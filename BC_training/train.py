"""Main training script for behavior cloning."""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax import serialization
import pickle
from omegaconf import OmegaConf
import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

console = Console()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from common.dataset import ZarrGameplayDataset, create_data_loader, split_episodes
from common.metrics import (
    compute_accuracy,
    compute_per_action_metrics,
    compute_action_distribution_distance,
    format_metrics_for_logging,
    print_metrics_summary,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
    """Extended train state with batch stats for batch norm."""
    batch_stats: Any = None


def save_checkpoint(ckpt_dir: Path, state: TrainState, name: str):
    """Save checkpoint using simple pickle + numpy.
    
    Args:
        ckpt_dir: Checkpoint directory
        state: TrainState to save
        name: Checkpoint name (e.g., 'best' or 'epoch_5')
    """
    ckpt_path = ckpt_dir / f"{name}.pkl"
    
    # Convert JAX arrays to numpy for serialization
    ckpt_data = {
        'step': int(state.step),
        'params': jax.tree.map(np.array, state.params),
        'opt_state': jax.tree.map(
            lambda x: np.array(x) if hasattr(x, 'shape') else x,
            state.opt_state
        ),
        'batch_stats': jax.tree.map(np.array, state.batch_stats) if state.batch_stats else None,
    }
    
    with open(ckpt_path, 'wb') as f:
        pickle.dump(ckpt_data, f)


def load_checkpoint(ckpt_path: Path, state: TrainState) -> TrainState:
    """Load checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        state: TrainState with initialized structure
        
    Returns:
        TrainState with loaded parameters
    """
    with open(ckpt_path, 'rb') as f:
        ckpt_data = pickle.load(f)
    
    # Convert numpy arrays back to JAX arrays
    params = jax.tree.map(jnp.array, ckpt_data['params'])
    opt_state = jax.tree.map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        ckpt_data['opt_state']
    )
    batch_stats = jax.tree.map(jnp.array, ckpt_data['batch_stats']) if ckpt_data['batch_stats'] else None
    
    return state.replace(
        step=ckpt_data['step'],
        params=params,
        opt_state=opt_state,
        batch_stats=batch_stats,
    )


def create_train_state(
    model,
    learning_rate: float,
    weight_decay: float,
    input_shape: tuple,
    rng,
) -> TrainState:
    """Initialize model and create train state.
    
    Args:
        model: Flax model
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        input_shape: Input shape for initialization
        rng: JAX random key
        
    Returns:
        TrainState with initialized parameters
    """
    # Initialize model
    init_rng, dropout_rng = jax.random.split(rng)
    variables = model.init(
        {'params': init_rng, 'dropout': dropout_rng},
        jnp.ones(input_shape),
        training=False,
    )
    
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)
    
    # Create optimizer
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
    )
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Model initialized with {param_count:,} parameters")
    
    return state


def create_learning_rate_schedule(config: Dict) -> optax.Schedule:
    """Create learning rate schedule.
    
    Args:
        config: Training config
        
    Returns:
        Optax learning rate schedule
    """
    lr_config = config['training']['lr_schedule']
    base_lr = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    
    if lr_config['type'] == 'cosine':
        warmup_epochs = lr_config.get('warmup_epochs', 0)
        min_lr = lr_config.get('min_lr', 0.0)
        
        schedules = []
        boundaries = []
        
        if warmup_epochs > 0:
            # Linear warmup
            schedules.append(optax.linear_schedule(
                init_value=0.0,
                end_value=base_lr,
                transition_steps=warmup_epochs,
            ))
            boundaries.append(warmup_epochs)
        
        # Cosine decay
        schedules.append(optax.cosine_decay_schedule(
            init_value=base_lr if warmup_epochs == 0 else base_lr,
            decay_steps=num_epochs - warmup_epochs,
            alpha=min_lr / base_lr,
        ))
        
        if len(schedules) > 1:
            schedule = optax.join_schedules(schedules, boundaries)
        else:
            schedule = schedules[0]
        
        logger.info(f"Created cosine LR schedule: {base_lr} -> {min_lr}, warmup={warmup_epochs}")
        
    else:  # constant
        schedule = optax.constant_schedule(base_lr)
        logger.info(f"Created constant LR schedule: {base_lr}")
    
    return schedule


def binary_cross_entropy_with_logits(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Binary cross entropy loss with logits.
    
    Args:
        logits: Model outputs [B, num_actions]
        labels: Ground truth [B, num_actions]
        weights: Optional per-action weights [num_actions]
        
    Returns:
        Loss value (scalar)
    """
    # Stable BCE computation
    log_sigmoid = jax.nn.log_sigmoid(logits)
    log_one_minus_sigmoid = jax.nn.log_sigmoid(-logits)
    
    loss = -(labels * log_sigmoid + (1 - labels) * log_one_minus_sigmoid)
    
    # Apply weights if provided
    if weights is not None:
        loss = loss * weights[None, :]
    
    return jnp.mean(loss)


def compute_accuracy_jax(predictions: jnp.ndarray, targets: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    """JAX-compatible accuracy computation for JIT.
    
    Args:
        predictions: Predicted probabilities [N, num_actions]
        targets: Ground truth binary labels [N, num_actions]
        threshold: Threshold for binary classification
        
    Returns:
        Accuracy as JAX scalar
    """
    pred_binary = (predictions > threshold).astype(jnp.float32)
    # Exact match: all actions must match
    exact_match = jnp.all(pred_binary == targets, axis=1)
    return jnp.mean(exact_match)


@jax.jit
def train_step(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Single training step.
    
    Args:
        state: TrainState
        batch: Batch of data
        action_weights: Per-action loss weights
        rng: JAX random key
        
    Returns:
        (new_state, metrics)
    """
    dropout_rng = jax.random.fold_in(rng, state.step)
    
    def loss_fn(params):
        # Forward pass
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats
        
        logits, new_variables = state.apply_fn(
            variables,
            batch['frames'],
            training=True,
            mutable=['batch_stats'] if state.batch_stats is not None else False,
            rngs={'dropout': dropout_rng},
        )
        
        # Compute loss
        loss = binary_cross_entropy_with_logits(
            logits,
            batch['actions'],
            weights=action_weights,
        )
        
        # Get updated batch stats
        new_batch_stats = new_variables.get('batch_stats', None)
        
        return loss, (logits, new_batch_stats)
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    # Update batch stats
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)
    
    # Compute metrics (using JAX arrays, no numpy conversion!)
    predictions = jax.nn.sigmoid(logits)
    accuracy = compute_accuracy_jax(predictions, batch['actions'])
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    
    return state, metrics


@jax.jit
def eval_step(state: TrainState, batch: Dict):
    """Single evaluation step.
    
    Args:
        state: TrainState
        batch: Batch of data
        
    Returns:
        (predictions, loss)
    """
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats
    
    logits = state.apply_fn(variables, batch['frames'], training=False)
    loss = binary_cross_entropy_with_logits(logits, batch['actions'])
    predictions = jax.nn.sigmoid(logits)
    
    return predictions, loss


def evaluate(state: TrainState, dataset: ZarrGameplayDataset, config: Dict, show_progress: bool = True) -> Dict:
    """Evaluate model on dataset with JIT-compiled steps and progress bar.
    
    Args:
        state: TrainState
        dataset: Evaluation dataset
        config: Config dict
        show_progress: Whether to show progress bar
        
    Returns:
        Dict of evaluation metrics
    """
    batch_size = config['training']['batch_size']
    
    # Pre-compute number of batches for progress bar
    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    batches_processed = 0
    
    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Evaluating"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Eval", total=num_batches)
            
            for batch in create_data_loader(dataset, batch_size, shuffle=False, drop_last=False):
                predictions, loss = eval_step(state, batch)
                
                # Block until computation is done, then convert
                all_predictions.append(np.asarray(predictions))
                all_targets.append(np.asarray(batch['actions']))
                total_loss += float(loss)
                batches_processed += 1
                progress.update(task, advance=1)
    else:
        for batch in create_data_loader(dataset, batch_size, shuffle=False, drop_last=False):
            predictions, loss = eval_step(state, batch)
            all_predictions.append(np.asarray(predictions))
            all_targets.append(np.asarray(batch['actions']))
            total_loss += float(loss)
            batches_processed += 1
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    avg_loss = total_loss / batches_processed
    accuracy = compute_accuracy(all_predictions, all_targets, threshold=0.5)
    per_action_metrics = compute_per_action_metrics(all_predictions, all_targets, threshold=0.5)
    dist_metrics = compute_action_distribution_distance(all_predictions, all_targets, threshold=0.5)
    
    # Ensure all scalar values are Python floats
    metrics = {
        'loss': float(avg_loss),
        'accuracy': float(accuracy) if hasattr(accuracy, '__float__') else accuracy,
        **per_action_metrics,
        **dist_metrics,
    }
    
    return metrics


def train(config_path: str):
    """Main training function.
    
    Args:
        config_path: Path to config YAML file
    """
    # Load config
    config = OmegaConf.load(config_path)
    console.print(f"[cyan]Loaded config from {config_path}[/cyan]")
    
    # Print config in a nice panel
    config_text = OmegaConf.to_yaml(config)
    console.print(Panel(config_text, title="[bold]Training Configuration[/bold]", expand=False))
    
    # Set random seed
    np.random.seed(config['system']['seed'])
    rng = jax.random.PRNGKey(config['system']['seed'])
    
    # Initialize wandb
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging']['wandb_entity'],
            name=config['logging']['wandb_run_name'],
            config=OmegaConf.to_container(config, resolve=True),
        )
        console.print("[green]✓ Initialized WandB[/green]")
    
    # Create checkpoint directory (must be absolute path for orbax)
    checkpoint_dir = Path(config['training']['checkpoint_dir']).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Checkpoint directory: {checkpoint_dir}[/cyan]")
    
    # Split episodes
    dataset_path = config['dataset']['path']
    zarr_root = __import__('zarr').open(dataset_path, mode='r')
    total_episodes = len([k for k in zarr_root.keys() if k.startswith('episode_')])
    
    train_indices, val_indices = split_episodes(
        total_episodes=total_episodes,
        train_ratio=config['dataset']['train_ratio'],
        val_ratio=config['dataset']['val_ratio'],
        seed=config['dataset']['split_seed'],
    )
    
    # Load datasets
    console.print("[cyan]Loading datasets...[/cyan]")
    train_dataset = ZarrGameplayDataset(
        dataset_path=dataset_path,
        episode_indices=train_indices,
        use_state=config['dataset']['use_state'],
        normalize_frames=config['dataset']['normalize_frames'],
        validate_episodes=config['dataset'].get('validate_episodes', True),
    )
    
    val_dataset = ZarrGameplayDataset(
        dataset_path=dataset_path,
        episode_indices=val_indices,
        use_state=config['dataset']['use_state'],
        normalize_frames=config['dataset']['normalize_frames'],
        validate_episodes=config['dataset'].get('validate_episodes', True),
    )
    
    # Get action names
    action_names = train_dataset.action_keys
    num_actions = len(action_names)
    console.print(f"[cyan]Actions ({num_actions}): {', '.join(action_names)}[/cyan]")
    
    # Compute action weights
    console.print("[cyan]Computing action weights...[/cyan]")
    if config['training']['use_class_weights']:
        action_weights = train_dataset.compute_action_weights()
        action_weights = jnp.array(action_weights)
    else:
        action_weights = jnp.ones(num_actions)
    
    # Create model
    console.print("[cyan]Creating model...[/cyan]")
    model_name = config['model']['name']
    if model_name == 'pure_cnn':
        from models.pure_cnn import create_model
        model = create_model(
            num_actions=num_actions,
            conv_features=tuple(config['model']['conv_features']),
            dense_features=tuple(config['model']['dense_features']),
            dropout_rate=config['model']['dropout_rate'],
            use_batch_norm=config['model']['use_batch_norm'],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get input shape from dataset
    sample = train_dataset[0]
    input_shape = (config['training']['batch_size'],) + sample['frames'].shape
    console.print(f"[cyan]Input shape: {input_shape}[/cyan]")
    
    # Create train state
    console.print("[cyan]Initializing model parameters (this may take a moment)...[/cyan]")
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        model=model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        input_shape=input_shape,
        rng=init_rng,
    )
    console.print("[green]✓ Model initialized and ready![/green]")
    
    # Training loop
    console.print("\n[bold cyan]Starting training...[/bold cyan]")
    best_val_accuracy = 0.0
    
    # Calculate total steps
    batch_size = config['training']['batch_size']
    steps_per_epoch = len(train_dataset) // batch_size
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    ) as progress:
        
        # Create progress bars
        epoch_task = progress.add_task(
            "[cyan]Epochs",
            total=config['training']['num_epochs']
        )
        
        batch_task = progress.add_task(
            "[green]Batches",
            total=steps_per_epoch
        )
        
        for epoch in range(config['training']['num_epochs']):
            epoch_start_time = time.time()
            
            # Reset batch progress
            progress.reset(batch_task, total=steps_per_epoch, description=f"[green]Epoch {epoch+1} Batches")
            
            # Training
            train_metrics = []
            
            for step, batch in enumerate(create_data_loader(train_dataset, batch_size, shuffle=True)):
                rng, step_rng = jax.random.split(rng)
                state, metrics = train_step(state, batch, action_weights, step_rng)
                train_metrics.append(metrics)
                
                # Update progress bar with current metrics
                current_loss = float(metrics['loss'])
                current_acc = float(metrics['accuracy'])
                progress.update(
                    batch_task, 
                    advance=1,
                    description=f"[green]Epoch {epoch+1} │ Loss: {current_loss:.4f} │ Acc: {current_acc:.4f}"
                )
                
                # Log step metrics
                if config['logging']['use_wandb'] and step % config['logging']['log_every_n_steps'] == 0:
                    wandb.log({
                        'train/step_loss': float(metrics['loss']),
                        'train/step_accuracy': float(metrics['accuracy']),
                        'step': int(state.step),
                    })
            
            # Aggregate training metrics
            avg_train_loss = np.mean([m['loss'] for m in train_metrics])
            avg_train_accuracy = np.mean([m['accuracy'] for m in train_metrics])
            
            # Evaluation
            if (epoch + 0) % config['evaluation']['eval_every_n_epochs'] == 0:
                progress.stop()  # Pause progress bar for evaluation output
                
                console.print(f"\n[bold yellow]Evaluating at epoch {epoch + 1}...[/bold yellow]")
                val_metrics = evaluate(state, val_dataset, config)
                
                # Create summary table
                table = Table(title=f"Epoch {epoch + 1}/{config['training']['num_epochs']} Summary", show_header=True, header_style="bold magenta")
                table.add_column("Split", style="cyan")
                table.add_column("Loss", justify="right")
                table.add_column("Accuracy", justify="right")
                table.add_row("Train", f"{avg_train_loss:.4f}", f"{avg_train_accuracy:.4f}")
                table.add_row("Val", f"{val_metrics['loss']:.4f}", f"{val_metrics['accuracy']:.4f}")
                
                console.print(table)
                print_metrics_summary(val_metrics, action_names, title=f"Validation Metrics (Epoch {epoch + 1})")
                
                progress.start()  # Resume progress bar
                
                # Log to wandb
                if config['logging']['use_wandb']:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train/loss': avg_train_loss,
                        'train/accuracy': avg_train_accuracy,
                        'val/loss': val_metrics['loss'],
                        'val/accuracy': val_metrics['accuracy'],
                    }
                    
                    # Add per-action metrics
                    per_action_log = format_metrics_for_logging(val_metrics, action_names, prefix='val/')
                    log_dict.update(per_action_log)
                    
                    wandb.log(log_dict)
                
                # Save best model
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    save_checkpoint(checkpoint_dir, state, 'best')
                    console.print(f"[bold green]✓ Saved best model with val accuracy: {best_val_accuracy:.4f}[/bold green]")
            
            # Save periodic checkpoint
            if (epoch + 1) % config['training']['save_every_n_epochs'] == 0:
                save_checkpoint(checkpoint_dir, state, f'epoch_{epoch + 1}')
                console.print(f"[cyan]Saved checkpoint at epoch {epoch + 1}[/cyan]")
            
            epoch_time = time.time() - epoch_start_time
            
            # Update epoch progress
            progress.update(epoch_task, advance=1)
    
    console.print("\n[bold green]✓ Training completed![/bold green]")
    console.print(f"[bold cyan]Best validation accuracy: {best_val_accuracy:.4f}[/bold cyan]")
    
    if config['logging']['use_wandb']:
        wandb.finish()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train behavior cloning model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pure_cnn.yaml',
        help='Path to config file',
    )
    
    args = parser.parse_args()
    
    train(args.config)

