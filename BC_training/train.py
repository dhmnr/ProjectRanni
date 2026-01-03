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

from common.dataset import (
    ZarrGameplayDataset, 
    create_data_loader, 
    split_episodes,
    TemporalGameplayDataset,
    create_temporal_data_loader,
)
from common.state_preprocessing import create_preprocessor
from common.metrics import (
    compute_accuracy,
    compute_per_action_metrics,
    compute_action_distribution_distance,
    compute_onset_metrics,
    compute_buffered_onset_metrics,
    format_metrics_for_logging,
    print_metrics_summary,
)
from common.losses import get_loss_fn, focal_loss_with_logits

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
    state_shape: tuple = None,
    use_anim_embeddings: bool = False,
    is_temporal: bool = False,
    action_history_shape: tuple = None,
) -> TrainState:
    """Initialize model and create train state.
    
    Args:
        model: Flax model
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        input_shape: Input shape for initialization (frames)
        rng: JAX random key
        state_shape: Optional state input shape for hybrid models
        use_anim_embeddings: Whether model uses animation embeddings
        is_temporal: Whether this is a temporal model
        action_history_shape: Shape for action history (temporal models)
        
    Returns:
        TrainState with initialized parameters
    """
    # Initialize model
    init_rng, dropout_rng = jax.random.split(rng)
    
    batch_size = input_shape[0]
    
    if is_temporal:
        # Temporal model initialization
        if state_shape is not None and use_anim_embeddings:
            # Temporal with state + embeddings
            variables = model.init(
                {'params': init_rng, 'dropout': dropout_rng},
                jnp.ones(input_shape),                      # frames [B, T, C, H, W]
                jnp.ones(action_history_shape),             # action_history [B, K, A]
                jnp.ones(state_shape),                      # continuous state
                jnp.zeros((batch_size,), dtype=jnp.int32),  # hero_anim_idx
                jnp.zeros((batch_size,), dtype=jnp.int32),  # npc_anim_idx
                training=False,
            )
        else:
            # Temporal without state
            variables = model.init(
                {'params': init_rng, 'dropout': dropout_rng},
                jnp.ones(input_shape),                      # frames [B, T, C, H, W]
                jnp.ones(action_history_shape),             # action_history [B, K, A]
                training=False,
            )
    elif state_shape is not None:
        if use_anim_embeddings:
            # Hybrid model with state + animation embeddings
            variables = model.init(
                {'params': init_rng, 'dropout': dropout_rng},
                jnp.ones(input_shape),                      # frames
                jnp.ones(state_shape),                      # continuous state
                jnp.zeros((batch_size,), dtype=jnp.int32),  # hero_anim_idx
                jnp.zeros((batch_size,), dtype=jnp.int32),  # npc_anim_idx
                training=False,
            )
        else:
            # Hybrid model with state input only (legacy)
            variables = model.init(
                {'params': init_rng, 'dropout': dropout_rng},
                jnp.ones(input_shape),  # frames
                jnp.ones(state_shape),  # state
                training=False,
            )
    else:
        # Vision-only model
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


# Module-level loss configuration - SET BEFORE FIRST JIT COMPILATION
# These values are captured at JIT trace time
_USE_FOCAL_LOSS = False
_FOCAL_GAMMA = 2.0
_FOCAL_ALPHA = None
_ONSET_WEIGHT = 1.0  # Weight multiplier for onset frames (>1 to penalize copying)


def _compute_onset_weights(
    labels: jnp.ndarray,
    previous_actions: jnp.ndarray,
    onset_weight: float,
) -> jnp.ndarray:
    """Compute per-sample weights based on onset detection."""
    action_changed = jnp.any(labels != previous_actions, axis=1)
    return jnp.where(action_changed, onset_weight, 1.0)


def compute_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
    previous_actions: Optional[jnp.ndarray] = None,
    onset_weight: float = 1.0,
) -> jnp.ndarray:
    """Compute loss with configurable loss function and onset weighting.
    
    Args:
        logits: Model outputs [B, num_actions]
        labels: Ground truth [B, num_actions]
        weights: Optional per-action weights [num_actions]
        use_focal: Whether to use focal loss
        focal_gamma: Focal loss gamma parameter
        focal_alpha: Focal loss alpha parameter
        previous_actions: Previous frame actions for onset weighting [B, num_actions]
        onset_weight: Weight multiplier for onset frames (>1 to upweight action changes)
        
    Returns:
        Loss value (scalar)
    """
    if use_focal:
        return focal_loss_with_logits(
            logits, labels, 
            gamma=focal_gamma, 
            alpha=focal_alpha,
            weights=weights if weights is not None else None,
            previous_actions=previous_actions,
            onset_weight=onset_weight,
        )
    else:
        # Standard BCE
        log_sigmoid = jax.nn.log_sigmoid(logits)
        log_one_minus_sigmoid = jax.nn.log_sigmoid(-logits)
        
        loss = -(labels * log_sigmoid + (1 - labels) * log_one_minus_sigmoid)
        
        if weights is not None:
            loss = loss * weights[None, :]
        
        # Apply onset weighting if previous_actions provided
        if previous_actions is not None and onset_weight > 1.0:
            sample_weights = _compute_onset_weights(labels, previous_actions, onset_weight)
            loss_per_sample = jnp.mean(loss, axis=1)  # [batch]
            return jnp.sum(loss_per_sample * sample_weights) / jnp.sum(sample_weights)
        
        return jnp.mean(loss)


# Wrapper functions that use module-level config
# These MUST be created before JIT compilation to capture config values
def _create_loss_fn(use_focal: bool, focal_gamma: float, focal_alpha: Optional[float], onset_weight: float = 1.0):
    """Create a loss function with the specified configuration."""
    def loss_fn(logits, labels, weights=None, previous_actions=None):
        return compute_loss(logits, labels, weights, use_focal, focal_gamma, focal_alpha, 
                           previous_actions, onset_weight)
    return loss_fn


# Default loss function (standard BCE)
_current_loss_fn = _create_loss_fn(False, 2.0, None, 1.0)


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
def train_step_vision(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Single training step for vision-only models.
    
    Args:
        state: TrainState
        batch: Batch of data (frames, actions)
        action_weights: Per-action loss weights
        rng: JAX random key
        
    Returns:
        (new_state, metrics)
    """
    dropout_rng = jax.random.fold_in(rng, state.step)
    
    def loss_fn(params):
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
        
        loss = _current_loss_fn(
            logits,
            batch['actions'],
            weights=action_weights,
        )
        
        new_batch_stats = new_variables.get('batch_stats', None)
        return loss, (logits, new_batch_stats)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)
    
    predictions = jax.nn.sigmoid(logits)
    accuracy = compute_accuracy_jax(predictions, batch['actions'])
    
    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def train_step_hybrid(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Single training step for hybrid models (vision + state + anim embeddings).
    
    Args:
        state: TrainState
        batch: Batch of data (frames, state, hero_anim_idx, npc_anim_idx, actions)
        action_weights: Per-action loss weights
        rng: JAX random key
        
    Returns:
        (new_state, metrics)
    """
    dropout_rng = jax.random.fold_in(rng, state.step)
    
    def loss_fn(params):
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats
        
        logits, new_variables = state.apply_fn(
            variables,
            batch['frames'],
            batch['state'],
            batch['hero_anim_idx'],
            batch['npc_anim_idx'],
            training=True,
            mutable=['batch_stats'] if state.batch_stats is not None else False,
            rngs={'dropout': dropout_rng},
        )
        
        loss = _current_loss_fn(
            logits,
            batch['actions'],
            weights=action_weights,
        )
        
        new_batch_stats = new_variables.get('batch_stats', None)
        return loss, (logits, new_batch_stats)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)
    
    predictions = jax.nn.sigmoid(logits)
    accuracy = compute_accuracy_jax(predictions, batch['actions'])
    
    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def eval_step_vision(state: TrainState, batch: Dict):
    """Single evaluation step for vision-only models."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats
    
    logits = state.apply_fn(variables, batch['frames'], training=False)
    loss = _current_loss_fn(logits, batch['actions'])
    predictions = jax.nn.sigmoid(logits)
    
    return predictions, loss


@jax.jit
def eval_step_hybrid(state: TrainState, batch: Dict):
    """Single evaluation step for hybrid models (vision + state + anim embeddings)."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats
    
    logits = state.apply_fn(
        variables, 
        batch['frames'], 
        batch['state'],
        batch['hero_anim_idx'],
        batch['npc_anim_idx'],
        training=False
    )
    loss = _current_loss_fn(logits, batch['actions'])
    predictions = jax.nn.sigmoid(logits)
    
    return predictions, loss


@jax.jit
def train_step_temporal(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Single training step for temporal models (frame stacking + action history).
    
    Args:
        state: TrainState
        batch: Batch of data (frames [B,T,C,H,W], action_history, state, actions)
        action_weights: Per-action loss weights
        rng: JAX random key
        
    Returns:
        (new_state, metrics)
    """
    dropout_rng = jax.random.fold_in(rng, state.step)
    
    def loss_fn(params):
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats
        
        # Check if we have state features
        has_state = 'state' in batch and batch['state'] is not None
        
        if has_state:
            logits, new_variables = state.apply_fn(
                variables,
                batch['frames'],
                batch['action_history'],
                batch['state'],
                batch['hero_anim_idx'],
                batch['npc_anim_idx'],
                training=True,
                mutable=['batch_stats'] if state.batch_stats is not None else False,
                rngs={'dropout': dropout_rng},
            )
        else:
            logits, new_variables = state.apply_fn(
                variables,
                batch['frames'],
                batch['action_history'],
                training=True,
                mutable=['batch_stats'] if state.batch_stats is not None else False,
                rngs={'dropout': dropout_rng},
            )
        
        # Get previous action for onset weighting (last in history = t-1)
        previous_actions = batch['action_history'][:, -1, :]
        
        loss = _current_loss_fn(
            logits,
            batch['actions'],
            weights=action_weights,
            previous_actions=previous_actions,
        )
        
        new_batch_stats = new_variables.get('batch_stats', None)
        return loss, (logits, new_batch_stats)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)
    
    predictions = jax.nn.sigmoid(logits)
    accuracy = compute_accuracy_jax(predictions, batch['actions'])
    
    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def train_step_temporal_with_state(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Training step for temporal models with state features."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    
    def loss_fn(params):
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats
        
        logits, new_variables = state.apply_fn(
            variables,
            batch['frames'],
            batch['action_history'],
            batch['state'],
            batch['hero_anim_idx'],
            batch['npc_anim_idx'],
            training=True,
            mutable=['batch_stats'] if state.batch_stats is not None else False,
            rngs={'dropout': dropout_rng},
        )
        
        # Get previous action for onset weighting (last in history = t-1)
        previous_actions = batch['action_history'][:, -1, :]
        
        loss = _current_loss_fn(
            logits,
            batch['actions'],
            weights=action_weights,
            previous_actions=previous_actions,
        )
        
        new_batch_stats = new_variables.get('batch_stats', None)
        return loss, (logits, new_batch_stats)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)
    
    predictions = jax.nn.sigmoid(logits)
    accuracy = compute_accuracy_jax(predictions, batch['actions'])
    
    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def train_step_temporal_no_state(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Training step for temporal models without state features."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    
    def loss_fn(params):
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats
        
        logits, new_variables = state.apply_fn(
            variables,
            batch['frames'],
            batch['action_history'],
            training=True,
            mutable=['batch_stats'] if state.batch_stats is not None else False,
            rngs={'dropout': dropout_rng},
        )
        
        # Get previous action for onset weighting (last in history = t-1)
        previous_actions = batch['action_history'][:, -1, :]
        
        loss = _current_loss_fn(
            logits,
            batch['actions'],
            weights=action_weights,
            previous_actions=previous_actions,
        )
        
        new_batch_stats = new_variables.get('batch_stats', None)
        return loss, (logits, new_batch_stats)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)
    
    predictions = jax.nn.sigmoid(logits)
    accuracy = compute_accuracy_jax(predictions, batch['actions'])
    
    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def eval_step_temporal_with_state(state: TrainState, batch: Dict):
    """Eval step for temporal models with state features."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats
    
    logits = state.apply_fn(
        variables,
        batch['frames'],
        batch['action_history'],
        batch['state'],
        batch['hero_anim_idx'],
        batch['npc_anim_idx'],
        training=False
    )
    loss = _current_loss_fn(logits, batch['actions'])
    predictions = jax.nn.sigmoid(logits)
    
    return predictions, loss


@jax.jit
def eval_step_temporal_no_state(state: TrainState, batch: Dict):
    """Eval step for temporal models without state features."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats
    
    logits = state.apply_fn(
        variables,
        batch['frames'],
        batch['action_history'],
        training=False
    )
    loss = _current_loss_fn(logits, batch['actions'])
    predictions = jax.nn.sigmoid(logits)
    
    return predictions, loss


def evaluate(
    state: TrainState, 
    dataset, 
    config: Dict, 
    show_progress: bool = True, 
    use_state: bool = False,
    is_temporal: bool = False,
) -> Dict:
    """Evaluate model on dataset with JIT-compiled steps and progress bar.
    
    Args:
        state: TrainState
        dataset: Evaluation dataset
        config: Config dict
        show_progress: Whether to show progress bar
        use_state: Whether model uses state features
        is_temporal: Whether this is a temporal model
        
    Returns:
        Dict of evaluation metrics
    """
    batch_size = config['training']['batch_size']
    
    # Select appropriate eval function and data loader
    if is_temporal:
        eval_fn = eval_step_temporal_with_state if use_state else eval_step_temporal_no_state
        data_loader_fn = create_temporal_data_loader
    else:
        eval_fn = eval_step_hybrid if use_state else eval_step_vision
        data_loader_fn = create_data_loader
    
    # Pre-compute number of batches for progress bar
    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    all_targets = []
    all_previous_actions = []  # For onset metrics
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
            
            for batch in data_loader_fn(dataset, batch_size, shuffle=False, drop_last=False):
                predictions, loss = eval_fn(state, batch)
                
                # Block until computation is done, then convert
                all_predictions.append(np.asarray(predictions))
                all_targets.append(np.asarray(batch['actions']))
                
                # Collect previous actions for onset metrics (temporal models have action_history)
                if is_temporal and 'action_history' in batch:
                    # Last action in history = previous action
                    prev_actions = np.asarray(batch['action_history'][:, -1, :])
                    all_previous_actions.append(prev_actions)
                
                total_loss += float(loss)
                batches_processed += 1
                progress.update(task, advance=1)
    else:
        for batch in data_loader_fn(dataset, batch_size, shuffle=False, drop_last=False):
            predictions, loss = eval_fn(state, batch)
            all_predictions.append(np.asarray(predictions))
            all_targets.append(np.asarray(batch['actions']))
            
            if is_temporal and 'action_history' in batch:
                prev_actions = np.asarray(batch['action_history'][:, -1, :])
                all_previous_actions.append(prev_actions)
            
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
    
    # Compute onset metrics for ALL models
    # For temporal models, we have action_history; for non-temporal, we infer from targets
    onset_metrics = {}
    buffered_onset_metrics = {}
    
    if is_temporal and all_previous_actions:
        # Temporal models: use action_history for previous actions
        all_previous_actions = np.concatenate(all_previous_actions, axis=0)
    else:
        # Non-temporal models: infer previous actions by shifting targets
        # First frame has no previous, so we use zeros (no action)
        all_previous_actions = np.zeros_like(all_targets)
        all_previous_actions[1:] = all_targets[:-1]
    
    # Always compute onset metrics
    onset_metrics = compute_onset_metrics(
        all_predictions, all_targets, all_previous_actions, threshold=0.5
    )
    
    # Compute buffered onset metrics
    # Create "previous predictions" by shifting predictions array
    prev_predictions = np.zeros_like(all_predictions)
    prev_predictions[1:] = all_predictions[:-1]
    
    # Get buffer size from config (default 5 frames = ~167ms at 30fps)
    buffer_frames = config.get('evaluation', {}).get('onset_buffer_frames', 5)
    buffered_onset_metrics = compute_buffered_onset_metrics(
        all_predictions, all_targets, all_previous_actions, prev_predictions,
        threshold=0.5, buffer_frames=buffer_frames
    )
    
    # Ensure all scalar values are Python floats
    metrics = {
        'loss': float(avg_loss),
        'accuracy': float(accuracy) if hasattr(accuracy, '__float__') else accuracy,
        **per_action_metrics,
        **dist_metrics,
        **onset_metrics,
        **buffered_onset_metrics,
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
    
    # Create state preprocessor (used by all models with state)
    state_preprocessor = create_preprocessor(config)
    
    # Check if this is a temporal model
    model_name = config['model']['name']
    is_temporal = model_name in ['temporal_cnn', 'gru', 'causal_transformer']
    
    if is_temporal:
        # Temporal dataset with frame stacking and action history
        temporal_config = config.get('temporal', {})
        num_history_frames = temporal_config.get('num_history_frames', 4)
        num_action_history = temporal_config.get('num_action_history', 4)
        frame_skip = temporal_config.get('frame_skip', 1)
        
        train_dataset = TemporalGameplayDataset(
            dataset_path=dataset_path,
            episode_indices=train_indices,
            use_state=config['dataset']['use_state'],
            normalize_frames=config['dataset']['normalize_frames'],
            validate_episodes=config['dataset'].get('validate_episodes', True),
            state_preprocessor=state_preprocessor,
            num_history_frames=num_history_frames,
            num_action_history=num_action_history,
            frame_skip=frame_skip,
        )
        
        val_dataset = TemporalGameplayDataset(
            dataset_path=dataset_path,
            episode_indices=val_indices,
            use_state=config['dataset']['use_state'],
            normalize_frames=config['dataset']['normalize_frames'],
            validate_episodes=config['dataset'].get('validate_episodes', True),
            state_preprocessor=state_preprocessor,
            num_history_frames=num_history_frames,
            num_action_history=num_action_history,
            frame_skip=frame_skip,
        )
    else:
        # Standard single-frame dataset
        train_dataset = ZarrGameplayDataset(
            dataset_path=dataset_path,
            episode_indices=train_indices,
            use_state=config['dataset']['use_state'],
            normalize_frames=config['dataset']['normalize_frames'],
            validate_episodes=config['dataset'].get('validate_episodes', True),
            state_preprocessor=state_preprocessor,
        )
        
        val_dataset = ZarrGameplayDataset(
            dataset_path=dataset_path,
            episode_indices=val_indices,
            use_state=config['dataset']['use_state'],
            normalize_frames=config['dataset']['normalize_frames'],
            validate_episodes=config['dataset'].get('validate_episodes', True),
            state_preprocessor=state_preprocessor,
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
    use_state = config['dataset'].get('use_state', False)
    
    if model_name == 'pure_cnn':
        from models.pure_cnn import create_model
        model = create_model(
            num_actions=num_actions,
            conv_features=tuple(config['model']['conv_features']),
            dense_features=tuple(config['model']['dense_features']),
            dropout_rate=config['model']['dropout_rate'],
            use_batch_norm=config['model']['use_batch_norm'],
        )
    elif model_name == 'hybrid_state':
        from models.hybrid_state import create_model
        num_state_features = state_preprocessor.continuous_dim
        console.print(f"[cyan]Continuous state features: {num_state_features}[/cyan]")
        console.print(f"[cyan]Hero anim vocab size: {state_preprocessor.hero_vocab_size}[/cyan]")
        console.print(f"[cyan]NPC anim vocab size: {state_preprocessor.npc_vocab_size}[/cyan]")
        
        # Get embedding config
        anim_embed_dim = config.get('state_preprocessing', {}).get('anim_embed_dim', 16)
        
        model = create_model(
            num_actions=num_actions,
            num_state_features=num_state_features,
            hero_anim_vocab_size=state_preprocessor.hero_vocab_size,
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size,
            anim_embed_dim=anim_embed_dim,
            conv_features=tuple(config['model']['conv_features']),
            dense_features=tuple(config['model']['dense_features']),
            state_encoder_features=tuple(config['model']['state_encoder_features']),
            state_output_features=config['model']['state_output_features'],
            dropout_rate=config['model']['dropout_rate'],
            use_batch_norm=config['model']['use_batch_norm'],
        )
    elif model_name == 'temporal_cnn':
        from models.temporal_cnn import create_model
        
        # Get temporal config
        temporal_config = config.get('temporal', {})
        num_history_frames = temporal_config.get('num_history_frames', 4)
        num_action_history = temporal_config.get('num_action_history', 4)
        
        # State features (optional)
        num_state_features = state_preprocessor.continuous_dim if use_state else 10
        anim_embed_dim = config.get('state_preprocessing', {}).get('anim_embed_dim', 16)
        
        console.print(f"[cyan]Temporal config: {num_history_frames} history frames, {num_action_history} action history[/cyan]")
        if use_state:
            console.print(f"[cyan]Continuous state features: {num_state_features}[/cyan]")
            console.print(f"[cyan]Hero anim vocab size: {state_preprocessor.hero_vocab_size}[/cyan]")
            console.print(f"[cyan]NPC anim vocab size: {state_preprocessor.npc_vocab_size}[/cyan]")
        
        model = create_model(
            num_actions=num_actions,
            num_history_frames=num_history_frames,
            num_action_history=num_action_history,
            use_state=use_state,
            num_state_features=num_state_features,
            hero_anim_vocab_size=state_preprocessor.hero_vocab_size if use_state else 67,
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size if use_state else 54,
            anim_embed_dim=anim_embed_dim,
            frame_mode=config['model'].get('frame_mode', 'channel_stack'),
            conv_features=tuple(config['model']['conv_features']),
            dense_features=tuple(config['model']['dense_features']),
            state_encoder_features=tuple(config['model'].get('state_encoder_features', [64, 64])),
            state_output_features=config['model'].get('state_output_features', 64),
            action_history_features=config['model'].get('action_history_features', 64),
            dropout_rate=config['model']['dropout_rate'],
            use_batch_norm=config['model']['use_batch_norm'],
        )
    elif model_name == 'gru':
        from models.gru import create_model
        
        # Get temporal config
        temporal_config = config.get('temporal', {})
        num_history_frames = temporal_config.get('num_history_frames', 4)
        num_action_history = temporal_config.get('num_action_history', 4)
        
        # GRU specific config
        gru_hidden_size = config['model'].get('gru_hidden_size', 256)
        gru_num_layers = config['model'].get('gru_num_layers', 1)
        
        # State features (optional)
        num_state_features = state_preprocessor.continuous_dim if use_state else 10
        anim_embed_dim = config.get('state_preprocessing', {}).get('anim_embed_dim', 16)
        
        console.print(f"[cyan]GRU config: hidden_size={gru_hidden_size}, layers={gru_num_layers}[/cyan]")
        console.print(f"[cyan]Temporal config: {num_history_frames} history frames, {num_action_history} action history[/cyan]")
        if use_state:
            console.print(f"[cyan]Continuous state features: {num_state_features}[/cyan]")
            console.print(f"[cyan]Hero anim vocab size: {state_preprocessor.hero_vocab_size}[/cyan]")
            console.print(f"[cyan]NPC anim vocab size: {state_preprocessor.npc_vocab_size}[/cyan]")
        
        model = create_model(
            num_actions=num_actions,
            num_history_frames=num_history_frames,
            num_action_history=num_action_history,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            use_state=use_state,
            num_state_features=num_state_features,
            hero_anim_vocab_size=state_preprocessor.hero_vocab_size if use_state else 67,
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size if use_state else 54,
            anim_embed_dim=anim_embed_dim,
            conv_features=tuple(config['model']['conv_features']),
            dense_features=tuple(config['model']['dense_features']),
            state_encoder_features=tuple(config['model'].get('state_encoder_features', [64, 64])),
            state_output_features=config['model'].get('state_output_features', 64),
            action_history_features=config['model'].get('action_history_features', 64),
            dropout_rate=config['model']['dropout_rate'],
            use_batch_norm=config['model']['use_batch_norm'],
        )
    elif model_name == 'causal_transformer':
        from models.causal_transformer import create_model
        
        # Get temporal config
        temporal_config = config.get('temporal', {})
        num_history_frames = temporal_config.get('num_history_frames', 4)
        num_action_history = temporal_config.get('num_action_history', 4)
        
        # Transformer specific config
        d_model = config['model'].get('d_model', 256)
        num_heads = config['model'].get('num_heads', 4)
        num_layers = config['model'].get('num_layers', 2)
        d_ff = config['model'].get('d_ff', 512)
        max_seq_len = config['model'].get('max_seq_len', 32)
        
        # State features (optional)
        num_state_features = state_preprocessor.continuous_dim if use_state else 10
        anim_embed_dim = config.get('state_preprocessing', {}).get('anim_embed_dim', 16)
        
        console.print(f"[cyan]Transformer config: d_model={d_model}, heads={num_heads}, layers={num_layers}, d_ff={d_ff}[/cyan]")
        console.print(f"[cyan]Temporal config: {num_history_frames} history frames, {num_action_history} action history[/cyan]")
        if use_state:
            console.print(f"[cyan]Continuous state features: {num_state_features}[/cyan]")
            console.print(f"[cyan]Hero anim vocab size: {state_preprocessor.hero_vocab_size}[/cyan]")
            console.print(f"[cyan]NPC anim vocab size: {state_preprocessor.npc_vocab_size}[/cyan]")
        
        model = create_model(
            num_actions=num_actions,
            num_history_frames=num_history_frames,
            num_action_history=num_action_history,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            use_state=use_state,
            num_state_features=num_state_features,
            hero_anim_vocab_size=state_preprocessor.hero_vocab_size if use_state else 67,
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size if use_state else 54,
            anim_embed_dim=anim_embed_dim,
            conv_features=tuple(config['model']['conv_features']),
            dense_features=tuple(config['model']['dense_features']),
            state_encoder_features=tuple(config['model'].get('state_encoder_features', [64, 64])),
            state_output_features=config['model'].get('state_output_features', 64),
            action_history_features=config['model'].get('action_history_features', 64),
            dropout_rate=config['model']['dropout_rate'],
            use_batch_norm=config['model']['use_batch_norm'],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get input shape from dataset
    sample = train_dataset[0]
    input_shape = (config['training']['batch_size'],) + sample['frames'].shape
    console.print(f"[cyan]Input shape: {input_shape}[/cyan]")
    
    # Get state shape if using state
    state_shape = None
    if use_state and 'state' in sample:
        state_shape = (config['training']['batch_size'],) + sample['state'].shape
        console.print(f"[cyan]State shape: {state_shape}[/cyan]")
    
    # Get action history shape for temporal models
    action_history_shape = None
    if is_temporal and 'action_history' in sample:
        action_history_shape = (config['training']['batch_size'],) + sample['action_history'].shape
        console.print(f"[cyan]Action history shape: {action_history_shape}[/cyan]")
    
    # Create train state
    console.print("[cyan]Initializing model parameters (this may take a moment)...[/cyan]")
    rng, init_rng = jax.random.split(rng)
    
    # Hybrid state models use animation embeddings
    use_anim_embeddings = (model_name == 'hybrid_state') or (model_name in ['temporal_cnn', 'gru', 'causal_transformer'] and use_state)
    
    state = create_train_state(
        model=model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        input_shape=input_shape,
        rng=init_rng,
        state_shape=state_shape,
        use_anim_embeddings=use_anim_embeddings,
        is_temporal=is_temporal,
        action_history_shape=action_history_shape,
    )
    console.print("[green]✓ Model initialized and ready![/green]")
    
    # Training loop
    console.print("\n[bold cyan]Starting training...[/bold cyan]")
    best_val_accuracy = 0.0
    
    # Configure loss function BEFORE any JIT compilation
    global _current_loss_fn
    loss_config = config.get('training', {}).get('loss', {})
    loss_type = loss_config.get('type', 'bce')
    onset_weight = loss_config.get('onset_weight', 1.0)  # Weight for onset frames
    
    if loss_type == 'focal':
        focal_gamma = loss_config.get('gamma', 2.0)
        focal_alpha = loss_config.get('alpha', None)
        _current_loss_fn = _create_loss_fn(use_focal=True, focal_gamma=focal_gamma, focal_alpha=focal_alpha, onset_weight=onset_weight)
        console.print(f"[cyan]Using Focal Loss (γ={focal_gamma}, α={focal_alpha}, onset_weight={onset_weight})[/cyan]")
    else:
        _current_loss_fn = _create_loss_fn(use_focal=False, focal_gamma=2.0, focal_alpha=None, onset_weight=onset_weight)
        if onset_weight > 1.0:
            console.print(f"[cyan]Using BCE Loss with onset weighting ({onset_weight}x)[/cyan]")
        else:
            console.print("[cyan]Using Binary Cross Entropy Loss[/cyan]")
    
    # Calculate total steps
    batch_size = config['training']['batch_size']
    steps_per_epoch = len(train_dataset) // batch_size
    
    # Select appropriate step functions and data loader based on model type
    if is_temporal:
        train_fn = train_step_temporal_with_state if use_state else train_step_temporal_no_state
        data_loader_fn = create_temporal_data_loader
    else:
        train_fn = train_step_hybrid if use_state else train_step_vision
        data_loader_fn = create_data_loader
    
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
            
            for step, batch in enumerate(data_loader_fn(train_dataset, batch_size, shuffle=True)):
                rng, step_rng = jax.random.split(rng)
                state, metrics = train_fn(state, batch, action_weights, step_rng)
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
                val_metrics = evaluate(state, val_dataset, config, use_state=use_state, is_temporal=is_temporal)
                
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

