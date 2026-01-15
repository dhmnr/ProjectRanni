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
import zarr
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
    ActionChunkingDataset,
    create_action_chunking_data_loader,
    OnsetOffsetDataset,
    create_onset_offset_data_loader,
    ActionComboDataset,
    create_action_combo_data_loader,
    StateChunkingDataset,
    create_state_chunking_data_loader,
)
from common.state_preprocessing import create_preprocessor
from common.metrics import (
    compute_per_action_metrics,
    load_action_names,
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
    is_action_chunking: bool = False,
    is_state_only: bool = False,
    is_state_chunking: bool = False,
    num_temporal_states: int = 8,
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
        is_action_chunking: Whether this is an action chunking model
        is_state_only: Whether this is a state-only model (no frames)
        is_state_chunking: Whether this is a state chunking model (temporal states, no frames)
        num_temporal_states: Number of temporal states for state_chunking

    Returns:
        TrainState with initialized parameters
    """
    # Initialize model
    init_rng, dropout_rng = jax.random.split(rng)

    batch_size = input_shape[0]

    if is_state_chunking:
        # State chunking model (temporal states, no frames)
        # state_shape is [B, num_states, num_features]
        variables = model.init(
            {'params': init_rng, 'dropout': dropout_rng},
            jnp.ones(state_shape),                                        # states [B, num_states, num_features]
            jnp.zeros((batch_size, num_temporal_states), dtype=jnp.int32),  # hero_anim_ids [B, num_states]
            jnp.zeros((batch_size, num_temporal_states), dtype=jnp.int32),  # npc_anim_ids [B, num_states]
            training=False,
        )
    elif is_state_only:
        # State-only model (e.g., action_combo_transformer)
        variables = model.init(
            {'params': init_rng, 'dropout': dropout_rng},
            jnp.ones(state_shape),                          # state
            jnp.zeros((batch_size,), dtype=jnp.int32),      # hero_anim_idx
            jnp.zeros((batch_size,), dtype=jnp.int32),      # npc_anim_idx
            training=False,
        )
    elif is_action_chunking:
        if state_shape is not None and use_anim_embeddings:
            # Action chunking with state + animation embeddings
            variables = model.init(
                {'params': init_rng, 'dropout': dropout_rng},
                jnp.ones(input_shape),                      # frames [B, T, C, H, W]
                state=jnp.ones(state_shape),                # continuous state
                hero_anim_idx=jnp.zeros((batch_size,), dtype=jnp.int32),
                npc_anim_idx=jnp.zeros((batch_size,), dtype=jnp.int32),
                training=False,
            )
        else:
            # Action chunking model: only takes frames as input
            variables = model.init(
                {'params': init_rng, 'dropout': dropout_rng},
                jnp.ones(input_shape),  # frames [B, T, C, H, W]
                training=False,
            )
    elif is_temporal:
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
    pos_weight: float = 1.0,
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
        pos_weight: Weight for positive class (>1 penalizes false negatives more)

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
        # BCE with pos_weight: penalize missed positives (FN) more than missed negatives (FP)
        # loss = -(pos_weight * y * log(p) + (1-y) * log(1-p))
        log_sigmoid = jax.nn.log_sigmoid(logits)
        log_one_minus_sigmoid = jax.nn.log_sigmoid(-logits)

        loss = -(pos_weight * labels * log_sigmoid + (1 - labels) * log_one_minus_sigmoid)

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
def _create_loss_fn(use_focal: bool, focal_gamma: float, focal_alpha: Optional[float], onset_weight: float = 1.0, pos_weight: float = 1.0):
    """Create a loss function with the specified configuration."""
    def loss_fn(logits, labels, weights=None, previous_actions=None):
        return compute_loss(logits, labels, weights, use_focal, focal_gamma, focal_alpha,
                           previous_actions, onset_weight, pos_weight)
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
        # Only if action history is non-empty (shape known at trace time)
        if batch['action_history'].shape[1] > 0:
            previous_actions = batch['action_history'][:, -1, :]
        else:
            previous_actions = None
        
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
        if batch['action_history'].shape[1] > 0:
            previous_actions = batch['action_history'][:, -1, :]
        else:
            previous_actions = None
        
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
        if batch['action_history'].shape[1] > 0:
            previous_actions = batch['action_history'][:, -1, :]
        else:
            previous_actions = None
        
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


# ============================================================================
# Action Chunking Transformer Training/Eval Steps
# ============================================================================

def compute_chunked_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
) -> jnp.ndarray:
    """Compute loss for action chunking (multi-step predictions).

    Args:
        logits: Model outputs [B, chunk_size, num_actions]
        labels: Ground truth [B, chunk_size, num_actions]
        weights: Optional per-action weights [num_actions]
        use_focal: Whether to use focal loss
        focal_gamma: Focal loss gamma parameter
        focal_alpha: Focal loss alpha parameter

    Returns:
        Loss value (scalar)
    """
    batch_size, chunk_size, num_actions = logits.shape

    # Reshape to [B * chunk_size, num_actions] for loss computation
    logits_flat = logits.reshape(-1, num_actions)
    labels_flat = labels.reshape(-1, num_actions)

    if use_focal:
        # Focal loss
        p = jax.nn.sigmoid(logits_flat)
        ce_loss = -(labels_flat * jax.nn.log_sigmoid(logits_flat) +
                    (1 - labels_flat) * jax.nn.log_sigmoid(-logits_flat))

        # Focal modulation
        p_t = labels_flat * p + (1 - labels_flat) * (1 - p)
        focal_weight = (1 - p_t) ** focal_gamma

        loss = focal_weight * ce_loss

        if focal_alpha is not None:
            alpha_t = labels_flat * focal_alpha + (1 - labels_flat) * (1 - focal_alpha)
            loss = alpha_t * loss
    else:
        # Standard BCE
        log_sigmoid = jax.nn.log_sigmoid(logits_flat)
        log_one_minus_sigmoid = jax.nn.log_sigmoid(-logits_flat)
        loss = -(labels_flat * log_sigmoid + (1 - labels_flat) * log_one_minus_sigmoid)

    if weights is not None:
        loss = loss * weights[None, :]

    return jnp.mean(loss)


# Module-level config for action chunking loss
_USE_FOCAL_LOSS_CHUNKED = False
_FOCAL_GAMMA_CHUNKED = 2.0
_FOCAL_ALPHA_CHUNKED = None


def _create_chunked_loss_fn(use_focal: bool, focal_gamma: float, focal_alpha: Optional[float]):
    """Create a loss function for action chunking."""
    def loss_fn(logits, labels, weights=None):
        return compute_chunked_loss(logits, labels, weights, use_focal, focal_gamma, focal_alpha)
    return loss_fn


_current_chunked_loss_fn = _create_chunked_loss_fn(False, 2.0, None)


def compute_chunked_accuracy(predictions: jnp.ndarray, targets: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    """Compute accuracy for chunked action predictions.

    Args:
        predictions: Predicted probabilities [B, chunk_size, num_actions]
        targets: Ground truth binary labels [B, chunk_size, num_actions]
        threshold: Threshold for binary classification

    Returns:
        Accuracy as JAX scalar (per-element accuracy across all predictions)
    """
    pred_binary = (predictions > threshold).astype(jnp.float32)
    correct = pred_binary == targets
    return jnp.mean(correct)


@jax.jit
def train_step_action_chunking(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Training step for action chunking transformer (vision-only).

    Args:
        state: TrainState
        batch: Batch with 'frames' [B, T, C, H, W] and 'actions' [B, chunk_size, num_actions]
        action_weights: Per-action loss weights
        rng: JAX random key

    Returns:
        (new_state, metrics)
    """
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        variables = {'params': params}

        # ActionChunkingTransformer uses LayerNorm (no batch_stats)
        logits = state.apply_fn(
            variables,
            batch['frames'],
            training=True,
            rngs={'dropout': dropout_rng},
        )

        loss = _current_chunked_loss_fn(
            logits,
            batch['actions'],
            weights=action_weights,
        )

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    predictions = jax.nn.sigmoid(logits)
    accuracy = compute_chunked_accuracy(predictions, batch['actions'])

    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def train_step_action_chunking_with_state(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Training step for action chunking transformer with state encoder.

    Args:
        state: TrainState
        batch: Batch with 'frames', 'actions', 'state', 'hero_anim_idx', 'npc_anim_idx'
        action_weights: Per-action loss weights
        rng: JAX random key

    Returns:
        (new_state, metrics)
    """
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        variables = {'params': params}

        logits = state.apply_fn(
            variables,
            batch['frames'],
            state=batch['state'],
            hero_anim_idx=batch['hero_anim_idx'],
            npc_anim_idx=batch['npc_anim_idx'],
            training=True,
            rngs={'dropout': dropout_rng},
        )

        loss = _current_chunked_loss_fn(
            logits,
            batch['actions'],
            weights=action_weights,
        )

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    predictions = jax.nn.sigmoid(logits)
    accuracy = compute_chunked_accuracy(predictions, batch['actions'])

    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def eval_step_action_chunking(state: TrainState, batch: Dict):
    """Eval step for action chunking transformer (vision-only)."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats

    logits = state.apply_fn(
        variables,
        batch['frames'],
        training=False
    )
    loss = _current_chunked_loss_fn(logits, batch['actions'])
    predictions = jax.nn.sigmoid(logits)

    return predictions, loss


@jax.jit
def eval_step_action_chunking_with_state(state: TrainState, batch: Dict):
    """Eval step for action chunking transformer with state encoder."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats

    logits = state.apply_fn(
        variables,
        batch['frames'],
        state=batch['state'],
        hero_anim_idx=batch['hero_anim_idx'],
        npc_anim_idx=batch['npc_anim_idx'],
        training=False
    )
    loss = _current_chunked_loss_fn(logits, batch['actions'])
    predictions = jax.nn.sigmoid(logits)

    return predictions, loss


# Global loss function for onset_offset models
_current_onset_offset_loss_fn = None


def _create_onset_offset_loss_fn(use_focal=False, focal_gamma=2.0, focal_alpha=None):
    """Create loss function for onset/offset predictions.

    Handles shape [B, chunk_size, num_actions, 2] where last dim is (onset, offset).
    """
    if use_focal:
        def loss_fn(logits, labels, weights=None):
            # logits: [B, chunk_size, num_actions, 2]
            # labels: [B, chunk_size, num_actions, 2]
            p = jax.nn.sigmoid(logits)
            p = jnp.clip(p, 1e-7, 1 - 1e-7)

            pt = labels * p + (1 - labels) * (1 - p)
            focal_weight = jnp.power(1 - pt, focal_gamma)

            bce_pos = -labels * jnp.log(p)
            bce_neg = -(1 - labels) * jnp.log(1 - p)

            if focal_alpha is not None:
                alpha_weight = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
                focal_loss = alpha_weight * focal_weight * (bce_pos + bce_neg)
            else:
                focal_loss = focal_weight * (bce_pos + bce_neg)

            # Apply per-action weights if provided [num_actions]
            if weights is not None:
                # Expand weights to match: [1, 1, num_actions, 1]
                weights_expanded = weights[None, None, :, None]
                focal_loss = focal_loss * weights_expanded

            return jnp.mean(focal_loss)
        return loss_fn
    else:
        def loss_fn(logits, labels, weights=None):
            log_sigmoid = jax.nn.log_sigmoid(logits)
            log_one_minus_sigmoid = jax.nn.log_sigmoid(-logits)
            bce = -labels * log_sigmoid - (1 - labels) * log_one_minus_sigmoid

            if weights is not None:
                weights_expanded = weights[None, None, :, None]
                bce = bce * weights_expanded

            return jnp.mean(bce)
        return loss_fn


@jax.jit
def train_step_onset_offset(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Training step for onset/offset transformer.

    Args:
        state: TrainState
        batch: Batch with 'frames' [B, T, C, H, W] and 'onset_offset' [B, chunk_size, num_actions, 2]
        action_weights: Per-action loss weights
        rng: JAX random key

    Returns:
        (new_state, metrics)
    """
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        variables = {'params': params}

        logits = state.apply_fn(
            variables,
            batch['frames'],
            training=True,
            rngs={'dropout': dropout_rng},
        )

        loss = _current_onset_offset_loss_fn(
            logits,
            batch['onset_offset'],
            weights=action_weights,
        )

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    # Compute accuracy for onset/offset predictions
    predictions = jax.nn.sigmoid(logits)
    pred_binary = (predictions > 0.5).astype(jnp.float32)
    correct = (pred_binary == batch['onset_offset']).astype(jnp.float32)
    accuracy = jnp.mean(correct)

    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def eval_step_onset_offset(state: TrainState, batch: Dict):
    """Eval step for onset/offset transformer."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats

    logits = state.apply_fn(
        variables,
        batch['frames'],
        training=False
    )
    loss = _current_onset_offset_loss_fn(logits, batch['onset_offset'])
    predictions = jax.nn.sigmoid(logits)

    return predictions, batch['onset_offset'], loss


# ============================================================================
# Action Combo Transformer Training/Eval Steps (softmax classification)
# ============================================================================

# Global loss function for action_combo models
_current_combo_loss_fn = None


def _create_combo_loss_fn(label_smoothing: float = 0.0):
    """Create cross-entropy loss function for combo classification.

    Args:
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Loss function that takes (logits, targets, weights) and returns scalar loss
    """
    def loss_fn(logits, targets, weights=None):
        """Compute cross-entropy loss for combo classification.

        Args:
            logits: [B, chunk_size, num_combos] raw logits
            targets: [B, chunk_size] int32 combo IDs
            weights: Optional [num_combos] class weights

        Returns:
            Scalar loss value
        """
        # Flatten for cross-entropy: [B * chunk_size, num_combos]
        batch_size, chunk_size, num_combos = logits.shape
        logits_flat = logits.reshape(-1, num_combos)
        targets_flat = targets.reshape(-1)

        # One-hot encode targets
        targets_one_hot = jax.nn.one_hot(targets_flat, num_combos)

        # Apply label smoothing
        if label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1.0 - label_smoothing) + label_smoothing / num_combos

        # Log softmax for numerical stability
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)

        # Cross-entropy loss
        loss_per_sample = -jnp.sum(targets_one_hot * log_probs, axis=-1)

        # Apply class weights if provided
        if weights is not None:
            sample_weights = weights[targets_flat]
            loss_per_sample = loss_per_sample * sample_weights

        return jnp.mean(loss_per_sample)

    return loss_fn


def train_step_action_combo(state: TrainState, batch: Dict, combo_weights: jnp.ndarray, rng):
    """Training step for action combo transformer (state-only) with cross-entropy loss.

    Args:
        state: TrainState
        batch: Batch with 'combo_ids', 'state', 'hero_anim_idx', 'npc_anim_idx'
        combo_weights: Per-combo class weights
        rng: JAX random key

    Returns:
        (new_state, metrics)
    """
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        variables = {'params': params}

        # State-only model: no frames input
        logits = state.apply_fn(
            variables,
            batch['state'],
            batch['hero_anim_idx'],
            batch['npc_anim_idx'],
            training=True,
            rngs={'dropout': dropout_rng},
        )

        loss = _current_combo_loss_fn(
            logits,
            batch['combo_ids'],
            weights=combo_weights,
        )

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    # Compute top-1 accuracy
    predictions = jnp.argmax(logits, axis=-1)  # [B, chunk_size]
    accuracy = jnp.mean(predictions == batch['combo_ids'])

    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def eval_step_action_combo(state: TrainState, batch: Dict):
    """Eval step for action combo transformer (state-only)."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats

    # State-only model: no frames input
    logits = state.apply_fn(
        variables,
        batch['state'],
        batch['hero_anim_idx'],
        batch['npc_anim_idx'],
        training=False
    )
    loss = _current_combo_loss_fn(logits, batch['combo_ids'])

    # Return logits and targets for metric computation
    return logits, batch['combo_ids'], loss


# Global loss function for state_chunking models
_current_state_chunking_loss_fn = None


def _create_state_chunking_loss_fn(use_focal=False, focal_gamma=2.0, focal_alpha=None, onset_weight=1.0):
    """Create loss function for state chunking predictions.

    Handles shape [B, chunk_size, num_actions] with binary action targets.
    Similar to action chunking loss but for state-only models.
    """
    if use_focal:
        def loss_fn(logits, labels, weights=None, prev_actions=None):
            # logits: [B, chunk_size, num_actions]
            # labels: [B, chunk_size, num_actions]
            p = jax.nn.sigmoid(logits)
            p = jnp.clip(p, 1e-7, 1 - 1e-7)

            pt = labels * p + (1 - labels) * (1 - p)
            focal_weight = jnp.power(1 - pt, focal_gamma)

            bce_pos = -labels * jnp.log(p)
            bce_neg = -(1 - labels) * jnp.log(1 - p)

            if focal_alpha is not None:
                alpha_weight = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
                focal_loss = alpha_weight * focal_weight * (bce_pos + bce_neg)
            else:
                focal_loss = focal_weight * (bce_pos + bce_neg)

            # Apply per-action weights if provided
            if weights is not None:
                weights_expanded = weights[None, None, :]  # [1, 1, num_actions]
                focal_loss = focal_loss * weights_expanded

            # Apply onset weighting if prev_actions provided
            if prev_actions is not None and onset_weight != 1.0:
                # First action in chunk compared to prev_actions
                first_actions = labels[:, 0, :]  # [B, num_actions]
                onset_mask = (first_actions > 0.5) & (prev_actions < 0.5)
                onset_weights = jnp.where(onset_mask, onset_weight, 1.0)
                # Apply to first timestep only
                focal_loss = focal_loss.at[:, 0, :].multiply(onset_weights)

            return jnp.mean(focal_loss)
        return loss_fn
    else:
        def loss_fn(logits, labels, weights=None, prev_actions=None):
            log_sigmoid = jax.nn.log_sigmoid(logits)
            log_one_minus_sigmoid = jax.nn.log_sigmoid(-logits)
            bce = -labels * log_sigmoid - (1 - labels) * log_one_minus_sigmoid

            if weights is not None:
                weights_expanded = weights[None, None, :]
                bce = bce * weights_expanded

            # Apply onset weighting if prev_actions provided
            if prev_actions is not None and onset_weight != 1.0:
                first_actions = labels[:, 0, :]
                onset_mask = (first_actions > 0.5) & (prev_actions < 0.5)
                onset_weights = jnp.where(onset_mask, onset_weight, 1.0)
                bce = bce.at[:, 0, :].multiply(onset_weights)

            return jnp.mean(bce)
        return loss_fn


@jax.jit
def train_step_state_chunking(state: TrainState, batch: Dict, action_weights: jnp.ndarray, rng):
    """Training step for state chunking transformer (state-only, temporal).

    Args:
        state: TrainState
        batch: Batch with 'states' [B, num_states, num_features], 'hero_anim_ids' [B, num_states],
               'npc_anim_ids' [B, num_states], and 'actions' [B, chunk_size, num_actions]
        action_weights: Per-action loss weights
        rng: JAX random key

    Returns:
        (new_state, metrics)
    """
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        variables = {'params': params}

        # StateChunkingTransformer: state-only with temporal input
        logits = state.apply_fn(
            variables,
            batch['states'],
            batch['hero_anim_ids'],
            batch['npc_anim_ids'],
            training=True,
            rngs={'dropout': dropout_rng},
        )

        loss = _current_state_chunking_loss_fn(
            logits,
            batch['actions'],
            weights=action_weights,
        )

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    predictions = jax.nn.sigmoid(logits)
    accuracy = compute_chunked_accuracy(predictions, batch['actions'])

    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def eval_step_state_chunking(state: TrainState, batch: Dict):
    """Eval step for state chunking transformer (state-only, temporal)."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats

    # State-only model with temporal input
    logits = state.apply_fn(
        variables,
        batch['states'],
        batch['hero_anim_ids'],
        batch['npc_anim_ids'],
        training=False
    )
    loss = _current_state_chunking_loss_fn(logits, batch['actions'])
    predictions = jax.nn.sigmoid(logits)

    return predictions, loss


def evaluate(
    state: TrainState,
    dataset,
    config: Dict,
    show_progress: bool = True,
    use_state: bool = False,
    is_temporal: bool = False,
    is_action_chunking: bool = False,
    is_onset_offset: bool = False,
    is_action_combo: bool = False,
    is_state_chunking: bool = False,
) -> Dict:
    """Evaluate model on dataset with JIT-compiled steps and progress bar.

    Args:
        state: TrainState
        dataset: Evaluation dataset
        config: Config dict
        show_progress: Whether to show progress bar
        use_state: Whether model uses state features
        is_temporal: Whether this is a temporal model
        is_action_chunking: Whether this is an action chunking model
        is_onset_offset: Whether this is an onset/offset model
        is_action_combo: Whether this is an action combo model
        is_state_chunking: Whether this is a state chunking model

    Returns:
        Dict of evaluation metrics
    """
    batch_size = config['training']['batch_size']

    # Select appropriate eval function and data loader
    if is_state_chunking:
        eval_fn = eval_step_state_chunking
        data_loader_fn = create_state_chunking_data_loader
    elif is_action_combo:
        eval_fn = eval_step_action_combo
        data_loader_fn = create_action_combo_data_loader
    elif is_onset_offset:
        eval_fn = eval_step_onset_offset
        data_loader_fn = create_onset_offset_data_loader
    elif is_action_chunking:
        # Check if dataset uses state (for ActionChunkingDataset with state)
        dataset_uses_state = getattr(dataset, 'use_state', False)
        eval_fn = eval_step_action_chunking_with_state if dataset_uses_state else eval_step_action_chunking
        data_loader_fn = create_action_chunking_data_loader
    elif is_temporal:
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
                if is_action_combo:
                    logits, targets, loss = eval_fn(state, batch)
                    all_predictions.append(np.asarray(logits))
                    all_targets.append(np.asarray(targets))
                elif is_onset_offset:
                    predictions, targets, loss = eval_fn(state, batch)
                    all_predictions.append(np.asarray(predictions))
                    all_targets.append(np.asarray(targets))
                else:
                    predictions, loss = eval_fn(state, batch)
                    all_predictions.append(np.asarray(predictions))
                    all_targets.append(np.asarray(batch['actions']))

                # Collect previous actions for onset metrics (temporal models have action_history)
                if is_temporal and 'action_history' in batch and batch['action_history'].shape[1] > 0:
                    # Last action in history = previous action
                    prev_actions = np.asarray(batch['action_history'][:, -1, :])
                    all_previous_actions.append(prev_actions)

                total_loss += float(loss)
                batches_processed += 1
                progress.update(task, advance=1)
    else:
        for batch in data_loader_fn(dataset, batch_size, shuffle=False, drop_last=False):
            if is_action_combo:
                logits, targets, loss = eval_fn(state, batch)
                all_predictions.append(np.asarray(logits))
                all_targets.append(np.asarray(targets))
            elif is_onset_offset:
                predictions, targets, loss = eval_fn(state, batch)
                all_predictions.append(np.asarray(predictions))
                all_targets.append(np.asarray(targets))
            else:
                predictions, loss = eval_fn(state, batch)
                all_predictions.append(np.asarray(predictions))
                all_targets.append(np.asarray(batch['actions']))

            if is_temporal and 'action_history' in batch and batch['action_history'].shape[1] > 0:
                prev_actions = np.asarray(batch['action_history'][:, -1, :])
                all_previous_actions.append(prev_actions)

            total_loss += float(loss)
            batches_processed += 1

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Handle different output shapes
    if is_action_combo:
        # Shape: all_predictions = [N, chunk_size, num_combos] (logits)
        # Shape: all_targets = [N, chunk_size] (int32 combo IDs)
        n_samples, chunk_size, num_combos = all_predictions.shape

        # Flatten predictions and targets
        logits_flat = all_predictions.reshape(-1, num_combos)  # [N*chunk, num_combos]
        targets_flat = all_targets.reshape(-1)  # [N*chunk]

        # Compute top-1 accuracy
        predicted_ids = np.argmax(logits_flat, axis=-1)
        top1_accuracy = float(np.mean(predicted_ids == targets_flat))

        # Compute top-k accuracy
        top_k_list = config.get('evaluation', {}).get('top_k', [1, 3, 5])
        top_k_accuracies = {}
        for k in top_k_list:
            if k <= num_combos:
                # Get top-k predicted indices
                top_k_indices = np.argsort(logits_flat, axis=-1)[:, -k:]  # [N*chunk, k]
                # Check if true label is in top-k
                matches = np.any(top_k_indices == targets_flat[:, None], axis=-1)
                top_k_accuracies[f'top_{k}_accuracy'] = float(np.mean(matches))

        avg_loss = total_loss / batches_processed

        metrics = {
            'loss': float(avg_loss),
            'accuracy': top1_accuracy,
            **top_k_accuracies,
        }
        return metrics

    elif is_onset_offset:
        # Shape: [N, chunk_size, num_actions, 2] -> compute metrics for onset and offset separately
        n_samples, chunk_size, num_actions, _ = all_predictions.shape

        # Split onset and offset
        onset_preds = all_predictions[..., 0].reshape(-1, num_actions)  # [N*chunk, num_actions]
        onset_targets = all_targets[..., 0].reshape(-1, num_actions)
        offset_preds = all_predictions[..., 1].reshape(-1, num_actions)
        offset_targets = all_targets[..., 1].reshape(-1, num_actions)

        # Compute metrics for onset
        onset_metrics = compute_per_action_metrics(onset_preds, onset_targets, threshold=0.5)
        # Compute metrics for offset
        offset_metrics = compute_per_action_metrics(offset_preds, offset_targets, threshold=0.5)

        avg_loss = total_loss / batches_processed

        # Compute mean metrics for onset and offset
        onset_precision = float(np.mean(onset_metrics['per_action_precision']))
        onset_recall = float(np.mean(onset_metrics['per_action_recall']))
        onset_f1 = float(np.mean(onset_metrics['per_action_f1_score']))

        offset_precision = float(np.mean(offset_metrics['per_action_precision']))
        offset_recall = float(np.mean(offset_metrics['per_action_recall']))
        offset_f1 = float(np.mean(offset_metrics['per_action_f1_score']))

        metrics = {
            'loss': float(avg_loss),
            'onset_precision': onset_precision,
            'onset_recall': onset_recall,
            'onset_f1': onset_f1,
            'offset_precision': offset_precision,
            'offset_recall': offset_recall,
            'offset_f1': offset_f1,
            # Overall averages
            'precision': (onset_precision + offset_precision) / 2,
            'recall': (onset_recall + offset_recall) / 2,
            'f1': (onset_f1 + offset_f1) / 2,
            # Per-action metrics for onset
            'onset_per_action_precision': onset_metrics['per_action_precision'],
            'onset_per_action_recall': onset_metrics['per_action_recall'],
            'onset_per_action_f1_score': onset_metrics['per_action_f1_score'],
            # Per-action metrics for offset
            'offset_per_action_precision': offset_metrics['per_action_precision'],
            'offset_per_action_recall': offset_metrics['per_action_recall'],
            'offset_per_action_f1_score': offset_metrics['per_action_f1_score'],
            # Standard per_action for compatibility (use onset)
            'per_action_precision': onset_metrics['per_action_precision'],
            'per_action_recall': onset_metrics['per_action_recall'],
            'per_action_f1_score': onset_metrics['per_action_f1_score'],
        }
        return metrics

    # For action chunking and state chunking, reshape from [N, chunk_size, num_actions] to [N*chunk_size, num_actions]
    if (is_action_chunking or is_state_chunking) and len(all_predictions.shape) == 3:
        n_samples, chunk_size, num_actions = all_predictions.shape
        all_predictions = all_predictions.reshape(-1, num_actions)
        all_targets = all_targets.reshape(-1, num_actions)

    # Compute metrics - simplified: only loss + per-action precision/recall/f1
    avg_loss = total_loss / batches_processed
    per_action_metrics = compute_per_action_metrics(all_predictions, all_targets, threshold=0.5)

    # Compute mean precision, recall, f1
    mean_precision = float(np.mean(per_action_metrics['per_action_precision']))
    mean_recall = float(np.mean(per_action_metrics['per_action_recall']))
    mean_f1 = float(np.mean(per_action_metrics['per_action_f1_score']))

    # Compute per-action distributions for logging
    # Target distribution: mean activation per action (how often each action is active in data)
    target_distribution = np.mean(all_targets, axis=0)  # [num_actions]
    # Prediction distribution: mean predicted probability per action
    pred_distribution = np.mean(all_predictions, axis=0)  # [num_actions]
    # Binary prediction distribution (after thresholding)
    binary_preds = (all_predictions > 0.5).astype(np.float32)
    binary_pred_distribution = np.mean(binary_preds, axis=0)  # [num_actions]

    metrics = {
        'loss': float(avg_loss),
        'precision': mean_precision,
        'recall': mean_recall,
        'f1': mean_f1,
        **per_action_metrics,
        # Distributions for wandb logging
        '_target_distribution': target_distribution,
        '_pred_distribution': pred_distribution,
        '_binary_pred_distribution': binary_pred_distribution,
        '_all_predictions': all_predictions,  # Raw predictions for histogram
        '_all_targets': all_targets,  # Raw targets for histogram
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
    
    # Check if this is a temporal model, action chunking, onset_offset, action_combo, or state_chunking model
    model_name = config['model']['name']
    is_temporal = model_name in ['temporal_cnn', 'gru', 'causal_transformer']
    is_action_chunking = model_name == 'action_chunking_transformer'
    is_onset_offset = model_name == 'onset_offset_transformer'
    is_action_combo = model_name == 'action_combo_transformer'
    is_state_chunking = model_name == 'state_chunking_transformer'

    if is_state_chunking:
        # State chunking dataset (state-only, temporal) - predicts binary actions
        state_chunk_config = config.get('state_chunking', {})
        num_states = state_chunk_config.get('num_states', 8)
        chunk_size = state_chunk_config.get('chunk_size', 8)

        train_dataset = StateChunkingDataset(
            dataset_path=dataset_path,
            episode_indices=train_indices,
            validate_episodes=config['dataset'].get('validate_episodes', True),
            num_states=num_states,
            chunk_size=chunk_size,
            state_preprocessor=state_preprocessor,
        )

        val_dataset = StateChunkingDataset(
            dataset_path=dataset_path,
            episode_indices=val_indices,
            validate_episodes=config['dataset'].get('validate_episodes', True),
            num_states=num_states,
            chunk_size=chunk_size,
            state_preprocessor=state_preprocessor,
        )

        console.print(f"[cyan]State chunking config: {num_states} temporal states -> {chunk_size} future actions[/cyan]")
        console.print(f"[cyan]Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}[/cyan]")

    elif is_action_combo:
        # Action combo dataset (state-only) - discovers unique combos and predicts combo IDs
        combo_config = config.get('action_combo', {})
        chunk_size = combo_config.get('chunk_size', 8)

        train_dataset = ActionComboDataset(
            dataset_path=dataset_path,
            episode_indices=train_indices,
            validate_episodes=config['dataset'].get('validate_episodes', True),
            chunk_size=chunk_size,
            state_preprocessor=state_preprocessor,
            combo_mapping=None,  # Will discover combos from training set
        )

        # Get combo mapping from training set for validation set
        combo_mapping = train_dataset.get_combo_mapping()
        console.print(f"[cyan]Discovered {combo_mapping['num_combos']} unique action combos from training set[/cyan]")

        val_dataset = ActionComboDataset(
            dataset_path=dataset_path,
            episode_indices=val_indices,
            validate_episodes=config['dataset'].get('validate_episodes', True),
            chunk_size=chunk_size,
            state_preprocessor=state_preprocessor,
            combo_mapping=combo_mapping,  # Use training set's combo mapping
        )
    elif is_onset_offset:
        # Onset/offset dataset for predicting action transitions
        onset_offset_config = config.get('onset_offset', {})
        num_frames = onset_offset_config.get('num_frames', 8)
        chunk_size = onset_offset_config.get('chunk_size', 8)

        train_dataset = OnsetOffsetDataset(
            dataset_path=dataset_path,
            episode_indices=train_indices,
            normalize_frames=config['dataset']['normalize_frames'],
            validate_episodes=config['dataset'].get('validate_episodes', True),
            num_frames=num_frames,
            chunk_size=chunk_size,
        )

        val_dataset = OnsetOffsetDataset(
            dataset_path=dataset_path,
            episode_indices=val_indices,
            normalize_frames=config['dataset']['normalize_frames'],
            validate_episodes=config['dataset'].get('validate_episodes', True),
            num_frames=num_frames,
            chunk_size=chunk_size,
        )
    elif is_action_chunking:
        # Action chunking dataset with frame sequences and future action chunks
        chunking_config = config.get('action_chunking', {})
        num_frames = chunking_config.get('num_frames', 4)
        chunk_size = chunking_config.get('chunk_size', 16)
        use_state_chunking = config['dataset'].get('use_state', False)

        train_dataset = ActionChunkingDataset(
            dataset_path=dataset_path,
            episode_indices=train_indices,
            normalize_frames=config['dataset']['normalize_frames'],
            validate_episodes=config['dataset'].get('validate_episodes', True),
            num_frames=num_frames,
            chunk_size=chunk_size,
            use_state=use_state_chunking,
            state_preprocessor=state_preprocessor if use_state_chunking else None,
        )

        val_dataset = ActionChunkingDataset(
            dataset_path=dataset_path,
            episode_indices=val_indices,
            normalize_frames=config['dataset']['normalize_frames'],
            validate_episodes=config['dataset'].get('validate_episodes', True),
            num_frames=num_frames,
            chunk_size=chunk_size,
            use_state=use_state_chunking,
            state_preprocessor=state_preprocessor if use_state_chunking else None,
        )
    elif is_temporal:
        # Temporal dataset with frame stacking and action history
        temporal_config = config.get('temporal', {})
        num_history_frames = temporal_config.get('num_history_frames', 4)
        num_action_history = temporal_config.get('num_action_history', 4)
        frame_skip = temporal_config.get('frame_skip', 1)

        # Parse oversampling config
        oversample_config = config['dataset'].get('oversample', {})
        oversample_actions = None
        oversample_ratio = 1.0

        if oversample_config:
            oversample_ratio = oversample_config.get('ratio', 1.0)
            raw_actions = oversample_config.get('actions', [])

            if raw_actions:
                # Get action names from dataset to resolve string names to indices
                temp_zarr = zarr.open(str(dataset_path), mode='r')
                action_keys = temp_zarr.attrs.get('keys', [])

                oversample_actions = []
                for action in raw_actions:
                    if isinstance(action, int):
                        oversample_actions.append(action)
                    elif isinstance(action, str):
                        if action in action_keys:
                            oversample_actions.append(action_keys.index(action))
                        else:
                            logger.warning(f"Unknown action name for oversampling: {action}")
                    else:
                        logger.warning(f"Invalid action specifier: {action}")

                if oversample_actions:
                    action_names = [action_keys[i] for i in oversample_actions]
                    console.print(f"[cyan]Oversampling actions: {action_names} (indices: {oversample_actions}) with ratio {oversample_ratio}[/cyan]")

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
            oversample_actions=oversample_actions,
            oversample_ratio=oversample_ratio,
        )

        # No oversampling for validation set
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
    
    # Get action info
    num_actions = len(train_dataset.action_keys)
    action_names = load_action_names(train_dataset.action_keys)  # Semantic names from keybinds
    console.print(f"[cyan]Actions ({num_actions}): {', '.join(action_names)}[/cyan]")
    
    # Compute action/combo weights
    console.print("[cyan]Computing class weights...[/cyan]")
    if is_action_combo:
        # Action combo uses combo weights (for softmax classification)
        if config['training']['use_class_weights']:
            action_weights = train_dataset.compute_combo_weights()
            action_weights = jnp.array(action_weights)
            console.print(f"[cyan]Combo weights computed for {train_dataset.num_combos} combos[/cyan]")
        else:
            action_weights = jnp.ones(train_dataset.num_combos)
    else:
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
    elif model_name == 'action_chunking_transformer':
        from models.action_chunking_transformer import create_model

        # Get action chunking config
        chunking_config = config.get('action_chunking', {})
        num_frames = chunking_config.get('num_frames', 4)
        chunk_size = chunking_config.get('chunk_size', 16)

        # Transformer specific config
        d_model = config['model'].get('d_model', 512)
        num_heads = config['model'].get('num_heads', 8)
        num_encoder_layers = config['model'].get('num_encoder_layers', 4)
        num_decoder_layers = config['model'].get('num_decoder_layers', 4)
        d_ff = config['model'].get('d_ff', 2048)
        dropout_rate = config['model'].get('dropout_rate', 0.1)

        # State encoder config
        use_state_model = config['dataset'].get('use_state', False)
        num_state_features = state_preprocessor.continuous_dim if use_state_model else 10
        anim_embed_dim = config.get('state_preprocessing', {}).get('anim_embed_dim', 16)

        console.print(f"[cyan]Action Chunking Transformer config:[/cyan]")
        console.print(f"[cyan]  d_model={d_model}, heads={num_heads}[/cyan]")
        console.print(f"[cyan]  encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}[/cyan]")
        console.print(f"[cyan]  d_ff={d_ff}, dropout={dropout_rate}[/cyan]")
        console.print(f"[cyan]  Input: {num_frames} frames -> Output: {chunk_size} action chunks[/cyan]")
        if use_state_model:
            console.print(f"[cyan]  State encoder: enabled ({num_state_features} continuous features)[/cyan]")

        model = create_model(
            num_actions=num_actions,
            num_frames=num_frames,
            chunk_size=chunk_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            use_state=use_state_model,
            num_state_features=num_state_features,
            state_hidden_features=tuple(config['model'].get('state_encoder_features', [64, 64])),
            state_output_features=config['model'].get('state_output_features', 64),
            hero_anim_vocab_size=state_preprocessor.hero_vocab_size if use_state_model else 67,
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size if use_state_model else 54,
            anim_embed_dim=anim_embed_dim,
        )
    elif model_name == 'onset_offset_transformer':
        from models.onset_offset_transformer import create_model

        # Get onset_offset config
        onset_offset_config = config.get('onset_offset', {})
        num_frames = onset_offset_config.get('num_frames', 8)
        chunk_size = onset_offset_config.get('chunk_size', 8)

        # Transformer specific config
        d_model = config['model'].get('d_model', 512)
        num_heads = config['model'].get('num_heads', 8)
        num_encoder_layers = config['model'].get('num_encoder_layers', 4)
        num_decoder_layers = config['model'].get('num_decoder_layers', 4)
        d_ff = config['model'].get('d_ff', 2048)
        dropout_rate = config['model'].get('dropout_rate', 0.1)

        console.print(f"[cyan]Onset/Offset Transformer config:[/cyan]")
        console.print(f"[cyan]  d_model={d_model}, heads={num_heads}[/cyan]")
        console.print(f"[cyan]  encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}[/cyan]")
        console.print(f"[cyan]  d_ff={d_ff}, dropout={dropout_rate}[/cyan]")
        console.print(f"[cyan]  Input: {num_frames} frames -> Output: {chunk_size} onset/offset predictions[/cyan]")

        model = create_model(
            num_actions=num_actions,
            num_frames=num_frames,
            chunk_size=chunk_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
        )
    elif model_name == 'state_chunking_transformer':
        from models.state_chunking_transformer import create_model

        # Get state chunking config (state-only, temporal)
        state_chunk_config = config.get('state_chunking', {})
        num_states = state_chunk_config.get('num_states', 8)
        chunk_size = state_chunk_config.get('chunk_size', 8)

        # Transformer specific config
        d_model = config['model'].get('d_model', 512)
        num_heads = config['model'].get('num_heads', 8)
        num_encoder_layers = config['model'].get('num_encoder_layers', 4)
        num_decoder_layers = config['model'].get('num_decoder_layers', 4)
        d_ff = config['model'].get('d_ff', 2048)
        dropout_rate = config['model'].get('dropout_rate', 0.1)

        # State encoder config
        num_state_features = state_preprocessor.continuous_dim
        anim_embed_dim = config.get('state_preprocessing', {}).get('anim_embed_dim', 16)

        console.print(f"[cyan]State Chunking Transformer config (STATE-ONLY, temporal):[/cyan]")
        console.print(f"[cyan]  d_model={d_model}, heads={num_heads}[/cyan]")
        console.print(f"[cyan]  encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}[/cyan]")
        console.print(f"[cyan]  d_ff={d_ff}, dropout={dropout_rate}[/cyan]")
        console.print(f"[cyan]  Input: {num_states} temporal states[/cyan]")
        console.print(f"[cyan]  Output: {chunk_size} future actions x {train_dataset.num_actions} actions[/cyan]")
        console.print(f"[cyan]  State encoder: {num_state_features} continuous features[/cyan]")

        model = create_model(
            num_actions=train_dataset.num_actions,
            num_states=num_states,
            chunk_size=chunk_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            num_state_features=num_state_features,
            state_hidden_features=tuple(config['model'].get('state_encoder_features', [64, 64])),
            state_output_features=config['model'].get('state_output_features', 64),
            hero_anim_vocab_size=state_preprocessor.hero_vocab_size,
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size,
            anim_embed_dim=anim_embed_dim,
        )
    elif model_name == 'action_combo_transformer':
        from models.action_combo_transformer import create_model

        # Get action combo config (state-only model - no frames)
        combo_config = config.get('action_combo', {})
        chunk_size = combo_config.get('chunk_size', 8)

        # Transformer specific config
        d_model = config['model'].get('d_model', 512)
        num_heads = config['model'].get('num_heads', 8)
        num_encoder_layers = config['model'].get('num_encoder_layers', 4)
        num_decoder_layers = config['model'].get('num_decoder_layers', 4)
        d_ff = config['model'].get('d_ff', 2048)
        dropout_rate = config['model'].get('dropout_rate', 0.1)

        # State encoder config (always enabled for action_combo)
        num_state_features = state_preprocessor.continuous_dim
        anim_embed_dim = config.get('state_preprocessing', {}).get('anim_embed_dim', 16)

        # Get num_combos from training dataset
        num_combos = train_dataset.num_combos

        console.print(f"[cyan]Action Combo Transformer config (STATE-ONLY):[/cyan]")
        console.print(f"[cyan]  d_model={d_model}, heads={num_heads}[/cyan]")
        console.print(f"[cyan]  encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}[/cyan]")
        console.print(f"[cyan]  d_ff={d_ff}, dropout={dropout_rate}[/cyan]")
        console.print(f"[cyan]  Output: {chunk_size} combo predictions[/cyan]")
        console.print(f"[cyan]  Num combos (classes): {num_combos}[/cyan]")
        console.print(f"[cyan]  State encoder: {num_state_features} continuous features[/cyan]")

        model = create_model(
            num_combos=num_combos,
            chunk_size=chunk_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            num_state_features=num_state_features,
            state_hidden_features=tuple(config['model'].get('state_encoder_features', [64, 64])),
            state_output_features=config['model'].get('state_output_features', 64),
            hero_anim_vocab_size=state_preprocessor.hero_vocab_size,
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size,
            anim_embed_dim=anim_embed_dim,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get input shape from dataset
    sample = train_dataset[0]
    if is_action_combo or is_state_chunking:
        # State-only model - no frames
        input_shape = (config['training']['batch_size'],)  # Dummy shape, not used
        console.print(f"[cyan]State-only model (no frame input)[/cyan]")
    else:
        input_shape = (config['training']['batch_size'],) + sample['frames'].shape
        console.print(f"[cyan]Input shape: {input_shape}[/cyan]")

    # Get state shape if using state (action_combo/state_chunking always use state)
    state_shape = None
    if is_state_chunking and 'states' in sample:
        # State chunking uses temporal states: [B, num_states, num_features]
        state_shape = (config['training']['batch_size'],) + sample['states'].shape
        console.print(f"[cyan]Temporal state shape: {state_shape}[/cyan]")
    elif (use_state or is_action_combo) and 'state' in sample:
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
    use_anim_embeddings = (model_name == 'hybrid_state') or (model_name in ['temporal_cnn', 'gru', 'causal_transformer'] and use_state) or (model_name == 'action_chunking_transformer' and use_state) or (model_name == 'action_combo_transformer') or (model_name == 'state_chunking_transformer')

    # Get num_states for state_chunking
    num_temporal_states = config.get('state_chunking', {}).get('num_states', 8) if is_state_chunking else 8

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
        is_action_chunking=is_action_chunking or is_onset_offset,
        is_state_only=is_action_combo,  # Action combo is state-only (no frames)
        is_state_chunking=is_state_chunking,  # State chunking is temporal state-only
        num_temporal_states=num_temporal_states,
    )
    console.print("[green]✓ Model initialized and ready![/green]")
    
    # Training loop
    console.print("\n[bold cyan]Starting training...[/bold cyan]")
    best_val_accuracy = 0.0
    
    # Configure loss function BEFORE any JIT compilation
    global _current_loss_fn, _current_chunked_loss_fn, _current_onset_offset_loss_fn, _current_combo_loss_fn, _current_state_chunking_loss_fn
    loss_config = config.get('training', {}).get('loss', {})
    loss_type = loss_config.get('type', 'bce')
    onset_weight = loss_config.get('onset_weight', 1.0)  # Weight for onset frames
    pos_weight = loss_config.get('pos_weight', 1.0)  # Weight for positive class (penalize FN more)

    if is_state_chunking:
        # State chunking uses BCE or focal loss for binary actions
        focal_gamma = loss_config.get('focal_gamma', 2.0)
        use_focal = loss_type == 'focal'
        _current_state_chunking_loss_fn = _create_state_chunking_loss_fn(
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            focal_alpha=loss_config.get('alpha', None),
            onset_weight=onset_weight,
        )
        if use_focal:
            console.print(f"[cyan]Using Focal Loss for state chunking (γ={focal_gamma}, onset_weight={onset_weight})[/cyan]")
        else:
            console.print(f"[cyan]Using BCE Loss for state chunking (onset_weight={onset_weight})[/cyan]")
    elif is_action_combo:
        # Action combo uses cross-entropy loss
        label_smoothing = loss_config.get('label_smoothing', 0.0)
        _current_combo_loss_fn = _create_combo_loss_fn(label_smoothing=label_smoothing)
        console.print(f"[cyan]Using Cross-Entropy Loss for combo classification (label_smoothing={label_smoothing})[/cyan]")
    elif loss_type == 'focal':
        focal_gamma = loss_config.get('gamma', 2.0)
        focal_alpha = loss_config.get('alpha', None)
        _current_loss_fn = _create_loss_fn(use_focal=True, focal_gamma=focal_gamma, focal_alpha=focal_alpha, onset_weight=onset_weight)
        _current_chunked_loss_fn = _create_chunked_loss_fn(use_focal=True, focal_gamma=focal_gamma, focal_alpha=focal_alpha)
        _current_onset_offset_loss_fn = _create_onset_offset_loss_fn(use_focal=True, focal_gamma=focal_gamma, focal_alpha=focal_alpha)
        console.print(f"[cyan]Using Focal Loss (γ={focal_gamma}, α={focal_alpha}, onset_weight={onset_weight})[/cyan]")
    else:
        _current_loss_fn = _create_loss_fn(use_focal=False, focal_gamma=2.0, focal_alpha=None, onset_weight=onset_weight, pos_weight=pos_weight)
        _current_chunked_loss_fn = _create_chunked_loss_fn(use_focal=False, focal_gamma=2.0, focal_alpha=None)
        _current_onset_offset_loss_fn = _create_onset_offset_loss_fn(use_focal=False, focal_gamma=2.0, focal_alpha=None)
        if pos_weight > 1.0:
            console.print(f"[cyan]Using BCE Loss with pos_weight={pos_weight} (penalizes missed positives {pos_weight}x more)[/cyan]")
        elif onset_weight > 1.0:
            console.print(f"[cyan]Using BCE Loss with onset weighting ({onset_weight}x)[/cyan]")
        else:
            console.print("[cyan]Using Binary Cross Entropy Loss[/cyan]")

    # Calculate total steps
    batch_size = config['training']['batch_size']
    steps_per_epoch = len(train_dataset) // batch_size

    # Select appropriate step functions and data loader based on model type
    if is_state_chunking:
        train_fn = train_step_state_chunking
        data_loader_fn = create_state_chunking_data_loader
    elif is_action_combo:
        train_fn = train_step_action_combo
        data_loader_fn = create_action_combo_data_loader
    elif is_onset_offset:
        train_fn = train_step_onset_offset
        data_loader_fn = create_onset_offset_data_loader
    elif is_action_chunking:
        # Check if using state for action chunking
        use_state_chunking = train_dataset.use_state
        train_fn = train_step_action_chunking_with_state if use_state_chunking else train_step_action_chunking
        data_loader_fn = create_action_chunking_data_loader
    elif is_temporal:
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
                progress.update(
                    batch_task,
                    advance=1,
                    description=f"[green]Epoch {epoch+1} │ Loss: {current_loss:.4f}"
                )

                # Log step metrics
                if config['logging']['use_wandb'] and step % config['logging']['log_every_n_steps'] == 0:
                    wandb.log({
                        'train/step_loss': float(metrics['loss']),
                        'step': int(state.step),
                    })

            # Aggregate training metrics
            avg_train_loss = np.mean([m['loss'] for m in train_metrics])

            # Evaluation
            if (epoch + 0) % config['evaluation']['eval_every_n_epochs'] == 0:
                progress.stop()  # Pause progress bar for evaluation output

                console.print(f"\n[bold yellow]Evaluating at epoch {epoch + 1}...[/bold yellow]")
                val_metrics = evaluate(state, val_dataset, config, use_state=use_state, is_temporal=is_temporal, is_action_chunking=is_action_chunking, is_onset_offset=is_onset_offset, is_action_combo=is_action_combo, is_state_chunking=is_state_chunking)

                # Create summary table
                table = Table(title=f"Epoch {epoch + 1}/{config['training']['num_epochs']} Summary", show_header=True, header_style="bold magenta")
                table.add_column("Split", style="cyan")
                table.add_column("Loss", justify="right")

                if is_action_combo:
                    # Action combo uses accuracy metrics
                    table.add_column("Top-1 Acc", justify="right")
                    table.add_column("Top-3 Acc", justify="right")
                    table.add_column("Top-5 Acc", justify="right")
                    table.add_row("Train", f"{avg_train_loss:.4f}", "-", "-", "-")
                    top1 = val_metrics.get('accuracy', 0)
                    top3 = val_metrics.get('top_3_accuracy', 0)
                    top5 = val_metrics.get('top_5_accuracy', 0)
                    table.add_row("Val", f"{val_metrics['loss']:.4f}", f"{top1:.4f}", f"{top3:.4f}", f"{top5:.4f}")
                else:
                    table.add_column("Precision", justify="right")
                    table.add_column("Recall", justify="right")
                    table.add_column("F1", justify="right")
                    table.add_row("Train", f"{avg_train_loss:.4f}", "-", "-", "-")
                    table.add_row("Val", f"{val_metrics['loss']:.4f}", f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}", f"{val_metrics['f1']:.4f}")

                console.print(table)

                # Print per-action metrics table (not for action_combo)
                if not is_action_combo:
                    action_table = Table(title="Per-Action Validation Metrics", show_header=True, header_style="bold cyan")
                    action_table.add_column("Key", style="cyan")
                    action_table.add_column("Precision", justify="right")
                    action_table.add_column("Recall", justify="right")
                    action_table.add_column("F1", justify="right")
                    for i, name in enumerate(action_names):
                        prec = val_metrics['per_action_precision'][i]
                        rec = val_metrics['per_action_recall'][i]
                        f1 = val_metrics['per_action_f1_score'][i]
                        action_table.add_row(name, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}")
                    console.print(action_table)

                progress.start()  # Resume progress bar

                # Log to wandb
                if config['logging']['use_wandb']:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train/loss': avg_train_loss,
                        'val/loss': val_metrics['loss'],
                    }

                    if is_action_combo:
                        # Action combo uses accuracy metrics
                        log_dict['val/accuracy'] = val_metrics.get('accuracy', 0)
                        log_dict['val/top_3_accuracy'] = val_metrics.get('top_3_accuracy', 0)
                        log_dict['val/top_5_accuracy'] = val_metrics.get('top_5_accuracy', 0)
                    else:
                        log_dict['val/precision'] = val_metrics['precision']
                        log_dict['val/recall'] = val_metrics['recall']
                        log_dict['val/f1'] = val_metrics['f1']

                        # Add per-action metrics: val_{KEY}/precision, val_{KEY}/recall, val_{KEY}/f1
                        for i, name in enumerate(action_names):
                            log_dict[f'val_{name}/precision'] = float(val_metrics['per_action_precision'][i])
                            log_dict[f'val_{name}/recall'] = float(val_metrics['per_action_recall'][i])
                            log_dict[f'val_{name}/f1'] = float(val_metrics['per_action_f1_score'][i])

                        # Add distribution logging
                        if '_target_distribution' in val_metrics:
                            # Bar chart data for target vs prediction distribution
                            target_dist = val_metrics['_target_distribution']
                            pred_dist = val_metrics['_pred_distribution']
                            binary_pred_dist = val_metrics['_binary_pred_distribution']

                            # Log per-action activation rates
                            for i, name in enumerate(action_names):
                                log_dict[f'dist_{name}/target_rate'] = float(target_dist[i])
                                log_dict[f'dist_{name}/pred_prob_mean'] = float(pred_dist[i])
                                log_dict[f'dist_{name}/pred_binary_rate'] = float(binary_pred_dist[i])

                            # Log histograms of prediction probabilities per action
                            all_preds = val_metrics.get('_all_predictions')
                            if all_preds is not None:
                                for i, name in enumerate(action_names):
                                    log_dict[f'hist_{name}/pred_probs'] = wandb.Histogram(all_preds[:, i])

                    wandb.log(log_dict)

                # Save best model
                if is_action_combo:
                    # Action combo: use top-1 accuracy
                    current_metric = val_metrics.get('accuracy', 0)
                    if current_metric > best_val_accuracy:
                        best_val_accuracy = current_metric
                        save_checkpoint(checkpoint_dir, state, 'best')
                        console.print(f"[bold green]✓ Saved best model with val accuracy: {best_val_accuracy:.4f}[/bold green]")
                else:
                    # Other models: use F1 score
                    if val_metrics['f1'] > best_val_accuracy:
                        best_val_accuracy = val_metrics['f1']
                        save_checkpoint(checkpoint_dir, state, 'best')
                        console.print(f"[bold green]✓ Saved best model with val F1: {best_val_accuracy:.4f}[/bold green]")
            
            # Save periodic checkpoint
            if (epoch + 1) % config['training']['save_every_n_epochs'] == 0:
                save_checkpoint(checkpoint_dir, state, f'epoch_{epoch + 1}')
                console.print(f"[cyan]Saved checkpoint at epoch {epoch + 1}[/cyan]")
            
            epoch_time = time.time() - epoch_start_time
            
            # Update epoch progress
            progress.update(epoch_task, advance=1)
    
    console.print("\n[bold green]✓ Training completed![/bold green]")
    if is_action_combo:
        console.print(f"[bold cyan]Best validation accuracy: {best_val_accuracy:.4f}[/bold cyan]")
    else:
        console.print(f"[bold cyan]Best validation F1: {best_val_accuracy:.4f}[/bold cyan]")
    
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

