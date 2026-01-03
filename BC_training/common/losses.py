"""Loss functions for behavior cloning training."""

import jax
import jax.numpy as jnp
from typing import Optional, Literal


def compute_onset_weights(
    labels: jnp.ndarray,
    previous_actions: jnp.ndarray,
    onset_weight: float = 1.0,
) -> jnp.ndarray:
    """Compute per-sample weights based on onset (action change) detection.
    
    Args:
        labels: Current frame labels [batch, num_actions]
        previous_actions: Previous frame actions [batch, num_actions]
        onset_weight: Weight multiplier for onset frames (>1 to upweight onsets)
        
    Returns:
        Per-sample weights [batch]
    """
    # Onset = any action changed from previous frame
    action_changed = jnp.any(labels != previous_actions, axis=1)  # [batch]
    
    # Weight: onset_weight for onset frames, 1.0 for non-onset frames
    sample_weights = jnp.where(action_changed, onset_weight, 1.0)
    
    return sample_weights


def binary_cross_entropy_with_logits(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    previous_actions: Optional[jnp.ndarray] = None,
    onset_weight: float = 1.0,
) -> jnp.ndarray:
    """Compute binary cross entropy loss from logits.
    
    Args:
        logits: Raw model outputs (before sigmoid) [batch, num_actions]
        labels: Binary target labels [batch, num_actions]
        weights: Optional per-class weights [num_actions]
        label_smoothing: Label smoothing factor
        previous_actions: Optional previous frame actions for onset weighting [batch, num_actions]
        onset_weight: Weight multiplier for onset frames (only used if previous_actions provided)
        
    Returns:
        Loss value (scalar)
    """
    # Apply label smoothing
    if label_smoothing > 0:
        labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
    
    # Stable BCE computation
    log_sigmoid = jax.nn.log_sigmoid(logits)
    log_one_minus_sigmoid = jax.nn.log_sigmoid(-logits)
    
    bce = -labels * log_sigmoid - (1 - labels) * log_one_minus_sigmoid
    
    # Apply class weights if provided [num_actions]
    if weights is not None:
        bce = bce * weights
    
    # Apply onset weighting if previous_actions provided [batch]
    if previous_actions is not None and onset_weight > 1.0:
        sample_weights = compute_onset_weights(labels, previous_actions, onset_weight)
        # Mean over actions first, then weighted mean over samples
        bce_per_sample = jnp.mean(bce, axis=1)  # [batch]
        return jnp.sum(bce_per_sample * sample_weights) / jnp.sum(sample_weights)
    
    return jnp.mean(bce)


def focal_loss_with_logits(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    previous_actions: Optional[jnp.ndarray] = None,
    onset_weight: float = 1.0,
) -> jnp.ndarray:
    """Compute focal loss from logits.
    
    Focal loss down-weights easy examples to focus on hard ones.
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    
    Args:
        logits: Raw model outputs (before sigmoid) [batch, num_actions]
        labels: Binary target labels [batch, num_actions]
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is equivalent to standard BCE.
               gamma=2 is the standard choice.
        alpha: Class balancing weight for positive class (0-1).
               If None, no class balancing is applied.
               alpha=0.25 is common for rare positive classes.
        weights: Optional per-class weights [num_actions]
        label_smoothing: Label smoothing factor
        previous_actions: Optional previous frame actions for onset weighting [batch, num_actions]
        onset_weight: Weight multiplier for onset frames (only used if previous_actions provided)
        
    Returns:
        Loss value (scalar)
    """
    # Apply label smoothing
    if label_smoothing > 0:
        labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
    
    # Get probabilities via sigmoid
    p = jax.nn.sigmoid(logits)
    
    # Clip for numerical stability
    p = jnp.clip(p, 1e-7, 1 - 1e-7)
    
    # Compute pt (probability of correct class)
    # pt = p if label=1, else (1-p) if label=0
    pt = labels * p + (1 - labels) * (1 - p)
    
    # Focal weight: (1 - pt)^gamma
    # When model is correct and confident (pt high), weight is low
    # When model is wrong or uncertain (pt low), weight is high
    focal_weight = jnp.power(1 - pt, gamma)
    
    # Standard BCE terms
    bce_pos = -labels * jnp.log(p)
    bce_neg = -(1 - labels) * jnp.log(1 - p)
    
    # Apply alpha balancing if specified
    if alpha is not None:
        # alpha for positive class, (1-alpha) for negative class
        alpha_weight = labels * alpha + (1 - labels) * (1 - alpha)
        focal_loss = alpha_weight * focal_weight * (bce_pos + bce_neg)
    else:
        focal_loss = focal_weight * (bce_pos + bce_neg)
    
    # Apply per-class weights if provided
    if weights is not None:
        focal_loss = focal_loss * weights
    
    # Apply onset weighting if previous_actions provided
    if previous_actions is not None and onset_weight > 1.0:
        sample_weights = compute_onset_weights(labels, previous_actions, onset_weight)
        # Mean over actions first, then weighted mean over samples
        loss_per_sample = jnp.mean(focal_loss, axis=1)  # [batch]
        return jnp.sum(loss_per_sample * sample_weights) / jnp.sum(sample_weights)
    
    return jnp.mean(focal_loss)


def get_loss_fn(
    loss_type: Literal["bce", "focal"] = "bce",
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    onset_weight: float = 1.0,
):
    """Get loss function based on configuration.
    
    Args:
        loss_type: Type of loss function ("bce" or "focal")
        gamma: Focal loss gamma parameter (ignored for BCE)
        alpha: Focal loss alpha parameter (ignored for BCE)
        onset_weight: Weight multiplier for onset frames (>1 to upweight action changes)
        
    Returns:
        Loss function with signature (logits, labels, weights, label_smoothing, previous_actions) -> loss
    """
    if loss_type == "focal":
        def loss_fn(logits, labels, weights=None, label_smoothing=0.0, previous_actions=None):
            return focal_loss_with_logits(
                logits, labels, 
                gamma=gamma, 
                alpha=alpha,
                weights=weights,
                label_smoothing=label_smoothing,
                previous_actions=previous_actions,
                onset_weight=onset_weight,
            )
        return loss_fn
    else:
        def loss_fn(logits, labels, weights=None, label_smoothing=0.0, previous_actions=None):
            return binary_cross_entropy_with_logits(
                logits, labels,
                weights=weights,
                label_smoothing=label_smoothing,
                previous_actions=previous_actions,
                onset_weight=onset_weight,
            )
        return loss_fn


# Convenience functions for common configurations
def focal_loss_gamma2(logits, labels, weights=None, label_smoothing=0.0, previous_actions=None, onset_weight=1.0):
    """Focal loss with gamma=2 (standard setting)."""
    return focal_loss_with_logits(logits, labels, gamma=2.0, weights=weights, label_smoothing=label_smoothing,
                                  previous_actions=previous_actions, onset_weight=onset_weight)


def focal_loss_gamma2_alpha25(logits, labels, weights=None, label_smoothing=0.0, previous_actions=None, onset_weight=1.0):
    """Focal loss with gamma=2, alpha=0.25 (for rare positive classes)."""
    return focal_loss_with_logits(logits, labels, gamma=2.0, alpha=0.25, weights=weights, label_smoothing=label_smoothing,
                                  previous_actions=previous_actions, onset_weight=onset_weight)

