"""Evaluation metrics for behavior cloning."""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """Compute overall accuracy for multi-label classification.
    
    Args:
        predictions: Predicted probabilities [N, num_actions]
        targets: Ground truth binary labels [N, num_actions]
        threshold: Threshold for binary classification
        
    Returns:
        Accuracy (exact match ratio)
    """
    pred_binary = (predictions > threshold).astype(np.float32)
    # Exact match: all actions must match
    exact_match = np.all(pred_binary == targets, axis=1)
    return np.mean(exact_match)


def compute_per_action_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Compute per-action metrics (accuracy, precision, recall, F1).
    
    Vectorized implementation - no Python loops.
    
    Args:
        predictions: Predicted probabilities [N, num_actions]
        targets: Ground truth binary labels [N, num_actions]
        threshold: Threshold for binary classification
        
    Returns:
        Dict with per-action metrics
    """
    pred_binary = (predictions > threshold).astype(np.float32)
    targets = targets.astype(np.float32)
    
    # Vectorized accuracy: mean of correct predictions per action
    accuracy = np.mean(pred_binary == targets, axis=0)
    
    # Vectorized TP, FP, FN computation
    tp = np.sum((pred_binary == 1) & (targets == 1), axis=0)
    fp = np.sum((pred_binary == 1) & (targets == 0), axis=0)
    fn = np.sum((pred_binary == 0) & (targets == 1), axis=0)
    
    # Precision, Recall, F1 - all vectorized
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'per_action_accuracy': accuracy,
        'per_action_precision': precision,
        'per_action_recall': recall,
        'per_action_f1_score': f1_score,
    }


def compute_action_distribution_distance(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute distance between predicted and target action distributions.
    
    Args:
        predictions: Predicted probabilities [N, num_actions]
        targets: Ground truth binary labels [N, num_actions]
        threshold: Threshold for binary classification
        
    Returns:
        Dict with distribution metrics
    """
    pred_binary = (predictions > threshold).astype(np.float32)
    
    # Action frequencies
    pred_freq = np.mean(pred_binary, axis=0)
    target_freq = np.mean(targets, axis=0)
    
    # L1 distance
    l1_distance = np.sum(np.abs(pred_freq - target_freq))
    
    # L2 distance
    l2_distance = np.sqrt(np.sum((pred_freq - target_freq) ** 2))
    
    # KL divergence (add small epsilon for numerical stability)
    eps = 1e-8
    kl_div = np.sum(target_freq * np.log((target_freq + eps) / (pred_freq + eps)))
    
    return {
        'l1_distance': float(l1_distance),
        'l2_distance': float(l2_distance),
        'kl_divergence': float(kl_div),
        'pred_freq': pred_freq,
        'target_freq': target_freq,
    }


def compute_onset_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    previous_actions: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute metrics on action ONSET frames (where action changes from previous).
    
    This is the key metric to detect if model is just copying previous actions.
    A model that copies will have low onset accuracy.
    
    Args:
        predictions: Predicted probabilities [N, num_actions]
        targets: Ground truth binary labels [N, num_actions] (current frame)
        previous_actions: Actions from previous frame [N, num_actions]
        threshold: Threshold for binary classification
        
    Returns:
        Dict with onset-specific metrics
    """
    pred_binary = (predictions > threshold).astype(np.float32)
    targets = targets.astype(np.float32)
    previous_actions = previous_actions.astype(np.float32)
    
    # Find frames where ANY action changed from previous
    action_changed = ~np.all(targets == previous_actions, axis=1)  # [N]
    num_onset_frames = np.sum(action_changed)
    
    if num_onset_frames == 0:
        return {
            'onset_accuracy': 0.0,
            'onset_count': 0,
            'onset_ratio': 0.0,
            'per_action_onset_accuracy': np.zeros(targets.shape[1]),
            'per_action_onset_recall': np.zeros(targets.shape[1]),
        }
    
    # Filter to onset frames only
    onset_preds = pred_binary[action_changed]
    onset_targets = targets[action_changed]
    onset_previous = previous_actions[action_changed]
    
    # Exact match accuracy on onset frames
    exact_match = np.all(onset_preds == onset_targets, axis=1)
    onset_accuracy = np.mean(exact_match)
    
    # Per-action onset metrics
    num_actions = targets.shape[1]
    per_action_onset_acc = np.zeros(num_actions)
    per_action_onset_recall = np.zeros(num_actions)
    
    for i in range(num_actions):
        # Frames where THIS action changed
        action_i_changed = onset_targets[:, i] != onset_previous[:, i]
        if np.sum(action_i_changed) > 0:
            # Accuracy: when action changed, did we predict correctly?
            correct = onset_preds[action_i_changed, i] == onset_targets[action_i_changed, i]
            per_action_onset_acc[i] = np.mean(correct)
            
            # Recall for action START: when action turned ON, did we predict ON?
            action_started = (onset_previous[:, i] == 0) & (onset_targets[:, i] == 1)
            if np.sum(action_started) > 0:
                recalled = onset_preds[action_started, i] == 1
                per_action_onset_recall[i] = np.mean(recalled)
    
    return {
        'onset_accuracy': float(onset_accuracy),
        'onset_count': int(num_onset_frames),
        'onset_ratio': float(num_onset_frames / len(targets)),
        'per_action_onset_accuracy': per_action_onset_acc,
        'per_action_onset_recall': per_action_onset_recall,  # Key metric!
    }


def compute_buffered_onset_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    previous_actions: np.ndarray,
    previous_predictions: np.ndarray,
    threshold: float = 0.5,
    buffer_frames: int = 5,
) -> Dict[str, float]:
    """Compute onset metrics with temporal buffer (±k frames tolerance).
    
    Instead of requiring exact frame matching, allows predictions within
    a buffer window to count as correct. More realistic for gameplay.
    
    Args:
        predictions: Predicted probabilities [N, num_actions]
        targets: Ground truth binary labels [N, num_actions]
        previous_actions: Target actions from previous frame [N, num_actions]
        previous_predictions: Predictions from previous frame [N, num_actions]
        threshold: Threshold for binary classification
        buffer_frames: Tolerance window (±k frames)
        
    Returns:
        Dict with buffered onset metrics
    """
    pred_binary = (predictions > threshold).astype(np.float32)
    prev_pred_binary = (previous_predictions > threshold).astype(np.float32)
    targets = targets.astype(np.float32)
    previous_actions = previous_actions.astype(np.float32)
    
    num_samples, num_actions = targets.shape
    
    # Find TRUE onset frames (where target action changed)
    true_onset_mask = ~np.all(targets == previous_actions, axis=1)  # [N]
    true_onset_indices = np.where(true_onset_mask)[0]
    
    # Find PREDICTED onset frames (where prediction changed)
    pred_onset_mask = ~np.all(pred_binary == prev_pred_binary, axis=1)  # [N]
    pred_onset_indices = np.where(pred_onset_mask)[0]
    
    num_true_onsets = len(true_onset_indices)
    num_pred_onsets = len(pred_onset_indices)
    
    if num_true_onsets == 0:
        return {
            'buffered_onset_recall': 0.0,
            'buffered_onset_precision': 0.0,
            'buffered_onset_f1': 0.0,
            'num_true_onsets': 0,
            'num_pred_onsets': num_pred_onsets,
            'buffer_frames': buffer_frames,
            'per_action_buffered_recall': np.zeros(num_actions),
            'per_action_buffered_precision': np.zeros(num_actions),
        }
    
    # Buffered Recall: For each TRUE onset, is there a PRED onset within ±k frames?
    matched_true_onsets = 0
    for true_idx in true_onset_indices:
        # Check if any prediction onset is within buffer
        if len(pred_onset_indices) > 0:
            distances = np.abs(pred_onset_indices - true_idx)
            if np.min(distances) <= buffer_frames:
                matched_true_onsets += 1
    
    buffered_recall = matched_true_onsets / num_true_onsets
    
    # Buffered Precision: For each PRED onset, is there a TRUE onset within ±k frames?
    if num_pred_onsets == 0:
        buffered_precision = 0.0
    else:
        matched_pred_onsets = 0
        for pred_idx in pred_onset_indices:
            distances = np.abs(true_onset_indices - pred_idx)
            if np.min(distances) <= buffer_frames:
                matched_pred_onsets += 1
        buffered_precision = matched_pred_onsets / num_pred_onsets
    
    # F1 score
    if buffered_precision + buffered_recall > 0:
        buffered_f1 = 2 * buffered_precision * buffered_recall / (buffered_precision + buffered_recall)
    else:
        buffered_f1 = 0.0
    
    # Per-action buffered metrics
    per_action_buffered_recall = np.zeros(num_actions)
    per_action_buffered_precision = np.zeros(num_actions)
    
    for action_idx in range(num_actions):
        # True onsets for this action
        action_true_onset = (targets[:, action_idx] != previous_actions[:, action_idx])
        action_true_indices = np.where(action_true_onset)[0]
        
        # Predicted onsets for this action
        action_pred_onset = (pred_binary[:, action_idx] != prev_pred_binary[:, action_idx])
        action_pred_indices = np.where(action_pred_onset)[0]
        
        # Recall for this action
        if len(action_true_indices) > 0:
            matched = 0
            for true_idx in action_true_indices:
                if len(action_pred_indices) > 0:
                    if np.min(np.abs(action_pred_indices - true_idx)) <= buffer_frames:
                        matched += 1
            per_action_buffered_recall[action_idx] = matched / len(action_true_indices)
        
        # Precision for this action
        if len(action_pred_indices) > 0:
            matched = 0
            for pred_idx in action_pred_indices:
                if len(action_true_indices) > 0:
                    if np.min(np.abs(action_true_indices - pred_idx)) <= buffer_frames:
                        matched += 1
            per_action_buffered_precision[action_idx] = matched / len(action_pred_indices)
    
    return {
        'buffered_onset_recall': float(buffered_recall),
        'buffered_onset_precision': float(buffered_precision),
        'buffered_onset_f1': float(buffered_f1),
        'num_true_onsets': int(num_true_onsets),
        'num_pred_onsets': int(num_pred_onsets),
        'buffer_frames': int(buffer_frames),
        'per_action_buffered_recall': per_action_buffered_recall,
        'per_action_buffered_precision': per_action_buffered_precision,
    }


def compute_aggregate_metrics(per_action_metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute aggregate (mean) metrics from per-action metrics.
    
    Args:
        per_action_metrics: Dict with per-action metric arrays
        
    Returns:
        Dict with aggregated scalar metrics
    """
    agg = {}
    for key, values in per_action_metrics.items():
        if isinstance(values, np.ndarray) and values.ndim == 1:
            agg[f'{key}_mean'] = float(np.mean(values))
            agg[f'{key}_std'] = float(np.std(values))
            agg[f'{key}_min'] = float(np.min(values))
            agg[f'{key}_max'] = float(np.max(values))
    return agg


def format_metrics_for_logging(
    metrics: Dict[str, np.ndarray],
    action_names: List[str],
    prefix: str = "",
) -> Dict[str, float]:
    """Format metrics for WandB logging with proper grouping.
    
    Groups:
        - {prefix}aggregate/  - Mean, std, min, max of per-action metrics
        - {prefix}per_action/{action_name}/  - Per-action breakdown
        - {prefix}onset/  - Onset-specific metrics (if present)
        - {prefix}distribution/  - Distribution distance metrics
    
    Args:
        metrics: Dict with per-action metric arrays
        action_names: List of action names
        prefix: Prefix for metric names (e.g., 'val/')
        
    Returns:
        Flattened dict for logging
    """
    log_dict = {}
    
    # Separate metrics by type
    per_action_metrics = {}
    onset_metrics = {}
    distribution_metrics = {}
    scalar_metrics = {}
    
    for metric_name, values in metrics.items():
        if 'buffered' in metric_name or 'onset' in metric_name:
            onset_metrics[metric_name] = values
        elif metric_name in ['l1_distance', 'l2_distance', 'kl_divergence', 'pred_freq', 'target_freq']:
            distribution_metrics[metric_name] = values
        elif isinstance(values, np.ndarray) and values.ndim == 1 and len(values) == len(action_names):
            per_action_metrics[metric_name] = values
        elif np.isscalar(values) or (isinstance(values, np.ndarray) and values.ndim == 0):
            scalar_metrics[metric_name] = values
    
    # Log scalar metrics directly
    for metric_name, value in scalar_metrics.items():
        log_dict[f"{prefix}{metric_name}"] = float(value)
    
    # Log per-action metrics with grouping
    for metric_name, values in per_action_metrics.items():
        # Clean up metric name (remove 'per_action_' prefix for cleaner logging)
        clean_name = metric_name.replace('per_action_', '')
        
        # Aggregate stats
        log_dict[f"{prefix}aggregate/{clean_name}_mean"] = float(np.mean(values))
        log_dict[f"{prefix}aggregate/{clean_name}_std"] = float(np.std(values))
        
        # Per-action breakdown
        for i, action_name in enumerate(action_names):
            log_dict[f"{prefix}per_action/{action_name}/{clean_name}"] = float(values[i])
    
    # Log onset metrics
    for metric_name, values in onset_metrics.items():
        if isinstance(values, np.ndarray) and values.ndim == 1:
            # Per-action onset metrics
            clean_name = metric_name.replace('per_action_', '')
            log_dict[f"{prefix}onset/{clean_name}_mean"] = float(np.mean(values))
            for i, action_name in enumerate(action_names):
                log_dict[f"{prefix}onset/{action_name}/{clean_name}"] = float(values[i])
        else:
            # Scalar onset metrics
            log_dict[f"{prefix}onset/{metric_name}"] = float(values) if not isinstance(values, (int, float)) else values
    
    # Log distribution metrics
    for metric_name, values in distribution_metrics.items():
        if isinstance(values, np.ndarray):
            for i, action_name in enumerate(action_names):
                log_dict[f"{prefix}distribution/{action_name}/{metric_name}"] = float(values[i])
        else:
            log_dict[f"{prefix}distribution/{metric_name}"] = float(values)
    
    return log_dict


def print_metrics_summary(
    metrics: Dict[str, np.ndarray],
    action_names: List[str],
    title: str = "Metrics",
):
    """Pretty print metrics summary.
    
    Args:
        metrics: Dict with per-action metric arrays
        action_names: List of action names
        title: Title for the summary
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    # Header - standard metrics
    print(f"{'Action':<15} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'OnsetRec':>10}")
    print(f"{'-'*70}")
    
    # Per-action
    for i, action_name in enumerate(action_names):
        acc = metrics['per_action_accuracy'][i] if 'per_action_accuracy' in metrics else 0
        prec = metrics['per_action_precision'][i] if 'per_action_precision' in metrics else 0
        rec = metrics['per_action_recall'][i] if 'per_action_recall' in metrics else 0
        f1 = metrics['per_action_f1_score'][i] if 'per_action_f1_score' in metrics else 0
        onset_rec = metrics['per_action_onset_recall'][i] if 'per_action_onset_recall' in metrics else 0
        
        print(f"{action_name:<15} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {onset_rec:>10.4f}")
    
    # Mean
    print(f"{'-'*70}")
    mean_acc = np.mean(metrics['per_action_accuracy']) if 'per_action_accuracy' in metrics else 0
    mean_prec = np.mean(metrics['per_action_precision']) if 'per_action_precision' in metrics else 0
    mean_rec = np.mean(metrics['per_action_recall']) if 'per_action_recall' in metrics else 0
    mean_f1 = np.mean(metrics['per_action_f1_score']) if 'per_action_f1_score' in metrics else 0
    mean_onset_rec = np.mean(metrics['per_action_onset_recall']) if 'per_action_onset_recall' in metrics else 0
    
    print(f"{'Mean':<15} {mean_acc:>8.4f} {mean_prec:>8.4f} {mean_rec:>8.4f} {mean_f1:>8.4f} {mean_onset_rec:>10.4f}")
    
    # Onset summary if available
    if 'onset_accuracy' in metrics:
        print(f"{'-'*70}")
        print(f"Onset Frames: {metrics.get('onset_count', 0):,} ({metrics.get('onset_ratio', 0)*100:.1f}% of data)")
        print(f"Onset Exact Match Accuracy: {metrics.get('onset_accuracy', 0):.4f}")
    
    # Buffered onset summary if available
    if 'buffered_onset_recall' in metrics:
        buffer_k = metrics.get('buffer_frames', 5)
        print(f"{'-'*70}")
        print(f"Buffered Onset Metrics (±{buffer_k} frames tolerance):")
        print(f"  Recall@{buffer_k}:    {metrics.get('buffered_onset_recall', 0):.4f}  (caught {metrics.get('buffered_onset_recall', 0)*100:.1f}% of expert onsets)")
        print(f"  Precision@{buffer_k}: {metrics.get('buffered_onset_precision', 0):.4f}  ({metrics.get('buffered_onset_precision', 0)*100:.1f}% of predictions were real)")
        print(f"  F1@{buffer_k}:        {metrics.get('buffered_onset_f1', 0):.4f}")
        print(f"  True onsets: {metrics.get('num_true_onsets', 0):,}, Pred onsets: {metrics.get('num_pred_onsets', 0):,}")
        
        # Per-action buffered recall if available
        if 'per_action_buffered_recall' in metrics:
            print(f"\n  Per-Action Buffered Recall@{buffer_k}:")
            for i, action_name in enumerate(action_names):
                recall = metrics['per_action_buffered_recall'][i]
                prec = metrics['per_action_buffered_precision'][i] if 'per_action_buffered_precision' in metrics else 0
                print(f"    {action_name:<15} Recall: {recall:.4f}  Precision: {prec:.4f}")
    
    print(f"{'='*70}\n")

