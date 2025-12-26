"""Evaluation metrics for behavior cloning."""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List
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


def format_metrics_for_logging(
    metrics: Dict[str, np.ndarray],
    action_names: List[str],
    prefix: str = "",
) -> Dict[str, float]:
    """Format per-action metrics for WandB logging.
    
    Args:
        metrics: Dict with per-action metric arrays
        action_names: List of action names
        prefix: Prefix for metric names (e.g., 'train/', 'val/')
        
    Returns:
        Flattened dict for logging
    """
    log_dict = {}
    
    for metric_name, values in metrics.items():
        if isinstance(values, np.ndarray) and values.ndim == 1:
            # Per-action metrics
            for i, action_name in enumerate(action_names):
                key = f"{prefix}{metric_name}/{action_name}"
                log_dict[key] = float(values[i])
            # Also log mean
            log_dict[f"{prefix}{metric_name}/mean"] = float(np.mean(values))
        else:
            # Scalar metrics
            log_dict[f"{prefix}{metric_name}"] = float(values)
    
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
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Header
    print(f"{'Action':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"{'-'*60}")
    
    # Per-action
    for i, action_name in enumerate(action_names):
        acc = metrics['per_action_accuracy'][i] if 'per_action_accuracy' in metrics else 0
        prec = metrics['per_action_precision'][i] if 'per_action_precision' in metrics else 0
        rec = metrics['per_action_recall'][i] if 'per_action_recall' in metrics else 0
        f1 = metrics['per_action_f1_score'][i] if 'per_action_f1_score' in metrics else 0
        
        print(f"{action_name:<20} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")
    
    # Mean
    print(f"{'-'*60}")
    mean_acc = np.mean(metrics['per_action_accuracy']) if 'per_action_accuracy' in metrics else 0
    mean_prec = np.mean(metrics['per_action_precision']) if 'per_action_precision' in metrics else 0
    mean_rec = np.mean(metrics['per_action_recall']) if 'per_action_recall' in metrics else 0
    mean_f1 = np.mean(metrics['per_action_f1_score']) if 'per_action_f1_score' in metrics else 0
    
    print(f"{'Mean':<20} {mean_acc:>8.4f} {mean_prec:>8.4f} {mean_rec:>8.4f} {mean_f1:>8.4f}")
    print(f"{'='*60}\n")

