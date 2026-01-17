"""Threshold sweep for evaluating precision/recall tradeoff.

Usage:
    python labs/threshold_sweep.py --config configs/temporal_cnn.yaml --checkpoint path/to/checkpoint.pkl

Or run after training by importing and calling sweep_thresholds() with predictions.
"""

import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


def sweep_thresholds(predictions: np.ndarray, targets: np.ndarray, action_names: list = None,
                     thresholds: list = None, target_action_idx: int = 2):
    """Sweep thresholds and compute precision/recall/F1 for a specific action.

    Args:
        predictions: [N, num_actions] float probabilities
        targets: [N, num_actions] binary targets
        action_names: List of action names
        thresholds: List of thresholds to try (default: 0.3 to 0.9)
        target_action_idx: Index of action to analyze (default: 2 = dodge)

    Returns:
        Dict with results per threshold
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

    action_name = action_names[target_action_idx] if action_names else f"action_{target_action_idx}"

    # Get predictions and targets for this action
    action_preds = predictions[:, target_action_idx]
    action_targets = targets[:, target_action_idx]

    print(f"\n{'='*60}")
    print(f"THRESHOLD SWEEP FOR: {action_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(action_targets)}")
    print(f"Positive samples: {action_targets.sum():.0f} ({100*action_targets.mean():.2f}%)")
    print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Pred Pos':>10}")
    print("-" * 60)

    results = []
    for thresh in thresholds:
        binary_preds = (action_preds >= thresh).astype(float)

        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            action_targets, binary_preds, average='binary', zero_division=0
        )

        pred_positives = binary_preds.sum()

        print(f"{thresh:>10.2f} | {precision:>10.4f} | {recall:>10.4f} | {f1:>10.4f} | {pred_positives:>10.0f}")

        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pred_positives': pred_positives,
        })

    # Find best F1
    best_result = max(results, key=lambda x: x['f1'])
    print(f"\nBest F1: {best_result['f1']:.4f} at threshold {best_result['threshold']}")

    # Find best for high precision (>0.5)
    high_prec_results = [r for r in results if r['precision'] >= 0.5]
    if high_prec_results:
        best_high_prec = max(high_prec_results, key=lambda x: x['f1'])
        print(f"Best F1 with precision >= 0.5: {best_high_prec['f1']:.4f} at threshold {best_high_prec['threshold']}")

    return results


def plot_threshold_sweep(results: list, action_name: str = "dodge"):
    """Plot precision/recall/F1 vs threshold."""
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'g-o', label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, 'r-o', label='F1', linewidth=2)

    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Precision/Recall/F1 vs Threshold for {action_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.25, 0.95)
    plt.ylim(0, 1)

    # Add vertical line at best F1
    best_idx = np.argmax(f1s)
    plt.axvline(x=thresholds[best_idx], color='red', linestyle='--', alpha=0.5,
                label=f'Best F1 @ {thresholds[best_idx]}')

    plt.tight_layout()
    plt.show()

    return plt.gcf()


def analyze_prediction_distribution(predictions: np.ndarray, targets: np.ndarray,
                                    action_idx: int = 2, action_name: str = "dodge"):
    """Analyze the distribution of prediction probabilities for positives vs negatives."""
    action_preds = predictions[:, action_idx]
    action_targets = targets[:, action_idx]

    pos_preds = action_preds[action_targets == 1]
    neg_preds = action_preds[action_targets == 0]

    print(f"\n{'='*60}")
    print(f"PREDICTION DISTRIBUTION FOR: {action_name}")
    print(f"{'='*60}")
    print(f"\nPositive samples (actual {action_name}):")
    print(f"  Mean pred: {pos_preds.mean():.4f}")
    print(f"  Median pred: {np.median(pos_preds):.4f}")
    print(f"  Std pred: {pos_preds.std():.4f}")
    print(f"  Min/Max: {pos_preds.min():.4f} / {pos_preds.max():.4f}")

    print(f"\nNegative samples (not {action_name}):")
    print(f"  Mean pred: {neg_preds.mean():.4f}")
    print(f"  Median pred: {np.median(neg_preds):.4f}")
    print(f"  Std pred: {neg_preds.std():.4f}")
    print(f"  Min/Max: {neg_preds.min():.4f} / {neg_preds.max():.4f}")

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(neg_preds, bins=50, alpha=0.7, label=f'Not {action_name}', color='blue')
    axes[0].hist(pos_preds, bins=50, alpha=0.7, label=f'Actual {action_name}', color='red')
    axes[0].set_xlabel('Prediction Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Prediction Distribution for {action_name}')
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    # Zoomed in on higher probabilities
    axes[1].hist(neg_preds[neg_preds > 0.3], bins=30, alpha=0.7, label=f'Not {action_name}', color='blue')
    axes[1].hist(pos_preds[pos_preds > 0.3], bins=30, alpha=0.7, label=f'Actual {action_name}', color='red')
    axes[1].set_xlabel('Prediction Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Prediction Distribution (zoomed: p > 0.3)')
    axes[1].legend()
    axes[1].set_xlim(0.3, 1)

    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold sweep for action prediction")
    parser.add_argument("--predictions", type=str, help="Path to saved predictions .npy file")
    parser.add_argument("--targets", type=str, help="Path to saved targets .npy file")
    parser.add_argument("--action_idx", type=int, default=2, help="Action index to analyze (default: 2 = dodge)")

    args = parser.parse_args()

    if args.predictions and args.targets:
        predictions = np.load(args.predictions)
        targets = np.load(args.targets)

        results = sweep_thresholds(predictions, targets, target_action_idx=args.action_idx)
        analyze_prediction_distribution(predictions, targets, action_idx=args.action_idx)
        plot_threshold_sweep(results)
    else:
        print("Usage: python threshold_sweep.py --predictions preds.npy --targets targets.npy")
        print("\nOr import and use in a notebook:")
        print("  from labs.threshold_sweep import sweep_thresholds, analyze_prediction_distribution")
        print("  results = sweep_thresholds(predictions, targets, action_names=['w', 'a', 'dodge', ...])")
