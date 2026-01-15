"""Standalone script to verify saved samples against the model.

Usage:
    uv run python BC_eval/verify_samples.py BC_eval/logs/samples/<episode_dir>
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import jax.numpy as jnp


def load_and_preprocess_frame(frame_path: str, target_shape=(3, 144, 256)):
    """Load and preprocess frame exactly as the agent does.

    Args:
        frame_path: Path to raw PNG frame
        target_shape: (C, H, W) expected by model

    Returns:
        Preprocessed frame ready for model [1, C, H, W]
    """
    # Load frame (BGR from cv2)
    frame_bgr = cv2.imread(frame_path)
    if frame_bgr is None:
        raise ValueError(f"Failed to load frame: {frame_path}")

    # Convert to RGB (as eldengym provides)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    print(f"  Raw frame shape: {frame_rgb.shape} dtype: {frame_rgb.dtype}")

    # Resize to target (same as agent)
    target_h, target_w = target_shape[1], target_shape[2]
    frame_resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    print(f"  Resized shape: {frame_resized.shape}")

    # Normalize to [0, 1]
    frame_norm = frame_resized.astype(np.float32) / 255.0
    print(f"  Normalized range: [{frame_norm.min():.3f}, {frame_norm.max():.3f}]")

    # Transpose HWC -> CHW
    frame_chw = np.transpose(frame_norm, (2, 0, 1))
    print(f"  Transposed shape: {frame_chw.shape}")

    # Add batch dimension
    frame_batch = frame_chw[np.newaxis, ...]
    print(f"  Batch shape: {frame_batch.shape}")

    return jnp.array(frame_batch)


def run_verification(samples_dir: str, checkpoint_path: str = None, config_path: str = None):
    """Verify samples against model.

    Args:
        samples_dir: Directory containing saved samples
        checkpoint_path: Path to model checkpoint (default: pure_cnn best)
        config_path: Path to training config (default: pure_cnn config)
    """
    samples_dir = Path(samples_dir)

    # Default paths
    if checkpoint_path is None:
        checkpoint_path = "./BC_training/checkpoints/pure_cnn/best.pkl"
    if config_path is None:
        config_path = "./BC_training/configs/pure_cnn.yaml"

    print("=" * 60)
    print("Sample Verification Script")
    print("=" * 60)
    print(f"Samples dir: {samples_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print()

    # Load model
    print("Loading model...")
    from BC_eval.model_loader import load_model
    model = load_model(checkpoint_path, config_path)
    print(f"  Model: {model.model_name}")
    print(f"  Num actions: {model.num_actions}")
    print()

    # Action names
    action_names = ['move_left', 'lock_on', 'dodge', 'move_right', 'attack', 'move_back', 'move_forward']

    # Find all samples
    sample_files = sorted(samples_dir.glob("*_raw.png"))
    print(f"Found {len(sample_files)} samples")
    print()

    for frame_path in sample_files:
        step_id = frame_path.stem.replace("_raw", "")
        meta_path = frame_path.parent / f"{step_id}_meta.json"

        print("-" * 60)
        print(f"Sample: {step_id}")

        # Load metadata
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  Original probs: {meta.get('probs', [])}")
            print(f"  Original actions: {meta.get('active_actions', [])}")
        else:
            meta = {}
            print("  No metadata found")

        print()
        print("  Preprocessing:")

        # Load and preprocess
        try:
            frame_batch = load_and_preprocess_frame(str(frame_path))
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Run inference
        print()
        print("  Running inference...")
        probs = model.get_action_probs(frame_batch)
        probs = np.array(probs[0])  # Remove batch dim

        print(f"  Output probs: {probs}")
        print()
        print("  Per-action breakdown:")
        for i, (name, p) in enumerate(zip(action_names, probs)):
            marker = ">>>" if p > 0.5 else "   "
            orig_p = meta.get('probs', [0]*7)[i] if i < len(meta.get('probs', [])) else 0
            diff = p - orig_p
            print(f"    {marker} {name:15s}: {p:.4f} (orig: {orig_p:.4f}, diff: {diff:+.4f})")

        print()

    print("=" * 60)
    print("Verification complete")


def compare_with_training_sample():
    """Compare preprocessing with a sample from training data."""
    import zarr

    print("=" * 60)
    print("Comparing with training data sample")
    print("=" * 60)

    # Load a training sample
    z = zarr.open('./dataset/margit_100_256x144.zarr', mode='r')

    # Get first frame from first episode
    ep = z['episode_0']
    train_frame = np.array(ep['frames'][0])  # [C, H, W] already preprocessed
    train_action = np.array(ep['actions'][0])

    print(f"Training frame shape: {train_frame.shape}")
    print(f"Training frame dtype: {train_frame.dtype}")
    print(f"Training frame range: [{train_frame.min()}, {train_frame.max()}]")
    print(f"Training action: {train_action}")
    print()

    # Load model
    from BC_eval.model_loader import load_model
    model = load_model(
        "./BC_training/checkpoints/pure_cnn/best.pkl",
        "./BC_training/configs/pure_cnn.yaml"
    )

    # Run inference on training frame (already preprocessed!)
    # Just need to normalize and add batch dim
    if train_frame.max() > 1:
        train_frame_norm = train_frame.astype(np.float32) / 255.0
    else:
        train_frame_norm = train_frame.astype(np.float32)

    train_batch = jnp.array(train_frame_norm[np.newaxis, ...])
    print(f"Input batch shape: {train_batch.shape}")
    print(f"Input range: [{float(train_batch.min()):.3f}, {float(train_batch.max()):.3f}]")

    probs = model.get_action_probs(train_batch)
    probs = np.array(probs[0])

    action_names = ['move_left', 'lock_on', 'dodge', 'move_right', 'attack', 'move_back', 'move_forward']
    print()
    print("Model output on TRAINING frame:")
    for i, (name, p) in enumerate(zip(action_names, probs)):
        gt = train_action[i]
        marker = ">>>" if p > 0.5 else "   "
        gt_marker = "[GT=1]" if gt > 0.5 else ""
        print(f"  {marker} {name:15s}: {p:.4f} {gt_marker}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python BC_eval/verify_samples.py <samples_dir>")
        print("       uv run python BC_eval/verify_samples.py --training")
        sys.exit(1)

    if sys.argv[1] == "--training":
        compare_with_training_sample()
    else:
        run_verification(sys.argv[1])
