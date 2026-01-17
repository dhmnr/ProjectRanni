"""Evaluate model checkpoint with threshold sweep analysis.

Usage:
    python labs/eval_threshold_sweep.py --config configs/temporal_cnn_aux.yaml \
        --checkpoint /home/dm/ProjectRanni/checkpoints/temporal_cnn_aux/best.pkl

Collects all predictions on validation set, then runs threshold sweep.
"""

import os
import sys
from pathlib import Path
import logging
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pickle
from omegaconf import OmegaConf
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from common.dataset import (
    TemporalGameplayDataset,
    create_temporal_data_loader,
    split_episodes,
)
from common.state_preprocessing import create_preprocessor
from common.metrics import compute_per_action_metrics, load_action_names
from models.temporal_cnn.model import create_model as create_temporal_cnn
from labs.threshold_sweep import sweep_thresholds, analyze_prediction_distribution, plot_threshold_sweep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainState:
    """Minimal train state for loading checkpoints."""
    def __init__(self, params, batch_stats, apply_fn):
        self.params = params
        self.batch_stats = batch_stats
        self.apply_fn = apply_fn


def load_checkpoint_params(ckpt_path: Path):
    """Load checkpoint parameters only (no optimizer state needed)."""
    with open(ckpt_path, 'rb') as f:
        ckpt_data = pickle.load(f)

    params = jax.tree.map(jnp.array, ckpt_data['params'])
    batch_stats = jax.tree.map(jnp.array, ckpt_data['batch_stats']) if ckpt_data['batch_stats'] else None

    return params, batch_stats


def create_model_from_config(config: dict, num_actions: int, state_dim: int,
                              hero_vocab: int, npc_vocab: int):
    """Create model from config."""
    model_config = config.get('model', {})
    temporal_config = config.get('temporal', {})
    attention_config = config.get('attention', {})
    preprocess_config = config.get('state_preprocessing', {})
    auxiliary_config = config.get('auxiliary', {})

    num_history_frames = temporal_config.get('num_history_frames', 4)
    num_action_history = temporal_config.get('num_action_history', 4)
    stack_states = temporal_config.get('stack_states', False)
    use_anim_onset_timing = temporal_config.get('use_anim_onset_timing', False)
    anim_onset_encoding_dim = temporal_config.get('anim_onset_encoding_dim', 16)

    # Auxiliary task
    use_aux_npc_anim = auxiliary_config.get('use_npc_anim_prediction', False)
    aux_npc_anim_classes = auxiliary_config.get('npc_anim_classes', npc_vocab)

    model = create_temporal_cnn(
        num_actions=num_actions,
        num_history_frames=num_history_frames,
        num_action_history=num_action_history,
        use_state=config.get('dataset', {}).get('use_state', False),
        stack_states=stack_states,
        num_state_features=state_dim,
        hero_anim_vocab_size=hero_vocab,
        npc_anim_vocab_size=npc_vocab,
        anim_embed_dim=preprocess_config.get('anim_embed_dim', 16),
        frame_mode=model_config.get('frame_mode', 'channel_stack'),
        conv_features=tuple(model_config.get('conv_features', [32, 64, 128, 256])),
        dense_features=tuple(model_config.get('dense_features', [512, 256])),
        state_encoder_features=tuple(model_config.get('state_encoder_features', [64, 64])),
        state_output_features=model_config.get('state_output_features', 64),
        action_history_features=model_config.get('action_history_features', 64),
        dropout_rate=model_config.get('dropout_rate', 0.0),
        use_batch_norm=model_config.get('use_batch_norm', True),
        use_frame_attention=attention_config.get('use_frame_attention', False),
        use_state_attention=attention_config.get('use_state_attention', False),
        use_cross_attention=attention_config.get('use_cross_attention', False),
        attention_num_heads=attention_config.get('num_heads', 4),
        attention_head_dim=attention_config.get('head_dim', 32),
        use_aux_npc_anim=use_aux_npc_anim,
        aux_npc_anim_classes=aux_npc_anim_classes,
        use_anim_onset_timing=use_anim_onset_timing,
        anim_onset_encoding_dim=anim_onset_encoding_dim,
    )

    return model, use_aux_npc_anim


def run_evaluation(config_path: str, checkpoint_path: str, thresholds: list = None):
    """Run evaluation with threshold sweep."""

    # Load config
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    dataset_config = config['dataset']

    # Resolve relative dataset path - relative to project root, not config file
    # The config paths are written relative to where train.py runs (BC_training dir)
    # But dataset is actually at project root level
    dataset_path = Path(dataset_config['path'])
    if not dataset_path.is_absolute():
        # Try relative to BC_training first
        bc_training_dir = Path(__file__).parent.parent
        candidate = (bc_training_dir / dataset_path).resolve()
        if not candidate.exists():
            # Try relative to project root
            project_root = bc_training_dir.parent
            candidate = (project_root / dataset_path).resolve()
        dataset_path = candidate
    dataset_config['path'] = str(dataset_path)

    # Also update config for preprocessor to find anim_mappings
    config['dataset']['path'] = str(dataset_path)

    logger.info(f"Using dataset at: {dataset_path}")

    temporal_config = config.get('temporal', {})

    # Create state preprocessor
    preprocessor = create_preprocessor(config)

    # Get episode split
    import zarr
    zarr_root = zarr.open(dataset_config['path'], mode='r')
    all_episodes = sorted([k for k in zarr_root.keys() if k.startswith('episode_')])
    num_episodes = len(all_episodes)

    train_indices, val_indices = split_episodes(
        num_episodes,
        train_ratio=dataset_config.get('train_ratio', 0.8),
        val_ratio=dataset_config.get('val_ratio', 0.2),
        seed=dataset_config.get('split_seed', 42),
    )

    logger.info(f"Validation episodes: {len(val_indices)}")

    # Create validation dataset (no oversampling for eval)
    num_history_frames = temporal_config.get('num_history_frames', 4)
    num_action_history = temporal_config.get('num_action_history', 4)
    stack_states = temporal_config.get('stack_states', False)
    use_anim_onset_timing = temporal_config.get('use_anim_onset_timing', False)
    anim_onset_encoding_dim = temporal_config.get('anim_onset_encoding_dim', 16)

    val_dataset = TemporalGameplayDataset(
        dataset_path=dataset_config['path'],
        episode_indices=val_indices,
        use_state=dataset_config.get('use_state', True),
        normalize_frames=dataset_config.get('normalize_frames', True),
        validate_episodes=dataset_config.get('validate_episodes', True),
        state_preprocessor=preprocessor,
        num_history_frames=num_history_frames,
        num_action_history=num_action_history,
        frame_skip=temporal_config.get('frame_skip', 1),
        stack_states=stack_states,
        use_anim_onset_timing=use_anim_onset_timing,
        anim_onset_encoding_dim=anim_onset_encoding_dim,
    )

    num_actions = val_dataset.num_actions
    action_names = val_dataset.action_keys

    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Action names: {action_names}")

    # Create model
    state_dim = preprocessor.continuous_dim
    hero_vocab = preprocessor.hero_vocab_size
    npc_vocab = preprocessor.npc_vocab_size

    model, use_aux = create_model_from_config(
        config, num_actions, state_dim, hero_vocab, npc_vocab
    )

    logger.info(f"Model has auxiliary task: {use_aux}")

    # Load checkpoint
    params, batch_stats = load_checkpoint_params(Path(checkpoint_path))
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Create apply function
    def predict_batch(batch):
        variables = {'params': params}
        if batch_stats is not None:
            variables['batch_stats'] = batch_stats

        outputs = model.apply(
            variables,
            batch['frames'],
            batch['action_history'],
            batch['state'],
            batch['hero_anim_idx'],
            batch['npc_anim_idx'],
            batch.get('anim_onset_encoding', None),
            training=False,
        )

        if use_aux:
            action_logits = outputs['action_logits']
        else:
            action_logits = outputs

        return jax.nn.sigmoid(action_logits)

    # JIT compile
    predict_batch_jit = jax.jit(predict_batch)

    # Collect all predictions
    all_predictions = []
    all_targets = []

    batch_size = config.get('training', {}).get('batch_size', 32)

    logger.info("Running evaluation on validation set...")

    for batch in tqdm(create_temporal_data_loader(val_dataset, batch_size, shuffle=False)):
        # Convert to JAX arrays
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        predictions = predict_batch_jit(batch_jax)

        all_predictions.append(np.array(predictions))
        all_targets.append(np.array(batch['actions']))

    # Concatenate all
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    logger.info(f"Collected {len(all_predictions)} predictions")

    # Run threshold sweep for dodge (index 2)
    dodge_idx = action_names.index('dodge_roll/dash') if 'dodge_roll/dash' in action_names else 2

    results = sweep_thresholds(
        all_predictions, all_targets,
        action_names=action_names,
        thresholds=thresholds,
        target_action_idx=dodge_idx,
    )

    # Analyze distribution
    analyze_prediction_distribution(
        all_predictions, all_targets,
        action_idx=dodge_idx,
        action_name=action_names[dodge_idx] if dodge_idx < len(action_names) else f"action_{dodge_idx}",
    )

    # Also do attack if available
    if 'attack' in action_names:
        attack_idx = action_names.index('attack')
        print("\n\n")
        sweep_thresholds(
            all_predictions, all_targets,
            action_names=action_names,
            thresholds=thresholds,
            target_action_idx=attack_idx,
        )
        analyze_prediction_distribution(
            all_predictions, all_targets,
            action_idx=attack_idx,
            action_name='attack',
        )

    # Standard metrics at 0.5 threshold
    print("\n" + "="*60)
    print("STANDARD METRICS (threshold=0.5)")
    print("="*60)
    compute_per_action_metrics(
        all_predictions, all_targets,
        action_names=action_names,
        threshold=0.5,
    )

    # Save predictions for further analysis
    output_dir = Path(checkpoint_path).parent
    np.save(output_dir / 'val_predictions.npy', all_predictions)
    np.save(output_dir / 'val_targets.npy', all_targets)
    logger.info(f"Saved predictions to {output_dir}")

    return all_predictions, all_targets, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model with threshold sweep")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds (default: 0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9)")

    args = parser.parse_args()

    thresholds = None
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(',')]

    run_evaluation(args.config, args.checkpoint, thresholds)
