"""Model loader for BC evaluation.

Loads trained model checkpoints and reconstructs them for inference.
"""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

# Add BC_training to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / "BC_training"))

from BC_training.common.state_preprocessing import StatePreprocessor, create_preprocessor

logger = logging.getLogger(__name__)


class LoadedModel:
    """Container for a loaded BC model with its params and config."""

    def __init__(
        self,
        model: Any,
        params: Dict,
        batch_stats: Optional[Dict],
        config: Dict,
        model_name: str,
        state_preprocessor: Optional[StatePreprocessor] = None,
    ):
        self.model = model
        self.params = params
        self.batch_stats = batch_stats
        self.config = config
        self.model_name = model_name
        self.state_preprocessor = state_preprocessor

        # Derived properties
        self.use_state = config.get("dataset", {}).get("use_state", False)
        self.is_temporal = model_name in ["temporal_cnn", "gru", "causal_transformer"]
        self.num_actions = self._get_num_actions()

        # Temporal config
        if self.is_temporal:
            temporal_config = config.get("temporal", {})
            self.num_history_frames = temporal_config.get("num_history_frames", 4)
            self.num_action_history = temporal_config.get("num_action_history", 4)
            self.frame_skip = temporal_config.get("frame_skip", 1)
        else:
            self.num_history_frames = 0
            self.num_action_history = 0
            self.frame_skip = 1

    def _get_num_actions(self) -> int:
        """Extract num_actions from model config."""
        # Most configs store this in the model params directly or we infer from output layer
        if "num_actions" in self.config.get("model", {}):
            return self.config["model"]["num_actions"]
        # Default for Elden Ring dataset (13 semantic actions from keybinds_v2.json)
        return 13

    def __call__(
        self,
        frames: jnp.ndarray,
        action_history: Optional[jnp.ndarray] = None,
        state: Optional[jnp.ndarray] = None,
        hero_anim_idx: Optional[jnp.ndarray] = None,
        npc_anim_idx: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Run inference on the model.

        Args:
            frames: Input frames [B, C, H, W] or [B, T, C, H, W] for temporal
            action_history: Past actions [B, K, num_actions] for temporal models
            state: Continuous state [B, S] for hybrid/temporal models
            hero_anim_idx: Hero animation index [B] for hybrid models
            npc_anim_idx: NPC animation index [B] for hybrid models

        Returns:
            Action logits [B, num_actions]
        """
        variables = {"params": self.params}
        if self.batch_stats is not None:
            variables["batch_stats"] = self.batch_stats

        if self.is_temporal:
            # Temporal models
            if self.use_state:
                logits = self.model.apply(
                    variables,
                    frames,
                    action_history,
                    state,
                    hero_anim_idx,
                    npc_anim_idx,
                    training=False,
                )
            else:
                logits = self.model.apply(
                    variables, frames, action_history, training=False
                )
        elif self.model_name == "hybrid_state":
            # Hybrid state model
            logits = self.model.apply(
                variables,
                frames,
                state,
                hero_anim_idx,
                npc_anim_idx,
                training=False,
            )
        else:
            # Pure vision model
            logits = self.model.apply(variables, frames, training=False)

        return logits

    def get_action_probs(self, *args, **kwargs) -> jnp.ndarray:
        """Get action probabilities (sigmoid of logits)."""
        logits = self(*args, **kwargs)
        return jax.nn.sigmoid(logits)


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict, int]:
    """Load checkpoint pickle file.

    Args:
        checkpoint_path: Path to .pkl checkpoint

    Returns:
        (checkpoint_data, step)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        ckpt_data = pickle.load(f)

    # Convert numpy arrays to JAX arrays
    params = jax.tree.map(jnp.array, ckpt_data["params"])
    batch_stats = (
        jax.tree.map(jnp.array, ckpt_data["batch_stats"])
        if ckpt_data.get("batch_stats")
        else None
    )
    step = ckpt_data.get("step", 0)

    return {"params": params, "batch_stats": batch_stats}, step


def load_model(
    checkpoint_path: str,
    training_config_path: str,
    num_actions: int = 13,  # 13 semantic actions from keybinds_v2.json
    anim_mappings_path: Optional[str] = None,
) -> LoadedModel:
    """Load a trained BC model for inference.

    Args:
        checkpoint_path: Path to checkpoint .pkl file
        training_config_path: Path to training config .yaml file
        num_actions: Number of actions (override if not in config)
        anim_mappings_path: Optional path to animation ID mappings JSON
            (overrides auto-detection from dataset path)

    Returns:
        LoadedModel instance ready for inference
    """
    # Load training config
    config = OmegaConf.load(training_config_path)
    config = OmegaConf.to_container(config, resolve=True)

    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")

    # Create state preprocessor if needed
    state_preprocessor = None
    if config.get("dataset", {}).get("use_state", False):
        # Override anim_mappings_path if provided
        if anim_mappings_path:
            config.setdefault("state_preprocessing", {})["anim_mappings_path"] = anim_mappings_path
            logger.info(f"Using animation mappings from: {anim_mappings_path}")
        state_preprocessor = create_preprocessor(config)

    # Load checkpoint
    ckpt_data, step = load_checkpoint(checkpoint_path)
    params = ckpt_data["params"]
    batch_stats = ckpt_data["batch_stats"]
    logger.info(f"Loaded checkpoint at step {step}")

    # Create model based on type
    if model_name == "pure_cnn":
        from BC_training.models.pure_cnn import create_model

        model = create_model(
            num_actions=num_actions,
            conv_features=tuple(config["model"]["conv_features"]),
            dense_features=tuple(config["model"]["dense_features"]),
            dropout_rate=config["model"].get("dropout_rate", 0.0),
            use_batch_norm=config["model"].get("use_batch_norm", True),
        )

    elif model_name == "hybrid_state":
        from BC_training.models.hybrid_state import create_model

        num_state_features = state_preprocessor.continuous_dim
        anim_embed_dim = config.get("state_preprocessing", {}).get("anim_embed_dim", 16)

        model = create_model(
            num_actions=num_actions,
            num_state_features=num_state_features,
            hero_anim_vocab_size=state_preprocessor.hero_vocab_size,
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size,
            anim_embed_dim=anim_embed_dim,
            conv_features=tuple(config["model"]["conv_features"]),
            dense_features=tuple(config["model"]["dense_features"]),
            state_encoder_features=tuple(config["model"]["state_encoder_features"]),
            state_output_features=config["model"]["state_output_features"],
            dropout_rate=config["model"].get("dropout_rate", 0.0),
            use_batch_norm=config["model"].get("use_batch_norm", True),
        )

    elif model_name == "temporal_cnn":
        from BC_training.models.temporal_cnn import create_model

        temporal_config = config.get("temporal", {})
        use_state = config.get("dataset", {}).get("use_state", False)
        num_state_features = state_preprocessor.continuous_dim if use_state else 10
        anim_embed_dim = config.get("state_preprocessing", {}).get("anim_embed_dim", 16)

        model = create_model(
            num_actions=num_actions,
            num_history_frames=temporal_config.get("num_history_frames", 4),
            num_action_history=temporal_config.get("num_action_history", 4),
            use_state=use_state,
            num_state_features=num_state_features,
            hero_anim_vocab_size=(
                state_preprocessor.hero_vocab_size if use_state else 67
            ),
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size if use_state else 54,
            anim_embed_dim=anim_embed_dim,
            frame_mode=config["model"].get("frame_mode", "channel_stack"),
            conv_features=tuple(config["model"]["conv_features"]),
            dense_features=tuple(config["model"]["dense_features"]),
            state_encoder_features=tuple(
                config["model"].get("state_encoder_features", [64, 64])
            ),
            state_output_features=config["model"].get("state_output_features", 64),
            action_history_features=config["model"].get("action_history_features", 64),
            dropout_rate=config["model"].get("dropout_rate", 0.0),
            use_batch_norm=config["model"].get("use_batch_norm", True),
        )

    elif model_name == "gru":
        from BC_training.models.gru import create_model

        temporal_config = config.get("temporal", {})
        use_state = config.get("dataset", {}).get("use_state", False)
        num_state_features = state_preprocessor.continuous_dim if use_state else 10
        anim_embed_dim = config.get("state_preprocessing", {}).get("anim_embed_dim", 16)

        model = create_model(
            num_actions=num_actions,
            num_history_frames=temporal_config.get("num_history_frames", 4),
            num_action_history=temporal_config.get("num_action_history", 4),
            gru_hidden_size=config["model"].get("gru_hidden_size", 256),
            gru_num_layers=config["model"].get("gru_num_layers", 1),
            use_state=use_state,
            num_state_features=num_state_features,
            hero_anim_vocab_size=(
                state_preprocessor.hero_vocab_size if use_state else 67
            ),
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size if use_state else 54,
            anim_embed_dim=anim_embed_dim,
            conv_features=tuple(config["model"]["conv_features"]),
            dense_features=tuple(config["model"]["dense_features"]),
            state_encoder_features=tuple(
                config["model"].get("state_encoder_features", [64, 64])
            ),
            state_output_features=config["model"].get("state_output_features", 64),
            action_history_features=config["model"].get("action_history_features", 64),
            dropout_rate=config["model"].get("dropout_rate", 0.0),
            use_batch_norm=config["model"].get("use_batch_norm", True),
        )

    elif model_name == "causal_transformer":
        from BC_training.models.causal_transformer import create_model

        temporal_config = config.get("temporal", {})
        use_state = config.get("dataset", {}).get("use_state", False)
        num_state_features = state_preprocessor.continuous_dim if use_state else 10
        anim_embed_dim = config.get("state_preprocessing", {}).get("anim_embed_dim", 16)

        model = create_model(
            num_actions=num_actions,
            num_history_frames=temporal_config.get("num_history_frames", 4),
            num_action_history=temporal_config.get("num_action_history", 4),
            d_model=config["model"].get("d_model", 256),
            num_heads=config["model"].get("num_heads", 4),
            num_layers=config["model"].get("num_layers", 2),
            d_ff=config["model"].get("d_ff", 512),
            max_seq_len=config["model"].get("max_seq_len", 32),
            use_state=use_state,
            num_state_features=num_state_features,
            hero_anim_vocab_size=(
                state_preprocessor.hero_vocab_size if use_state else 67
            ),
            npc_anim_vocab_size=state_preprocessor.npc_vocab_size if use_state else 54,
            anim_embed_dim=anim_embed_dim,
            conv_features=tuple(config["model"]["conv_features"]),
            dense_features=tuple(config["model"]["dense_features"]),
            state_encoder_features=tuple(
                config["model"].get("state_encoder_features", [64, 64])
            ),
            state_output_features=config["model"].get("state_output_features", 64),
            action_history_features=config["model"].get("action_history_features", 64),
            dropout_rate=config["model"].get("dropout_rate", 0.0),
            use_batch_norm=config["model"].get("use_batch_norm", True),
        )

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Model has {param_count:,} parameters")

    return LoadedModel(
        model=model,
        params=params,
        batch_stats=batch_stats,
        config=config,
        model_name=model_name,
        state_preprocessor=state_preprocessor,
    )

