"""Base agent for single-frame BC models (pure_cnn, hybrid_state).

Preprocessing to match BC training:
1. Frames: [H, W, C] uint8 (from eldengym) -> [C, H, W] float32 normalized to [0, 1]
2. State (if use_state=True):
   - Extract raw 19-dim state from eldengym observation
   - Use StatePreprocessor to compute:
     * continuous: [10] normalized HP/SP/FP ratios + distance features
     * hero_anim_idx: int32 embedding index for hero animation
     * npc_anim_idx: int32 embedding index for NPC animation
"""

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, Optional
import logging

# Add BC_training to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "BC_training"))

from ..model_loader import LoadedModel
from BC_training.common.state_preprocessing import STATE_INDICES

logger = logging.getLogger(__name__)


class BaseAgent:
    """Agent wrapper for single-frame BC models.

    Handles observation preprocessing and action selection for:
    - pure_cnn: Vision-only models
    - hybrid_state: Vision + game state + animation embeddings

    Converts eldengym observations to model input format and
    model outputs to environment action format.
    """

    def __init__(
        self,
        model: LoadedModel,
        action_threshold: float = 0.5,
        frame_shape: tuple = (3, 144, 256),  # C, H, W expected by model
    ):
        """Initialize base agent.

        Args:
            model: Loaded BC model
            action_threshold: Threshold for converting probabilities to binary actions
            frame_shape: Expected frame shape (C, H, W) for the model
        """
        self.model = model
        self.action_threshold = action_threshold
        self.frame_shape = frame_shape
        self.num_actions = model.num_actions

        # Expose model properties
        self.is_temporal = model.is_temporal
        self.use_state = model.use_state

        if self.is_temporal:
            raise ValueError(
                f"BaseAgent does not support temporal models. "
                f"Use TemporalAgent for model '{model.model_name}'"
            )

        logger.info(f"Initialized BaseAgent for {model.model_name}")
        logger.info(f"  use_state: {self.use_state}")
        logger.info(f"  action_threshold: {action_threshold}")
        logger.info(f"  frame_shape: {frame_shape}")

    def reset(self):
        """Reset agent state (no-op for single-frame models)."""
        pass

    def _preprocess_frame(self, frame: np.ndarray) -> jnp.ndarray:
        """Preprocess frame from eldengym format to model input.

        Args:
            frame: RGB frame [H, W, C] uint8 from eldengym

        Returns:
            Preprocessed frame [1, C, H, W] float32 normalized
        """
        # eldengym returns [H, W, C] uint8
        # Model expects [B, C, H, W] float32 normalized to [0, 1]

        # Resize if needed (using simple slicing/padding for now)
        # TODO: Use proper resizing if frame sizes don't match
        target_h, target_w = self.frame_shape[1], self.frame_shape[2]
        h, w = frame.shape[:2]

        if h != target_h or w != target_w:
            # Simple center crop/pad - in production use proper resize
            import cv2

            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Convert to float and normalize
        frame = frame.astype(np.float32) / 255.0

        # Transpose to [C, H, W] and add batch dimension
        frame = np.transpose(frame, (2, 0, 1))  # [H, W, C] -> [C, H, W]
        frame = frame[np.newaxis, ...]  # [C, H, W] -> [1, C, H, W]

        return jnp.array(frame)

    def _preprocess_state(self, obs: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Preprocess game state from eldengym observation.

        Args:
            obs: Dictionary observation from eldengym with game state

        Returns:
            Dict with 'continuous', 'hero_anim_idx', 'npc_anim_idx'
        """
        if not self.use_state or self.model.state_preprocessor is None:
            return {}

        # Extract raw state values from eldengym observation
        # eldengym provides these in the info dict or observation dict
        raw_state = self._extract_raw_state(obs)

        # Use the state preprocessor from training
        processed = self.model.state_preprocessor.process(raw_state)

        # Convert to JAX arrays with batch dimension
        return {
            "state": jnp.array(processed["continuous"][np.newaxis, ...]),
            "hero_anim_idx": jnp.array([processed["hero_anim_idx"]]),
            "npc_anim_idx": jnp.array([processed["npc_anim_idx"]]),
        }

    def _extract_raw_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract raw state array from eldengym observation.

        Builds state array matching the exact format used in BC training.
        Uses STATE_INDICES from BC_training to ensure correct ordering.

        Args:
            obs: Observation dict from eldengym (combined obs + info)

        Returns:
            Raw state array [19] matching training data format
        """
        # Build state array matching STATE_INDICES order from BC_training
        state = np.zeros(19, dtype=np.float32)

        # Map eldengym keys to state indices using STATE_INDICES
        # eldengym uses both CamelCase (from memory) and snake_case (from info dict)
        state[STATE_INDICES["HeroHp"]] = obs.get("HeroHp", obs.get("player_hp", 0))
        state[STATE_INDICES["HeroMaxHp"]] = obs.get("HeroMaxHp", obs.get("player_max_hp", 1))
        state[STATE_INDICES["HeroSp"]] = obs.get("HeroSp", obs.get("player_sp", 0))
        state[STATE_INDICES["HeroMaxSp"]] = obs.get("HeroMaxSp", obs.get("player_max_sp", 1))
        state[STATE_INDICES["HeroFp"]] = obs.get("HeroFp", obs.get("player_fp", 0))
        state[STATE_INDICES["HeroMaxFp"]] = obs.get("HeroMaxFp", obs.get("player_max_fp", 1))
        state[STATE_INDICES["HeroGlobalPosX"]] = obs.get("HeroGlobalPosX", obs.get("player_x", 0))
        state[STATE_INDICES["HeroGlobalPosY"]] = obs.get("HeroGlobalPosY", obs.get("player_y", 0))
        state[STATE_INDICES["HeroGlobalPosZ"]] = obs.get("HeroGlobalPosZ", obs.get("player_z", 0))
        state[STATE_INDICES["HeroAngle"]] = obs.get("HeroAngle", obs.get("player_angle", 0))
        state[STATE_INDICES["HeroAnimId"]] = obs.get("HeroAnimId", obs.get("player_animation_id", 0))
        state[STATE_INDICES["NpcHp"]] = obs.get("NpcHp", obs.get("target_hp", 0))
        state[STATE_INDICES["NpcMaxHp"]] = obs.get("NpcMaxHp", obs.get("target_max_hp", 1))
        state[STATE_INDICES["NpcId"]] = obs.get("NpcId", obs.get("target_id", 0))
        state[STATE_INDICES["NpcGlobalPosX"]] = obs.get("NpcGlobalPosX", obs.get("target_x", 0))
        state[STATE_INDICES["NpcGlobalPosY"]] = obs.get("NpcGlobalPosY", obs.get("target_y", 0))
        state[STATE_INDICES["NpcGlobalPosZ"]] = obs.get("NpcGlobalPosZ", obs.get("target_z", 0))
        state[STATE_INDICES["NpcGlobalPosAngle"]] = obs.get("NpcGlobalPosAngle", obs.get("target_angle", 0))
        state[STATE_INDICES["NpcAnimId"]] = obs.get("NpcAnimId", obs.get("target_animation_id", 0))

        return state

    def act(self, observation: Dict[str, Any]) -> np.ndarray:
        """Select action based on observation.

        Args:
            observation: Environment observation dict with 'frame' and optionally state info

        Returns:
            Multi-binary action array [num_actions]
        """
        probs = self.get_action_probs(observation)
        action = (probs > self.action_threshold).astype(np.float32)
        return action

    def get_action_probs(self, observation: Dict[str, Any]) -> np.ndarray:
        """Get action probabilities from model.

        Args:
            observation: Environment observation dict

        Returns:
            Action probabilities [num_actions]
        """
        # Extract and preprocess frame
        if isinstance(observation, dict):
            frame = observation.get("frame", observation.get("rgb", observation))
        else:
            frame = observation

        frames = self._preprocess_frame(frame)

        # Get state features if needed
        if self.use_state:
            state_data = self._preprocess_state(observation)
            probs = self.model.get_action_probs(
                frames,
                state=state_data.get("state"),
                hero_anim_idx=state_data.get("hero_anim_idx"),
                npc_anim_idx=state_data.get("npc_anim_idx"),
            )
        else:
            probs = self.model.get_action_probs(frames)

        # Remove batch dimension and convert to numpy
        return np.array(probs[0])

    def __repr__(self) -> str:
        return (
            f"BaseAgent(model={self.model.model_name}, "
            f"use_state={self.use_state}, "
            f"threshold={self.action_threshold})"
        )

