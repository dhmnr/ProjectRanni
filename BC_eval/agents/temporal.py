"""Temporal agent for BC models with frame/action history (temporal_cnn, gru, causal_transformer).

Preprocessing to match BC training:
1. Frames: [H, W, C] uint8 (from eldengym) -> [C, H, W] float32 normalized to [0, 1]
   - Stacked into [T, C, H, W] where T = num_history_frames + 1 (oldest to newest)
2. Action history: [K, num_actions] float32 from past K actions
3. State (if use_state=True):
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
from collections import deque
import logging

# Add BC_training to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "BC_training"))

from ..model_loader import LoadedModel
from BC_training.common.state_preprocessing import STATE_INDICES

logger = logging.getLogger(__name__)


class TemporalAgent(BaseAgent):
    """Agent wrapper for temporal BC models with history buffers.

    Handles observation preprocessing and action selection for:
    - temporal_cnn: Frame stacking with action history
    - gru: Recurrent model with frame/action history
    - causal_transformer: Attention-based temporal model

    Maintains ring buffers for:
    - Frame history: Past N frames for temporal context
    - Action history: Past K actions for autoregressive conditioning

    Converts eldengym observations to model input format and
    model outputs to environment action format.
    """

    def __init__(
        self,
        model: LoadedModel,
        action_threshold: float = 0.5,
        frame_shape: tuple = (3, 144, 256),  # C, H, W expected by model
    ):
        """Initialize temporal agent.

        Args:
            model: Loaded BC model (must be temporal)
            action_threshold: Threshold for converting probabilities to binary actions
            frame_shape: Expected frame shape (C, H, W) for the model
        """
        # Don't call super().__init__ to avoid the is_temporal check
        self.model = model
        self.action_threshold = action_threshold
        self.frame_shape = frame_shape
        self.num_actions = model.num_actions

        # Expose model properties
        self.is_temporal = model.is_temporal
        self.use_state = model.use_state

        if not self.is_temporal:
            raise ValueError(
                f"TemporalAgent requires temporal models. "
                f"Use BaseAgent for model '{model.model_name}'"
            )

        # Temporal config from model
        self.num_history_frames = model.num_history_frames
        self.num_action_history = model.num_action_history
        self.total_frames = self.num_history_frames + 1  # history + current

        # Initialize history buffers
        self._init_buffers()

        logger.info(f"Initialized TemporalAgent for {model.model_name}")
        logger.info(f"  use_state: {self.use_state}")
        logger.info(f"  num_history_frames: {self.num_history_frames}")
        logger.info(f"  num_action_history: {self.num_action_history}")
        logger.info(f"  action_threshold: {action_threshold}")
        logger.info(f"  frame_shape: {frame_shape}")

    def _init_buffers(self):
        """Initialize frame and action history buffers."""
        # Frame buffer: stores preprocessed frames [C, H, W]
        # Use deque with maxlen for automatic FIFO behavior
        self.frame_buffer = deque(maxlen=self.total_frames)

        # Action buffer: stores past actions [num_actions]
        self.action_buffer = deque(maxlen=self.num_action_history)

        # Initialize with zeros
        zero_frame = np.zeros(self.frame_shape, dtype=np.float32)
        zero_action = np.zeros(self.num_actions, dtype=np.float32)

        for _ in range(self.total_frames):
            self.frame_buffer.append(zero_frame.copy())

        for _ in range(self.num_action_history):
            self.action_buffer.append(zero_action.copy())

    def reset(self):
        """Reset agent state - clear history buffers."""
        self._init_buffers()
        logger.debug("TemporalAgent buffers reset")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame from eldengym format.

        Args:
            frame: RGB frame [H, W, C] uint8 from eldengym

        Returns:
            Preprocessed frame [C, H, W] float32 normalized (no batch dim)
        """
        # Resize if needed
        target_h, target_w = self.frame_shape[1], self.frame_shape[2]
        h, w = frame.shape[:2]

        if h != target_h or w != target_w:
            import cv2

            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Convert to float and normalize
        frame = frame.astype(np.float32) / 255.0

        # Transpose to [C, H, W]
        frame = np.transpose(frame, (2, 0, 1))

        return frame

    def _get_stacked_frames(self) -> jnp.ndarray:
        """Get stacked frames from buffer.

        Returns:
            Stacked frames [1, T, C, H, W] where T = total_frames
        """
        # Stack frames: list of [C, H, W] -> [T, C, H, W]
        stacked = np.stack(list(self.frame_buffer), axis=0)
        # Add batch dimension
        stacked = stacked[np.newaxis, ...]  # [1, T, C, H, W]
        return jnp.array(stacked)

    def _get_action_history(self) -> jnp.ndarray:
        """Get action history from buffer.

        Returns:
            Action history [1, K, num_actions] where K = num_action_history
        """
        # Stack actions: list of [num_actions] -> [K, num_actions]
        history = np.stack(list(self.action_buffer), axis=0)
        # Add batch dimension
        history = history[np.newaxis, ...]  # [1, K, num_actions]
        return jnp.array(history)

    def _update_buffers(self, frame: np.ndarray, action: np.ndarray):
        """Update frame and action buffers with new data.

        Args:
            frame: Preprocessed frame [C, H, W]
            action: Action taken [num_actions]
        """
        self.frame_buffer.append(frame)
        self.action_buffer.append(action.astype(np.float32))

    def _preprocess_state(self, obs: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Preprocess game state from eldengym observation.

        Args:
            obs: Dictionary observation from eldengym with game state

        Returns:
            Dict with 'state', 'hero_anim_idx', 'npc_anim_idx'
        """
        if not self.use_state or self.model.state_preprocessor is None:
            return {}

        # Extract raw state values from eldengym observation
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

        Processes observation, updates history buffers, and returns action.

        Args:
            observation: Environment observation dict with 'frame' and optionally state info

        Returns:
            Multi-binary action array [num_actions]
        """
        probs = self.get_action_probs(observation)
        action = (probs > self.action_threshold).astype(np.float32)

        # Update action buffer with the action we're about to take
        # Note: The current frame was already added in get_action_probs
        self.action_buffer.append(action)

        return action

    def get_action_probs(self, observation: Dict[str, Any]) -> np.ndarray:
        """Get action probabilities from model.

        Updates frame buffer with current observation before inference.

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

        # Preprocess current frame
        processed_frame = self._preprocess_frame(frame)

        # Update frame buffer with current frame
        self.frame_buffer.append(processed_frame)

        # Get stacked frames and action history
        frames = self._get_stacked_frames()  # [1, T, C, H, W]
        action_history = self._get_action_history()  # [1, K, num_actions]

        # Get state features if needed
        if self.use_state:
            state_data = self._preprocess_state(observation)
            probs = self.model.get_action_probs(
                frames,
                action_history=action_history,
                state=state_data.get("state"),
                hero_anim_idx=state_data.get("hero_anim_idx"),
                npc_anim_idx=state_data.get("npc_anim_idx"),
            )
        else:
            probs = self.model.get_action_probs(frames, action_history=action_history)

        # Remove batch dimension and convert to numpy
        return np.array(probs[0])

    def __repr__(self) -> str:
        return (
            f"TemporalAgent(model={self.model.model_name}, "
            f"use_state={self.use_state}, "
            f"history_frames={self.num_history_frames}, "
            f"action_history={self.num_action_history}, "
            f"threshold={self.action_threshold})"
        )

