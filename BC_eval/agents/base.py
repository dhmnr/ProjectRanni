"""Base agent for single-frame BC models (pure_cnn, hybrid_state).

Preprocessing to match BC training:
1. Frames: [H, W, C] uint8 (from eldengym) -> [C, H, W] float32 normalized to [0, 1]
2. State (if use_state=True):
   - Extract raw 19-dim state from eldengym observation
   - Use StatePreprocessor to compute:
     * continuous: [10] normalized HP/SP/FP ratios + distance features
     * hero_anim_idx: int32 embedding index for hero animation
     * npc_anim_idx: int32 embedding index for NPC animation
3. Actions: Maps model's semantic actions to environment's raw key actions
"""

import json
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, List, Optional
import logging

# Add BC_training to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "BC_training"))

from ..model_loader import LoadedModel
from BC_training.common.state_preprocessing import STATE_INDICES

logger = logging.getLogger(__name__)

# Actions to ignore (gym handles these automatically)
DISABLED_ACTIONS = {'lock_on'}


def load_training_actions_from_keybinds(keybinds_path: str) -> List[str]:
    """Load training action names from keybinds JSON file.

    Args:
        keybinds_path: Path to keybinds JSON file (e.g., keybinds_v2_7.json)

    Returns:
        List of action names ordered by their index
    """
    with open(keybinds_path) as f:
        keybinds = json.load(f)

    # Build list ordered by index
    actions = keybinds.get("actions", {})
    indexed_actions = [(info["index"], name) for name, info in actions.items()]
    indexed_actions.sort(key=lambda x: x[0])

    return [name for _, name in indexed_actions]


def build_action_mapping(
    env_action_keys: List[str],
    env_keybinds: Dict[str, str],
    training_actions: List[str],
    disabled_actions: Optional[set] = None,
) -> Dict[int, List[int]]:
    """Build mapping from training action indices to environment action indices.

    Args:
        env_action_keys: List of raw key names from env (e.g., ['W', 'S', 'A', ...])
        env_keybinds: Dict mapping key names to semantic actions (e.g., {'W': 'move_forward'})
        training_actions: List of semantic action names from training data
        disabled_actions: Set of action names to disable (won't be sent to env)

    Returns:
        Dict mapping training action index -> list of env action indices
    """
    disabled_actions = disabled_actions or DISABLED_ACTIONS

    # Build reverse mapping: semantic action -> list of env key indices
    semantic_to_env_indices = {}
    for env_idx, key in enumerate(env_action_keys):
        semantic = env_keybinds.get(key)
        if semantic:
            if semantic not in semantic_to_env_indices:
                semantic_to_env_indices[semantic] = []
            semantic_to_env_indices[semantic].append(env_idx)

    # Build training index -> env indices mapping
    mapping = {}
    for train_idx, action_name in enumerate(training_actions):
        # Skip disabled actions
        if action_name in disabled_actions:
            logger.info(f"Action '{action_name}' disabled (handled by env)")
            mapping[train_idx] = []
            continue

        env_indices = semantic_to_env_indices.get(action_name, [])
        if env_indices:
            # Use first matching key (e.g., LEFT_SHIFT for dodge, not BUTTON5)
            mapping[train_idx] = [env_indices[0]]
        else:
            logger.warning(f"Training action '{action_name}' not found in env keybinds")
            mapping[train_idx] = []

    return mapping


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
        env_action_keys: Optional[List[str]] = None,
        env_keybinds: Optional[Dict[str, str]] = None,
        training_keybinds_path: Optional[str] = None,
    ):
        """Initialize base agent.

        Args:
            model: Loaded BC model
            action_threshold: Threshold for converting probabilities to binary actions
            frame_shape: Expected frame shape (C, H, W) for the model
            env_action_keys: List of raw key names from environment
            env_keybinds: Dict mapping key names to semantic actions
            training_keybinds_path: Path to keybinds JSON used during training (e.g., keybinds_v2_7.json)
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

        # Setup action mapping from model outputs to env actions
        self.action_mapping = None
        self.env_num_actions = None
        self.training_actions = None
        if env_action_keys is not None and env_keybinds is not None:
            if training_keybinds_path:
                self.training_actions = load_training_actions_from_keybinds(training_keybinds_path)
                logger.info(f"Loaded {len(self.training_actions)} training actions from {training_keybinds_path}")
            else:
                raise ValueError("training_keybinds_path is required for action mapping")
            self.action_mapping = build_action_mapping(
                env_action_keys, env_keybinds, self.training_actions
            )
            self.env_num_actions = len(env_action_keys)
            logger.info(f"Action mapping: {self.action_mapping}")

        logger.info(f"Initialized BaseAgent for {model.model_name}")
        logger.info(f"  use_state: {self.use_state}")
        logger.info(f"  action_threshold: {action_threshold}")
        logger.info(f"  frame_shape: {frame_shape}")
        logger.info(f"  model_num_actions: {self.num_actions}")
        logger.info(f"  env_num_actions: {self.env_num_actions}")

    def reset(self):
        """Reset agent state (no-op for single-frame models)."""
        pass

    def _preprocess_frame(self, frame: np.ndarray) -> jnp.ndarray:
        """Preprocess frame from eldengym format to model input.

        Args:
            frame: BGR frame [H, W, C] uint8 from eldengym (cv2.imdecode returns BGR)

        Returns:
            Preprocessed frame [1, C, H, W] float32 normalized
        """
        # eldengym returns [H, W, C] uint8 in BGR format (from cv2.imdecode)
        # Training data was also BGR, so no conversion needed

        # Resize if needed
        target_h, target_w = self.frame_shape[1], self.frame_shape[2]
        h, w = frame.shape[:2]

        if h != target_h or w != target_w:
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

    def _map_action_to_env(self, model_action: np.ndarray) -> np.ndarray:
        """Map model's action output to environment's action space.

        Args:
            model_action: Binary action array from model [model_num_actions]

        Returns:
            Binary action array for environment [env_num_actions]
        """
        if self.action_mapping is None:
            # No mapping configured, return as-is
            return model_action

        env_action = np.zeros(self.env_num_actions, dtype=np.float32)
        for model_idx, env_indices in self.action_mapping.items():
            if model_action[model_idx] > 0:
                for env_idx in env_indices:
                    env_action[env_idx] = 1.0

        return env_action

    def act(self, observation: Dict[str, Any]) -> np.ndarray:
        """Select action based on observation.

        Args:
            observation: Environment observation dict with 'frame' and optionally state info

        Returns:
            Multi-binary action array [env_num_actions] mapped to environment action space
        """
        probs = self.get_action_probs(observation)
        self._last_probs = probs  # Store for logging
        model_action = (probs > self.action_threshold).astype(np.float32)

        # Map to environment action space
        return self._map_action_to_env(model_action)

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

