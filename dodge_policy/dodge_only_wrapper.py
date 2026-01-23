"""Wrapper for dodge-only action space with continuous forward movement."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DodgeOnlyWrapper(gym.ActionWrapper):
    """Converts Discrete(2) dodge decision to MultiBinary(5) with constant forward.

    Actions:
        0: No dodge (keep walking forward)
        1: Dodge (forward + dodge roll)

    The agent always walks forward. The only decision is when to dodge.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Change action space to Discrete(2)
        self.action_space = spaces.Discrete(2)

        # Action indices in underlying MultiBinary(5)
        # [forward, back, left, right, dodge]
        self.ACTION_FORWARD = 0
        self.ACTION_DODGE = 4

    def action(self, action: int) -> np.ndarray:
        """Convert discrete action to MultiBinary.

        Args:
            action: 0 = no dodge, 1 = dodge

        Returns:
            MultiBinary(5) action with forward always held
        """
        # Always hold forward
        multi_action = np.zeros(5, dtype=np.float32)
        multi_action[self.ACTION_FORWARD] = 1.0

        # Add dodge if action == 1
        if action == 1:
            multi_action[self.ACTION_DODGE] = 1.0

        return multi_action


class DodgeOnlyObsWrapper(gym.ObservationWrapper):
    """Filters observation to only boss_anim_id and elapsed_frames.

    Output observation dict contains only:
        - boss_anim_id: Boss animation ID (for embedding)
        - elapsed_frames: Frames since animation started (for sinusoidal encoding)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # We'll return a dict with just the two keys we need
        # The actual observation space shape will be determined by preprocessing

    def observation(self, obs: dict) -> dict:
        """Filter observation to timing features only."""
        return {
            'boss_anim_id': obs['boss_anim_id'],
            'elapsed_frames': obs['elapsed_frames'],
        }
