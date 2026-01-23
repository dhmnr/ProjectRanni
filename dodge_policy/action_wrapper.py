"""Action space wrapper to convert Discrete to MultiBinary with proper roll timing."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACTION_NAMES = ["nothing", "roll_fwd", "roll_back", "roll_left", "roll_right"]

# Movement directions for each roll action (without dodge)
MOVEMENT_MAP = {
    0: np.array([0, 0, 0, 0, 0], dtype=np.float32),  # Nothing
    1: np.array([1, 0, 0, 0, 0], dtype=np.float32),  # Forward
    2: np.array([0, 1, 0, 0, 0], dtype=np.float32),  # Backward
    3: np.array([0, 0, 1, 0, 0], dtype=np.float32),  # Left
    4: np.array([0, 0, 0, 1, 0], dtype=np.float32),  # Right
}

# Movement + dodge for the tap frame
ROLL_MAP = {
    0: np.array([0, 0, 0, 0, 0], dtype=np.float32),  # Nothing
    1: np.array([1, 0, 0, 0, 1], dtype=np.float32),  # Forward + dodge
    2: np.array([0, 1, 0, 0, 1], dtype=np.float32),  # Backward + dodge
    3: np.array([0, 0, 1, 0, 1], dtype=np.float32),  # Left + dodge
    4: np.array([0, 0, 0, 1, 1], dtype=np.float32),  # Right + dodge
}


class DiscreteActionWrapper(gym.Wrapper):
    """Convert Discrete(5) action space to MultiBinary(5) with proper roll timing.

    For roll actions (1-4), executes a multi-frame sequence:
        1. Movement direction only (pre_roll_frames)
        2. Movement + dodge tap (1 frame)
        3. Release all (post_roll_frames)

    This prevents holding dodge (which causes sprint) and ensures proper roll input.

    Actions:
        0: Nothing - stand still
        1: Roll forward
        2: Roll backward
        3: Roll left
        4: Roll right
    """

    def __init__(
        self,
        env: gym.Env,
        pre_roll_frames: int = 2,
        post_roll_frames: int = 1,
    ):
        """Initialize wrapper.

        Args:
            env: Environment to wrap
            pre_roll_frames: Frames to hold movement before dodge tap
            post_roll_frames: Frames to wait after dodge tap (release all)
        """
        super().__init__(env)
        self.action_space = spaces.Discrete(5)
        self.pre_roll_frames = pre_roll_frames
        self.post_roll_frames = post_roll_frames

    def step(self, action: int):
        """Execute action with proper roll timing.

        For action 0 (nothing): single frame, no buttons
        For actions 1-4 (rolls): multi-frame sequence
        """
        if action == 0:
            # Nothing - single frame
            return self.env.step(MOVEMENT_MAP[0])

        # Roll action - execute sequence
        total_reward = 0.0
        obs = None
        terminated = False
        truncated = False
        info = {}

        # Phase 1: Movement only (pre-roll frames)
        for _ in range(self.pre_roll_frames):
            obs, reward, terminated, truncated, info = self.env.step(MOVEMENT_MAP[action])
            total_reward += reward
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info

        # Phase 2: Movement + dodge tap (1 frame)
        obs, reward, terminated, truncated, info = self.env.step(ROLL_MAP[action])
        total_reward += reward
        if terminated or truncated:
            return obs, total_reward, terminated, truncated, info

        # Phase 3: Release all (post-roll frames)
        for _ in range(self.post_roll_frames):
            obs, reward, terminated, truncated, info = self.env.step(MOVEMENT_MAP[0])
            total_reward += reward
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)
