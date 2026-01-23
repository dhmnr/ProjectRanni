"""Environment factory for dodge policy training."""

from typing import List
import gymnasium as gym

import eldengym
from eldengym import (
    ArenaBoundary,
    AnimFrameWrapper,
    SDFObsWrapper,
    OOBSafetyWrapper,
    HPRefundWrapper,
    DodgePolicyRewardWrapper,
)

from .config import EnvConfig


# Actions for dodge policy (movement + dodge only)
DODGE_POLICY_ACTIONS: List[str] = [
    "move_forward",
    "move_back",
    "move_left",
    "move_right",
    "dodge_roll/dash",
]

# Action indices
ACTION_FORWARD = 0
ACTION_BACK = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_DODGE = 4

# Observation keys to use for dodge policy
#
# Full observation dict from wrapped environment contains:
#   frame             - RGB image (excluded, use for visualization only)
#   HeroHp            - Player current HP (int)
#   HeroMaxHp         - Player max HP (int)
#   NpcHp             - Boss current HP (int)
#   NpcMaxHp          - Boss max HP (int)
#   HeroAnimId        - Player animation ID (int)
#   NpcAnimId         - Boss animation ID (int, same as boss_anim_id)
#   player_x/y/z      - Player world coordinates (float)
#   boss_x/y/z        - Boss world coordinates (float)
#   dist_to_boss      - Euclidean distance to boss (float)
#   boss_z_relative   - Boss Z relative to player (float)
#   boss_anim_id      - Boss animation ID from AnimFrameWrapper (int)
#   elapsed_frames    - Frames since animation started (int)
#   sdf_value         - Signed distance to arena boundary (float, negative=outside)
#   sdf_normal_x/y    - Normal vector pointing toward boundary center (float)
#
# Minimal obs for dodge policy (5 continuous + anim_id + elapsed_frames):
OBS_KEYS = [
    # Distance features
    "dist_to_boss",      # 0: how far is boss
    "boss_z_relative",   # 1: vertical offset
    # Animation tracking (handled specially: embedding + sinusoidal)
    "boss_anim_id",      # 2: boss animation ID (embedding)
    "elapsed_frames",    # 3: frames into animation (sinusoidal)
    # SDF boundary awareness
    "sdf_value",         # 4: distance to boundary
    "sdf_normal_x",      # 5: boundary direction x
    "sdf_normal_y",      # 6: boundary direction y
]


def make_env(config: EnvConfig) -> gym.Env:
    """Create dodge policy environment with full wrapper stack.

    Args:
        config: Environment configuration

    Returns:
        Wrapped environment ready for training

    Wrapper order (inside to outside):
        1. Base EldenGym environment with filtered action space
        2. HPRefundWrapper - Refund player HP for infinite exploration
        3. AnimFrameWrapper - Add animation frame tracking
        4. SDFObsWrapper - Add SDF observations for boundary awareness
        5. OOBSafetyWrapper - Teleport on OOB detection
        6. DodgePolicyRewardWrapper - Reward shaping
    """
    # Load arena boundary
    boundary = ArenaBoundary.load(config.boundary_path)

    # Create base environment with filtered action space
    env = eldengym.make(
        config.env_id,
        launch_game=config.launch_game,
        host=config.host,
        actions=DODGE_POLICY_ACTIONS,
        save_file_name=config.save_file_name, 
        save_file_dir=config.save_file_dir
    )

    # Apply wrappers (order matters!)
    env = HPRefundWrapper(
        env,
        refund_player=config.refund_player_hp,
        refund_boss=config.refund_boss_hp,
    )

    env = AnimFrameWrapper(env)

    env = SDFObsWrapper(
        env,
        boundary=boundary,
        live_plot=config.live_plot,
    )

    env = OOBSafetyWrapper(
        env,
        boundary=boundary,
        soft_margin=config.soft_margin,
        hard_margin=config.hard_margin,
    )

    env = DodgePolicyRewardWrapper(
        env,
        dodge_action_idx=ACTION_DODGE,
        hit_penalty=config.hit_penalty,
        dodge_penalty=config.dodge_penalty,
        danger_zone_penalty=config.danger_zone_penalty,
        oob_penalty=config.oob_penalty,
    )

    return env


def get_obs_shape(env: gym.Env) -> tuple:
    """Get flattened observation shape from environment.

    Returns shape based on OBS_KEYS only.
    """
    return (len(OBS_KEYS),)


def get_num_actions(env: gym.Env) -> int:
    """Get number of actions from environment.

    Args:
        env: Environment

    Returns:
        Number of actions (MultiBinary dimension)
    """
    return env.action_space.shape[0]
