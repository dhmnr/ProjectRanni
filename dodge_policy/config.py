"""Configuration loading utilities for dodge policy training."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Tuple, Optional, Any, Dict
import yaml

from .ppo import PPOConfig


@dataclass
class EnvConfig:
    """Environment configuration."""
    # Environment
    env_id: str = "Margit-v0"
    host: str = "192.168.48.1:50051"
    launch_game: bool = False

    # Arena boundary
    boundary_path: str = "paths/margit_arena_boundary.json"
    soft_margin: float = 5.0
    hard_margin: float = 0.0
    save_file_name: str = "ER0000.Margit-v0.sl2"
    save_file_dir: str = "C:\\Users\\DM\\AppData\\Roaming\\EldenRing\\76561198217475593"
    # Reward shaping (penalties from wrapper)
    hit_penalty: float = -10.0
    dodge_penalty: float = 0.0
    danger_zone_penalty: float = 0.0
    oob_penalty: float = -1.0

    # Engagement reward (added in training loop)
    engage_reward: float = 0.05        # Reward per step when within engage_distance
    engage_distance: float = 7.0       # Distance threshold to be "engaging" with boss
    disengage_penalty: float = -0.005  # Penalty per step when beyond engage_distance

    # HP management
    refund_player_hp: bool = True
    refund_boss_hp: bool = False

    # Visualization
    live_plot: bool = False


@dataclass
class TrainConfig:
    """Full training configuration."""
    # Sub-configs
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvConfig = field(default_factory=EnvConfig)

    # Experiment
    exp_name: str = "dodge_ppo"
    seed: int = 42

    # Logging
    log_dir: str = "logs"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    track: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 10
    resume_from: Optional[str] = None

    # Dodge window model (learned from human demonstrations)
    dodge_window_model: Optional[str] = None  # Path to model JSON
    dodge_window_reward: float = 1.0          # Reward scale for dodging in window

    # Player dodge animation detection (HeroAnimId range for dodge/roll)
    player_dodge_anim_min: int = 0
    player_dodge_anim_max: int = 999

    # RUDDER reward shaping (online credit assignment)
    rudder_model: Optional[str] = None  # Path to pre-trained RUDDER model
    rudder_credit_scale: float = 1.0    # Scale factor for RUDDER credit
    rudder_update_freq: int = 5         # Update RUDDER every N rollouts
    rudder_save_freq: int = 50          # Save RUDDER model every N updates


def _update_dataclass(dc: Any, updates: Dict[str, Any]) -> None:
    """Update dataclass fields from dictionary."""
    dc_fields = {f.name for f in fields(dc)}
    for key, value in updates.items():
        if key in dc_fields:
            setattr(dc, key, value)


def load_config(config_path: str) -> TrainConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        TrainConfig with loaded values
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    config = TrainConfig()

    # Update top-level fields
    top_level_fields = {f.name for f in fields(config)} - {'ppo', 'env'}
    for key in top_level_fields:
        if key in raw_config:
            setattr(config, key, raw_config[key])

    # Update ppo config
    if 'ppo' in raw_config:
        ppo_dict = raw_config['ppo']
        # Handle hidden_dims as tuple
        if 'hidden_dims' in ppo_dict:
            ppo_dict['hidden_dims'] = tuple(ppo_dict['hidden_dims'])
        _update_dataclass(config.ppo, ppo_dict)

    # Update env config
    if 'env' in raw_config:
        _update_dataclass(config.env, raw_config['env'])

    return config


def save_config(config: TrainConfig, config_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path to save YAML file
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary
    config_dict = {
        'exp_name': config.exp_name,
        'seed': config.seed,
        'log_dir': config.log_dir,
        'wandb_project': config.wandb_project,
        'wandb_entity': config.wandb_entity,
        'track': config.track,
        'checkpoint_dir': config.checkpoint_dir,
        'save_freq': config.save_freq,
        'resume_from': config.resume_from,
        'dodge_window_model': config.dodge_window_model,
        'dodge_window_reward': config.dodge_window_reward,
        'ppo': {
            'num_envs': config.ppo.num_envs,
            'num_steps': config.ppo.num_steps,
            'total_timesteps': config.ppo.total_timesteps,
            'learning_rate': config.ppo.learning_rate,
            'gamma': config.ppo.gamma,
            'gae_lambda': config.ppo.gae_lambda,
            'clip_eps': config.ppo.clip_eps,
            'clip_vloss': config.ppo.clip_vloss,
            'ent_coef': config.ppo.ent_coef,
            'vf_coef': config.ppo.vf_coef,
            'max_grad_norm': config.ppo.max_grad_norm,
            'update_epochs': config.ppo.update_epochs,
            'num_minibatches': config.ppo.num_minibatches,
            'normalize_advantages': config.ppo.normalize_advantages,
            'hidden_dims': list(config.ppo.hidden_dims),
            'log_interval': config.ppo.log_interval,
            'save_interval': config.ppo.save_interval,
            'eval_interval': config.ppo.eval_interval,
        },
        'env': {
            'env_id': config.env.env_id,
            'host': config.env.host,
            'launch_game': config.env.launch_game,
            'boundary_path': config.env.boundary_path,
            'soft_margin': config.env.soft_margin,
            'hard_margin': config.env.hard_margin,
            'hit_penalty': config.env.hit_penalty,
            'dodge_penalty': config.env.dodge_penalty,
            'danger_zone_penalty': config.env.danger_zone_penalty,
            'oob_penalty': config.env.oob_penalty,
            'refund_player_hp': config.env.refund_player_hp,
            'refund_boss_hp': config.env.refund_boss_hp,
            'live_plot': config.env.live_plot,
            'engage_reward': config.env.engage_reward,
            'engage_distance': config.env.engage_distance,
            'disengage_penalty': config.env.disengage_penalty,
        },
    }

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
