"""PPO training script for dodge policy."""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

from .agent import Agent, create_agent
from .ppo import PPOTrainer, NUM_CONTINUOUS, ANIM_VOCAB_SIZE
from .config import load_config, save_config, TrainConfig
from .env_factory import make_env, get_obs_shape, get_num_actions
from .dodge_window_model import DodgeWindowModel


def train(config: TrainConfig):
    """Run PPO training.

    Args:
        config: Training configuration
    """
    # Set random seed
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config.log_dir) / f"{config.exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Save config
    save_config(config, str(exp_dir / "config.yaml"))

    print("=" * 80)
    print("DODGE POLICY - PPO TRAINING")
    print("=" * 80)
    print(f"\nExperiment: {config.exp_name}")
    print(f"Log directory: {exp_dir}")
    print(f"Seed: {config.seed}")

    # Create environment
    print("\nCreating environment...")
    env = make_env(config.env)
    obs_shape = get_obs_shape(env)
    num_actions = get_num_actions(env)
    print(f"  Observation shape: {obs_shape}")
    print(f"  Number of actions: {num_actions}")

    # Create agent
    print("\nCreating agent...")
    key, agent_key = jax.random.split(key)
    agent, train_state = create_agent(
        key=agent_key,
        num_continuous=NUM_CONTINUOUS,
        num_actions=num_actions,
        learning_rate=config.ppo.learning_rate,
        hidden_dims=config.ppo.hidden_dims,
        max_grad_norm=config.ppo.max_grad_norm,
        anim_embed_dim=32,
        anim_vocab_size=ANIM_VOCAB_SIZE,
        sinusoidal_scales=8,
    )
    print(f"  Hidden dims: {config.ppo.hidden_dims}")
    print(f"  Learning rate: {config.ppo.learning_rate}")
    print(f"  Continuous features: {NUM_CONTINUOUS}")
    print(f"  Anim embedding: 32 dim, {ANIM_VOCAB_SIZE} vocab")
    print(f"  Sinusoidal scales: 8 (16 dim)")

    # Load dodge window model if provided
    dodge_window_model = None
    if config.dodge_window_model:
        print(f"\nLoading dodge window model...")
        dodge_window_model = DodgeWindowModel.load(config.dodge_window_model)
        print(f"  Dodge window reward scale: {config.dodge_window_reward}")

    # Create trainer
    print("\nCreating trainer...")
    key, trainer_key = jax.random.split(key)
    trainer = PPOTrainer(
        config=config.ppo,
        agent=agent,
        train_state=train_state,
        env=env,
        obs_shape=obs_shape,
        num_actions=num_actions,
        key=trainer_key,
        live_obs_plot=getattr(config, '_live_obs', False),
        env_config=config.env,
        dodge_window_model=dodge_window_model,
        dodge_window_reward=config.dodge_window_reward,
    )

    # Training info
    print(f"\nTraining configuration:")
    print(f"  Total timesteps: {config.ppo.total_timesteps:,}")
    print(f"  Batch size: {config.ppo.batch_size}")
    print(f"  Minibatch size: {config.ppo.minibatch_size}")
    print(f"  Number of updates: {config.ppo.num_updates}")
    print(f"  Update epochs: {config.ppo.update_epochs}")

    # Initialize wandb if tracking enabled
    wandb_run = None
    if config.track:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb_project or "dodge-policy",
            entity=config.wandb_entity,
            name=f"{config.exp_name}_{timestamp}",
            config={
                "exp_name": config.exp_name,
                "seed": config.seed,
                "ppo": {
                    "num_steps": config.ppo.num_steps,
                    "total_timesteps": config.ppo.total_timesteps,
                    "learning_rate": config.ppo.learning_rate,
                    "gamma": config.ppo.gamma,
                    "gae_lambda": config.ppo.gae_lambda,
                    "clip_eps": config.ppo.clip_eps,
                    "ent_coef": config.ppo.ent_coef,
                    "vf_coef": config.ppo.vf_coef,
                    "update_epochs": config.ppo.update_epochs,
                    "num_minibatches": config.ppo.num_minibatches,
                    "hidden_dims": config.ppo.hidden_dims,
                },
                "env": {
                    "env_id": config.env.env_id,
                    "soft_margin": config.env.soft_margin,
                    "hard_margin": config.env.hard_margin,
                    "hit_penalty": config.env.hit_penalty,
                    "dodge_penalty": config.env.dodge_penalty,
                    "danger_zone_penalty": config.env.danger_zone_penalty,
                    "oob_penalty": config.env.oob_penalty,
                },
            },
            sync_tensorboard=False,
        )
        print(f"  Wandb: {wandb_run.url}")

    start_time = time.time()

    def training_callback(info: dict):
        update = info['update']
        global_step = info['global_step']

        # Save checkpoint periodically
        if update % config.save_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{update}.npz"
            np.savez(
                checkpoint_path,
                params=jax.device_get(trainer.train_state.params),
                step=global_step,
                update=update,
            )

    print("\nStarting training...\n")

    try:
        final_state = trainer.train(callback=training_callback, wandb_run=wandb_run)

        # Save final checkpoint
        final_path = checkpoint_dir / "final.npz"
        np.savez(
            final_path,
            params=jax.device_get(final_state.params),
            step=trainer.global_step,
            update=trainer.update_count,
        )
        print(f"\nSaved final checkpoint: {final_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        # Save interrupt checkpoint
        interrupt_path = checkpoint_dir / "interrupt.npz"
        np.savez(
            interrupt_path,
            params=jax.device_get(trainer.train_state.params),
            step=trainer.global_step,
            update=trainer.update_count,
        )
        print(f"Saved interrupt checkpoint: {interrupt_path}")

    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()

    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTotal timesteps: {trainer.global_step:,}")
    print(f"Total updates: {trainer.update_count}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Steps/second: {trainer.global_step / elapsed:.1f}")
    print(f"\nExperiment saved to: {exp_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PPO training for dodge policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="dodge_policy/configs/default.yaml",
        help="Path to YAML configuration file",
    )

    # Override common options
    parser.add_argument("--exp-name", type=str, help="Experiment name")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, help="Total timesteps")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--host", type=str, help="Siphon server address")
    parser.add_argument("--boundary", type=str, help="Path to boundary JSON")
    parser.add_argument("--launch-game", action="store_true", help="Launch game")
    parser.add_argument("--live-plot", action="store_true", help="Enable live SDF plot")
    parser.add_argument("--live-obs", action="store_true", help="Enable live obs debug plot")

    # Wandb
    parser.add_argument("--track", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, help="Wandb project name")

    # Dodge window model
    parser.add_argument("--dodge-window-model", type=str, help="Path to dodge window model JSON")
    parser.add_argument("--dodge-window-reward", type=float, default=1.0,
                        help="Reward scale for dodging in window (default: 1.0)")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.seed:
        config.seed = args.seed
    if args.total_timesteps:
        config.ppo.total_timesteps = args.total_timesteps
    if args.learning_rate:
        config.ppo.learning_rate = args.learning_rate
    if args.host:
        config.env.host = args.host
    if args.boundary:
        config.env.boundary_path = args.boundary
    if args.launch_game:
        config.env.launch_game = True
    if args.live_plot:
        config.env.live_plot = True
    if args.live_obs:
        config._live_obs = True
    if args.track:
        config.track = True
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.dodge_window_model:
        config.dodge_window_model = args.dodge_window_model
    if args.dodge_window_reward != 1.0:  # Only override if explicitly set
        config.dodge_window_reward = args.dodge_window_reward

    train(config)


if __name__ == "__main__":
    main()
