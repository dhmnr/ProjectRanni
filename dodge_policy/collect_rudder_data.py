"""Collect data for RUDDER credit assignment model.

Random dodge policy with constant forward movement.
Collects observations, actions, rewards per episode.
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

from .dodge_only_wrapper import DodgeOnlyWrapper
from .config import load_config
from .env_factory import make_env


def collect_episode(env, steps: int, rng: np.random.Generator) -> dict:
    """Collect one episode of random dodge data.

    Args:
        env: DodgeOnlyWrapper environment
        steps: Number of steps per episode
        rng: Random number generator

    Returns:
        Dictionary with episode data arrays
    """
    # Pre-allocate arrays
    boss_anim_id = np.zeros(steps, dtype=np.int32)
    elapsed_frames = np.zeros(steps, dtype=np.int32)
    hero_anim_id = np.zeros(steps, dtype=np.int32)
    dist_to_boss = np.zeros(steps, dtype=np.float32)
    hero_hp = np.zeros(steps, dtype=np.int32)
    npc_hp = np.zeros(steps, dtype=np.int32)
    actions = np.zeros(steps, dtype=np.int32)
    rewards = np.zeros(steps, dtype=np.float32)
    damage_taken = np.zeros(steps, dtype=np.float32)
    dones = np.zeros(steps, dtype=np.bool_)

    obs_dict, _ = env.reset()

    for step in range(steps):
        # Store observation
        boss_anim_id[step] = int(obs_dict.get('boss_anim_id', 0))
        elapsed_frames[step] = int(obs_dict.get('elapsed_frames', 0))
        hero_anim_id[step] = int(obs_dict.get('HeroAnimId', 0))
        dist_to_boss[step] = float(obs_dict.get('dist_to_boss', 0))
        hero_hp[step] = int(obs_dict.get('HeroHp', 0))
        npc_hp[step] = int(obs_dict.get('NpcHp', 0))

        # Random dodge action (Discrete(2): 0=no dodge, 1=dodge)
        # Use low probability for dodge to avoid spam
        action = int(rng.random() < 0.1)  # 10% dodge probability
        actions[step] = action

        # Step environment
        next_obs_dict, reward, terminated, truncated, info = env.step(action)

        # Store results
        rewards[step] = float(reward)
        damage_taken[step] = float(info.get('player_damage_taken', 0))
        done = terminated or truncated
        dones[step] = done

        # Reset if done
        if done:
            obs_dict, _ = env.reset()
        else:
            obs_dict = next_obs_dict

    return {
        'boss_anim_id': boss_anim_id,
        'elapsed_frames': elapsed_frames,
        'hero_anim_id': hero_anim_id,
        'dist_to_boss': dist_to_boss,
        'hero_hp': hero_hp,
        'npc_hp': npc_hp,
        'actions': actions,
        'rewards': rewards,
        'damage_taken': damage_taken,
        'dones': dones,
        'total_reward': float(rewards.sum()),
        'total_damage': float(damage_taken.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect RUDDER training data")
    parser.add_argument("--config", default="dodge_policy/configs/dodge_only.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="rudder_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dodge-prob", type=float, default=0.1,
                        help="Probability of random dodge per step")

    args = parser.parse_args()
    console = Console()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and create environment
    console.print(f"[bold]RUDDER Data Collection[/bold]")
    console.print(f"  Episodes: {args.episodes}")
    console.print(f"  Steps per episode: {args.steps}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Dodge probability: {args.dodge_prob}")
    console.print()

    config = load_config(args.config)

    console.print("Creating environment...")
    base_env = make_env(config.env)
    env = DodgeOnlyWrapper(base_env)
    console.print(f"  Action space: {env.action_space}")
    console.print()

    # Random generator
    rng = np.random.default_rng(args.seed)

    # Collect episodes
    console.print(f"Collecting {args.episodes} episodes...\n")

    total_reward = 0
    total_damage = 0

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Episodes", total=args.episodes)

        for ep_idx in range(args.episodes):
            # Collect episode
            episode_data = collect_episode(env, args.steps, rng)

            # Save episode
            ep_path = output_dir / f"episode_{ep_idx:04d}.npz"
            np.savez_compressed(ep_path, **episode_data)

            total_reward += episode_data['total_reward']
            total_damage += episode_data['total_damage']

            progress.update(task, advance=1)

    env.close()

    # Summary
    console.print()
    console.print("[bold green]Collection complete![/bold green]")
    console.print(f"  Total episodes: {args.episodes}")
    console.print(f"  Total steps: {args.episodes * args.steps:,}")
    console.print(f"  Avg reward/episode: {total_reward / args.episodes:.2f}")
    console.print(f"  Avg damage/episode: {total_damage / args.episodes:.2f}")
    console.print(f"  Saved to: {output_dir}")

    # Save metadata
    metadata = {
        'episodes': args.episodes,
        'steps_per_episode': args.steps,
        'dodge_prob': args.dodge_prob,
        'seed': args.seed,
        'total_reward': total_reward,
        'total_damage': total_damage,
        'config_path': args.config,
        'timestamp': datetime.now().isoformat(),
    }
    np.savez(output_dir / "metadata.npz", **metadata)


if __name__ == "__main__":
    main()
