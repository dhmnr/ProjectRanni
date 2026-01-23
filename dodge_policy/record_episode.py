"""Record human play episode with model-ready observations."""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import time

from .config import load_config, EnvConfig
from .env_factory import make_env, OBS_KEYS
from .ppo import (
    anim_id_to_index,
    NORM_ARENA_DIST, NORM_Z_DIFF, NORM_SDF, NORM_ELAPSED,
    IDX_DIST_TO_BOSS, IDX_BOSS_Z_REL, IDX_ANIM_ID, IDX_ELAPSED,
    IDX_SDF_VALUE, IDX_SDF_NORMAL_X, IDX_SDF_NORMAL_Y,
)


def sinusoidal_encoding(x: np.ndarray, num_scales: int = 8) -> np.ndarray:
    """Sinusoidal positional encoding for temporal values.

    Args:
        x: Input value (scalar or array)
        num_scales: Number of frequency scales (output dim = 2 * num_scales)

    Returns:
        Encoded values [2 * num_scales]
    """
    x = np.atleast_1d(x).astype(np.float32)
    if x.ndim == 1:
        x = x[:, None]

    freqs = 2.0 ** np.arange(num_scales)
    angles = x * freqs * np.pi

    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def flatten_obs(obs_dict: dict) -> np.ndarray:
    """Flatten observation dict to array."""
    values = []
    for k in OBS_KEYS:
        v = obs_dict[k]
        if k == "boss_anim_id":
            v = anim_id_to_index(v)
        elif k == "sdf_value":
            v = np.clip(v, -NORM_SDF, NORM_SDF)
        values.append(v)
    return np.array(values, dtype=np.float32)


def get_hero_anim_id(obs_dict: dict) -> int:
    """Extract player animation ID from observation dict."""
    return int(obs_dict.get("HeroAnimId", 0))


def preprocess_obs(obs: np.ndarray) -> dict:
    """Preprocess observation into model-ready format.

    Args:
        obs: Flat observation array [7] from OBS_KEYS

    Returns:
        Dict with all model inputs
    """
    # Raw values
    dist_to_boss = obs[IDX_DIST_TO_BOSS]
    boss_z_rel = obs[IDX_BOSS_Z_REL]
    anim_idx = int(obs[IDX_ANIM_ID])
    elapsed_frames = obs[IDX_ELAPSED]
    sdf_value = obs[IDX_SDF_VALUE]
    sdf_normal_x = obs[IDX_SDF_NORMAL_X]
    sdf_normal_y = obs[IDX_SDF_NORMAL_Y]

    # Normalized continuous features
    continuous_obs = np.array([
        dist_to_boss / NORM_ARENA_DIST,
        boss_z_rel / NORM_Z_DIFF,
        sdf_value / NORM_SDF,
        sdf_normal_x,
        sdf_normal_y,
    ], dtype=np.float32)

    # Elapsed frames: both raw normalized and sinusoidal
    elapsed_normalized = elapsed_frames / NORM_ELAPSED
    elapsed_sinusoidal = sinusoidal_encoding(
        np.array([elapsed_normalized]), num_scales=8
    )[0]  # [16]

    return {
        # Raw obs
        "obs_raw": obs,
        # Individual raw values
        "dist_to_boss": dist_to_boss,
        "boss_z_rel": boss_z_rel,
        "anim_idx": anim_idx,
        "elapsed_frames": elapsed_frames,
        "sdf_value": sdf_value,
        "sdf_normal_x": sdf_normal_x,
        "sdf_normal_y": sdf_normal_y,
        # Model-ready features
        "continuous_obs": continuous_obs,  # [5]
        "elapsed_normalized": elapsed_normalized,
        "elapsed_sinusoidal": elapsed_sinusoidal,  # [16]
    }


def record_episode(config: EnvConfig, output_dir: str = "recordings"):
    """Record a human play episode.

    Args:
        config: Environment configuration
        output_dir: Directory to save recordings
    """
    print("=" * 60)
    print("EPISODE RECORDER")
    print("=" * 60)
    print("\nCreating environment...")

    env = make_env(config)
    print(f"  Action space: {env.action_space}")
    print(f"  Obs keys: {OBS_KEYS}")

    # Storage
    all_obs_raw = []
    all_continuous_obs = []
    all_anim_idx = []
    all_boss_anim_id_raw = []  # Raw boss animation ID (before vocab conversion)
    all_elapsed_frames = []
    all_elapsed_normalized = []
    all_elapsed_sinusoidal = []
    all_dist_to_boss = []
    all_sdf_value = []
    all_hero_anim_id = []  # Player animation ID for dodge detection
    all_actions = []
    all_rewards = []
    all_dones = []
    all_player_damage = []
    all_boss_damage = []

    print("\nStarting recording...")
    print("Press Ctrl+C to stop recording\n")

    obs_dict, info = env.reset()

    # Debug: print all observation keys and sample values
    print("Observation dict keys:")
    for k, v in sorted(obs_dict.items()):
        print(f"  {k}: {v}")
    print()

    # Track HP for damage calculation (in case wrapper doesn't provide it)
    prev_player_hp = obs_dict.get("HeroHp", 0)
    prev_boss_hp = obs_dict.get("NpcHp", 0)

    obs = flatten_obs(obs_dict)
    step = 0
    start_time = time.time()
    step_times = []

    try:
        while True:
            step_start = time.time()

            # Preprocess current observation
            processed = preprocess_obs(obs)

            # Store observation data
            all_obs_raw.append(processed["obs_raw"])
            all_continuous_obs.append(processed["continuous_obs"])
            all_anim_idx.append(processed["anim_idx"])
            all_elapsed_frames.append(processed["elapsed_frames"])
            all_elapsed_normalized.append(processed["elapsed_normalized"])
            all_elapsed_sinusoidal.append(processed["elapsed_sinusoidal"])
            all_dist_to_boss.append(processed["dist_to_boss"])
            all_sdf_value.append(processed["sdf_value"])
            all_hero_anim_id.append(get_hero_anim_id(obs_dict))

            # No action from agent - human is playing
            # Send zeros (no buttons pressed from our side)
            action = np.zeros(5, dtype=np.float32)

            # Step environment
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)

            # Warn if step is very slow (indicates connection issue)
            if step_elapsed > 1.0:
                print(f"  !!! SLOW STEP: {step_elapsed:.2f}s - possible connection issue")

            next_obs = flatten_obs(next_obs_dict)
            done = terminated or truncated

            # Calculate damage from HP deltas (more reliable than wrapper)
            curr_player_hp = next_obs_dict.get("HeroHp", 0)
            curr_boss_hp = next_obs_dict.get("NpcHp", 0)
            player_dmg = max(0, prev_player_hp - curr_player_hp)
            boss_dmg = max(0, prev_boss_hp - curr_boss_hp)
            prev_player_hp = curr_player_hp
            prev_boss_hp = curr_boss_hp

            # Store transition
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)
            all_player_damage.append(player_dmg)
            all_boss_damage.append(boss_dmg)

            # Print damage events immediately
            if player_dmg > 0:
                print(f"  >>> HIT! Player took {player_dmg} damage (HP: {curr_player_hp})")
            if boss_dmg > 0:
                print(f"  >>> Boss took {boss_dmg} damage (HP: {curr_boss_hp})")

            # Print progress with hero anim for dodge visibility
            hero_anim = get_hero_anim_id(obs_dict)
            wall_time = time.time() - start_time
            real_fps = step / wall_time if wall_time > 0 else 0
            if step % 15 == 0:  # ~1 second at 15fps
                print(
                    f"Step {step:5d} | "
                    f"wall={wall_time:5.1f}s | "
                    f"fps={real_fps:4.1f} | "
                    f"anim={processed['anim_idx']:2d} | "
                    f"hero={hero_anim} | "
                    f"hp={curr_player_hp}/{curr_boss_hp}"
                )

            obs = next_obs
            obs_dict = next_obs_dict  # Update obs_dict for next iteration!
            step += 1

            if done:
                print(f"\nEpisode ended at step {step}")
                if terminated:
                    print("  Reason: terminated (player died)")
                if truncated:
                    print("  Reason: truncated")
                break

    except KeyboardInterrupt:
        print(f"\n\nRecording stopped by user at step {step}")

    finally:
        env.close()

    # Save recording
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"episode_{timestamp}.npz"

    print(f"\nSaving {len(all_obs_raw)} frames to {filename}...")

    np.savez_compressed(
        filename,
        # Raw observations
        obs_raw=np.array(all_obs_raw),  # [T, 7]
        # Model-ready features
        continuous_obs=np.array(all_continuous_obs),  # [T, 5]
        anim_idx=np.array(all_anim_idx),  # [T] - boss animation vocab index
        elapsed_frames=np.array(all_elapsed_frames),  # [T]
        elapsed_normalized=np.array(all_elapsed_normalized),  # [T]
        elapsed_sinusoidal=np.array(all_elapsed_sinusoidal),  # [T, 16]
        dist_to_boss=np.array(all_dist_to_boss),  # [T]
        sdf_value=np.array(all_sdf_value),  # [T]
        hero_anim_id=np.array(all_hero_anim_id),  # [T] - player animation for dodge detection
        # Actions and rewards
        actions=np.array(all_actions),  # [T, 5]
        rewards=np.array(all_rewards),  # [T]
        dones=np.array(all_dones),  # [T]
        # Damage taken
        player_damage=np.array(all_player_damage),  # [T]
        boss_damage=np.array(all_boss_damage),  # [T]
        # Metadata
        num_steps=step,
        fps=15,
    )

    print(f"Saved!")

    # Timing analysis
    total_wall_time = time.time() - start_time
    avg_step_time = np.mean(step_times) if step_times else 0
    max_step_time = np.max(step_times) if step_times else 0

    print(f"\nRecording summary:")
    print(f"  Total steps: {step}")
    print(f"  Game time: {step / 15:.1f} seconds (at 15fps)")
    print(f"  Wall time: {total_wall_time:.1f} seconds")
    print(f"  Real FPS: {step / total_wall_time:.1f}" if total_wall_time > 0 else "  Real FPS: N/A")
    print(f"  Avg step time: {avg_step_time*1000:.1f}ms")
    print(f"  Max step time: {max_step_time*1000:.1f}ms")
    print(f"  Player damage taken: {sum(all_player_damage)}")
    print(f"  Boss damage dealt: {sum(all_boss_damage)}")
    print(f"  Hits taken: {sum(1 for d in all_player_damage if d > 0)}")


def main():
    parser = argparse.ArgumentParser(description="Record human play episode")
    parser.add_argument(
        "--config",
        type=str,
        default="dodge_policy/configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recordings",
        help="Directory to save recordings",
    )
    parser.add_argument("--host", type=str, help="Siphon server address")

    args = parser.parse_args()

    config = load_config(args.config)
    if args.host:
        config.env.host = args.host

    record_episode(config.env, args.output_dir)


if __name__ == "__main__":
    main()
