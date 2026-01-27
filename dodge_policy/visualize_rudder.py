"""Visualize RUDDER credit assignment on episodes (JAX version)."""

import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from .rudder_model import load_model, get_credit
from .dodge_window_model import DodgeWindowModel
from .ppo import anim_id_to_index, ANIM_VOCAB

# Colorblind-friendly palette
COLOR_POSITIVE = '#0072B2'  # Blue - positive credit
COLOR_NEGATIVE = '#D55E00'  # Orange - negative credit
COLOR_GT_WINDOW = '#CC79A7'  # Pink/magenta - ground truth window
COLOR_GT_LINE = '#882255'    # Dark magenta - ground truth center
COLOR_DAMAGE = '#D55E00'     # Orange - damage events


def get_credit_per_attack(model, params, data: dict, stats: dict, max_len: int = 64):
    """Compute credit by segmenting episode into attacks (for per-attack trained models).

    Optimized: collects all segments first, then runs batched inference.
    """
    T = len(data['actions'])
    boss_anims = data['boss_anim_id']

    # Normalize continuous features
    dist = (data['dist_to_boss'] - stats['dist_mean']) / stats['dist_std']
    hp = (data['hero_hp'] - stats['hp_mean']) / stats['hp_std']
    action = data['actions'].astype(np.float32)

    # Check if old model (with elapsed_frames) or new model (without)
    if 'elapsed_mean' in stats:
        elapsed = (data['elapsed_frames'] - stats['elapsed_mean']) / stats['elapsed_std']
        cont_stack = np.stack([elapsed, dist, hp, action], axis=-1).astype(np.float32)
    else:
        cont_stack = np.stack([dist, hp, action], axis=-1).astype(np.float32)

    # Collect all segments first
    segments = []  # List of (start_idx, length, boss_id, hero_arr, cont_arr)

    start_idx = 0
    for t in range(1, T + 1):
        is_boundary = (t == T) or (boss_anims[t] != boss_anims[t-1])

        if is_boundary:
            end_idx = t
            length = end_idx - start_idx

            if length >= 1:
                seg_boss = int(boss_anims[start_idx])
                seg_hero = data['hero_anim_id'][start_idx:end_idx].copy()
                seg_cont = cont_stack[start_idx:end_idx].copy()

                # Pad or truncate to max_len
                if length > max_len:
                    seg_hero = seg_hero[:max_len]
                    seg_cont = seg_cont[:max_len]
                elif length < max_len:
                    pad_len = max_len - length
                    seg_hero = np.pad(seg_hero, (0, pad_len), mode='edge')
                    seg_cont = np.pad(seg_cont, ((0, pad_len), (0, 0)), mode='edge')

                segments.append((start_idx, min(length, max_len), seg_boss, seg_hero, seg_cont))

            start_idx = t

    if not segments:
        return np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)

    # Batch all segments together
    n_segs = len(segments)
    batch_boss = np.array([s[2] for s in segments], dtype=np.int32)  # (N,)
    batch_hero = np.stack([s[3] for s in segments], axis=0)  # (N, max_len)
    batch_cont = np.stack([s[4] for s in segments], axis=0)  # (N, max_len, C)

    # Single batched inference
    all_preds = np.array(model.apply(
        {'params': params},
        jnp.array(batch_boss),
        jnp.array(batch_hero),
        jnp.array(batch_cont),
    ))  # (N, max_len)

    # Compute credit and scatter back to full episode
    all_credit = np.zeros(T, dtype=np.float32)
    all_predictions = np.zeros(T, dtype=np.float32)

    for i, (start_idx, store_len, _, _, _) in enumerate(segments):
        preds = all_preds[i]  # (max_len,)

        # Credit = diff of predictions
        credit = np.zeros(max_len)
        credit[0] = preds[0]
        credit[1:] = preds[1:] - preds[:-1]

        all_credit[start_idx:start_idx + store_len] = credit[:store_len]
        all_predictions[start_idx:start_idx + store_len] = preds[:store_len]

    return all_credit, all_predictions


def visualize_episode(
    model,
    params,
    episode_path: str,
    stats: dict,
    save_path: str = None,
):
    """Visualize credit assignment for a single episode."""
    data = np.load(episode_path)
    T = len(data['actions'])

    # Check if per-attack mode
    mode = stats.get('mode', 'full_episode')
    max_len = stats.get('max_len', 64)

    if mode == 'per_attack':
        # Segment episode into attacks and compute credit per-attack
        credit, predictions = get_credit_per_attack(model, params, dict(data), stats, max_len)
        credit_denorm = credit * stats['return_std']
        predictions = predictions * stats['return_std'] + stats['return_mean']
    else:
        # Full episode mode
        # Prepare inputs (normalized)
        boss_anim = jnp.array(data['boss_anim_id'], dtype=jnp.int32)
        hero_anim = jnp.array(data['hero_anim_id'], dtype=jnp.int32)

        dist = (data['dist_to_boss'] - stats['dist_mean']) / stats['dist_std']
        hp = (data['hero_hp'] - stats['hp_mean']) / stats['hp_std']
        action = data['actions'].astype(np.float32)

        # Check if old model (with elapsed_frames) or new model (without)
        if 'elapsed_mean' in stats:
            elapsed = (data['elapsed_frames'] - stats['elapsed_mean']) / stats['elapsed_std']
            continuous = np.stack([elapsed, dist, hp, action], axis=-1).astype(np.float32)
        else:
            continuous = np.stack([dist, hp, action], axis=-1).astype(np.float32)
        continuous = jnp.array(continuous)

        # Get credit
        credit = np.array(get_credit(model, params, boss_anim, hero_anim, continuous))

        # Get predictions
        predictions = np.array(model.apply(
            {'params': params},
            boss_anim[None, :],
            hero_anim[None, :],
            continuous[None, :, :],
        )[0])

        # Denormalize
        predictions = predictions * stats['return_std'] + stats['return_mean']
        credit_denorm = credit * stats['return_std']

    # Find events
    damage_idx = np.where(data['damage_taken'] > 0)[0]
    dodge_idx = np.where(data['actions'] == 1)[0]

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Credit assignment
    ax = axes[0]
    colors = [COLOR_POSITIVE if c > 0 else COLOR_NEGATIVE for c in credit_denorm]
    ax.bar(range(T), credit_denorm, color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Credit')
    ax.set_title(f'RUDDER Credit Assignment (Return: {data["total_reward"]:.2f})')
    for idx in damage_idx:
        ax.axvline(idx, color=COLOR_DAMAGE, linestyle='--', alpha=0.5)

    # Predictions
    ax = axes[1]
    ax.plot(predictions, color=COLOR_POSITIVE, linewidth=1)
    ax.axhline(data['total_reward'], color='#009E73', linestyle='--', label='Actual')  # Teal
    ax.set_ylabel('Predicted Return')
    ax.legend()
    for idx in damage_idx:
        ax.axvline(idx, color=COLOR_DAMAGE, linestyle='--', alpha=0.5)

    # Boss animation
    ax = axes[2]
    ax.plot(data['boss_anim_id'], color='#7570B3', alpha=0.7, label='Boss Anim')  # Muted purple
    ax2 = ax.twinx()
    ax2.plot(data['elapsed_frames'], color='#E6AB02', alpha=0.7, label='Elapsed')  # Gold
    ax.set_ylabel('Boss Anim', color='#7570B3')
    ax2.set_ylabel('Elapsed', color='#E6AB02')

    # Actions
    ax = axes[3]
    ax.fill_between(range(T), 0, data['actions'], alpha=0.3, color=COLOR_POSITIVE, label='Dodge')
    ax.scatter(damage_idx, np.ones(len(damage_idx)) * 0.5, c=COLOR_NEGATIVE, s=50, marker='x', label='Damage')
    ax.set_ylabel('Action')
    ax.set_xlabel('Timestep')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_credit(model, params, data_dir: str, stats: dict):
    """Analyze credit at damage events."""
    from rich.console import Console
    console = Console()

    data_path = Path(data_dir)
    episode_files = sorted(data_path.glob("episode_*.npz"))

    all_credit_at_damage = []
    all_credit_before = []
    window = 30

    # Check mode
    mode = stats.get('mode', 'full_episode')
    max_len = stats.get('max_len', 64)

    for ep_file in episode_files:
        data = np.load(ep_file)

        if mode == 'per_attack':
            # Use per-attack credit computation
            credit, _ = get_credit_per_attack(model, params, dict(data), stats, max_len)
            credit = credit * stats['return_std']
        else:
            # Full episode mode
            boss_anim = jnp.array(data['boss_anim_id'], dtype=jnp.int32)
            hero_anim = jnp.array(data['hero_anim_id'], dtype=jnp.int32)

            dist = (data['dist_to_boss'] - stats['dist_mean']) / stats['dist_std']
            hp = (data['hero_hp'] - stats['hp_mean']) / stats['hp_std']
            action = data['actions'].astype(np.float32)

            if 'elapsed_mean' in stats:
                elapsed = (data['elapsed_frames'] - stats['elapsed_mean']) / stats['elapsed_std']
                continuous = np.stack([elapsed, dist, hp, action], axis=-1).astype(np.float32)
            else:
                continuous = np.stack([dist, hp, action], axis=-1).astype(np.float32)

            credit = np.array(get_credit(model, params, boss_anim, hero_anim, jnp.array(continuous)))
            credit = credit * stats['return_std']

        damage_idx = np.where(data['damage_taken'] > 0)[0]
        for idx in damage_idx:
            all_credit_at_damage.append(credit[idx])
            start = max(0, idx - window)
            all_credit_before.append(credit[start:idx].sum() if idx > start else 0)

    console.print("\n[bold]Credit at Damage Events[/bold]")
    console.print(f"  Events: {len(all_credit_at_damage)}")
    console.print(f"  Avg credit at damage: {np.mean(all_credit_at_damage):.4f}")
    console.print(f"  Avg credit before ({window} steps): {np.mean(all_credit_before):.4f}")

    neg = sum(1 for c in all_credit_at_damage if c < 0)
    console.print(f"  Negative at damage: {neg}/{len(all_credit_at_damage)} ({100*neg/len(all_credit_at_damage):.1f}%)")


def compare_with_groundtruth(
    model,
    params,
    data_dir: str,
    stats: dict,
    dodge_window_path: str = "dodge_windows.json",
    save_path: str = None,
):
    """Compare RUDDER credit with ground truth DodgeWindowModel - visual."""
    from rich.console import Console
    console = Console()

    # Load ground truth
    gt_model = DodgeWindowModel.load(dodge_window_path)
    console.print(f"[bold]Comparing RUDDER with ground truth[/bold]")
    console.print(f"  Animations with windows: {list(gt_model.windows.keys())}")

    data_path = Path(data_dir)
    episode_files = sorted(data_path.glob("episode_*.npz"))

    # Collect credit aligned by (anim_idx, elapsed_frames)
    # For each animation with a known window, collect all credits at each elapsed frame
    anim_credits = {}  # anim_idx -> {elapsed: [credits]}

    # Check mode
    mode = stats.get('mode', 'full_episode')
    max_len = stats.get('max_len', 64)

    for ep_file in episode_files:
        data = np.load(ep_file)
        T = len(data['actions'])

        if mode == 'per_attack':
            # Use per-attack credit computation
            credit, _ = get_credit_per_attack(model, params, dict(data), stats, max_len)
            credit = credit * stats['return_std']
        else:
            # Full episode mode
            boss_anim = jnp.array(data['boss_anim_id'], dtype=jnp.int32)
            hero_anim = jnp.array(data['hero_anim_id'], dtype=jnp.int32)

            dist = (data['dist_to_boss'] - stats['dist_mean']) / stats['dist_std']
            hp = (data['hero_hp'] - stats['hp_mean']) / stats['hp_std']
            action = data['actions'].astype(np.float32)

            if 'elapsed_mean' in stats:
                elapsed_norm = (data['elapsed_frames'] - stats['elapsed_mean']) / stats['elapsed_std']
                continuous = np.stack([elapsed_norm, dist, hp, action], axis=-1).astype(np.float32)
            else:
                continuous = np.stack([dist, hp, action], axis=-1).astype(np.float32)

            credit = np.array(get_credit(model, params, boss_anim, hero_anim, jnp.array(continuous)))
            credit = credit * stats['return_std']

        # Still use elapsed_frames for bucketing visualization (not model input)
        for t in range(T):
            raw_anim_id = int(data['boss_anim_id'][t])
            anim_idx = anim_id_to_index(raw_anim_id)
            elapsed = int(data['elapsed_frames'][t])

            if anim_idx not in anim_credits:
                anim_credits[anim_idx] = {}
            if elapsed not in anim_credits[anim_idx]:
                anim_credits[anim_idx][elapsed] = []
            anim_credits[anim_idx][elapsed].append(credit[t])

    # Plot for each animation that has a ground truth window
    anims_with_windows = [idx for idx in gt_model.windows.keys() if idx in anim_credits]
    n_anims = len(anims_with_windows)

    if n_anims == 0:
        console.print("[red]No matching animations found![/red]")
        return

    console.print(f"  Plotting {n_anims} animations...")

    cols = 4
    rows = (n_anims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()

    for i, anim_idx in enumerate(anims_with_windows):
        ax = axes[i]

        # Get credits by elapsed frame
        elapsed_credits = anim_credits[anim_idx]
        frames = sorted(elapsed_credits.keys())
        max_frame = max(frames) if frames else 100

        # Compute mean credit at each frame
        mean_credits = []
        frame_range = range(0, max_frame + 1)
        for f in frame_range:
            if f in elapsed_credits:
                mean_credits.append(np.mean(elapsed_credits[f]))
            else:
                mean_credits.append(0)

        # Normalize to [-1, 1] by dividing by max absolute value
        mean_credits = np.array(mean_credits)
        max_abs = np.abs(mean_credits).max()
        if max_abs > 0:
            mean_credits = mean_credits / max_abs

        # Plot credit (colorblind-friendly: blue=positive, orange=negative)
        colors = [COLOR_POSITIVE if c > 0 else COLOR_NEGATIVE for c in mean_credits]
        ax.bar(frame_range, mean_credits, color=colors, alpha=0.6, width=1.0)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylim(-1.1, 1.1)

        # Overlay ground truth windows (pink/magenta)
        windows = gt_model.windows.get(anim_idx, [])
        for w in windows:
            mean_f = w.dodge_mean_frames
            std_f = w.dodge_std_frames
            # Draw window as shaded region
            ax.axvspan(mean_f - 2*std_f, mean_f + 2*std_f, alpha=0.3, color=COLOR_GT_WINDOW, label='GT Window')
            ax.axvline(mean_f, color=COLOR_GT_LINE, linestyle='--', linewidth=2)

        # Get animation name
        anim_name = windows[0].anim_name if windows else f"idx_{anim_idx}"
        ax.set_title(f"{anim_name}\n(idx={anim_idx})", fontsize=10)
        ax.set_xlabel("Elapsed Frames")
        ax.set_ylabel("RUDDER Credit")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("RUDDER Credit vs Ground Truth Dodge Windows\n(Pink = GT window, Orange bars = negative credit)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"  Saved to {save_path}")
    else:
        plt.show()

    plt.close()

    # Compute summary: is credit more negative inside windows?
    in_window_credits = []
    out_window_credits = []

    for anim_idx in anims_with_windows:
        windows = gt_model.windows.get(anim_idx, [])
        elapsed_credits = anim_credits[anim_idx]

        for elapsed, credits in elapsed_credits.items():
            in_any_window = False
            for w in windows:
                if abs(elapsed - w.dodge_mean_frames) <= 2 * w.dodge_std_frames:
                    in_any_window = True
                    break

            if in_any_window:
                in_window_credits.extend(credits)
            else:
                out_window_credits.extend(credits)

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Avg credit IN dodge windows: {np.mean(in_window_credits):.4f}")
    console.print(f"  Avg credit OUTSIDE windows: {np.mean(out_window_credits):.4f}")

    if np.mean(in_window_credits) < np.mean(out_window_credits):
        console.print("  [green]✓ RUDDER assigns MORE NEGATIVE credit in dodge windows (correct!)[/green]")
    else:
        console.print("  [red]✗ RUDDER does not distinguish dodge windows[/red]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rudder_model")
    parser.add_argument("--data-dir", default="rudder_data")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare with ground truth DodgeWindowModel")
    parser.add_argument("--dodge-windows", default="dodge_windows.json", help="Path to ground truth model")

    args = parser.parse_args()

    model, params, config, stats = load_model(args.model)

    if args.compare:
        compare_with_groundtruth(model, params, args.data_dir, stats, args.dodge_windows, args.output)
    elif args.analyze:
        analyze_credit(model, params, args.data_dir, stats)
    else:
        ep_path = Path(args.data_dir) / f"episode_{args.episode:04d}.npz"
        visualize_episode(model, params, str(ep_path), stats, args.output)


if __name__ == "__main__":
    main()
