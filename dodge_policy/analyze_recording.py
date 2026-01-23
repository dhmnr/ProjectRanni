"""Analyze recorded episode data for dodge timing patterns."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Set

# Load anim vocab for names
import json
_VOCAB_PATH = Path(__file__).parent / "anim_vocab.json"
with open(_VOCAB_PATH) as f:
    _ANIM_VOCAB_DATA = json.load(f)
# Reverse lookup: index -> anim_id
IDX_TO_ANIM = {v: k for k, v in _ANIM_VOCAB_DATA["vocab"].items()}

# Common player dodge animation ID patterns (Elden Ring)
# These are approximate - actual IDs may vary
DODGE_ANIM_PATTERNS = [
    (60000, 70000),   # Roll animations
    (61000, 62000),   # Backstep
]


def is_dodge_anim(hero_anim_id: int) -> bool:
    """Check if player animation ID is a dodge/roll."""
    for low, high in DODGE_ANIM_PATTERNS:
        if low <= hero_anim_id < high:
            return True
    return False


def load_recording(filepath: str) -> dict:
    """Load a recording file."""
    data = np.load(filepath)
    return {k: data[k] for k in data.files}


def load_all_recordings(directory: str) -> List[dict]:
    """Load all recordings from a directory."""
    recordings = []
    for f in Path(directory).glob("*.npz"):
        print(f"Loading {f.name}...")
        recordings.append(load_recording(str(f)))
    return recordings


def find_dodge_onsets(recordings: List[dict]) -> List[dict]:
    """Find frames where player starts a dodge.

    Returns:
        List of dicts with dodge onset info:
        - frame: frame index
        - boss_anim_idx: boss animation at dodge time
        - boss_elapsed: boss animation elapsed frames
        - boss_sinusoidal: sinusoidal encoding at dodge
        - hero_anim_id: player dodge animation ID
    """
    dodge_onsets = []

    for rec in recordings:
        hero_anim = rec.get("hero_anim_id")
        if hero_anim is None:
            print("Warning: Recording missing hero_anim_id, skipping dodge detection")
            continue

        boss_anim_idx = rec["anim_idx"]
        boss_elapsed = rec["elapsed_frames"]
        boss_sinusoidal = rec["elapsed_sinusoidal"]

        prev_hero_anim = None
        for i, h_anim in enumerate(hero_anim):
            h_anim = int(h_anim)
            # Detect onset: transition TO dodge animation
            if is_dodge_anim(h_anim) and (prev_hero_anim is None or not is_dodge_anim(prev_hero_anim)):
                dodge_onsets.append({
                    'frame': i,
                    'boss_anim_idx': int(boss_anim_idx[i]),
                    'boss_elapsed': float(boss_elapsed[i]),
                    'boss_sinusoidal': boss_sinusoidal[i],
                    'hero_anim_id': h_anim,
                })
            prev_hero_anim = h_anim

    return dodge_onsets


def analyze_dodge_timing(dodge_onsets: List[dict]) -> Dict[int, List[float]]:
    """Analyze when dodges occur for each boss animation.

    Returns:
        Dict mapping boss_anim_idx -> list of elapsed_frames when dodge started
    """
    dodge_timing = defaultdict(list)
    for onset in dodge_onsets:
        dodge_timing[onset['boss_anim_idx']].append(onset['boss_elapsed'])
    return dodge_timing


def analyze_hit_timing(recordings: List[dict]) -> Dict[int, List[float]]:
    """Analyze when hits occur for each animation.

    Returns:
        Dict mapping anim_idx -> list of elapsed_frames when hit occurred
    """
    hit_timing = defaultdict(list)

    for rec in recordings:
        anim_idx = rec["anim_idx"]
        elapsed = rec["elapsed_frames"]
        player_damage = rec["player_damage"]

        # Find frames where player took damage
        hit_frames = np.where(player_damage > 0)[0]

        for frame in hit_frames:
            anim = int(anim_idx[frame])
            elapsed_at_hit = float(elapsed[frame])
            hit_timing[anim].append(elapsed_at_hit)

    return hit_timing


def print_hero_anim_summary(recordings: List[dict]):
    """Print summary of player animations seen (to identify dodge IDs)."""
    all_hero_anims = Counter()

    for rec in recordings:
        hero_anim = rec.get("hero_anim_id")
        if hero_anim is None:
            continue
        all_hero_anims.update(hero_anim.astype(int))

    print("\n" + "=" * 50)
    print("PLAYER ANIMATION IDS (to identify dodge animations)")
    print("=" * 50)
    print(f"{'Anim ID':<12} {'Count':>8} {'Is Dodge?':>10}")
    print("-" * 50)

    for anim_id, count in sorted(all_hero_anims.items(), key=lambda x: -x[1])[:30]:
        is_dodge = "YES" if is_dodge_anim(anim_id) else ""
        print(f"{anim_id:<12} {count:>8} {is_dodge:>10}")

    print("\nIf dodge animations are misidentified, update DODGE_ANIM_PATTERNS in analyze_recording.py")


def analyze_animation_durations(recordings: List[dict]) -> Dict[int, List[float]]:
    """Analyze how long each animation typically lasts.

    Returns:
        Dict mapping anim_idx -> list of durations (in frames)
    """
    durations = defaultdict(list)

    for rec in recordings:
        anim_idx = rec["anim_idx"]
        elapsed = rec["elapsed_frames"]

        # Find animation transitions
        prev_anim = None
        start_frame = 0

        for i, (anim, el) in enumerate(zip(anim_idx, elapsed)):
            anim = int(anim)
            if anim != prev_anim:
                if prev_anim is not None and prev_anim != 0:
                    # Animation ended, record duration
                    duration = i - start_frame
                    durations[prev_anim].append(duration)
                prev_anim = anim
                start_frame = i

    return durations


def plot_hit_timing_histogram(hit_timing: Dict[int, List[float]], output_path: str = None):
    """Plot histogram of hit timing for each animation."""
    # Filter to animations with enough hits
    min_hits = 3
    anims_with_hits = {k: v for k, v in hit_timing.items() if len(v) >= min_hits}

    if not anims_with_hits:
        print("Not enough hit data to plot histograms")
        return

    n_anims = len(anims_with_hits)
    cols = min(4, n_anims)
    rows = (n_anims + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_anims == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (anim_idx, timings) in enumerate(sorted(anims_with_hits.items())):
        ax = axes[idx]
        anim_name = IDX_TO_ANIM.get(anim_idx, f"idx_{anim_idx}")

        ax.hist(timings, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(timings), color='red', linestyle='--',
                   label=f'mean={np.mean(timings):.1f}')
        ax.axvline(np.median(timings), color='green', linestyle=':',
                   label=f'median={np.median(timings):.1f}')
        ax.set_xlabel('Elapsed Frames at Hit')
        ax.set_ylabel('Count')
        ax.set_title(f'Anim {anim_name}\n(n={len(timings)}, std={np.std(timings):.1f})')
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_anims, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved histogram to {output_path}")
    else:
        plt.show()


def plot_sinusoidal_at_events(
    event_data: Dict[int, List[dict]],
    event_name: str,
    output_path: str = None
):
    """Plot sinusoidal encoding values at event frames for each animation.

    Args:
        event_data: Dict mapping anim_idx -> list of {'elapsed', 'sinusoidal'}
        event_name: Name of event (e.g., "Hits", "Dodges")
        output_path: Path to save plot
    """
    # Filter to animations with enough events
    min_events = 3
    anims_with_events = {k: v for k, v in event_data.items() if len(v) >= min_events}

    if not anims_with_events:
        print(f"Not enough {event_name.lower()} data to plot sinusoidal patterns")
        return

    n_anims = len(anims_with_events)
    fig, axes = plt.subplots(n_anims, 1, figsize=(12, 3 * n_anims))
    if n_anims == 1:
        axes = [axes]

    for idx, (anim_idx, events) in enumerate(sorted(anims_with_events.items())):
        ax = axes[idx]
        anim_name = IDX_TO_ANIM.get(anim_idx, f"idx_{anim_idx}")

        # Stack sinusoidal vectors
        sin_matrix = np.array([e['sinusoidal'] for e in events])  # [n_events, 16]
        elapsed_vals = [e['elapsed'] for e in events]

        # Plot each event's sinusoidal as a line
        for i, (sin_vec, el) in enumerate(zip(sin_matrix, elapsed_vals)):
            ax.plot(sin_vec, alpha=0.5, label=f'el={el:.0f}' if i < 5 else None)

        # Plot mean
        ax.plot(sin_matrix.mean(axis=0), 'k-', linewidth=2, label='mean')

        ax.set_xlabel('Sinusoidal Dimension')
        ax.set_ylabel('Value')
        ax.set_title(f'Sinusoidal at {event_name} - Anim {anim_name} (n={len(events)}, std={np.std(elapsed_vals):.1f})')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xticks(range(16))
        ax.set_xticklabels([f's{i}' if i < 8 else f'c{i-8}' for i in range(16)])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved sinusoidal {event_name.lower()} plot to {output_path}")
    else:
        plt.show()


def collect_event_sinusoidal(recordings: List[dict], event_type: str = "dodge") -> Dict[int, List[dict]]:
    """Collect sinusoidal encoding at event frames.

    Args:
        recordings: List of recording dicts
        event_type: "dodge" or "hit"

    Returns:
        Dict mapping boss_anim_idx -> list of {'elapsed', 'sinusoidal'}
    """
    event_data = defaultdict(list)

    for rec in recordings:
        anim_idx = rec["anim_idx"]
        elapsed = rec["elapsed_frames"]
        sinusoidal = rec["elapsed_sinusoidal"]

        if event_type == "hit":
            player_damage = rec["player_damage"]
            event_frames = np.where(player_damage > 0)[0]
        elif event_type == "dodge":
            hero_anim = rec.get("hero_anim_id")
            if hero_anim is None:
                continue
            # Find dodge onset frames
            event_frames = []
            prev_hero = None
            for i, h in enumerate(hero_anim):
                h = int(h)
                if is_dodge_anim(h) and (prev_hero is None or not is_dodge_anim(prev_hero)):
                    event_frames.append(i)
                prev_hero = h
        else:
            raise ValueError(f"Unknown event_type: {event_type}")

        for frame in event_frames:
            anim = int(anim_idx[frame])
            event_data[anim].append({
                'elapsed': float(elapsed[frame]),
                'sinusoidal': sinusoidal[frame],
            })

    return event_data


def plot_timing_consistency(hit_timing: Dict[int, List[float]], output_path: str = None):
    """Plot timing consistency (std/mean) for each animation."""
    stats = []
    for anim_idx, timings in hit_timing.items():
        if len(timings) >= 2:
            anim_name = IDX_TO_ANIM.get(anim_idx, f"idx_{anim_idx}")
            mean = np.mean(timings)
            std = np.std(timings)
            cv = std / mean if mean > 0 else 0  # Coefficient of variation
            stats.append({
                'anim': anim_name,
                'anim_idx': anim_idx,
                'mean': mean,
                'std': std,
                'cv': cv,
                'n': len(timings),
            })

    if not stats:
        print("Not enough data for timing consistency plot")
        return

    # Sort by coefficient of variation (lower = more consistent)
    stats.sort(key=lambda x: x['cv'])

    fig, ax = plt.subplots(figsize=(12, 6))

    anims = [s['anim'] for s in stats]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]

    x = np.arange(len(anims))
    ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(anims, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Animation')
    ax.set_ylabel('Elapsed Frames at Hit (mean ± std)')
    ax.set_title('Hit Timing Consistency by Animation\n(sorted by coefficient of variation)')

    # Add CV annotation
    for i, s in enumerate(stats):
        ax.annotate(f'CV={s["cv"]:.2f}\nn={s["n"]}',
                    xy=(i, s['mean'] + s['std']),
                    ha='center', fontsize=7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved consistency plot to {output_path}")
    else:
        plt.show()


def print_timing_summary(
    timing_data: Dict[int, List[float]],
    durations: Dict[int, List[float]],
    event_name: str = "Event"
):
    """Print summary statistics for timing data."""
    print("\n" + "=" * 70)
    print(f"{event_name.upper()} TIMING ANALYSIS")
    print("=" * 70)

    print(f"\n{'Anim ID':<12} {'Name':<15} {'Count':>5} {'Mean':>8} {'Std':>8} {'CV':>8} {'Avg Dur':>8}")
    print("-" * 70)

    for anim_idx in sorted(timing_data.keys()):
        timings = timing_data[anim_idx]
        anim_name = IDX_TO_ANIM.get(anim_idx, "???")[:15]

        if len(timings) >= 1:
            mean = np.mean(timings)
            std = np.std(timings) if len(timings) > 1 else 0
            cv = std / mean if mean > 0 else 0

            # Get duration if available
            dur = np.mean(durations.get(anim_idx, [0]))

            print(f"{anim_idx:<12} {anim_name:<15} {len(timings):>5} {mean:>8.1f} {std:>8.1f} {cv:>8.2f} {dur:>8.1f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("- Low CV (< 0.2): Consistent timing - sinusoidal clock should work well")
    print("- High CV (> 0.5): Variable timing - may need other features")
    print("=" * 70)


def plot_sinusoidal_timeline(
    recordings: List[dict],
    target_anim_idx: int,
    num_scales: int = 8,
    output_path: str = None
):
    """Plot sinusoidal encoding over time for a specific boss animation.

    Shows smooth sine/cos waves over the animation duration with dodge onset markers.

    Args:
        recordings: List of recording dicts
        target_anim_idx: Boss animation index to analyze
        num_scales: Number of sinusoidal scales (default 8 = 16 dimensions)
        output_path: Path to save plot
    """
    from .ppo import NORM_ELAPSED

    # Collect all instances of this animation
    animation_instances = []  # List of {'elapsed': [...], 'dodge_frames': [...]}

    for rec in recordings:
        anim_idx = rec["anim_idx"]
        elapsed = rec["elapsed_frames"]
        hero_anim = rec.get("hero_anim_id")

        # Find segments where boss is in target animation
        in_anim = False
        current_instance = {'elapsed': [], 'dodge_frames': [], 'frame_indices': []}
        prev_hero = None

        for i, (anim, el) in enumerate(zip(anim_idx, elapsed)):
            anim = int(anim)

            if anim == target_anim_idx:
                if not in_anim:
                    # Starting new instance
                    in_anim = True
                    current_instance = {'elapsed': [], 'dodge_frames': [], 'frame_indices': []}

                current_instance['elapsed'].append(float(el))
                current_instance['frame_indices'].append(i)

                # Check for dodge onset
                if hero_anim is not None:
                    h = int(hero_anim[i])
                    if is_dodge_anim(h) and (prev_hero is None or not is_dodge_anim(prev_hero)):
                        current_instance['dodge_frames'].append(len(current_instance['elapsed']) - 1)
                    prev_hero = h
            else:
                if in_anim and len(current_instance['elapsed']) > 5:  # Min 5 frames
                    animation_instances.append(current_instance)
                in_anim = False
                prev_hero = None

        # Don't forget last instance
        if in_anim and len(current_instance['elapsed']) > 5:
            animation_instances.append(current_instance)

    if not animation_instances:
        print(f"No instances of animation {target_anim_idx} found")
        return

    anim_name = IDX_TO_ANIM.get(target_anim_idx, f"idx_{target_anim_idx}")
    print(f"\nFound {len(animation_instances)} instances of animation {anim_name}")

    # Find max duration for x-axis
    max_frames = max(len(inst['elapsed']) for inst in animation_instances)
    max_elapsed = max(max(inst['elapsed']) for inst in animation_instances)

    # Create figure: one subplot per frequency scale (show first 4 scales)
    scales_to_show = min(4, num_scales)
    fig, axes = plt.subplots(scales_to_show, 1, figsize=(14, 3 * scales_to_show), sharex=True)
    if scales_to_show == 1:
        axes = [axes]

    # Generate smooth reference sine waves
    t_ref = np.linspace(0, max_elapsed, 200)
    t_norm_ref = t_ref / NORM_ELAPSED

    # Colors for different instances
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(animation_instances))))

    for scale_idx, ax in enumerate(axes):
        freq = 2.0 ** scale_idx

        # Plot reference sine wave (light gray)
        sin_ref = np.sin(t_norm_ref * freq * np.pi)
        ax.plot(t_ref, sin_ref, 'k-', alpha=0.2, linewidth=2, label='sin reference')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.3)

        # Collect all dodge points for this scale
        all_dodge_elapsed = []
        all_dodge_sin_values = []

        # Plot each instance
        for inst_idx, inst in enumerate(animation_instances):
            elapsed_arr = np.array(inst['elapsed'])
            elapsed_norm = elapsed_arr / NORM_ELAPSED
            sin_values = np.sin(elapsed_norm * freq * np.pi)

            color = colors[inst_idx % len(colors)]
            ax.plot(elapsed_arr, sin_values, '-', color=color, alpha=0.6, linewidth=1)

            # Mark dodge onsets
            for dodge_frame in inst['dodge_frames']:
                dodge_elapsed = elapsed_arr[dodge_frame]
                dodge_sin = sin_values[dodge_frame]
                ax.plot(dodge_elapsed, dodge_sin, 'o', color=color, markersize=8,
                       markeredgecolor='black', markeredgewidth=1)
                all_dodge_elapsed.append(dodge_elapsed)
                all_dodge_sin_values.append(dodge_sin)

        # Stats on dodge timing
        if all_dodge_elapsed:
            mean_elapsed = np.mean(all_dodge_elapsed)
            std_elapsed = np.std(all_dodge_elapsed)
            mean_sin = np.mean(all_dodge_sin_values)
            std_sin = np.std(all_dodge_sin_values)

            # Mark mean dodge timing
            ax.axvline(mean_elapsed, color='red', linestyle='--', alpha=0.7,
                      label=f'dodge mean={mean_elapsed:.1f}±{std_elapsed:.1f}')

            ax.set_title(f'Scale {scale_idx} (freq=2^{scale_idx}): '
                        f'sin(dodge)={mean_sin:.2f}±{std_sin:.2f}')
        else:
            ax.set_title(f'Scale {scale_idx} (freq=2^{scale_idx}): no dodges')

        ax.set_ylabel(f'sin(t·{int(freq)}π)')
        ax.set_ylim(-1.3, 1.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Elapsed Frames')

    fig.suptitle(f'Sinusoidal Clock for Animation {anim_name}\n'
                 f'({len(animation_instances)} instances, circles=dodge onset)',
                 fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved timeline plot to {output_path}")
    else:
        plt.show()

    # Print dodge timing summary
    if any(inst['dodge_frames'] for inst in animation_instances):
        all_dodge_elapsed = []
        for inst in animation_instances:
            for df in inst['dodge_frames']:
                all_dodge_elapsed.append(inst['elapsed'][df])

        print(f"\nDodge timing for {anim_name}:")
        print(f"  Count: {len(all_dodge_elapsed)}")
        print(f"  Mean elapsed: {np.mean(all_dodge_elapsed):.1f}")
        print(f"  Std elapsed: {np.std(all_dodge_elapsed):.1f}")
        print(f"  CV: {np.std(all_dodge_elapsed) / np.mean(all_dodge_elapsed):.2f}")
        print(f"  Range: [{min(all_dodge_elapsed):.0f}, {max(all_dodge_elapsed):.0f}]")


def main():
    parser = argparse.ArgumentParser(description="Analyze recorded episodes for dodge timing")
    parser.add_argument(
        "input",
        type=str,
        help="Path to recording file (.npz) or directory of recordings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (if not specified, shows interactively)",
    )
    parser.add_argument(
        "--anim",
        type=int,
        default=None,
        help="Plot sinusoidal timeline for specific boss animation index",
    )
    parser.add_argument(
        "--all-anims",
        action="store_true",
        help="Plot sinusoidal timeline for all animations with dodges",
    )

    args = parser.parse_args()

    # Load recordings
    input_path = Path(args.input)
    if input_path.is_file():
        recordings = [load_recording(str(input_path))]
    elif input_path.is_dir():
        recordings = load_all_recordings(str(input_path))
    else:
        print(f"Error: {args.input} not found")
        return

    if not recordings:
        print("No recordings found")
        return

    print(f"\nLoaded {len(recordings)} recording(s)")
    total_frames = sum(rec['num_steps'] for rec in recordings)
    print(f"Total frames: {total_frames} ({total_frames / 15:.1f} seconds)")

    # Check for hero_anim_id
    has_hero_anim = any("hero_anim_id" in rec for rec in recordings)
    if has_hero_anim:
        print_hero_anim_summary(recordings)

    # Analyze hit timing
    hit_timing = analyze_hit_timing(recordings)
    durations = analyze_animation_durations(recordings)

    total_hits = sum(len(v) for v in hit_timing.values())
    print(f"\nTotal hits: {total_hits}")
    print(f"Animations with hits: {len(hit_timing)}")

    # Analyze dodge timing
    if has_hero_anim:
        dodge_onsets = find_dodge_onsets(recordings)
        dodge_timing = analyze_dodge_timing(dodge_onsets)
        total_dodges = len(dodge_onsets)
        print(f"Total dodges detected: {total_dodges}")
        print(f"Animations with dodges: {len(dodge_timing)}")
    else:
        dodge_timing = {}
        print("\nNo hero_anim_id in recordings - cannot detect dodges")
        print("Re-record with updated record_episode.py to capture player animations")

    # Print summaries
    if hit_timing:
        print_timing_summary(hit_timing, durations, "Hit")

    if dodge_timing:
        print_timing_summary(dodge_timing, durations, "Dodge")

    # Plot
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Hit timing plots
    if hit_timing:
        plot_hit_timing_histogram(
            hit_timing,
            str(output_dir / "hit_timing_histogram.png") if output_dir else None
        )

        hit_event_data = collect_event_sinusoidal(recordings, "hit")
        plot_sinusoidal_at_events(
            hit_event_data,
            "Hits",
            str(output_dir / "sinusoidal_at_hits.png") if output_dir else None
        )

        plot_timing_consistency(
            hit_timing,
            str(output_dir / "hit_timing_consistency.png") if output_dir else None
        )

    # Dodge timing plots
    if dodge_timing:
        plot_hit_timing_histogram(
            dodge_timing,
            str(output_dir / "dodge_timing_histogram.png") if output_dir else None
        )

        dodge_event_data = collect_event_sinusoidal(recordings, "dodge")
        plot_sinusoidal_at_events(
            dodge_event_data,
            "Dodges",
            str(output_dir / "sinusoidal_at_dodges.png") if output_dir else None
        )

        plot_timing_consistency(
            dodge_timing,
            str(output_dir / "dodge_timing_consistency.png") if output_dir else None
        )

    # Sinusoidal timeline for specific animation
    if args.anim is not None:
        plot_sinusoidal_timeline(
            recordings,
            args.anim,
            output_path=str(output_dir / f"sinusoidal_timeline_anim{args.anim}.png") if output_dir else None
        )

    # Sinusoidal timeline for all animations with dodges
    if args.all_anims and dodge_timing:
        for anim_idx in sorted(dodge_timing.keys()):
            if len(dodge_timing[anim_idx]) >= 2:  # At least 2 dodges
                plot_sinusoidal_timeline(
                    recordings,
                    anim_idx,
                    output_path=str(output_dir / f"sinusoidal_timeline_anim{anim_idx}.png") if output_dir else None
                )


if __name__ == "__main__":
    main()
