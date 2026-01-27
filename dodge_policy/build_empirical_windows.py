"""Build empirical dodge windows from recorded episodes.

Analyzes dodge attempts and their outcomes to find safe dodge timings.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Animation vocab for readable names
ANIM_VOCAB_PATH = Path(__file__).parent / "anim_vocab.json"


@dataclass
class DodgeAttempt:
    """A single dodge attempt."""
    boss_anim_id: int
    elapsed_frames: int
    got_hit: bool  # Did we get hit within the hit window?
    damage: float


def load_anim_names() -> Dict[int, str]:
    """Load animation ID to name mapping."""
    with open(ANIM_VOCAB_PATH) as f:
        data = json.load(f)
    # Reverse: index -> name
    return {int(k): v for k, v in data.get("id_to_name", {}).items()}


def analyze_episode(filepath: Path, hit_window: int = 30) -> List[DodgeAttempt]:
    """Analyze a single episode for dodge attempts and outcomes.

    Args:
        filepath: Path to episode .npz file
        hit_window: Frames after dodge to check for hits

    Returns:
        List of DodgeAttempt records
    """
    data = np.load(filepath)

    boss_anim_id = data['boss_anim_id']
    elapsed_frames = data['elapsed_frames']
    actions = data['actions']
    damage_taken = data['damage_taken']

    T = len(actions)
    attempts = []

    for t in range(T):
        if actions[t] == 1:  # Dodge action
            # Check if hit within window
            end = min(t + hit_window, T)
            damage_in_window = damage_taken[t:end].sum()
            got_hit = damage_in_window > 0

            attempts.append(DodgeAttempt(
                boss_anim_id=int(boss_anim_id[t]),
                elapsed_frames=int(elapsed_frames[t]),
                got_hit=got_hit,
                damage=float(damage_in_window),
            ))

    return attempts


def build_windows(
    data_dir: str = "rudder_data",
    hit_window: int = 30,
    bin_size: int = 5,
    min_samples: int = 3,
) -> Dict:
    """Build empirical dodge windows from episode data.

    Args:
        data_dir: Directory with episode_*.npz files
        hit_window: Frames after dodge to check for hits
        bin_size: Group elapsed_frames into bins of this size
        min_samples: Minimum samples needed to consider a bin valid

    Returns:
        Dictionary with window data
    """
    data_path = Path(data_dir)
    episode_files = sorted(data_path.glob("episode_*.npz"))

    print(f"Analyzing {len(episode_files)} episodes...")

    # Collect all dodge attempts
    all_attempts = []
    for ep_file in episode_files:
        attempts = analyze_episode(ep_file, hit_window)
        all_attempts.extend(attempts)

    print(f"Found {len(all_attempts)} total dodge attempts")

    # Group by (anim_id, elapsed_bin)
    # stats[anim_id][elapsed_bin] = {'hits': n, 'total': m}
    stats = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0, 'damage': 0}))

    for attempt in all_attempts:
        elapsed_bin = (attempt.elapsed_frames // bin_size) * bin_size
        stats[attempt.boss_anim_id][elapsed_bin]['total'] += 1
        if attempt.got_hit:
            stats[attempt.boss_anim_id][elapsed_bin]['hits'] += 1
            stats[attempt.boss_anim_id][elapsed_bin]['damage'] += attempt.damage

    # Load animation names
    anim_names = load_anim_names()

    # Build windows: find bins with zero hit rate
    windows = {}

    print("\n" + "="*80)
    print("EMPIRICAL DODGE WINDOWS")
    print("="*80)

    for anim_id in sorted(stats.keys()):
        anim_name = anim_names.get(anim_id, f"anim_{anim_id}")
        bins = stats[anim_id]

        # Find safe bins (hit rate = 0)
        safe_bins = []
        unsafe_bins = []

        for elapsed_bin in sorted(bins.keys()):
            s = bins[elapsed_bin]
            if s['total'] >= min_samples:
                hit_rate = s['hits'] / s['total']
                if hit_rate == 0:
                    safe_bins.append(elapsed_bin)
                else:
                    unsafe_bins.append((elapsed_bin, hit_rate, s['total']))

        if safe_bins or unsafe_bins:
            print(f"\n{anim_name} ({anim_id}):")

            if safe_bins:
                # Compute window from safe bins
                min_frame = min(safe_bins)
                max_frame = max(safe_bins) + bin_size
                mean_frame = (min_frame + max_frame) / 2
                std_frame = (max_frame - min_frame) / 4  # ~2 std covers range

                print(f"  SAFE: frames {min_frame}-{max_frame} (mean={mean_frame:.0f}, std={std_frame:.1f})")
                for b in safe_bins:
                    s = bins[b]
                    print(f"    [{b:3d}-{b+bin_size:3d}]: 0/{s['total']} hits")

                windows[anim_id] = {
                    'anim_name': anim_name,
                    'safe_bins': safe_bins,
                    'min_frame': min_frame,
                    'max_frame': max_frame,
                    'mean_frame': mean_frame,
                    'std_frame': max(std_frame, 3.0),  # Minimum std
                }

            if unsafe_bins:
                print(f"  UNSAFE:")
                for (b, hit_rate, total) in unsafe_bins:
                    print(f"    [{b:3d}-{b+bin_size:3d}]: {hit_rate*100:.0f}% hit rate ({int(hit_rate*total)}/{total})")

    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY: Found {len(windows)} animations with safe dodge windows")
    print("="*80)

    return {
        'windows': windows,
        'stats': {str(k): dict(v) for k, v in stats.items()},
        'config': {
            'hit_window': hit_window,
            'bin_size': bin_size,
            'min_samples': min_samples,
            'num_episodes': len(episode_files),
            'total_attempts': len(all_attempts),
        }
    }


def export_to_dodge_windows_format(empirical: Dict, output_path: str = "dodge_windows_empirical.json"):
    """Export to format compatible with DodgeWindowModel."""
    windows = empirical['windows']

    output = {}
    for anim_id, w in windows.items():
        output[str(anim_id)] = [{
            'anim_id': anim_id,
            'anim_name': w['anim_name'],
            'dodge_mean_frames': w['mean_frame'],
            'dodge_std_frames': w['std_frame'],
            'count': len(w['safe_bins']),
        }]

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build empirical dodge windows")
    parser.add_argument("--data-dir", default="rudder_data")
    parser.add_argument("--hit-window", type=int, default=30, help="Frames after dodge to check for hits")
    parser.add_argument("--bin-size", type=int, default=5, help="Elapsed frame bin size")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum samples per bin")
    parser.add_argument("--output", default="dodge_windows_empirical.json")

    args = parser.parse_args()

    result = build_windows(
        data_dir=args.data_dir,
        hit_window=args.hit_window,
        bin_size=args.bin_size,
        min_samples=args.min_samples,
    )

    export_to_dodge_windows_format(result, args.output)
