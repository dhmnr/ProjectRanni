"""Build dodge windows from expert gameplay recordings.

Analyzes when the expert dodges relative to boss animations,
and when attacks land if they don't dodge.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

ANIM_VOCAB_PATH = Path(__file__).parent / "anim_vocab.json"

# Dodge animation range
DODGE_ANIM_MIN = 27000
DODGE_ANIM_MAX = 28000


def load_anim_vocab():
    """Load animation vocab for index -> anim_id mapping."""
    with open(ANIM_VOCAB_PATH) as f:
        data = json.load(f)
    # vocab: anim_id_str -> index
    # reverse it: index -> anim_id
    vocab = data.get('vocab', {})
    idx_to_anim = {v: int(k) for k, v in vocab.items() if k.isdigit()}
    return idx_to_anim


def analyze_recording(filepath: Path, hit_check_window: int = 30) -> Tuple[List, List, List]:
    """Analyze a single recording for dodge timing and hit timing.

    Args:
        filepath: Path to recording .npz file
        hit_check_window: Frames after dodge to check for hits (to validate success)

    Returns:
        successful_dodges: List of (anim_idx, elapsed_frames) for successful dodges (no hit after)
        failed_dodges: List of (anim_idx, elapsed_frames) for failed dodges (got hit after)
        hit_events: List of (anim_idx, elapsed_frames) when expert got hit without dodging
    """
    data = np.load(filepath)

    anim_idx = data['anim_idx']
    elapsed_frames = data['elapsed_frames']
    hero_anim_id = data['hero_anim_id']
    player_damage = data['player_damage']

    T = len(anim_idx)

    # First pass: collect all dodge onsets and hit times
    dodge_times = []  # (t, anim_idx, elapsed_frames)
    hit_times = set()  # timesteps where damage occurred

    prev_hero_anim = 0

    for t in range(T):
        curr_anim = int(hero_anim_id[t])
        is_dodging = DODGE_ANIM_MIN <= curr_anim <= DODGE_ANIM_MAX
        was_dodging = DODGE_ANIM_MIN <= prev_hero_anim <= DODGE_ANIM_MAX

        if is_dodging and not was_dodging:
            dodge_times.append((t, int(anim_idx[t]), int(elapsed_frames[t])))

        if player_damage[t] > 0:
            hit_times.add(t)

        prev_hero_anim = curr_anim

    # Second pass: classify dodges as successful or failed
    successful_dodges = []
    failed_dodges = []

    for t, a_idx, elapsed in dodge_times:
        # Check if hit occurred within window after dodge
        got_hit = any(
            hit_t in hit_times
            for hit_t in range(t, min(t + hit_check_window, T))
        )
        if got_hit:
            failed_dodges.append((a_idx, elapsed))
        else:
            successful_dodges.append((a_idx, elapsed))

    # Hits that occurred without a recent dodge are "no-dodge hits"
    hit_events = []
    for t in sorted(hit_times):
        # Check if there was a dodge within the window before this hit
        had_recent_dodge = any(
            dodge_t <= t < dodge_t + hit_check_window
            for dodge_t, _, _ in dodge_times
        )
        if not had_recent_dodge:
            hit_events.append((int(anim_idx[t]), int(elapsed_frames[t])))

    return successful_dodges, failed_dodges, hit_events


def build_windows(data_dir: str = "recordings", lead_time: int = 8, min_samples: int = 3):
    """Build dodge windows from expert recordings.

    Only uses SUCCESSFUL dodges (where no hit followed).
    Falls back to hit timing only for animations with no successful dodge data.
    """
    data_path = Path(data_dir)
    recordings = sorted(data_path.glob("*.npz"))

    if not recordings:
        print(f"No recordings found in {data_dir}")
        return {}

    print(f"Analyzing {len(recordings)} recordings...")

    # Collect all events
    all_successful_dodges = defaultdict(list)  # anim_idx -> list of elapsed_frames
    all_failed_dodges = defaultdict(list)      # anim_idx -> list of elapsed_frames
    all_hits = defaultdict(list)               # anim_idx -> list of elapsed_frames (no-dodge hits)

    for rec_path in recordings:
        successful, failed, hits = analyze_recording(rec_path)

        for anim_idx, elapsed in successful:
            all_successful_dodges[anim_idx].append(elapsed)

        for anim_idx, elapsed in failed:
            all_failed_dodges[anim_idx].append(elapsed)

        for anim_idx, elapsed in hits:
            all_hits[anim_idx].append(elapsed)

    # Load vocab for anim_id lookup
    idx_to_anim = load_anim_vocab()

    total_successful = sum(len(v) for v in all_successful_dodges.values())
    total_failed = sum(len(v) for v in all_failed_dodges.values())
    total_hits = sum(len(v) for v in all_hits.values())

    print(f"\nSuccessful dodges: {total_successful} across {len(all_successful_dodges)} animations")
    print(f"Failed dodges: {total_failed} across {len(all_failed_dodges)} animations")
    print(f"No-dodge hits: {total_hits} across {len(all_hits)} animations")

    # Build windows
    windows = {}

    print("\n" + "="*70)
    print("DODGE WINDOWS FROM SUCCESSFUL EXPERT DODGES")
    print("="*70)

    # Combine info from all sources
    all_anims = set(all_successful_dodges.keys()) | set(all_failed_dodges.keys()) | set(all_hits.keys())

    for anim_idx in sorted(all_anims):
        dodges = all_successful_dodges.get(anim_idx, [])
        failed = all_failed_dodges.get(anim_idx, [])
        hits = all_hits.get(anim_idx, [])

        anim_id = idx_to_anim.get(anim_idx, 0)
        anim_name = f"anim_{anim_id}"

        print(f"\n{anim_name} (idx={anim_idx}):")

        # Priority: successful dodges > failed dodges > hits
        if len(dodges) > 0:
            dodge_mean = np.mean(dodges)
            dodge_std = max(np.std(dodges), 2.0) if len(dodges) > 1 else 5.0

            print(f"  Successful dodges: {len(dodges)} samples")
            if failed:
                print(f"  Failed dodges: {len(failed)} (excluded)")
            print(f"  Dodge timing: {dodge_mean:.1f} +/- {dodge_std:.1f}")

            windows[anim_idx] = {
                'anim_idx': anim_idx,
                'anim_id': anim_id,
                'anim_name': anim_name,
                'dodge_mean_frames': float(dodge_mean),
                'dodge_std_frames': float(dodge_std),
                'source': 'successful_dodges',
                'n_samples': len(dodges),
                'n_failed': len(failed),
            }

        # Use failed dodges if no successful ones (at least we tried)
        elif len(failed) > 0:
            dodge_mean = np.mean(failed)
            dodge_std = max(np.std(failed), 2.0) if len(failed) > 1 else 5.0

            print(f"  No successful dodges")
            print(f"  Failed dodges: {len(failed)} samples (using anyway)")
            print(f"  Dodge timing: {dodge_mean:.1f} +/- {dodge_std:.1f}")

            windows[anim_idx] = {
                'anim_idx': anim_idx,
                'anim_id': anim_id,
                'anim_name': anim_name,
                'dodge_mean_frames': float(dodge_mean),
                'dodge_std_frames': float(dodge_std),
                'source': 'failed_dodges',
                'n_samples': len(failed),
                'n_failed': len(failed),
            }

        # Fall back to hit timing
        elif len(hits) > 0:
            hit_mean = np.mean(hits)
            hit_std = max(np.std(hits), 2.0) if len(hits) > 1 else 5.0
            dodge_mean = max(0, hit_mean - lead_time)

            print(f"  No dodge data")
            print(f"  No-dodge hits: {len(hits)} samples")
            print(f"  Hit timing: {hit_mean:.1f} +/- {hit_std:.1f}")
            print(f"  Inferred dodge: {dodge_mean:.1f} +/- {hit_std:.1f}")

            windows[anim_idx] = {
                'anim_idx': anim_idx,
                'anim_id': anim_id,
                'anim_name': anim_name,
                'dodge_mean_frames': float(dodge_mean),
                'dodge_std_frames': float(hit_std),
                'source': 'inferred_from_hits',
                'n_samples': len(hits),
                'n_failed': len(failed),
            }

        else:
            print(f"  No data at all")

    return windows


def export_windows(windows: dict, output_path: str):
    """Export to DodgeWindowModel format."""
    with open(ANIM_VOCAB_PATH) as f:
        vocab_data = json.load(f)

    # Build anim_vocab: index -> anim_id string
    anim_vocab = {}
    for anim_id_str, idx in vocab_data.get('vocab', {}).items():
        anim_vocab[str(idx)] = anim_id_str

    # Build windows dict
    max_frames = 120.0
    windows_out = {}

    for anim_idx, w in windows.items():
        windows_out[str(anim_idx)] = [{
            'anim_idx': anim_idx,
            'anim_name': w['anim_name'],
            'window_idx': 0,
            'dodge_mean_frames': w['dodge_mean_frames'],
            'dodge_std_frames': w['dodge_std_frames'],
            'dodge_mean_norm': w['dodge_mean_frames'] / max_frames,
            'dodge_std_norm': w['dodge_std_frames'] / max_frames,
            'duration_mean': max_frames,
            'duration_std': 10.0,
            'n_dodges': w['n_samples'],
            'n_instances': w['n_samples'],
        }]

    output = {
        'windows': windows_out,
        'anim_vocab': anim_vocab,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExported {len(windows_out)} windows to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dodge windows from expert recordings")
    parser.add_argument("--data-dir", default="recordings")
    parser.add_argument("--lead-time", type=int, default=8, help="Frames before hit to dodge")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--output", default="dodge_windows_expert.json")

    args = parser.parse_args()

    windows = build_windows(args.data_dir, args.lead_time, args.min_samples)

    if windows:
        export_windows(windows, args.output)
    else:
        print("No windows built - need more data")
