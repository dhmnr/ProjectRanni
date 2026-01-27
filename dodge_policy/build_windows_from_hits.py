"""Build dodge windows from hit timing data.

The optimal dodge timing is BEFORE the attack lands, so i-frames cover the hit.
- I-frames in Elden Ring: ~13 frames at 60fps
- Dodge startup: ~3-4 frames
- So dodge at (hit_frame - 10) to (hit_frame - 5) roughly
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


ANIM_VOCAB_PATH = Path(__file__).parent / "anim_vocab.json"
IFRAMES_DURATION = 13  # i-frames last ~13 frames
DODGE_STARTUP = 4      # frames before i-frames activate
DODGE_LEAD_TIME = 8    # how many frames before hit to start dodge


def load_anim_names():
    with open(ANIM_VOCAB_PATH) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.get("id_to_name", {}).items()}


def analyze_hit_timing(data_dir: str = "rudder_data"):
    """Find when each animation's attack actually connects."""
    hit_frames = defaultdict(list)

    data_path = Path(data_dir)
    for ep_file in sorted(data_path.glob("episode_*.npz")):
        data = np.load(ep_file)
        boss_anim = data['boss_anim_id']
        elapsed = data['elapsed_frames']
        damage = data['damage_taken']

        for t in range(len(damage)):
            if damage[t] > 0:
                hit_frames[int(boss_anim[t])].append(int(elapsed[t]))

    return hit_frames


def build_windows_from_hits(hit_frames: dict, min_hits: int = 5, lead_time: int = 8):
    """Build dodge windows from hit timing.

    Dodge window = hit_timing - lead_time
    """
    anim_names = load_anim_names()
    windows = {}

    print("DODGE WINDOWS FROM HIT TIMING")
    print("=" * 70)
    print(f"Lead time: {lead_time} frames before hit")
    print("=" * 70)

    for anim_id in sorted(hit_frames.keys()):
        frames = np.array(hit_frames[anim_id])
        if len(frames) < min_hits:
            continue

        anim_name = anim_names.get(anim_id, f"anim_{anim_id}")

        # Calculate hit statistics
        hit_mean = frames.mean()
        hit_std = max(frames.std(), 2.0)  # minimum std

        # Dodge window = before the hit
        dodge_mean = max(0, hit_mean - lead_time)
        dodge_std = hit_std

        # Window bounds
        window_start = max(0, int(dodge_mean - 2 * dodge_std))
        window_end = int(dodge_mean + 2 * dodge_std)

        print(f"\n{anim_name} ({anim_id}):")
        print(f"  Hit timing: {hit_mean:.1f} +/- {hit_std:.1f} (n={len(frames)})")
        print(f"  Dodge window: {dodge_mean:.1f} +/- {dodge_std:.1f}")
        print(f"  Safe range: frames {window_start}-{window_end}")

        windows[anim_id] = {
            'anim_id': anim_id,
            'anim_name': anim_name,
            'hit_mean': float(hit_mean),
            'hit_std': float(hit_std),
            'dodge_mean_frames': float(dodge_mean),
            'dodge_std_frames': float(dodge_std),
            'count': len(frames),
        }

    return windows


def export_windows(windows: dict, output_path: str):
    """Export to DodgeWindowModel format."""
    # Load anim vocab for the model
    with open(ANIM_VOCAB_PATH) as f:
        vocab_data = json.load(f)

    # Build anim_vocab: index -> anim_id string
    anim_vocab = {}
    for anim_id_str, idx in vocab_data.get('vocab', {}).items():
        anim_vocab[str(idx)] = anim_id_str

    # Build windows dict keyed by vocab index
    windows_by_idx = {}
    for anim_id, w in windows.items():
        # Find the vocab index for this anim_id
        anim_id_str = str(anim_id)
        idx = vocab_data.get('vocab', {}).get(anim_id_str)
        if idx is not None:
            # Estimate normalized timing (assuming max 120 frames per attack)
            max_frames = 120.0
            windows_by_idx[str(idx)] = [{
                'anim_idx': idx,
                'anim_name': w['anim_name'],
                'window_idx': 0,
                'dodge_mean_frames': w['dodge_mean_frames'],
                'dodge_std_frames': w['dodge_std_frames'],
                'dodge_mean_norm': w['dodge_mean_frames'] / max_frames,
                'dodge_std_norm': w['dodge_std_frames'] / max_frames,
                'duration_mean': max_frames,  # Estimated
                'duration_std': 10.0,  # Estimated
                'n_dodges': w['count'],
                'n_instances': w['count'],
            }]

    output = {
        'windows': windows_by_idx,
        'anim_vocab': anim_vocab,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExported {len(windows_by_idx)} windows to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="rudder_data")
    parser.add_argument("--min-hits", type=int, default=5)
    parser.add_argument("--lead-time", type=int, default=8, help="Frames before hit to dodge")
    parser.add_argument("--output", default="dodge_windows_from_hits.json")

    args = parser.parse_args()

    hit_frames = analyze_hit_timing(args.data_dir)
    windows = build_windows_from_hits(hit_frames, args.min_hits, args.lead_time)
    export_windows(windows, args.output)
