"""Dodge Window Model - learns optimal dodge timing from human demonstrations."""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class DodgeWindow:
    """Dodge timing window for a single boss animation."""
    anim_idx: int
    anim_name: str
    window_idx: int  # 0 for single window, 0/1 for bimodal
    # Raw frame timing
    dodge_mean_frames: float
    dodge_std_frames: float
    # Normalized timing (0-1 scale)
    dodge_mean_norm: float
    dodge_std_norm: float
    # Animation duration stats
    duration_mean: float
    duration_std: float
    # Sample count
    n_dodges: int
    n_instances: int

    def is_dodge_window(
        self,
        elapsed_frames: float,
        sigma: float = 2.0,
        use_normalized: bool = False,
        estimated_duration: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Check if current timing is within the dodge window.

        Args:
            elapsed_frames: Current elapsed frames in boss animation
            sigma: Number of standard deviations for window width
            use_normalized: If True, normalize elapsed_frames by duration
            estimated_duration: Override duration estimate (if None, use stored mean)

        Returns:
            (is_in_window, distance_to_center)
            - is_in_window: True if within sigma std devs of mean
            - distance_to_center: Signed distance from mean (negative=early, positive=late)
        """
        if use_normalized:
            duration = estimated_duration or self.duration_mean
            if duration <= 0:
                return False, 0.0
            normalized = elapsed_frames / duration
            distance = normalized - self.dodge_mean_norm
            threshold = sigma * self.dodge_std_norm
        else:
            distance = elapsed_frames - self.dodge_mean_frames
            threshold = sigma * self.dodge_std_frames

        is_in_window = abs(distance) <= threshold
        return is_in_window, distance

    def get_reward(
        self,
        elapsed_frames: float,
        use_normalized: bool = False,
        estimated_duration: Optional[float] = None,
    ) -> float:
        """Get reward for dodging at this timing.

        Returns:
            Gaussian reward based on distance from mean dodge timing.
            1.0 at the mean, smoothly decaying with distance.
        """
        if use_normalized:
            duration = estimated_duration or self.duration_mean
            if duration <= 0:
                return 0.0
            normalized = elapsed_frames / duration
            z = (normalized - self.dodge_mean_norm) / max(self.dodge_std_norm, 1e-6)
        else:
            z = (elapsed_frames - self.dodge_mean_frames) / max(self.dodge_std_frames, 1e-6)

        # Gaussian reward: exp(-0.5 * z^2)
        reward = np.exp(-0.5 * z * z)
        return float(reward)


def detect_bimodal(timings: List[float], gap_threshold: float = 8.0) -> Tuple[bool, List[List[float]]]:
    """Detect if timing distribution is bimodal.

    Args:
        timings: List of dodge timings
        gap_threshold: Minimum gap between clusters to consider bimodal

    Returns:
        (is_bimodal, clusters) where clusters is list of 1 or 2 timing lists
    """
    if len(timings) < 4:
        return False, [timings]

    sorted_t = sorted(timings)

    # Find largest gap
    gaps = [(sorted_t[i+1] - sorted_t[i], i) for i in range(len(sorted_t)-1)]
    max_gap, gap_idx = max(gaps, key=lambda x: x[0])

    if max_gap > gap_threshold:
        cluster1 = sorted_t[:gap_idx+1]
        cluster2 = sorted_t[gap_idx+1:]
        # Only bimodal if both clusters have enough samples
        if len(cluster1) >= 2 and len(cluster2) >= 2:
            return True, [cluster1, cluster2]

    return False, [timings]


def remove_outliers(timings: List[float], threshold: float = 2.5) -> List[float]:
    """Remove outliers from timing distribution.

    Args:
        timings: List of dodge timings
        threshold: Number of std devs from median to consider outlier

    Returns:
        Filtered list with outliers removed
    """
    if len(timings) < 3:
        return timings

    arr = np.array(timings)
    median = np.median(arr)
    # Use MAD (median absolute deviation) for robust std estimate
    mad = np.median(np.abs(arr - median))
    # Convert MAD to std estimate (for normal distribution)
    std_estimate = mad * 1.4826

    if std_estimate < 1e-6:
        return timings

    mask = np.abs(arr - median) <= threshold * std_estimate
    return arr[mask].tolist()


class DodgeWindowModel:
    """Model of optimal dodge timing windows learned from human demonstrations."""

    def __init__(self):
        # anim_idx -> List[DodgeWindow] (usually 1, but 2 for bimodal)
        self.windows: Dict[int, List[DodgeWindow]] = {}
        self.anim_vocab: Dict[int, str] = {}  # anim_idx -> name

    @classmethod
    def from_recordings(
        cls,
        recordings: List[dict],
        anim_vocab: Dict[int, str],
        dodge_anim_ranges: List[Tuple[int, int]] = [(27000, 28000)],
        min_dodges: int = 2,
        min_instances: int = 2,
        min_elapsed: float = 3.0,
        outlier_threshold: float = 2.5,
        bimodal_gap: float = 8.0,
    ) -> "DodgeWindowModel":
        """Build model from recorded episodes.

        Args:
            recordings: List of recording dicts from record_episode.py
            anim_vocab: Mapping of anim_idx -> animation name
            dodge_anim_ranges: Ranges of player animation IDs that are dodges
            min_dodges: Minimum dodges required to create a window
            min_instances: Minimum animation instances required
            min_elapsed: Filter out dodges before this frame (fuzzy onset filter)
            outlier_threshold: Std devs from median to consider outlier
            bimodal_gap: Minimum gap between clusters to detect bimodal

        Returns:
            DodgeWindowModel with learned windows
        """
        model = cls()
        model.anim_vocab = anim_vocab

        def is_dodge(hero_anim_id: int) -> bool:
            for low, high in dodge_anim_ranges:
                if low <= hero_anim_id < high:
                    return True
            return False

        # Collect data per animation
        anim_data = defaultdict(lambda: {
            'dodge_frames': [],      # elapsed frames at dodge onset
            'durations': [],         # max elapsed per instance
        })

        total_dodges = 0
        failed_dodges = 0
        filtered_early = 0

        for rec in recordings:
            hero_anim = rec.get("hero_anim_id")
            if hero_anim is None:
                continue

            anim_idx = rec["anim_idx"]
            elapsed = rec["elapsed_frames"]
            player_damage = rec.get("player_damage")

            # Default to zeros if no damage data
            if player_damage is None:
                player_damage = np.zeros(len(anim_idx))

            # Track animation instances and dodges
            in_anim = None
            instance_max_elapsed = 0
            instance_dodges = []  # dodges in current instance
            instance_got_hit = False  # did player get hit in this instance
            prev_hero = None

            for i, (anim, el, hero, dmg) in enumerate(zip(anim_idx, elapsed, hero_anim, player_damage)):
                anim = int(anim)
                hero = int(hero)
                el = float(el)
                dmg = int(dmg)

                if anim != in_anim:
                    # Animation changed - record previous instance
                    if in_anim is not None and instance_max_elapsed > 0:
                        anim_data[in_anim]['durations'].append(instance_max_elapsed)
                        # Only add dodges if player didn't get hit
                        if instance_dodges:
                            total_dodges += len(instance_dodges)
                            if instance_got_hit:
                                failed_dodges += len(instance_dodges)
                            else:
                                # Filter early dodges
                                for d in instance_dodges:
                                    if d >= min_elapsed:
                                        anim_data[in_anim]['dodge_frames'].append(d)
                                    else:
                                        filtered_early += 1

                    in_anim = anim
                    instance_max_elapsed = 0
                    instance_dodges = []
                    instance_got_hit = False

                instance_max_elapsed = max(instance_max_elapsed, el)

                # Track if player got hit
                if dmg > 0:
                    instance_got_hit = True

                # Check for dodge onset
                if is_dodge(hero) and (prev_hero is None or not is_dodge(prev_hero)):
                    instance_dodges.append(el)

                prev_hero = hero

            # Don't forget last instance
            if in_anim is not None and instance_max_elapsed > 0:
                anim_data[in_anim]['durations'].append(instance_max_elapsed)
                if instance_dodges:
                    total_dodges += len(instance_dodges)
                    if instance_got_hit:
                        failed_dodges += len(instance_dodges)
                    else:
                        for d in instance_dodges:
                            if d >= min_elapsed:
                                anim_data[in_anim]['dodge_frames'].append(d)
                            else:
                                filtered_early += 1

        successful = total_dodges - failed_dodges
        print(f"Total dodges: {total_dodges}, Failed (hit): {failed_dodges}, "
              f"Filtered (early): {filtered_early}, Used: {successful - filtered_early}")

        # Build windows for animations with enough data
        outliers_removed = 0
        bimodal_count = 0

        for anim_idx, data in anim_data.items():
            raw_dodges = data['dodge_frames']
            durations = data['durations']

            if len(raw_dodges) < min_dodges or len(durations) < min_instances:
                continue

            # Remove outliers
            cleaned_dodges = remove_outliers(raw_dodges, outlier_threshold)
            outliers_removed += len(raw_dodges) - len(cleaned_dodges)

            if len(cleaned_dodges) < min_dodges:
                continue

            # Check for bimodal distribution
            is_bimodal, clusters = detect_bimodal(cleaned_dodges, bimodal_gap)

            if is_bimodal:
                bimodal_count += 1

            # Compute duration stats
            duration_mean = float(np.mean(durations))
            duration_std = float(np.std(durations)) if len(durations) > 1 else 5.0

            windows = []
            for window_idx, cluster in enumerate(clusters):
                if len(cluster) < min_dodges:
                    continue

                # Compute normalized timings
                dodge_normalized = [d / duration_mean for d in cluster]

                window = DodgeWindow(
                    anim_idx=anim_idx,
                    anim_name=anim_vocab.get(anim_idx, f"anim_{anim_idx}"),
                    window_idx=window_idx,
                    dodge_mean_frames=float(np.mean(cluster)),
                    dodge_std_frames=float(np.std(cluster)) if len(cluster) > 1 else 2.0,
                    dodge_mean_norm=float(np.mean(dodge_normalized)),
                    dodge_std_norm=float(np.std(dodge_normalized)) if len(dodge_normalized) > 1 else 0.1,
                    duration_mean=duration_mean,
                    duration_std=duration_std,
                    n_dodges=len(cluster),
                    n_instances=len(durations),
                )
                windows.append(window)

            if windows:
                model.windows[anim_idx] = windows

        print(f"Outliers removed: {outliers_removed}, Bimodal animations: {bimodal_count}")
        return model

    def get_windows(self, anim_idx: int) -> List[DodgeWindow]:
        """Get all dodge windows for an animation (1 or 2 for bimodal)."""
        return self.windows.get(anim_idx, [])

    def get_window(self, anim_idx: int, window_idx: int = 0) -> Optional[DodgeWindow]:
        """Get specific dodge window for an animation."""
        windows = self.windows.get(anim_idx, [])
        if window_idx < len(windows):
            return windows[window_idx]
        return None

    def should_dodge(
        self,
        anim_idx: int,
        elapsed_frames: float,
        sigma: float = 2.0,
    ) -> Tuple[bool, float]:
        """Check if agent should dodge now.

        Args:
            anim_idx: Current boss animation index
            elapsed_frames: Elapsed frames in current animation
            sigma: Window width in standard deviations

        Returns:
            (should_dodge, reward)
            - should_dodge: True if within any dodge window
            - reward: Best Gaussian reward across all windows (0-1)
        """
        windows = self.windows.get(anim_idx, [])
        if not windows:
            return False, 0.0

        best_reward = 0.0
        in_any_window = False

        for window in windows:
            in_window, _ = window.is_dodge_window(elapsed_frames, sigma=sigma)
            if in_window:
                in_any_window = True
                reward = window.get_reward(elapsed_frames)
                best_reward = max(best_reward, reward)

        return in_any_window, best_reward

    def save(self, path: str):
        """Save model to JSON."""
        data = {
            'windows': {
                str(k): [asdict(w) for w in v]
                for k, v in self.windows.items()
            },
            'anim_vocab': {str(k): v for k, v in self.anim_vocab.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved dodge window model to {path}")

    @classmethod
    def load(cls, path: str) -> "DodgeWindowModel":
        """Load model from JSON."""
        with open(path) as f:
            data = json.load(f)

        model = cls()
        model.anim_vocab = {int(k): v for k, v in data['anim_vocab'].items()}

        for k, v in data['windows'].items():
            windows = [DodgeWindow(**w) for w in v]
            model.windows[int(k)] = windows

        total_windows = sum(len(w) for w in model.windows.values())
        bimodal = sum(1 for w in model.windows.values() if len(w) > 1)
        print(f"Loaded dodge window model: {len(model.windows)} animations, "
              f"{total_windows} windows ({bimodal} bimodal)")
        return model

    def print_summary(self):
        """Print summary of learned windows."""
        print("\n" + "=" * 90)
        print("DODGE WINDOW MODEL")
        print("=" * 90)
        print(f"{'Anim':<6} {'Name':<15} {'Win':>3} {'N':>4} {'Mean':>7} {'Std':>5} "
              f"{'Norm':>6} {'Dur':>6} {'Conf':<6}")
        print("-" * 90)

        for anim_idx in sorted(self.windows.keys()):
            for w in self.windows[anim_idx]:
                # Confidence rating
                cv = (w.dodge_std_frames / w.dodge_mean_frames * 100) if w.dodge_mean_frames > 0 else 999
                if w.n_dodges >= 5 and cv < 20:
                    conf = 'HIGH'
                elif w.n_dodges >= 3 and cv < 35:
                    conf = 'MED'
                else:
                    conf = 'LOW'

                bimodal_marker = '*' if len(self.windows[anim_idx]) > 1 else ' '
                print(f"{anim_idx:<6} {w.anim_name[:15]:<15} {w.window_idx:>3}{bimodal_marker} "
                      f"{w.n_dodges:>4} {w.dodge_mean_frames:>7.1f} {w.dodge_std_frames:>5.1f} "
                      f"{w.dodge_mean_norm:>6.2f} {w.duration_mean:>6.1f} {conf:<6}")

        print("=" * 90)
        bimodal = sum(1 for w in self.windows.values() if len(w) > 1)
        print(f"Total: {len(self.windows)} animations, {bimodal} bimodal (* = bimodal)")


def build_model_from_recordings(
    recording_dir: str,
    output_path: str,
    dodge_anim_ranges: List[Tuple[int, int]] = [(27000, 28000)],
    min_dodges: int = 2,
    min_elapsed: float = 3.0,
    outlier_threshold: float = 2.5,
    bimodal_gap: float = 8.0,
):
    """Build and save dodge window model from recordings.

    Args:
        recording_dir: Directory containing .npz recordings
        output_path: Path to save model JSON
        dodge_anim_ranges: Player animation ID ranges for dodge detection
        min_dodges: Minimum dodges required per animation
        min_elapsed: Filter dodges before this frame
        outlier_threshold: Std devs from median for outlier removal
        bimodal_gap: Min gap to detect bimodal distribution
    """
    # Load animation vocab
    vocab_path = Path(__file__).parent / "anim_vocab.json"
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    anim_vocab = {v: k for k, v in vocab_data["vocab"].items()}

    # Load recordings
    recordings = []
    rec_path = Path(recording_dir)
    for f in sorted(rec_path.glob("*.npz")):
        print(f"Loading {f.name}...")
        data = np.load(str(f))
        recordings.append({k: data[k] for k in data.files})

    if not recordings:
        print(f"No recordings found in {recording_dir}")
        return

    print(f"\nLoaded {len(recordings)} recordings")

    # Build model
    model = DodgeWindowModel.from_recordings(
        recordings=recordings,
        anim_vocab=anim_vocab,
        dodge_anim_ranges=dodge_anim_ranges,
        min_dodges=min_dodges,
        min_elapsed=min_elapsed,
        outlier_threshold=outlier_threshold,
        bimodal_gap=bimodal_gap,
    )

    model.print_summary()
    model.save(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dodge window model from recordings")
    parser.add_argument("recording_dir", help="Directory containing .npz recordings")
    parser.add_argument("--output", "-o", default="dodge_windows.json", help="Output path")
    parser.add_argument("--dodge-range", type=int, nargs=2, default=[27000, 28000],
                        help="Player dodge animation ID range (default: 27000 28000)")
    parser.add_argument("--min-dodges", type=int, default=2, help="Minimum dodges per window")
    parser.add_argument("--min-elapsed", type=float, default=3.0,
                        help="Filter dodges before this frame (default: 3)")
    parser.add_argument("--outlier-threshold", type=float, default=2.5,
                        help="Std devs from median for outlier removal (default: 2.5)")
    parser.add_argument("--bimodal-gap", type=float, default=8.0,
                        help="Min gap between clusters for bimodal detection (default: 8)")

    args = parser.parse_args()

    build_model_from_recordings(
        recording_dir=args.recording_dir,
        output_path=args.output,
        dodge_anim_ranges=[(args.dodge_range[0], args.dodge_range[1])],
        min_dodges=args.min_dodges,
        min_elapsed=args.min_elapsed,
        outlier_threshold=args.outlier_threshold,
        bimodal_gap=args.bimodal_gap,
    )
