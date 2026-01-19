"""Profile Z value difference between Hero and NPC.

Uses the same transform as trace_paths.py:
  transform_z = HeroGlobalPosZ - HeroLocalPosZ
  npc_corrected_z = NpcGlobalPosZ + transform_z
"""

import time
import argparse
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pysiphon import SiphonClient
from rich.console import Console

console = Console()


def run_z_profile(
    host: str = "192.168.48.1:50051",
    config_path: str = "gameplay_pipeline/configs/er_siphon_config.toml",
    window_seconds: float = 30.0,
    sample_rate: float = 30.0,
):
    """
    Live plot Z values with local→global transform.

    Shows:
    - HeroGlobalPosZ (truly global)
    - NpcGlobalPosZ (actually local, needs transform)
    - Transform (HeroGlobal - HeroLocal)
    - NPC Corrected Z (NpcGlobal + transform)
    - Difference between HeroGlobalZ and NPC Corrected Z

    Args:
        host: Siphon server address
        config_path: Path to siphon config
        window_seconds: Time window to display
        sample_rate: Samples per second
    """
    console.print(f"[cyan]Connecting to {host}...[/cyan]")

    client = SiphonClient(host)
    client.init_all(config_path)
    console.print("[green]Connected![/green]")

    # Data buffers
    max_samples = int(window_seconds * sample_rate)
    timestamps = deque(maxlen=max_samples)
    hero_global_z = deque(maxlen=max_samples)
    hero_local_z = deque(maxlen=max_samples)
    npc_raw_z = deque(maxlen=max_samples)
    transform_z = deque(maxlen=max_samples)
    npc_corrected_z = deque(maxlen=max_samples)
    z_diff = deque(maxlen=max_samples)

    # Set up live plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Z Value Profile with Local→Global Transform', fontsize=14, fontweight='bold')

    # Row 0: Hero Global Z vs Hero Local Z
    line_hero_global, = axes[0].plot([], [], 'c-', linewidth=1.5, label='HeroGlobalPosZ')
    line_hero_local, = axes[0].plot([], [], 'c--', linewidth=1.5, alpha=0.6, label='HeroLocalPosZ')

    # Row 1: Transform (HeroGlobal - HeroLocal)
    line_transform, = axes[1].plot([], [], 'g-', linewidth=1.5, label='Transform (Global - Local)')

    # Row 2: NPC Raw vs Corrected
    line_npc_raw, = axes[2].plot([], [], 'm--', linewidth=1.5, alpha=0.6, label='NpcGlobalPosZ (raw)')
    line_npc_corrected, = axes[2].plot([], [], 'm-', linewidth=1.5, label='NPC Corrected (raw + transform)')

    # Row 3: Difference (HeroGlobalZ - NpcCorrectedZ)
    line_diff, = axes[3].plot([], [], 'y-', linewidth=1.5, label='HeroGlobalZ - NpcCorrectedZ')
    axes[3].axhline(y=0, color='white', linestyle='--', alpha=0.5)

    for ax in axes:
        ax.set_xlim(0, window_seconds)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[0].set_ylabel('Hero Z')
    axes[1].set_ylabel('Transform')
    axes[2].set_ylabel('NPC Z')
    axes[3].set_ylabel('Diff (Hero - NPC)')
    axes[3].set_xlabel('Time (s)')

    # Stats text
    stats_text = fig.text(0.02, 0.02, '', fontsize=10, family='monospace')

    start_time = time.time()
    sample_interval = 1.0 / sample_rate
    last_sample = 0

    def update(frame):
        nonlocal last_sample

        now = time.time()
        if now - last_sample >= sample_interval:
            last_sample = now

            hgz = client.get_attribute("HeroGlobalPosZ")
            hlz = client.get_attribute("HeroLocalPosZ")
            nrz = client.get_attribute("NpcGlobalPosZ")

            if hgz and hlz and nrz:
                t = now - start_time
                hgz_val = hgz["value"]
                hlz_val = hlz["value"]
                nrz_val = nrz["value"]

                # Compute transform (same as trace_paths.py)
                tz = hgz_val - hlz_val
                ncz = nrz_val + tz
                diff = hgz_val - ncz

                timestamps.append(t)
                hero_global_z.append(hgz_val)
                hero_local_z.append(hlz_val)
                npc_raw_z.append(nrz_val)
                transform_z.append(tz)
                npc_corrected_z.append(ncz)
                z_diff.append(diff)

        if len(timestamps) > 1:
            t_arr = np.array(timestamps)
            t_offset = t_arr - t_arr[0]

            # Update data
            line_hero_global.set_data(t_offset, np.array(hero_global_z))
            line_hero_local.set_data(t_offset, np.array(hero_local_z))
            line_transform.set_data(t_offset, np.array(transform_z))
            line_npc_raw.set_data(t_offset, np.array(npc_raw_z))
            line_npc_corrected.set_data(t_offset, np.array(npc_corrected_z))
            line_diff.set_data(t_offset, np.array(z_diff))

            # Update x limits
            t_max = max(window_seconds, t_offset[-1])
            t_min = max(0, t_offset[-1] - window_seconds)
            for ax in axes:
                ax.set_xlim(t_min, t_max)

            # Auto-scale y
            all_hero = list(hero_global_z) + list(hero_local_z)
            axes[0].set_ylim(min(all_hero) - 1, max(all_hero) + 1)

            axes[1].set_ylim(min(transform_z) - 1, max(transform_z) + 1)

            all_npc = list(npc_raw_z) + list(npc_corrected_z)
            axes[2].set_ylim(min(all_npc) - 1, max(all_npc) + 1)

            diff_arr = np.array(z_diff)
            diff_range = max(abs(diff_arr.min()), abs(diff_arr.max()), 0.5)
            axes[3].set_ylim(-diff_range - 0.5, diff_range + 0.5)

            # Update stats
            stats_text.set_text(
                f"HeroGlobalZ: {hero_global_z[-1]:8.3f}  |  "
                f"HeroLocalZ: {hero_local_z[-1]:8.3f}  |  "
                f"Transform: {transform_z[-1]:+8.3f}  |  "
                f"NpcRawZ: {npc_raw_z[-1]:8.3f}  |  "
                f"NpcCorrectedZ: {npc_corrected_z[-1]:8.3f}  |  "
                f"Diff: {z_diff[-1]:+8.4f}"
            )

        return line_hero_global, line_hero_local, line_transform, line_npc_raw, line_npc_corrected, line_diff, stats_text

    console.print("[yellow]Starting live plot. Close window to stop.[/yellow]")

    ani = animation.FuncAnimation(
        fig, update, interval=1000 // sample_rate, blit=False, cache_frame_data=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    plt.show()

    client.close()

    # Print final stats
    if z_diff:
        console.print(f"\n[cyan]Final Statistics:[/cyan]")
        console.print(f"  Samples: {len(z_diff)}")
        console.print(f"  Transform Mean: {np.mean(transform_z):+.4f}")
        console.print(f"  Transform Std:  {np.std(transform_z):.4f}")
        console.print(f"  Diff Mean (Hero - NPC Corrected): {np.mean(z_diff):+.4f}")
        console.print(f"  Diff Std:  {np.std(z_diff):.4f}")
        console.print(f"  Diff Min:  {min(z_diff):+.4f}")
        console.print(f"  Diff Max:  {max(z_diff):+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Profile Z value difference")
    parser.add_argument("--host", default="192.168.48.1:50051", help="Siphon server address")
    parser.add_argument("--config", default="gameplay_pipeline/configs/er_siphon_config.toml", help="Config path")
    parser.add_argument("--window", type=float, default=30.0, help="Display window (seconds)")
    parser.add_argument("--rate", type=float, default=30.0, help="Sample rate (Hz)")

    args = parser.parse_args()

    run_z_profile(
        host=args.host,
        config_path=args.config,
        window_seconds=args.window,
        sample_rate=args.rate,
    )


if __name__ == "__main__":
    main()
