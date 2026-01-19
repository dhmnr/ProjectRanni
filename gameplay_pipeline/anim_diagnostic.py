"""Diagnostic tool to analyze NpcAnimLength behavior."""

import time
import argparse
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pysiphon import SiphonClient
from rich.console import Console

console = Console()


def run_diagnostic(
    host: str = "192.168.48.1:50051",
    config_path: str = "gameplay_pipeline/configs/er_siphon_config.toml",
    duration: float = 30.0,
    sample_rate: float = 30.0,
):
    """
    Record NpcAnimId and NpcAnimLength over time.

    Args:
        host: Siphon server address
        config_path: Path to siphon config
        duration: Recording duration in seconds
        sample_rate: Samples per second
    """
    console.print(f"[cyan]Connecting to {host}...[/cyan]")

    with SiphonClient(host) as client:
        client.init_all(config_path)
        console.print("[green]Connected![/green]")
        console.print(f"[yellow]Recording for {duration}s at {sample_rate}Hz. Move the boss around![/yellow]")

        # Data collection
        timestamps = []
        anim_ids = []
        anim_lengths = []

        sample_interval = 1.0 / sample_rate
        start_time = time.time()
        last_sample = 0

        try:
            while time.time() - start_time < duration:
                now = time.time()
                if now - last_sample >= sample_interval:
                    last_sample = now

                    anim_id = client.get_attribute("NpcAnimId")
                    anim_length = client.get_attribute("NpcAnimLength")

                    if anim_id and anim_length:
                        t = now - start_time
                        timestamps.append(t)
                        anim_ids.append(anim_id["value"])
                        anim_lengths.append(anim_length["value"])

                        console.print(
                            f"[dim]t={t:6.2f}s | AnimId: {anim_id['value']:10d} | "
                            f"AnimLength: {anim_length['value']:8.4f}[/dim]",
                            end="\r"
                        )

                time.sleep(0.001)

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user[/yellow]")

    console.print(f"\n[green]Collected {len(timestamps)} samples[/green]")

    if not timestamps:
        console.print("[red]No data collected![/red]")
        return

    # Analyze data
    timestamps = np.array(timestamps)
    anim_ids = np.array(anim_ids)
    anim_lengths = np.array(anim_lengths)

    # Get unique anim IDs and their stats
    unique_ids = np.unique(anim_ids)
    console.print(f"\n[cyan]Found {len(unique_ids)} unique AnimIds:[/cyan]")

    stats = []
    for aid in unique_ids:
        mask = anim_ids == aid
        lengths = anim_lengths[mask]
        count = len(lengths)
        if count > 0:
            stats.append({
                'id': aid,
                'count': count,
                'min': lengths.min(),
                'max': lengths.max(),
                'mean': lengths.mean(),
                'std': lengths.std(),
            })

    # Sort by count
    stats.sort(key=lambda x: x['count'], reverse=True)

    console.print("\n[bold]AnimId Statistics (sorted by frequency):[/bold]")
    console.print(f"{'AnimId':>12} | {'Count':>6} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Std':>8}")
    console.print("-" * 70)
    for s in stats[:20]:  # Top 20
        console.print(
            f"{s['id']:>12d} | {s['count']:>6d} | {s['min']:>8.4f} | "
            f"{s['max']:>8.4f} | {s['mean']:>8.4f} | {s['std']:>8.4f}"
        )

    # Create one long scrollable timeline
    # Width scales with duration for scrollability
    width_per_second = 2.0  # inches per second of data
    total_width = max(14, duration * width_per_second)

    fig, ax = plt.subplots(figsize=(total_width, 6))

    # Color map for different anim IDs - use distinct colors
    cmap = plt.cm.tab20
    id_to_color = {aid: cmap(i % 20) for i, aid in enumerate(unique_ids)}

    # Plot each AnimId as a separate series for legend
    for aid in unique_ids:
        mask = anim_ids == aid
        t_anim = timestamps[mask]
        l_anim = anim_lengths[mask]
        color = id_to_color[aid]
        ax.scatter(t_anim, l_anim, c=[color], s=20, alpha=0.8, label=f'{aid}')
        # Connect points within same animation
        ax.plot(t_anim, l_anim, c=color, alpha=0.3, linewidth=1)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('AnimLength', fontsize=12)
    ax.set_title('NpcAnimLength Timeline (color = AnimId)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend outside plot, scrollable
    ax.legend(
        bbox_to_anchor=(1.01, 1), loc='upper left',
        fontsize=8, title='AnimId', title_fontsize=10,
        ncol=max(1, len(unique_ids) // 20)
    )

    plt.tight_layout()

    # Save as wide image
    output_path = "paths/anim_timeline.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Saved timeline to {output_path}[/green]")

    # --- Find animation durations ---
    # When AnimId changes, the LAST AnimLength before the change is the duration
    durations = defaultdict(list)  # AnimId -> list of durations

    for i in range(1, len(anim_ids)):
        if anim_ids[i] != anim_ids[i-1]:
            # Animation changed - record the last AnimLength of previous animation
            prev_id = anim_ids[i-1]
            prev_duration = anim_lengths[i-1]
            durations[prev_id].append(prev_duration)

    console.print(f"\n[cyan]Animation Durations (last value before AnimId change):[/cyan]")
    console.print(f"{'AnimId':>12} | {'Count':>6} | {'Min':>8} | {'Max':>8} | {'Mean':>8}")
    console.print("-" * 55)

    # Sort by count
    sorted_ids = sorted(durations.keys(), key=lambda x: len(durations[x]), reverse=True)
    for aid in sorted_ids[:15]:
        d = durations[aid]
        console.print(f"{aid:>12d} | {len(d):>6d} | {min(d):>8.4f} | {max(d):>8.4f} | {np.mean(d):>8.4f}")

    # Select AnimIds: idle (2002000) + top frequent ones
    target_ids = []
    if 2002000 in durations:
        target_ids.append(2002000)
    for aid in sorted_ids:
        if aid not in target_ids and len(target_ids) < 12:
            target_ids.append(aid)

    # Grid of separate histograms
    n_plots = len(target_ids)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig2, axes2 = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes2 = axes2.flatten() if n_plots > 1 else [axes2]

    for i, aid in enumerate(target_ids):
        ax = axes2[i]
        d = durations[aid]

        ax.hist(d, bins=min(20, len(d)), edgecolor='black', alpha=0.7, color=plt.cm.tab10(i % 10))
        ax.axvline(np.mean(d), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(d):.3f}')
        ax.axvline(max(d), color='orange', linestyle=':', linewidth=2, label=f'Max: {max(d):.3f}')

        ax.set_xlabel('Duration')
        ax.set_ylabel('Count')
        ax.set_title(f'AnimId {aid} (n={len(d)})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(target_ids), len(axes2)):
        axes2[i].set_visible(False)

    plt.suptitle('Animation Duration Distributions (last AnimLength before transition)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path2 = "paths/anim_duration_hist.png"
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Saved duration histograms to {output_path2}[/green]")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Diagnose NpcAnimLength behavior")
    parser.add_argument("--host", default="192.168.48.1:50051", help="Siphon server address")
    parser.add_argument("--config", default="gameplay_pipeline/configs/er_siphon_config.toml", help="Config path")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration (seconds)")
    parser.add_argument("--rate", type=float, default=30.0, help="Sample rate (Hz)")

    args = parser.parse_args()

    run_diagnostic(
        host=args.host,
        config_path=args.config,
        duration=args.duration,
        sample_rate=args.rate,
    )


if __name__ == "__main__":
    main()
