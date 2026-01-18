"""Trace and visualize player and boss paths on the game map."""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pysiphon import SiphonClient
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()


@dataclass
class PathPoint:
    """A single point in a path with timestamp."""
    timestamp: float
    global_x: float
    global_y: float
    chunk_x: float
    chunk_y: float
    local_x: float
    local_y: float


@dataclass
class PathData:
    """Container for path tracing data."""
    player_path: List[PathPoint] = field(default_factory=list)
    boss_path: List[PathPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metadata": self.metadata,
            "player_path": [asdict(p) for p in self.player_path],
            "boss_path": [asdict(p) for p in self.boss_path],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathData":
        """Load from dictionary."""
        path_data = cls()
        path_data.metadata = data.get("metadata", {})
        path_data.player_path = [PathPoint(**p) for p in data.get("player_path", [])]
        path_data.boss_path = [PathPoint(**p) for p in data.get("boss_path", [])]
        return path_data


class PathTracer:
    """Trace player and boss paths using the Siphon service."""

    # Attributes needed for path tracing
    REQUIRED_ATTRIBUTES = [
        "ChunkX", "ChunkY",
        "HeroLocalPosX", "HeroLocalPosY",
        "NpcGlobalPosX", "NpcGlobalPosY",
    ]

    # Optional attributes for context
    OPTIONAL_ATTRIBUTES = [
        "HeroHp", "NpcHp", "NpcId",
    ]

    def __init__(
        self,
        host: str = "192.168.48.1:50051",
        config_path: Optional[str] = None,
    ):
        """
        Initialize the path tracer.

        Args:
            host: Siphon server address
            config_path: Path to Siphon config file
        """
        self.host = host
        self.config_path = config_path
        self.client: Optional[SiphonClient] = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        """Connect to the Siphon server."""
        console.print(f"[cyan]Connecting to Siphon server at {self.host}...[/cyan]")
        self.client = SiphonClient(self.host)
        self.client.__enter__()

        if self.config_path:
            console.print(f"[cyan]Initializing with config: {self.config_path}[/cyan]")
            self.client.init_all(self.config_path)

        status = self.client.get_server_status()
        console.print(f"[green]Connected to Siphon server[/green]")
        console.print(f"  Status: {status}")

    def disconnect(self):
        """Disconnect from the Siphon server."""
        if self.client:
            self.client.__exit__(None, None, None)
            console.print("[cyan]Disconnected from Siphon server[/cyan]")

    def get_attribute(self, name: str) -> Optional[float]:
        """Get a single attribute value."""
        try:
            result = self.client.get_attribute(name)
            return result["value"]
        except Exception as e:
            console.print(f"[yellow]Warning: Could not get '{name}': {e}[/yellow]")
            return None

    def get_current_positions(self) -> Dict[str, Optional[float]]:
        """
        Get current chunk and local positions.

        Returns:
            Dictionary with chunk_x, chunk_y, hero_local_x, hero_local_y,
            npc_global_x, npc_global_y
        """
        return {
            "chunk_x": self.get_attribute("ChunkX"),
            "chunk_y": self.get_attribute("ChunkY"),
            "hero_local_x": self.get_attribute("HeroLocalPosX"),
            "hero_local_y": self.get_attribute("HeroLocalPosY"),
            "npc_global_x": self.get_attribute("NpcGlobalPosX"),
            "npc_global_y": self.get_attribute("NpcGlobalPosY"),
        }

    @staticmethod
    def compute_global_position(chunk: float, local: float) -> float:
        """
        Compute global position from chunk and local coordinates.

        The coordinate system works as follows:
        - Local coords run from +32 (left/back) to -32 (right/front)
        - When local hits -32, chunk increases by 32 and local resets to 0
        - Global = chunk - local gives continuous coordinates

        Args:
            chunk: Chunk coordinate
            local: Local coordinate within chunk

        Returns:
            Global coordinate
        """
        return chunk - local

    def trace_paths(
        self,
        output_path: str = "./path_data.json",
        sample_rate: float = 10.0,
        max_duration: Optional[float] = None,
        stop_on_death: bool = True,
        show_live: bool = True,
        live_plot: bool = False,
    ) -> PathData:
        """
        Trace player and boss paths.

        Args:
            output_path: Where to save the path data
            sample_rate: Samples per second
            max_duration: Maximum recording duration in seconds (None for unlimited)
            stop_on_death: Stop when player or boss HP reaches 0
            show_live: Show live position updates
            live_plot: Show real-time plot of paths

        Returns:
            PathData containing the recorded paths
        """
        sample_interval = 1.0 / sample_rate
        path_data = PathData()
        path_data.metadata = {
            "sample_rate": sample_rate,
            "start_time": time.time(),
            "output_path": output_path,
        }

        console.print("\n[bold cyan]Starting path tracing...[/bold cyan]")
        console.print(f"  Sample rate: {sample_rate} Hz")
        console.print(f"  Output: {output_path}")
        if max_duration:
            console.print(f"  Max duration: {max_duration}s")
        if live_plot:
            console.print(f"  Live plot: enabled")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")

        # Track offsets for tile boundary jump compensation (shared for player and boss)
        offset_x, offset_y = 0.0, 0.0
        last_player_local_x, last_player_local_y = None, None
        last_chunk_x, last_chunk_y = None, None

        # Continuous plot coordinates (separate from PathPoint storage)
        plot_player_xs, plot_player_ys = [], []
        plot_boss_xs, plot_boss_ys = [], []

        # Setup live plot if enabled
        fig, ax, player_line, boss_line, player_marker, boss_marker = None, None, None, None, None, None
        if live_plot:
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 8))
            player_line, = ax.plot([], [], 'b-', linewidth=2, label='Player')
            boss_line, = ax.plot([], [], 'r-', linewidth=2, label='Boss')
            player_marker, = ax.plot([], [], 'bo', markersize=10)
            boss_marker, = ax.plot([], [], 'ro', markersize=10)
            ax.set_xlabel('Global Y', fontsize=12)
            ax.set_ylabel('Global X', fontsize=12)
            ax.set_title('Live Path Trace', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.show(block=False)
            plt.pause(0.01)

        start_time = time.time()
        last_sample_time = 0

        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_time

                # Check max duration
                if max_duration and elapsed >= max_duration:
                    console.print("\n[yellow]Max duration reached[/yellow]")
                    break

                # Sample at specified rate
                if current_time - last_sample_time >= sample_interval:
                    last_sample_time = current_time
                    timestamp = elapsed

                    # Get positions
                    positions = self.get_current_positions()

                    # Skip if any required value is None
                    if any(v is None for v in positions.values()):
                        continue

                    # Get current values
                    curr_chunk_x = positions["chunk_x"]
                    curr_chunk_y = positions["chunk_y"]
                    curr_player_x = positions["hero_local_x"]
                    curr_player_y = positions["hero_local_y"]
                    curr_boss_x = positions["npc_global_x"]
                    curr_boss_y = positions["npc_global_y"]

                    # Detect tile boundary crossing by chunk change
                    if last_chunk_x is not None:
                        chunk_delta_x = curr_chunk_x - last_chunk_x
                        chunk_delta_y = curr_chunk_y - last_chunk_y
                        # If chunk changed, compensate for the jump in local coords
                        if abs(chunk_delta_x) > 1.0:
                            local_delta_x = curr_player_x - last_player_local_x
                            offset_x -= local_delta_x
                        if abs(chunk_delta_y) > 1.0:
                            local_delta_y = curr_player_y - last_player_local_y
                            offset_y -= local_delta_y

                    last_chunk_x, last_chunk_y = curr_chunk_x, curr_chunk_y
                    last_player_local_x, last_player_local_y = curr_player_x, curr_player_y

                    # Compute continuous global positions (shared offset for both)
                    player_global_x = curr_player_x + offset_x
                    player_global_y = curr_player_y + offset_y
                    boss_global_x = curr_boss_x + offset_x
                    boss_global_y = curr_boss_y + offset_y

                    # Append to plot lists (swap X/Y, flip X axis only)
                    plot_player_xs.append(player_global_y)
                    plot_player_ys.append(-player_global_x)
                    plot_boss_xs.append(boss_global_y)
                    plot_boss_ys.append(-boss_global_x)

                    # Record player path point
                    player_point = PathPoint(
                        timestamp=timestamp,
                        global_x=player_global_x,
                        global_y=player_global_y,
                        chunk_x=positions["chunk_x"],
                        chunk_y=positions["chunk_y"],
                        local_x=positions["hero_local_x"],
                        local_y=positions["hero_local_y"],
                    )
                    path_data.player_path.append(player_point)

                    # Record boss path point
                    boss_point = PathPoint(
                        timestamp=timestamp,
                        global_x=boss_global_x,
                        global_y=boss_global_y,
                        chunk_x=positions["chunk_x"],
                        chunk_y=positions["chunk_y"],
                        local_x=positions["npc_global_x"],
                        local_y=positions["npc_global_y"],
                    )
                    path_data.boss_path.append(boss_point)

                    # Update live plot
                    if live_plot and fig is not None:
                        # Use continuous plot lists
                        player_line.set_data(plot_player_xs, plot_player_ys)
                        boss_line.set_data(plot_boss_xs, plot_boss_ys)
                        player_marker.set_data([plot_player_xs[-1]], [plot_player_ys[-1]])
                        boss_marker.set_data([plot_boss_xs[-1]], [plot_boss_ys[-1]])

                        # Auto-scale axes
                        ax.relim()
                        ax.autoscale_view()

                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()

                    # Check for death if enabled
                    if stop_on_death:
                        hero_hp = self.get_attribute("HeroHp")
                        npc_hp = self.get_attribute("NpcHp")
                        if hero_hp is not None and hero_hp <= 0:
                            console.print("\n[red]Player died - stopping trace[/red]")
                            break
                        if npc_hp is not None and npc_hp <= 0:
                            console.print("\n[green]Boss defeated - stopping trace[/green]")
                            break

                    # Show live update (use continuous plot values)
                    if show_live:
                        console.print(
                            f"[dim]t={timestamp:6.1f}s | "
                            f"Player: ({plot_player_xs[-1]:8.2f}, {plot_player_ys[-1]:8.2f}) | "
                            f"Boss: ({plot_boss_xs[-1]:8.2f}, {plot_boss_ys[-1]:8.2f}) | "
                            f"Chunk: ({curr_chunk_x:.0f}, {curr_chunk_y:.0f})[/dim]",
                            end="\r"
                        )

                # Small sleep to avoid busy loop
                time.sleep(0.001)

        except KeyboardInterrupt:
            console.print("\n[yellow]Tracing stopped by user[/yellow]")

        # Close live plot
        if live_plot and fig is not None:
            plt.ioff()
            plt.close(fig)

        # Finalize metadata
        path_data.metadata["end_time"] = time.time()
        path_data.metadata["duration"] = path_data.metadata["end_time"] - path_data.metadata["start_time"]
        path_data.metadata["total_samples"] = len(path_data.player_path)

        # Save data
        self.save_path_data(path_data, output_path)

        console.print(f"\n[green]Path tracing complete[/green]")
        console.print(f"  Duration: {path_data.metadata['duration']:.1f}s")
        console.print(f"  Samples: {path_data.metadata['total_samples']}")
        console.print(f"  Saved to: {output_path}")

        return path_data

    def save_path_data(self, path_data: PathData, output_path: str):
        """Save path data to file (JSON or CSV based on extension)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".csv":
            self._save_csv(path_data, output_path)
        else:
            self._save_json(path_data, output_path)

    def _save_json(self, path_data: PathData, output_path: Path):
        """Save as JSON."""
        with open(output_path, "w") as f:
            json.dump(path_data.to_dict(), f, indent=2)

    def _save_csv(self, path_data: PathData, output_path: Path):
        """Save as CSV with both paths merged."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "player_global_x", "player_global_y",
                "player_chunk_x", "player_chunk_y",
                "player_local_x", "player_local_y",
                "boss_global_x", "boss_global_y",
                "boss_local_x", "boss_local_y",
            ])

            for player_pt, boss_pt in zip(path_data.player_path, path_data.boss_path):
                writer.writerow([
                    player_pt.timestamp,
                    player_pt.global_x, player_pt.global_y,
                    player_pt.chunk_x, player_pt.chunk_y,
                    player_pt.local_x, player_pt.local_y,
                    boss_pt.global_x, boss_pt.global_y,
                    boss_pt.local_x, boss_pt.local_y,
                ])

    @staticmethod
    def load_path_data(input_path: str) -> PathData:
        """Load path data from file."""
        input_path = Path(input_path)

        if input_path.suffix == ".csv":
            return PathTracer._load_csv(input_path)
        else:
            with open(input_path) as f:
                return PathData.from_dict(json.load(f))

    @staticmethod
    def _load_csv(input_path: Path) -> PathData:
        """Load from CSV."""
        path_data = PathData()

        with open(input_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_pt = PathPoint(
                    timestamp=float(row["timestamp"]),
                    global_x=float(row["player_global_x"]),
                    global_y=float(row["player_global_y"]),
                    chunk_x=float(row["player_chunk_x"]),
                    chunk_y=float(row["player_chunk_y"]),
                    local_x=float(row["player_local_x"]),
                    local_y=float(row["player_local_y"]),
                )
                boss_pt = PathPoint(
                    timestamp=float(row["timestamp"]),
                    global_x=float(row["boss_global_x"]),
                    global_y=float(row["boss_global_y"]),
                    chunk_x=float(row["player_chunk_x"]),
                    chunk_y=float(row["player_chunk_y"]),
                    local_x=float(row["boss_local_x"]),
                    local_y=float(row["boss_local_y"]),
                )
                path_data.player_path.append(player_pt)
                path_data.boss_path.append(boss_pt)

        return path_data


def visualize_paths(
    path_data: PathData,
    output_path: Optional[str] = None,
    title: str = "Player and Boss Paths",
    show_grid: bool = True,
    show_time_gradient: bool = True,
    figsize: tuple = (12, 10),
    background_image: Optional[str] = None,
    image_bounds: Optional[tuple] = None,
) -> plt.Figure:
    """
    Visualize paths on a 2D plot.

    Args:
        path_data: PathData containing player and boss paths
        output_path: Where to save the figure (None to just display)
        title: Plot title
        show_grid: Show grid lines
        show_time_gradient: Color paths by time (start->end gradient)
        figsize: Figure size
        background_image: Optional path to background image
        image_bounds: Bounds for background image (x_min, x_max, y_min, y_max)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Load background image if provided
    if background_image and os.path.exists(background_image):
        img = plt.imread(background_image)
        if image_bounds:
            ax.imshow(img, extent=image_bounds, aspect='auto', alpha=0.7)
        else:
            ax.imshow(img, aspect='auto', alpha=0.7)

    # Extract coordinates (use compensated global coords, swap X/Y, flip X axis only)
    player_x = [p.global_y for p in path_data.player_path]
    player_y = [-p.global_x for p in path_data.player_path]
    boss_x = [p.global_y for p in path_data.boss_path]
    boss_y = [-p.global_x for p in path_data.boss_path]

    if not player_x:
        console.print("[yellow]No path data to visualize[/yellow]")
        return fig

    if show_time_gradient:
        # Create time-based color gradient
        times = [p.timestamp for p in path_data.player_path]
        t_norm = np.array(times)
        t_norm = (t_norm - t_norm.min()) / (t_norm.max() - t_norm.min() + 1e-6)

        # Plot player path with gradient (blue to cyan)
        for i in range(len(player_x) - 1):
            color = plt.cm.Blues(0.3 + 0.7 * t_norm[i])
            ax.plot(player_x[i:i+2], player_y[i:i+2], color=color, linewidth=2)

        # Plot boss path with gradient (red to orange)
        for i in range(len(boss_x) - 1):
            color = plt.cm.Reds(0.3 + 0.7 * t_norm[i])
            ax.plot(boss_x[i:i+2], boss_y[i:i+2], color=color, linewidth=2)
    else:
        # Simple solid color paths
        ax.plot(player_x, player_y, 'b-', linewidth=2, label='Player')
        ax.plot(boss_x, boss_y, 'r-', linewidth=2, label='Boss')

    # Mark start and end points
    ax.scatter([player_x[0]], [player_y[0]], c='blue', s=100, marker='o',
               edgecolors='white', linewidths=2, zorder=5, label='Player Start')
    ax.scatter([player_x[-1]], [player_y[-1]], c='cyan', s=100, marker='s',
               edgecolors='white', linewidths=2, zorder=5, label='Player End')

    ax.scatter([boss_x[0]], [boss_y[0]], c='red', s=100, marker='o',
               edgecolors='white', linewidths=2, zorder=5, label='Boss Start')
    ax.scatter([boss_x[-1]], [boss_y[-1]], c='orange', s=100, marker='s',
               edgecolors='white', linewidths=2, zorder=5, label='Boss End')

    # Styling
    ax.set_xlabel('Global Y', fontsize=12)
    ax.set_ylabel('Global X', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add chunk boundaries (every 64 units since tiles are 64 wide)
        all_x = player_x + boss_x
        all_y = player_y + boss_y
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Extend bounds slightly
        margin = 10
        x_min, x_max = x_min - margin, x_max + margin
        y_min, y_max = y_min - margin, y_max + margin

        # Draw chunk boundaries
        chunk_size = 64
        for x in range(int(x_min // chunk_size) * chunk_size,
                       int(x_max // chunk_size + 2) * chunk_size, chunk_size):
            ax.axvline(x=x, color='gray', alpha=0.2, linewidth=0.5)
        for y in range(int(y_min // chunk_size) * chunk_size,
                       int(y_max // chunk_size + 2) * chunk_size, chunk_size):
            ax.axhline(y=y, color='gray', alpha=0.2, linewidth=0.5)

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved visualization to {output_path}[/green]")

    return fig


def main():
    """CLI entry point for path tracing."""
    parser = argparse.ArgumentParser(
        description="Trace and visualize player/boss paths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Trace command
    trace_parser = subparsers.add_parser("trace", help="Record path data")
    trace_parser.add_argument(
        "--host", default="192.168.48.1:50051",
        help="Siphon server address"
    )
    trace_parser.add_argument(
        "--siphon-config", required=True,
        help="Path to Siphon config file"
    )
    trace_parser.add_argument(
        "--output", "-o", default="./path_data.json",
        help="Output file path (.json or .csv)"
    )
    trace_parser.add_argument(
        "--sample-rate", type=float, default=10.0,
        help="Samples per second"
    )
    trace_parser.add_argument(
        "--duration", type=float, default=None,
        help="Maximum duration in seconds"
    )
    trace_parser.add_argument(
        "--no-stop-on-death", action="store_true",
        help="Don't stop when player/boss dies"
    )
    trace_parser.add_argument(
        "--quiet", action="store_true",
        help="Disable live position updates"
    )
    trace_parser.add_argument(
        "--live-plot", action="store_true",
        help="Show real-time plot while tracing"
    )
    trace_parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization after tracing"
    )

    # Visualize command (for existing data)
    viz_parser = subparsers.add_parser("visualize", help="Visualize existing path data")
    viz_parser.add_argument(
        "input", help="Path to input data file (.json or .csv)"
    )
    viz_parser.add_argument(
        "--output", "-o", default=None,
        help="Output image path (displays if not specified)"
    )
    viz_parser.add_argument(
        "--title", default="Player and Boss Paths",
        help="Plot title"
    )
    viz_parser.add_argument(
        "--background", default=None,
        help="Background image path"
    )
    viz_parser.add_argument(
        "--bounds", nargs=4, type=float, default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Image bounds for alignment"
    )
    viz_parser.add_argument(
        "--no-grid", action="store_true",
        help="Hide grid"
    )
    viz_parser.add_argument(
        "--no-gradient", action="store_true",
        help="Use solid colors instead of time gradient"
    )

    args = parser.parse_args()

    if args.command == "trace":
        try:
            with PathTracer(host=args.host, config_path=args.siphon_config) as tracer:
                path_data = tracer.trace_paths(
                    output_path=args.output,
                    sample_rate=args.sample_rate,
                    max_duration=args.duration,
                    stop_on_death=not args.no_stop_on_death,
                    show_live=not args.quiet,
                    live_plot=args.live_plot,
                )

                if args.visualize:
                    viz_output = Path(args.output).with_suffix(".png")
                    fig = visualize_paths(path_data, output_path=str(viz_output))
                    plt.show()

        except KeyboardInterrupt:
            console.print("\n[yellow]Aborted[/yellow]")
            return 130
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return 1

    elif args.command == "visualize":
        try:
            path_data = PathTracer.load_path_data(args.input)
            console.print(f"[cyan]Loaded {len(path_data.player_path)} samples[/cyan]")

            bounds = tuple(args.bounds) if args.bounds else None

            fig = visualize_paths(
                path_data,
                output_path=args.output,
                title=args.title,
                show_grid=not args.no_grid,
                show_time_gradient=not args.no_gradient,
                background_image=args.background,
                image_bounds=bounds,
            )

            if not args.output:
                plt.show()

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return 1
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
