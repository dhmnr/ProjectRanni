# video_to_pickle.py
"""
Unified video processing pipeline: OCR boundary detection + HP extraction + frame downsampling.
Outputs a single pickle file containing all processed video data.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

# Import from our existing modules
from data_pipeline.trim_videos import find_boss_name_boundaries
from data_pipeline.hp_detection import detect_all_hp, BOSS_HP_COORDS, PLAYER_HP_COORDS

# =============================================================================
# CONFIGURATION
# =============================================================================

# Frame sampling: skip every N frames (60fps -> 30fps with skip=2)
FRAME_SKIP = 2

# Output frame resolution (width, height)
OUTPUT_RESOLUTION = (256, 144)

# Boss name detection sample interval (check every N frames for OCR)
BOSS_NAME_SAMPLE_INTERVAL = 30

# Boss name region coordinates (from trim_videos.py)
BOSS_NAME_COORDS = {
    'y_start': 0.775,
    'y_end': 0.80,
    'x_start': 0.24,
    'x_end': 0.355,
}

console = Console()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_debug_frame(frame_bgr: np.ndarray, detection_result: dict, video_id: str, 
                       frame_idx: int) -> np.ndarray:
    """
    Create an annotated debug frame with bounding boxes and HP information.
    
    Args:
        frame_bgr: Original frame (BGR format)
        detection_result: Result from detect_all_hp()
        video_id: Video identifier
        frame_idx: Sequential frame index
    
    Returns:
        Annotated frame
    """
    debug_frame = frame_bgr.copy()
    h, w = debug_frame.shape[:2]
    
    # Draw boss HP region
    boss = detection_result['boss']
    bx1, by1, bx2, by2 = boss['region_bounds']
    cv2.rectangle(debug_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    
    # Draw boss HP fill indicator
    if 'right_edge' in boss and boss['right_edge'] > 0:
        right_edge = boss['right_edge']
        cv2.rectangle(debug_frame, (bx1, by1), (bx1 + right_edge, by2), (255, 255, 0), 1)
        cv2.line(debug_frame, (bx1 + right_edge, by1 - 5), 
                (bx1 + right_edge, by2 + 5), (0, 255, 255), 3)
    
    # Draw player HP region
    player = detection_result['player']
    px1, py1, px2, py2 = player['region_bounds']
    cv2.rectangle(debug_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
    
    # Draw player HP fill indicator
    if 'right_edge' in player and player['right_edge'] > 0:
        right_edge = player['right_edge']
        cv2.rectangle(debug_frame, (px1, py1), (px1 + right_edge, py2), (255, 255, 0), 1)
        cv2.line(debug_frame, (px1 + right_edge, py1 - 5), 
                (px1 + right_edge, py2 + 5), (0, 255, 255), 3)
    
    # Draw boss name region
    ny_start = int(h * BOSS_NAME_COORDS['y_start'])
    ny_end = int(h * BOSS_NAME_COORDS['y_end'])
    nx_start = int(w * BOSS_NAME_COORDS['x_start'])
    nx_end = int(w * BOSS_NAME_COORDS['x_end'])
    cv2.rectangle(debug_frame, (nx_start, ny_start), (nx_end, ny_end), (255, 0, 255), 2)
    
    # Add text overlays
    cv2.putText(debug_frame, f"Boss HP: {boss['hp']:.1%}", (bx1, by2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(debug_frame, f"Player HP: {player['hp']:.1%}", (px1, py1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(debug_frame, f"{video_id} | Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return debug_frame


def save_pickle(data: dict, output_path: Path) -> None:
    """
    Save processed data to pickle file.
    
    Args:
        data: Dictionary containing processed video data
        output_path: Path to output pickle file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    console.print(f"[green]✓[/green] Saved pickle to: {output_path}")


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def process_single_video(
    video_path: Path,
    debug_dir: Optional[Path] = None,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None
) -> Dict[str, List]:
    """
    Process a single video: detect boundaries, extract frames, detect HP.
    
    Args:
        video_path: Path to video file
        debug_dir: Optional debug output directory
        progress: Rich progress bar instance
        task_id: Task ID for progress updates
    
    Returns:
        Dictionary with lists of video_ids, frame_indices, frames, boss_hps, player_hps
    """
    video_id = video_path.stem
    
    if debug_dir:
        console.print(f"\n[cyan]Processing: {video_id}[/cyan]")
    
    # Step 1: Find boss name boundaries
    if debug_dir:
        console.print(f"[yellow]→[/yellow] Finding boss name boundaries...")
    
    first_frame, last_frame = find_boss_name_boundaries(
        video_path,
        sample_interval=BOSS_NAME_SAMPLE_INTERVAL,
        debug=False
    )
    
    if first_frame is None or last_frame is None:
        console.print(f"[red]✗[/red] Boss name not found in {video_id}, skipping")
        return {
            'video_ids': [],
            'frame_indices': [],
            'frames': np.array([]),
            'boss_hps': [],
            'player_hps': []
        }
    
    if debug_dir:
        console.print(f"[green]✓[/green] Boundaries: frames {first_frame}-{last_frame}")
    
    # Step 2: Open video and process frames
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_to_process = (last_frame - first_frame) // FRAME_SKIP
    
    if debug_dir:
        console.print(f"[yellow]→[/yellow] Video resolution: {frame_width}x{frame_height}")
        console.print(f"[yellow]→[/yellow] Processing frames with skip={FRAME_SKIP}...")
        console.print(f"[yellow]→[/yellow] Expected frames: {total_frames_to_process}")
        console.print(f"[yellow]→[/yellow] HP detection: ON ORIGINAL RESOLUTION (before resize to {OUTPUT_RESOLUTION})")
    
    # Initialize result lists
    video_ids = []
    frame_indices = []
    frames = []
    boss_hps = []
    player_hps = []
    
    # Get player max HP from first valid frame (within boundaries)
    player_max_hp_extent = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    ret, first_frame_img = cap.read()
    if ret:
        first_result = detect_all_hp(first_frame_img, player_max_hp_extent=None)
        player_max_hp_extent = first_result['player']['right_edge']
        if debug_dir:
            console.print(f"[green]✓[/green] Player max HP extent: {player_max_hp_extent} pixels")
            console.print(f"[green]✓[/green] First frame boss HP: {first_result['boss']['hp']:.1%}")
            console.print(f"[green]✓[/green] First frame player HP: {first_result['player']['hp']:.1%}")
    
    # Create debug directory for this video if needed
    video_debug_dir = None
    if debug_dir:
        video_debug_dir = debug_dir / video_id
        video_debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Update progress bar total
    if progress and task_id is not None:
        progress.update(task_id, total=total_frames_to_process)
    
    # Process frames
    sequential_idx = 0
    current_frame_num = first_frame
    
    while current_frame_num <= last_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
        ret, frame = cap.read()
        
        if not ret:
            if debug_dir:
                console.print(f"[yellow]![/yellow] Failed to read frame {current_frame_num}")
            current_frame_num += FRAME_SKIP
            continue
        
        # STEP 1: Detect HP on ORIGINAL RESOLUTION frame (e.g., 1920x1080)
        # This ensures accurate HP bar pixel detection
        hp_result = detect_all_hp(frame, player_max_hp_extent)
        boss_hp = hp_result['boss']['hp']
        player_hp = hp_result['player']['hp']
        
        # STEP 2: Resize frame to output resolution for storage (256x144)
        # HP values are already extracted, this is just for visual data
        resized_frame = cv2.resize(frame, OUTPUT_RESOLUTION, interpolation=cv2.INTER_AREA)
        
        # STEP 3: Convert BGR to RGB for consistency with ML pipelines
        resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Append to lists
        video_ids.append(video_id)
        frame_indices.append(sequential_idx)
        frames.append(resized_frame_rgb)
        boss_hps.append(boss_hp)
        player_hps.append(player_hp)
        
        # Debug output
        if video_debug_dir:
            # Create annotated debug frame (on original resolution)
            debug_frame = create_debug_frame(frame, hp_result, video_id, sequential_idx)
            debug_path = video_debug_dir / f"frame_{sequential_idx:06d}.jpg"
            cv2.imwrite(str(debug_path), debug_frame)
            
            # Print every 50 frames for more visibility
            if sequential_idx % 50 == 0:
                console.print(
                    f"  Frame {sequential_idx} (orig frame {current_frame_num}): "
                    f"Boss={boss_hp:.1%} (edge={hp_result['boss']['right_edge']}px), "
                    f"Player={player_hp:.1%} (edge={hp_result['player']['right_edge']}px)"
                )
        
        # Update progress
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
        
        sequential_idx += 1
        current_frame_num += FRAME_SKIP
    
    cap.release()
    
    if debug_dir:
        console.print(f"[green]✓[/green] Processed {len(frames)} frames from {video_id}")
    
    return {
        'video_ids': video_ids,
        'frame_indices': frame_indices,
        'frames': np.array(frames, dtype=np.uint8),
        'boss_hps': boss_hps,
        'player_hps': player_hps
    }


def process_batch(
    input_path: Path,
    output_path: Path,
    debug: bool = False
) -> None:
    """
    Process a video file or folder of videos.
    
    Args:
        input_path: Path to video file or folder
        output_path: Output pickle file path
        debug: Enable debug mode
    """
    # Determine video files to process
    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        video_files = sorted(input_path.glob("*.mp4"))
    else:
        console.print(f"[red]Error:[/red] Invalid input path: {input_path}")
        return
    
    if not video_files:
        console.print(f"[red]Error:[/red] No video files found in {input_path}")
        return
    
    console.print(f"\n[bold cyan]Video to Pickle Pipeline[/bold cyan]")
    console.print(f"[cyan]━[/cyan]" * 60)
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_path}")
    console.print(f"Videos to process: {len(video_files)}")
    console.print(f"Frame skip: {FRAME_SKIP} (effective fps: ~{60/FRAME_SKIP:.0f})")
    console.print(f"Output resolution: {OUTPUT_RESOLUTION}")
    console.print(f"Debug mode: {'ON' if debug else 'OFF'}")
    console.print(f"[cyan]━[/cyan]" * 60 + "\n")
    
    # Setup debug directory
    debug_dir = None
    if debug:
        debug_dir = output_path.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]Debug output:[/yellow] {debug_dir}\n")
    
    # Initialize accumulator
    all_video_ids = []
    all_frame_indices = []
    all_frames = []
    all_boss_hps = []
    all_player_hps = []
    
    # Create progress bars
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Outer progress: videos
        video_task = progress.add_task(
            "[cyan]Processing videos...",
            total=len(video_files)
        )
        
        # Inner progress: frames (per video)
        frame_task = progress.add_task(
            "[green]Processing frames...",
            total=0,
            visible=False
        )
        
        for video_file in video_files:
            # Update video task description
            progress.update(
                video_task,
                description=f"[cyan]Processing: {video_file.name}"
            )
            
            # Reset and show frame task
            progress.reset(frame_task, visible=True)
            progress.update(
                frame_task,
                description=f"[green]Frames from {video_file.stem}",
                total=0
            )
            
            # Process video
            result = process_single_video(
                video_file,
                debug_dir=debug_dir,
                progress=progress,
                task_id=frame_task
            )
            
            # Accumulate results
            if len(result['video_ids']) > 0:
                all_video_ids.extend(result['video_ids'])
                all_frame_indices.extend(result['frame_indices'])
                all_frames.append(result['frames'])
                all_boss_hps.extend(result['boss_hps'])
                all_player_hps.extend(result['player_hps'])
            
            # Hide frame task and advance video task
            progress.update(frame_task, visible=False)
            progress.advance(video_task)
    
    # Combine all frames
    if all_frames:
        all_frames_array = np.concatenate(all_frames, axis=0)
    else:
        all_frames_array = np.array([])
        console.print("[red]Warning:[/red] No frames were processed!")
    
    # Create final data structure
    final_data = {
        'video_ids': all_video_ids,
        'frame_indices': all_frame_indices,
        'frames': all_frames_array,
        'boss_hps': all_boss_hps,
        'player_hps': all_player_hps
    }
    
    # Print summary
    console.print(f"\n[bold cyan]Processing Complete![/bold cyan]")
    console.print(f"[cyan]━[/cyan]" * 60)
    console.print(f"Total frames processed: {len(all_video_ids)}")
    console.print(f"Frames array shape: {all_frames_array.shape}")
    console.print(f"Videos included: {len(set(all_video_ids))}")
    console.print(f"[cyan]━[/cyan]" * 60 + "\n")
    
    # Save pickle
    save_pickle(final_data, output_path)
    
    if debug_dir:
        console.print(f"\n[yellow]Debug frames saved to:[/yellow] {debug_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process videos to extract frames and HP data into pickle format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video
  python -m data_pipeline.video_to_pickle path/to/video.mp4
  
  # Process all videos in a folder
  python -m data_pipeline.video_to_pickle data/videos/downloaded
  
  # Enable debug mode with visualization
  python -m data_pipeline.video_to_pickle data/videos/downloaded --debug
  
  # Specify custom output path
  python -m data_pipeline.video_to_pickle data/videos/downloaded --output data/output.pkl
        """
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to video file or folder containing videos'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (saves annotated frames, verbose output)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output pickle file path (default: derived from input)'
    )
    
    args = parser.parse_args()
    
    # Parse paths
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input path does not exist: {input_path}")
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if input_path.is_file():
            output_path = input_path.parent / f"{input_path.stem}_processed.pkl"
        else:
            output_path = input_path.parent / f"{input_path.name}_processed.pkl"
    
    # Run processing
    try:
        process_batch(input_path, output_path, debug=args.debug)
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise


if __name__ == "__main__":
    main()

