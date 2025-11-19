"""
Convert gameplay recordings to framewise zarr format for training.

This script processes all recordings in a directory and creates a unified zarr dataset with:
- frames: Video frames [N, C, H, W] as uint8
- state: Game state values [N, num_attributes] as float32
- actions: Input actions [N, num_keys] as bool
- metadata: Key mappings, attribute names, etc.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import zarr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

console = Console()


def load_keybinds(keybinds_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load keybinds mapping from JSON file.
    
    Args:
        keybinds_path: Path to keybinds.json file
        
    Returns:
        Dict mapping raw keys to semantic action names
    """
    if keybinds_path is None:
        # Default to configs/keybinds.json
        keybinds_path = Path(__file__).parent / "configs" / "keybinds.json"
    else:
        keybinds_path = Path(keybinds_path)
    
    if not keybinds_path.exists():
        console.print(f"[yellow]Warning: Keybinds file not found at {keybinds_path}[/yellow]")
        console.print("[yellow]Using raw key names instead of semantic names[/yellow]")
        return {}
    
    try:
        with open(keybinds_path, 'r') as f:
            data = json.load(f)
            return data.get('keybinds', {})
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load keybinds: {e}[/yellow]")
        return {}


class RecordingToZarrConverter:
    """Convert gameplay recordings to zarr format."""
    
    def __init__(
        self, 
        recordings_dir: str, 
        output_zarr: str, 
        keybinds_path: Optional[str] = None,
        target_resolution: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the converter.
        
        Args:
            recordings_dir: Directory containing recording session folders
            output_zarr: Path to output zarr file
            keybinds_path: Path to keybinds.json file (optional)
            target_resolution: Target resolution (width, height) to resize frames (optional)
        """
        self.recordings_dir = Path(recordings_dir)
        self.output_zarr = Path(output_zarr)
        self.target_resolution = target_resolution
        
        # Load keybinds mapping
        self.keybinds = load_keybinds(keybinds_path)
        if self.keybinds:
            console.print(f"[cyan]Loaded {len(self.keybinds)} keybind mappings[/cyan]")
        
        if self.target_resolution:
            console.print(f"[cyan]Will resize frames to {target_resolution[0]}x{target_resolution[1]}[/cyan]")
        
        # Find all recording sessions
        self.sessions = self._find_sessions()
        console.print(f"[cyan]Found {len(self.sessions)} recording sessions[/cyan]")
        
    def _find_sessions(self) -> List[Path]:
        """Find all recording session directories."""
        sessions = []
        for item in self.recordings_dir.iterdir():
            if item.is_dir() and (item / "video.mp4").exists():
                sessions.append(item)
        return sorted(sessions)
    
    def _load_session_info(self, session_dir: Path) -> Dict:
        """
        Load session information without loading full data.
        
        Returns:
            Dict with keys, attributes, frame count, etc.
        """
        # Load inputs to get key names
        inputs_df = pd.read_csv(session_dir / "inputs.csv")
        inputs_df = inputs_df[
            inputs_df['event_type'].isin(['KEY_DOWN', 'KEY_UP', 'MOUSE_DOWN', 'MOUSE_UP'])
        ].copy()
        key_events = inputs_df[inputs_df['key_or_button'] != 'MOVE']
        raw_keys = sorted(key_events['key_or_button'].unique().tolist())
        
        # Convert to semantic action names if keybinds available
        if self.keybinds:
            keys = [self.keybinds.get(k, k) for k in raw_keys]
            # Also store the mapping
            raw_to_semantic = {raw: sem for raw, sem in zip(raw_keys, keys)}
        else:
            keys = raw_keys
            raw_to_semantic = {k: k for k in raw_keys}
        
        # Load memory data to get attribute names
        memory_df = pd.read_csv(session_dir / "memory_data.csv")
        attributes = [col for col in memory_df.columns if col != 'timestamp_us']
        
        # Get video info
        video_capture = cv2.VideoCapture(str(session_dir / "video.mp4"))
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_capture.release()
        
        return {
            'keys': keys,
            'raw_to_semantic': raw_to_semantic,
            'attributes': attributes,
            'frame_count': frame_count,
            'fps': fps,
            'frame_width': frame_width,
            'frame_height': frame_height,
        }
    
    def _convert_inputs_to_framewise(
        self, 
        inputs_df: pd.DataFrame, 
        keys: List[str],
        raw_to_semantic: Dict[str, str],
        start_timestamp_us: int,
        fps: float,
        num_frames: int
    ) -> np.ndarray:
        """
        Convert event-based inputs to framewise boolean array.
        
        Args:
            inputs_df: Input events dataframe
            keys: List of SEMANTIC key names in order
            raw_to_semantic: Mapping from raw keys to semantic names
            start_timestamp_us: Starting timestamp
            fps: Frames per second
            num_frames: Number of frames to generate
            
        Returns:
            Array of shape [num_frames, num_keys] with boolean values
        """
        actions = np.zeros((num_frames, len(keys)), dtype=bool)
        
        # Track key state using SEMANTIC names
        pressed_keys = set()
        
        # Sort events by timestamp
        inputs_df = inputs_df.sort_values('timestamp_us').reset_index(drop=True)
        event_idx = 0
        
        for frame_idx in range(num_frames):
            # Calculate timestamp for this frame
            frame_timestamp = start_timestamp_us + int((frame_idx / fps) * 1000000)
            
            # Process all events up to this frame's timestamp
            while event_idx < len(inputs_df):
                event = inputs_df.iloc[event_idx]
                if event['timestamp_us'] > frame_timestamp:
                    break
                
                raw_key = event['key_or_button']
                event_type = event['event_type']
                
                # Convert raw key to semantic name
                semantic_key = raw_to_semantic.get(raw_key, raw_key)
                
                if event_type in ['KEY_DOWN', 'MOUSE_DOWN']:
                    pressed_keys.add(semantic_key)
                elif event_type in ['KEY_UP', 'MOUSE_UP']:
                    pressed_keys.discard(semantic_key)
                
                event_idx += 1
            
            # Set action values for this frame
            for key_idx, key in enumerate(keys):
                actions[frame_idx, key_idx] = key in pressed_keys
        
        return actions
    
    def _convert_memory_to_framewise(
        self,
        memory_df: pd.DataFrame,
        attributes: List[str],
        start_timestamp_us: int,
        fps: float,
        num_frames: int
    ) -> np.ndarray:
        """
        Convert memory data to framewise array.
        
        Args:
            memory_df: Memory data dataframe
            attributes: List of attribute names in order
            start_timestamp_us: Starting timestamp
            fps: Frames per second
            num_frames: Number of frames to generate
            
        Returns:
            Array of shape [num_frames, num_attributes] with float32 values
        """
        state = np.zeros((num_frames, len(attributes)), dtype=np.float32)
        
        for frame_idx in range(num_frames):
            # Calculate timestamp for this frame
            frame_timestamp = start_timestamp_us + int((frame_idx / fps) * 1000000)
            
            # Find closest memory data entry
            idx = (memory_df['timestamp_us'] - frame_timestamp).abs().idxmin()
            row = memory_df.loc[idx]
            
            # Set state values for this frame
            for attr_idx, attr in enumerate(attributes):
                state[frame_idx, attr_idx] = row[attr]
        
        return state
    
    def _load_video_frames(self, video_path: Path, num_frames: int) -> np.ndarray:
        """
        Load video frames as numpy array.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to load
            
        Returns:
            Array of shape [num_frames, height, width, channels] as uint8
        """
        video_capture = cv2.VideoCapture(str(video_path))
        
        # Get original dimensions
        orig_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Determine output dimensions
        if self.target_resolution:
            target_width, target_height = self.target_resolution
        else:
            target_width, target_height = orig_width, orig_height
        
        frames = np.zeros((num_frames, target_height, target_width, 3), dtype=np.uint8)
        
        for i in range(num_frames):
            ret, frame = video_capture.read()
            if not ret:
                console.print(f"[yellow]Warning: Could only read {i}/{num_frames} frames[/yellow]")
                frames = frames[:i]
                break
            
            # Resize if needed
            if self.target_resolution and (orig_width != target_width or orig_height != target_height):
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            frames[i] = frame
        
        video_capture.release()
        
        # Convert to CHW format (channels first)
        frames = np.transpose(frames, (0, 3, 1, 2))
        
        return frames
    
    def convert_session(
        self, 
        session_idx: int, 
        session_dir: Path,
        zarr_root: zarr.Group,
        global_metadata: Dict
    ) -> bool:
        """
        Convert a single session to zarr format.
        
        Args:
            session_idx: Episode index
            session_dir: Path to session directory
            zarr_root: Zarr root group
            global_metadata: Global metadata dict to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            console.print(f"\n[bold cyan]Processing {session_dir.name}[/bold cyan]")
            
            # Load session info
            info = self._load_session_info(session_dir)
            
            # Update global metadata with keys, attributes, and mappings
            if 'keys' not in global_metadata:
                global_metadata['keys'] = info['keys']
            if 'attributes' not in global_metadata:
                global_metadata['attributes'] = info['attributes']
            if 'raw_to_semantic' not in global_metadata:
                global_metadata['raw_to_semantic'] = info['raw_to_semantic']
            
            # Load data
            console.print("  Loading inputs...")
            inputs_df = pd.read_csv(session_dir / "inputs.csv")
            inputs_df = inputs_df[
                inputs_df['event_type'].isin(['KEY_DOWN', 'KEY_UP', 'MOUSE_DOWN', 'MOUSE_UP'])
            ].copy()
            
            console.print("  Loading memory data...")
            memory_df = pd.read_csv(session_dir / "memory_data.csv")
            start_timestamp_us = memory_df['timestamp_us'].iloc[0]
            
            # Convert to framewise
            console.print("  Converting actions to framewise...")
            actions = self._convert_inputs_to_framewise(
                inputs_df,
                info['keys'],
                info['raw_to_semantic'],
                start_timestamp_us,
                info['fps'],
                info['frame_count']
            )
            
            console.print("  Converting state to framewise...")
            state = self._convert_memory_to_framewise(
                memory_df,
                info['attributes'],
                start_timestamp_us,
                info['fps'],
                info['frame_count']
            )
            
            console.print("  Loading video frames...")
            frames = self._load_video_frames(
                session_dir / "video.mp4",
                info['frame_count']
            )
            
            # Create episode group
            episode_name = f"episode_{session_idx}"
            episode_group = zarr_root.create_group(episode_name)
            
            # Store data with compression
            console.print(f"  Storing to zarr as {episode_name}...")
            episode_group.array(
                'frames',
                frames,
                chunks=(1, 3, frames.shape[2], frames.shape[3]),
                dtype='uint8',
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
            
            episode_group.array(
                'state',
                state,
                chunks=(100, len(info['attributes'])),
                dtype='float32',
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
            
            episode_group.array(
                'actions',
                actions,
                chunks=(100, len(info['keys'])),
                dtype='bool',
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
            
            # Store episode metadata
            episode_group.attrs['session_name'] = session_dir.name
            episode_group.attrs['fps'] = info['fps']
            episode_group.attrs['frame_count'] = info['frame_count']
            episode_group.attrs['original_resolution'] = [info['frame_height'], info['frame_width']]
            episode_group.attrs['stored_resolution'] = [frames.shape[2], frames.shape[3]]
            episode_group.attrs['action_keys'] = info['keys']
            episode_group.attrs['state_attributes'] = info['attributes']
            
            console.print(f"[green]✓ Converted {info['frame_count']} frames[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Failed to convert {session_dir.name}: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def convert_all(self):
        """Convert all sessions to zarr format."""
        console.print(f"\n[bold cyan]Converting recordings to zarr format[/bold cyan]")
        console.print(f"  Output: {self.output_zarr}")
        
        # Ensure output directory exists and is clean
        if self.output_zarr.exists():
            import shutil
            console.print(f"[yellow]Removing existing {self.output_zarr}[/yellow]")
            shutil.rmtree(self.output_zarr)
        
        # Create zarr root (zarr will create the directory)
        zarr_root = zarr.open_group(str(self.output_zarr), mode='w')
        
        # Global metadata
        global_metadata = {
            'version': '1.0',
            'format': 'framewise',
            'source': str(self.recordings_dir),
        }
        
        if self.target_resolution:
            global_metadata['target_resolution'] = list(self.target_resolution)
        
        # Convert each session
        successful = 0
        failed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Converting sessions...",
                total=len(self.sessions)
            )
            
            for idx, session_dir in enumerate(self.sessions):
                if self.convert_session(idx, session_dir, zarr_root, global_metadata):
                    successful += 1
                else:
                    failed += 1
                
                progress.update(task, advance=1)
        
        # Store global metadata
        zarr_root.attrs.update(global_metadata)
        
        console.print(f"\n[bold green]✓ Conversion complete![/bold green]")
        console.print(f"  Successful: {successful}")
        console.print(f"  Failed: {failed}")
        console.print(f"  Dataset: {self.output_zarr}")
        
        # Print dataset info
        self._print_dataset_info(zarr_root)
    
    def _print_dataset_info(self, zarr_root: zarr.Group):
        """Print information about the created dataset."""
        console.print("\n[bold cyan]Dataset Structure:[/bold cyan]")
        
        num_episodes = len([k for k in zarr_root.keys() if k.startswith('episode_')])
        console.print(f"  Episodes: {num_episodes}")
        
        if num_episodes > 0:
            ep0 = zarr_root['episode_0']
            console.print(f"  Frames shape: {ep0['frames'].shape} ({ep0['frames'].dtype})")
            console.print(f"  State shape: {ep0['state'].shape} ({ep0['state'].dtype})")
            console.print(f"  Actions shape: {ep0['actions'].shape} ({ep0['actions'].dtype})")
            
            # Show resolution info
            orig_res = ep0.attrs.get('original_resolution', 'N/A')
            stored_res = ep0.attrs.get('stored_resolution', 'N/A')
            if orig_res != 'N/A' and stored_res != 'N/A':
                console.print(f"  Original resolution: {orig_res[0]}x{orig_res[1]}")
                console.print(f"  Stored resolution: {stored_res[0]}x{stored_res[1]}")
        
        # Print metadata
        if 'keys' in zarr_root.attrs:
            console.print(f"\n  Action keys ({len(zarr_root.attrs['keys'])}):")
            console.print(f"    {', '.join(zarr_root.attrs['keys'])}")
            
            # Show raw -> semantic mapping if available
            if 'raw_to_semantic' in zarr_root.attrs:
                console.print(f"\n  Key Mappings (raw -> semantic):")
                mapping = zarr_root.attrs['raw_to_semantic']
                for raw, semantic in sorted(mapping.items()):
                    if raw != semantic:  # Only show if different
                        console.print(f"    {raw:15s} → {semantic}")
        
        if 'attributes' in zarr_root.attrs:
            console.print(f"\n  State attributes ({len(zarr_root.attrs['attributes'])}):")
            for i in range(0, len(zarr_root.attrs['attributes']), 5):
                attrs = zarr_root.attrs['attributes'][i:i+5]
                console.print(f"    {', '.join(attrs)}")
        
        # Calculate total size
        try:
            total_size = sum(
                f.stat().st_size
                for f in self.output_zarr.rglob('*')
                if f.is_file()
            )
            console.print(f"\n  Total size: {total_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            console.print(f"\n  [yellow]Could not calculate size: {e}[/yellow]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert gameplay recordings to framewise zarr format'
    )
    parser.add_argument(
        'recordings_dir',
        type=str,
        help='Directory containing recording session folders'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='dataset.zarr',
        help='Output zarr path (default: dataset.zarr)'
    )
    parser.add_argument(
        '--keybinds',
        type=str,
        default=None,
        help='Path to keybinds.json file (default: configs/keybinds.json)'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default=None,
        help='Target resolution as WIDTHxHEIGHT (e.g., 1920x1080, 1280x720). If not specified, uses original resolution.'
    )
    
    args = parser.parse_args()
    
    # Parse resolution
    target_resolution = None
    if args.resolution:
        try:
            width, height = args.resolution.lower().split('x')
            target_resolution = (int(width), int(height))
        except ValueError:
            console.print(f"[red]Error: Invalid resolution format '{args.resolution}'. Use WIDTHxHEIGHT (e.g., 1920x1080)[/red]")
            return 1
    
    try:
        converter = RecordingToZarrConverter(
            args.recordings_dir, 
            args.output, 
            args.keybinds,
            target_resolution
        )
        converter.convert_all()
    except KeyboardInterrupt:
        console.print("\n[yellow]Conversion interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

