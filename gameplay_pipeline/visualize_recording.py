"""
Visualize gameplay recordings with overlays showing inputs and game state data.

This script takes recorded videos and CSV data files and adds overlays showing:
- Button presses
- Player health, stamina, FP
- Enemy health
- Position data
- Animation IDs
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import argparse


class RecordingVisualizer:
    """Visualize gameplay recordings with data overlays."""
    
    def __init__(self, recording_dir: str, output_path: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            recording_dir: Path to the recording directory (containing video.mp4, inputs.csv, memory_data.csv)
            output_path: Path for output video (default: <recording_dir>/visualized.mp4)
        """
        self.recording_dir = Path(recording_dir)
        
        if output_path is None:
            self.output_path = self.recording_dir / "visualized.mp4"
        else:
            self.output_path = Path(output_path)
        
        # Load files
        print(f"Loading recording from: {self.recording_dir}")
        
        # Load video
        video_path = self.recording_dir / "video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_capture = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Load inputs CSV (event-based format)
        inputs_path = self.recording_dir / "inputs.csv"
        if not inputs_path.exists():
            raise FileNotFoundError(f"Inputs file not found: {inputs_path}")
        
        self.inputs_df = pd.read_csv(inputs_path)
        
        # Filter out MOUSE_MOVE events, keep key and button presses
        self.inputs_df = self.inputs_df[
            self.inputs_df['event_type'].isin(['KEY_DOWN', 'KEY_UP', 'MOUSE_DOWN', 'MOUSE_UP'])
        ].copy()
        
        # Load memory data CSV
        memory_path = self.recording_dir / "memory_data.csv"
        if not memory_path.exists():
            raise FileNotFoundError(f"Memory data file not found: {memory_path}")
        
        self.memory_df = pd.read_csv(memory_path)
        # Get attribute columns (all columns except timestamp_us)
        self.attribute_names = [col for col in self.memory_df.columns if col != 'timestamp_us']
        
        # Get all unique keys/buttons from inputs (excluding "MOVE")
        key_events = self.inputs_df[self.inputs_df['key_or_button'] != 'MOVE']
        self.key_mapping = sorted(key_events['key_or_button'].unique().tolist())
        self.key_indices = list(range(len(self.key_mapping)))
        
        print(f"  Video: {video_path.name}")
        print(f"  Frames: {self.total_frames}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Keys tracked: {', '.join(self.key_mapping)}")
        print(f"  Attributes: {len(self.attribute_names)}")
        
        # Calculate duration
        if len(self.memory_df) > 0:
            self.start_timestamp_us = self.memory_df['timestamp_us'].iloc[0]
            duration_us = self.memory_df['timestamp_us'].iloc[-1] - self.start_timestamp_us
            print(f"  Duration: {duration_us/1000000:.2f}s")
        else:
            self.start_timestamp_us = 0
        
        # Both inputs and memory now use the same timestamp source
        # Just sort inputs by timestamp for efficient lookup
        if len(self.inputs_df) > 0:
            self.inputs_df = self.inputs_df.sort_values('timestamp_us').reset_index(drop=True)
    
    def _get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get a specific frame from the video.
        
        Args:
            frame_idx: Index of the frame to retrieve
            
        Returns:
            Frame as numpy array (BGR format)
        """
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_capture.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx}")
        return frame
    
    def _get_pressed_keys_at_time(self, timestamp_us: int) -> set:
        """
        Get the set of keys/buttons that are pressed at a given timestamp.
        
        SIMPLE LOGIC:
        1. Get all events that happened BEFORE this timestamp
        2. Go through them in order
        3. Track which keys are currently down
        4. Return the set of pressed keys
        
        Args:
            timestamp_us: Timestamp in microseconds
            
        Returns:
            Set of key/button names that are currently pressed
        """
        # Get all events up to and including this timestamp
        past_events = self.inputs_df[self.inputs_df['timestamp_us'] <= timestamp_us]
        
        # Track key state by going through events in order
        pressed_keys = set()
        
        for _, event in past_events.iterrows():
            key = event['key_or_button']
            event_type = event['event_type']
            
            if event_type in ['KEY_DOWN', 'MOUSE_DOWN']:
                # Key/button pressed down
                pressed_keys.add(key)
            elif event_type in ['KEY_UP', 'MOUSE_UP']:
                # Key/button released
                pressed_keys.discard(key)
        
        return pressed_keys
    
    def _get_memory_data_at_time(self, timestamp_us: int) -> dict:
        """
        Get the memory data closest to the given timestamp.
        
        Args:
            timestamp_us: Timestamp in microseconds
            
        Returns:
            Dictionary of attribute names to values
        """
        # Find the closest memory data entry (by timestamp)
        if len(self.memory_df) == 0:
            return {attr: 0 for attr in self.attribute_names}
        
        # Find closest timestamp
        idx = (self.memory_df['timestamp_us'] - timestamp_us).abs().idxmin()
        row = self.memory_df.loc[idx]
        
        return {attr: row[attr] for attr in self.attribute_names}
    
    def _draw_text_with_background(
        self, 
        img: np.ndarray, 
        text: str, 
        pos: Tuple[int, int],
        font_scale: float = 0.6,
        color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = 1,
        padding: int = 5
    ):
        """Draw text with a background rectangle."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = pos
        
        # Draw background rectangle
        cv2.rectangle(
            img,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
        )
        
        return text_height + baseline + padding * 2
    
    def _create_overlay_frame(self, frame_idx: int) -> np.ndarray:
        """
        Create a frame with overlays.
        
        Args:
            frame_idx: Index of the frame to process
            
        Returns:
            Frame with overlays as numpy array
        """
        # Get the frame from video
        frame = self._get_frame(frame_idx)
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Use the frame directly (no side panel)
        canvas = frame.copy()
        
        # Calculate timestamp for this frame
        frame_time_s = frame_idx / self.fps
        timestamp_us = self.start_timestamp_us + int(frame_time_s * 1000000)
        
        # Get current data at this timestamp
        pressed_keys = self._get_pressed_keys_at_time(timestamp_us)
        memory_data = self._get_memory_data_at_time(timestamp_us)
        timestamp = frame_time_s
        
        # === Draw Input Overlay (on game frame - TOP LEFT) ===
        input_y = 60
        input_x = 30
        
        self._draw_text_with_background(
            canvas, 
            f"Time: {timestamp:.2f}s", 
            (input_x, input_y),
            font_scale=1.8,
            color=(0, 255, 255),
            bg_color=(0, 0, 0),
            thickness=4,
            padding=12
        )
        
        input_y += 90
        
        # Draw ALL keys - make pressed ones MUCH more obvious
        for key_name in self.key_mapping:
            is_pressed = key_name in pressed_keys
            
            if is_pressed:
                # PRESSED: Large, bright green with thick border
                height_used = self._draw_text_with_background(
                    canvas,
                    f">>> {key_name} <<<",
                    (input_x, input_y),
                    font_scale=2.5,
                    color=(0, 255, 0),  # Bright green
                    bg_color=(0, 100, 0),  # Dark green background
                    thickness=5,
                    padding=15
                )
            else:
                # NOT PRESSED: Small, dark gray
                height_used = self._draw_text_with_background(
                    canvas,
                    key_name,
                    (input_x, input_y),
                    font_scale=0.9,
                    color=(80, 80, 80),  # Dark gray
                    bg_color=(0, 0, 0),
                    thickness=1,
                    padding=5
                )
            
            input_y += height_used + 8
        
        # === Draw Memory Data (on game frame - TOP RIGHT) ===
        data_x = width - 450  # Position from right side
        data_y = 60
        
        # Title
        height_used = self._draw_text_with_background(
            canvas,
            "GAME STATE",
            (data_x, data_y),
            font_scale=1.8,
            color=(255, 255, 0),
            bg_color=(0, 0, 0),
            thickness=4,
            padding=12
        )
        data_y += height_used + 15
        
        # Draw important stats
        for attr_name in self.attribute_names:
            value = memory_data.get(attr_name, 0)
            
            # Color code based on attribute type
            if 'Hp' in attr_name and 'Max' not in attr_name:
                color = (0, 100, 255)  # Red-ish for HP
            elif 'Sp' in attr_name and 'Max' not in attr_name:
                color = (0, 255, 0)  # Green for Stamina
            elif 'Fp' in attr_name and 'Max' not in attr_name:
                color = (255, 200, 0)  # Blue-ish for FP
            elif 'Npc' in attr_name:
                color = (200, 0, 255)  # Purple for NPC/Boss
            else:
                color = (200, 200, 200)  # Gray for others
            
            # Format value
            if 'Pos' in attr_name or 'Angle' in attr_name:
                text = f"{attr_name}: {value:.1f}"
            elif 'Id' in attr_name or 'Anim' in attr_name:
                text = f"{attr_name}: {int(value)}"
            else:
                text = f"{attr_name}: {value:.0f}"
            
            height_used = self._draw_text_with_background(
                canvas,
                text,
                (data_x, data_y),
                font_scale=1.0,
                color=color,
                bg_color=(0, 0, 0),
                thickness=2,
                padding=8
            )
            data_y += height_used + 5
        
        return canvas
    
    def generate_video(self, start_frame: int = 0, end_frame: Optional[int] = None):
        """
        Generate the visualization video.
        
        Args:
            start_frame: First frame to process
            end_frame: Last frame to process (None = all frames)
        """
        if end_frame is None:
            end_frame = self.total_frames
        
        print(f"\nGenerating video: {self.output_path}")
        print(f"  Processing frames {start_frame} to {end_frame}")
        
        # Get dimensions from first frame
        sample_frame = self._create_overlay_frame(start_frame)
        height, width = sample_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (width, height)
        )
        
        # Process each frame
        for i in range(start_frame, end_frame):
            if i % 10 == 0:
                progress = ((i - start_frame) / (end_frame - start_frame)) * 100
                print(f"  Progress: {progress:.1f}% (frame {i}/{end_frame})", end='\r')
            
            overlay_frame = self._create_overlay_frame(i)
            out.write(overlay_frame)
        
        out.release()
        print(f"\n✓ Video saved: {self.output_path}")
        print(f"  Duration: {(end_frame - start_frame) / self.fps:.2f}s")
        print(f"  Size: {self.output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    def preview_frame(self, frame_idx: int):
        """
        Show a preview of a single frame.
        
        Args:
            frame_idx: Index of frame to preview
        """
        overlay_frame = self._create_overlay_frame(frame_idx)
        
        # Resize if too large
        height, width = overlay_frame.shape[:2]
        if width > 1920:
            scale = 1920 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            overlay_frame = cv2.resize(overlay_frame, (new_width, new_height))
        
        cv2.imshow(f'Frame {frame_idx}', overlay_frame)
        print(f"\nShowing frame {frame_idx}. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def close(self):
        """Release the video capture."""
        if self.video_capture:
            self.video_capture.release()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize gameplay recordings with overlays'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to recording directory (containing video.mp4, inputs.csv, memory_data.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output video path (default: <input_dir>/visualized.mp4)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start frame (default: 0)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End frame (default: all frames)'
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=None,
        help='Preview a single frame instead of generating video'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = RecordingVisualizer(args.input, args.output)
    
    try:
        if args.preview is not None:
            # Preview mode
            viz.preview_frame(args.preview)
        else:
            # Generate video
            viz.generate_video(args.start, args.end)
    finally:
        viz.close()


if __name__ == '__main__':
    main()

