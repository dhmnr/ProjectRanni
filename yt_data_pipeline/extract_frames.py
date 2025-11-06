# extract_frames.py
import cv2
import numpy as np
from pathlib import Path

def extract_frames_from_video(video_path, target_fps=15, target_size=(84, 84)):
    """Extract frames at target FPS and resize"""
    
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / target_fps))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract every Nth frame to get target FPS
        if frame_count % frame_interval == 0:
            # Resize
            frame_resized = cv2.resize(frame, target_size)
            # Normalize to [0, 1]
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            frames.append(frame_normalized)
        
        frame_count += 1
    
    cap.release()
    return np.array(frames)

def test_extract_frames_from_video():
    video_files = list(Path("data/test_videos").glob("*.mp4"))
    if video_files:
        frames = extract_frames_from_video(video_files[0])
        print(f"✓ Extracted {len(frames)} frames from {video_files[0].name}")
        print(f"  Shape: {frames.shape}")  # Should be (N, 84, 84, 3)

