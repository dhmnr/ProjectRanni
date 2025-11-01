# validate_pipeline.py
import cv2
import numpy as np
from pathlib import Path
from hp_detection import detect_hp_bars

def validate_video(video_path, num_samples=10):
    """
    Validate HP detection on multiple frames from video
    
    Returns: success_rate (0.0-1.0)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    sample_frames = np.linspace(100, total_frames - 100, num_samples, dtype=int)
    
    successful_detections = 0
    
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        player_hp, boss_hp = detect_hp_bars(frame, debug=False)
        
        # Success if both HP values are reasonable (not 0, not 1)
        if 0.05 < player_hp < 1.0 and 0.05 < boss_hp < 1.0:
            successful_detections += 1
    
    cap.release()
    
    success_rate = successful_detections / num_samples
    return success_rate

def validate_all_videos():
    """Validate HP detection on all test videos"""
    
    video_files = list(Path("data/test_videos").glob("*.mp4"))
    
    if not video_files:
        print("❌ No videos found in data/test_videos/")
        return
    
    print(f"Validating {len(video_files)} videos...\n")
    
    results = []
    
    for video_path in video_files:
        print(f"Testing: {video_path.name}...")
        success_rate = validate_video(video_path, num_samples=10)
        results.append((video_path.name, success_rate))
        
        status = "✓" if success_rate >= 0.8 else "✗"
        print(f"  {status} Success rate: {success_rate:.0%}\n")
    
    # Summary
    print("=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, rate in results if rate >= 0.8)
    total = len(results)
    
    print(f"\nVideos passed (≥80% detection): {passed}/{total}")
    
    if passed >= 8:
        print("\n✓ MILESTONE 1 PASSED - Ready to scale to 200 videos")
    else:
        print("\n✗ MILESTONE 1 FAILED - Need to adjust HP detection")
        print("\nFailing videos:")
        for name, rate in results:
            if rate < 0.8:
                print(f"  - {name}: {rate:.0%}")

if __name__ == "__main__":
    validate_all_videos()