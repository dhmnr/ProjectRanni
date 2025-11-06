# trim_videos.py
import cv2
import numpy as np
from pathlib import Path
import subprocess
from typing import Optional, Tuple
import easyocr

# Initialize EasyOCR reader (only once, globally)
_easyocr_reader = None

def get_ocr_reader():
    """Get or create the EasyOCR reader instance"""
    global _easyocr_reader
    if _easyocr_reader is None:
        print("Initializing EasyOCR (this may take a moment on first run)...")
        # Try to use GPU, fall back to CPU if not available
        try:
            import torch
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                print("Using GPU for EasyOCR")
            else:
                print("GPU not available, using CPU for EasyOCR")
        except ImportError:
            use_gpu = False
            print("PyTorch not found, using CPU for EasyOCR")
        
        _easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
    return _easyocr_reader

def detect_boss_name_in_frame(frame_bgr, debug=False, show_window=False):
    """
    Detect if a boss name is present in the frame using OCR
    
    Args:
        frame_bgr: OpenCV frame (BGR format, 0-255 uint8)
        debug: If True, print debug information
        show_window: If True, show detection visualization window
    
    Returns:
        (bool, str, dict): (is_boss_name_present, detected_text, debug_info)
    """
    h, w = frame_bgr.shape[:2]
    
    # Boss name typically appears in top-center of screen
    # Usually around 10-20% from top, centered horizontally
    name_y_start = int(h * 0.775)
    name_y_end = int(h * 0.80)
    name_x_start = int(w * 0.24)
    name_x_end = int(w * 0.355)
    
    name_region = frame_bgr[name_y_start:name_y_end, name_x_start:name_x_end]
    
    debug_info = {
        'region_coords': (name_x_start, name_y_start, name_x_end, name_y_end),
        'region': name_region.copy() if name_region.size > 0 else None
    }
    
    if name_region.size == 0:
        return False, "", debug_info
    
    # Preprocess for better OCR
    # Convert to grayscale
    gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
    
    # Try multiple preprocessing methods since background varies (dark vs light)
    scale_factor = 2
    h_region, w_region = gray.shape
    
    # Method 1: Original grayscale upscaled (for white text on dark background)
    upscaled_original = cv2.resize(gray, (w_region * scale_factor, h_region * scale_factor), 
                                  interpolation=cv2.INTER_CUBIC)
    upscaled_original = cv2.GaussianBlur(upscaled_original, (3, 3), 0)
    
    # Method 2: Inverted (for potential dark text on light background)
    inverted = cv2.bitwise_not(gray)
    upscaled_inverted = cv2.resize(inverted, (w_region * scale_factor, h_region * scale_factor), 
                                  interpolation=cv2.INTER_CUBIC)
    upscaled_inverted = cv2.GaussianBlur(upscaled_inverted, (3, 3), 0)
    
    debug_info['gray'] = gray
    debug_info['inverted'] = inverted
    debug_info['processed_original'] = upscaled_original
    debug_info['processed_inverted'] = upscaled_inverted
    
    if show_window:
        # Create visualization with frame and detected region
        debug_frame = frame_bgr.copy()
        cv2.rectangle(debug_frame, (name_x_start, name_y_start), (name_x_end, name_y_end), (0, 255, 0), 2)
        cv2.imshow("Boss Name Detection", debug_frame)
        cv2.imshow("Boss Name Region", name_region)
        cv2.imshow("Boss Name Grayscale", gray)
        cv2.imshow("Boss Name Processed - Original", upscaled_original)
        cv2.imshow("Boss Name Processed - Inverted", upscaled_inverted)
    
    # Run OCR with EasyOCR
    best_text = ""
    best_confidence = 0.0
    
    try:
        reader = get_ocr_reader()
        
        # EasyOCR works better on the original region (it handles varying backgrounds)
        # Convert BGR to RGB for EasyOCR
        region_rgb = cv2.cvtColor(name_region, cv2.COLOR_BGR2RGB)
        
        # Run EasyOCR - returns list of (bbox, text, confidence)
        results = reader.readtext(region_rgb)
        
        if debug:
            print(f"  [DEBUG] EasyOCR found {len(results)} text regions")
        
        # Combine all detected text and find the best one
        all_texts = []
        for (bbox, text, confidence) in results:
            if debug:
                print(f"  [DEBUG] OCR detected: '{text}' (confidence: {confidence:.2f})")
            all_texts.append((text, confidence))
            
            # Keep track of best result
            if confidence > best_confidence:
                best_text = text
                best_confidence = confidence
        
        # If multiple texts detected, try to combine them (boss names might be split)
        if len(all_texts) > 1:
            combined_text = ' '.join([t for t, _ in all_texts])
            if debug:
                print(f"  [DEBUG] Combined text: '{combined_text}'")
            # Use combined text if it looks more complete
            if len(combined_text) > len(best_text):
                best_text = combined_text
        
        debug_info['ocr_results'] = all_texts
        debug_info['ocr_text'] = best_text
        debug_info['confidence'] = best_confidence
        
        # Check if text looks like a boss name
        # Boss names are typically:
        # - At least a few characters long
        # - Contain letters
        # - Common boss names: "Margit", "Godrick", "Radahn", etc.
        # - Confidence threshold (EasyOCR gives confidence 0-1)
        if len(best_text) >= 3 and any(c.isalpha() for c in best_text) and best_confidence > 0.1:
            return True, best_text, debug_info
        
    except Exception as e:
        if debug:
            print(f"  [DEBUG] OCR error: {e}")
        debug_info['error'] = str(e)
    
    return False, best_text, debug_info


def find_boss_name_boundaries(video_path, sample_interval=30, debug=False, expected_boss_name=None):
    """
    Find the first and last frames where a boss name appears
    
    Args:
        video_path: Path to the video file
        sample_interval: Check every Nth frame (default: 30, i.e., ~1 second at 30fps)
        debug: If True, show detection visualization
        expected_boss_name: Optional boss name to look for (case-insensitive partial match)
    
    Returns:
        (first_frame, last_frame) or (None, None) if not found
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return None, None
    
    print(f"Analyzing: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames} ({total_frames/fps:.1f}s @ {fps:.1f} fps)")
    print(f"Sampling every {sample_interval} frames...")
    
    first_frame = None
    last_frame = None
    boss_name_detected = None
    
    # === PHASE 1: Find first occurrence (scan from start) ===
    print("\n[PHASE 1] Scanning from start to find first occurrence...")
    frame_num = 0
    
    while frame_num < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        is_present, text, _ = detect_boss_name_in_frame(frame, debug=debug)
        
        if is_present:
            # Check if it matches expected boss name (if provided)
            if expected_boss_name is None or expected_boss_name.lower() in text.lower():
                first_frame = frame_num
                boss_name_detected = text
                print(f"  ✓ Boss name found at frame {frame_num} ({frame_num/fps:.1f}s): '{text}'")
                break
        
        if frame_num % (sample_interval * 10) == 0:
            print(f"  Checked up to frame {frame_num} ({frame_num/fps:.1f}s)...")
        
        frame_num += sample_interval
    
    if first_frame is None:
        print("  ✗ Boss name not found in video")
        cap.release()
        cv2.destroyAllWindows()
        return None, None
    
    # === PHASE 2: Find last occurrence (scan from end backwards) ===
    print("\n[PHASE 2] Scanning from end to find last occurrence...")
    frame_num = total_frames - 1
    
    while frame_num > first_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            frame_num -= sample_interval
            continue
        
        is_present, text, _ = detect_boss_name_in_frame(frame, debug=debug)
        
        if is_present:
            # Check if it matches the boss name we found earlier
            if expected_boss_name is None or expected_boss_name.lower() in text.lower():
                last_frame = frame_num
                print(f"  ✓ Last occurrence at frame {frame_num} ({frame_num/fps:.1f}s): '{text}'")
                break
        
        if frame_num % (sample_interval * 10) == 0:
            print(f"  Checked down to frame {frame_num} ({frame_num/fps:.1f}s)...")
        
        frame_num -= sample_interval
    
    if last_frame is None:
        # Boss name found only once or near the start
        last_frame = total_frames - 1
        print(f"  → Using end of video as last frame: {last_frame}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Add small buffer (1 second before and after)
    buffer_frames = int(fps)
    first_frame = max(0, first_frame - buffer_frames)
    last_frame = min(total_frames - 1, last_frame + buffer_frames)
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Boss name: '{boss_name_detected}'")
    print(f"  First frame: {first_frame} ({first_frame/fps:.1f}s)")
    print(f"  Last frame: {last_frame} ({last_frame/fps:.1f}s)")
    print(f"  Duration: {(last_frame - first_frame)/fps:.1f}s")
    print(f"{'='*60}\n")
    
    return first_frame, last_frame


def trim_video_ffmpeg(input_path, output_path, start_frame, end_frame, fps):
    """
    Trim video using ffmpeg
    
    Args:
        input_path: Input video path
        output_path: Output video path
        start_frame: Starting frame number
        end_frame: Ending frame number
        fps: Video frame rate
    """
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    
    print(f"Trimming video with ffmpeg...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Start: {start_time:.2f}s")
    print(f"  Duration: {duration:.2f}s")
    
    # Use GPU encoding (NVENC) for much faster processing
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-hwaccel", "cuda",  # Use CUDA hardware acceleration
        "-hwaccel_output_format", "cuda",  # Keep frames on GPU
        "-i", str(input_path),
        "-ss", str(start_time),  # Seek to start time
        "-t", str(duration),  # Duration
        "-c:v", "h264_nvenc",  # NVIDIA GPU encoding
        "-preset", "p4",  # NVENC preset (p1=fastest, p7=slowest/best quality)
        "-cq", "23",  # Quality (similar to CRF, lower = better)
        "-c:a", "aac",  # Re-encode audio with AAC
        "-b:a", "192k",  # Audio bitrate
        "-avoid_negative_ts", "make_zero",  # Fix timestamp issues
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Video trimmed successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ FFmpeg error: {e.stderr.decode()}")
        raise


def trim_video_by_boss_name(input_path, output_path=None, sample_interval=30, 
                             debug=False, expected_boss_name=None):
    """
    Main function to trim a video based on boss name detection
    
    Args:
        input_path: Path to input video
        output_path: Path to output video (default: input_path with '_trimmed' suffix)
        sample_interval: Frame sampling interval for detection
        debug: Enable debug visualization
        expected_boss_name: Optional expected boss name to look for
    
    Returns:
        Path to trimmed video or None if failed
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_trimmed{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Find boss name boundaries
    first_frame, last_frame = find_boss_name_boundaries(
        input_path, 
        sample_interval=sample_interval,
        debug=debug,
        expected_boss_name=expected_boss_name
    )
    
    if first_frame is None or last_frame is None:
        print("Could not detect boss name boundaries")
        return None
    
    # Get FPS
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Trim video
    trim_video_ffmpeg(input_path, output_path, first_frame, last_frame, fps)
    
    return output_path


def batch_trim_videos(input_dir, output_dir=None, sample_interval=30, expected_boss_name=None):
    """
    Trim all videos in a directory
    
    Args:
        input_dir: Directory containing videos
        output_dir: Output directory (default: input_dir/trimmed)
        sample_interval: Frame sampling interval
        expected_boss_name: Optional expected boss name
    """
    input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir / "trimmed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    video_files = list(input_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos to process\n")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(video_files)}: {video_path.name}")
        print(f"{'='*60}")
        
        output_path = output_dir / video_path.name
        
        try:
            trim_video_by_boss_name(
                video_path,
                output_path,
                sample_interval=sample_interval,
                expected_boss_name=expected_boss_name
            )
        except Exception as e:
            print(f"✗ Failed to process {video_path.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def test_boss_name_detection(n_frames=10, video_path=None, start_frame=1000):
    """
    Test boss name OCR detection on multiple frames from a video
    
    Args:
        n_frames: Number of equidistant frames to test
        video_path: Path to specific video file (default: first video in data/videos)
        start_frame: Starting frame number for testing
    """
    from pathlib import Path
    
    # Load video
    if video_path is None:
        video_files = list(Path("data/videos/downloaded").glob("*.mp4"))
        if not video_files:
            print("No videos found! Run download_videos.py first")
            return
        video_path = video_files[4]
    else:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return
    
    print(f"Testing Boss Name OCR detection on: {video_path.name}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames} ({total_frames/fps:.1f}s @ {fps:.1f} fps)")
    print(f"Testing {n_frames} equidistant frames starting from frame {start_frame}")
    print("Press any key to advance to next frame, ESC to quit\n")
    
    # Generate equidistant frame numbers
    if n_frames == 1:
        test_frames = [start_frame]
    else:
        end_frame = total_frames - 1000  # Leave buffer at end
        frame_step = max(1, (end_frame - start_frame) // (n_frames - 1))
        test_frames = [start_frame + i * frame_step for i in range(n_frames)]
    
    for i, frame_num in enumerate(test_frames):
        if frame_num >= total_frames:
            print(f"Frame {frame_num} exceeds video length, stopping")
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            print(f"\n{'='*60}")
            print(f"Frame {frame_num} ({i+1}/{n_frames}) [{frame_num/fps:.1f}s]")
            print(f"{'='*60}")
            
            # Detect boss name with visualization
            is_present, text, debug_info = detect_boss_name_in_frame(frame, debug=True, show_window=True)
            
            # Add text overlay on the main detection window
            if 'region_coords' in debug_info:
                x1, y1, x2, y2 = debug_info['region_coords']
                debug_frame = frame.copy()
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add OCR result text
                result_text = f"OCR: '{text}'" if text else "OCR: (no text)"
                cv2.putText(debug_frame, result_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                status_text = "BOSS NAME DETECTED" if is_present else "No boss name"
                color = (0, 255, 0) if is_present else (0, 0, 255)
                cv2.putText(debug_frame, status_text, (x1, y2+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Boss Name Detection", debug_frame)
            
            print(f"\n{'='*60}")
            print(f"RESULTS:")
            print(f"  Boss name detected: {is_present}")
            print(f"  OCR text: '{text}'")
            if 'confidence' in debug_info:
                print(f"  Confidence: {debug_info['confidence']:.2f}")
            if 'error' in debug_info:
                print(f"  Error: {debug_info['error']}")
            print(f"{'='*60}\n")
            
            # Wait for key press
            key = cv2.waitKey(0)
            if key == 27:  # ESC key
                print("Exiting...")
                break
    
    cap.release()
    cv2.destroyAllWindows()


# === TEST SCRIPT ===
if __name__ == "__main__":
    import sys
    
    # To run OCR test: python -m data_pipeline.trim_videos --test
    if "--test" in sys.argv:
        test_boss_name_detection(n_frames=10, start_frame=1000)
    elif len(sys.argv) > 1 and sys.argv[1] != "--test":
        # Trim specific video
        video_path = sys.argv[1]
        trim_video_by_boss_name(video_path, sample_interval=30, debug=False)
    else:
        # Batch process all videos in data/videos
        print("Starting batch video trimming...")
        batch_trim_videos("data/videos/downloaded", output_dir="data/videos/trimmed", sample_interval=10)

