# hp_detection.py
import cv2
import numpy as np
from pathlib import Path

def detect_hp_bars(frame_bgr, debug=False):
    """
    Detect player and boss HP bars from frame
    
    Args:
        frame_bgr: OpenCV frame (BGR format, 0-255 uint8)
        debug: If True, show detection visualization
    
    Returns:
        (player_hp_ratio, boss_hp_ratio) or (None, None) if failed
    """
    
    # Frame is usually original size (before 84x84 resize)
    h, w = frame_bgr.shape[:2]
    
    # === BOSS HP BAR (top center) ===
    # Margit's HP is usually at top, ~20-30% from top, centered
    boss_y_start = int(h * 0.80)  # Near bottom
    boss_y_end = int(h * 0.815)
    boss_x_start = int(w * 0.24)  # Centered but wide
    boss_x_end = int(w * 0.76)
    
    boss_region = frame_bgr[boss_y_start:boss_y_end, boss_x_start:boss_x_end]
    
    # === PLAYER HP BAR (top left - red bar above equipment) ===
    player_y_start = int(h * 0.02)  # Very top
    player_y_end = int(h * 0.06)
    player_x_start = int(w * 0.08)  # Left edge
    player_x_end = int(w * 0.20)
    
    player_region = frame_bgr[player_y_start:player_y_end, player_x_start:player_x_end]
    
    # Detect HP from regions
    boss_hp = _detect_hp_from_region(boss_region)
    player_hp = _detect_hp_from_region(player_region)
    
    if debug:
        # Draw rectangles on frame
        debug_frame = frame_bgr.copy()
        cv2.rectangle(debug_frame, (boss_x_start, boss_y_start), (boss_x_end, boss_y_end), (0, 255, 0), 2)
        cv2.rectangle(debug_frame, (player_x_start, player_y_start), (player_x_end, player_y_end), (255, 0, 0), 2)
        
        # Add text
        cv2.putText(debug_frame, f"Boss: {boss_hp:.2f}", (boss_x_start, boss_y_start-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Player: {player_hp:.2f}", (player_x_start, player_y_start-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Show
        cv2.imshow("HP Detection", debug_frame)
        cv2.waitKey(0)
    
    return player_hp, boss_hp

def _detect_hp_from_region(region):
    """
    Detect HP ratio from a region by looking for red/yellow bar
    
    Returns: float 0.0-1.0, or 0.0 if detection failed
    """
    
    if region.size == 0:
        return 0.0
    
    # Convert BGR to HSV for color detection
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # HP bars in Elden Ring are typically red or yellow
    # Red hue: 0-10 or 170-180
    # Yellow hue: 20-30
    
    # Mask for red
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Mask for yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    mask = mask_red1 | mask_red2 | mask_yellow
    
    # Calculate filled ratio
    # HP ratio = horizontal extent of colored pixels / total width
    
    # Find horizontal extent
    horizontal_sum = np.sum(mask, axis=0)  # Sum vertically to get horizontal profile
    filled_columns = horizontal_sum > 0
    
    if np.any(filled_columns):
        # Find leftmost and rightmost filled columns
        filled_indices = np.where(filled_columns)[0]
        leftmost = filled_indices[0]
        rightmost = filled_indices[-1]
        filled_width = rightmost - leftmost + 1
        total_width = region.shape[1]
        hp_ratio = filled_width / total_width
    else:
        hp_ratio = 0.0
    
    return np.clip(hp_ratio, 0.0, 1.0)


# === TEST SCRIPT ===
if __name__ == "__main__":
    from pathlib import Path
    
    # Load a test video
    video_files = list(Path("data/test_videos").glob("*.mp4"))
    
    if not video_files:
        print("No videos found! Run test_download.py first")
        exit(1)
    
    video_path = video_files[0]
    print(f"Testing HP detection on: {video_path.name}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    # Test on frame 100 (skip intro)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    ret, frame = cap.read()
    
    if ret:
        player_hp, boss_hp = detect_hp_bars(frame, debug=True)
        print(f"\n✓ HP Detection:")
        print(f"  Player HP: {player_hp:.2%}")
        print(f"  Boss HP: {boss_hp:.2%}")
    
    cap.release()
    cv2.destroyAllWindows()