# hp_detection.py
import cv2
import numpy as np
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Boss HP bar coordinates (as fractions of frame dimensions)
BOSS_HP_COORDS = {
    'y_start': 0.804,
    'y_end': 0.810,
    'x_start': 0.242,
    'x_end': 0.764,
}

# Player HP bar coordinates (as fractions of frame dimensions)
PLAYER_HP_COORDS = {
    'y_start': 0.047,
    'y_end': 0.053,
    'x_start': 0.08,
    'x_end': 0.20,  # Search region end
}


# =============================================================================
# BOSS HP DETECTION
# =============================================================================

def detect_boss_hp(frame_bgr):
    """
    Detect boss HP by scanning for red pixels.
    
    Boss HP bar has fixed coordinates, so we use the full bar width as reference.
    HP = rightmost red pixel position / total bar width
    
    Args:
        frame_bgr: Full frame (BGR format)
    
    Returns:
        dict with 'hp' (0.0-1.0), 'right_edge' (pixel position), 'region_bounds' (coordinates)
    """
    h, w = frame_bgr.shape[:2]
    
    # Extract boss HP region
    y_start = int(h * BOSS_HP_COORDS['y_start'])
    y_end = int(h * BOSS_HP_COORDS['y_end'])
    x_start = int(w * BOSS_HP_COORDS['x_start'])
    x_end = int(w * BOSS_HP_COORDS['x_end'])
    
    region = frame_bgr[y_start:y_end, x_start:x_end]
    
    if region.size == 0:
        return {'hp': 0.0, 'right_edge': 0, 'region_bounds': (x_start, y_start, x_end, y_end)}
    
    region_h, region_w = region.shape[:2]
    
    # Find red pixels (boss HP bar)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Red hue: 0-10 or 170-180
    mask_red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_mask = mask_red1 | mask_red2
    
    # Get horizontal profile of red pixels
    red_horizontal = np.sum(red_mask, axis=0)
    red_columns = red_horizontal > 0
    
    if not np.any(red_columns):
        return {'hp': 0.0, 'right_edge': 0, 'region_bounds': (x_start, y_start, x_end, y_end)}
    
    # Find rightmost red pixel (current HP extent)
    red_indices = np.where(red_columns)[0]
    right_edge = red_indices[-1]
    
    # Calculate HP: rightmost red pixel / total bar width
    hp_ratio = right_edge / region_w
    hp_ratio = np.clip(hp_ratio, 0.0, 1.0)
    
    return {
        'hp': hp_ratio,
        'right_edge': right_edge,
        'region_bounds': (x_start, y_start, x_end, y_end)
    }


# =============================================================================
# PLAYER HP DETECTION
# =============================================================================

def detect_player_hp(frame_bgr, max_hp_extent=None):
    """
    Detect player HP by scanning for red pixels.
    
    First frame: Find where red pixels end - this is max HP
    Subsequent frames: Current HP = current red extent / max red extent
    
    Args:
        frame_bgr: Full frame (BGR format)
        max_hp_extent: Max HP extent from first frame (int), or None for first frame
    
    Returns:
        dict with 'hp' (0.0-1.0), 'right_edge' (pixel position), 'region_bounds' (coordinates)
    """
    h, w = frame_bgr.shape[:2]
    
    # Extract player HP region
    y_start = int(h * PLAYER_HP_COORDS['y_start'])
    y_end = int(h * PLAYER_HP_COORDS['y_end'])
    x_start = int(w * PLAYER_HP_COORDS['x_start'])
    x_end = int(w * PLAYER_HP_COORDS['x_end'])
    
    region = frame_bgr[y_start:y_end, x_start:x_end]
    
    if region.size == 0:
        return {'hp': 0.0, 'right_edge': 0, 'region_bounds': (x_start, y_start, x_end, y_end)}
    
    # Find red pixels
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Red hue: 0-10 or 170-180
    mask_red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_mask = mask_red1 | mask_red2
    
    # Get horizontal profile of red pixels
    red_horizontal = np.sum(red_mask, axis=0)
    red_columns = red_horizontal > 0
    
    if not np.any(red_columns):
        return {'hp': 0.0, 'right_edge': 0, 'region_bounds': (x_start, y_start, x_end, y_end)}
    
    # Find rightmost red pixel (current HP extent)
    red_indices = np.where(red_columns)[0]
    current_right_edge = red_indices[-1]
    
    # First frame: This is max HP
    if max_hp_extent is None:
        return {
            'hp': 1.0,
            'right_edge': current_right_edge,
            'region_bounds': (x_start, y_start, x_end, y_end),
            'is_first_frame': True
        }
    
    # Subsequent frames: Calculate HP ratio
    if max_hp_extent <= 0:
        hp_ratio = 0.0
    else:
        hp_ratio = current_right_edge / max_hp_extent
        hp_ratio = np.clip(hp_ratio, 0.0, 1.0)
    
    return {
        'hp': hp_ratio,
        'right_edge': current_right_edge,
        'region_bounds': (x_start, y_start, x_end, y_end)
    }


# =============================================================================
# COMBINED DETECTION
# =============================================================================

def detect_all_hp(frame_bgr, player_max_hp_extent=None):
    """
    Detect both player and boss HP from a frame.
    
    Args:
        frame_bgr: Full frame (BGR format)
        player_max_hp_extent: Max player HP extent from first frame, or None
    
    Returns:
        dict with 'player' and 'boss' sub-dicts containing detection results
    """
    player_result = detect_player_hp(frame_bgr, player_max_hp_extent)
    boss_result = detect_boss_hp(frame_bgr)
    
    return {
        'player': player_result,
        'boss': boss_result
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_hp_detection(frame_bgr, detection_result):
    """
    Draw HP detection boxes and text on frame.
    
    Args:
        frame_bgr: Frame to draw on (will be modified)
        detection_result: Result from detect_all_hp()
    
    Returns:
        Modified frame
    """
    player = detection_result['player']
    boss = detection_result['boss']
    
    # Boss HP visualization
    bx1, by1, bx2, by2 = boss['region_bounds']
    
    # Green box: Full bar region
    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 255, 0), 1)
    
    # Cyan box: Filled portion (left edge to right edge of red pixels)
    if 'right_edge' in boss:
        right_edge = boss['right_edge']
        cv2.rectangle(frame_bgr, (bx1, by1), (bx1 + right_edge, by2), (255, 255, 0), 1)
        cv2.line(frame_bgr, (bx1 + right_edge, by1 - 5), (bx1 + right_edge, by2 + 5), (0, 255, 255), 3)
    
    # Boss HP text
    cv2.putText(frame_bgr, f"Boss: {boss['hp']:.1%}", (bx1, by2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Player HP visualization
    px1, py1, px2, py2 = player['region_bounds']
    
    # Green box: Search region
    cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), (0, 255, 0), 1)
    
    # Cyan box: Current HP extent (left edge to right edge of red pixels)
    if 'right_edge' in player:
        right_edge = player['right_edge']
        cv2.rectangle(frame_bgr, (px1, py1), (px1 + right_edge, py2), (255, 255, 0), 1)
        cv2.line(frame_bgr, (px1 + right_edge, py1 - 5), (px1 + right_edge, py2 + 5), (0, 255, 255), 3)
    
    # Player HP text
    cv2.putText(frame_bgr, f"Player: {player['hp']:.1%}", (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame_bgr


def print_detection_debug(detection_result, frame_num=None):
    """
    Print debug information about detection result.
    
    Args:
        detection_result: Result from detect_all_hp()
        frame_num: Optional frame number to display
    """
    player = detection_result['player']
    boss = detection_result['boss']
    
    if frame_num is not None:
        print(f"Frame {frame_num}:")
    
    print(f"  Boss HP: {boss['hp']:.1%} | Right edge: {boss.get('right_edge', 'N/A')}")
    print(f"  Player HP: {player['hp']:.1%} | Right edge: {player.get('right_edge', 'N/A')}")


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_hp_detection_interactive(video_path=None, n_frames=5, start_frame=100):
    """
    Test HP detection on multiple frames with visualization.
    
    Args:
        video_path: Path to video, or None to use first video in trimmed folder
        n_frames: Number of frames to test
        start_frame: Starting frame number
    """
    # Load video
    if video_path is None:
        video_files = list(Path("data/videos/trimmed").glob("*.mp4"))
        if not video_files:
            print("No videos found in data/videos/trimmed/")
            return
        video_path = video_files[9]
    else:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return
    
    print(f"Testing HP detection on: {video_path.name}\n")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total_frames} frames @ {fps:.1f}fps ({total_frames/fps:.1f}s)")
    print(f"Testing {n_frames} frames starting from frame {start_frame}")
    print("Press any key to advance, ESC to quit\n")
    
    # Generate frame numbers to test
    if n_frames == 1:
        test_frames = [start_frame]
    else:
        end_frame = total_frames - 1000
        frame_step = max(1, (end_frame - start_frame) // (n_frames - 1))
        test_frames = [start_frame + i * frame_step for i in range(n_frames)]
    
    # Get first frame to establish max player HP
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frames[0])
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        cap.release()
        return
    
    # Detect max player HP from first frame
    first_result = detect_player_hp(first_frame, max_hp_extent=None)
    player_max_hp_extent = first_result['right_edge']
    
    print(f"Max player HP extent: {player_max_hp_extent} pixels\n")
    print("="*60)
    
    # Process each test frame
    for i, frame_num in enumerate(test_frames):
        if frame_num >= total_frames:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Detect HP
        result = detect_all_hp(frame, player_max_hp_extent)
        
        # Print info
        print(f"\nFrame {frame_num} ({i+1}/{n_frames}) @ {frame_num/fps:.1f}s")
        print(f"  Boss: {result['boss']['hp']:.1%}")
        print(f"  Player: {result['player']['hp']:.1%}")
        
        # Draw and show
        draw_hp_detection(frame, result)
        cv2.imshow("HP Detection", frame)
        
        # Wait for key
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break
    
    print("\n" + "="*60)
    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_hp_detection_interactive(n_frames=5, start_frame=100)
