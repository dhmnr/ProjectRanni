import subprocess
from pathlib import Path

def remux_to_mp4(input_path, output_path):
    """
    Convert MPEGTS to MP4 (no re-encoding, just container change)
    Fast: ~10 seconds per video
    """
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c', 'copy',  # Copy streams without re-encoding
        '-y',  # Overwrite
        str(output_path)
    ]
    
    subprocess.run(cmd, capture_output=True)

# Convert all videos
video_dir = Path('data/test_videos')
output_dir = Path('data/test_videos_mp4')
output_dir.mkdir(exist_ok=True)

for video in video_dir.glob('*.mp4'):
    output = output_dir / video.name
    print(f"Converting {video.name}...")
    remux_to_mp4(video, output)