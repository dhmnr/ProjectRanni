# test_download.py
import yt_dlp
import os

def download_test_videos(num_videos=10):
    os.makedirs("data/test_videos", exist_ok=True)
    
    ydl_opts = {
        'format': 'mp4[height<=1080]/best[height<=1080]',  # Prefer MP4
        'outtmpl': 'data/videos/%(id)s.mp4',  # Force .mp4 extension
        'merge_output_format': 'mp4',
    }
    
    # Search for Margit fights
    query = "margit the fell omen boss fight melee no summons"
    search_url = f"ytsearch{num_videos}:{query}"
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([search_url])
    
    print(f"✓ Downloaded videos to data/test_videos/")
    