# test_download.py
import yt_dlp
import os

def download_test_videos(num_videos=10):
    os.makedirs("data/videos/downloaded", exist_ok=True)
    
    ydl_opts = {
        'format': 'mp4[height=1080]/best[height=1080]',  # Prefer MP4
        'outtmpl': 'data/videos/downloaded/%(id)s.mp4',  # Force .mp4 extension
        'merge_output_format': 'mp4',
    }
    
    # Search for Margit fights
    query = "elden ring margit level 1 fight"
    search_url = f"ytsearch{num_videos}:{query}"
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([search_url])
    
    print(f"✓ Downloaded videos to data/videos/downloaded/")

if __name__ == "__main__":
    download_test_videos(10)