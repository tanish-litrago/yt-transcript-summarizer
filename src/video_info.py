"""
src/video_info.py
─────────────────
Fetches video metadata: title, thumbnail, channel, duration.
Uses yt-dlp which is already installed in the project.
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_video_info(url: str) -> dict:
    """
    Returns video metadata dict:
    {
        "title"      : str,
        "thumbnail"  : str,
        "channel"    : str,
        "duration"   : str,
        "view_count" : str,
        "video_id"   : str,
    }
    """
    try:
        import yt_dlp

        ydl_opts = {
            "quiet"        : True,
            "no_warnings"  : True,
            "skip_download": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        # Format duration
        duration_secs = info.get("duration", 0) or 0
        mins = duration_secs // 60
        secs = duration_secs % 60
        duration_str = f"{mins}:{secs:02d}"

        # Format view count
        views = info.get("view_count", 0) or 0
        if views >= 1_000_000:
            views_str = f"{views/1_000_000:.1f}M views"
        elif views >= 1_000:
            views_str = f"{views/1_000:.1f}K views"
        else:
            views_str = f"{views} views"

        # Best thumbnail
        thumbnails = info.get("thumbnails", [])
        thumbnail_url = ""
        if thumbnails:
            best = max(thumbnails, key=lambda t: (t.get("width") or 0))
            thumbnail_url = best.get("url", "")
        if not thumbnail_url:
            vid = info.get("id", "")
            thumbnail_url = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"

        return {
            "title"      : info.get("title", "YouTube Video"),
            "thumbnail"  : thumbnail_url,
            "channel"    : info.get("uploader", "Unknown Channel"),
            "duration"   : duration_str,
            "view_count" : views_str,
            "video_id"   : info.get("id", ""),
        }

    except Exception as e:
        print(f"[VideoInfo] Could not fetch metadata: {e}")
        video_id = ""
        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        if match:
            video_id = match.group(1)
        return {
            "title"      : "YouTube Video",
            "thumbnail"  : f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            "channel"    : "Unknown",
            "duration"   : "Unknown",
            "view_count" : "Unknown",
            "video_id"   : video_id,
        }
