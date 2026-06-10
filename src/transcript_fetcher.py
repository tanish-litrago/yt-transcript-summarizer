"""
src/transcript_fetcher.py
─────────────────────────
Fetches the full transcript text from a YouTube video.
Updated for youtube-transcript-api v1.x (new instance-based API).
"""

import os
import sys
import re
import subprocess
import tempfile

from youtube_transcript_api import YouTubeTranscriptApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, WHISPER_MODEL


def extract_video_id(url: str) -> str:
    patterns = [r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})"]
    for pat in patterns:
        match = re.search(pat, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract a video ID from: {url}")


def _merge_transcript_segments(segments) -> str:
    lines = []
    for seg in segments:
        # New API returns FetchedTranscriptSnippet objects
        text = seg.text if hasattr(seg, 'text') else seg.get("text", "")
        if text.strip():
            lines.append(text.strip())
    return " ".join(lines)


def fetch_transcript(url: str, preferred_language: str = "en") -> dict:
    video_id = extract_video_id(url)
    print(f"[Fetcher] Video ID: {video_id}")

    # New instance-based API
    ytt_api = YouTubeTranscriptApi()

    try:
        # Try direct fetch first (simplest approach)
        try:
            fetched = ytt_api.fetch(video_id, languages=[preferred_language])
            full_text = _merge_transcript_segments(fetched.snippets)
            language = fetched.language_code
        except Exception:
            # Fallback: list all transcripts and pick first available
            transcript_list = ytt_api.list(video_id)
            transcript = None
            for t in transcript_list:
                transcript = t
                break
            if transcript is None:
                raise RuntimeError("No transcripts found.")
            fetched = transcript.fetch()
            full_text = _merge_transcript_segments(fetched.snippets)
            language = fetched.language_code

        if len(full_text.split()) < 20:
            raise ValueError("Transcript too short.")

        print(f"[Fetcher] Caption source: YouTube API  |  Language: {language}")
        return {
            "video_id"  : video_id,
            "transcript": full_text,
            "source"    : "youtube_api",
            "language"  : language,
        }

    except Exception as e:
        print(f"[Fetcher] YouTube captions not available ({e}). Falling back to Whisper ...")
        return _transcribe_with_whisper(url, video_id)


def _transcribe_with_whisper(url: str, video_id: str) -> dict:
    try:
        import whisper
    except ImportError:
        raise RuntimeError("openai-whisper is not installed. Run: pip install openai-whisper")

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = os.path.join(tmp_dir, f"{video_id}.mp3")

        print("[Fetcher] Downloading audio with yt-dlp ...")
        cmd = [
            "yt-dlp", "--extract-audio", "--audio-format", "mp3",
            "--audio-quality", "5", "--no-playlist",
            "-o", audio_path,
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

        print(f"[Fetcher] Transcribing with Whisper ({WHISPER_MODEL}) on {DEVICE.upper()} ...")
        model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
        result = model.transcribe(audio_path, fp16=(DEVICE == "cuda"))
        full_text = result["text"].strip()
        language  = result.get("language", "unknown")

    print(f"[Fetcher] Whisper done  |  Language: {language}")
    return {
        "video_id"  : video_id,
        "transcript": full_text,
        "source"    : "whisper",
        "language"  : language,
    }