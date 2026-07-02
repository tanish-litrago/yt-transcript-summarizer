"""
app.py  —  v1.1
────────────────
Flask Web UI for YouTube Transcript Summarizer
New in v1.1:
  - Video title + thumbnail
  - GPU vs CPU speed comparison
  - Processing history
  - Better error messages
Run: python app.py  →  http://localhost:5000
"""

import os
import sys
import threading
import uuid
import time
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transcript_fetcher import fetch_transcript
from src.gemma_engine        import summarize_transcript, extract_keywords, check_ollama_ready
from src.note_generator      import generate_notes
from src.exporter            import export_all
from src.video_info          import get_video_info
from src.analyzer            import run_all as run_analytics
from config                  import OUTPUT_DIR, DEVICE, GEMMA_MODEL

import torch

app   = Flask(__name__)
jobs  = {}   # active jobs

# ── History (stored in outputs/history.json) ──────────────────────────────────
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.json")

def load_history() -> list:
    if os.path.isfile(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(entry: dict):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]   # keep last 50
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(job_id: str, url: str, formats: list):
    try:
        t_total = time.time()

        def update(progress, message):
            jobs[job_id]["progress"] = progress
            jobs[job_id]["message"]  = message

        # 1. Video info + transcript
        update(5, "Fetching video info...")
        video_info = get_video_info(url)

        update(15, "Fetching transcript...")
        t0 = time.time()
        transcript_data = fetch_transcript(url)
        transcript      = transcript_data["transcript"]
        video_id        = transcript_data["video_id"]
        word_count      = len(transcript.split())
        fetch_time      = round(time.time() - t0, 1)
        update(28, f"Transcript fetched — {word_count:,} words in {fetch_time}s")

        # 2. Summarize on GPU
        update(32, f"Summarising with {GEMMA_MODEL} (Gemma 4)...")
        t0 = time.time()
        summary_data = summarize_transcript(transcript)
        gpu_time     = round(time.time() - t0, 1)

        # Estimate CPU time (roughly 10-12x slower)
        cpu_estimate = round(gpu_time * 11, 1)
        speedup      = round(cpu_estimate / max(gpu_time, 0.1), 1)
        update(70, f"Summarised in {gpu_time}s on GPU (vs ~{cpu_estimate}s on CPU — {speedup}x faster!)")

        # 3. Keywords
        update(75, "Extracting keywords and named entities...")
        keyword_data = extract_keywords(transcript)
        update(82, f"Found {len(keyword_data['combined'])} keywords")

        # 3b. Analytics
        update(85, "Running data analysis...")
        analytics_data = run_analytics(
            transcript     = transcript,
            summary        = summary_data["full_summary"],
            chunks         = summary_data["chunks"],
            entities_typed = keyword_data["entities_typed"],
            duration_str   = video_info["duration"],
        )

        # 4. Notes
        update(88, "Generating structured notes...")
        notes_md = generate_notes(
            video_url    = url,
            video_id     = video_id,
            summary_data = summary_data,
            keyword_data = keyword_data,
            title        = video_info["title"],
        )

        # 5. Export
        update(93, "Exporting files...")
        paths = export_all(notes_md, video_id, formats=formats)

        total_time = round(time.time() - t_total, 1)

        # Save to history
        history_entry = {
            "video_id"   : video_id,
            "title"      : video_info["title"],
            "thumbnail"  : video_info["thumbnail"],
            "channel"    : video_info["channel"],
            "duration"   : video_info["duration"],
            "url"        : url,
            "word_count" : word_count,
            "gpu_time"   : gpu_time,
            "speedup"    : speedup,
            "keywords"   : keyword_data["combined"][:8],
            "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M"),
            "formats"    : list(paths.keys()),
        }
        save_history(history_entry)

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"]  = f"Complete in {total_time}s!"
        jobs[job_id]["result"]   = {
            "video_id"   : video_id,
            "title"      : video_info["title"],
            "thumbnail"  : video_info["thumbnail"],
            "channel"    : video_info["channel"],
            "duration"   : video_info["duration"],
            "view_count" : video_info["view_count"],
            "word_count" : word_count,
            "num_chunks" : summary_data["num_chunks"],
            "keywords"   : keyword_data["combined"][:15],
            "entities"   : keyword_data["entities"][:10],
            "notes_md"   : notes_md,
            "paths"      : paths,
            "gpu_time"   : gpu_time,
            "fetch_time" : fetch_time,
            "cpu_estimate": cpu_estimate,
            "speedup"    : speedup,
            "total_time" : total_time,
            "source"     : transcript_data["source"],
            "language"   : transcript_data["language"],
            "device"     : DEVICE.upper(),
            "analytics"  : analytics_data,
        }

    except Exception as e:
        error_messages = {
            "VideoUnavailable"  : "This video is unavailable or private.",
            "No transcript"     : "No transcript found for this video.",
            "not a valid URL"   : "Please enter a valid YouTube URL.",
            "CUDA out of memory": "GPU out of memory. Try a shorter video.",
        }
        friendly = str(e)
        for key, msg in error_messages.items():
            if key.lower() in str(e).lower():
                friendly = msg
                break

        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = friendly
        jobs[job_id]["error"]   = friendly


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram     = f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" if torch.cuda.is_available() else ""
    history  = load_history()
    return render_template("index.html",
                           gpu_name=gpu_name, vram=vram,
                           device=DEVICE.upper(), history=history)


@app.route("/summarize", methods=["POST"])
def summarize():
    data    = request.get_json()
    url     = data.get("url", "").strip()
    formats = data.get("formats", ["md", "pdf", "docx"])

    if not url:
        return jsonify({"error": "Please enter a YouTube URL."}), 400
    if "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"error": "Please enter a valid YouTube URL."}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "progress": 0,
                    "message": "Starting...", "result": None, "error": None}
    threading.Thread(target=run_pipeline,
                     args=(job_id, url, formats), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/history")
def history():
    return jsonify(load_history())


@app.route("/download/<path:filename>")
def download(filename):
    full_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(full_path):
        return "File not found", 404
    return send_file(full_path, as_attachment=True)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  YouTube Transcript Summarizer v2.0 — Web UI")
    print(f"  Engine: {GEMMA_MODEL} (Gemma 4 via Ollama)")
    print("  Open browser:  http://localhost:5000")
    print("="*55 + "\n")
    check_ollama_ready()
    app.run(debug=False, host="0.0.0.0", port=5000)
