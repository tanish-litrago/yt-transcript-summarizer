"""
app.py
──────
Flask Web UI for YouTube Transcript Summarizer
Run:  python app.py
Then open:  http://localhost:5000
"""

import os
import sys
import threading
import uuid
from flask import Flask, render_template, request, jsonify, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transcript_fetcher  import fetch_transcript
from src.summarizer          import summarize_transcript
from src.keyword_extractor   import extract_keywords
from src.note_generator      import generate_notes
from src.exporter            import export_all
from config                  import OUTPUT_DIR, DEVICE

import torch

app = Flask(__name__)

# ── In-memory job store ───────────────────────────────────────────────────────
# { job_id: { status, progress, message, result, error } }
jobs = {}


def run_pipeline(job_id: str, url: str, formats: list):
    """Runs the full pipeline in a background thread."""
    try:
        def update(progress, message):
            jobs[job_id]["progress"] = progress
            jobs[job_id]["message"]  = message

        update(5,  "Fetching transcript from YouTube...")
        transcript_data = fetch_transcript(url)
        transcript      = transcript_data["transcript"]
        video_id        = transcript_data["video_id"]
        word_count      = len(transcript.split())
        update(25, f"Transcript fetched — {word_count:,} words ({transcript_data['source']})")

        update(30, f"Loading BART model on {DEVICE.upper()}...")
        summary_data = summarize_transcript(transcript)
        update(70, f"Summarised into {summary_data['num_chunks']} section(s) — {len(summary_data['full_summary'].split())} words")

        update(75, "Extracting keywords and named entities...")
        keyword_data = extract_keywords(transcript)
        kw_preview   = ", ".join(keyword_data["combined"][:5])
        update(85, f"Keywords found: {kw_preview}...")

        update(88, "Generating structured notes...")
        notes_md = generate_notes(
            video_url    = url,
            video_id     = video_id,
            summary_data = summary_data,
            keyword_data = keyword_data,
        )

        update(92, "Exporting files...")
        paths = export_all(notes_md, video_id, formats=formats)

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"]  = "Complete!"
        jobs[job_id]["result"]   = {
            "video_id"    : video_id,
            "word_count"  : word_count,
            "source"      : transcript_data["source"],
            "language"    : transcript_data["language"],
            "num_chunks"  : summary_data["num_chunks"],
            "keywords"    : keyword_data["combined"][:15],
            "entities"    : keyword_data["entities"][:10],
            "notes_md"    : notes_md,
            "paths"       : paths,
        }

    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)
        jobs[job_id]["error"]   = str(e)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram     = f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" if torch.cuda.is_available() else ""
    return render_template("index.html", gpu_name=gpu_name, vram=vram, device=DEVICE.upper())


@app.route("/summarize", methods=["POST"])
def summarize():
    data    = request.get_json()
    url     = data.get("url", "").strip()
    formats = data.get("formats", ["md", "pdf", "docx"])

    if not url:
        return jsonify({"error": "Please enter a YouTube URL."}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status"  : "running",
        "progress": 0,
        "message" : "Starting...",
        "result"  : None,
        "error"   : None,
    }

    thread = threading.Thread(target=run_pipeline, args=(job_id, url, formats), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/download/<path:filename>")
def download(filename):
    full_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(full_path):
        return "File not found", 404
    return send_file(full_path, as_attachment=True)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  YouTube Transcript Summarizer — Web UI")
    print("  Open your browser and go to:  http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
