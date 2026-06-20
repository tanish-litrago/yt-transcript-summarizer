# YouTube Transcript Summarizer & Note Maker
### v1.1 — CUDA-Accelerated NLP on NVIDIA RTX GPU

> Paste a YouTube URL → get structured Markdown / PDF / DOCX notes in seconds,  
> powered by BART transformer running on your NVIDIA RTX GPU via CUDA.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![CI](https://github.com/tanish-litrago/yt-transcript-summarizer/tree/main/tests)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## What's New in v1.1

- Video title + thumbnail display in Web UI
- GPU vs CPU speed comparison with visual bar chart
- Processing history with search and sort
- Dark / Light mode toggle
- Copy notes to clipboard
- Real-time URL validation
- Font size adjuster for notes viewer
- Better friendly error messages

---

## Features

| Feature | Detail |
|---|---|
| Transcript Extraction | YouTube Transcript API → Whisper GPU fallback |
| Summarization | BART-large-CNN on CUDA (10x+ faster than CPU) |
| Keyword Extraction | TF-IDF + spaCy Named Entity Recognition |
| Note Structuring | Auto-sectioned Markdown with headings and bullets |
| Export | .md + .pdf + .docx in one command |
| Web UI | Flask + real-time progress bar + history tab |
| Batch Mode | Process many URLs from a text file |
| CLI Support | Full command-line interface via main.py |

---

## Requirements

| Component | Minimum |
|---|---|
| OS | Windows 11 / Ubuntu 22.04 |
| Python | 3.11 |
| GPU | NVIDIA RTX Series |
| VRAM | 6 GB+ (BART-large-CNN) |
| CUDA | 12.x |
| RAM | 16 GB |
| Storage | 10 GB free (for model cache) |

---

## Setup

### 1 — Install CUDA PyTorch
```bash
# CUDA 12.x (RTX 40xx series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 (RTX 30xx series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2 — Install all dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3 — Run Web UI
```bash
python app.py
```
Open browser → `http://localhost:5000`

### 4 — Or use CLI
```bash
# Single video
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Choose export formats
python main.py --url "..." --formats md pdf

# Use lighter model (less VRAM)
python main.py --url "..." --model t5-base

# Batch mode
python main.py --batch urls.txt
```

---

## Project Structure

```
yt_summarizer/
├── app.py                     # Flask Web UI backend
├── main.py                    # CLI entry point
├── config.py                  # All settings (model, device, paths)
├── requirements.txt
├── src/
│   ├── transcript_fetcher.py  # YouTube API + Whisper GPU fallback
│   ├── summarizer.py          # CUDA-accelerated BART inference
│   ├── keyword_extractor.py   # TF-IDF + spaCy NER pipeline
│   ├── note_generator.py      # Structured Markdown note builder
│   ├── exporter.py            # MD + PDF + DOCX export
│   └── video_info.py          # Title, thumbnail, channel metadata
└── templates/
    └── index.html             # Web UI (dark/light, history, stats)
```

---

## How It Works

```
YouTube URL
     │
     ▼
 Video Info Fetcher
 (title, thumbnail, channel, duration)
     │
     ▼
 Transcript Fetcher
 ├── YouTube Transcript API  (fast, no download)
 └── OpenAI Whisper on CUDA  (fallback for uncaptioned videos)
     │
     ▼
 BART-large-CNN on NVIDIA RTX GPU
 ├── Split transcript into 1024-token chunks
 ├── torch.autocast fp16 inference on CUDA
 └── 10x+ faster than CPU
     │
     ▼
 Keyword Extractor
 ├── TF-IDF scoring (pure Python)
 └── spaCy Named Entity Recognition
     │
     ▼
 Note Generator
 └── Auto-sectioned Markdown notes
     │
     ▼
 Exporter
 ├── .md  (Markdown)
 ├── .pdf (fpdf2)
 └── .docx (python-docx)
```

---

## Configuration

Edit `config.py` to change model or settings:

```python
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # or "t5-base", "t5-small"
WHISPER_MODEL       = "base"                      # tiny, base, small, medium
CHUNK_TOKEN_LIMIT   = 1024
SUMMARY_MIN_LENGTH  = 60
SUMMARY_MAX_LENGTH  = 200
TOP_KEYWORDS        = 10
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA not available | Reinstall PyTorch with cu124 index URL |
| No captions found | Install FFmpeg — Whisper fallback will handle it |
| spaCy model not found | Run `python -m spacy download en_core_web_sm` |
| yt-dlp error | Run `pip install -U yt-dlp` |
| CUDA out of memory | Switch to `--model t5-small` in CLI |

---

## Versions

| Version | Features | Status |
|---|---|---|
| v1.0 | BART + Flask Web UI + CLI + Export | Released |
| v1.1 | Video info + Speed stats + History + Dark mode + Copy | Released |
| v2.0 | Phi-3 INT4 Quantized + RAG + ChromaDB + Chat UI | Coming Soon |
| v2.1 | Exam Q&A Generator + Flashcard Export | Planned |
| v3.0 | Multi-language + Chrome Extension + Cloud Deploy | Planned |

---

## Tech Stack

`Python 3.11` &nbsp;·&nbsp; `PyTorch (CUDA)` &nbsp;·&nbsp; `HuggingFace Transformers` &nbsp;·&nbsp; `BART-large-CNN`  
`OpenAI Whisper` &nbsp;·&nbsp; `spaCy` &nbsp;·&nbsp; `Flask` &nbsp;·&nbsp; `fpdf2` &nbsp;·&nbsp; `python-docx` &nbsp;·&nbsp; `yt-dlp`

---

## Author

**Tanish Kumar** &nbsp;·&nbsp; B.Tech CSE &nbsp;·&nbsp; Government Engineering College, Bilaspur  
GitHub: [@tanish-litrago](https://github.com/tanish-litrago)
