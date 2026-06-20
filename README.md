# YouTube Transcript Summarizer & Note Maker
### v1.2 — CUDA-Accelerated NLP on NVIDIA RTX GPU

> Paste a YouTube URL → get structured Markdown / PDF / DOCX notes in seconds,  
> powered by BART transformer running on your NVIDIA RTX GPU via CUDA.

[![CI](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## What's New in v1.2

- Analytics dashboard with interactive Plotly charts
- Compression ratio, readability score, and speaking pace
- Word frequency, sentiment timeline, entity types, parts of speech
- Unit tests + GitHub Actions CI

---

## Features

- Transcript extraction via YouTube API with Whisper GPU fallback
- CUDA-accelerated summarization using BART-large-CNN
- Keyword extraction using TF-IDF + spaCy NER
- Auto-structured Markdown notes
- Export to `.md`, `.pdf`, `.docx`
- Flask Web UI with live progress, history, and analytics
- CLI support with batch processing

---

## Requirements

- Python 3.11
- NVIDIA RTX GPU with CUDA 12.x
- 6GB+ VRAM
- 16GB RAM

---

## Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
python app.py
```
Open `http://localhost:5000`

Or use the CLI:
```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

---

## Project Structure

```
yt_summarizer/
├── app.py
├── main.py
├── config.py
├── src/
│   ├── transcript_fetcher.py
│   ├── summarizer.py
│   ├── keyword_extractor.py
│   ├── note_generator.py
│   ├── exporter.py
│   ├── video_info.py
│   └── analyzer.py
├── templates/
│   └── index.html
└── tests/
```

---

## Versions

| Version | Highlights |
|---|---|
| v1.0 | BART + Flask Web UI + CLI |
| v1.1 | Video info, speed stats, history, dark mode |
| v1.2 | Analytics dashboard, unit tests, CI |

---

## Tech Stack

Python · PyTorch (CUDA) · HuggingFace Transformers · BART · Whisper · spaCy · Flask · Plotly · fpdf2 · python-docx

---

## Author

**Tanish** — B.Tech CSE, Government Engineering College, Bilaspur  
GitHub: [@tanish-litrago](https://github.com/tanish-litrago)
