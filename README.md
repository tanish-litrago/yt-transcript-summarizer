# YouTube Transcript Summarizer & Note Maker
### v2.0 — Local LLM-Powered NLP via Gemma 4 (Ollama)

> Paste a YouTube URL → get structured Markdown / PDF / DOCX notes in seconds,
> powered by **Gemma 4 (e4b)** running fully locally on your NVIDIA RTX GPU via Ollama.

[![CI](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Gemma_4_e4b-black?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## What's New in v2.0

| | Before (v1.x) | After (v2.0) |
|---|---|---|
| **Summarization** | `facebook/bart-large-cnn` (HuggingFace) | Gemma 4 e4b via Ollama |
| **Keywords** | TF-IDF | Gemma 4 e4b via Ollama |
| **Named Entities** | spaCy `en_core_web_sm` | Gemma 4 e4b via Ollama |
| **Entity Types chart** | Single "Named Entity 100%" bucket | Real type breakdown (Person / Org / Location / Product / Event) |
| **Source files** | `summarizer.py` + `keyword_extractor.py` | Single `gemma_engine.py` |
| **Dependencies** | `transformers`, `sentencepiece`, `accelerate`, `spacy` | `requests` (talks to Ollama) |

**Also fixed in v2.0:**
- Ollama bug where `think: false` + `format: json` silently breaks Gemma 4 output — resolved by strong prompting + defensive parsing instead
- Truncated JSON responses caused by low default `num_predict` / `num_ctx` — now explicitly set to `1024` / `8192`
- JSON extraction rewritten with brace-counting (handles nested objects correctly) + partial-JSON repair for truncated responses
- 3-attempt retry logic for both summarization and keyword extraction calls
- Transcript truncated to 2500 words before Gemma calls to stay within context limits on long videos

---

## Features

- **Transcript extraction** via YouTube Transcript API, with OpenAI Whisper (CUDA) as fallback for uncaptioned videos
- **Local LLM pipeline** — Gemma 4 (e4b) via Ollama handles summarization, keyword extraction, and typed named-entity recognition — no cloud API calls
- **Structured Markdown notes** with section summaries, keywords, and entities
- **Export** to `.md`, `.pdf`, `.docx`
- **Flask Web UI** with live progress bar, processing history, and an analytics dashboard (Plotly)
- **CLI** with batch processing support

---

## Requirements

- Python 3.11
- [Ollama](https://ollama.com/download) installed and running
- NVIDIA RTX GPU + CUDA 12.x — used by Whisper fallback only; Gemma 4 is served by Ollama independently
- ~5 GB VRAM free for `gemma4:e4b`
- 16 GB RAM recommended

---

## Setup

**1 — Install Ollama and pull the model**
```bash
ollama serve
ollama pull gemma4:e4b
```

**2 — Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3 — (Optional) Whisper fallback requires CUDA PyTorch**

Only needed if you want audio transcription for videos with no captions:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Run

**Web UI**
```bash
python app.py
```
Open `http://localhost:5000`

**CLI**
```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

---

## Project Structure

```
yt_summarizer/
├── app.py                          # Flask Web UI
├── main.py                         # CLI entry point
├── config.py                       # Model name, paths, Ollama host
├── requirements.txt
├── src/
│   ├── gemma_engine.py             # Gemma 4 (Ollama) — summarization + keywords + entities
│   ├── transcript_fetcher.py       # YouTube Transcript API + Whisper fallback
│   ├── analyzer.py                 # Analytics: readability, sentiment, word freq, entity types
│   ├── note_generator.py           # Builds structured Markdown notes
│   ├── exporter.py                 # MD / PDF / DOCX export
│   └── video_info.py               # yt-dlp — title, thumbnail, channel, duration
├── templates/
│   └── index.html                  # Web UI (Plotly analytics dashboard)
├── outputs/                        # Generated notes + history.json
└── tests/
    ├── test_analyzer.py
    ├── test_config.py
    └── test_imports.py
```

---

## How It Works

```
YouTube URL
    │
    ▼
video_info.py          →  title, thumbnail, channel, duration
transcript_fetcher.py  →  transcript text (YouTube API / Whisper fallback)
    │
    ▼
gemma_engine.py
    ├── summarize_transcript()   →  Gemma 4 divides transcript into 4–8 sections
    └── extract_keywords()       →  Gemma 4 extracts keywords + typed named entities
                                    {"name": "MIT", "type": "Organization"}
    │
    ▼
analyzer.py        →  word frequency, compression ratio, readability,
                       sentiment timeline, speaking pace, entity type distribution
note_generator.py  →  builds structured Markdown
exporter.py        →  saves .md / .pdf / .docx  →  outputs/
```

---

## Analytics Dashboard

The **Analytics tab** in the Web UI shows six charts, all computed locally with no extra model calls:

| Chart | Source |
|---|---|
| Word Frequency | `analyzer.word_frequency()` |
| Compression Ratio | `analyzer.compression_ratio()` |
| Readability Score | `textstat` (Flesch–Kincaid) |
| Sentiment Timeline | Keyword-based scoring per section |
| Speaking Pace | Words per minute from video duration |
| Named Entity Types | Gemma's typed entities — Person / Organization / Location / Product / Event |

---

## Versions

| Version | Highlights |
|---|---|
| v1.0 | BART summarization + Flask Web UI + CLI |
| v1.1 | Video info, GPU speed stats, processing history, dark mode |
| v1.2 | Analytics dashboard (Plotly), unit tests, GitHub Actions CI |
| **v2.0** | **Replaced BART + spaCy/TF-IDF with Gemma 4 (e4b) via Ollama; typed entity extraction; smaller dependency footprint** |

**Planned:**
- v2.5 — RAG (ChromaDB + LangChain) for chat-with-video Q&A
- v2.6 — Open Knowledge Graph (OKG) + KG-RAG: D3.js force-directed graph (Outer Wilds Ship Log aesthetic) where the knowledge graph structure actively guides retrieval — clicking a node traverses graph edges to pull related chunks → Gemma 4 answers. The graph IS the query interface, KG-RAG is the backend engine.
- v3.0 — Claim extraction + fact-checking against web sources
- v4.0 — Docker + live demo deployment

---

## Tech Stack

Python · Gemma 4 e4b (Ollama) · OpenAI Whisper · Flask · Plotly · fpdf2 · python-docx · yt-dlp · textstat

---

## Author

**Tanish Kumar** — B.Tech CSE, Government Engineering College, Bilaspur
GitHub: [@tanish-litrago](https://github.com/tanish-litrago)
