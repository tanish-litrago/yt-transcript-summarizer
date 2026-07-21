# YouTube Transcript Summarizer & Note Maker
### v2.5 — Local LLM · RAG Chat-with-Video

> Paste a YouTube URL → get structured Markdown / PDF / DOCX notes **and chat with the video**,
> all powered by **Gemma 4 (e4b)** running fully locally on your NVIDIA RTX GPU via Ollama.
> No cloud APIs. No data leaves your machine.

[![CI](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Gemma_4_e4b-black?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-1.x-teal?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## What's New in v2.5 — RAG Chat-with-Video

| | v2.0 | v2.5 |
|---|---|---|
| **Chat tab** | — | ✅ Ask any question about the video |
| **Retrieval** | — | ChromaDB vector search (top-4 chunks) |
| **Embeddings** | — | Ollama `nomic-embed-text` (local, no cloud) |
| **Chunking** | — | LangChain `RecursiveCharacterTextSplitter` |
| **Index persistence** | — | Cached per `video_id` in `outputs/chroma/` |
| **Answer grounding** | — | Gemma answers **only** from retrieved transcript excerpts |
| **Source transparency** | — | Each answer shows collapsible source excerpts |

**How RAG chat works:**
```
Your question  →  nomic-embed-text (embed)
               →  ChromaDB similarity search  →  top-4 transcript chunks
               →  Grounded prompt to Gemma 4
               →  Answer + source excerpts shown in Chat tab
```

The vector store is built automatically when you summarize a video and is reused on every revisit — **no re-embedding cost** when you open a past video from History.

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

---

## Features

- **Transcript extraction** via YouTube Transcript API, with OpenAI Whisper (CUDA) as fallback for uncaptioned videos
- **Local LLM pipeline** — Gemma 4 (e4b) via Ollama handles summarization, keyword extraction, and named-entity recognition — no cloud API calls
- **RAG Chat-with-Video** — ask natural-language questions, get Gemma-grounded answers with source excerpts shown
- **Structured Markdown notes** with section summaries, keywords, and entities
- **Export** to `.md`, `.pdf`, `.docx`
- **Flask Web UI** — Summary · Chat · Analytics · History tabs, live progress bar, dark/light mode
- **Analytics dashboard** (Plotly) — word frequency, compression ratio, readability, sentiment, speaking pace, entity types
- **CLI** with batch processing support

---

## Requirements

- Python 3.11
- [Ollama](https://ollama.com/download) installed and running
- NVIDIA RTX GPU + CUDA 12.8+ — used by Whisper fallback only; Gemma 4 and nomic-embed-text are served by Ollama
- ~5 GB VRAM free for `gemma4:e4b` + ~1 GB for `nomic-embed-text`
- 16 GB RAM recommended

---

## Setup

**1 — Install Ollama and pull the models**
```bash
ollama serve
ollama pull gemma4:e4b
ollama pull nomic-embed-text   # used for RAG embeddings (v2.5)
```

**2 — Install Python dependencies**
```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

> **GPU note:** `requirements.txt` includes `--extra-index-url https://download.pytorch.org/whl/cu128`,
> so `torch` is installed with **CUDA 12.8** support automatically — no manual step needed.
> After install, verify your GPU is detected:
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> # Expected: True   NVIDIA GeForce RTX 4060 ...
> ```

**3 — (Optional) Reinstall torch with CUDA if you get CPU-only**

If pip pulled the CPU build (e.g. from a cached install), force the CUDA version:
```bash
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
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
├── app.py                          # Flask Web UI (v2.5: + /chat route)
├── main.py                         # CLI entry point
├── config.py                       # Model name, paths, Ollama host, RAG config
├── requirements.txt
├── src/
│   ├── gemma_engine.py             # Gemma 4 (Ollama) — summarization + keywords + entities
│   ├── rag_engine.py               # v2.5: RAG engine — ChromaDB + LangChain + Ollama embeddings
│   ├── transcript_fetcher.py       # YouTube Transcript API + Whisper fallback
│   ├── analyzer.py                 # Analytics: readability, sentiment, word freq, entity types
│   ├── note_generator.py           # Builds structured Markdown notes
│   ├── exporter.py                 # MD / PDF / DOCX export
│   └── video_info.py               # yt-dlp — title, thumbnail, channel, duration
├── templates/
│   └── index.html                  # Web UI — Summary · Chat · Analytics · History tabs
├── outputs/
│   ├── chroma/                     # v2.5: ChromaDB vector stores (one subdir per video_id)
│   └── history.json
└── tests/
    ├── test_analyzer.py
    ├── test_config.py
    ├── test_imports.py
    └── test_rag_engine.py          # v2.5: 3 mocked RAG tests
```

---

## How It Works

**Summarization pipeline:**
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
rag_engine.py          →  chunk transcript → embed (nomic-embed-text) → ChromaDB
analyzer.py            →  word frequency, compression ratio, readability,
                           sentiment timeline, speaking pace, entity type distribution
note_generator.py      →  builds structured Markdown
exporter.py            →  saves .md / .pdf / .docx  →  outputs/
```

**RAG chat pipeline (after summarization):**
```
User question
    │
    ▼
nomic-embed-text (via Ollama)  →  question embedding
    │
    ▼
ChromaDB similarity_search()   →  top-4 most relevant transcript chunks
    │
    ▼
Gemma 4 (grounded prompt)      →  answer + source excerpts
    │
    ▼
Chat tab                       →  answer bubble + collapsible source quotes
```

---

## Versions

| Version | Highlights |
|---|---|
| v1.0 | BART summarization + Flask Web UI + CLI |
| v1.1 | Video info, GPU speed stats, processing history, dark mode |
| v1.2 | Analytics dashboard (Plotly), unit tests, GitHub Actions CI |
| v2.0 | Replaced BART + spaCy/TF-IDF with Gemma 4 (e4b) via Ollama; typed entity extraction; smaller dependency footprint |
| **v2.5** | **RAG Chat-with-Video: ChromaDB + LangChain + Ollama nomic-embed-text; Chat tab; per-video index caching** |

**Planned:**
- v2.6 — Open Knowledge Graph (OKG) + KG-RAG: D3.js force-directed graph where the knowledge graph structure actively guides retrieval — clicking a node traverses graph edges to pull related chunks → Gemma 4 answers. The graph IS the query interface; KG-RAG is the retrieval engine.
- v3.0 — Claim extraction + fact-checking against web sources
- v4.0 — Docker + live demo deployment

---

## Analytics Dashboard

The **Analytics tab** shows six charts, all computed locally with no extra model calls:

| Chart | Source |
|---|---|
| Word Frequency | `analyzer.word_frequency()` |
| Compression Ratio | `analyzer.compression_ratio()` |
| Readability Score | `textstat` (Flesch–Kincaid) |
| Sentiment Timeline | Keyword-based scoring per section |
| Speaking Pace | Words per minute from video duration |
| Named Entity Types | Gemma's typed entities — Person / Organization / Location / Product / Event |

---

## Tech Stack

Python · Gemma 4 e4b (Ollama) · nomic-embed-text (Ollama) · LangChain · ChromaDB · OpenAI Whisper · Flask · Plotly · fpdf2 · python-docx · yt-dlp · textstat

---

## Author

**Tanish Kumar** — B.Tech CSE, Government Engineering College, Bilaspur
GitHub: [@tanish-litrago](https://github.com/tanish-litrago)
