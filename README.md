# YouTube Transcript Summarizer & Note Maker
### v2.6 — Local LLM · RAG Chat · Knowledge Graph

> Paste a YouTube URL → get structured Markdown / PDF / DOCX notes, **chat with the video**,
> and **explore its knowledge graph** — all powered by **Gemma 4 (e4b)** running fully locally
> on your NVIDIA RTX GPU via Ollama. No cloud APIs. No data leaves your machine.

[![CI](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Gemma_4_e4b-black?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-1.x-teal?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-orange?style=flat-square)
![D3.js](https://img.shields.io/badge/D3.js-v7-f9a825?style=flat-square)
![License](https://img.shields.io/badge/License-AGPL--3.0-purple?style=flat-square)

---

## What's New in v2.6 — Knowledge Graph

| | v2.5 | v2.6 |
|---|---|---|
| **Graph tab** | — | D3.js force-directed knowledge graph |
| **KG extraction** | — | Gemma 4 extracts nodes + typed edges from transcript |
| **Node click** | — | Ask Gemma what the video says about that concept |
| **Edge click** | — | Ask Gemma how two concepts relate in the video |
| **Graph Q&A** | — | Free-form question box with node highlighting |
| **Weak edges** | — | Dimmed/dashed for low-confidence links (< 0.7) |
| **Graph caching** | — | Per-video graph saved in `outputs/kg/` — no re-extraction |

**How KG-RAG works:**
```
Your question / node click
    │
    ▼
knowledge_graph.py   →  find relevant node(s) in graph
                     →  1-hop traversal: pull in neighbour nodes
    │
    ▼
kg_rag_engine.py     →  ChromaDB similarity search using node labels + neighbours
                     →  grounded prompt to Gemma 4
    │
    ▼
Graph tab            →  answer panel + collapsible source excerpts
                     →  involved nodes pulse gold on graph
```

Plain RAG (v2.5) only retrieves chunks similar to the question. KG-RAG also traverses graph edges — so asking about "BERT" also retrieves "Transformer" chunks (its parent via `gave_rise_to`), giving Gemma richer context.

---

## What's New in v2.5 — RAG Chat-with-Video

| | v2.0 | v2.5 |
|---|---|---|
| **Chat tab** | — | Ask any question about the video |
| **Retrieval** | — | ChromaDB vector search (top-4 chunks) |
| **Embeddings** | — | Ollama `nomic-embed-text` (local, no cloud) |
| **Index persistence** | — | Cached per `video_id` in `outputs/chroma/` |
| **Answer grounding** | — | Gemma answers **only** from retrieved transcript excerpts |

---

## Features

- **Transcript extraction** via YouTube Transcript API, with OpenAI Whisper (CUDA) as fallback for uncaptioned videos
- **Local LLM pipeline** — Gemma 4 (e4b) via Ollama handles summarization, keyword extraction, named-entity recognition, and knowledge graph extraction — no cloud API calls
- **Knowledge Graph** — D3.js force-directed graph auto-built from transcript; click nodes/edges to query Gemma via KG-RAG
- **RAG Chat-with-Video** — ask natural-language questions, get Gemma-grounded answers with source excerpts
- **Structured Markdown notes** with section summaries, keywords, and entities
- **Export** to `.md`, `.pdf`, `.docx`
- **Flask Web UI** — Summary · Chat · Analytics · **Graph** · History tabs, live progress bar, dark fantasy aesthetic
- **Analytics dashboard** (Plotly) — word frequency, compression ratio, readability, sentiment, speaking pace, entity types

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
ollama pull nomic-embed-text
```

**2 — Install Python dependencies**
```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install PyTorch with CUDA 12.8 FIRST (must be a separate step — pip's
# --extra-index-url is unreliable for torch and may silently install CPU-only)
pip install torch torchaudio --no-cache-dir --index-url https://download.pytorch.org/whl/cu128

# Then install the rest
pip install -r requirements.txt
```

> **Verify GPU is detected after install:**
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> # Expected: True   NVIDIA GeForce RTX 4060 ...
> ```

**3 — (Optional) Fix CPU-only torch**

If `torch.cuda.is_available()` returns `False`:
```bash
pip uninstall torch torchaudio -y
pip install torch torchaudio --no-cache-dir --index-url https://download.pytorch.org/whl/cu128
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
yt-transcript-summarizer/
├── app.py                          # Flask Web UI (v2.6: + /kg/query route)
├── main.py                         # CLI entry point
├── config.py                       # Model name, paths, Ollama host, RAG + KG config
├── requirements.txt
├── src/
│   ├── gemma_engine.py             # Gemma 4 (Ollama) — summarization + keywords + entities
│   ├── rag_engine.py               # RAG engine — ChromaDB + LangChain + Ollama embeddings
│   ├── knowledge_graph.py          # v2.6: Gemma KG extraction → nodes + edges JSON
│   ├── kg_rag_engine.py            # v2.6: KG-RAG — graph traversal + ChromaDB retrieval
│   ├── transcript_fetcher.py       # YouTube Transcript API + Whisper fallback
│   ├── analyzer.py                 # Analytics: readability, sentiment, word freq, entity types
│   ├── note_generator.py           # Builds structured Markdown notes
│   ├── exporter.py                 # MD / PDF / DOCX export
│   └── video_info.py               # yt-dlp — title, thumbnail, channel, duration
├── templates/
│   └── index.html                  # Web UI — Summary · Chat · Analytics · Graph · History
├── outputs/
│   ├── chroma/                     # ChromaDB vector stores (one subdir per video_id)
│   ├── kg/                         # v2.6: Knowledge graph JSON cache (one file per video_id)
│   └── history.json
└── tests/
    ├── test_analyzer.py
    ├── test_config.py
    ├── test_imports.py
    ├── test_rag_engine.py
    └── test_knowledge_graph.py     # v2.6: 3 mocked KG tests
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
    └── extract_keywords()       →  keywords + typed named entities
    │
    ▼
rag_engine.py          →  chunk → embed (nomic-embed-text) → ChromaDB
knowledge_graph.py     →  Gemma 4 extracts KG nodes + edges → cached in outputs/kg/
analyzer.py            →  word frequency, compression ratio, readability,
                           sentiment timeline, speaking pace, entity type distribution
note_generator.py      →  builds structured Markdown
exporter.py            →  saves .md / .pdf / .docx → outputs/
```

**KG-RAG query pipeline (Graph tab):**
```
Node click / edge click / typed question
    │
    ▼
kg_rag_engine.py    →  find relevant nodes in the cached graph
                    →  1-hop traversal to collect neighbour node labels
                    →  ChromaDB similarity search (node labels + neighbours as query)
                    →  grounded prompt: "Based only on these excerpts, explain X"
    │
    ▼
Gemma 4             →  grounded answer
    │
    ▼
Graph tab           →  answer panel + source excerpts + involved nodes pulse gold
```

---

## Versions

| Version | Highlights |
|---|---|
| v1.0 | BART summarization + Flask Web UI + CLI |
| v1.1 | Video info, GPU speed stats, processing history, dark mode |
| v1.2 | Analytics dashboard (Plotly), unit tests, GitHub Actions CI |
| v2.0 | Replaced BART + spaCy/TF-IDF with Gemma 4 (e4b) via Ollama; typed entity extraction |
| v2.5 | RAG Chat-with-Video: ChromaDB + LangChain + Ollama nomic-embed-text; Chat tab |
| **v2.6** | **Knowledge Graph: D3.js force graph + KG-RAG (graph-guided retrieval); Graph tab; node/edge Q&A; dark fantasy UI** |

**Planned:**
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

Python · Gemma 4 e4b (Ollama) · nomic-embed-text (Ollama) · LangChain · ChromaDB · D3.js · OpenAI Whisper · Flask · Plotly · fpdf2 · python-docx · yt-dlp · textstat

---

## Author

**Tanish Kumar** — B.Tech CSE, Government Engineering College, Bilaspur
GitHub: [@tanish-litrago](https://github.com/tanish-litrago)
