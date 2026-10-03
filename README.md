# YouTube Transcript Summarizer & Note Maker
### v3.0 — Local LLM · RAG Chat · Knowledge Graph · Fact-Checking

> Paste a YouTube URL → get structured Markdown / PDF / DOCX notes, **chat with the video**,
> **explore its knowledge graph**, and **fact-check key claims against the web** — all powered
> by **Gemma 4 (e4b)** running fully locally on your NVIDIA RTX GPU via Ollama.
> No cloud APIs. No data leaves your machine.

[![CI](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/tanish-litrago/yt-transcript-summarizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Gemma_4_e4b-black?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-1.x-teal?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-orange?style=flat-square)
![D3.js](https://img.shields.io/badge/D3.js-v7-f9a825?style=flat-square)
![License](https://img.shields.io/badge/License-AGPL--3.0-purple?style=flat-square)

---

## What's New in v3.0 — Fact-Checking

| | v2.6 | v3.0 |
|---|---|---|
| **Fact Check tab** | — | Dedicated tab with verdict cards |
| **Claim extraction** | — | Gemma extracts 5–8 checkable factual claims from the summary |
| **Web search** | — | DuckDuckGo search per claim (free, no API key) |
| **Verdict** | — | Gemma cross-references claim vs. web snippets → `supported / contradicted / partially / unverifiable` |
| **Inline highlights** | — | Matched sentences in the notes get colour-coded underlines |
| **Caching** | — | Results cached per `video_id` in `outputs/fact_checks/` |
| **On-demand** | — | Triggered manually after summarizing — doesn't slow the pipeline |

**How fact-checking works:**
```
User clicks "⚑ Fact Check"
    │
    ▼
fact_checker.py
    ├── extract_claims(summary)   → Gemma → JSON list of 5–8 factual claims
    ├── search_claim(claim)       → DuckDuckGo DDGS → top 4 web snippets per claim
    └── verify_claim(claim, hits) → Gemma → { verdict, explanation, sources }
    │
    ▼
Fact Check tab  →  verdict cards (✓ Supported / ✗ Contradicted / ~ Partial / ? Unverifiable)
Summary tab     →  inline highlights on matched sentences
```

> **Note on "unverifiable":** original educational videos (e.g. 3Blue1Brown, lecture recordings)
> will show mostly "unverifiable" — they present original explanations, not citable external facts.
> Fact-checking works best on news summaries, tech talks citing papers, or videos with statistics.

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
kg_rag_engine.py   →  find relevant node(s) in graph
                   →  1-hop traversal: pull in neighbour nodes
                   →  ChromaDB similarity search (label + neighbours, 2× top-k)
                   →  grounded prompt to Gemma 4
    │
    ▼
Graph tab          →  answer panel + source excerpts + involved nodes pulse gold
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
- **Local LLM pipeline** — Gemma 4 (e4b) via Ollama handles summarization, keyword extraction, named-entity recognition, knowledge graph extraction, and fact-checking — no cloud API calls
- **Fact-Checking** — DuckDuckGo search + Gemma verifies key claims from the summary against live web results
- **Knowledge Graph** — D3.js force-directed graph auto-built from transcript; click nodes/edges to query Gemma via KG-RAG
- **RAG Chat-with-Video** — ask natural-language questions, get Gemma-grounded answers with source excerpts
- **Structured Markdown notes** with section summaries, keywords, and entities
- **Export** to `.md`, `.pdf`, `.docx`
- **Flask Web UI** — Summary · Chat · Analytics · Graph · **Fact Check** · History tabs, live progress bar, dark fantasy aesthetic
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
├── app.py                          # Flask Web UI (v3.0: + /fact-check route)
├── main.py                         # CLI entry point
├── config.py                       # Model name, paths, Ollama host, RAG + KG + FC config
├── requirements.txt
├── src/
│   ├── gemma_engine.py             # Gemma 4 (Ollama) — summarization + keywords + entities
│   ├── rag_engine.py               # RAG engine — ChromaDB + LangChain + Ollama embeddings
│   ├── knowledge_graph.py          # Gemma KG extraction → nodes + edges JSON
│   ├── kg_rag_engine.py            # KG-RAG — graph traversal + ChromaDB retrieval
│   ├── fact_checker.py             # v3.0: claim extraction + DuckDuckGo + Gemma verification
│   ├── transcript_fetcher.py       # YouTube Transcript API + Whisper fallback
│   ├── analyzer.py                 # Analytics: readability, sentiment, word freq, entity types
│   ├── note_generator.py           # Builds structured Markdown notes
│   ├── exporter.py                 # MD / PDF / DOCX export
│   └── video_info.py               # yt-dlp — title, thumbnail, channel, duration
├── templates/
│   └── index.html                  # Web UI — Summary · Chat · Analytics · Graph · Fact Check · History
├── outputs/
│   ├── chroma/                     # ChromaDB vector stores (one subdir per video_id)
│   ├── kg/                         # Knowledge graph JSON cache (one file per video_id)
│   ├── fact_checks/                # v3.0: Fact-check results cache (one file per video_id)
│   └── history.json
└── tests/
    ├── test_analyzer.py
    ├── test_config.py
    ├── test_imports.py
    ├── test_rag_engine.py
    ├── test_knowledge_graph.py
    └── test_fact_checker.py        # v3.0: 10 mocked fact-checker tests
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

**Fact-checking pipeline (on-demand):**
```
User clicks "⚑ Fact Check" after summarizing
    │
    ▼
POST /fact-check  { video_id, summary }
    │
    ▼
fact_checker.py
    ├── extract_claims()   →  Gemma → up to 8 checkable factual claims (JSON)
    ├── search_claim()     →  DuckDuckGo DDGS → 4 web snippets per claim
    └── verify_claim()     →  Gemma → { verdict, explanation, sources }
    │
    ▼
Fact Check tab    →  verdict cards with source links
Summary tab       →  inline highlights on matching sentences
outputs/fact_checks/<video_id>.json  →  cached for re-runs
```

**KG-RAG query pipeline (Graph tab):**
```
Node click / edge click / typed question
    │
    ▼
kg_rag_engine.py    →  find relevant nodes in the cached graph
                    →  1-hop traversal to collect neighbour node labels
                    →  ChromaDB similarity search (node labels + neighbours, 2× top-k)
                    →  two-pass fallback: bare label if augmented query returns nothing
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
| v2.6 | Knowledge Graph: D3.js force graph + KG-RAG (graph-guided retrieval); Graph tab; node/edge Q&A; dark fantasy UI |
| **v3.0** | **Fact-Checking: Gemma claim extraction + DuckDuckGo web search + verdict cards; inline highlights; Fact Check tab** |

**Planned:**
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

Python · Gemma 4 e4b (Ollama) · nomic-embed-text (Ollama) · LangChain · ChromaDB · D3.js · DuckDuckGo Search · OpenAI Whisper · Flask · Plotly · fpdf2 · python-docx · yt-dlp · textstat

---

## Author

**Tanish Kumar** — B.Tech CSE, Government Engineering College, Bilaspur
GitHub: [@tanish-litrago](https://github.com/tanish-litrago)
