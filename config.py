import os
import torch

# ── Device ───────────────────────────────────────────────────────────────────
# Used only by Whisper (audio fallback). Gemma 4 runs through Ollama, which
# manages its own GPU usage independently.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Gemma 4 via Ollama ─────────────────────────────────────────────────────────
# Replaces BART (summarization) + spaCy/TF-IDF (keyword extraction).
# Make sure Ollama is running and the model is pulled:
#   ollama serve
#   ollama pull gemma4:e4b
OLLAMA_HOST = "http://localhost:11434"
GEMMA_MODEL = "gemma4:e4b"

# ── Whisper speech-to-text model (fallback when no captions exist) ────────────
# Options: "tiny", "base", "small", "medium", "large"
WHISPER_MODEL = "base"

# ── Keyword extraction ─────────────────────────────────────────────────────────
TOP_KEYWORDS = 10

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "models_cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Whisper model cache stays inside the project folder.
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR
os.environ["HF_HOME"]            = MODELS_DIR

# ── RAG / Chat-with-Video (v2.5) ──────────────────────────────────────────────
# ChromaDB vector stores — one subdirectory per video_id, persisted across runs.
# Embeddings via Ollama (nomic-embed-text).
# Before first use:  ollama pull nomic-embed-text
CHROMA_DIR        = os.path.join(OUTPUT_DIR, "chroma")
EMBED_MODEL       = "nomic-embed-text"
RAG_CHUNK_SIZE    = 500   # characters per chunk
RAG_CHUNK_OVERLAP = 50    # character overlap between adjacent chunks
RAG_TOP_K         = 4     # chunks retrieved per user question

os.makedirs(CHROMA_DIR, exist_ok=True)
