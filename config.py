"""
config.py
─────────
Central configuration for the YouTube Transcript Summarizer.
Edit this file to switch models, paths, or summarization parameters.
"""

import os
import torch

# ── Device ───────────────────────────────────────────────────────────────────
# Automatically uses NVIDIA RTX GPU (CUDA) if available, else CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Summarization model ───────────────────────────────────────────────────────
# Options (all work with CUDA):
#   "facebook/bart-large-cnn"   — Best quality, slower, needs ~2 GB VRAM
#   "t5-base"                   — Faster, lower VRAM (~600 MB)
#   "t5-small"                  — Fastest, minimal VRAM (~250 MB)
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# ── Whisper speech-to-text model (fallback when no captions exist) ────────────
# Options: "tiny", "base", "small", "medium", "large"
# "base" is a good balance of speed and accuracy on GPU.
WHISPER_MODEL = "base"

# ── spaCy NLP model for keyword extraction ────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ── Summarization parameters ──────────────────────────────────────────────────
# Max token length each transcript chunk is split into before summarizing.
CHUNK_TOKEN_LIMIT = 1024          # BART supports up to 1024 input tokens

# Min/max tokens for each generated summary chunk.
SUMMARY_MIN_LENGTH = 60
SUMMARY_MAX_LENGTH = 200

# Number of top keywords to extract per section.
TOP_KEYWORDS = 10

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "models_cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Cache HuggingFace model downloads inside the project folder.
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR
os.environ["HF_HOME"]            = MODELS_DIR
