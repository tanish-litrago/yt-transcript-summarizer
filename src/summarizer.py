"""
src/summarizer.py
─────────────────
CUDA-accelerated abstractive summarisation using BART or T5 via
HuggingFace Transformers.

The transcript is split into chunks that fit inside the model's
input window, summarised in parallel on the GPU, then merged.
"""

import os
import sys
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE,
    SUMMARIZATION_MODEL,
    CHUNK_TOKEN_LIMIT,
    SUMMARY_MIN_LENGTH,
    SUMMARY_MAX_LENGTH,
)


# ── Model loading (singleton) ─────────────────────────────────────────────────

_tokenizer = None
_model     = None


def _load_model() -> tuple:
    """Loads tokenizer + model once, caches them in module-level globals."""
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print(f"[Summarizer] Loading model: {SUMMARIZATION_MODEL}")
        print(f"[Summarizer] Target device: {DEVICE.upper()}")

        _tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)

        _model = AutoModelForSeq2SeqLM.from_pretrained(
            SUMMARIZATION_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)

        _model.eval()   # Disable dropout for deterministic inference

        if DEVICE == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[Summarizer] GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
        else:
            print("[Summarizer] WARNING: CUDA not available — running on CPU (slow).")

    return _tokenizer, _model


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_into_chunks(text: str, tokenizer, max_tokens: int) -> list[str]:
    """
    Splits `text` into sentence-aware chunks, each fitting within
    `max_tokens` tokens (measured by the model tokenizer).
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks    = []
    current   = []
    current_len = 0

    for sentence in sentences:
        s_len = len(tokenizer.encode(sentence, add_special_tokens=False))

        # If a single sentence exceeds the limit, hard-split by words
        if s_len > max_tokens:
            words = sentence.split()
            partial = []
            partial_len = 0
            for word in words:
                w_len = len(tokenizer.encode(word, add_special_tokens=False))
                if partial_len + w_len > max_tokens:
                    if partial:
                        chunks.append(" ".join(partial))
                    partial     = [word]
                    partial_len = w_len
                else:
                    partial.append(word)
                    partial_len += w_len
            if partial:
                current.extend(partial)
                current_len += partial_len
            continue

        if current_len + s_len > max_tokens:
            if current:
                chunks.append(" ".join(current))
            current     = [sentence]
            current_len = s_len
        else:
            current.append(sentence)
            current_len += s_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ── Summarisation ─────────────────────────────────────────────────────────────

def summarize_chunk(chunk: str, tokenizer, model) -> str:
    """
    Summarises a single text chunk using the loaded model on GPU/CPU.
    Uses torch.no_grad() and (on CUDA) torch.autocast for speed + memory.
    """
    # BART expects "summarize: " prefix for some variants; T5 always needs it.
    prefix = "summarize: " if "t5" in SUMMARIZATION_MODEL.lower() else ""
    inputs = tokenizer(
        prefix + chunk,
        return_tensors="pt",
        truncation=True,
        max_length=CHUNK_TOKEN_LIMIT,
    ).to(DEVICE)

    with torch.no_grad():
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output_ids = model.generate(
                    **inputs,
                    min_new_tokens=SUMMARY_MIN_LENGTH,
                    max_new_tokens=SUMMARY_MAX_LENGTH,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                )
        else:
            output_ids = model.generate(
                **inputs,
                min_new_tokens=SUMMARY_MIN_LENGTH,
                max_new_tokens=SUMMARY_MAX_LENGTH,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary.strip()


def summarize_transcript(transcript: str) -> dict:
    """
    Main entry point.
    Returns:
        {
            "full_summary": str,       # all chunk summaries joined
            "chunks"      : list[str], # individual chunk summaries
            "num_chunks"  : int,
        }
    """
    tokenizer, model = _load_model()
    chunks = _split_into_chunks(transcript, tokenizer, CHUNK_TOKEN_LIMIT)

    print(f"[Summarizer] Transcript split into {len(chunks)} chunk(s).")

    summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarising", unit="chunk")):
        summary = summarize_chunk(chunk, tokenizer, model)
        summaries.append(summary)

    # Free GPU cache after inference
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    full_summary = " ".join(summaries)
    print(f"[Summarizer] Done. Summary: {len(full_summary.split())} words.")

    return {
        "full_summary": full_summary,
        "chunks"      : summaries,
        "num_chunks"  : len(summaries),
    }
