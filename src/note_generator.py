"""
src/note_generator.py
─────────────────────
Organises summarisation output into structured Markdown notes:

  # Video Title (or ID)
  **Source:** <url>
  **Keywords:** …

  ## Section 1
  - <bullet point from summary>
  - …

  ## Section 2
  …
"""

import os
import sys
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """Splits a paragraph into individual sentences."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


def _sentences_to_bullets(sentences: list[str]) -> list[str]:
    """Formats a list of sentences as Markdown bullet points."""
    return [f"- {s}" for s in sentences]


def _auto_section_title(chunk_idx: int, total: int) -> str:
    """
    Generates a section heading when no chapter markers are present.
    For single-chunk videos it returns 'Summary'.
    """
    if total == 1:
        return "Summary"
    # Estimate rough position in video as a percentage
    pct = int((chunk_idx / max(total - 1, 1)) * 100)
    labels = {
        range(0, 20)  : "Introduction",
        range(20, 40) : "Core Concepts",
        range(40, 60) : "Main Discussion",
        range(60, 80) : "Key Insights",
        range(80, 101): "Conclusion",
    }
    for rng, label in labels.items():
        if pct in rng:
            return f"Section {chunk_idx + 1} — {label}"
    return f"Section {chunk_idx + 1}"


# ── Main function ─────────────────────────────────────────────────────────────

def generate_notes(
    video_url    : str,
    video_id     : str,
    summary_data : dict,      # output of summarizer.summarize_transcript()
    keyword_data : dict,      # output of keyword_extractor.extract_keywords()
    title        : str = "",
) -> str:
    """
    Builds a full Markdown notes document and returns it as a string.

    Args:
        video_url    : Original YouTube URL.
        video_id     : 11-char YouTube video ID.
        summary_data : {'full_summary': str, 'chunks': list[str], …}
        keyword_data : {'keywords': list, 'entities': list, 'combined': list}
        title        : Optional video title.

    Returns:
        Markdown string ready to display or write to file.
    """
    title_str   = title if title else f"YouTube Video — {video_id}"
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M")
    chunks      = summary_data.get("chunks", [summary_data.get("full_summary", "")])
    combined_kw = keyword_data.get("combined", [])
    entities    = keyword_data.get("entities", [])

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"# {title_str}")
    lines.append("")
    lines.append(f"**Source:** {video_url}")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Model:** CUDA-accelerated BART/T5 via HuggingFace Transformers")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Keywords ──────────────────────────────────────────────────────────────
    if combined_kw:
        lines.append("## 🔑 Key Terms & Entities")
        lines.append("")
        kw_line = "  ".join(f"`{k}`" for k in combined_kw[:15])
        lines.append(kw_line)
        lines.append("")

    if entities:
        lines.append("## 👤 Named Entities")
        lines.append("")
        for ent in entities[:10]:
            lines.append(f"- **{ent}**")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── Section-by-section notes ──────────────────────────────────────────────
    lines.append("## 📝 Notes")
    lines.append("")

    total = len(chunks)
    for idx, chunk_summary in enumerate(chunks):
        section_title = _auto_section_title(idx, total)
        lines.append(f"### {section_title}")
        lines.append("")

        sentences = _split_into_sentences(chunk_summary)
        if sentences:
            bullets = _sentences_to_bullets(sentences)
            lines.extend(bullets)
        else:
            lines.append(f"- {chunk_summary.strip()}")

        lines.append("")

    # ── Full summary (condensed single paragraph) ─────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 📄 Full Summary")
    lines.append("")
    lines.append(summary_data.get("full_summary", "").strip())
    lines.append("")

    return "\n".join(lines)
