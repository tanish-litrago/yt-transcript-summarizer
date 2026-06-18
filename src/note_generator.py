"""
src/note_generator.py  — v1.1 Fixed
─────────────────────────────────────
Organises summarisation output into structured Markdown notes.
Fixed: date formatting, duplicate headings, meta line rendering.
"""

import os
import sys
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _split_into_sentences(text: str) -> list:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


def _sentences_to_bullets(sentences: list) -> list:
    return [f"- {s}" for s in sentences]


def _auto_section_title(chunk_idx: int, total: int) -> str:
    if total == 1:
        return "Summary"
    pct = int((chunk_idx / max(total - 1, 1)) * 100)
    labels = {
        range(0,  20) : "Introduction",
        range(20, 40) : "Core Concepts",
        range(40, 60) : "Main Discussion",
        range(60, 80) : "Key Insights",
        range(80, 101): "Conclusion",
    }
    for rng, label in labels.items():
        if pct in rng:
            return f"Section {chunk_idx + 1} -- {label}"
    return f"Section {chunk_idx + 1}"


def generate_notes(
    video_url    : str,
    video_id     : str,
    summary_data : dict,
    keyword_data : dict,
    title        : str = "",
) -> str:
    title_str   = title if title else f"YouTube Video -- {video_id}"
    # Full date and time — no truncation
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M")
    chunks      = summary_data.get("chunks", [summary_data.get("full_summary", "")])
    combined_kw = keyword_data.get("combined", [])
    entities    = keyword_data.get("entities", [])

    lines = []

    # Header
    lines.append(f"# {title_str}")
    lines.append("")
    lines.append(f"**Source:** {video_url}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Keywords
    if combined_kw:
        lines.append("## Key Terms & Entities")
        lines.append("")
        kw_line = "  ".join(f"`{k}`" for k in combined_kw[:15])
        lines.append(kw_line)
        lines.append("")

    # Named Entities
    if entities:
        lines.append("## Named Entities")
        lines.append("")
        for ent in entities[:10]:
            lines.append(f"- **{ent}**")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Notes sections
    lines.append("## Notes")
    lines.append("")

    total = len(chunks)
    for idx, chunk_summary in enumerate(chunks):
        section_title = _auto_section_title(idx, total)
        lines.append(f"### {section_title}")
        lines.append("")
        sentences = _split_into_sentences(chunk_summary)
        if sentences:
            lines.extend(_sentences_to_bullets(sentences))
        else:
            lines.append(f"- {chunk_summary.strip()}")
        lines.append("")

    # Full summary
    lines.append("---")
    lines.append("")
    lines.append("## Full Summary")
    lines.append("")
    lines.append(summary_data.get("full_summary", "").strip())
    lines.append("")

    return "\n".join(lines)
