"""
src/keyword_extractor.py
────────────────────────
Extracts:
  - Named entities  (people, organisations, locations, etc.) via spaCy NER
  - Top keywords    via TF-IDF scoring (no sklearn dependency — pure Python)

Both are merged and deduplicated for use in note headers and highlights.
"""

import os
import sys
import re
import math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPACY_MODEL, TOP_KEYWORDS


# ── spaCy loader (singleton) ──────────────────────────────────────────────────

_nlp = None


def _load_spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(SPACY_MODEL)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{SPACY_MODEL}' not found.\n"
                f"Run:  python -m spacy download {SPACY_MODEL}"
            )
    return _nlp


# ── TF-IDF (pure Python, no sklearn needed) ───────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "this", "that", "it", "is",
    "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "not", "no", "so", "if", "as", "up", "out", "he",
    "she", "they", "we", "you", "i", "me", "us", "him", "her", "them",
    "his", "its", "our", "their", "my", "your", "also", "just", "more",
    "very", "about", "like", "than", "then", "when", "where", "which",
    "who", "what", "how", "all", "any", "each", "into", "over", "such",
    "even", "only", "now", "well", "back", "after", "here", "there",
    "because", "through", "during", "before", "both", "these", "those",
}


def _tokenize(text: str) -> list[str]:
    """Lowercases and splits text into alpha tokens of length >= 3."""
    return [
        w for w in re.findall(r"[a-z]{3,}", text.lower())
        if w not in _STOPWORDS
    ]


def _tfidf_keywords(text: str, top_n: int) -> list[str]:
    """
    Computes TF-IDF across sentences of `text` and returns the top N terms.
    Treats each sentence as a 'document'.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    if not sentences:
        return []

    # Term frequency per sentence
    tf_per_doc = [Counter(_tokenize(s)) for s in sentences]

    # Document frequency
    all_terms = set(t for c in tf_per_doc for t in c)
    df = {term: sum(1 for c in tf_per_doc if term in c) for term in all_terms}

    n_docs = len(sentences)
    scores: dict[str, float] = {}

    for term in all_terms:
        tf_avg = sum(c.get(term, 0) for c in tf_per_doc) / n_docs
        idf    = math.log((n_docs + 1) / (df[term] + 1)) + 1
        scores[term] = tf_avg * idf

    top = sorted(scores, key=lambda t: scores[t], reverse=True)[:top_n]
    return top


# ── Named entity extraction ───────────────────────────────────────────────────

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT", "LAW", "NORP"}


def _extract_entities(text: str) -> list[str]:
    """Returns unique named entities (people, orgs, places, products…)."""
    nlp = _load_spacy()
    # spaCy processes up to 100k chars by default — limit to avoid memory issues
    doc     = nlp(text[:100_000])
    seen    = set()
    entities = []
    for ent in doc.ents:
        label = ent.label_
        norm  = ent.text.strip()
        if label in _ENTITY_LABELS and norm.lower() not in seen and len(norm) > 1:
            seen.add(norm.lower())
            entities.append(norm)
    return entities


# ── Public API ────────────────────────────────────────────────────────────────

def extract_keywords(text: str) -> dict:
    """
    Returns:
        {
            "keywords" : list[str],   # TF-IDF top keywords
            "entities" : list[str],   # Named entities
            "combined" : list[str],   # Merged, deduplicated, entities first
        }
    """
    keywords = _tfidf_keywords(text, TOP_KEYWORDS)
    entities = _extract_entities(text)

    # Merge: entities first (higher priority), then TF-IDF keywords
    seen     = set()
    combined = []
    for item in entities + keywords:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            combined.append(item)

    print(f"[Keywords] Found {len(entities)} entities, {len(keywords)} TF-IDF keywords.")
    return {
        "keywords" : keywords,
        "entities" : entities,
        "combined" : combined,
    }
