"""
Validates the analyzer.py functions using pure-Python logic only —
spaCy-dependent functions are tested for graceful fallback behaviour
so CI doesn't require the en_core_web_sm model to be installed.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.analyzer import (
    word_frequency,
    compression_ratio,
    readability,
    sentiment_timeline,
    speaking_pace,
)


SAMPLE_TEXT = (
    "Neural networks learn from data using weights and biases. "
    "The network adjusts its weights through a process called backpropagation. "
    "This is a great and powerful way to solve difficult problems."
)


def test_word_frequency_returns_list_of_dicts():
    result = word_frequency(SAMPLE_TEXT, top_n=5)
    assert isinstance(result, list)
    assert len(result) <= 5
    for item in result:
        assert "word" in item and "count" in item


def test_word_frequency_excludes_stopwords():
    result = word_frequency("the a an and the the the", top_n=10)
    assert result == []


def test_compression_ratio_basic():
    original = "word " * 100
    summary  = "word " * 10
    result = compression_ratio(original, summary)
    assert result["original_words"] == 100
    assert result["summary_words"] == 10
    assert result["compression_pct"] == 90.0


def test_compression_ratio_handles_empty_summary():
    result = compression_ratio("word " * 50, "")
    assert result["compression_pct"] == 100.0


def test_readability_returns_expected_keys():
    result = readability(SAMPLE_TEXT)
    assert "flesch_reading_ease" in result
    assert "flesch_kincaid_grade" in result
    assert "reading_level" in result


def test_sentiment_timeline_length_matches_chunks():
    chunks = ["This is great and useful.", "This is a difficult problem."]
    result = sentiment_timeline(chunks)
    assert len(result) == len(chunks)
    for item in result:
        assert item["label"] in ("Positive", "Negative", "Neutral")


def test_speaking_pace_valid_duration():
    transcript = "word " * 300
    result = speaking_pace(transcript, "2:00")
    assert result["wpm"] == 150
    assert result["pace"] in ("Slow", "Normal", "Fast", "Very Fast")


def test_speaking_pace_invalid_duration_returns_unknown():
    result = speaking_pace("some text", "not-a-duration")
    assert result["pace"] == "Unknown"
