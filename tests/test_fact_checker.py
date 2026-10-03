"""
tests/test_fact_checker.py
Tests for src/fact_checker.py (v3.0 fact-checking pipeline).

All Ollama and DuckDuckGo calls are mocked — tests run in CI without either.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExtractJson(unittest.TestCase):
    """_extract_json handles clean JSON and messy LLM output."""

    def setUp(self):
        from src.fact_checker import _extract_json
        self._fn = _extract_json

    def test_clean_array(self):
        result = self._fn('["claim one", "claim two"]')
        self.assertEqual(result, ["claim one", "claim two"])

    def test_clean_object(self):
        result = self._fn('{"verdict": "supported", "explanation": "ok"}')
        self.assertEqual(result["verdict"], "supported")

    def test_wrapped_in_prose(self):
        messy = 'Here are the claims:\n["alpha", "beta"]\nDone.'
        result = self._fn(messy)
        self.assertIsInstance(result, list)
        self.assertIn("alpha", result)

    def test_invalid_returns_none(self):
        result = self._fn("not json at all")
        self.assertIsNone(result)


class TestVerifyClaimNoHits(unittest.TestCase):
    """verify_claim returns unverifiable when search returns nothing."""

    def test_empty_hits(self):
        from src.fact_checker import verify_claim
        result = verify_claim("Some claim", [])
        self.assertEqual(result["verdict"], "unverifiable")
        self.assertEqual(result["sources"], [])


class TestVerifyClaimGemmaResponse(unittest.TestCase):
    """verify_claim correctly parses Gemma's verdict."""

    def test_supported_verdict(self):
        from src.fact_checker import verify_claim

        hits = [{"title": "Source A", "url": "https://a.com", "snippet": "..."}]
        mock_response = '{"verdict": "supported", "explanation": "Source A confirms."}'

        with patch("src.fact_checker._call_gemma", return_value=mock_response):
            result = verify_claim("Earth orbits the Sun", hits)

        self.assertEqual(result["verdict"], "supported")
        self.assertIn("Source A", result["explanation"])
        self.assertEqual(len(result["sources"]), 1)

    def test_unknown_verdict_becomes_unverifiable(self):
        from src.fact_checker import verify_claim

        hits = [{"title": "X", "url": "https://x.com", "snippet": "..."}]
        mock_response = '{"verdict": "maybe", "explanation": "unclear"}'

        with patch("src.fact_checker._call_gemma", return_value=mock_response):
            result = verify_claim("Some claim", hits)

        self.assertEqual(result["verdict"], "unverifiable")


class TestCacheRoundTrip(unittest.TestCase):
    """Cache save/load round-trip works correctly."""

    def test_save_and_load(self):
        from src.fact_checker import _save_cache, _load_cached

        data = [
            {"claim": "c1", "verdict": "supported", "explanation": "ok", "sources": []},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.fact_checker.FACT_CHECK_DIR", tmpdir):
                _save_cache("vid123", data)
                loaded = _load_cached("vid123")

        self.assertEqual(loaded, data)

    def test_load_nonexistent_returns_none(self):
        from src.fact_checker import _load_cached

        with patch("src.fact_checker.FACT_CHECK_DIR", "/nonexistent/path/xyz"):
            result = _load_cached("does_not_exist")

        self.assertIsNone(result)


class TestFactCheckUsesCacheFirst(unittest.TestCase):
    """fact_check() returns cached results without calling Gemma or search."""

    def test_cache_hit_skips_gemma(self):
        from src.fact_checker import fact_check

        cached = [{"claim": "cached", "verdict": "supported",
                   "explanation": "cached", "sources": []}]

        with patch("src.fact_checker._load_cached", return_value=cached) as mock_load, \
             patch("src.fact_checker.extract_claims") as mock_extract:
            result = fact_check("any summary", "vid_cached")

        mock_load.assert_called_once_with("vid_cached")
        mock_extract.assert_not_called()
        self.assertEqual(result, cached)


if __name__ == "__main__":
    unittest.main()
