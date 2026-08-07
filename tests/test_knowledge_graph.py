import os
import sys
import json
import types
import pytest
from unittest.mock import MagicMock, patch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


def _config_stub(tmp_path):
    cfg = types.ModuleType("config")
    cfg.OLLAMA_HOST             = "http://localhost:11434"
    cfg.GEMMA_MODEL             = "gemma4:e4b"
    cfg.KG_DIR                  = str(tmp_path)
    cfg.KG_CONFIDENCE_THRESHOLD = 0.7
    cfg.KG_MAX_NODES            = 30
    return cfg


def _gemma_stub():
    stub = types.ModuleType("src.gemma_engine")
    stub._extract_json       = MagicMock(return_value={
        "nodes": [
            {"id": "transformer", "label": "Transformer", "type": "Model",   "explained": True},
            {"id": "attention",   "label": "Attention",   "type": "Concept", "explained": True},
            {"id": "google",      "label": "Google",      "type": "Organization", "explained": False},
        ],
        "edges": [
            {"source": "transformer", "target": "attention", "relation": "uses",       "confidence": 0.92},
            {"source": "google",      "target": "transformer","relation": "founded_by", "confidence": 0.55},
        ],
    })
    stub._truncate_transcript = MagicMock(side_effect=lambda t, max_words=1200: t)
    return stub


# ---------------------------------------------------------------------------
# 1. _parse_kg splits edges correctly by confidence threshold
# ---------------------------------------------------------------------------

def test_parse_kg_splits_edges_by_confidence(tmp_path):
    sys.modules["config"] = _config_stub(tmp_path)
    sys.modules["src.gemma_engine"] = _gemma_stub()
    sys.modules.pop("src.knowledge_graph", None)

    import src.knowledge_graph as kg

    raw = {
        "nodes": [
            {"id": "a", "label": "Alpha", "type": "Concept", "explained": True},
            {"id": "b", "label": "Beta",  "type": "Person",  "explained": False},
        ],
        "edges": [
            {"source": "a", "target": "b", "relation": "uses",         "confidence": 0.9},
            {"source": "b", "target": "a", "relation": "mentioned_with","confidence": 0.5},
        ],
    }

    result = kg._parse_kg(raw)

    assert len(result["edges"])      == 1, "one strong edge expected"
    assert len(result["weak_edges"]) == 1, "one weak edge expected"
    assert result["edges"][0]["confidence"] >= 0.7
    assert result["weak_edges"][0]["confidence"] < 0.7


# ---------------------------------------------------------------------------
# 2. _parse_kg drops edges referencing unknown node IDs
# ---------------------------------------------------------------------------

def test_parse_kg_drops_orphan_edges(tmp_path):
    sys.modules["config"] = _config_stub(tmp_path)
    sys.modules["src.gemma_engine"] = _gemma_stub()
    sys.modules.pop("src.knowledge_graph", None)

    import src.knowledge_graph as kg

    raw = {
        "nodes": [
            {"id": "x", "label": "X", "type": "Concept", "explained": True},
        ],
        "edges": [
            # 'y' does not exist in nodes — should be silently dropped
            {"source": "x", "target": "y", "relation": "uses", "confidence": 0.95},
        ],
    }

    result = kg._parse_kg(raw)
    assert result["edges"] == []
    assert result["weak_edges"] == []


# ---------------------------------------------------------------------------
# 3. extract_knowledge_graph returns cached result without calling Gemma
# ---------------------------------------------------------------------------

def test_extract_kg_uses_cache(tmp_path):
    sys.modules["config"] = _config_stub(tmp_path)
    sys.modules["src.gemma_engine"] = _gemma_stub()
    sys.modules.pop("src.knowledge_graph", None)

    import src.knowledge_graph as kg

    cached = {
        "nodes":      [{"id": "n1", "label": "Node1", "type": "Concept", "explained": True}],
        "edges":      [{"source": "n1", "target": "n1", "relation": "is_a", "confidence": 0.9}],
        "weak_edges": [],
    }

    # Write a fake cache file
    cache_path = os.path.join(str(tmp_path), "cached_video.json")
    with open(cache_path, "w") as f:
        json.dump(cached, f)

    with patch.object(kg, "_call_gemma_kg") as mock_call:
        result = kg.extract_knowledge_graph(
            transcript     = "some transcript text",
            entities_typed = [],
            video_id       = "cached_video",
        )

    # Gemma should NOT have been called — cache hit
    mock_call.assert_not_called()
    assert result["nodes"] == cached["nodes"]
