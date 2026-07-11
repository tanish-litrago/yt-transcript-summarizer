"""
tests/test_rag_engine.py  -  v2.5
Tests for src/rag_engine.py.

These tests mock all Ollama and ChromaDB calls so they run in CI
without a live Ollama server or installed chromadb package.
"""

import os
import sys
import types
import importlib
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------------
# Helper: build a minimal config stub so rag_engine.py can be imported
# in CI without torch / chromadb / langchain installed.
# ---------------------------------------------------------------------------

def _make_config_stub():
    cfg = types.ModuleType("config")
    cfg.OLLAMA_HOST      = "http://localhost:11434"
    cfg.CHROMA_DIR       = os.path.join(ROOT_DIR, "outputs", "chroma")
    cfg.EMBED_MODEL      = "nomic-embed-text"
    cfg.RAG_CHUNK_SIZE   = 500
    cfg.RAG_CHUNK_OVERLAP = 50
    cfg.RAG_TOP_K        = 4
    return cfg


# ---------------------------------------------------------------------------
# 1. load_vector_store returns None for a non-existent video_id
# ---------------------------------------------------------------------------

def test_load_nonexistent_store_returns_none():
    """load_vector_store should return None when the store dir does not exist."""
    # Inject config stub
    sys.modules["config"] = _make_config_stub()
    sys.modules.pop("src.rag_engine", None)

    import src.rag_engine as rag

    result = rag.load_vector_store("__nonexistent_test_video_id_xyz__")
    assert result is None


# ---------------------------------------------------------------------------
# 2. build_vector_store returns a store and creates the directory (mocked)
# ---------------------------------------------------------------------------

def test_build_vector_store_calls_chroma(tmp_path):
    """
    build_vector_store should call Chroma.from_texts with the chunked transcript.
    ChromaDB and Ollama are mocked -- no real IO or network calls are made.
    """
    sys.modules["config"] = _make_config_stub()
    # Point CHROMA_DIR at a temp dir so no real files are written
    sys.modules["config"].CHROMA_DIR = str(tmp_path)
    sys.modules.pop("src.rag_engine", None)

    # Stub out the heavy lazy-import deps
    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = ["chunk one", "chunk two"]

    mock_chroma_cls = MagicMock()
    mock_store = MagicMock()
    mock_chroma_cls.from_texts.return_value = mock_store

    mock_embeddings = MagicMock()

    import src.rag_engine as rag

    with patch.object(rag, "_get_splitter", return_value=mock_splitter), \
         patch.object(rag, "_get_embeddings", return_value=mock_embeddings), \
         patch.object(rag, "_get_chroma", return_value=mock_chroma_cls):

        store = rag.build_vector_store("This is a test transcript.", "test_video_abc")

    # Chroma.from_texts should have been called once
    mock_chroma_cls.from_texts.assert_called_once()
    call_kwargs = mock_chroma_cls.from_texts.call_args
    texts_arg = call_kwargs[1].get("texts") or call_kwargs[0][0]
    assert texts_arg == ["chunk one", "chunk two"]

    assert store is mock_store


# ---------------------------------------------------------------------------
# 3. answer_question returns the required keys
# ---------------------------------------------------------------------------

def test_answer_question_returns_required_keys(tmp_path):
    """
    answer_question should return a dict with 'answer' and 'sources' keys.
    ChromaDB, Ollama embeddings, and _call_gemma are all mocked.
    """
    sys.modules["config"] = _make_config_stub()
    sys.modules["config"].CHROMA_DIR = str(tmp_path)
    sys.modules.pop("src.rag_engine", None)

    # Create a fake store directory so load_vector_store thinks a store exists
    fake_store_dir = tmp_path / "fake_video_id"
    fake_store_dir.mkdir()
    (fake_store_dir / "chroma.sqlite3").write_text("fake")  # non-empty dir

    mock_doc = MagicMock()
    mock_doc.page_content = "This is a retrieved chunk about machine learning."

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [mock_doc]

    mock_chroma_cls = MagicMock()
    mock_chroma_cls.return_value = mock_store

    mock_embeddings = MagicMock()

    # Pre-inject a src.gemma_engine stub so the `from src.gemma_engine import
    # _call_gemma` inside answer_question resolves without loading the real module.
    gemma_stub = types.ModuleType("src.gemma_engine")
    gemma_stub._call_gemma = MagicMock(return_value="Machine learning is a subset of AI.")
    sys.modules["src.gemma_engine"] = gemma_stub

    import src.rag_engine as rag

    with patch.object(rag, "_get_chroma", return_value=mock_chroma_cls), \
         patch.object(rag, "_get_embeddings", return_value=mock_embeddings):

        result = rag.answer_question("What is machine learning?", "fake_video_id")

    assert "answer" in result,  "answer_question must return 'answer' key"
    assert "sources" in result, "answer_question must return 'sources' key"
    assert isinstance(result["answer"],  str)
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) == 1
    assert "machine learning" in result["sources"][0].lower()
