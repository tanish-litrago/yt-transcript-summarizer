"""
Tests that all Python source files parse as valid syntax and that
config constants have the expected types. Heavy ML libraries (torch,
transformers, spacy, whisper) are mocked so they never need to be installed
in CI.
"""

import ast
import os
import sys
import types
import importlib
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Syntax checks
# ---------------------------------------------------------------------------

def _python_files(directory):
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


@pytest.mark.parametrize("filepath", list(_python_files(SRC_DIR)))
def test_src_syntax(filepath):
    with open(filepath, "r", encoding="utf-8") as fh:
        source = fh.read()
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"Syntax error in {filepath}: {exc}")


def test_app_syntax():
    path = os.path.join(ROOT_DIR, "app.py")
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"Syntax error in app.py: {exc}")


def test_config_syntax():
    path = os.path.join(ROOT_DIR, "config.py")
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"Syntax error in config.py: {exc}")


# ---------------------------------------------------------------------------
# Requirements file is well-formed
# ---------------------------------------------------------------------------

def test_requirements_parseable():
    req_path = os.path.join(ROOT_DIR, "requirements.txt")
    assert os.path.isfile(req_path), "requirements.txt not found"
    with open(req_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Each non-comment line must contain a package name (no spaces at start)
        assert not line.startswith(" "), f"Unexpected indentation in requirements.txt: {line!r}"


# ---------------------------------------------------------------------------
# Config constants — loaded with torch mocked out
# ---------------------------------------------------------------------------

def _load_config_with_mock():
    """Import config.py after injecting a minimal torch stub so CI doesn't
    need PyTorch installed."""
    torch_stub = types.ModuleType("torch")
    cuda_stub = types.ModuleType("torch.cuda")
    cuda_stub.is_available = lambda: False
    torch_stub.cuda = cuda_stub

    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("torch.cuda", cuda_stub)

    # Force a fresh import of config
    sys.modules.pop("config", None)
    sys.path.insert(0, ROOT_DIR)
    return importlib.import_module("config")


def test_config_device_is_string():
    cfg = _load_config_with_mock()
    assert isinstance(cfg.DEVICE, str)
    assert cfg.DEVICE in ("cuda", "cpu")


def test_config_paths_are_strings():
    cfg = _load_config_with_mock()
    assert isinstance(cfg.OUTPUT_DIR, str)
    assert isinstance(cfg.MODELS_DIR, str)
    assert isinstance(cfg.BASE_DIR, str)


def test_config_ollama_params():
    cfg = _load_config_with_mock()
    assert isinstance(cfg.OLLAMA_HOST, str)
    assert cfg.OLLAMA_HOST.startswith("http")
    assert isinstance(cfg.GEMMA_MODEL, str)
    assert len(cfg.GEMMA_MODEL) > 0


def test_config_rag_params():
    """v2.5 — RAG constants exist and are correctly typed."""
    cfg = _load_config_with_mock()
    assert isinstance(cfg.CHROMA_DIR, str),   "CHROMA_DIR must be a str path"
    assert isinstance(cfg.EMBED_MODEL, str),  "EMBED_MODEL must be a str"
    assert len(cfg.EMBED_MODEL) > 0,          "EMBED_MODEL must not be empty"
    assert isinstance(cfg.RAG_CHUNK_SIZE, int)
    assert cfg.RAG_CHUNK_SIZE > 0
    assert isinstance(cfg.RAG_CHUNK_OVERLAP, int)
    assert cfg.RAG_CHUNK_OVERLAP >= 0
    assert isinstance(cfg.RAG_TOP_K, int)
    assert cfg.RAG_TOP_K > 0
