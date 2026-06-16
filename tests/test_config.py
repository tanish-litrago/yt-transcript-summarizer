"""
Validates the structure and content of config.py values — no ML libraries needed.
"""

import sys
import os
import types
import importlib
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    torch_stub = types.ModuleType("torch")
    cuda_stub = types.ModuleType("torch.cuda")
    cuda_stub.is_available = lambda: False
    torch_stub.cuda = cuda_stub
    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("torch.cuda", cuda_stub)
    sys.modules.pop("config", None)
    sys.path.insert(0, ROOT_DIR)
    return importlib.import_module("config")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_output_dir_under_project(cfg):
    assert ROOT_DIR in cfg.OUTPUT_DIR


def test_models_dir_under_project(cfg):
    assert ROOT_DIR in cfg.MODELS_DIR


def test_summarization_model_is_nonempty_string(cfg):
    assert isinstance(cfg.SUMMARIZATION_MODEL, str)
    assert len(cfg.SUMMARIZATION_MODEL) > 0


def test_whisper_model_valid_size(cfg):
    valid_sizes = {"tiny", "base", "small", "medium", "large"}
    assert cfg.WHISPER_MODEL in valid_sizes


def test_top_keywords_positive(cfg):
    assert isinstance(cfg.TOP_KEYWORDS, int)
    assert cfg.TOP_KEYWORDS > 0
