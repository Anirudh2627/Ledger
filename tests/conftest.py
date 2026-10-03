"""Shared pytest fixtures.

All tests run fully offline: the mock provider backs every pipeline test, so
no test requires an API key or network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import (
    FIXTURES,
    REPO_ROOT,
    ScriptedProvider,
    base_config_dict,
    deep_merge,
    make_case,
    make_chunk,
    make_judge_score,
    make_output,
    make_result,
    make_summary,
    write_config_file,
    write_jsonl,
)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def mini_dataset_path() -> Path:
    return FIXTURES / "mini_dataset.jsonl"


@pytest.fixture
def mini_corpus_dir() -> Path:
    return FIXTURES / "corpus"


@pytest.fixture
def mini_labels_path() -> Path:
    return FIXTURES / "human_labels_mini.jsonl"


@pytest.fixture
def make_config_file(tmp_path: Path):
    """Factory: write a (optionally overridden) YAML config into tmp_path."""

    def _make(overrides: dict | None = None, name: str = "config.yaml") -> Path:
        return write_config_file(tmp_path, overrides, name)

    return _make


@pytest.fixture
def make_config(make_config_file):
    """Factory: a loaded LedgerConfig (mock mode, tmp outputs)."""
    from ledger.config.settings import load_config

    def _make(overrides: dict | None = None):
        return load_config(make_config_file(overrides))

    return _make


@pytest.fixture
def result_factory():
    return make_result


@pytest.fixture
def summary_factory():
    return make_summary


@pytest.fixture
def case_factory():
    return make_case


@pytest.fixture
def scripted_provider():
    return ScriptedProvider


# Re-export helpers used directly by some tests.
__all__ = [
    "ScriptedProvider",
    "base_config_dict",
    "deep_merge",
    "make_case",
    "make_chunk",
    "make_judge_score",
    "make_output",
    "make_result",
    "make_summary",
    "write_config_file",
    "write_jsonl",
]
