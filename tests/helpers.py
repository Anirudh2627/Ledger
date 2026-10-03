"""Reusable factories for tests (importable via ``from tests.helpers import ...``).

``tests/`` is added to mypy_path and to sys.path by pytest's rootdir
handling, so this module imports directly.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ledger.core.interfaces import LLMProvider
from ledger.core.models import (
    CompletionRequest,
    CompletionResponse,
    DatasetInfo,
    EvaluationResult,
    JudgeScore,
    ProviderInfo,
    RetrievedChunk,
    RunSummary,
    SystemOutput,
)
from ledger.dataset.schema import TestCase
from ledger.metrics.aggregate import category_aggregates, count_stats, latency_stats, metric_means

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------


def base_config_dict(tmp_path: Path) -> dict[str, Any]:
    """A hermetic mock-mode config rooted in ``tmp_path``."""
    return {
        "config_version": 1,
        "run": {
            "system_version": "test-system",
            "seed": 7,
            "max_concurrency": 2,
            "sut_retries": 0,
        },
        "dataset": {"path": str(FIXTURES / "mini_dataset.jsonl")},
        "system": {
            "kind": "rag",
            "corpus_dir": str(FIXTURES / "corpus"),
            "chunk_size": 400,
            "chunk_overlap": 40,
            "top_k": 3,
            "prompt_version": "v1_grounding",
            "embedder": {"name": "hashing_tfidf", "dim": 256},
        },
        "provider": {"name": "mock", "model": "mock-test-llm"},
        "judge": {
            "enabled": True,
            "provider": {"name": "mock", "model": "mock-test-judge"},
            "pass_threshold": 3.5,
            "critical_pass_threshold": 4.0,
            "on_failure": "exclude",
            "max_retries": 1,
        },
        "policy_path": None,
        "output": {
            "results_dir": str(tmp_path / "results"),
            "reports_dir": str(tmp_path / "reports"),
            "experiments_dir": str(tmp_path / "experiments"),
        },
        "logging": {"level": "WARNING"},
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Test-side merge mirroring the loader semantics (system kind replaces)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            if key == "system" and "kind" in value and merged[key].get("kind") != value.get("kind"):
                merged[key] = copy.deepcopy(value)
            else:
                merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def write_config_file(
    tmp_path: Path, overrides: dict[str, Any] | None = None, name: str = "config.yaml"
) -> Path:
    data = deep_merge(base_config_dict(tmp_path), overrides or {})
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Domain factories
# ---------------------------------------------------------------------------


def make_case(
    test_id: str = "t1",
    *,
    question: str = "What is the answer?",
    category: str = "factual_correctness",
    critical: bool = False,
    reference: str | None = "The answer is 42.",
    behavior: str | None = "Answer accurately from the documentation.",
    **kwargs: Any,
) -> TestCase:
    return TestCase(
        id=test_id,
        question=question,
        category=category,
        critical=critical,
        reference_answer=reference,
        expected_behavior=behavior,
        **kwargs,
    )


def make_output(answer: str = "The answer is 42.", **kwargs: Any) -> SystemOutput:
    return SystemOutput(answer=answer, **kwargs)


def make_chunk(doc_id: str = "doc.md", index: int = 0, text: str = "chunk text") -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc_id, chunk_index=index, text=text, score=0.9)


def make_result(
    test_id: str,
    *,
    overall: float | None = 0.8,
    passed: bool | None = True,
    category: str = "factual_correctness",
    critical: bool = False,
    run_id: str = "run-x",
    system_version: str = "sys",
    det: dict[str, float | None] | None = None,
    judge_dims: dict[str, int] | None = None,
    judge_failed: bool = False,
    error: str | None = None,
    latency_ms: float | None = 100.0,
) -> EvaluationResult:
    """A persisted-style result for comparator/detector/report tests."""
    judge_overall = None
    if judge_dims:
        judge_overall = sum(judge_dims.values()) / len(judge_dims)
    return EvaluationResult(
        test_id=test_id,
        run_id=run_id,
        system_version=system_version,
        category=category,
        critical=critical,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        question=f"question for {test_id}",
        reference_answer="reference",
        actual_answer="answer",
        latency_ms=latency_ms,
        error=error,
        deterministic_scores=det if det is not None else {},
        judge_scores=judge_dims,
        judge_overall=judge_overall,
        judge_failed=judge_failed,
        overall_score=overall,
        passed=passed,
    )


def make_summary(
    run_id: str,
    system_version: str,
    results: list[EvaluationResult],
    *,
    fingerprint: str = "fp-dataset",
) -> RunSummary:
    """Build a RunSummary from results using the real aggregation code."""
    return RunSummary(
        run_id=run_id,
        system_version=system_version,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset=DatasetInfo(path="mini.jsonl", fingerprint=fingerprint, n_cases=len(results)),
        provider=ProviderInfo(name="mock", model="mock-test-llm", is_mock=True),
        judge_enabled=True,
        metrics={k: round(v, 6) for k, v in metric_means(results).items()},
        categories=category_aggregates(results),
        counts=count_stats(results),
        latency=latency_stats(results),
    )


def make_judge_score(
    dims: dict[str, int] | None = None,
    *,
    passed: bool = True,
    failed: bool = False,
) -> JudgeScore:
    dims = dims or {
        "correctness": 4,
        "relevance": 4,
        "groundedness": 4,
        "instruction_following": 4,
    }
    overall = sum(dims.values()) / len(dims) if dims and not failed else None
    return JudgeScore(
        dimensions=dims if not failed else {},
        overall=overall,
        normalized=(overall - 1) / 4 if overall is not None else None,
        passed=None if failed else passed,
        failed=failed,
        error="boom" if failed else None,
    )


# ---------------------------------------------------------------------------
# JSONL + scripted provider
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


class ScriptedProvider(LLMProvider):
    """LLMProvider double replaying a fixed list of responses (or exceptions)."""

    name = "scripted"

    def __init__(self, responses: list[str | Exception], model: str = "scripted-judge") -> None:
        self.responses = list(responses)
        self.model = model
        self.requests: list[Any] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        item = self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return CompletionResponse(text=item, model=self.model, finish_reason="stop")
