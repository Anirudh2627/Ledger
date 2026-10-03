"""Integration tests: full evaluation pipeline (offline, mock providers)."""

from __future__ import annotations

import json

import pytest

from ledger.dataset.loader import DatasetError
from ledger.evaluation.pipeline import run_evaluation
from tests.helpers import write_config_file

pytestmark = pytest.mark.integration


def test_pipeline_end_to_end_writes_artifacts(tmp_path, make_config) -> None:
    config = make_config()
    artifacts = run_evaluation(config, progress=False)

    # artifacts on disk
    assert artifacts.run_dir.is_dir()
    assert artifacts.results_path.is_file()
    assert artifacts.summary_path.is_file()
    assert (artifacts.run_dir / "config.snapshot.json").is_file()

    # results.jsonl: one row per case, parseable, fully populated
    rows = [json.loads(line) for line in artifacts.results_path.read_text().splitlines() if line]
    assert len(rows) == 6
    required_fields = {
        "test_id",
        "run_id",
        "system_version",
        "category",
        "critical",
        "timestamp",
        "question",
        "actual_answer",
        "deterministic_scores",
        "judge_scores",
        "overall_score",
        "passed",
        "model_name",
        "prompt_version",
    }
    for row in rows:
        assert required_fields <= set(row)
        assert row["run_id"] == artifacts.run_id

    # summary consistency
    summary = artifacts.summary
    assert summary.dataset.n_cases == 6
    assert summary.provider.is_mock
    assert summary.judge_enabled
    assert 0.0 <= summary.metrics["overall"] <= 1.0
    assert 0.0 <= summary.metrics["pass_rate"] <= 1.0
    assert summary.counts.total == 6
    assert summary.counts.passed + summary.counts.failed <= 6
    assert summary.runtime["ledger_version"]

    # experiment index recorded
    index = tmp_path / "experiments" / "index.jsonl"
    assert index.is_file()
    entry = json.loads(index.read_text().strip())
    assert entry["run_id"] == artifacts.run_id


def test_pipeline_mock_sut_answers_reference(tmp_path) -> None:
    """MockSystem in reference mode must pass (plumbing upper bound)."""
    from ledger.config.settings import load_config

    path = write_config_file(
        tmp_path,
        {"system": {"kind": "mock", "mode": "reference"}, "run": {"system_version": "upper"}},
    )
    artifacts = run_evaluation(load_config(path), progress=False)
    assert artifacts.summary.metrics["pass_rate"] == pytest.approx(1.0)
    # upper bound modulo the mock judge's known lexical biases (format penalty
    # on the JSON case, neutral groundedness without context)
    assert artifacts.summary.metrics["overall"] > 0.75


def test_pipeline_degraded_mock_sut_scores_low(tmp_path) -> None:
    from ledger.config.settings import load_config

    path = write_config_file(
        tmp_path,
        {"system": {"kind": "mock", "mode": "degraded"}, "run": {"system_version": "lower"}},
    )
    artifacts = run_evaluation(load_config(path), progress=False)
    assert artifacts.summary.metrics["pass_rate"] <= 0.5


def test_pipeline_is_deterministic(make_config) -> None:
    """Same config + seed => identical answers and metrics across runs."""
    config = make_config()
    first = run_evaluation(config, progress=False)
    second = run_evaluation(config, progress=False)

    answers_a = [(r.test_id, r.actual_answer) for r in first.results]
    answers_b = [(r.test_id, r.actual_answer) for r in second.results]
    assert answers_a == answers_b
    assert first.summary.metrics == second.summary.metrics
    assert first.run_id != second.run_id  # ids stay unique


def test_pipeline_limit_and_category_filter(make_config, mini_dataset_path) -> None:
    config = make_config()
    limited = run_evaluation(config, limit=2, progress=False)
    assert len(limited.results) == 2

    filtered = run_evaluation(config, dataset_path=mini_dataset_path, progress=False)
    assert len(filtered.results) == 6


def test_pipeline_judge_disabled_falls_back_to_deterministic(tmp_path) -> None:
    from ledger.config.settings import load_config

    path = write_config_file(tmp_path, {"judge": {"enabled": False}})
    artifacts = run_evaluation(load_config(path), progress=False)
    summary = artifacts.summary
    assert summary.judge_enabled is False
    assert summary.counts.judge_failed == 0
    for result in artifacts.results:
        assert result.judge_scores is None
        assert result.judge_failed is False
        # overall now comes from the deterministic composite
        applicable = [v for v in result.deterministic_scores.values() if v is not None]
        if applicable:
            assert result.overall_score == pytest.approx(
                sum(applicable) / len(applicable), abs=1e-6
            )


def test_pipeline_sut_error_recorded_not_fatal(tmp_path, monkeypatch) -> None:
    """A system that always raises must yield errored results, not a crash."""
    import ledger.systems.mock as mock_module
    from ledger.config.settings import load_config
    from ledger.systems.mock import MockSystem

    class ExplodingSystem(MockSystem):
        def generate(self, test_case):
            raise RuntimeError("SUT exploded")

    monkeypatch.setattr(mock_module, "MockSystem", ExplodingSystem)
    path = write_config_file(
        tmp_path,
        {"system": {"kind": "mock", "mode": "reference"}, "run": {"sut_retries": 1}},
    )
    artifacts = run_evaluation(load_config(path), progress=False)

    assert artifacts.summary.counts.errored == 6
    assert artifacts.summary.counts.passed == 0
    for result in artifacts.results:
        assert result.error is not None and "exploded" in result.error
        assert result.passed is False
        assert result.overall_score == 0.0


def test_pipeline_rejects_broken_dataset(tmp_path, make_config) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text("", encoding="utf-8")  # zero cases
    config = make_config()
    with pytest.raises(DatasetError):
        run_evaluation(config, dataset_path=broken, progress=False)


def test_pipeline_rejects_invalid_dataset_content(tmp_path, make_config) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "question": "q", "category": "c"}\n', encoding="utf-8")
    with pytest.raises(DatasetError):
        run_evaluation(make_config(), dataset_path=bad, progress=False)


def test_pipeline_records_prompt_version_and_retrieved_context(make_config) -> None:
    artifacts = run_evaluation(make_config(), progress=False)
    rag_results = [r for r in artifacts.results if r.retrieved_context]
    assert rag_results, "RAG system should expose retrieved context"
    for result in artifacts.results:
        assert result.prompt_version == "v1_grounding"
        assert result.model_name == "mock-test-llm"
