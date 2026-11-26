"""Unit tests for evaluation-result and persisted-run integrity."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from ledger.config.settings import JudgeConfig, LedgerConfig
from ledger.core.interfaces import Judge
from ledger.core.models import EvaluationResult, JudgeScore, RunSummary, SystemOutput
from ledger.evaluation.compare import load_run_data
from ledger.evaluation.evaluator import Evaluator
from ledger.regression.comparator import RunData, compare_runs
from ledger.regression.detector import detect_regressions
from ledger.regression.policy import RegressionPolicy
from ledger.tracking.artifacts import ArtifactStore
from tests.helpers import make_case, make_result, make_summary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_evaluation_result_rejects_nonfinite_scores(value: float) -> None:
    payload = make_result("t1").model_dump()
    payload["overall_score"] = value
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_evaluation_result_rejects_nonfinite_latency(value: float) -> None:
    payload = make_result("t1").model_dump()
    payload["latency_ms"] = value
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metrics", {"overall": math.nan}),
        ("latency", {"mean_ms": math.inf}),
    ],
)
def test_run_summary_rejects_nonfinite_aggregates(field: str, value: object) -> None:
    payload = make_summary("run-a", "sys", [make_result("t1")]).model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        RunSummary.model_validate(payload)


def test_run_summary_rejects_inconsistent_aggregate_counts() -> None:
    payload = make_summary("run-a", "sys", [make_result("t1")]).model_dump()
    payload["counts"]["total"] = 0

    with pytest.raises(ValidationError, match="total does not match"):
        RunSummary.model_validate(payload)


def test_run_summary_rejects_inconsistent_pass_rate() -> None:
    payload = make_summary("run-a", "sys", [make_result("t1", passed=True)]).model_dump()
    payload["metrics"]["pass_rate"] = 0.0

    with pytest.raises(ValidationError, match="pass_rate"):
        RunSummary.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("overall_score", "0.8"),
        ("overall_score", True),
        ("overall_score", 1.01),
        ("overall_score", -0.01),
        ("deterministic_scores", {"token_f1": 1.01}),
        ("deterministic_scores", {"token_f1": -0.01}),
        ("judge_overall", 5.01),
        ("judge_scores", {"correctness": 6}),
    ],
)
def test_evaluation_result_rejects_out_of_range_scores(field: str, value: object) -> None:
    payload = make_result("t1").model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(payload)


def test_run_data_rejects_duplicate_test_ids() -> None:
    results = [
        make_result("t1", run_id="run-a"),
        make_result("t1", overall=0.0, passed=False, run_id="run-a"),
    ]
    summary = make_summary("run-a", "sys", results)

    with pytest.raises(ValueError, match="duplicate result test id"):
        RunData(summary, results)


def test_run_data_rejects_incomplete_result_rows() -> None:
    results = [make_result(f"t{i}", run_id="run-a") for i in range(3)]
    summary = make_summary("run-a", "sys", results)

    with pytest.raises(ValueError, match="summary declares 3 cases but results contain 2"):
        RunData(summary, results[:2])


def test_empty_candidate_run_fails_gate_without_overall_signal() -> None:
    baseline_results = [make_result("t1", run_id="base", system_version="sys-a")]
    baseline = RunData(make_summary("base", "sys-a", baseline_results), baseline_results)
    candidate = RunData(make_summary("candidate", "sys-b", []), [])

    comparison = compare_runs(baseline, candidate)
    decision = detect_regressions(
        comparison, baseline.summary, candidate.summary, RegressionPolicy()
    )

    assert comparison.overall is None
    assert decision.status == "FAIL"
    assert any(violation.rule == "overall_missing" for violation in decision.fails)


def test_run_data_rejects_inconsistent_summary_counts() -> None:
    results = [make_result("t1", passed=False, run_id="run-a")]
    summary = make_summary("run-a", "sys", results)
    summary.counts = summary.counts.model_copy(update={"failed": 0, "not_passed": 0})

    with pytest.raises(ValueError, match="summary counts do not match result rows"):
        RunData(summary, results)


def test_run_data_rejects_inconsistent_summary_latency() -> None:
    results = [make_result("t1", run_id="run-a", latency_ms=10.0)]
    summary = make_summary("run-a", "sys", results)
    summary.latency.mean_ms = 99.0

    with pytest.raises(ValueError, match="run summary latency"):
        RunData(summary, results)


def test_evaluator_recomputes_judge_score_and_pass_from_dimensions() -> None:
    class InconsistentJudge(Judge):
        def score(self, case, output) -> JudgeScore:
            return JudgeScore(
                dimensions=dict.fromkeys(JudgeConfig().dimensions, 1),
                overall=5.0,
                normalized=1.0,
                passed=True,
            )

    evaluator = Evaluator(
        metrics=[],
        judge=InconsistentJudge(),
        judge_config=JudgeConfig(),
    )
    result = evaluator.evaluate_case(
        make_case(), SystemOutput(answer="answer"), run_id="run-a", system_version="sys"
    )

    assert result.judge_overall == 1.0
    assert result.overall_score == 0.0
    assert result.passed is False


def test_evaluator_marks_incomplete_judge_dimensions_as_failed() -> None:
    class IncompleteJudge(Judge):
        def score(self, case, output) -> JudgeScore:
            return JudgeScore(
                dimensions={"correctness": 1},
                overall=1.0,
                normalized=0.0,
                passed=False,
            )

    evaluator = Evaluator(
        metrics=[],
        judge=IncompleteJudge(),
        judge_config=JudgeConfig(on_failure="exclude"),
    )
    result = evaluator.evaluate_case(
        make_case(), SystemOutput(answer="answer"), run_id="run-a", system_version="sys"
    )

    assert result.judge_failed is True
    assert result.passed is None


def test_evaluator_uses_custom_judge_dimensions_without_explicit_config() -> None:
    class CustomJudge(Judge):
        def score(self, case, output) -> JudgeScore:
            return JudgeScore(
                dimensions={"quality": 4},
                overall=1.0,
                normalized=0.0,
                passed=False,
            )

    evaluator = Evaluator(metrics=[], judge=CustomJudge())
    result = evaluator.evaluate_case(
        make_case(), SystemOutput(answer="answer"), run_id="run-a", system_version="sys"
    )

    assert result.judge_failed is False
    assert result.judge_overall == 4.0
    assert result.overall_score == 0.75
    assert result.passed is True


def test_evaluator_still_enforces_explicit_judge_dimensions() -> None:
    class CustomJudge(Judge):
        def score(self, case, output) -> JudgeScore:
            return JudgeScore(dimensions={"quality": 4}, overall=4.0, normalized=0.75, passed=True)

    evaluator = Evaluator(
        metrics=[],
        judge=CustomJudge(),
        judge_config=JudgeConfig(dimensions=["correctness"]),
    )
    result = evaluator.evaluate_case(
        make_case(), SystemOutput(answer="answer"), run_id="run-a", system_version="sys"
    )

    assert result.judge_failed is True


def test_load_run_data_rejects_forged_decision_using_saved_custom_thresholds(tmp_path) -> None:
    config = LedgerConfig.model_validate(
        {
            "run": {"system_version": "sys"},
            "judge": {
                "dimensions": ["quality"],
                "weights": {"quality": 1.0},
                "pass_threshold": 4.5,
                "critical_pass_threshold": 4.8,
            },
        }
    )
    store = ArtifactStore(results_dir=tmp_path / "results")
    valid = make_result(
        "t1",
        overall=0.25,
        passed=False,
        run_id="run-a",
        system_version="sys",
        judge_dims={"quality": 2},
    )
    forged_payload = valid.model_dump(mode="json")
    forged_payload.update(judge_overall=5.0, overall_score=1.0, passed=True)
    forged = EvaluationResult.model_validate(forged_payload)

    store.save_results("run-a", [forged])
    store.save_summary("run-a", make_summary("run-a", "sys", [forged]))
    store.save_config_snapshot("run-a", config.model_dump(mode="json"))

    with pytest.raises(ValueError, match="decision"):
        load_run_data(store, "run-a")


def test_load_run_data_requires_saved_decision_config(tmp_path) -> None:
    store = ArtifactStore(results_dir=tmp_path / "results")
    results = [make_result("t1", run_id="run-a", system_version="sys")]
    store.save_results("run-a", results)
    store.save_summary("run-a", make_summary("run-a", "sys", results))

    with pytest.raises(ValueError, match="no config snapshot"):
        load_run_data(store, "run-a")


def test_load_run_data_accepts_valid_custom_judge_failure_fallback(tmp_path) -> None:
    config = LedgerConfig.model_validate(
        {
            "run": {"system_version": "sys"},
            "judge": {
                "dimensions": ["quality"],
                "weights": {"quality": 1.0},
                "pass_threshold": 4.5,
                "critical_pass_threshold": 4.8,
                "on_failure": "fail_case",
            },
            "evaluation": {"fallback_pass_threshold": 0.6},
        }
    )
    store = ArtifactStore(results_dir=tmp_path / "results")
    results = [
        make_result(
            "healthy",
            overall=1.0,
            passed=True,
            run_id="run-a",
            system_version="sys",
            judge_dims={"quality": 5},
        ),
        make_result(
            "judge-failed",
            overall=0.9,
            passed=False,
            run_id="run-a",
            system_version="sys",
            det={"token_f1": 0.9},
            judge_failed=True,
        ),
    ]
    store.save_results("run-a", results)
    store.save_summary("run-a", make_summary("run-a", "sys", results))
    store.save_config_snapshot("run-a", config.model_dump(mode="json"))

    loaded = load_run_data(store, "run-a")
    assert [result.test_id for result in loaded.results] == ["healthy", "judge-failed"]


def test_artifact_store_rejects_nonfinite_serialized_score(tmp_path) -> None:
    store = ArtifactStore(results_dir=tmp_path / "results")
    directory = store.ensure_run_dir("run-bad")
    payload = make_result("t1", run_id="run-bad").model_dump(mode="json")
    payload["overall_score"] = math.nan
    (directory / "results.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt result row"):
        store.load_results("run-bad")


def test_artifact_store_rejects_duplicate_result_ids(tmp_path) -> None:
    store = ArtifactStore(results_dir=tmp_path / "results")
    results = [make_result("t1"), make_result("t1", overall=0.0)]

    with pytest.raises(ValueError, match="duplicate result test id"):
        store.save_results("run-duplicate", results)
