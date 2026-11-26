"""Unit tests: regression policy application (the gate decision)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ledger.regression.comparator import MetricComparison, RunData, compare_runs
from ledger.regression.detector import _significant, detect_regressions
from ledger.regression.policy import CountPolicy, MetricThreshold, RegressionPolicy, ThresholdGroup
from tests.helpers import make_result, make_summary

pytestmark = pytest.mark.unit

DIMS = {"correctness": 4, "relevance": 4, "groundedness": 4, "instruction_following": 4}


def _runs(
    deltas: list[float],
    *,
    base_passed: list[bool] | None = None,
    cand_passed: list[bool] | None = None,
    critical_idx: tuple[int, ...] = (0, 1),
    det_delta: float | None = None,
    judge_failed_cand: int = 0,
):
    n = len(deltas)
    base_passed = base_passed or [True] * n
    cand_passed = cand_passed or [True] * n
    base_results = [
        make_result(
            f"t{i}",
            overall=0.8,
            passed=base_passed[i],
            critical=i in critical_idx,
            det={"token_f1": 0.6} if det_delta is not None else {},
            judge_dims=DIMS,
            run_id="b",
            system_version="A",
        )
        for i in range(n)
    ]
    cand_results = [
        make_result(
            f"t{i}",
            overall=0.8 + deltas[i],
            passed=cand_passed[i],
            critical=i in critical_idx,
            det={"token_f1": 0.6 + det_delta} if det_delta is not None else {},
            judge_dims=DIMS,
            judge_failed=i < judge_failed_cand,
            run_id="c",
            system_version="B",
        )
        for i in range(n)
    ]
    base = RunData(make_summary("b", "A", base_results), base_results)
    cand = RunData(make_summary("c", "B", cand_results), cand_results)
    return base, cand


def _decide(base, cand, policy=None):
    comparison = compare_runs(base, cand, n_samples=800, seed=11)
    return detect_regressions(comparison, base.summary, cand.summary, policy or RegressionPolicy())


def _compare_and_decide(base, cand, policy=None):
    policy = policy or RegressionPolicy()
    comparison = compare_runs(
        base,
        cand,
        n_samples=policy.bootstrap.n_samples,
        confidence=policy.bootstrap.confidence,
        seed=policy.bootstrap.seed,
    )
    decision = detect_regressions(comparison, base.summary, cand.summary, policy)
    return comparison, decision


def test_identical_runs_pass() -> None:
    base, cand = _runs([0.0] * 12)
    decision = _decide(base, cand)
    assert decision.status == "PASS"
    assert decision.violations == []


def test_significant_overall_regression_fails() -> None:
    base, cand = _runs([-0.1] * 12)
    decision = _decide(base, cand)
    assert decision.status == "FAIL"
    rules = {v.rule for v in decision.fails}
    assert "overall_regression" in rules


def test_improvement_never_fails() -> None:
    base, cand = _runs([+0.15] * 12)
    decision = _decide(base, cand)
    assert decision.status == "PASS"


def test_nonsignificant_degradation_warns_when_significance_required() -> None:
    # mean delta -0.067 (beyond 0.03 threshold) but wildly spread -> CI spans 0
    deltas = [-0.3, +0.2, -0.2, +0.1, -0.25, +0.05, -0.02, +0.01, -0.01, 0.0, +0.02, -0.08]
    base, cand = _runs(deltas)
    decision = _decide(base, cand)
    assert decision.status == "WARNING"
    rules = {v.rule for v in decision.warnings}
    assert "overall_nonsignificant" in rules


def test_nonsignificant_degradation_fails_when_significance_not_required() -> None:
    deltas = [-0.3, +0.2, -0.2, +0.1, -0.25, +0.05, -0.02, +0.01, -0.01, 0.0, +0.02, -0.08]
    base, cand = _runs(deltas)
    policy = RegressionPolicy(require_significance=False)
    decision = _decide(base, cand, policy)
    assert decision.status == "FAIL"
    assert any(v.rule == "overall_regression" for v in decision.fails)


def test_warning_band() -> None:
    # delta -0.02: below fail threshold (0.03) but above warning_fraction*0.03
    base, cand = _runs([-0.02] * 12)
    decision = _decide(base, cand)
    assert decision.status == "WARNING"
    assert any(v.rule == "overall_warning_band" for v in decision.warnings)


@pytest.mark.parametrize(
    ("candidate_score", "expected_status"),
    [(0.095, "PASS"), (0.094, "WARNING")],
)
def test_relative_threshold_warning_band(candidate_score: float, expected_status: str) -> None:
    base_results = [
        make_result(
            f"t{i}",
            overall=0.1,
            passed=True,
            critical=False,
            run_id="b",
            system_version="A",
        )
        for i in range(12)
    ]
    candidate_results = [
        make_result(
            f"t{i}",
            overall=candidate_score,
            passed=True,
            critical=False,
            run_id="c",
            system_version="B",
        )
        for i in range(12)
    ]
    base = RunData(make_summary("b", "A", base_results), base_results)
    candidate = RunData(make_summary("c", "B", candidate_results), candidate_results)
    policy = RegressionPolicy(
        overall=ThresholdGroup(absolute=0.5, relative=0.1, warning_fraction=0.5),
        min_paired_cases=10,
    )

    comparison, decision = _compare_and_decide(base, candidate, policy)

    assert comparison.overall is not None
    expected_delta = candidate_score - 0.1
    assert comparison.overall.relative_delta == pytest.approx(expected_delta / 0.1)
    assert decision.status == expected_status
    if expected_status == "WARNING":
        warning = next(v for v in decision.warnings if v.rule == "overall_warning_band")
        assert "relative" in warning.message


def test_critical_newly_failing_blocks() -> None:
    base, cand = _runs(
        [0.0] * 12,
        cand_passed=[False] + [True] * 11,  # t0 is critical
    )
    decision = _decide(base, cand)
    assert decision.status == "FAIL"
    assert any(v.rule == "critical_failures" for v in decision.fails)


def test_noncritical_newly_failing_within_budget_passes() -> None:
    cand_passed = [True] * 12
    cand_passed[5] = False  # t5 not critical, single flip <= budget 3
    base, cand = _runs([0.0] * 12, cand_passed=cand_passed)
    decision = _decide(base, cand)
    # a single flip triggers no count rule; score deltas are zero
    assert decision.status in {"PASS", "WARNING"}
    assert not any(v.rule == "newly_failing_tests" for v in decision.fails)


def test_too_many_newly_failing_blocks() -> None:
    cand_passed = [True] * 12
    for i in (4, 5, 6, 7):  # non-critical ids, 4 flips > budget 3
        cand_passed[i] = False
    base, cand = _runs([0.0] * 12, cand_passed=cand_passed)
    decision = _decide(base, cand)
    assert decision.status == "FAIL"
    assert any(v.rule == "newly_failing_tests" for v in decision.fails)


def test_absolute_failed_tests_budget() -> None:
    policy = RegressionPolicy.model_validate({"counts": {"max_failed_tests": 2}})
    cand_passed = [True] * 12
    cand_passed[4] = cand_passed[5] = cand_passed[6] = False
    base, cand = _runs([0.0] * 12, cand_passed=cand_passed)
    decision = _decide(base, cand, policy)
    assert any(v.rule == "failed_tests" for v in decision.fails)


def test_one_sut_error_counts_once_against_failed_case_budget() -> None:
    base_results = [make_result("t1", overall=0.8, passed=True, run_id="b", system_version="A")]
    candidate_results = [
        make_result(
            "t1",
            overall=0.0,
            passed=False,
            error="SUT failed",
            run_id="c",
            system_version="B",
        )
    ]
    base = RunData(make_summary("b", "A", base_results), base_results)
    candidate = RunData(make_summary("c", "B", candidate_results), candidate_results)
    policy = RegressionPolicy(
        min_paired_cases=10,
        counts=CountPolicy(max_failed_tests=1),
    )

    decision = _decide(base, candidate, policy)

    assert not any(v.rule == "failed_tests" for v in decision.fails)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metric_comparison_rejects_nonfinite_statistics(value: float) -> None:
    with pytest.raises(ValidationError):
        MetricComparison(
            metric="overall",
            baseline_mean=0.8,
            candidate_mean=0.7,
            delta=value,
        )


def test_metric_threshold_regression() -> None:
    policy = RegressionPolicy(metrics={"token_f1": MetricThreshold(absolute=0.05)})
    base, cand = _runs([0.0] * 12, det_delta=-0.1)
    decision = _decide(base, cand, policy)
    assert decision.status == "FAIL"
    assert any(
        v.rule == "metric_regression" and v.details.get("metric") == "token_f1"
        for v in decision.fails
    )


def test_metric_missing_warns() -> None:
    policy = RegressionPolicy(metrics={"does_not_exist": MetricThreshold(absolute=0.05)})
    base, cand = _runs([0.0] * 12)
    decision = _decide(base, cand, policy)
    assert any(v.rule == "metric_missing" for v in decision.warnings)


def test_judge_failure_rate_blocks() -> None:
    base, cand = _runs([0.0] * 12, judge_failed_cand=6)
    decision = _decide(base, cand)
    assert decision.status == "FAIL"
    assert any(v.rule == "judge_failure_rate" for v in decision.fails)


def test_small_sample_warns() -> None:
    base, cand = _runs([0.0] * 6)
    decision = _decide(base, cand)
    assert any(v.rule == "small_sample" for v in decision.warnings)


def test_percentile_interval_alone_does_not_establish_gate_significance() -> None:
    comparison = MetricComparison(
        metric="overall",
        n_paired=10,
        baseline_mean=0.8,
        candidate_mean=0.7,
        delta=-0.1,
        ci_low=-0.15,
        ci_high=-0.05,
        p_value=0.2,
    )

    assert not _significant(comparison, RegressionPolicy(min_paired_cases=10))


def test_one_pair_threshold_drop_warns_without_significance_failure() -> None:
    base, cand = _runs([-0.1], critical_idx=())
    comparison, decision = _compare_and_decide(base, cand)

    assert comparison.overall is not None
    assert comparison.overall.n_paired == 1
    assert decision.status == "WARNING"
    assert not any(v.rule == "overall_regression" for v in decision.fails)
    assert any(v.rule == "small_sample" for v in decision.warnings)
    warning = next(v for v in decision.warnings if v.rule == "overall_nonsignificant")
    assert "1 paired case" in warning.message
    assert "policy minimum: 10" in warning.message


def test_below_minimum_paired_sample_warns_without_significance_failure() -> None:
    policy = RegressionPolicy(min_paired_cases=10)
    base, cand = _runs([-0.1] * 9, critical_idx=())
    comparison, decision = _compare_and_decide(base, cand, policy)

    assert comparison.overall is not None
    assert comparison.overall.n_paired == 9
    assert decision.status == "WARNING"
    assert not any(v.rule == "overall_regression" for v in decision.fails)
    assert any(v.rule == "small_sample" for v in decision.warnings)


def test_sufficient_paired_sample_still_fails_supported_regression() -> None:
    policy = RegressionPolicy(min_paired_cases=10)
    deltas = [-0.06, -0.07, -0.08, -0.09, -0.10, -0.10, -0.11, -0.12, -0.13, -0.14]
    base, cand = _runs(deltas, critical_idx=())
    comparison, decision = _compare_and_decide(base, cand, policy)

    assert comparison.overall is not None
    assert comparison.overall.n_paired == 10
    assert comparison.overall.ci_high is not None
    assert comparison.overall.ci_high < 0
    assert comparison.overall.p_value is not None
    assert comparison.overall.p_value < policy.significance_alpha
    assert decision.status == "FAIL"
    assert any(v.rule == "overall_regression" for v in decision.fails)


def test_small_sample_does_not_suppress_newly_failing_critical_case() -> None:
    base, cand = _runs(
        [0.0],
        base_passed=[True],
        cand_passed=[False],
        critical_idx=(0,),
    )
    comparison, decision = _compare_and_decide(base, cand)

    assert comparison.n_common == 1
    assert decision.status == "FAIL"
    assert any(v.rule == "critical_failures" for v in decision.fails)
    assert any(v.rule == "small_sample" for v in decision.warnings)


def test_category_regression_fails() -> None:
    # make refusal-category cases (odd ids) collapse
    deltas = [-0.4 if i % 2 == 1 else 0.0 for i in range(12)]
    base_results = [
        make_result(
            f"t{i}",
            overall=0.8,
            passed=True,
            category="refusal" if i % 2 == 1 else "factual_correctness",
            critical=False,
            judge_dims=DIMS,
            run_id="b",
            system_version="A",
        )
        for i in range(12)
    ]
    cand_results = [
        make_result(
            f"t{i}",
            overall=0.8 + deltas[i],
            passed=True,
            category="refusal" if i % 2 == 1 else "factual_correctness",
            critical=False,
            judge_dims=DIMS,
            run_id="c",
            system_version="B",
        )
        for i in range(12)
    ]
    base = RunData(make_summary("b", "A", base_results), base_results)
    cand = RunData(make_summary("c", "B", cand_results), cand_results)
    decision = _decide(base, cand, RegressionPolicy(min_paired_cases=5))
    rules = {v.rule for v in decision.fails}
    assert "category_regression" in rules
    assert any(v.details.get("category") == "refusal" for v in decision.fails)


def test_decision_summary_payload() -> None:
    base, cand = _runs([-0.1] * 12)
    decision = _decide(base, cand)
    assert decision.summary["overall_delta"] == pytest.approx(-0.1)
    assert decision.summary["baseline_version"] == "A"
    assert decision.policy  # policy snapshot embedded
