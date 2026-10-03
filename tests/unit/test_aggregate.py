"""Unit tests: aggregation of per-case results."""

from __future__ import annotations

import pytest

from ledger.metrics.aggregate import (
    category_aggregates,
    count_stats,
    latency_stats,
    metric_means,
)
from tests.helpers import make_result

pytestmark = pytest.mark.unit

DIMS = {"correctness": 4, "relevance": 5, "groundedness": 4, "instruction_following": 3}


def _sample_results():
    return [
        make_result(
            "t1",
            overall=0.8,
            passed=True,
            det={"token_f1": 0.5, "exact_match": None},
            judge_dims=DIMS,
            latency_ms=10.0,
        ),
        make_result(
            "t2",
            overall=0.6,
            passed=False,
            category="refusal",
            det={"token_f1": 0.25},
            judge_dims={
                "correctness": 2,
                "relevance": 3,
                "groundedness": 5,
                "instruction_following": 4,
            },
            latency_ms=20.0,
        ),
        make_result("t3", overall=None, passed=None, error="boom", latency_ms=30.0),
    ]


def test_metric_means_includes_deterministic_and_judge() -> None:
    means = metric_means(_sample_results())
    assert means["token_f1"] == pytest.approx((0.5 + 0.25) / 2)
    assert "exact_match" not in means  # all-None metrics are excluded
    assert means["judge_correctness"] == pytest.approx(((4 - 1) / 4 + (2 - 1) / 4) / 2)


def test_metric_means_judge_overall_normalization() -> None:
    means = metric_means(_sample_results())
    # t1 judge overall = 4.0 -> normalized 0.75 ; t2 = 3.5 -> 0.625
    assert means["judge_overall"] == pytest.approx((0.75 + 0.625) / 2)


def test_metric_means_overall_and_pass_rate() -> None:
    means = metric_means(_sample_results())
    assert means["overall"] == pytest.approx((0.8 + 0.6) / 2)  # None excluded
    # 2 decidable cases (t3 has passed=None): 1 pass / 2
    assert means["pass_rate"] == pytest.approx(0.5)


def test_judge_failed_results_excluded_from_judge_metrics() -> None:
    results = [make_result("t1", judge_dims=DIMS, judge_failed=True)]
    means = metric_means(results)
    assert "judge_correctness" not in means


def test_count_stats() -> None:
    counts = count_stats(_sample_results())
    assert counts.total == 3
    assert counts.passed == 1
    assert counts.failed == 1
    assert counts.errored == 1
    assert counts.not_passed == 2
    assert counts.judge_failed == 0


def test_count_stats_judge_failures() -> None:
    counts = count_stats([make_result("t1", judge_failed=True)])
    assert counts.judge_failed == 1


def test_latency_stats() -> None:
    stats = latency_stats(_sample_results())
    assert stats.mean_ms == pytest.approx(20.0)
    assert stats.p50_ms == pytest.approx(20.0)
    assert stats.p95_ms == pytest.approx(29.0)


def test_latency_stats_empty() -> None:
    stats = latency_stats([make_result("t1", latency_ms=None)])
    assert stats.mean_ms is None


def test_category_aggregates() -> None:
    aggs = category_aggregates(_sample_results())
    assert set(aggs) == {"factual_correctness", "refusal"}
    fc = aggs["factual_correctness"]
    assert fc.n == 2
    assert fc.overall_mean == pytest.approx(0.8)  # t3 overall is None -> excluded
    # t3 has passed=None (undecided) -> only t1 counts for pass rate
    assert fc.pass_rate == pytest.approx(1.0)
    refusal = aggs["refusal"]
    assert refusal.n == 1
    assert refusal.pass_rate == pytest.approx(0.0)
