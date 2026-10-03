"""Aggregation of per-case results into run-level statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ledger.core.models import (
    CategoryAggregate,
    CountStats,
    EvaluationResult,
    LatencyStats,
)
from ledger.metrics.llm_judge import judge_dimension_metrics


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def metric_means(results: Sequence[EvaluationResult]) -> dict[str, float]:
    """Mean of every metric across applicable cases.

    Includes deterministic metrics, normalized judge dimensions
    (``judge_*``), the primary ``overall`` signal and ``pass_rate``.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for name, value in result.deterministic_scores.items():
            if value is not None:
                buckets[name].append(float(value))
        for name, value in _result_judge_metrics(result).items():
            buckets[name].append(value)
        if result.overall_score is not None:
            buckets["overall"].append(result.overall_score)

    means = {name: float(np.mean(values)) for name, values in buckets.items() if values}

    decidable = [r for r in results if r.passed is not None]
    if decidable:
        means["pass_rate"] = sum(1 for r in decidable if r.passed) / len(decidable)
    return means


def _result_judge_metrics(result: EvaluationResult) -> dict[str, float]:
    """Reconstruct normalized judge metrics from a persisted result."""
    if result.judge_failed or not result.judge_scores:
        return {}
    from ledger.core.models import JudgeScore

    judge = JudgeScore(
        dimensions=result.judge_scores,
        overall=result.judge_overall,
        normalized=(
            (result.judge_overall - 1.0) / 4.0 if result.judge_overall is not None else None
        ),
        failed=False,
    )
    return judge_dimension_metrics(judge)


def count_stats(results: Sequence[EvaluationResult]) -> CountStats:
    """Pass/fail/error bookkeeping used by the regression policy."""
    stats = CountStats(total=len(results))
    for result in results:
        if result.error:
            stats.errored += 1
        if result.judge_failed:
            stats.judge_failed += 1
        if result.passed is True:
            stats.passed += 1
        elif result.passed is False:
            stats.failed += 1
    stats.not_passed = stats.failed + stats.errored
    return stats


def latency_stats(results: Sequence[EvaluationResult]) -> LatencyStats:
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    if not latencies:
        return LatencyStats()
    array = np.asarray(latencies, dtype=float)
    return LatencyStats(
        mean_ms=round(float(np.mean(array)), 2),
        p50_ms=round(float(np.percentile(array, 50)), 2),
        p95_ms=round(float(np.percentile(array, 95)), 2),
    )


def category_aggregates(results: Sequence[EvaluationResult]) -> dict[str, CategoryAggregate]:
    """Per-category overall mean, pass rate and metric means."""
    by_category: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)

    aggregates: dict[str, CategoryAggregate] = {}
    for category, group in sorted(by_category.items()):
        overalls = [r.overall_score for r in group if r.overall_score is not None]
        decidable = [r for r in group if r.passed is not None]
        pass_rate = sum(1 for r in decidable if r.passed) / len(decidable) if decidable else None
        overall_mean = _mean_or_none(overalls)
        aggregates[category] = CategoryAggregate(
            n=len(group),
            overall_mean=round(overall_mean, 6) if overall_mean is not None else None,
            pass_rate=round(pass_rate, 6) if pass_rate is not None else None,
            metrics={k: round(v, 6) for k, v in metric_means(group).items()},
        )
    return aggregates
