"""Bridge between the judge subsystem and per-result score records.

Keeps the mapping from a structured :class:`JudgeScore` to normalized metric
values in one place, so aggregation, comparison and reports all agree on
what "judge_correctness = 0.75" means.
"""

from __future__ import annotations

from ledger.core.models import JudgeScore

#: Judge dimension scores live on a 1-5 integer scale; everything downstream
#: (aggregation, bootstrap, policy thresholds) uses the [0, 1] normalization.
JUDGE_SCALE_MIN = 1.0
JUDGE_SCALE_MAX = 5.0


def normalize_judge_score(raw: float) -> float:
    """Map a 1-5 score onto [0, 1]."""
    clamped = max(JUDGE_SCALE_MIN, min(JUDGE_SCALE_MAX, float(raw)))
    return (clamped - JUDGE_SCALE_MIN) / (JUDGE_SCALE_MAX - JUDGE_SCALE_MIN)


def judge_dimension_metrics(judge: JudgeScore | None) -> dict[str, float]:
    """Return normalized per-dimension metrics prefixed ``judge_<dim>``.

    Returns ``{}`` for missing or failed judgments so callers can merge
    unconditionally.
    """
    if judge is None or judge.failed:
        return {}
    metrics: dict[str, float] = {}
    for dim, raw in judge.dimensions.items():
        metrics[f"judge_{dim}"] = normalize_judge_score(raw)
    if judge.normalized is not None:
        metrics["judge_overall"] = judge.normalized
    return metrics
