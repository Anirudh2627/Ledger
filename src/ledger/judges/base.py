"""Shared judge primitives: score math and pass/fail rules.

The judge's raw output is *never* trusted for arithmetic: overall scores are
recomputed from dimension scores and configured weights, and pass decisions
are re-derived from configured thresholds. This keeps gating consistent even
when the judge model miscalculates its own average.
"""

from __future__ import annotations

from ledger.config.settings import JudgeConfig
from ledger.core.models import JudgeScore
from ledger.metrics.llm_judge import normalize_judge_score

DEFAULT_WEIGHT = 1.0


def resolve_weights(config: JudgeConfig) -> dict[str, float]:
    """Per-dimension weights; missing dimensions default to 1.0."""
    return {dim: float(config.weights.get(dim, DEFAULT_WEIGHT)) for dim in config.dimensions}


def weighted_overall(dimensions: dict[str, int], weights: dict[str, float]) -> float:
    """Weighted mean of dimension scores on the 1-5 scale."""
    total_weight = 0.0
    total = 0.0
    for dim, weight in weights.items():
        if dim in dimensions:
            total += dimensions[dim] * weight
            total_weight += weight
    if total_weight == 0:
        raise ValueError("no dimension scores available to compute overall")
    return total / total_weight


def pass_threshold_for(critical: bool, config: JudgeConfig) -> float:
    """Threshold on the 1-5 scale; critical cases can be held to a higher bar."""
    if critical and config.critical_pass_threshold is not None:
        return max(config.pass_threshold, config.critical_pass_threshold)
    return config.pass_threshold


def finalize_score(
    dimensions: dict[str, int],
    *,
    critical: bool,
    config: JudgeConfig,
    reason: str = "",
    judge_model: str | None = None,
    latency_ms: float | None = None,
    attempts: int = 1,
    raw: str | None = None,
) -> JudgeScore:
    """Recompute overall/normalized/pass from dimensions and policy config."""
    weights = resolve_weights(config)
    overall = weighted_overall(dimensions, weights)
    threshold = pass_threshold_for(critical, config)
    return JudgeScore(
        dimensions=dimensions,
        overall=round(overall, 4),
        normalized=round(normalize_judge_score(overall), 6),
        passed=overall >= threshold,
        reason=reason,
        judge_model=judge_model,
        latency_ms=latency_ms,
        attempts=attempts,
        raw=raw,
    )
