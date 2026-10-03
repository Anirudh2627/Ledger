"""Paired comparison of two evaluation runs.

Aligns per-case results by ``test_id``, then computes for every tracked
metric (overall quality, each judge dimension, each deterministic metric,
pass rate) and every category slice:

    baseline mean, candidate mean, absolute delta, relative delta,
    paired-bootstrap CI + p-value.

Plus per-case movement: regressed / improved / stable, newly failing and
newly passing tests, and critical-case regressions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ledger.core.models import EvaluationResult, RunSummary
from ledger.regression.bootstrap import (
    BootstrapResult,
    InsufficientDataError,
    means,
    paired_bootstrap,
    sign_test_on_flips,
)
from ledger.utils.logging import get_logger

logger = get_logger(__name__)

CaseStatus = Literal[
    "regressed",
    "improved",
    "stable_pass",
    "stable_fail",
    "undecided",
    "baseline_only",
    "candidate_only",
]

#: Score movements smaller than this count as "no change".
MOVEMENT_EPS = 1e-6


class MetricComparison(BaseModel):
    """Statistical comparison of one metric across two runs."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    n_paired: int = 0
    baseline_mean: float
    candidate_mean: float
    delta: float
    relative_delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    significant_regression: bool = False
    significant_improvement: bool = False


class CaseDelta(BaseModel):
    """Per-case movement between two runs."""

    model_config = ConfigDict(extra="forbid")

    test_id: str
    category: str
    critical: bool = False
    baseline_overall: float | None = None
    candidate_overall: float | None = None
    delta: float | None = None
    baseline_passed: bool | None = None
    candidate_passed: bool | None = None
    status: CaseStatus = "undecided"


class ComparisonResult(BaseModel):
    """Everything the regression engine and reports need from a comparison."""

    model_config = ConfigDict(extra="forbid")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    baseline_run_id: str
    baseline_version: str
    candidate_run_id: str
    candidate_version: str
    dataset_fingerprint_match: bool = True
    n_baseline: int = 0
    n_candidate: int = 0
    n_common: int = 0
    overall: MetricComparison | None = None
    #: Per-metric comparisons keyed by metric name (includes judge_* dims).
    metrics: dict[str, MetricComparison] = Field(default_factory=dict)
    #: Per-category comparisons of the overall score, plus the special
    #: ``__critical__`` slice over all critical cases.
    categories: dict[str, MetricComparison] = Field(default_factory=dict)
    cases: list[CaseDelta] = Field(default_factory=list)
    newly_failing: list[str] = Field(default_factory=list)
    newly_passing: list[str] = Field(default_factory=list)
    critical_regressions: list[str] = Field(default_factory=list)
    sign_test_p_value: float | None = None
    sign_test_flips: dict[str, int] = Field(default_factory=dict)
    method: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class RunData:
    """A loaded run: its summary plus per-case results."""

    summary: RunSummary
    results: list[EvaluationResult]

    def by_id(self) -> dict[str, EvaluationResult]:
        return {r.test_id: r for r in self.results}


def _per_case_values(results: Sequence[EvaluationResult], metric: str) -> dict[str, float]:
    """Extract per-case values for a metric name (deterministic/judge/overall)."""
    values: dict[str, float] = {}
    for result in results:
        value: float | None = None
        if metric == "overall":
            value = result.overall_score
        elif metric == "pass_rate":
            value = 1.0 if result.passed else (0.0 if result.passed is not None else None)
        elif metric.startswith("judge_"):
            dim = metric[len("judge_") :]
            if result.judge_scores and dim in result.judge_scores:
                value = (result.judge_scores[dim] - 1) / 4.0
            elif dim == "overall" and result.judge_overall is not None:
                value = (result.judge_overall - 1) / 4.0
        else:
            value = result.deterministic_scores.get(metric)
        if value is not None:
            values[result.test_id] = float(value)
    return values


def _compare_metric(
    name: str,
    base_values: dict[str, float],
    cand_values: dict[str, float],
    *,
    n_samples: int,
    confidence: float,
    seed: int,
) -> MetricComparison | None:
    common_ids = sorted(set(base_values) & set(cand_values))
    if not common_ids:
        return None
    base_seq = [base_values[i] for i in common_ids]
    cand_seq = [cand_values[i] for i in common_ids]
    b_mean, c_mean, delta, rel = means(base_seq, cand_seq)
    boot: BootstrapResult | None = None
    try:
        boot = paired_bootstrap(
            base_seq, cand_seq, n_samples=n_samples, confidence=confidence, seed=seed
        )
    except InsufficientDataError as exc:  # pragma: no cover - guarded above
        logger.warning("bootstrap skipped for %s: %s", name, exc)
    return MetricComparison(
        metric=name,
        n_paired=len(common_ids),
        baseline_mean=round(b_mean, 6),
        candidate_mean=round(c_mean, 6),
        delta=round(delta, 6),
        relative_delta=round(rel, 6) if rel is not None else None,
        ci_low=round(boot.ci_low, 6) if boot and boot.ci_low is not None else None,
        ci_high=round(boot.ci_high, 6) if boot and boot.ci_high is not None else None,
        p_value=round(boot.p_value, 6) if boot and boot.p_value is not None else None,
        significant_regression=bool(boot and boot.significant_regression),
        significant_improvement=bool(boot and boot.significant_improvement),
    )


def _case_status(delta: CaseDelta) -> CaseStatus:
    if delta.baseline_passed is True and delta.candidate_passed is False:
        return "regressed"
    if delta.baseline_passed is False and delta.candidate_passed is True:
        return "improved"
    if delta.delta is not None:
        if delta.delta < -MOVEMENT_EPS:
            return "regressed"
        if delta.delta > MOVEMENT_EPS:
            return "improved"
    if delta.candidate_passed is True:
        return "stable_pass"
    if delta.candidate_passed is False:
        return "stable_fail"
    return "undecided"


def compare_runs(
    baseline: RunData,
    candidate: RunData,
    *,
    n_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 1337,
    extra_metrics: Sequence[str] = (),
) -> ComparisonResult:
    """Build a full paired comparison between two runs."""
    base_by_id = baseline.by_id()
    cand_by_id = candidate.by_id()
    common_ids = sorted(set(base_by_id) & set(cand_by_id))

    result = ComparisonResult(
        baseline_run_id=baseline.summary.run_id,
        baseline_version=baseline.summary.system_version,
        candidate_run_id=candidate.summary.run_id,
        candidate_version=candidate.summary.system_version,
        dataset_fingerprint_match=(
            baseline.summary.dataset.fingerprint == candidate.summary.dataset.fingerprint
        ),
        n_baseline=len(baseline.results),
        n_candidate=len(candidate.results),
        n_common=len(common_ids),
        method={
            "bootstrap": "paired_percentile",
            "n_samples": n_samples,
            "confidence": confidence,
            "seed": seed,
        },
    )

    if not result.dataset_fingerprint_match:
        result.warnings.append(
            "dataset fingerprints differ between runs; comparison covers the "
            f"intersection of test ids only ({len(common_ids)} cases)"
        )
    only_baseline = sorted(set(base_by_id) - set(cand_by_id))
    only_candidate = sorted(set(cand_by_id) - set(base_by_id))
    if only_baseline:
        result.warnings.append(f"{len(only_baseline)} cases present only in baseline run")
    if only_candidate:
        result.warnings.append(f"{len(only_candidate)} cases present only in candidate run")

    # -- metric set ------------------------------------------------------------
    metric_names = set(baseline.summary.metrics) | set(candidate.summary.metrics)
    metric_names |= set(extra_metrics)
    metric_names.update({"overall", "pass_rate"})
    metric_names = {m for m in metric_names if not m.startswith("__")}

    for metric in sorted(metric_names):
        comparison = _compare_metric(
            metric,
            _per_case_values(baseline.results, metric),
            _per_case_values(candidate.results, metric),
            n_samples=n_samples,
            confidence=confidence,
            seed=seed,
        )
        if comparison is None:
            continue
        if metric == "overall":
            result.overall = comparison
        result.metrics[metric] = comparison

    # -- per-case deltas ---------------------------------------------------------
    for test_id in common_ids:
        b = base_by_id[test_id]
        c = cand_by_id[test_id]
        delta_overall = (
            round(c.overall_score - b.overall_score, 6)
            if b.overall_score is not None and c.overall_score is not None
            else None
        )
        case_delta = CaseDelta(
            test_id=test_id,
            category=c.category,
            critical=c.critical or b.critical,
            baseline_overall=b.overall_score,
            candidate_overall=c.overall_score,
            delta=delta_overall,
            baseline_passed=b.passed,
            candidate_passed=c.passed,
        )
        case_delta.status = _case_status(case_delta)
        result.cases.append(case_delta)
    for test_id in only_baseline:
        b = base_by_id[test_id]
        result.cases.append(
            CaseDelta(
                test_id=test_id,
                category=b.category,
                critical=b.critical,
                baseline_overall=b.overall_score,
                baseline_passed=b.passed,
                status="baseline_only",
            )
        )
    for test_id in only_candidate:
        c = cand_by_id[test_id]
        result.cases.append(
            CaseDelta(
                test_id=test_id,
                category=c.category,
                critical=c.critical,
                candidate_overall=c.overall_score,
                candidate_passed=c.passed,
                status="candidate_only",
            )
        )

    result.newly_failing = [
        cd.test_id
        for cd in result.cases
        if cd.baseline_passed is True and cd.candidate_passed is False
    ]
    result.newly_passing = [
        cd.test_id
        for cd in result.cases
        if cd.baseline_passed is False and cd.candidate_passed is True
    ]
    result.critical_regressions = [
        cd.test_id for cd in result.cases if cd.critical and cd.status == "regressed"
    ]

    # -- category + critical slices -------------------------------------------
    by_category: dict[str, list[str]] = defaultdict(list)
    critical_ids: list[str] = []
    for cd in result.cases:
        if cd.test_id in base_by_id and cd.test_id in cand_by_id:
            by_category[cd.category].append(cd.test_id)
            if cd.critical:
                critical_ids.append(cd.test_id)

    overall_base = _per_case_values(baseline.results, "overall")
    overall_cand = _per_case_values(candidate.results, "overall")
    for category, ids in sorted(by_category.items()):
        comp = _compare_metric(
            category,
            {i: overall_base[i] for i in ids if i in overall_base},
            {i: overall_cand[i] for i in ids if i in overall_cand},
            n_samples=n_samples,
            confidence=confidence,
            seed=seed,
        )
        if comp is not None:
            result.categories[category] = comp
    critical_comp = _compare_metric(
        "__critical__",
        {i: overall_base[i] for i in critical_ids if i in overall_base},
        {i: overall_cand[i] for i in critical_ids if i in overall_cand},
        n_samples=n_samples,
        confidence=confidence,
        seed=seed,
    )
    if critical_comp is not None:
        result.categories["__critical__"] = critical_comp

    # -- pass/fail flip test -------------------------------------------------------
    flips = sign_test_on_flips(
        [base_by_id[i].passed for i in common_ids],
        [cand_by_id[i].passed for i in common_ids],
    )
    result.sign_test_p_value = flips.p_value
    result.sign_test_flips = {
        "n_flips": flips.n_flips,
        "newly_failing": flips.n_newly_failing,
        "newly_passing": flips.n_newly_passing,
    }

    return result
