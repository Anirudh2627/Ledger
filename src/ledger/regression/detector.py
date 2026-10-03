"""Regression detection: apply the policy to a comparison, emit a decision.

The detector implements the gate logic as a set of independent, named rules.
Each rule produces either a FAIL-level violation or a WARNING, with a
machine-readable ``rule`` id and human-readable message, so CI output and
reports can explain *why* a change was blocked.

FAIL conditions (any of):
  R1  statistically-supported overall degradation beyond ``overall`` threshold
  R2  any category degrades beyond ``categories`` threshold (with significance
      when required)
  R3  critical-case slice degrades beyond ``critical`` threshold, or newly
      failing critical tests exceed ``max_critical_failures``
  R4  a configured per-metric threshold is exceeded (with significance when
      required)
  R5  count limits exceeded (newly failing tests, total failures)
  R6  evaluation integrity: judge failure rate above ``max_judge_failure_rate``

WARNING conditions: degradations inside the warning band
(``threshold * warning_fraction`` .. threshold), degradations beyond a
threshold but *not* statistically supported while ``require_significance``
is on, dataset fingerprint mismatch, tiny paired sample.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ledger.core.models import RunSummary
from ledger.regression.comparator import ComparisonResult, MetricComparison
from ledger.regression.policy import RegressionPolicy, ThresholdGroup
from ledger.utils.logging import get_logger

logger = get_logger(__name__)

GateStatus = Literal["PASS", "WARNING", "FAIL"]


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["fail", "warning"]
    rule: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class GateDecision(BaseModel):
    """The verdict of the regression gate."""

    model_config = ConfigDict(extra="forbid")

    status: GateStatus = "PASS"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    violations: list[Violation] = Field(default_factory=list)
    policy: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)

    @property
    def fails(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "fail"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]


def _relative_drop(comparison: MetricComparison) -> float | None:
    if comparison.baseline_mean == 0:
        return None
    return -comparison.delta / abs(comparison.baseline_mean)


def _exceeds(
    comparison: MetricComparison,
    group: ThresholdGroup | None,
    absolute: float | None,
    relative: float | None,
) -> bool:
    """True when the (negative) delta exceeds the configured drop thresholds."""
    drop = -comparison.delta  # positive = degradation
    abs_hit = False
    if absolute is not None and drop > absolute:
        abs_hit = True
    if group is not None and drop > group.absolute:
        abs_hit = True
    rel_hit = False
    rel_drop = _relative_drop(comparison)
    if relative is not None and rel_drop is not None and rel_drop > relative:
        rel_hit = True
    if (
        group is not None
        and group.relative is not None
        and rel_drop is not None
        and rel_drop > group.relative
    ):
        rel_hit = True
    return abs_hit or rel_hit


def _significant(comparison: MetricComparison, policy: RegressionPolicy) -> bool:
    """Statistical support for a directional change."""
    ci_conclusive = (
        comparison.ci_high is not None
        and comparison.ci_low is not None
        and (comparison.ci_high < 0 or comparison.ci_low > 0)
    )
    if ci_conclusive:
        return True
    return comparison.p_value is not None and comparison.p_value < policy.significance_alpha


def _fmt_delta(delta: float) -> str:
    return f"{delta:+.4f}"


def _fmt_ci(comparison: MetricComparison) -> str:
    if comparison.ci_low is None or comparison.ci_high is None:
        return "n/a"
    return f"[{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]"


def detect_regressions(
    comparison: ComparisonResult,
    baseline_summary: RunSummary,
    candidate_summary: RunSummary,
    policy: RegressionPolicy,
) -> GateDecision:
    """Apply every policy rule and produce the gate decision."""
    violations: list[Violation] = []

    def add_fail(rule: str, message: str, **details: Any) -> None:
        violations.append(Violation(severity="fail", rule=rule, message=message, details=details))

    def add_warning(rule: str, message: str, **details: Any) -> None:
        violations.append(
            Violation(severity="warning", rule=rule, message=message, details=details)
        )

    # -- R1: overall quality ------------------------------------------------
    overall = comparison.overall
    if overall is None:
        add_fail(
            "overall_missing",
            "no paired 'overall' scores available; cannot evaluate the primary quality signal",
        )
    else:
        _apply_threshold_rule(
            overall,
            policy.overall,
            policy,
            label="overall quality",
            rule_prefix="overall",
            add_fail=add_fail,
            add_warning=add_warning,
        )

    # -- R2: per-category ----------------------------------------------------
    for name, comp in sorted(comparison.categories.items()):
        if name == "__critical__":
            continue  # handled by R3
        _apply_threshold_rule(
            comp,
            policy.categories,
            policy,
            label=f"category '{name}'",
            rule_prefix="category",
            add_fail=add_fail,
            add_warning=add_warning,
            details_extra={"category": name},
        )

    # -- R3: critical slice ----------------------------------------------------
    critical_comp = comparison.categories.get("__critical__")
    if critical_comp is not None:
        _apply_threshold_rule(
            critical_comp,
            policy.critical,
            policy,
            label="critical cases",
            rule_prefix="critical",
            add_fail=add_fail,
            add_warning=add_warning,
        )
    if policy.counts.max_critical_failures is not None:
        critical_newly_failing = [
            cd.test_id
            for cd in comparison.cases
            if cd.critical and cd.baseline_passed is True and cd.candidate_passed is False
        ]
        if len(critical_newly_failing) > policy.counts.max_critical_failures:
            add_fail(
                "critical_failures",
                f"{len(critical_newly_failing)} critical test(s) newly failing "
                f"(max allowed: {policy.counts.max_critical_failures}): "
                f"{', '.join(critical_newly_failing[:8])}",
                tests=critical_newly_failing,
            )

    # -- R4: configured per-metric thresholds ------------------------------------
    for metric_name, threshold in sorted(policy.metrics.items()):
        metric_comp = comparison.metrics.get(metric_name)
        if metric_comp is None:
            add_warning(
                "metric_missing",
                f"policy references metric '{metric_name}' which was not measured in these runs",
                metric=metric_name,
            )
            continue
        if metric_comp.delta >= 0:
            continue
        exceeded = _exceeds(metric_comp, None, threshold.absolute, threshold.relative)
        if not exceeded:
            continue
        if policy.require_significance and not _significant(metric_comp, policy):
            add_warning(
                "metric_nonsignificant",
                f"metric '{metric_name}' degraded {_fmt_delta(metric_comp.delta)} beyond "
                f"threshold but is not statistically supported (CI {_fmt_ci(metric_comp)})",
                metric=metric_name,
                delta=metric_comp.delta,
            )
        else:
            add_fail(
                "metric_regression",
                f"metric '{metric_name}' regressed {_fmt_delta(metric_comp.delta)} "
                f"(CI {_fmt_ci(metric_comp)}, p={metric_comp.p_value})",
                metric=metric_name,
                delta=metric_comp.delta,
                p_value=metric_comp.p_value,
            )

    # -- R5: count limits ----------------------------------------------------------
    counts = policy.counts
    n_newly_failing = len(comparison.newly_failing)
    if (
        counts.max_newly_failing_tests is not None
        and n_newly_failing > counts.max_newly_failing_tests
    ):
        add_fail(
            "newly_failing_tests",
            f"{n_newly_failing} test(s) newly failing (max allowed: "
            f"{counts.max_newly_failing_tests}): {', '.join(comparison.newly_failing[:8])}",
            tests=comparison.newly_failing,
        )
    candidate_failures = candidate_summary.counts.not_passed
    if counts.max_failed_tests is not None and candidate_failures > counts.max_failed_tests:
        add_fail(
            "failed_tests",
            f"candidate has {candidate_failures} failing test(s) in total "
            f"(max allowed: {counts.max_failed_tests})",
            failed=candidate_failures,
        )
    if (
        comparison.sign_test_p_value is not None
        and n_newly_failing > 0
        and comparison.sign_test_p_value < policy.significance_alpha
        and n_newly_failing > (counts.max_newly_failing_tests or n_newly_failing)
    ):
        logger.debug(
            "sign test corroborates flips: p=%.4f (%d newly failing vs %d newly passing)",
            comparison.sign_test_p_value,
            n_newly_failing,
            comparison.sign_test_flips.get("newly_passing", 0),
        )

    # -- R6: evaluation integrity --------------------------------------------------
    if counts.max_judge_failure_rate is not None and candidate_summary.counts.total:
        rate = candidate_summary.counts.judge_failed / candidate_summary.counts.total
        if rate > counts.max_judge_failure_rate:
            add_fail(
                "judge_failure_rate",
                f"judge failed on {candidate_summary.counts.judge_failed}/"
                f"{candidate_summary.counts.total} candidate cases "
                f"({rate:.1%} > allowed {counts.max_judge_failure_rate:.1%}); "
                "the verdict is not trustworthy",
                rate=rate,
            )

    # -- structural warnings -------------------------------------------------------
    if not comparison.dataset_fingerprint_match:
        add_warning(
            "dataset_mismatch",
            "baseline and candidate were evaluated on different dataset versions",
        )
    if comparison.n_common < policy.min_paired_cases:
        add_warning(
            "small_sample",
            f"only {comparison.n_common} paired cases (policy minimum: "
            f"{policy.min_paired_cases}); statistical power is low",
            n_common=comparison.n_common,
        )
    for message in comparison.warnings:
        add_warning("comparison", message)

    status: GateStatus = "PASS"
    if any(v.severity == "fail" for v in violations):
        status = "FAIL"
    elif violations:
        status = "WARNING"

    decision = GateDecision(
        status=status,
        violations=violations,
        policy=policy.model_dump(mode="json"),
        summary={
            "baseline_run": comparison.baseline_run_id,
            "baseline_version": comparison.baseline_version,
            "candidate_run": comparison.candidate_run_id,
            "candidate_version": comparison.candidate_version,
            "n_common": comparison.n_common,
            "overall_baseline": overall.baseline_mean if overall else None,
            "overall_candidate": overall.candidate_mean if overall else None,
            "overall_delta": overall.delta if overall else None,
            "overall_ci": (
                [overall.ci_low, overall.ci_high]
                if overall and overall.ci_low is not None
                else None
            ),
            "newly_failing": n_newly_failing,
            "newly_passing": len(comparison.newly_passing),
            "critical_regressions": len(comparison.critical_regressions),
            "baseline_pass_rate": baseline_summary.metrics.get("pass_rate"),
            "candidate_pass_rate": candidate_summary.metrics.get("pass_rate"),
        },
    )
    logger.info(
        "gate decision: %s (%d fail, %d warning)",
        status,
        len(decision.fails),
        len(decision.warnings),
    )
    return decision


def _apply_threshold_rule(
    comp: MetricComparison,
    group: ThresholdGroup,
    policy: RegressionPolicy,
    *,
    label: str,
    rule_prefix: str,
    add_fail: Any,
    add_warning: Any,
    details_extra: dict[str, Any] | None = None,
) -> None:
    """Shared FAIL/WARNING logic for a threshold group."""
    if comp.delta >= 0:
        return
    drop = -comp.delta
    extra = details_extra or {}
    if _exceeds(comp, group, None, None):
        if policy.require_significance and not _significant(comp, policy):
            add_warning(
                f"{rule_prefix}_nonsignificant",
                f"{label} degraded {_fmt_delta(comp.delta)} beyond the configured threshold "
                f"but lacks statistical support (CI {_fmt_ci(comp)}, p={comp.p_value})",
                delta=comp.delta,
                **extra,
            )
        else:
            add_fail(
                f"{rule_prefix}_regression",
                f"{label} regressed {_fmt_delta(comp.delta)} "
                f"({comp.baseline_mean:.4f} -> {comp.candidate_mean:.4f}, "
                f"CI {_fmt_ci(comp)}, p={comp.p_value})",
                delta=comp.delta,
                baseline_mean=comp.baseline_mean,
                candidate_mean=comp.candidate_mean,
                p_value=comp.p_value,
                **extra,
            )
    elif drop > group.absolute * group.warning_fraction:
        add_warning(
            f"{rule_prefix}_warning_band",
            f"{label} degraded {_fmt_delta(comp.delta)}, approaching the fail threshold "
            f"({group.absolute:.3f})",
            delta=comp.delta,
            **extra,
        )
