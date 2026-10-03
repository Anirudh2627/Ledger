"""The evaluation report model and its builder.

A :class:`Report` is the single source of truth rendered by every output
format (JSON, Markdown, console). Reports embed the full comparison, the
gate decision, run references and readable excerpts of the worst
regressions - enough context for a reviewer to triage without opening the
raw artifacts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ledger import __version__
from ledger.core.models import RunSummary
from ledger.regression.comparator import ComparisonResult, RunData
from ledger.regression.detector import GateDecision
from ledger.utils.text import truncate

REPORT_SCHEMA_VERSION = "1.0"
_EXCERPT_CHARS = 240
_TOP_REGRESSIONS = 10


class RunReference(BaseModel):
    """Identifying metadata of one evaluated run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    system_version: str
    created_at: datetime | None = None
    provider: str | None = None
    model: str | None = None
    judge_model: str | None = None
    prompt_version: str | None = None
    dataset_fingerprint: str | None = None
    n_cases: int = 0
    overall_mean: float | None = None
    pass_rate: float | None = None
    failed: int = 0
    errored: int = 0
    judge_failed: int = 0


class CaseExcerpt(BaseModel):
    """Readable slice of one regressed case for human triage."""

    model_config = ConfigDict(extra="forbid")

    test_id: str
    category: str
    critical: bool = False
    status: str
    delta: float | None = None
    baseline_overall: float | None = None
    candidate_overall: float | None = None
    baseline_passed: bool | None = None
    candidate_passed: bool | None = None
    question: str = ""
    baseline_answer: str = ""
    candidate_answer: str = ""
    candidate_judge_reason: str | None = None


class Report(BaseModel):
    """Complete comparison report (rendered to JSON / Markdown / console)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = REPORT_SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ledger_version: str = __version__
    baseline: RunReference
    candidate: RunReference
    comparison: ComparisonResult
    decision: GateDecision
    top_regressions: list[CaseExcerpt] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)


def _run_reference(summary: RunSummary) -> RunReference:
    return RunReference(
        run_id=summary.run_id,
        system_version=summary.system_version,
        created_at=summary.created_at,
        provider=summary.provider.name,
        model=summary.provider.model,
        judge_model=summary.judge.model if summary.judge else None,
        prompt_version=summary.prompt_version,
        dataset_fingerprint=summary.dataset.fingerprint,
        n_cases=summary.dataset.n_cases,
        overall_mean=summary.metrics.get("overall"),
        pass_rate=summary.metrics.get("pass_rate"),
        failed=summary.counts.failed,
        errored=summary.counts.errored,
        judge_failed=summary.counts.judge_failed,
    )


def build_report(
    comparison: ComparisonResult,
    decision: GateDecision,
    baseline: RunData,
    candidate: RunData,
    *,
    notes: list[str] | None = None,
    artifacts: dict[str, str] | None = None,
) -> Report:
    """Assemble the report model from comparison + decision + raw run data."""
    base_by_id = baseline.by_id()
    cand_by_id = candidate.by_id()

    regressed = [c for c in comparison.cases if c.status == "regressed"]
    regressed.sort(key=lambda c: c.delta if c.delta is not None else 0.0)
    excerpts: list[CaseExcerpt] = []
    for case_delta in regressed[:_TOP_REGRESSIONS]:
        base_result = base_by_id.get(case_delta.test_id)
        cand_result = cand_by_id.get(case_delta.test_id)
        excerpts.append(
            CaseExcerpt(
                test_id=case_delta.test_id,
                category=case_delta.category,
                critical=case_delta.critical,
                status=case_delta.status,
                delta=case_delta.delta,
                baseline_overall=case_delta.baseline_overall,
                candidate_overall=case_delta.candidate_overall,
                baseline_passed=case_delta.baseline_passed,
                candidate_passed=case_delta.candidate_passed,
                question=truncate(base_result.question if base_result else "", _EXCERPT_CHARS),
                baseline_answer=truncate(
                    base_result.actual_answer if base_result else "", _EXCERPT_CHARS
                ),
                candidate_answer=truncate(
                    cand_result.actual_answer if cand_result else "", _EXCERPT_CHARS
                ),
                candidate_judge_reason=truncate(
                    cand_result.judge_reason if cand_result and cand_result.judge_reason else "",
                    _EXCERPT_CHARS,
                )
                or None,
            )
        )

    default_notes = [
        "Scores are on a normalized 0-1 scale (judge dimension scores mapped from 1-5).",
        "Confidence intervals come from a paired percentile bootstrap over test cases.",
    ]
    if comparison.dataset_fingerprint_match is False:
        default_notes.append(
            "WARNING: the two runs used different dataset versions; treat deltas with caution."
        )
    if decision.status != "PASS":
        default_notes.append(
            f"Gate decision: {decision.status} - see violations for the exact rules triggered."
        )

    return Report(
        baseline=_run_reference(baseline.summary),
        candidate=_run_reference(candidate.summary),
        comparison=comparison,
        decision=decision,
        top_regressions=excerpts,
        notes=list(notes) if notes is not None else default_notes,
        artifacts=dict(artifacts or {}),
    )


def report_headline(report: Report) -> dict[str, Any]:
    """Tiny dict for logs/step-summaries: the numbers that matter most."""
    overall = report.comparison.overall
    return {
        "decision": report.decision.status,
        "baseline": report.baseline.system_version,
        "candidate": report.candidate.system_version,
        "overall_delta": overall.delta if overall else None,
        "newly_failing": len(report.comparison.newly_failing),
        "critical_regressions": len(report.comparison.critical_regressions),
    }
