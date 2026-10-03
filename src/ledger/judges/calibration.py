"""Judge calibration against human labels.

Why this exists: an LLM judge is itself a model with biases (verbosity
preference, self-preference, position bias, harsh/lenient drift). Before its
verdicts gate releases, we measure how well it reproduces human judgement on
a labeled sample, and keep the evidence next to the evaluation artifacts.

Produced statistics (per dimension and overall):
  * MAE / RMSE / mean bias (judge - human) on the 1-5 scale
  * Pearson r and Spearman rho (rank correlation)
  * agreement-within-1 rate
  * pass-decision agreement rate, Cohen's kappa and a confusion matrix

Interpretation guide (documented in docs/evaluation-methodology.md): kappa
follows the Landis & Koch bands; MAE <= 0.5 and within-1 >= 0.8 are common
working targets, but teams should set their own bars.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ledger.config.settings import JudgeConfig
from ledger.core.interfaces import Judge
from ledger.core.models import SystemOutput
from ledger.dataset.schema import TestCase
from ledger.judges.base import resolve_weights, weighted_overall
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


class HumanLabel(BaseModel):
    """One human-labeled (question, answer) pair used for calibration."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    answer: str
    context: str | None = None
    reference_answer: str | None = None
    expected_behavior: str | None = None
    category: str = "calibration"
    critical: bool = False
    human_scores: dict[str, int] = Field(min_length=1)
    human_pass: bool
    labeler: str | None = None
    notes: str | None = None

    def to_test_case(self) -> TestCase:
        return TestCase(
            id=self.id,
            question=self.question,
            context=self.context,
            reference_answer=self.reference_answer,
            expected_behavior=self.expected_behavior,
            category=self.category,
            critical=self.critical,
        )


class DimensionCalibration(BaseModel):
    n: int = 0
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None
    pearson_r: float | None = None
    spearman_rho: float | None = None
    agreement_within_1: float | None = None


class CalibrationResult(BaseModel):
    """Full calibration report for one judge configuration."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    judge_model: str | None = None
    n_labels: int = 0
    n_judged: int = 0
    n_judge_failures: int = 0
    dimensions: dict[str, DimensionCalibration] = Field(default_factory=dict)
    overall: DimensionCalibration = Field(default_factory=DimensionCalibration)
    pass_agreement_rate: float | None = None
    cohen_kappa: float | None = None
    kappa_interpretation: str | None = None
    confusion: dict[str, int] = Field(default_factory=dict)  # tp/tn/fp/fn (judge vs human)
    per_label: list[dict[str, Any]] = Field(default_factory=list)


def load_human_labels(path: str | Path) -> list[HumanLabel]:
    """Load calibration labels from JSONL (comments/blank lines allowed)."""
    labels: list[HumanLabel] = []
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"human labels file not found: {file_path}")
    with file_path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                labels.append(HumanLabel.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{file_path}:{line_no}: invalid label: {exc}") from exc
    if not labels:
        raise ValueError(f"no labels found in {file_path}")
    return labels


# ---------------------------------------------------------------------------
# Statistics helpers (dependency-light, numpy only)
# ---------------------------------------------------------------------------


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None  # constant input: correlation undefined
    return sxy / math.sqrt(sxx * syy)


def _rank(values: Sequence[float]) -> list[float]:
    """Average ranks with ties (for Spearman)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    return _pearson(_rank(x), _rank(y))


def _dimension_calibration(human: Sequence[float], judge: Sequence[float]) -> DimensionCalibration:
    n = len(human)
    if n == 0:
        return DimensionCalibration()
    diffs = [j - h for h, j in zip(human, judge, strict=True)]
    mae = sum(abs(d) for d in diffs) / n
    rmse = math.sqrt(sum(d * d for d in diffs) / n)
    bias = sum(diffs) / n
    within_1 = sum(1 for d in diffs if abs(d) <= 1.0) / n
    return DimensionCalibration(
        n=n,
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        bias=round(bias, 4),
        pearson_r=_round_or_none(_pearson(list(human), list(judge))),
        spearman_rho=_round_or_none(_spearman(list(human), list(judge))),
        agreement_within_1=round(within_1, 4),
    )


def _round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Kappa for judge_pass vs human_pass. ``None`` when undefined."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for j, h in pairs if j == h) / n
    pj_true = sum(1 for j, _ in pairs if j) / n
    ph_true = sum(1 for _, h in pairs if h) / n
    pe = pj_true * ph_true + (1 - pj_true) * (1 - ph_true)
    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1 - pe)


def interpret_kappa(kappa: float | None) -> str | None:
    """Landis & Koch (1977) agreement bands, with continuous boundaries."""
    if kappa is None:
        return None
    if kappa > 0.80:
        return "almost perfect"
    if kappa > 0.60:
        return "substantial"
    if kappa > 0.40:
        return "moderate"
    if kappa > 0.20:
        return "fair"
    if kappa >= 0.0:
        return "slight"
    return "poor (worse than chance)"


# ---------------------------------------------------------------------------
# Calibration pipeline
# ---------------------------------------------------------------------------


def _judge_model_name(judge: Judge) -> str | None:
    """Best-effort model name for reporting (mock and HTTP providers)."""
    provider = getattr(judge, "provider", None)
    model = getattr(provider, "model", None)
    if isinstance(model, str) and model:
        return model
    provider_config = getattr(provider, "_config", None)
    model = getattr(provider_config, "model", None)
    return model if isinstance(model, str) and model else None


def calibrate_judge(
    judge: Judge,
    labels: Sequence[HumanLabel],
    *,
    judge_config: JudgeConfig | None = None,
) -> CalibrationResult:
    """Run the judge over labeled answers and compare with human scores."""
    weights = resolve_weights(judge_config) if judge_config else None
    result = CalibrationResult(
        judge_model=_judge_model_name(judge),
        n_labels=len(labels),
    )

    human_by_dim: dict[str, list[float]] = {}
    judge_by_dim: dict[str, list[float]] = {}
    human_overalls: list[float] = []
    judge_overalls: list[float] = []
    pass_pairs: list[tuple[bool, bool]] = []

    for label in labels:
        case = label.to_test_case()
        output = SystemOutput(answer=label.answer)
        score = judge.score(case, output)
        if score.failed:
            result.n_judge_failures += 1
            logger.warning("judge failed on calibration label %s: %s", label.id, score.error)
            continue
        result.n_judged += 1

        for dim, human_value in label.human_scores.items():
            if dim not in score.dimensions:
                continue
            human_by_dim.setdefault(dim, []).append(float(human_value))
            judge_by_dim.setdefault(dim, []).append(float(score.dimensions[dim]))

        # Overall on the 1-5 scale (weighted when config provides weights).
        human_dims = {
            d: v for d, v in label.human_scores.items() if weights is None or d in weights
        }
        try:
            if weights and human_dims:
                human_overall = weighted_overall(human_dims, weights)
            elif human_dims:
                human_overall = sum(human_dims.values()) / len(human_dims)
            else:
                human_overall = None
        except ValueError:
            human_overall = None
        if human_overall is not None and score.overall is not None:
            human_overalls.append(human_overall)
            judge_overalls.append(score.overall)

        if score.passed is not None:
            pass_pairs.append((score.passed, label.human_pass))

        result.per_label.append(
            {
                "id": label.id,
                "category": label.category,
                "human_scores": label.human_scores,
                "judge_scores": score.dimensions,
                "human_pass": label.human_pass,
                "judge_pass": score.passed,
                "judge_overall": score.overall,
                "judge_reason": score.reason,
            }
        )

    for dim in sorted(human_by_dim):
        result.dimensions[dim] = _dimension_calibration(human_by_dim[dim], judge_by_dim[dim])
    result.overall = _dimension_calibration(human_overalls, judge_overalls)

    if pass_pairs:
        result.pass_agreement_rate = round(
            sum(1 for j, h in pass_pairs if j == h) / len(pass_pairs), 4
        )
        kappa = _cohen_kappa(pass_pairs)
        result.cohen_kappa = _round_or_none(kappa)
        result.kappa_interpretation = interpret_kappa(kappa)
        result.confusion = {
            "tp": sum(1 for j, h in pass_pairs if j and h),
            "tn": sum(1 for j, h in pass_pairs if not j and not h),
            "fp": sum(1 for j, h in pass_pairs if j and not h),
            "fn": sum(1 for j, h in pass_pairs if not j and h),
        }
    return result


def render_calibration_markdown(result: CalibrationResult) -> str:
    """Human-readable calibration report."""
    lines = [
        "# Judge Calibration Report",
        "",
        f"- Generated: {result.created_at.isoformat(timespec='seconds')}",
        f"- Judge model: `{result.judge_model or 'unknown'}`",
        f"- Labels: {result.n_labels} | judged: {result.n_judged} | "
        f"judge failures: {result.n_judge_failures}",
        "",
        "## Agreement with human labels (1-5 scale)",
        "",
        "| Dimension | n | MAE | RMSE | Bias (J-H) | Pearson r | Spearman rho | Within +-1 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def row(name: str, d: DimensionCalibration) -> str:
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        return (
            f"| {name} | {d.n} | {fmt(d.mae)} | {fmt(d.rmse)} | {fmt(d.bias)} "
            f"| {fmt(d.pearson_r)} | {fmt(d.spearman_rho)} | {fmt(d.agreement_within_1)} |"
        )

    for dim, dcal in result.dimensions.items():
        lines.append(row(dim, dcal))
    lines.append(row("**overall**", result.overall))

    lines += [
        "",
        "## Pass-decision agreement",
        "",
        f"- Agreement rate: {_fmt_opt(result.pass_agreement_rate)}",
        f"- Cohen's kappa: {_fmt_opt(result.cohen_kappa)}"
        + (f" ({result.kappa_interpretation})" if result.kappa_interpretation else ""),
    ]
    if result.confusion:
        c = result.confusion
        lines += [
            "",
            "|  | Human: PASS | Human: FAIL |",
            "|---|---|---|",
            f"| Judge: PASS | {c.get('tp', 0)} | {c.get('fp', 0)} |",
            f"| Judge: FAIL | {c.get('fn', 0)} | {c.get('tn', 0)} |",
        ]
    lines += [
        "",
        "## Notes",
        "",
        "- Bias > 0 means the judge scores *higher* than humans (lenient); < 0 means harsher.",
        "- Correlations are undefined (shown as `-`) when a column is constant.",
        "- Calibration results describe the judge on THIS label sample; re-run after any "
        "change to judge model, prompt or rubric.",
        "",
    ]
    return "\n".join(lines)


def _fmt_opt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
