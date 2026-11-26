"""Core domain models shared across the evaluation pipeline.

Everything that crosses a module boundary (provider I/O, system outputs,
per-case results, run summaries) is a pydantic model so that artifacts are
validatable, serializable and self-documenting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ledger.dataset.schema import TestCase


def _validate_latency(value: Any, field_name: str) -> Any:
    if value is None:
        return value
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return value


def _validate_normalized_metrics(metrics: Any, field_name: str) -> Any:
    if not isinstance(metrics, dict):
        return metrics
    for name, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{field_name} '{name}' must be finite and in [0, 1]")
    return metrics


def _validate_normalized_score(value: Any, field_name: str) -> Any:
    if value is None:
        return value
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return value


# ---------------------------------------------------------------------------
# Provider I/O
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """One chat message (OpenAI-compatible role/content pair)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    """A chat-completion request handed to an :class:`LLMProvider`.

    ``purpose`` lets deterministic/test providers route behavior (e.g. the
    mock provider answers judge prompts differently from generation prompts)
    without leaking harness internals into prompt text.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    purpose: Literal["generation", "judge", "other"] = "generation"
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: Literal["json", "text"] | None = None
    seed: int | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class CompletionResponse(BaseModel):
    text: str
    model: str | None = None
    finish_reason: str | None = None
    latency_ms: float = 0.0
    usage: CompletionUsage | None = None
    #: Number of transport-level attempts that were needed.
    attempts: int = 1

    @field_validator("latency_ms", mode="before")
    @classmethod
    def _validate_latency_ms(cls, value: Any) -> Any:
        return _validate_latency(value, "latency_ms")


# ---------------------------------------------------------------------------
# System-under-test output
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A unit of retrieved context returned by the system under test."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    chunk_index: int = 0
    text: str
    score: float | None = None


class SystemOutput(BaseModel):
    """Everything the evaluation engine needs from one SUT invocation."""

    answer: str = ""
    retrieved_context: list[RetrievedChunk] = Field(default_factory=list)
    model_name: str | None = None
    prompt_version: str | None = None
    latency_ms: float | None = None
    #: Populated when generation failed after retries; such results are
    #: counted as errors instead of silently scoring as empty answers.
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("latency_ms", mode="before")
    @classmethod
    def _validate_latency_ms(cls, value: Any) -> Any:
        return _validate_latency(value, "latency_ms")


# ---------------------------------------------------------------------------
# Judge output
# ---------------------------------------------------------------------------


class JudgeScore(BaseModel):
    """Structured LLM-as-a-judge verdict for one case."""

    #: Dimension name -> integer score in [1, 5].
    dimensions: dict[str, int] = Field(default_factory=dict)
    #: Weighted overall on the 1-5 scale, recomputed by Ledger (never trusted
    #: blindly from the judge model).
    overall: float | None = None
    #: ``overall`` normalized to [0, 1]: ``(overall - 1) / 4``.
    normalized: float | None = None
    passed: bool | None = None
    reason: str = ""
    judge_model: str | None = None
    latency_ms: float | None = None
    attempts: int = 1
    #: Set when judging ultimately failed (malformed output, API error, ...).
    failed: bool = False
    error: str | None = None
    #: Raw judge text, truncated; useful for debugging calibration issues.
    raw: str | None = None

    @field_validator("dimensions", mode="before")
    @classmethod
    def _validate_dimensions(cls, dimensions: Any) -> Any:
        if not isinstance(dimensions, dict):
            return dimensions
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"judge dimension '{name}' must be an integer in [1, 5]")
        return dimensions

    @field_validator("overall", mode="before")
    @classmethod
    def _validate_overall(cls, value: Any) -> Any:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 1.0 <= value <= 5.0
        ):
            raise ValueError("judge overall must be finite and in [1, 5]")
        return value

    @field_validator("normalized", mode="before")
    @classmethod
    def _validate_normalized(cls, value: Any) -> Any:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("normalized judge score must be finite and in [0, 1]")
        return value

    @field_validator("latency_ms", mode="before")
    @classmethod
    def _validate_latency_ms(cls, value: Any) -> Any:
        return _validate_latency(value, "latency_ms")


# ---------------------------------------------------------------------------
# Per-case evaluation result
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Fully self-contained record of evaluating one case with one system.

    Field names intentionally mirror the flattened artifact schema so that
    ``results.jsonl`` rows are directly consumable by notebooks/BI tools.
    """

    test_id: str
    run_id: str
    system_version: str
    category: str
    critical: bool = False
    timestamp: datetime

    # Inputs
    question: str
    context: str | None = None
    reference_answer: str | None = None
    expected_behavior: str | None = None

    # System output
    actual_answer: str = ""
    retrieved_context: list[RetrievedChunk] = Field(default_factory=list)
    model_name: str | None = None
    prompt_version: str | None = None
    latency_ms: float | None = None
    error: str | None = None

    # Scores
    deterministic_scores: dict[str, float | None] = Field(default_factory=dict)
    judge_scores: dict[str, int] | None = None
    judge_overall: float | None = None
    judge_reason: str | None = None
    judge_failed: bool = False
    overall_score: float | None = None
    passed: bool | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("latency_ms", mode="before")
    @classmethod
    def _validate_latency_ms(cls, value: Any) -> Any:
        return _validate_latency(value, "latency_ms")

    @field_validator("deterministic_scores", mode="before")
    @classmethod
    def _validate_deterministic_scores(cls, scores: Any) -> Any:
        if not isinstance(scores, dict):
            return scores
        for name, value in scores.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"deterministic score '{name}' must be numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"deterministic score '{name}' must be finite and in [0, 1]")
        return scores

    @field_validator("judge_scores", mode="before")
    @classmethod
    def _validate_judge_score_values(cls, scores: Any) -> Any:
        if scores is None:
            return scores
        if not isinstance(scores, dict):
            return scores
        for name, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"judge score '{name}' must be an integer in [1, 5]")
        return scores

    @field_validator("judge_overall", mode="before")
    @classmethod
    def _validate_judge_overall(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("judge_overall must be numeric")
        if not math.isfinite(value) or not 1.0 <= value <= 5.0:
            raise ValueError("judge_overall must be finite and in [1, 5]")
        return value

    @field_validator("overall_score", mode="before")
    @classmethod
    def _validate_overall_score(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("overall_score must be numeric")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("overall_score must be finite and in [0, 1]")
        return value

    @classmethod
    def create(
        cls,
        *,
        case: TestCase,
        output: SystemOutput,
        run_id: str,
        system_version: str,
        deterministic_scores: dict[str, float | None],
        judge: JudgeScore | None = None,
        overall_score: float | None = None,
        passed: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        return cls(
            test_id=case.id,
            run_id=run_id,
            system_version=system_version,
            category=case.category,
            critical=case.critical,
            timestamp=datetime.now(UTC),
            question=case.question,
            context=case.context,
            reference_answer=case.reference_answer,
            expected_behavior=case.expected_behavior,
            actual_answer=output.answer,
            retrieved_context=output.retrieved_context,
            model_name=output.model_name,
            prompt_version=output.prompt_version,
            latency_ms=output.latency_ms,
            error=output.error,
            deterministic_scores=deterministic_scores,
            judge_scores=judge.dimensions if judge and not judge.failed else None,
            judge_overall=judge.overall if judge and not judge.failed else None,
            judge_reason=judge.reason if judge and not judge.failed else None,
            judge_failed=bool(judge and judge.failed),
            overall_score=overall_score,
            passed=passed,
            metadata=metadata or {},
        )


def validate_result_integrity(results: Sequence[EvaluationResult]) -> None:
    """Revalidate result values and reject duplicate identities."""
    for result in results:
        EvaluationResult.model_validate(result.model_dump())
    seen: set[str] = set()
    for result in results:
        if result.test_id in seen:
            raise ValueError(f"duplicate result test id '{result.test_id}'")
        seen.add(result.test_id)


# ---------------------------------------------------------------------------
# Run summary / aggregates
# ---------------------------------------------------------------------------


class DatasetInfo(BaseModel):
    path: str
    fingerprint: str
    n_cases: int = Field(ge=0, strict=True)


class ProviderInfo(BaseModel):
    name: str
    model: str
    temperature: float | None = None
    is_mock: bool = False


class CountStats(BaseModel):
    total: int = Field(default=0, ge=0, strict=True)
    passed: int = Field(default=0, ge=0, strict=True)
    failed: int = Field(default=0, ge=0, strict=True)
    errored: int = Field(default=0, ge=0, strict=True)
    judge_failed: int = Field(default=0, ge=0, strict=True)
    not_passed: int = Field(default=0, ge=0, strict=True)


class LatencyStats(BaseModel):
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None

    @field_validator("mean_ms", "p50_ms", "p95_ms", mode="before")
    @classmethod
    def _validate_latency_stat(cls, value: Any) -> Any:
        return _validate_latency(value, "latency statistic")


class CategoryAggregate(BaseModel):
    n: int = Field(ge=0, strict=True)
    overall_mean: float | None = None
    pass_rate: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("overall_mean", "pass_rate", mode="before")
    @classmethod
    def _validate_normalized_value(cls, value: Any) -> Any:
        return _validate_normalized_score(value, "aggregate score")

    @field_validator("metrics", mode="before")
    @classmethod
    def _validate_metrics(cls, metrics: Any) -> Any:
        return _validate_normalized_metrics(metrics, "aggregate metric")


class RunSummary(BaseModel):
    """Aggregate outcome of one evaluation run - the unit of comparison."""

    run_id: str
    system_version: str
    created_at: datetime
    dataset: DatasetInfo
    provider: ProviderInfo
    judge: ProviderInfo | None = None
    judge_enabled: bool = False
    prompt_version: str | None = None
    #: Mean of every tracked metric across applicable cases, including
    #: ``overall`` (primary quality signal) and ``pass_rate``.
    metrics: dict[str, float] = Field(default_factory=dict)
    categories: dict[str, CategoryAggregate] = Field(default_factory=dict)
    counts: CountStats = Field(default_factory=CountStats)
    latency: LatencyStats = Field(default_factory=LatencyStats)
    runtime: dict[str, Any] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metrics", mode="before")
    @classmethod
    def _validate_metrics(cls, metrics: Any) -> Any:
        return _validate_normalized_metrics(metrics, "run metric")

    @model_validator(mode="after")
    def _validate_aggregate_counts(self) -> RunSummary:
        if self.counts.total != self.dataset.n_cases:
            raise ValueError("run summary total does not match dataset case count")
        minimum_not_passed = max(self.counts.failed, self.counts.errored)
        maximum_not_passed = min(self.counts.total, self.counts.failed + self.counts.errored)
        legacy_not_passed = self.counts.failed + self.counts.errored
        if not (
            minimum_not_passed <= self.counts.not_passed <= maximum_not_passed
            or self.counts.not_passed == legacy_not_passed
        ):
            raise ValueError("run summary not_passed count is inconsistent")
        if self.counts.passed + self.counts.failed > self.counts.total:
            raise ValueError("run summary pass/fail counts exceed total")
        if self.counts.errored > self.counts.total or self.counts.judge_failed > self.counts.total:
            raise ValueError("run summary error counts exceed total")
        if sum(category.n for category in self.categories.values()) != self.dataset.n_cases:
            raise ValueError("run summary category counts do not match dataset case count")
        decidable = self.counts.passed + self.counts.failed
        if decidable:
            expected_pass_rate = round(self.counts.passed / decidable, 6)
            if self.metrics.get("pass_rate") != expected_pass_rate:
                raise ValueError("run summary pass_rate does not match pass/fail counts")
        elif "pass_rate" in self.metrics:
            raise ValueError("run summary pass_rate exists without a decidable case")
        return self
