"""Core domain models shared across the evaluation pipeline.

Everything that crosses a module boundary (provider I/O, system outputs,
per-case results, run summaries) is a pydantic model so that artifacts are
validatable, serializable and self-documenting.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ledger.dataset.schema import TestCase

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


# ---------------------------------------------------------------------------
# Run summary / aggregates
# ---------------------------------------------------------------------------


class DatasetInfo(BaseModel):
    path: str
    fingerprint: str
    n_cases: int


class ProviderInfo(BaseModel):
    name: str
    model: str
    temperature: float | None = None
    is_mock: bool = False


class CountStats(BaseModel):
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    judge_failed: int = 0
    not_passed: int = 0  # failed + errored (what the gate counts)


class LatencyStats(BaseModel):
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None


class CategoryAggregate(BaseModel):
    n: int
    overall_mean: float | None = None
    pass_rate: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


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
