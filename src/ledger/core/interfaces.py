"""Core interfaces (ports) of the evaluation framework.

Ledger is built around small, explicit interfaces so that every moving part -
the application under test, the LLM provider, the judge, the metrics - can be
replaced without touching the pipeline:

* :class:`SystemUnderTest` - *what* is evaluated (RAG app, agent, raw API...)
* :class:`LLMProvider`     - *how* text is generated (mock, OpenAI-compatible)
* :class:`Judge`           - *who* scores open-ended quality
* :class:`DeterministicMetric` - *what* is checked mechanically
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ledger.core.models import (
    CompletionRequest,
    CompletionResponse,
    JudgeScore,
    SystemOutput,
)
from ledger.dataset.schema import TestCase


class SystemUnderTest(ABC):
    """The application under evaluation.

    Implementations wrap *any* LLM application - a RAG pipeline, an agent, a
    thin prompt, or a remote service. The evaluation engine only ever calls
    :meth:`generate` and :meth:`describe`; it never inspects internals.
    """

    #: Human-readable version identifier recorded with every result.
    version: str = "unknown"

    @abstractmethod
    def generate(self, test_case: TestCase) -> SystemOutput:
        """Produce an answer (and optional retrieved context) for one case.

        Implementations should raise on unrecoverable errors; the runner
        retries and ultimately records the failure on the result instead of
        crashing the whole evaluation.
        """

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable description for run metadata."""
        return {"kind": type(self).__name__, "version": self.version}


class LLMProvider(ABC):
    """Minimal chat-completion port shared by SUTs, judges and embedders."""

    name: str = "abstract"

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run one chat completion, applying provider-level retries."""


class Judge(ABC):
    """Scores one (case, output) pair across rubric dimensions."""

    name: str = "abstract"

    @abstractmethod
    def score(self, case: TestCase, output: SystemOutput) -> JudgeScore:
        """Return a structured verdict. Must not raise for malformed model
        output - failures are represented via ``JudgeScore.failed``."""

    def describe(self) -> dict[str, Any]:  # pragma: no cover - trivial
        return {"name": self.name}


@dataclass
class MetricContext:
    """Everything a deterministic metric may look at for one case."""

    case: TestCase
    output: SystemOutput
    #: Run-level extras (e.g. configured top_k) injected by the evaluator.
    extras: dict[str, Any] = field(default_factory=dict)


class DeterministicMetric(ABC):
    """A mechanical, reproducible check scoring in ``[0, 1]``.

    Return ``None`` when the metric does not apply to a case (e.g. retrieval
    hit for a system that exposes no retrieved context); non-applicable
    metrics are excluded from aggregation instead of counting as zero.
    """

    name: str = "abstract"

    @abstractmethod
    def compute(self, ctx: MetricContext) -> float | None:
        """Compute the metric score for one case."""
