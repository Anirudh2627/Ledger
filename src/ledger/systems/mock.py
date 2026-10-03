"""Trivial deterministic systems used to test the harness itself.

These are *not* realistic applications. They exist so that the evaluation
pipeline, statistics and CI gate can be exercised with a fully controlled
quality signal:

* ``reference`` mode returns the gold answer (plumbing upper bound),
* ``constant`` mode returns one fixed string,
* ``degraded`` mode deterministically corrupts the gold answer.
"""

from __future__ import annotations

import hashlib

from ledger.config.settings import MockSystemConfig
from ledger.core.interfaces import SystemUnderTest
from ledger.core.models import SystemOutput
from ledger.dataset.schema import TestCase
from ledger.systems.base import register_system


@register_system("mock")
class MockSystem(SystemUnderTest):
    """A configurable do-nothing SUT for harness tests."""

    def __init__(self, config: MockSystemConfig, *, version: str = "mock") -> None:
        self.config = config
        self.version = version

    def generate(self, test_case: TestCase) -> SystemOutput:
        mode = self.config.mode
        if mode == "reference":
            answer = test_case.reference_answer or (
                "I'm sorry, but the provided documents do not contain "
                "information to answer this question."
            )
        elif mode == "constant":
            answer = self.config.constant_answer
        else:  # degraded: deterministic corruption of the reference answer
            answer = self._degraded(test_case)
        return SystemOutput(
            answer=answer,
            model_name="mock-system",
            prompt_version="none",
            latency_ms=1.0,
            metadata={"mode": mode},
        )

    def _degraded(self, test_case: TestCase) -> str:
        reference = test_case.reference_answer or "answer unavailable"
        words = reference.split()
        # Drop every second word deterministically and add noise.
        kept = [w for i, w in enumerate(words) if i % 2 == 0]
        digest = hashlib.sha256(test_case.id.encode()).hexdigest()[:6]
        return f"maybe {' '.join(kept)} (see {digest})"

    def describe(self) -> dict[str, object]:
        return {"kind": "mock", "version": self.version, "mode": self.config.mode}
