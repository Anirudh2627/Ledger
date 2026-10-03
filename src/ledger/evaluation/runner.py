"""Execution of a system under test over a dataset.

The runner owns concurrency, retries and error capture. A failing case never
aborts the run: the exception is recorded on the ``SystemOutput`` so the
evaluator can score it as an error (which the regression policy counts).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ledger.core.interfaces import SystemUnderTest
from ledger.core.models import SystemOutput
from ledger.dataset.schema import TestCase
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CaseOutcome:
    case: TestCase
    output: SystemOutput


class EvaluationRunner:
    """Runs a SUT over test cases with bounded concurrency and retries."""

    def __init__(
        self,
        sut: SystemUnderTest,
        *,
        max_workers: int = 4,
        retries: int = 1,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.sut = sut
        self.max_workers = max(1, max_workers)
        self.retries = max(0, retries)
        self.progress_callback = progress_callback

    def run(self, cases: Sequence[TestCase]) -> list[CaseOutcome]:
        """Generate an output for every case, preserving dataset order."""
        outcomes: list[CaseOutcome | None] = [None] * len(cases)
        completed = 0
        started = time.perf_counter()

        def work(index: int, case: TestCase) -> tuple[int, CaseOutcome]:
            return index, self._run_case(case)

        if self.max_workers == 1 or len(cases) == 1:
            for index, case in enumerate(cases):
                i, outcome = work(index, case)
                outcomes[i] = outcome
                completed += 1
                self._report_progress(completed, len(cases))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(work, index, case) for index, case in enumerate(cases)]
                for future in futures:
                    i, outcome = future.result()
                    outcomes[i] = outcome
                    completed += 1
                    self._report_progress(completed, len(cases))

        elapsed = time.perf_counter() - started
        errors = sum(1 for o in outcomes if o is not None and o.output.error)
        logger.info(
            "SUT run complete: %d cases in %.1fs (%d errors) [%s]",
            len(cases),
            elapsed,
            errors,
            getattr(self.sut, "version", "unknown"),
        )
        return [o for o in outcomes if o is not None]

    def _run_case(self, case: TestCase) -> CaseOutcome:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                output = self.sut.generate(case)
                if attempt:
                    output.metadata["succeeded_after_retry"] = attempt
                return CaseOutcome(case=case, output=output)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "case %s failed on attempt %d/%d: %s",
                    case.id,
                    attempt + 1,
                    self.retries + 1,
                    exc,
                )
        return CaseOutcome(
            case=case,
            output=SystemOutput(
                answer="",
                error=f"{type(last_error).__name__}: {last_error}"
                if last_error
                else "unknown error",
                model_name=getattr(self.sut, "version", "unknown"),
            ),
        )

    def _report_progress(self, completed: int, total: int) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(completed, total)
        except Exception:  # pragma: no cover - progress must never break a run
            logger.debug("progress callback raised; ignored", exc_info=True)
