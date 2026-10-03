"""Per-case scoring: deterministic metrics + judge + pass/fail derivation."""

from __future__ import annotations

from ledger.config.settings import EvaluationConfig, JudgeConfig
from ledger.core.interfaces import DeterministicMetric, Judge, MetricContext
from ledger.core.models import EvaluationResult, JudgeScore, SystemOutput
from ledger.dataset.schema import TestCase
from ledger.metrics.deterministic import compute_all
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


class JudgeAbortError(RuntimeError):
    """Raised when judge.on_failure == 'abort' and judging failed."""


class Evaluator:
    """Turns (case, output) pairs into fully scored evaluation results."""

    def __init__(
        self,
        *,
        metrics: list[DeterministicMetric],
        judge: Judge | None = None,
        judge_config: JudgeConfig | None = None,
        evaluation_config: EvaluationConfig | None = None,
    ) -> None:
        self.metrics = metrics
        self.judge = judge
        self.judge_config = judge_config
        self.evaluation_config = evaluation_config or EvaluationConfig()

    # -- public API ---------------------------------------------------------

    def evaluate_case(
        self,
        case: TestCase,
        output: SystemOutput,
        *,
        run_id: str,
        system_version: str,
    ) -> EvaluationResult:
        """Score one case end-to-end (deterministic metrics, then judge)."""
        deterministic = compute_all(self.metrics, MetricContext(case=case, output=output))

        judge_score: JudgeScore | None = None
        if self.judge is not None and output.error is None:
            judge_score = self.judge.score(case, output)
            if judge_score.failed:
                judge_score = self._handle_judge_failure(case, judge_score)

        overall, passed = self._derive_scores(case, output, deterministic, judge_score)
        return EvaluationResult.create(
            case=case,
            output=output,
            run_id=run_id,
            system_version=system_version,
            deterministic_scores=deterministic,
            judge=judge_score,
            overall_score=overall,
            passed=passed,
        )

    # -- internals ------------------------------------------------------------

    def _handle_judge_failure(self, case: TestCase, score: JudgeScore) -> JudgeScore | None:
        on_failure = self.judge_config.on_failure if self.judge_config else "exclude"
        logger.error(
            "judge failed for case %s: %s (on_failure=%s)", case.id, score.error, on_failure
        )
        if on_failure == "abort":
            raise JudgeAbortError(f"judge failed on case {case.id}: {score.error}")
        # "exclude" and "fail_case" both keep the failure on record.
        return score

    def _derive_scores(
        self,
        case: TestCase,
        output: SystemOutput,
        deterministic: dict[str, float | None],
        judge_score: JudgeScore | None,
    ) -> tuple[float | None, bool | None]:
        """Compute the primary quality score and pass decision.

        Precedence:
          1. SUT error            -> score 0.0, fail.
          2. Healthy judge        -> judge normalized overall decides.
          3. Judge failed
             (on_failure=fail_case) -> deterministic fallback score, fail.
          4. Judge disabled/failed
             (on_failure=exclude)   -> deterministic composite vs fallback
                                       threshold.
        """
        det_values = [v for v in deterministic.values() if v is not None]
        det_composite = sum(det_values) / len(det_values) if det_values else None

        if output.error is not None:
            return 0.0, False

        if (
            judge_score is not None
            and not judge_score.failed
            and judge_score.normalized is not None
        ):
            return round(judge_score.normalized, 6), bool(judge_score.passed)

        judge_failed_hard = (
            judge_score is not None
            and judge_score.failed
            and self.judge_config is not None
            and self.judge_config.on_failure == "fail_case"
        )
        if judge_failed_hard:
            return det_composite, False

        if det_composite is None:
            # No signal at all (no judge, no applicable metrics).
            return None, None
        return round(
            det_composite, 6
        ), det_composite >= self.evaluation_config.fallback_pass_threshold
