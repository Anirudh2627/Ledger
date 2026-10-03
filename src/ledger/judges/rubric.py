"""Rubric-based LLM-as-a-judge with hardened response handling.

Failure modes explicitly handled:
  * malformed JSON (fences, prose around payload, trailing commas) -> robust
    extraction; on failure, a repair retry with the parse error;
  * missing / non-integer / out-of-range dimension scores -> validation
    error -> repair retry;
  * provider exceptions & timeouts -> captured after provider-level retries;
  * final failure -> ``JudgeScore(failed=True, error=...)`` so one broken
    verdict never crashes the suite (downstream policy decides the
    consequence via ``judge.on_failure``).

Trust boundaries: the judge's own ``overall``/``pass`` fields are parsed for
diagnostics but the authoritative values are recomputed from dimension
scores, configured weights and configured thresholds.
"""

from __future__ import annotations

import time
from typing import Any

from ledger.config.settings import JudgeConfig
from ledger.core.interfaces import Judge, LLMProvider
from ledger.core.models import CompletionRequest, JudgeScore, SystemOutput
from ledger.dataset.schema import TestCase
from ledger.judges.base import finalize_score, pass_threshold_for
from ledger.judges.prompts import build_judge_messages, build_repair_messages
from ledger.utils.json_extract import JSONExtractionError, extract_json_object
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


class JudgeParseError(ValueError):
    """Raised internally when a judge response fails validation."""


def validate_dimensions(payload: dict[str, Any], dimensions: list[str]) -> dict[str, int]:
    """Extract and validate integer dimension scores from a parsed payload.

    Accepts ints or integral floats; rejects missing keys, non-numerics and
    values outside [1, 5] (no silent clamping - a confused judge should
    trigger a repair retry, not a fabricated score).
    """
    result: dict[str, int] = {}
    missing = [d for d in dimensions if d not in payload]
    if missing:
        raise JudgeParseError(f"missing dimension keys: {missing}")
    for dim in dimensions:
        value = payload[dim]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JudgeParseError(f"dimension '{dim}' is not numeric: {value!r}")
        numeric = float(value)
        if not 1.0 <= numeric <= 5.0:
            raise JudgeParseError(f"dimension '{dim}' out of range [1,5]: {value!r}")
        result[dim] = round(numeric)
    return result


class RubricJudge(Judge):
    """Structured multi-dimension judge backed by any LLM provider."""

    name = "rubric"

    def __init__(self, config: JudgeConfig, provider: LLMProvider) -> None:
        self.config = config
        self.provider = provider

    # -- Judge API -------------------------------------------------------------

    def score(self, case: TestCase, output: SystemOutput) -> JudgeScore:
        started = time.perf_counter()
        messages = build_judge_messages(case, output, self.config)
        last_error = "unknown error"
        raw_text = ""

        for attempt in range(1, self.config.max_retries + 2):  # first try + retries
            try:
                response = self.provider.complete(
                    CompletionRequest(
                        messages=messages,
                        purpose="judge",
                        temperature=0.0,
                        response_format="json",
                    )
                )
            except Exception as exc:  # provider exhausted its own retries
                last_error = f"provider error: {exc}"
                logger.warning(
                    "judge provider failed for case %s (attempt %d): %s", case.id, attempt, exc
                )
                continue

            raw_text = response.text
            try:
                payload = extract_json_object(raw_text)
                dimensions = validate_dimensions(payload, self.config.dimensions)
            except (JSONExtractionError, JudgeParseError) as exc:
                last_error = str(exc)
                logger.warning(
                    "malformed judge response for case %s (attempt %d): %s",
                    case.id,
                    attempt,
                    last_error,
                )
                messages = build_repair_messages(messages, raw_text, last_error)
                continue

            reason = str(payload.get("reason", ""))[:2000]
            judge_overall_hint = payload.get("overall")
            score = finalize_score(
                dimensions,
                critical=case.critical,
                config=self.config,
                reason=reason,
                judge_model=response.model or getattr(self.provider, "model", None),
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                attempts=attempt,
                raw=raw_text[:2000],
            )
            self._log_divergence(case, score, judge_overall_hint, payload.get("pass"))
            return score

        return JudgeScore(
            failed=True,
            error=f"judge failed after {self.config.max_retries + 1} attempt(s): {last_error}",
            judge_model=getattr(self.provider, "model", None),
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            attempts=self.config.max_retries + 1,
            raw=raw_text[:2000] or None,
        )

    # -- internals ---------------------------------------------------------------

    def _log_divergence(
        self,
        case: TestCase,
        score: JudgeScore,
        judge_overall: Any,
        judge_pass: Any,
    ) -> None:
        """Warn when our recomputation disagrees with the judge's own math."""
        try:
            if (
                isinstance(judge_overall, (int, float))
                and score.overall is not None
                and abs(float(judge_overall) - score.overall) > 0.5
            ):
                logger.info(
                    "case %s: judge-reported overall %.2f differs from recomputed %.2f "
                    "(recomputed value is authoritative)",
                    case.id,
                    float(judge_overall),
                    score.overall,
                )
            if (
                isinstance(judge_pass, bool)
                and score.passed is not None
                and judge_pass != score.passed
            ):
                logger.info(
                    "case %s: judge-reported pass=%s overridden by threshold rule "
                    "(overall=%.2f, threshold=%.2f)",
                    case.id,
                    judge_pass,
                    score.overall,
                    pass_threshold_for(case.critical, self.config),
                )
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": getattr(self.provider, "name", "unknown"),
            "model": getattr(self.provider, "model", None),
            "dimensions": self.config.dimensions,
            "pass_threshold": self.config.pass_threshold,
            "critical_pass_threshold": self.config.critical_pass_threshold,
        }
