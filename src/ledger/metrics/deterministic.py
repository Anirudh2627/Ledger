"""Deterministic (mechanical) metrics.

These metrics are cheap, reproducible and provider-free. They do NOT pretend
to measure semantic quality - that is the judge's job. Each metric returns a
score in ``[0, 1]`` or ``None`` when it does not apply to the case (metrics
that don't apply are excluded from aggregation, never counted as zero).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from ledger.core.interfaces import DeterministicMetric, MetricContext
from ledger.utils.logging import get_logger
from ledger.utils.text import (
    normalize_text,
    sequence_similarity,
    token_f1,
)

logger = get_logger(__name__)

#: Refusal indicators used by :class:`RefusalMatch`. Deliberately broad;
#: a system that must refuse should say so in one of these forms.
REFUSAL_PATTERNS = (
    r"don'?t have enough information",
    r"do not have enough information",
    r"not enough information",
    r"cannot (?:reliably )?answer",
    r"can'?t (?:reliably )?answer",
    r"unable to answer",
    r"(?:is |are )?not (?:covered|provided|mentioned|available|contained) (?:in|by) the",
    r"no information (?:in|about|available)",
    r"documents? (?:do not|don't|do not) (?:contain|cover|mention|include)",
    r"i (?:must )?(?:refuse|decline)",
    r"cannot (?:comply|act on|fulfill|fulfil)",
    r"can'?t (?:comply|act on|fulfill|fulfil)",
    r"won'?t (?:comply|act on)",
    r"i'?m sorry,? but",
    r"cannot assist with",
    r"not answerable from",
)
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

_CITATION_RE = re.compile(
    r"\(\s*sources?:|\[\s*source|sources?:\s*\S+\.md|\w[\w-]*\.md|\[[^\]]+\]\([^)]+\)",
    re.IGNORECASE,
)


def contains_refusal(text: str) -> bool:
    """Detect refusal phrasing in an answer."""
    return bool(_REFUSAL_RE.search(text or ""))


class ExactMatch(DeterministicMetric):
    """Normalized exact string match against the reference answer."""

    name = "exact_match"

    def compute(self, ctx: MetricContext) -> float | None:
        if not ctx.case.reference_answer:
            return None
        return (
            1.0
            if normalize_text(ctx.output.answer) == normalize_text(ctx.case.reference_answer)
            else 0.0
        )


class TokenF1(DeterministicMetric):
    """Token-level F1 (SQuAD-style) against the reference answer."""

    name = "token_f1"

    def compute(self, ctx: MetricContext) -> float | None:
        if not ctx.case.reference_answer:
            return None
        return token_f1(ctx.output.answer, ctx.case.reference_answer)


class NormalizedSimilarity(DeterministicMetric):
    """Character-level sequence similarity against the reference answer.

    Catches word-order and morphology differences that token F1 ignores. It is
    a crude proxy for semantic equivalence - interpret trends, not absolutes.
    """

    name = "normalized_similarity"

    def compute(self, ctx: MetricContext) -> float | None:
        if not ctx.case.reference_answer:
            return None
        return sequence_similarity(ctx.output.answer, ctx.case.reference_answer)


class KeyTermCoverage(DeterministicMetric):
    """Coverage of ``must_contain`` terms; hard-fails on ``must_not_contain``."""

    name = "key_term_coverage"

    def compute(self, ctx: MetricContext) -> float | None:
        answer_norm = normalize_text(ctx.output.answer)
        prohibited = [normalize_text(t) for t in ctx.case.must_not_contain]
        if prohibited and any(term and term in answer_norm for term in prohibited):
            return 0.0
        required = [normalize_text(t) for t in ctx.case.must_contain]
        if not required:
            return None
        hits = sum(1 for term in required if term and term in answer_norm)
        return hits / len(required)


class CitationPresence(DeterministicMetric):
    """Does the answer cite sources, for cases that require citations?"""

    name = "citation_presence"

    def compute(self, ctx: MetricContext) -> float | None:
        if not ctx.case.expect_citations:
            return None
        return 1.0 if _CITATION_RE.search(ctx.output.answer or "") else 0.0


class RetrievalHit(DeterministicMetric):
    """Recall of ``expected_sources`` among retrieved chunks.

    Measures the retrieval stage independently from generation - a regression
    here localizes the problem even when end-to-end scores barely move.
    """

    name = "retrieval_hit"

    def compute(self, ctx: MetricContext) -> float | None:
        if not ctx.case.expected_sources:
            return None
        if not ctx.output.retrieved_context:
            return 0.0
        retrieved_ids = {chunk.doc_id for chunk in ctx.output.retrieved_context}
        expected = set(ctx.case.expected_sources)
        hits = sum(1 for source in expected if _matches(source, retrieved_ids))
        return hits / len(expected)


def _matches(expected_source: str, retrieved_ids: set[str]) -> bool:
    """Match with or without file extension / path prefix."""
    stem = re.sub(r"\.(md|txt)$", "", expected_source.strip())
    for doc_id in retrieved_ids:
        if doc_id == expected_source or re.sub(r"\.(md|txt)$", "", doc_id) == stem:
            return True
    return False


class RefusalMatch(DeterministicMetric):
    """For refusal-expected cases: did the system actually refuse?"""

    name = "refusal_match"

    def compute(self, ctx: MetricContext) -> float | None:
        if not ctx.case.expects_refusal:
            return None
        return 1.0 if contains_refusal(ctx.output.answer) else 0.0


class StructuredOutputValidity(DeterministicMetric):
    """Validates machine-readable output contracts (``metadata.output_format``)."""

    name = "structured_output_validity"

    def compute(self, ctx: MetricContext) -> float | None:
        output_format = ctx.case.metadata.get("output_format")
        answer = (ctx.output.answer or "").strip()
        if output_format == "json":
            # Tolerate fenced JSON blocks.
            candidate = answer
            fence = re.search(r"```(?:json)?\s*(.*?)```", answer, re.DOTALL)
            if fence:
                candidate = fence.group(1).strip()
            try:
                json.loads(candidate)
                return 1.0
            except (json.JSONDecodeError, ValueError):
                return 0.0
        if output_format == "markdown_list":
            lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
            if not lines:
                return 0.0
            good = sum(1 for ln in lines if re.match(r"^[-*\u2022]\s+", ln))
            return good / len(lines)
        return None


class LengthCompliance(DeterministicMetric):
    """Checks ``metadata.max_words`` constraints (binary compliance)."""

    name = "length_compliance"

    def compute(self, ctx: MetricContext) -> float | None:
        max_words = ctx.case.metadata.get("max_words")
        if max_words is None:
            return None
        try:
            limit = int(max_words)
        except (TypeError, ValueError):
            return None
        words = len((ctx.output.answer or "").split())
        return 1.0 if words <= limit else 0.0


METRIC_REGISTRY: dict[str, type[DeterministicMetric]] = {
    cls.name: cls
    for cls in (
        ExactMatch,
        TokenF1,
        NormalizedSimilarity,
        KeyTermCoverage,
        CitationPresence,
        RetrievalHit,
        RefusalMatch,
        StructuredOutputValidity,
        LengthCompliance,
    )
}


def build_metrics(names: Iterable[str]) -> list[DeterministicMetric]:
    """Instantiate metrics by name; unknown names raise ``KeyError``."""
    metrics: list[DeterministicMetric] = []
    for name in names:
        try:
            metrics.append(METRIC_REGISTRY[name]())
        except KeyError:
            raise KeyError(
                f"unknown deterministic metric '{name}'; available: {sorted(METRIC_REGISTRY)}"
            ) from None
    return metrics


def compute_all(
    metrics: Iterable[DeterministicMetric], ctx: MetricContext
) -> dict[str, float | None]:
    """Run every metric, isolating failures (a broken metric must not kill a run)."""
    scores: dict[str, float | None] = {}
    for metric in metrics:
        try:
            value = metric.compute(ctx)
        except Exception as exc:  # defensive: metrics see arbitrary user data
            logger.warning("metric %s failed on case %s: %s", metric.name, ctx.case.id, exc)
            value = None
        if value is not None:
            value = max(0.0, min(1.0, float(value)))
        scores[metric.name] = value
    return scores
