"""Unit tests: deterministic metrics."""

from __future__ import annotations

import pytest

from ledger.core.interfaces import DeterministicMetric, MetricContext
from ledger.metrics.deterministic import (
    METRIC_REGISTRY,
    CitationPresence,
    ExactMatch,
    KeyTermCoverage,
    LengthCompliance,
    NormalizedSimilarity,
    RefusalMatch,
    RetrievalHit,
    StructuredOutputValidity,
    TokenF1,
    build_metrics,
    compute_all,
    contains_refusal,
)
from tests.helpers import make_case, make_chunk, make_output

pytestmark = pytest.mark.unit


def ctx(case, answer="The answer is 42.", chunks=None) -> MetricContext:
    return MetricContext(case=case, output=make_output(answer, retrieved_context=chunks or []))


# -- exact match -------------------------------------------------------------


def test_exact_match_ignores_case_and_punctuation() -> None:
    case = make_case(reference="AES-256!")
    assert ExactMatch().compute(ctx(case, "aes 256")) == 1.0
    assert ExactMatch().compute(ctx(case, "AES-256.")) == 1.0
    assert ExactMatch().compute(ctx(case, "TLS 1.3")) == 0.0


def test_exact_match_not_applicable_without_reference() -> None:
    assert ExactMatch().compute(ctx(make_case(reference=None))) is None


# -- token f1 / similarity ----------------------------------------------------


def test_token_f1_bounds() -> None:
    case = make_case(reference="the cat sat on the mat")
    assert TokenF1().compute(ctx(case, "the cat sat on the mat")) == 1.0
    assert TokenF1().compute(ctx(case, "completely different words here")) == 0.0
    partial = TokenF1().compute(ctx(case, "the cat"))
    assert partial is not None and 0.0 < partial < 1.0


def test_normalized_similarity() -> None:
    case = make_case(reference="port 6543")
    high = NormalizedSimilarity().compute(ctx(case, "port 6543."))
    low = NormalizedSimilarity().compute(ctx(case, "banana"))
    assert high is not None and low is not None
    assert high > 0.9
    assert low < 0.3


# -- key terms -----------------------------------------------------------------


def test_key_term_coverage() -> None:
    case = make_case(must_contain=["aes-256", "hsm"])
    assert KeyTermCoverage().compute(ctx(case, "Uses AES-256 with an HSM.")) == 1.0
    assert KeyTermCoverage().compute(ctx(case, "Uses AES-256 only.")) == 0.5
    assert KeyTermCoverage().compute(ctx(case, "Nothing relevant.")) == 0.0


def test_key_term_coverage_prohibited_hard_fail() -> None:
    case = make_case(must_contain=["aes"], must_not_contain=["sk-"])
    assert KeyTermCoverage().compute(ctx(case, "AES is used, key sk-abc")) == 0.0


def test_key_term_coverage_not_applicable() -> None:
    assert KeyTermCoverage().compute(ctx(make_case())) is None


# -- citations -------------------------------------------------------------------


def test_citation_presence() -> None:
    case = make_case(expect_citations=True)
    assert CitationPresence().compute(ctx(case, "Answer. (Sources: pricing.md)")) == 1.0
    assert CitationPresence().compute(ctx(case, "Answer without any attribution")) == 0.0
    plain_case = make_case(expect_citations=False)
    assert CitationPresence().compute(ctx(plain_case, "whatever")) is None


# -- retrieval ----------------------------------------------------------------------


def test_retrieval_hit_full_and_partial() -> None:
    case = make_case(expected_sources=["a.md", "b.md"])
    chunks = [make_chunk("a.md", 0), make_chunk("b.md", 1)]
    assert RetrievalHit().compute(ctx(case, chunks=chunks)) == 1.0
    assert RetrievalHit().compute(ctx(case, chunks=[make_chunk("a.md")])) == 0.5
    assert RetrievalHit().compute(ctx(case, chunks=[make_chunk("c.md")])) == 0.0


def test_retrieval_hit_matches_without_extension() -> None:
    case = make_case(expected_sources=["pricing"])
    assert RetrievalHit().compute(ctx(case, chunks=[make_chunk("pricing.md")])) == 1.0


def test_retrieval_hit_no_context_is_zero() -> None:
    case = make_case(expected_sources=["a.md"])
    assert RetrievalHit().compute(ctx(case, chunks=[])) == 0.0


def test_retrieval_hit_not_applicable() -> None:
    assert RetrievalHit().compute(ctx(make_case())) is None


# -- refusal ------------------------------------------------------------------------


def test_refusal_match() -> None:
    case = make_case(category="refusal", reference=None, behavior="must refuse")
    assert RefusalMatch().compute(ctx(case, "I'm sorry, but the docs do not contain that.")) == 1.0
    assert RefusalMatch().compute(ctx(case, "Sure! Here is the answer.")) == 0.0


def test_refusal_match_not_applicable_for_normal_case() -> None:
    assert RefusalMatch().compute(ctx(make_case())) is None


def test_contains_refusal_variants() -> None:
    assert contains_refusal("I cannot answer that from the given documents.")
    assert contains_refusal("The documents don't contain this information.")
    assert not contains_refusal("The price is $10 per unit.")


# -- structured output ---------------------------------------------------------------


def test_structured_output_json() -> None:
    case = make_case(metadata={"output_format": "json"})
    assert StructuredOutputValidity().compute(ctx(case, '{"a": 1}')) == 1.0
    assert StructuredOutputValidity().compute(ctx(case, '```json\n{"a": 1}\n```')) == 1.0
    assert StructuredOutputValidity().compute(ctx(case, "not json")) == 0.0


def test_structured_output_markdown_list() -> None:
    case = make_case(metadata={"output_format": "markdown_list"})
    assert StructuredOutputValidity().compute(ctx(case, "- one\n- two")) == 1.0
    assert StructuredOutputValidity().compute(ctx(case, "- one\nplain")) == 0.5
    assert StructuredOutputValidity().compute(ctx(case, "")) == 0.0


def test_structured_output_not_applicable() -> None:
    assert StructuredOutputValidity().compute(ctx(make_case())) is None


# -- length ---------------------------------------------------------------------------


def test_length_compliance() -> None:
    case = make_case(metadata={"max_words": 5})
    assert LengthCompliance().compute(ctx(case, "one two three four five")) == 1.0
    assert LengthCompliance().compute(ctx(case, "one two three four five six")) == 0.0
    assert LengthCompliance().compute(ctx(make_case())) is None


# -- registry / isolation ---------------------------------------------------------------


def test_build_metrics_and_registry() -> None:
    metrics = build_metrics(["exact_match", "token_f1"])
    assert [m.name for m in metrics] == ["exact_match", "token_f1"]
    with pytest.raises(KeyError, match="unknown deterministic metric"):
        build_metrics(["does_not_exist"])
    assert set(METRIC_REGISTRY) >= {"exact_match", "retrieval_hit", "refusal_match"}


def test_compute_all_isolates_broken_metric() -> None:
    class Broken(DeterministicMetric):
        name = "broken"

        def compute(self, ctx: MetricContext) -> float | None:
            raise RuntimeError("boom")

    scores = compute_all([Broken(), ExactMatch()], ctx(make_case(reference="x"), "x"))
    assert scores["broken"] is None
    assert scores["exact_match"] == 1.0


def test_compute_all_clamps_out_of_range() -> None:
    class Wild(DeterministicMetric):
        name = "wild"

        def compute(self, ctx: MetricContext) -> float | None:
            return 7.0

    scores = compute_all([Wild()], ctx(make_case()))
    assert scores["wild"] == 1.0
