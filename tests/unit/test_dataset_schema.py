"""Unit tests: golden-dataset schema (TestCase)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ledger.dataset.schema import TestCase

pytestmark = pytest.mark.unit


def _valid_payload() -> dict:
    return {
        "id": "test_001",
        "question": "What encryption is used at rest?",
        "context": None,
        "reference_answer": "AES-256.",
        "expected_behavior": "State AES-256.",
        "category": "groundedness",
        "critical": False,
        "rubric": {"correctness": "Must name AES-256."},
    }


def test_valid_case_parses() -> None:
    case = TestCase.model_validate(_valid_payload())
    assert case.id == "test_001"
    assert case.category == "groundedness"
    assert case.rubric["correctness"].startswith("Must name")


def test_extra_fields_rejected() -> None:
    payload = _valid_payload()
    payload["not_a_field"] = 1
    with pytest.raises(ValidationError):
        TestCase.model_validate(payload)


def test_case_without_any_signal_rejected() -> None:
    payload = _valid_payload()
    payload["reference_answer"] = None
    payload["expected_behavior"] = None
    payload["rubric"] = {}
    with pytest.raises(ValidationError, match="no evaluation signal"):
        TestCase.model_validate(payload)


def test_unknown_rubric_dimension_rejected() -> None:
    payload = _valid_payload()
    payload["rubric"] = {"style": "must be pretty"}
    with pytest.raises(ValidationError, match="rubric keys"):
        TestCase.model_validate(payload)


def test_expects_refusal_from_category() -> None:
    payload = _valid_payload()
    payload["category"] = "refusal"
    payload["reference_answer"] = None
    case = TestCase.model_validate(payload)
    assert case.expects_refusal is True


def test_expects_refusal_from_behavior() -> None:
    payload = _valid_payload()
    payload["expected_behavior"] = "The system should refuse to answer."
    case = TestCase.model_validate(payload)
    assert case.expects_refusal is True


def test_expects_refusal_false_for_normal_case() -> None:
    case = TestCase.model_validate(_valid_payload())
    assert case.expects_refusal is False


def test_defaults() -> None:
    payload = _valid_payload()
    case = TestCase.model_validate(payload)
    assert case.critical is False
    assert case.expected_sources == []
    assert case.must_contain == []
    assert case.must_not_contain == []
    assert case.expect_citations is False
    assert case.metadata == {}
