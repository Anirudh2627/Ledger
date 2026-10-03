"""Unit tests: judge response parsing, validation and RubricJudge resilience."""

from __future__ import annotations

import json

import pytest

from ledger.config.settings import JudgeConfig
from ledger.judges.base import finalize_score, pass_threshold_for, resolve_weights, weighted_overall
from ledger.judges.prompts import build_judge_messages
from ledger.judges.rubric import JudgeParseError, RubricJudge, validate_dimensions
from ledger.utils.json_extract import JSONExtractionError, extract_json_object
from tests.helpers import ScriptedProvider, make_case, make_output

pytestmark = pytest.mark.unit

VALID_VERDICT = json.dumps(
    {
        "correctness": 4,
        "relevance": 5,
        "groundedness": 4,
        "instruction_following": 5,
        "overall": 4.5,
        "pass": True,
        "reason": "Mostly correct and well grounded.",
    }
)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_plain_json() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_fenced_json() -> None:
    text = 'Sure!\n```json\n{"a": 1}\n```\nHope that helps.'
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_with_surrounding_prose() -> None:
    text = 'Here is the verdict: {"a": {"b": 2}} - end.'
    assert extract_json_object(text) == {"a": {"b": 2}}


def test_extract_json_with_trailing_comma() -> None:
    assert extract_json_object('{"a": 1,}') == {"a": 1}


def test_extract_json_with_line_comments() -> None:
    text = '{\n// leading comment\n"a": 1\n}'
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_braces_inside_strings() -> None:
    text = '{"reason": "contains } and { braces"}'
    assert extract_json_object(text) == {"reason": "contains } and { braces"}


def test_extract_json_errors() -> None:
    with pytest.raises(JSONExtractionError):
        extract_json_object("no json here at all")
    with pytest.raises(JSONExtractionError):
        extract_json_object("")
    with pytest.raises(JSONExtractionError):
        extract_json_object("[1, 2, 3]")  # array, not object


# ---------------------------------------------------------------------------
# dimension validation
# ---------------------------------------------------------------------------

DIMS = ["correctness", "relevance", "groundedness", "instruction_following"]


def test_validate_dimensions_accepts_ints_and_integral_floats() -> None:
    payload = {"correctness": 4, "relevance": 5.0, "groundedness": 3, "instruction_following": 4}
    assert validate_dimensions(payload, DIMS) == {
        "correctness": 4,
        "relevance": 5,
        "groundedness": 3,
        "instruction_following": 4,
    }


def test_validate_dimensions_missing_key() -> None:
    with pytest.raises(JudgeParseError, match="missing dimension"):
        validate_dimensions({"correctness": 4}, DIMS)


def test_validate_dimensions_out_of_range() -> None:
    payload = {"correctness": 9, "relevance": 5, "groundedness": 3, "instruction_following": 4}
    with pytest.raises(JudgeParseError, match="out of range"):
        validate_dimensions(payload, DIMS)


def test_validate_dimensions_rejects_non_numeric() -> None:
    payload = {"correctness": "four", "relevance": 5, "groundedness": 3, "instruction_following": 4}
    with pytest.raises(JudgeParseError, match="not numeric"):
        validate_dimensions(payload, DIMS)


def test_validate_dimensions_rejects_bool() -> None:
    payload = {"correctness": True, "relevance": 5, "groundedness": 3, "instruction_following": 4}
    with pytest.raises(JudgeParseError, match="not numeric"):
        validate_dimensions(payload, DIMS)


# ---------------------------------------------------------------------------
# score math
# ---------------------------------------------------------------------------


def test_weighted_overall_and_finalization() -> None:
    config = JudgeConfig(weights={"correctness": 2.0, "groundedness": 2.0})
    weights = resolve_weights(config)
    dims = {"correctness": 5, "relevance": 3, "groundedness": 3, "instruction_following": 3}
    # (5*2 + 3*1 + 3*2 + 3*1) / 6 = 22/6
    assert weighted_overall(dims, weights) == pytest.approx(22 / 6)
    score = finalize_score(dims, critical=False, config=config, reason="r")
    assert score.overall == pytest.approx(22 / 6, abs=1e-4)
    assert score.normalized == pytest.approx((22 / 6 - 1) / 4)
    assert score.passed is (22 / 6 >= 3.5)


def test_pass_threshold_for_critical_cases() -> None:
    config = JudgeConfig(pass_threshold=3.5, critical_pass_threshold=4.0)
    assert pass_threshold_for(False, config) == 3.5
    assert pass_threshold_for(True, config) == 4.0
    no_critical = JudgeConfig(pass_threshold=3.5, critical_pass_threshold=None)
    assert pass_threshold_for(True, no_critical) == 3.5


# ---------------------------------------------------------------------------
# RubricJudge resilience
# ---------------------------------------------------------------------------


def _judge(
    responses: list, config: JudgeConfig | None = None
) -> tuple[RubricJudge, ScriptedProvider]:
    provider = ScriptedProvider(responses)
    return RubricJudge(config or JudgeConfig(max_retries=2), provider), provider


def test_judge_happy_path() -> None:
    judge, provider = _judge([VALID_VERDICT])
    score = judge.score(make_case(), make_output())
    assert not score.failed
    assert score.dimensions["correctness"] == 4
    assert score.attempts == 1
    assert provider.requests[0].purpose == "judge"
    assert provider.requests[0].response_format == "json"


def test_judge_recomputes_overall_and_pass() -> None:
    """The judge's self-reported overall/pass are diagnostics only."""
    bogus = json.dumps(
        {
            "correctness": 2,
            "relevance": 2,
            "groundedness": 2,
            "instruction_following": 2,
            "overall": 99,
            "pass": True,
            "reason": "lying judge",
        }
    )
    judge, _ = _judge([bogus])
    score = judge.score(make_case(), make_output())
    assert score.overall == pytest.approx(2.0)
    assert score.passed is False  # 2.0 < 3.5 regardless of judge's claim


def test_judge_repairs_malformed_response() -> None:
    judge, provider = _judge(["I think the answer is fine!", VALID_VERDICT])
    score = judge.score(make_case(), make_output())
    assert not score.failed
    assert score.attempts == 2
    # second request must include the repair follow-up
    assert any("could not be parsed" in m.content for m in provider.requests[1].messages)


def test_judge_repairs_invalid_scores() -> None:
    bad_scores = json.dumps(
        {"correctness": 7, "relevance": 2, "groundedness": 2, "instruction_following": 2}
    )
    judge, _ = _judge([bad_scores, VALID_VERDICT])
    score = judge.score(make_case(), make_output())
    assert not score.failed
    assert score.attempts == 2


def test_judge_fails_gracefully_after_exhausted_retries() -> None:
    judge, _ = _judge(["garbage"] * 5, config=JudgeConfig(max_retries=2))
    score = judge.score(make_case(), make_output())
    assert score.failed
    assert score.error is not None and "judge failed" in score.error
    assert score.attempts == 3  # initial + 2 retries


def test_judge_survives_provider_exceptions() -> None:
    judge, _ = _judge([RuntimeError("api down")] * 3, config=JudgeConfig(max_retries=1))
    score = judge.score(make_case(), make_output())
    assert score.failed
    assert "provider error" in (score.error or "")


def test_judge_critical_threshold_applied() -> None:
    middling = json.dumps(
        {
            "correctness": 4,
            "relevance": 4,
            "groundedness": 4,
            "instruction_following": 3,
            "overall": 3.75,
            "pass": True,
            "reason": "ok",
        }
    )
    config = JudgeConfig(pass_threshold=3.5, critical_pass_threshold=4.0)
    judge, _ = _judge([middling], config=config)
    assert judge.score(make_case(critical=True), make_output()).passed is False
    judge2, _ = _judge([middling], config=config)
    assert judge2.score(make_case(critical=False), make_output()).passed is True


def test_judge_prompt_contains_expected_sections() -> None:
    case = make_case(behavior="must refuse", rubric={"correctness": "strict"})
    messages = build_judge_messages(case, make_output(), JudgeConfig())
    system, user = messages[0].content, messages[1].content
    assert "1 to 5" in system
    for tag in (
        "<question>",
        "<reference_answer>",
        "<expected_behavior>",
        "<context>",
        "<candidate_answer>",
        "<pass_threshold>",
        "<category>",
        "<critical>",
    ):
        assert tag in user
    assert "strict" in user  # case rubric is included
