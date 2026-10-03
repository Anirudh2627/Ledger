"""Golden-dataset test-case schema.

A test case is the atomic unit of evaluation. The schema is deliberately
tolerant of optional fields (not every case has a reference answer, e.g.
refusal or ambiguity cases) but strict about types and identity.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Categories used by the bundled golden dataset. Free-form strings are
#: allowed (validator warns) so teams can extend the taxonomy without code
#: changes.
KNOWN_CATEGORIES = (
    "factual_correctness",
    "groundedness",
    "answer_relevance",
    "retrieval_correctness",
    "instruction_following",
    "refusal",
    "ambiguity",
    "edge_case",
    "adversarial",
)

#: Judge dimensions a per-case rubric may provide criteria for.
RUBRIC_DIMENSIONS = (
    "correctness",
    "relevance",
    "groundedness",
    "instruction_following",
)


class TestCase(BaseModel):
    """One versioned evaluation case from the golden dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=0, max_length=20_000)
    #: Optional static context supplied directly with the case (for
    #: context-injection style systems). RAG systems retrieve their own
    #: context and may ignore this field.
    context: str | None = None
    #: Gold answer used by lexical metrics and shown to the judge. May be
    #: absent for refusal/ambiguity cases where *behaviour* is the target.
    reference_answer: str | None = None
    #: Human-readable description of the required behaviour ("must refuse",
    #: "must answer in exactly three bullets", ...). Shown to the judge.
    expected_behavior: str | None = None
    category: str = Field(min_length=1, max_length=64)
    #: Critical cases gate the release independently of aggregate scores.
    critical: bool = False
    #: Per-case rubric criteria keyed by judge dimension.
    rubric: dict[str, str] = Field(default_factory=dict)
    #: Document ids/filenames that a correct retrieval must surface.
    expected_sources: list[str] = Field(default_factory=list)
    #: Substrings the answer must contain (case-insensitive, normalized).
    must_contain: list[str] = Field(default_factory=list)
    #: Substrings the answer must NOT contain (e.g. fabricated claims).
    must_not_contain: list[str] = Field(default_factory=list)
    #: Expect the answer to cite its sources.
    expect_citations: bool = False
    #: Free-form tags for slicing ("security", "billing", "long-context", ...).
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    #: Structured-output expectations, e.g. ``{"output_format": "json",
    #: "max_words": 20}`` used by deterministic metrics.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("category")
    @classmethod
    def _normalize_category(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("category must not be empty")
        return value

    @field_validator("rubric")
    @classmethod
    def _check_rubric_keys(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = set(value) - set(RUBRIC_DIMENSIONS)
        if unknown:
            raise ValueError(
                f"rubric keys must be a subset of {RUBRIC_DIMENSIONS}, "
                f"got unknown: {sorted(unknown)}"
            )
        return value

    @model_validator(mode="after")
    def _check_signal_presence(self) -> TestCase:
        """Every case must give the evaluator *something* to check against."""
        has_signal = (
            self.reference_answer
            or self.expected_behavior
            or self.must_contain
            or self.rubric
            or self.expected_sources
        )
        if not has_signal:
            raise ValueError(
                f"case '{self.id}' has no evaluation signal: provide at least one of "
                "reference_answer, expected_behavior, must_contain, rubric, expected_sources"
            )
        return self

    @property
    def expects_refusal(self) -> bool:
        """True when the required behaviour is to refuse / decline answering."""
        if self.category == "refusal":
            return True
        behavior = (self.expected_behavior or "").lower()
        return any(marker in behavior for marker in ("refuse", "decline", "must not answer"))
