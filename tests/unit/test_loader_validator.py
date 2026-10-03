"""Unit tests: dataset loading and dataset-level validation."""

from __future__ import annotations

import pytest

from ledger.dataset.loader import DatasetError, load_dataset
from ledger.dataset.validator import validate_cases
from tests.helpers import make_case

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


def test_load_mini_dataset(mini_dataset_path) -> None:
    cases = load_dataset(mini_dataset_path)
    assert len(cases) == 6
    assert {c.id for c in cases} >= {"mini-fc-1", "mini-ref-1"}


def test_load_skips_comments_and_blank_lines(tmp_path) -> None:
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '# comment line\n\n{"id": "a", "question": "q?", "category": "refusal",'
        ' "expected_behavior": "refuse"}\n',
        encoding="utf-8",
    )
    cases = load_dataset(path)
    assert [c.id for c in cases] == ["a"]


def test_load_invalid_json_reports_line(tmp_path) -> None:
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '{"id": "a", "question": "q", "category": "c", "reference_answer": "r"}\n{oops\n'
    )
    with pytest.raises(DatasetError, match=r"ds\.jsonl:2"):
        load_dataset(path)


def test_load_duplicate_ids_rejected(tmp_path) -> None:
    path = tmp_path / "ds.jsonl"
    row = '{"id": "dup", "question": "q", "category": "refusal", "expected_behavior": "refuse"}'
    path.write_text(f"{row}\n{row}\n")
    with pytest.raises(DatasetError, match="duplicate case id"):
        load_dataset(path)


def test_load_schema_violation_rejected(tmp_path) -> None:
    path = tmp_path / "ds.jsonl"
    path.write_text('{"id": "x", "question": "q", "category": "refusal"}\n')  # no signal
    with pytest.raises(DatasetError, match="schema violation"):
        load_dataset(path)


def test_load_missing_file(tmp_path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.jsonl")


def test_load_limit_and_category_filter(mini_dataset_path) -> None:
    cases = load_dataset(mini_dataset_path, limit=2)
    assert len(cases) == 2
    refusals = load_dataset(mini_dataset_path, categories=["refusal"])
    assert [c.id for c in refusals] == ["mini-ref-1"]


def test_load_golden_dataset(repo_root) -> None:
    """The committed golden dataset must always load and validate."""
    cases = load_dataset(repo_root / "datasets" / "golden.jsonl")
    assert len(cases) >= 30
    report = validate_cases(cases)
    assert report.ok, [str(e) for e in report.errors]
    assert report.critical_count >= 5


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------


def test_validate_healthy_dataset() -> None:
    cases = [
        make_case("a", category="refusal", reference=None, behavior="must refuse", critical=True),
        make_case("b"),
    ]
    report = validate_cases(cases)
    assert report.ok
    assert report.n_cases == 2
    assert report.critical_count == 1


def test_validate_duplicate_ids() -> None:
    cases = [make_case("dup"), make_case("dup")]
    report = validate_cases(cases)
    assert any("duplicate" in str(e) for e in report.errors)


def test_validate_critical_without_signal_is_error() -> None:
    cases = [make_case("c1", critical=True, reference="ok")]
    # reference present -> fine
    assert validate_cases(cases).ok
    broken = [
        make_case("c2", critical=True, reference=None, behavior=None, rubric={"correctness": "x"})
    ]
    # rubric alone keeps schema happy but critical needs reference/behavior
    assert not validate_cases(broken).ok


def test_validate_unknown_category_warns() -> None:
    cases = [make_case("u", category="made_up_category")]
    report = validate_cases(cases)
    assert report.ok  # warnings only
    assert any("made_up_category" in str(w) for w in report.warnings)


def test_validate_retrieval_case_without_sources_warns() -> None:
    cases = [make_case("r", category="retrieval_correctness")]
    report = validate_cases(cases)
    assert any("expected_sources" in str(w) for w in report.warnings)


def test_validate_empty_dataset_is_error() -> None:
    report = validate_cases([])
    assert not report.ok


def test_validate_no_critical_cases_warns() -> None:
    report = validate_cases([make_case("a"), make_case("b")])
    assert any("no critical" in str(w) for w in report.warnings)
