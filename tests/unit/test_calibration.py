"""Unit tests: judge calibration statistics and pipeline."""

from __future__ import annotations

import pytest

from ledger.config.settings import JudgeConfig
from ledger.core.interfaces import Judge
from ledger.core.models import JudgeScore, SystemOutput
from ledger.dataset.schema import TestCase
from ledger.judges.calibration import (
    HumanLabel,
    _cohen_kappa,
    _dimension_calibration,
    _pearson,
    _spearman,
    calibrate_judge,
    interpret_kappa,
    load_human_labels,
    render_calibration_markdown,
)
from ledger.judges.rubric import RubricJudge
from ledger.providers.mock import MockLLMProvider

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# statistics helpers (hand-computed values)
# ---------------------------------------------------------------------------


def test_dimension_calibration_hand_computed() -> None:
    human = [4.0, 5.0, 3.0]
    judge = [4.0, 4.0, 3.0]
    cal = _dimension_calibration(human, judge)
    assert cal.n == 3
    assert cal.mae == pytest.approx(1 / 3, abs=1e-4)
    assert cal.bias == pytest.approx(-1 / 3, abs=1e-4)
    assert cal.rmse == pytest.approx((1 / 3) ** 0.5, abs=1e-4)
    assert cal.agreement_within_1 == pytest.approx(1.0)
    # pearson: sxy = 1, sxx = 2, syy = 2/3 -> r = 1/sqrt(4/3) = 0.866
    assert cal.pearson_r == pytest.approx(0.866, abs=1e-3)


def test_dimension_calibration_empty() -> None:
    cal = _dimension_calibration([], [])
    assert cal.n == 0
    assert cal.mae is None


def test_pearson_constant_input_is_none() -> None:
    assert _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_pearson_perfect_correlation() -> None:
    assert _pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_spearman_monotonic_nonlinear() -> None:
    # y = x^2 on positive x is perfectly monotonic -> rho = 1
    assert _spearman([1.0, 2.0, 3.0], [1.0, 4.0, 9.0]) == pytest.approx(1.0)


def test_spearman_ties_handled() -> None:
    rho = _spearman([1.0, 1.0, 2.0], [3.0, 3.0, 5.0])
    assert rho is not None and rho == pytest.approx(1.0)


def test_cohen_kappa_hand_computed() -> None:
    # po = 0.75, pe = 0.5 -> kappa = 0.5
    pairs = [(True, True), (True, False), (False, False), (False, False)]
    assert _cohen_kappa(pairs) == pytest.approx(0.5)


def test_cohen_kappa_perfect_and_degenerate() -> None:
    assert _cohen_kappa([(True, True), (False, False)]) == pytest.approx(1.0)
    # all same on both sides: pe == 1, po == 1 -> defined as 1.0
    assert _cohen_kappa([(True, True), (True, True)]) == pytest.approx(1.0)
    assert _cohen_kappa([]) is None


def test_interpret_kappa_bands() -> None:
    assert interpret_kappa(0.9) == "almost perfect"
    assert interpret_kappa(0.7) == "substantial"
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.3) == "fair"
    assert interpret_kappa(0.1) == "slight"
    assert (interpret_kappa(-0.2) or "").startswith("poor")
    assert interpret_kappa(None) is None


# ---------------------------------------------------------------------------
# labels loading
# ---------------------------------------------------------------------------


def test_load_human_labels(mini_labels_path) -> None:
    labels = load_human_labels(mini_labels_path)
    assert len(labels) == 4
    assert all(isinstance(x, HumanLabel) for x in labels)
    assert sum(x.human_pass for x in labels) == 2


def test_load_human_labels_invalid_line(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")  # missing required fields
    with pytest.raises(ValueError, match="invalid label"):
        load_human_labels(path)


def test_load_human_labels_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_human_labels(tmp_path / "nope.jsonl")


def test_load_human_labels_empty_file(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no labels"):
        load_human_labels(path)


# ---------------------------------------------------------------------------
# calibration pipeline (mock judge - offline)
# ---------------------------------------------------------------------------


def test_calibrate_judge_with_mock(mini_labels_path) -> None:
    judge = RubricJudge(JudgeConfig(), MockLLMProvider())
    labels = load_human_labels(mini_labels_path)
    result = calibrate_judge(judge, labels, judge_config=JudgeConfig())

    assert result.n_labels == 4
    assert result.n_judged == 4
    assert result.n_judge_failures == 0
    assert set(result.dimensions) == {
        "correctness",
        "relevance",
        "groundedness",
        "instruction_following",
    }
    for dim_cal in result.dimensions.values():
        assert dim_cal.n == 4
        assert dim_cal.mae is not None and dim_cal.mae >= 0
    assert result.pass_agreement_rate is not None
    assert result.cohen_kappa is not None
    assert result.kappa_interpretation is not None
    assert sum(result.confusion.values()) == 4
    assert len(result.per_label) == 4


def test_calibrate_judge_counts_failures(mini_labels_path) -> None:
    class BrokenJudge(Judge):
        name = "broken"

        def score(self, case: TestCase, output: SystemOutput) -> JudgeScore:
            return JudgeScore(failed=True, error="no verdict")

    labels = load_human_labels(mini_labels_path)
    result = calibrate_judge(BrokenJudge(), labels)
    assert result.n_judged == 0
    assert result.n_judge_failures == 4
    assert result.pass_agreement_rate is None


def test_render_calibration_markdown(mini_labels_path) -> None:
    judge = RubricJudge(JudgeConfig(), MockLLMProvider())
    labels = load_human_labels(mini_labels_path)
    result = calibrate_judge(judge, labels, judge_config=JudgeConfig())
    markdown = render_calibration_markdown(result)
    assert "# Judge Calibration Report" in markdown
    assert "Cohen's kappa" in markdown
    assert "| correctness |" in markdown


def test_repo_labels_file_loads(repo_root) -> None:
    """The shipped human_labels.jsonl must always be loadable."""
    labels = load_human_labels(repo_root / "datasets" / "human_labels.jsonl")
    assert len(labels) >= 10
    assert any(x.human_pass for x in labels)
    assert any(not x.human_pass for x in labels)
