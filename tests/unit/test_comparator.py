"""Unit tests: paired run comparison."""

from __future__ import annotations

import pytest

from ledger.regression.comparator import RunData, compare_runs
from tests.helpers import make_result, make_summary

pytestmark = pytest.mark.unit

N = 12


def _runs(delta: float = -0.1, *, flip_ids: tuple[str, ...] = (), fingerprints=("fp", "fp")):
    base_results = [
        make_result(
            f"t{i}",
            overall=0.8,
            passed=True,
            category="factual_correctness" if i % 2 == 0 else "refusal",
            critical=(i < 3),
            det={"token_f1": 0.6},
            judge_dims={
                "correctness": 4,
                "relevance": 4,
                "groundedness": 4,
                "instruction_following": 4,
            },
            run_id="base",
            system_version="A",
        )
        for i in range(N)
    ]
    cand_results = [
        make_result(
            f"t{i}",
            overall=0.8 + delta,
            passed=f"t{i}" not in flip_ids,
            category="factual_correctness" if i % 2 == 0 else "refusal",
            critical=(i < 3),
            det={"token_f1": 0.6 + delta},
            judge_dims={
                "correctness": 4,
                "relevance": 4,
                "groundedness": 4,
                "instruction_following": 4,
            },
            run_id="cand",
            system_version="B",
        )
        for i in range(N)
    ]
    base = RunData(
        summary=make_summary("base", "A", base_results, fingerprint=fingerprints[0]),
        results=base_results,
    )
    cand = RunData(
        summary=make_summary("cand", "B", cand_results, fingerprint=fingerprints[1]),
        results=cand_results,
    )
    return base, cand


def test_overall_comparison_math() -> None:
    base, cand = _runs(delta=-0.1)
    comp = compare_runs(base, cand, n_samples=500, seed=1)
    assert comp.overall is not None
    assert comp.overall.baseline_mean == pytest.approx(0.8)
    assert comp.overall.candidate_mean == pytest.approx(0.7)
    assert comp.overall.delta == pytest.approx(-0.1)
    assert comp.overall.relative_delta == pytest.approx(-0.125)
    assert comp.overall.n_paired == N
    assert comp.overall.significant_regression is True


def test_metric_and_category_slices() -> None:
    base, cand = _runs(delta=-0.1)
    comp = compare_runs(base, cand, n_samples=500, seed=1)
    assert "token_f1" in comp.metrics
    assert comp.metrics["token_f1"].delta == pytest.approx(-0.1)
    assert "judge_correctness" in comp.metrics
    assert set(comp.categories) == {"factual_correctness", "refusal", "__critical__"}
    assert comp.categories["__critical__"].n_paired == 3


def test_flips_and_critical_regressions() -> None:
    base, cand = _runs(delta=0.0, flip_ids=("t0", "t5"))
    comp = compare_runs(base, cand, n_samples=500, seed=1)
    assert comp.newly_failing == ["t0", "t5"]
    assert comp.critical_regressions == ["t0"]  # t0 is critical, t5 is not
    by_id = {c.test_id: c for c in comp.cases}
    assert by_id["t0"].status == "regressed"
    assert by_id["t1"].status == "stable_pass"


def test_improved_cases() -> None:
    base_results = [make_result(f"t{i}", overall=0.4, passed=False) for i in range(N)]
    cand_results = [make_result(f"t{i}", overall=0.9, passed=True) for i in range(N)]
    base = RunData(make_summary("b", "A", base_results), base_results)
    cand = RunData(make_summary("c", "B", cand_results), cand_results)
    comp = compare_runs(base, cand, n_samples=500, seed=1)
    assert len(comp.newly_passing) == N
    assert comp.cases[0].status == "improved"
    assert comp.overall is not None and comp.overall.significant_improvement


def test_missing_cases_on_one_side() -> None:
    base, cand = _runs(delta=0.0)
    cand.results = cand.results[:-2]  # drop t10, t11
    comp = compare_runs(base, cand, n_samples=500, seed=1)
    assert comp.n_common == N - 2
    statuses = {c.test_id: c.status for c in comp.cases}
    assert statuses["t10"] == "baseline_only"
    assert any(w.startswith("2 cases present only in baseline") for w in comp.warnings)


def test_dataset_fingerprint_mismatch_warns() -> None:
    base, cand = _runs(delta=0.0, fingerprints=("fp1", "fp2"))
    comp = compare_runs(base, cand, n_samples=500, seed=1)
    assert comp.dataset_fingerprint_match is False
    assert any("fingerprint" in w for w in comp.warnings)


def test_sign_test_reported() -> None:
    base, cand = _runs(delta=0.0, flip_ids=("t1", "t2", "t3"))
    comp = compare_runs(base, cand, n_samples=500, seed=1)
    assert comp.sign_test_flips["newly_failing"] == 3
    assert comp.sign_test_p_value is not None and comp.sign_test_p_value <= 1.0


def test_pass_rate_metric_paired() -> None:
    base, cand = _runs(delta=0.0, flip_ids=("t0",))
    comp = compare_runs(base, cand, n_samples=500, seed=1)
    assert "pass_rate" in comp.metrics
    # comparison values are rounded to 6 decimals
    assert comp.metrics["pass_rate"].delta == pytest.approx(-1 / N, abs=1e-6)
