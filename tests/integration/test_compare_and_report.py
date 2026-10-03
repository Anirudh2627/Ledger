"""Integration tests: baseline-vs-candidate comparison, reports and gate wiring."""

from __future__ import annotations

import json

import pytest

from ledger.config.settings import load_config
from ledger.evaluation.compare import compare_and_gate, load_run_data
from ledger.evaluation.pipeline import run_evaluation
from ledger.regression.policy import RegressionPolicy
from ledger.reporting.json_report import write_reports
from ledger.reporting.markdown import render_markdown
from ledger.reporting.report import build_report
from ledger.tracking.artifacts import ArtifactStore
from tests.helpers import write_config_file

pytestmark = pytest.mark.integration


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        experiments_dir=tmp_path / "experiments",
    )


def _mock_config(tmp_path, name: str, version: str, mode: str):
    return load_config(
        write_config_file(
            tmp_path,
            {"system": {"kind": "mock", "mode": mode}, "run": {"system_version": version}},
            name=name,
        )
    )


def _run_pair(tmp_path):
    """Evaluate an upper-bound (reference) and a degraded mock system."""
    base = run_evaluation(
        _mock_config(tmp_path, "base.yaml", "good-v1", "reference"),
        store=_store(tmp_path),
        progress=False,
    )
    cand = run_evaluation(
        _mock_config(tmp_path, "cand.yaml", "bad-v2", "degraded"),
        store=_store(tmp_path),
        progress=False,
    )
    return base, cand


def test_degraded_candidate_fails_gate(tmp_path) -> None:
    base, cand = _run_pair(tmp_path)
    outcome = compare_and_gate(_store(tmp_path), base.run_id, cand.run_id, RegressionPolicy())

    assert outcome.decision.status == "FAIL"
    rules = {v.rule for v in outcome.decision.fails}
    assert "overall_regression" in rules
    assert "newly_failing_tests" in rules or "critical_failures" in rules
    assert outcome.comparison.overall is not None
    assert outcome.comparison.overall.delta < -0.1


def test_identical_systems_pass_gate(tmp_path) -> None:
    store = _store(tmp_path)
    a = run_evaluation(
        _mock_config(tmp_path, "a.yaml", "same-v1", "reference"), store=store, progress=False
    )
    b = run_evaluation(
        _mock_config(tmp_path, "b.yaml", "same-v2", "reference"), store=store, progress=False
    )
    # mini dataset has 6 cases; relax the low-power warning floor accordingly
    policy = RegressionPolicy(min_paired_cases=5)
    outcome = compare_and_gate(store, a.run_id, b.run_id, policy)
    assert outcome.decision.status == "PASS"
    assert outcome.comparison.overall is not None
    assert outcome.comparison.overall.delta == pytest.approx(0.0, abs=1e-9)


def test_full_report_written_for_regressed_pair(tmp_path) -> None:
    base, cand = _run_pair(tmp_path)
    outcome = compare_and_gate(_store(tmp_path), base.run_id, cand.run_id, RegressionPolicy())
    report = build_report(outcome.comparison, outcome.decision, outcome.baseline, outcome.candidate)
    paths = write_reports(report, tmp_path / "reports")

    markdown_text = paths["markdown"].read_text(encoding="utf-8")
    assert "# Ledger Evaluation Report" in markdown_text
    assert "FAIL" in markdown_text
    assert "good-v1" in markdown_text and "bad-v2" in markdown_text

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["decision"]["status"] == "FAIL"
    assert payload["comparison"]["n_common"] == 6
    assert render_markdown(report) == markdown_text


def test_load_run_data_round_trip(tmp_path) -> None:
    base, _cand = _run_pair(tmp_path)
    data = load_run_data(_store(tmp_path), base.run_id)
    assert data.summary.run_id == base.run_id
    assert len(data.results) == 6
    assert {r.test_id for r in data.results} == {r.test_id for r in base.results}


def test_shipped_configs_produce_comparable_runs(repo_root, tmp_path, monkeypatch) -> None:
    """The shipped baseline/candidate configs must run and compare cleanly.

    Uses the real golden dataset with a small limit and forced mock mode;
    artifacts are redirected into tmp_path so the workspace stays clean.
    """
    monkeypatch.setenv("LEDGER_FORCE_MOCK", "1")
    store = _store(tmp_path)
    base_cfg = load_config(repo_root / "configs" / "baseline.yaml")
    cand_cfg = load_config(repo_root / "configs" / "candidate.yaml")
    base = run_evaluation(base_cfg, limit=10, store=store, progress=False)
    cand = run_evaluation(cand_cfg, limit=10, store=store, progress=False)

    assert base.summary.dataset.fingerprint == cand.summary.dataset.fingerprint
    outcome = compare_and_gate(store, base.run_id, cand.run_id, RegressionPolicy())
    assert outcome.decision.status in {"PASS", "WARNING", "FAIL"}
    assert outcome.comparison.n_common == 10
