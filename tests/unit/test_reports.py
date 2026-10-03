"""Unit tests: report building and rendering (markdown, JSON, console)."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from ledger.regression.comparator import RunData, compare_runs
from ledger.regression.detector import detect_regressions
from ledger.regression.policy import RegressionPolicy
from ledger.reporting.console import render_console
from ledger.reporting.json_report import render_json, write_reports
from ledger.reporting.markdown import render_markdown, render_run_markdown
from ledger.reporting.report import Report, build_report, report_headline
from tests.helpers import make_result, make_summary

pytestmark = pytest.mark.unit

N = 12


def _outcome(delta: float = -0.1):
    base_results = [
        make_result(
            f"t{i}",
            overall=0.8,
            passed=True,
            category="factual_correctness" if i % 2 == 0 else "refusal",
            critical=i < 2,
            det={"token_f1": 0.6},
            judge_dims={
                "correctness": 4,
                "relevance": 4,
                "groundedness": 4,
                "instruction_following": 4,
            },
            run_id="b",
            system_version="A",
        )
        for i in range(N)
    ]
    cand_results = [
        make_result(
            f"t{i}",
            overall=max(0.0, 0.8 + delta),
            passed=0.8 + delta >= 0.5,
            category="factual_correctness" if i % 2 == 0 else "refusal",
            critical=i < 2,
            det={"token_f1": max(0.0, 0.6 + delta)},
            judge_dims={
                "correctness": 4,
                "relevance": 4,
                "groundedness": 4,
                "instruction_following": 4,
            },
            run_id="c",
            system_version="B",
        )
        for i in range(N)
    ]
    base = RunData(make_summary("b", "A", base_results), base_results)
    cand = RunData(make_summary("c", "B", cand_results), cand_results)
    comparison = compare_runs(base, cand, n_samples=500, seed=3)
    decision = detect_regressions(comparison, base.summary, cand.summary, RegressionPolicy())
    return base, cand, comparison, decision


@pytest.fixture
def regressed_report() -> Report:
    base, cand, comparison, decision = _outcome(delta=-0.1)
    return build_report(comparison, decision, base, cand)


def test_report_structure(regressed_report: Report) -> None:
    report = regressed_report
    assert report.schema_version == "1.0"
    assert report.baseline.system_version == "A"
    assert report.candidate.system_version == "B"
    assert report.decision.status == "FAIL"
    assert report.comparison.overall is not None
    assert report.comparison.overall.delta == pytest.approx(-0.1, abs=1e-6)
    # all 12 cases regressed by score; the report caps excerpts at 10
    assert len(report.top_regressions) == 10


def test_report_headline(regressed_report: Report) -> None:
    headline = report_headline(regressed_report)
    assert headline["decision"] == "FAIL"
    assert headline["overall_delta"] == pytest.approx(-0.1, abs=1e-6)


def test_markdown_rendering(regressed_report: Report) -> None:
    markdown = render_markdown(regressed_report)
    assert "# Ledger Evaluation Report" in markdown
    assert "FAIL" in markdown
    assert "| **overall** |" in markdown
    assert "## Categories" in markdown
    assert "critical cases" in markdown
    assert "## Gate decision" in markdown
    assert "overall_regression" in markdown  # violated rule id
    assert "## Top regressed cases" in markdown
    assert "paired_percentile" in markdown  # method disclosure


def test_markdown_pass_report() -> None:
    base, cand, comparison, decision = _outcome(delta=0.0)
    report = build_report(comparison, decision, base, cand)
    markdown = render_markdown(report)
    assert "PASS" in markdown
    assert "## Top regressed cases" not in markdown


def test_json_rendering_round_trip(regressed_report: Report) -> None:
    payload = render_json(regressed_report)
    parsed = json.loads(payload)
    assert parsed["decision"]["status"] == "FAIL"
    assert parsed["baseline"]["system_version"] == "A"
    restored = Report.model_validate_json(payload)
    assert restored == regressed_report


def test_write_reports_creates_files(regressed_report: Report, tmp_path) -> None:
    paths = write_reports(regressed_report, tmp_path / "out")
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert "comparison_A_vs_B" in paths["json"].name
    content = json.loads(paths["json"].read_text())
    assert content["candidate"]["system_version"] == "B"


def test_console_rendering_smoke(regressed_report: Report) -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    render_console(regressed_report, console=console)
    output = buffer.getvalue()
    assert "LEDGER EVALUATION REPORT" in output
    assert "Baseline:" in output
    assert "Candidate:" in output
    assert "Decision:" in output
    assert "FAIL" in output


def test_run_markdown_rendering() -> None:
    results = [make_result("t1", overall=0.8, det={"token_f1": 0.5})]
    summary = make_summary("run-1", "sysA", results)
    markdown = render_run_markdown(summary)
    assert "# Ledger Run Report - `sysA`" in markdown
    assert "run-1" in markdown
    assert "`token_f1`" in markdown
    assert "## Headline metrics" in markdown
