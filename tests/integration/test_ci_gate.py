"""Integration tests: CLI behaviour and the CI gate contract (exit codes)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ledger.cli.main import app
from tests.helpers import write_config_file

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """CLI tests must never pick up ambient Ledger env configuration."""
    for var in (
        "LEDGER_FORCE_MOCK",
        "LEDGER_PROVIDER_NAME",
        "LEDGER_JUDGE_NAME",
        "LEDGER_EMBEDDER_NAME",
        "LEDGER_LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _evaluate(runner_args: list[str]) -> str:
    result = runner.invoke(app, runner_args)
    assert result.exit_code == 0, result.output
    return result.output.strip().splitlines()[-1].strip()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_cli_evaluate_prints_run_id(make_config_file, tmp_path) -> None:
    config = make_config_file({"run": {"system_version": "cli-sys"}})
    run_id = _evaluate(["evaluate", "--config", str(config), "--quiet"])
    assert run_id.startswith("20") and "cli-sys" in run_id
    assert (tmp_path / "results" / run_id / "summary.json").is_file()


def test_cli_evaluate_table_output(make_config_file) -> None:
    config = make_config_file()
    result = runner.invoke(app, ["evaluate", "--config", str(config)])
    assert result.exit_code == 0
    assert "system_version" in result.output
    assert "overall" in result.output


def test_cli_evaluate_bad_config_exits_3(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("nonsense: [1, 2\n", encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--config", str(bad)])
    assert result.exit_code == 3


# ---------------------------------------------------------------------------
# gate exit-code contract
# ---------------------------------------------------------------------------


def _gate_configs(tmp_path):
    good = write_config_file(
        tmp_path,
        {"system": {"kind": "mock", "mode": "reference"}, "run": {"system_version": "g"}},
        name="good.yaml",
    )
    bad = write_config_file(
        tmp_path,
        {"system": {"kind": "mock", "mode": "degraded"}, "run": {"system_version": "b"}},
        name="bad.yaml",
    )
    return good, bad


def _small_policy(tmp_path):
    """Policy tuned for the 6-case mini dataset (avoids low-power warnings)."""
    import yaml

    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({"min_paired_cases": 2}), encoding="utf-8")
    return str(path)


def test_cli_gate_one_shot_fail_exit_1(tmp_path) -> None:
    good, bad = _gate_configs(tmp_path)
    result = runner.invoke(
        app,
        ["gate", "--config-baseline", str(good), "--config-candidate", str(bad), "--mock"],
    )
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_cli_gate_one_shot_pass_exit_0(tmp_path) -> None:
    good, _ = _gate_configs(tmp_path)
    result = runner.invoke(
        app,
        [
            "gate",
            "--config-baseline",
            str(good),
            "--config-candidate",
            str(good),
            "--mock",
            "--policy",
            _small_policy(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_cli_gate_by_run_refs_and_latest(make_config_file, tmp_path) -> None:
    config = make_config_file({"system": {"kind": "mock", "mode": "reference"}})
    policy = _small_policy(tmp_path)
    run_a = _evaluate(["evaluate", "--config", str(config), "--quiet"])
    run_b = _evaluate(["evaluate", "--config", str(config), "--quiet"])

    # identical systems -> PASS via explicit refs
    result = runner.invoke(
        app,
        [
            "gate",
            "--baseline",
            run_a,
            "--candidate",
            run_b,
            "--config",
            str(config),
            "--policy",
            policy,
        ],
    )
    assert result.exit_code == 0, result.output

    # --latest picks the two newest runs (same systems -> PASS)
    result = runner.invoke(app, ["gate", "--latest", "--config", str(config), "--policy", policy])
    assert result.exit_code == 0, result.output


def test_cli_gate_reports_written(make_config_file, tmp_path) -> None:
    config = make_config_file({"system": {"kind": "mock", "mode": "reference"}})
    _evaluate(["evaluate", "--config", str(config), "--quiet"])
    _evaluate(["evaluate", "--config", str(config), "--quiet"])
    result = runner.invoke(
        app, ["gate", "--latest", "--config", str(config), "--policy", _small_policy(tmp_path)]
    )
    assert result.exit_code == 0
    reports = list((tmp_path / "reports").glob("comparison_*"))
    assert any(p.suffix == ".md" for p in reports)
    assert any(p.suffix == ".json" for p in reports)


def test_cli_compare_no_write(make_config_file, tmp_path) -> None:
    config = make_config_file({"system": {"kind": "mock", "mode": "reference"}})
    _evaluate(["evaluate", "--config", str(config), "--quiet"])
    _evaluate(["evaluate", "--config", str(config), "--quiet"])
    result = runner.invoke(
        app,
        [
            "compare",
            "--latest",
            "--config",
            str(config),
            "--no-write",
            "--policy",
            _small_policy(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "LEDGER EVALUATION REPORT" in result.output
    assert list((tmp_path / "reports").glob("comparison_*")) == []


def test_cli_gate_needs_two_runs_for_latest(make_config_file) -> None:
    config = make_config_file()
    _evaluate(["evaluate", "--config", str(config), "--quiet"])
    result = runner.invoke(app, ["gate", "--latest", "--config", str(config)])
    assert result.exit_code == 3
    assert "at least two" in result.output


# ---------------------------------------------------------------------------
# validate-dataset / report / calibrate / list-runs
# ---------------------------------------------------------------------------


def test_cli_validate_dataset_ok(mini_dataset_path) -> None:
    result = runner.invoke(app, ["validate-dataset", str(mini_dataset_path)])
    assert result.exit_code == 0
    assert "dataset OK" in result.output


def test_cli_validate_dataset_broken(tmp_path) -> None:
    broken = tmp_path / "broken.jsonl"
    row = '{"id": "dup", "question": "q", "category": "refusal", "expected_behavior": "refuse"}'
    broken.write_text(f"{row}\n{row}\n", encoding="utf-8")
    result = runner.invoke(app, ["validate-dataset", str(broken)])
    assert result.exit_code != 0


def test_cli_report_latest(make_config_file, tmp_path) -> None:
    config = make_config_file()
    run_id = _evaluate(["evaluate", "--config", str(config), "--quiet"])
    result = runner.invoke(app, ["report", "--latest", "--config", str(config), "--format", "both"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "reports" / f"run_{run_id}.md").is_file()
    assert (tmp_path / "reports" / f"run_{run_id}.json").is_file()


def test_cli_calibrate_with_mock(mini_labels_path, make_config_file, tmp_path) -> None:
    config = make_config_file()
    result = runner.invoke(
        app, ["calibrate", "--labels", str(mini_labels_path), "--config", str(config), "--mock"]
    )
    assert result.exit_code == 0, result.output
    assert "cohen kappa" in result.output
    assert list((tmp_path / "reports").glob("calibration_*.json"))
    assert list((tmp_path / "reports").glob("calibration_*.md"))


def test_cli_list_runs(make_config_file) -> None:
    config = make_config_file({"run": {"system_version": "listed-sys"}})
    _evaluate(["evaluate", "--config", str(config), "--quiet"])
    result = runner.invoke(app, ["list-runs", "--config", str(config)])
    assert result.exit_code == 0
    assert "listed-sys" in result.output


def test_cli_version_flag() -> None:
    result = runner.invoke(app, ["--version", "list-runs"])
    # --version short-circuits through the callback when a command is present
    assert result.exit_code == 0
    assert "ledger" in result.output
