"""Unit tests: artifact store, run ids, redaction, experiment index."""

from __future__ import annotations

import json
import re
from typing import Any, cast

import pytest

from ledger.tracking.artifacts import ArtifactStore
from ledger.utils.env import load_dotenv, redact_secrets
from tests.helpers import make_result, make_summary

pytestmark = pytest.mark.unit

RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[a-z0-9-]+-[0-9a-f]{6}$")


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        experiments_dir=tmp_path / "experiments",
    )


def test_run_id_format_and_uniqueness(store) -> None:
    a = store.new_run_id("baseline-v1")
    b = store.new_run_id("baseline-v1")
    assert RUN_ID_RE.match(a), a
    assert a != b


def test_run_id_slugifies_version(store) -> None:
    run_id = store.new_run_id("Candidate V2 (weird/name)")
    assert "candidate-v2-weird-name" in run_id


def test_results_round_trip(store) -> None:
    results = [make_result("t1"), make_result("t2", passed=False, overall=0.2)]
    summary = make_summary("run-a", "sysA", results)
    store.save_results("run-a", results)
    store.save_summary("run-a", summary)

    loaded = store.load_results("run-a")
    assert [r.test_id for r in loaded] == ["t1", "t2"]
    assert loaded[0] == results[0]
    loaded_summary = store.load_summary("run-a")
    assert loaded_summary.run_id == "run-a"
    assert loaded_summary.metrics == summary.metrics


def test_config_snapshot_redacts_secrets(store) -> None:
    store.save_config_snapshot(
        "run-s",
        {
            "provider": {"api_key": "sk-live-SUPERSECRET", "api_key_env": "X"},
            "judge": {"provider": {"secret_token": "another-secret"}},
            "run": {"seed": 1},
        },
    )
    path = store.run_dir("run-s") / "config.snapshot.json"
    text = path.read_text(encoding="utf-8")
    assert "SUPERSECRET" not in text
    assert "another-secret" not in text
    assert "***redacted***" in text
    assert json.loads(text)["run"]["seed"] == 1


def test_resolve_run_by_id_prefix_and_path(store) -> None:
    results = [make_result("t1")]
    store.save_summary(
        "20260101T000000Z-sys-a1b2c3", make_summary("20260101T000000Z-sys-a1b2c3", "sys", results)
    )
    # exact id
    assert store.resolve_run("20260101T000000Z-sys-a1b2c3").name == "20260101T000000Z-sys-a1b2c3"
    # unique prefix
    assert store.resolve_run("20260101T000000Z").name == "20260101T000000Z-sys-a1b2c3"
    # direct path
    assert store.resolve_run(store.run_dir("20260101T000000Z-sys-a1b2c3")).is_dir()


def test_resolve_run_ambiguous_and_missing(store) -> None:
    for run_id in ("20260101T000000Z-a-111111", "20260101T000001Z-a-222222"):
        store.save_summary(run_id, make_summary(run_id, "sys", [make_result("t1")]))
    with pytest.raises(ValueError, match="ambiguous"):
        store.resolve_run("2026")
    with pytest.raises(FileNotFoundError):
        store.resolve_run("does-not-exist")


def test_list_runs_newest_first_and_latest(store) -> None:
    ids = [
        "20260101T000000Z-old-aaaaaa",
        "20260102T000000Z-new-bbbbbb",
        "20260101T120000Z-mid-cccccc",
    ]
    for run_id in ids:
        store.save_summary(run_id, make_summary(run_id, "sys", [make_result("t1")]))
    runs = store.list_runs()
    assert [r.run_id for r in runs] == sorted(ids, reverse=True)
    assert store.latest_run().run_id == "20260102T000000Z-new-bbbbbb"


def test_load_results_missing_run(store) -> None:
    with pytest.raises(FileNotFoundError):
        store.load_results("nope")


def test_experiment_index_appends(store) -> None:
    store.record_experiment({"run_id": "a", "overall": 0.8})
    store.record_experiment({"run_id": "b", "overall": 0.7})
    index = store.experiments_dir / "index.jsonl"
    lines = index.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["run_id"] == "b"


def test_corrupt_result_row_reports_line(store) -> None:
    directory = store.ensure_run_dir("run-bad")
    (directory / "results.jsonl").write_text('{"test_id": broken}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt result row"):
        store.load_results("run-bad")


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------


def test_load_dotenv_basic(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nLEDGER_DOTENV_TEST_A=plain\nLEDGER_DOTENV_TEST_B='quoted value'\n"
        'export LEDGER_DOTENV_TEST_C="double"\n\nBAD LINE\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("LEDGER_DOTENV_TEST_A", raising=False)
    loaded = load_dotenv(env_file)
    assert loaded["LEDGER_DOTENV_TEST_A"] == "plain"
    assert loaded["LEDGER_DOTENV_TEST_B"] == "quoted value"
    assert loaded["LEDGER_DOTENV_TEST_C"] == "double"


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LEDGER_DOTENV_KEEP", "original")
    env_file = tmp_path / ".env"
    env_file.write_text("LEDGER_DOTENV_KEEP=replaced\n", encoding="utf-8")
    load_dotenv(env_file)
    import os

    assert os.environ["LEDGER_DOTENV_KEEP"] == "original"


def test_load_dotenv_missing_file(tmp_path) -> None:
    assert load_dotenv(tmp_path / "missing.env") == {}


def test_redact_secrets_nested() -> None:
    data = {
        "a": {"api_key": "secret", "keep": "visible"},
        "list": [{"authorization": "Bearer x"}, {"ok": 1}],
        "password": "hunter2",
    }
    redacted = cast("dict[str, Any]", redact_secrets(data))
    inner = cast("dict[str, Any]", redacted["a"])
    listed = cast("list[dict[str, Any]]", redacted["list"])
    assert inner["api_key"] == "***redacted***"
    assert inner["keep"] == "visible"
    assert listed[0]["authorization"] == "***redacted***"
    assert listed[1]["ok"] == 1
    assert redacted["password"] == "***redacted***"
