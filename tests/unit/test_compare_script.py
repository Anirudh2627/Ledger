from __future__ import annotations

from types import SimpleNamespace

import pytest
from scripts import compare as compare_script


@pytest.mark.parametrize(
    ("status", "fail_on_warning", "expected_exit_code"),
    [
        ("PASS", False, 0),
        ("WARNING", False, 0),
        ("WARNING", True, 1),
        ("FAIL", False, 1),
    ],
)
def test_compare_script_exit_codes(
    monkeypatch, status: str, fail_on_warning: bool, expected_exit_code: int
) -> None:
    outcome = SimpleNamespace(
        comparison=object(),
        decision=SimpleNamespace(status=status),
        baseline=object(),
        candidate=object(),
    )
    report = SimpleNamespace(artifacts={})
    store = SimpleNamespace(reports_dir="reports")

    monkeypatch.setattr(compare_script, "load_dotenv", lambda: None)
    monkeypatch.setattr(compare_script, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compare_script, "load_config", lambda _path: None)
    monkeypatch.setattr(compare_script, "load_policy", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(compare_script, "ArtifactStore", lambda **_kwargs: store)
    monkeypatch.setattr(compare_script, "compare_and_gate", lambda *_args: outcome)
    monkeypatch.setattr(compare_script, "build_report", lambda *_args: report)
    monkeypatch.setattr(compare_script, "write_reports", lambda *_args: {})
    monkeypatch.setattr(compare_script, "render_console", lambda *_args: None)

    args = ["--baseline", "base", "--candidate", "candidate"]
    if fail_on_warning:
        args.append("--fail-on-warning")

    assert compare_script.main(args) == expected_exit_code
