"""Security checks for shell interpolation in GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


def test_workflow_context_is_not_interpolated_into_shell_scripts() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "llm-evaluation.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["evaluation-gate"]["steps"]

    assert steps
    assert all("${{" not in step.get("run", "") for step in steps)
