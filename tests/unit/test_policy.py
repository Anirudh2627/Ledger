"""Unit tests: regression policy loading and validation."""

from __future__ import annotations

import pytest
import yaml

from ledger.regression.policy import PolicyError, RegressionPolicy, load_policy

pytestmark = pytest.mark.unit


def _write_policy(tmp_path, data: dict, name: str = "policy.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_defaults() -> None:
    policy = RegressionPolicy()
    assert policy.require_significance is True
    assert policy.overall.absolute == pytest.approx(0.03)
    assert policy.categories.absolute == pytest.approx(0.05)
    assert policy.critical.absolute == pytest.approx(0.02)
    assert policy.counts.max_newly_failing_tests == 3
    assert policy.counts.max_critical_failures == 0
    assert policy.counts.max_failed_tests is None
    assert policy.bootstrap.n_samples == 2000
    assert policy.bootstrap.confidence == pytest.approx(0.95)


def test_from_yaml(tmp_path) -> None:
    path = _write_policy(
        tmp_path,
        {
            "require_significance": False,
            "overall": {"absolute": 0.1},
            "counts": {"max_failed_tests": 5},
            "metrics": {"token_f1": {"absolute": 0.05}},
        },
    )
    policy = RegressionPolicy.from_yaml(path)
    assert policy.require_significance is False
    assert policy.overall.absolute == pytest.approx(0.1)
    assert policy.counts.max_failed_tests == 5
    assert policy.metrics["token_f1"].absolute == pytest.approx(0.05)
    # untouched fields keep defaults
    assert policy.counts.max_newly_failing_tests == 3


def test_from_yaml_with_inline_overrides(tmp_path) -> None:
    path = _write_policy(tmp_path, {"overall": {"absolute": 0.1}})
    policy = RegressionPolicy.from_yaml(path, overrides={"overall": {"absolute": 0.2}})
    assert policy.overall.absolute == pytest.approx(0.2)


def test_missing_file(tmp_path) -> None:
    with pytest.raises(PolicyError, match="not found"):
        RegressionPolicy.from_yaml(tmp_path / "nope.yaml")


def test_invalid_values_rejected(tmp_path) -> None:
    path = _write_policy(tmp_path, {"overall": {"absolute": 1.5}})
    with pytest.raises(PolicyError):
        RegressionPolicy.from_yaml(path)


def test_unknown_keys_rejected(tmp_path) -> None:
    path = _write_policy(tmp_path, {"nonsense_key": True})
    with pytest.raises(PolicyError):
        RegressionPolicy.from_yaml(path)


def test_non_mapping_root_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="mapping"):
        RegressionPolicy.from_yaml(path)


def test_load_policy_fallbacks(tmp_path) -> None:
    path = _write_policy(tmp_path, {"overall": {"absolute": 0.07}})
    assert load_policy(path).overall.absolute == pytest.approx(0.07)
    assert load_policy(
        None, inline={"overall": {"absolute": 0.09}}
    ).overall.absolute == pytest.approx(0.09)
    assert load_policy(None).overall.absolute == pytest.approx(0.03)


def test_repo_policy_file_is_valid(repo_root) -> None:
    """The shipped configs/policy.yaml must load with the shipped defaults."""
    policy = RegressionPolicy.from_yaml(repo_root / "configs" / "policy.yaml")
    assert policy.overall.absolute == pytest.approx(0.03)
    assert policy.counts.max_newly_failing_tests == 3
    assert "retrieval_hit" in policy.metrics
