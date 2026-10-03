"""Unit tests: configuration loading, layering, interpolation and secrets."""

from __future__ import annotations

import pytest
import yaml

from ledger.config.settings import ConfigError, LedgerConfig, RagSystemConfig, load_config

pytestmark = pytest.mark.unit


def _write(tmp_path, name: str, data: dict):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_defaults_without_file() -> None:
    config = load_config(None)
    assert config.provider.name == "mock"
    assert config.judge.enabled is True
    assert isinstance(config.system, RagSystemConfig)
    assert config.run.seed == 1337


def test_extends_merges_parent(tmp_path) -> None:
    _write(
        tmp_path,
        "parent.yaml",
        {
            "run": {"seed": 1, "system_version": "parent"},
            "provider": {"name": "mock", "model": "m1"},
        },
    )
    child = _write(
        tmp_path, "child.yaml", {"extends": "parent.yaml", "run": {"system_version": "child"}}
    )
    config = load_config(child)
    assert config.run.system_version == "child"  # overridden
    assert config.run.seed == 1  # inherited
    assert config.provider.model == "m1"  # inherited sibling key


def test_extends_chain(tmp_path) -> None:
    _write(tmp_path, "a.yaml", {"run": {"seed": 5}})
    _write(tmp_path, "b.yaml", {"extends": "a.yaml", "run": {"system_version": "b"}})
    c = _write(tmp_path, "c.yaml", {"extends": "b.yaml", "logging": {"level": "DEBUG"}})
    config = load_config(c)
    assert config.run.seed == 5
    assert config.run.system_version == "b"
    assert config.logging.level == "DEBUG"


def test_circular_extends_detected(tmp_path) -> None:
    _write(tmp_path, "x.yaml", {"extends": "y.yaml"})
    _write(tmp_path, "y.yaml", {"extends": "x.yaml"})
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "x.yaml")


def test_missing_file(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_env_interpolation_with_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LEDGER_TEST_UNSET_VAR", raising=False)
    monkeypatch.setenv("LEDGER_TEST_SET_VAR", "from-env")
    path = _write(
        tmp_path,
        "cfg.yaml",
        {
            "provider": {
                "name": "mock",
                "model": "${LEDGER_TEST_SET_VAR:-fallback}",
                "base_url": "${LEDGER_TEST_UNSET_VAR:-http://localhost:8000/v1}",
            }
        },
    )
    config = load_config(path)
    assert config.provider.model == "from-env"
    assert config.provider.base_url == "http://localhost:8000/v1"


def test_unresolvable_placeholder_left_intact(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LEDGER_TEST_MISSING", raising=False)
    path = _write(tmp_path, "cfg.yaml", {"run": {"system_version": "${LEDGER_TEST_MISSING}"}})
    config = load_config(path)
    assert config.run.system_version == "${LEDGER_TEST_MISSING}"


def test_inline_api_key_rejected(tmp_path) -> None:
    path = _write(
        tmp_path,
        "cfg.yaml",
        {
            "provider": {
                "name": "openai_compat",
                "api_key": "sk-secret-123",
                "base_url": "http://x/v1",
            }
        },
    )
    with pytest.raises(ConfigError, match="api_key"):
        load_config(path)


def test_api_key_resolved_from_env_and_excluded_from_dump(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LEDGER_TEST_KEY_XYZ", "super-secret-value")
    path = _write(
        tmp_path,
        "cfg.yaml",
        {
            "provider": {
                "name": "openai_compat",
                "api_key_env": "LEDGER_TEST_KEY_XYZ",
                "base_url": "http://x/v1",
            }
        },
    )
    config = load_config(path)
    assert config.provider.resolved_api_key() == "super-secret-value"
    dumped = config.model_dump(mode="json")
    assert "super-secret-value" not in yaml.safe_dump(dumped)
    assert dumped["provider"].get("api_key") is None  # excluded from serialization


def test_force_mock_overrides_providers(tmp_path) -> None:
    path = _write(
        tmp_path,
        "cfg.yaml",
        {
            "provider": {
                "name": "openai_compat",
                "base_url": "http://x/v1",
                "api_key_env": "LEDGER_TEST_KEY_XYZ",
            },
            "judge": {"provider": {"name": "openai_compat", "base_url": "http://x/v1"}},
            "system": {
                "kind": "rag",
                "embedder": {"name": "openai_compat", "base_url": "http://x/v1"},
            },
        },
    )
    config = load_config(path, force_mock=True)
    assert config.provider.name == "mock"
    assert config.judge.provider.name == "mock"
    assert isinstance(config.system, RagSystemConfig)
    assert config.system.embedder.name == "hashing_tfidf"


def test_force_mock_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LEDGER_FORCE_MOCK", "1")
    path = _write(
        tmp_path, "cfg.yaml", {"provider": {"name": "openai_compat", "base_url": "http://x/v1"}}
    )
    config = load_config(path)
    assert config.provider.name == "mock"


def test_unknown_keys_rejected(tmp_path) -> None:
    path = _write(tmp_path, "cfg.yaml", {"not_a_section": {"x": 1}})
    with pytest.raises(ConfigError, match="invalid config"):
        load_config(path)


def test_chunk_overlap_must_be_smaller_than_chunk_size(tmp_path) -> None:
    path = _write(
        tmp_path, "cfg.yaml", {"system": {"kind": "rag", "chunk_size": 100, "chunk_overlap": 200}}
    )
    with pytest.raises(ConfigError):
        load_config(path)


def test_invalid_judge_weights_rejected(tmp_path) -> None:
    path = _write(tmp_path, "cfg.yaml", {"judge": {"weights": {"style": 1.0}}})
    with pytest.raises(ConfigError):
        load_config(path)


def test_repo_configs_load_and_extend_default(repo_root, monkeypatch) -> None:
    """The shipped baseline/candidate/degraded configs must load cleanly."""
    monkeypatch.setenv("LEDGER_FORCE_MOCK", "1")
    for name, version in [
        ("baseline.yaml", "baseline-v1"),
        ("candidate.yaml", "candidate-v2"),
        ("candidate_degraded.yaml", "candidate-degraded-demo"),
    ]:
        config = load_config(repo_root / "configs" / name)
        assert config.run.system_version == version
        assert config.provider.name == "mock"
        assert config.dataset.path == "datasets/golden.jsonl"
        assert config.policy_path == "configs/policy.yaml"


def test_config_round_trip_serialization() -> None:
    config = LedgerConfig()
    restored = LedgerConfig.model_validate_json(config.model_dump_json())
    assert restored == config
