"""Configuration models and YAML loading for Ledger.

Design principles
-----------------
* **YAML for structure, environment for secrets.** Config files may reference
  an API key only *indirectly* via ``api_key_env: NAME_OF_ENV_VAR``; a literal
  ``api_key`` inside YAML is rejected at load time.
* **Layering.** A config may declare ``extends: default.yaml``; child values
  deep-merge over the parent.
* **Interpolation.** ``${VAR}`` and ``${VAR:-fallback}`` inside string values
  are substituted from the process environment at load time.
* **Reproducibility.** Everything that can change behaviour (retrieval params,
  prompt version, thresholds, seed) lives in config, never in code constants.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from ledger.utils.env import resolve_api_key
from ledger.utils.logging import get_logger

logger = get_logger(__name__)

_INTERPOLATION_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    """Raised for invalid, unsafe or unreadable configuration."""


# ---------------------------------------------------------------------------
# Provider configuration (SUT generation, judge, embeddings)
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel):
    """An OpenAI-compatible chat provider (or the offline deterministic mock)."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["mock", "openai_compat"] = "mock"
    model: str = "mock-model"
    base_url: str | None = None
    #: Name of the environment variable holding the API key. The key itself is
    #: never stored in YAML.
    api_key_env: str | None = None
    #: Resolved at load time from the environment; excluded from serialization.
    api_key: SecretStr | None = Field(default=None, exclude=True)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=128_000)
    timeout_s: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    #: Ask the provider for JSON mode (``response_format={"type": "json_object"}``).
    json_mode: bool = False
    extra_headers: dict[str, str] = Field(default_factory=dict)

    def resolved_api_key(self) -> str | None:
        """Return the API key from env/secret field, or ``None`` in mock mode."""
        if self.api_key is not None:
            return self.api_key.get_secret_value() or None
        return resolve_api_key(self.api_key_env)


# ---------------------------------------------------------------------------
# System-under-test configuration
# ---------------------------------------------------------------------------


class EmbedderConfig(BaseModel):
    """Embedding backend for the sample RAG system."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["hashing_tfidf", "openai_compat"] = "hashing_tfidf"
    dim: int = Field(default=512, ge=32, le=8192)
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)
    timeout_s: float = Field(default=30.0, gt=0.0)
    batch_size: int = Field(default=64, ge=1, le=1024)

    def resolved_api_key(self) -> str | None:
        if self.api_key is not None:
            return self.api_key.get_secret_value() or None
        return resolve_api_key(self.api_key_env)


class RagSystemConfig(BaseModel):
    """Configuration of the bundled sample RAG system."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["rag"] = "rag"
    corpus_dir: str = "data/corpus"
    file_extensions: list[str] = Field(default_factory=lambda: [".md", ".txt"])
    chunk_size: int = Field(default=900, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=4000)
    top_k: int = Field(default=4, ge=1, le=50)
    prompt_version: str = "v1_grounding"
    system_instruction: str | None = None
    max_context_chars: int = Field(default=12_000, ge=200, le=200_000)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)

    @model_validator(mode="after")
    def _check_overlap(self) -> RagSystemConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class MockSystemConfig(BaseModel):
    """A trivial system used to exercise the harness itself.

    Modes:
      * ``reference``  - returns the reference answer (upper-bound plumbing check)
      * ``constant``   - returns a fixed string (baseline-ish neutral output)
      * ``degraded``   - deterministically mangles the reference answer
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["mock"] = "mock"
    mode: Literal["reference", "constant", "degraded"] = "reference"
    constant_answer: str = "I don't know."


SystemConfig = RagSystemConfig | MockSystemConfig
_SYSTEM_ADAPTER: TypeAdapter[SystemConfig] = TypeAdapter(SystemConfig)


# ---------------------------------------------------------------------------
# Run / dataset / judge / metrics / evaluation / output configuration
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_version: str = "dev"
    seed: int = 1337
    max_concurrency: int = Field(default=4, ge=1, le=64)
    #: Per-case retries when the SUT raises (not for judge retries).
    sut_retries: int = Field(default=1, ge=0, le=5)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "datasets/golden.jsonl"
    limit: int | None = Field(default=None, ge=1)
    categories: list[str] | None = None
    max_case_chars: int = Field(default=20_000, ge=100, le=500_000)


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    name: Literal["rubric"] = "rubric"
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    dimensions: list[str] = Field(
        default_factory=lambda: [
            "correctness",
            "relevance",
            "groundedness",
            "instruction_following",
        ]
    )
    #: Relative weights for the weighted overall score (1-5 scale).
    weights: dict[str, float] = Field(default_factory=dict)
    #: A case passes when the weighted overall score reaches this threshold.
    pass_threshold: float = Field(default=3.5, ge=1.0, le=5.0)
    #: Stricter threshold for critical cases (null disables the stricter rule).
    critical_pass_threshold: float | None = Field(default=4.0, ge=1.0, le=5.0)
    #: What to do when judging fails after all retries:
    #: ``exclude`` (drop from judge aggregates, still report), ``fail_case``
    #: (count the case as failed), ``abort`` (stop the whole run).
    on_failure: Literal["exclude", "fail_case", "abort"] = "exclude"
    #: Retries for malformed judge responses.
    max_retries: int = Field(default=2, ge=0, le=5)
    #: Truncate long fields sent to the judge (cost/latency control).
    max_input_chars: int = Field(default=12_000, ge=200, le=200_000)

    @model_validator(mode="after")
    def _check_weights(self) -> JudgeConfig:
        unknown = set(self.weights) - set(self.dimensions)
        if unknown:
            raise ConfigError(f"judge.weights reference unknown dimensions: {sorted(unknown)}")
        if any(w <= 0 for w in self.weights.values()):
            raise ConfigError("judge.weights must all be positive")
        return self


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deterministic: list[str] = Field(
        default_factory=lambda: [
            "exact_match",
            "token_f1",
            "normalized_similarity",
            "key_term_coverage",
            "citation_presence",
            "retrieval_hit",
            "refusal_match",
            "structured_output_validity",
            "length_compliance",
        ]
    )


class EvaluationConfig(BaseModel):
    """How per-case pass/fail and the overall score are derived."""

    model_config = ConfigDict(extra="forbid")

    #: Primary quality signal: judge overall normalized to [0,1]. When the
    #: judge is disabled/failed for a case, fall back to the mean of
    #: applicable deterministic metrics.
    fallback_pass_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_dir: str = "results"
    reports_dir: str = "reports"
    experiments_dir: str = "experiments"


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"


class LedgerConfig(BaseModel):
    """Root configuration object for an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    config_version: int = 1
    run: RunConfig = Field(default_factory=RunConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    system: SystemConfig = Field(default_factory=RagSystemConfig)
    #: Provider used by the system under test for generation.
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    #: Regression policy: path to a YAML file and/or an inline override dict.
    policy_path: str | None = "configs/policy.yaml"
    policy: dict[str, Any] | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("system", mode="before")
    @classmethod
    def _coerce_system(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return _SYSTEM_ADAPTER.validate_python(value)
        return value

    def force_mock(self) -> LedgerConfig:
        """Return a copy of this config with all providers switched to mock."""
        clone = self.model_copy(deep=True)
        clone.provider.name = "mock"
        clone.judge.provider.name = "mock"
        if isinstance(clone.system, RagSystemConfig):
            clone.system.embedder.name = "hashing_tfidf"
        return clone


_CONFIG_ADAPTER: TypeAdapter[LedgerConfig] = TypeAdapter(LedgerConfig)


# ---------------------------------------------------------------------------
# Loading: YAML + extends + env interpolation + secret injection
# ---------------------------------------------------------------------------


def _interpolate_env(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            env_value = os.environ.get(var)
            if env_value is not None and env_value != "":
                return env_value
            if default is not None:
                return default
            return match.group(0)  # leave unresolved placeholder intact

        return _INTERPOLATION_RE.sub(replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (lists and scalars replace).

    Special case: ``system`` is a discriminated union (rag | mock). When the
    override changes ``kind``, the whole section replaces the parent's -
    merging fields across different system kinds produces invalid configs.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            if key == "system" and "kind" in value and merged[key].get("kind") != value.get("kind"):
                merged[key] = copy.deepcopy(value)
            else:
                merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _reject_inline_secrets(node: Any, where: str) -> None:
    """Refuse to load configs that embed literal API keys."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "api_key" and value not in (None, ""):
                raise ConfigError(
                    f"{where}: literal 'api_key' is not allowed in YAML config. "
                    f"Use 'api_key_env: ENV_VAR_NAME' and set the value in the environment or .env."
                )
            _reject_inline_secrets(value, f"{where}.{key}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _reject_inline_secrets(item, f"{where}[{i}]")


def _load_yaml(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load one YAML config, resolving ``extends`` recursively (cycle-safe)."""
    seen = _seen or set()
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in (*seen, resolved))
        raise ConfigError(f"circular 'extends' chain detected: {chain}")
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {path}")
    seen = {*seen, resolved}

    raw_text = resolved.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {path}")

    extends = raw.pop("extends", None)
    if extends is not None:
        parent_path = (path.parent / str(extends)).resolve()
        parent = _load_yaml(parent_path, seen)
        raw = _deep_merge(parent, raw)
    return raw


def load_config(
    path: str | Path | None = None,
    *,
    force_mock: bool | None = None,
    defaults: bool = True,
) -> LedgerConfig:
    """Load a Ledger configuration.

    Args:
        path: YAML config file. ``None`` returns defaults (optionally merged
            with nothing).
        force_mock: force mock providers. ``None`` reads ``LEDGER_FORCE_MOCK``.
        defaults: when a config is loaded, unused sections fall back to the
            built-in defaults (pydantic model defaults).

    Raises:
        ConfigError: on missing files, invalid YAML, inline secrets, cycles or
            failed schema validation.
    """
    if force_mock is None:
        force_mock = os.environ.get("LEDGER_FORCE_MOCK", "0").lower() in {"1", "true", "yes"}

    if path is None:
        config = LedgerConfig()
    else:
        raw = _load_yaml(Path(path))
        _reject_inline_secrets(raw, str(path))
        raw = _interpolate_env(raw)
        try:
            config = _CONFIG_ADAPTER.validate_python(raw)
        except ConfigError:
            raise
        except Exception as exc:  # pydantic ValidationError and friends
            raise ConfigError(f"invalid config {path}: {exc}") from exc
        logger.debug("loaded config from %s", path)

    if not defaults:  # pragma: no cover - reserved for future partial configs
        pass

    if force_mock:
        logger.info("LEDGER_FORCE_MOCK active: all providers switched to deterministic mock")
        config = config.force_mock()

    _resolve_secret_fields(config)
    return config


def _resolve_secret_fields(config: LedgerConfig) -> None:
    """Inject API keys from the environment into SecretStr fields."""
    providers: list[tuple[ProviderConfig | EmbedderConfig, str | None]] = [
        (config.provider, config.provider.api_key_env),
        (config.judge.provider, config.judge.provider.api_key_env),
    ]
    if isinstance(config.system, RagSystemConfig):
        providers.append((config.system.embedder, config.system.embedder.api_key_env))
    for holder, env_name in providers:
        if env_name:
            key = resolve_api_key(env_name)
            if key:
                holder.api_key = SecretStr(key)
