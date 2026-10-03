"""Configurable regression policy.

All gate thresholds live in YAML (``configs/policy.yaml``) - never in code.
The policy model validates them and the detector applies them to a
:class:`~ledger.regression.comparator.ComparisonResult`.

Policy semantics (defaults in brackets):
  * ``require_significance`` [true] - a degradation only FAILs when the
    paired bootstrap supports it (CI excludes zero or p < alpha). Degradation
    beyond a threshold *without* statistical support degrades to WARNING.
  * ``overall`` [absolute 0.03, relative null] - allowed drop of the primary
    quality score (0-1 scale) before failing.
  * ``categories`` [absolute 0.05] - per-category allowed drop.
  * ``critical`` [absolute 0.02] - allowed drop across all critical cases and
    max newly-failing critical cases [0].
  * ``metrics`` - optional per-metric drop thresholds (e.g. retrieval_hit).
  * ``counts`` - max newly failing tests [3], max total failed tests [null =
    off], max judge failure rate [0.1].
  * ``warning_fraction`` [0.5] - degradations between ``threshold *
    warning_fraction`` and the threshold raise WARNINGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PolicyError(ValueError):
    """Raised for invalid policy configuration."""


class MetricThreshold(BaseModel):
    """Allowed degradation for one metric (absolute on the 0-1 scale)."""

    model_config = ConfigDict(extra="forbid")

    absolute: float | None = Field(default=None, ge=0.0, le=1.0)
    relative: float | None = Field(default=None, ge=0.0, le=1.0)


class ThresholdGroup(BaseModel):
    """Threshold + warning band used for overall/categories/critical."""

    model_config = ConfigDict(extra="forbid")

    absolute: float = Field(ge=0.0, le=1.0)
    relative: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Fraction of the fail threshold that triggers a WARNING instead.
    warning_fraction: float = Field(default=0.5, ge=0.0, le=1.0)


class CountPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Max tests that pass on baseline but fail on candidate.
    max_newly_failing_tests: int | None = Field(default=3, ge=0)
    #: Max total failing (not passed) candidate tests; null disables.
    max_failed_tests: int | None = Field(default=None, ge=0)
    #: Max newly-failing CRITICAL tests (0 = any critical regression fails).
    max_critical_failures: int | None = Field(default=0, ge=0)
    #: Max fraction of cases whose judge verdict failed (evaluation integrity).
    max_judge_failure_rate: float | None = Field(default=0.1, ge=0.0, le=1.0)


class BootstrapPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(default=2000, ge=100, le=100_000)
    seed: int = 1337
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)


class RegressionPolicy(BaseModel):
    """The complete, YAML-driven quality-gate policy."""

    model_config = ConfigDict(extra="forbid")

    require_significance: bool = True
    significance_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    #: Below this many paired cases the gate emits a low-power WARNING.
    min_paired_cases: int = Field(default=10, ge=1, le=100_000)
    overall: ThresholdGroup = Field(default_factory=lambda: ThresholdGroup(absolute=0.03))
    categories: ThresholdGroup = Field(default_factory=lambda: ThresholdGroup(absolute=0.05))
    critical: ThresholdGroup = Field(default_factory=lambda: ThresholdGroup(absolute=0.02))
    #: Per-metric drop thresholds keyed by metric name.
    metrics: dict[str, MetricThreshold] = Field(default_factory=dict)
    counts: CountPolicy = Field(default_factory=CountPolicy)
    bootstrap: BootstrapPolicy = Field(default_factory=BootstrapPolicy)

    @model_validator(mode="after")
    def _check_coherence(self) -> RegressionPolicy:
        if self.significance_alpha >= 1.0 - self.bootstrap.confidence + 1e-9:
            # e.g. alpha=0.05 with confidence=0.95 is the coherent pairing.
            pass  # informational only: alpha and CI level are independent knobs
        return self

    # -- loading ---------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, overrides: dict[str, Any] | None = None
    ) -> RegressionPolicy:
        """Load a policy file, optionally deep-merged with inline overrides."""
        file_path = Path(path)
        if not file_path.is_file():
            raise PolicyError(f"policy file not found: {file_path}")
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise PolicyError(f"policy root must be a mapping: {file_path}")
        merged = _deep_merge(raw, overrides or {})
        try:
            return cls.model_validate(merged)
        except ValueError as exc:
            raise PolicyError(f"invalid policy {file_path}: {exc}") from exc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionPolicy:
        try:
            return cls.model_validate(data)
        except ValueError as exc:
            raise PolicyError(f"invalid policy: {exc}") from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_policy(
    policy_path: str | Path | None = None,
    *,
    inline: dict[str, Any] | None = None,
) -> RegressionPolicy:
    """Resolve a policy from path + inline overrides, or fall back to defaults."""
    if policy_path:
        return RegressionPolicy.from_yaml(policy_path, overrides=inline)
    if inline:
        return RegressionPolicy.from_dict(inline)
    return RegressionPolicy()
