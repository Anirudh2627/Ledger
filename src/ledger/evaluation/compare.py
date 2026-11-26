"""High-level comparison service: load two runs, compare, apply the policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ledger.config.settings import LedgerConfig
from ledger.regression.comparator import ComparisonResult, RunData, compare_runs
from ledger.regression.detector import GateDecision, detect_regressions
from ledger.regression.policy import RegressionPolicy
from ledger.tracking.artifacts import ArtifactStore
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GateOutcome:
    """Bundle of everything produced by a baseline-vs-candidate comparison."""

    baseline: RunData
    candidate: RunData
    comparison: ComparisonResult
    decision: GateDecision


def load_run_data(
    store: ArtifactStore,
    run_ref: str | Path,
    *,
    verify_decisions: bool = True,
) -> RunData:
    """Load summary + per-case results for a run reference."""
    run_dir = store.resolve_run(run_ref)
    snapshot = store.load_config_snapshot(run_dir)
    if verify_decisions and snapshot is None:
        raise ValueError(f"run {run_dir} has no config snapshot; cannot verify result decisions")
    decision_config = _decision_config(snapshot) if snapshot is not None else None
    return RunData(
        summary=store.load_summary(run_dir),
        results=store.load_results(run_dir),
        decision_config=decision_config,
    )


def _decision_config(snapshot: dict[str, object]) -> LedgerConfig:
    """Validate the saved decision inputs without parsing redacted provider settings."""
    required = {
        "judge": {
            "enabled",
            "dimensions",
            "weights",
            "pass_threshold",
            "critical_pass_threshold",
            "on_failure",
        },
        "evaluation": {"fallback_pass_threshold"},
    }
    sections: dict[str, dict[str, object]] = {}
    for section, keys in required.items():
        values = snapshot.get(section)
        if not isinstance(values, dict) or not keys.issubset(values):
            raise ValueError(f"config snapshot is missing result decision inputs in '{section}'")
        sections[section] = {key: values[key] for key in keys}
    return LedgerConfig.model_validate(sections)


def compare_and_gate(
    store: ArtifactStore,
    baseline_ref: str | Path,
    candidate_ref: str | Path,
    policy: RegressionPolicy,
) -> GateOutcome:
    """Compare two persisted runs and evaluate the regression policy."""
    baseline = load_run_data(store, baseline_ref)
    candidate = load_run_data(store, candidate_ref)
    logger.info(
        "comparing baseline run %s (%s) vs candidate run %s (%s)",
        baseline.summary.run_id,
        baseline.summary.system_version,
        candidate.summary.run_id,
        candidate.summary.system_version,
    )
    comparison = compare_runs(
        baseline,
        candidate,
        n_samples=policy.bootstrap.n_samples,
        confidence=policy.bootstrap.confidence,
        seed=policy.bootstrap.seed,
    )
    decision = detect_regressions(comparison, baseline.summary, candidate.summary, policy)
    return GateOutcome(
        baseline=baseline, candidate=candidate, comparison=comparison, decision=decision
    )
