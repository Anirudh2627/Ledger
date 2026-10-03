"""High-level comparison service: load two runs, compare, apply the policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def load_run_data(store: ArtifactStore, run_ref: str | Path) -> RunData:
    """Load summary + per-case results for a run reference."""
    run_dir = store.resolve_run(run_ref)
    return RunData(summary=store.load_summary(run_dir), results=store.load_results(run_dir))


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
