"""End-to-end evaluation pipeline.

    load config -> load & validate dataset -> build provider + SUT
      -> run SUT over cases -> score (deterministic + judge)
      -> aggregate -> persist artifacts -> return run handle

This module is the single entry point used by the CLI, the scripts/ wrappers,
the CI gate and the test-suite; everything else composes it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ledger.config.settings import LedgerConfig
from ledger.core.models import (
    DatasetInfo,
    EvaluationResult,
    ProviderInfo,
    RunSummary,
)
from ledger.dataset.loader import load_dataset
from ledger.dataset.schema import TestCase
from ledger.dataset.validator import validate_cases
from ledger.evaluation.evaluator import Evaluator, JudgeAbortError
from ledger.evaluation.runner import CaseOutcome, EvaluationRunner
from ledger.judges.rubric import RubricJudge
from ledger.metrics.aggregate import (
    category_aggregates,
    count_stats,
    latency_stats,
    metric_means,
)
from ledger.metrics.deterministic import build_metrics
from ledger.providers.registry import build_provider
from ledger.systems.factory import build_system
from ledger.tracking.artifacts import ArtifactStore
from ledger.utils.logging import get_logger, setup_logging
from ledger.utils.provenance import runtime_fingerprint
from ledger.utils.text import sha256_file

logger = get_logger(__name__)


class PipelineError(RuntimeError):
    """Raised when an evaluation run cannot be completed."""


@dataclass
class RunArtifacts:
    """Handle to everything a finished run produced."""

    run_id: str
    run_dir: Path
    results_path: Path
    summary_path: Path
    summary: RunSummary
    results: list[EvaluationResult] = field(default_factory=list, repr=False)


def run_evaluation(
    config: LedgerConfig,
    *,
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    run_id: str | None = None,
    store: ArtifactStore | None = None,
    progress: bool = True,
) -> RunArtifacts:
    """Execute one full evaluation run and persist its artifacts.

    Args:
        config: fully resolved Ledger configuration.
        dataset_path: override ``config.dataset.path``.
        limit: override ``config.dataset.limit`` (smoke runs, CI fast path).
        run_id: force a specific run id (CI caching); generated when ``None``.
        store: injectable artifact store (tests point it at tmp dirs).
        progress: log progress while running.

    Raises:
        PipelineError: on invalid datasets or aborted judging.
    """
    setup_logging(config.logging.level)
    started = time.perf_counter()

    store = store or ArtifactStore(
        results_dir=config.output.results_dir,
        reports_dir=config.output.reports_dir,
        experiments_dir=config.output.experiments_dir,
    )

    # -- dataset ------------------------------------------------------------
    ds_path = Path(dataset_path or config.dataset.path)
    cases = load_dataset(
        ds_path,
        limit=limit if limit is not None else config.dataset.limit,
        categories=config.dataset.categories,
    )
    validation = validate_cases(cases)
    for issue in validation.warnings:
        logger.warning("dataset: %s", issue)
    if validation.errors:
        details = "; ".join(str(e) for e in validation.errors[:10])
        raise PipelineError(f"golden dataset failed validation: {details}")
    fingerprint = sha256_file(ds_path)

    # -- system + judge -------------------------------------------------------
    provider = build_provider(config.provider, seed=config.run.seed)
    sut = build_system(config, provider=provider)
    judge = None
    if config.judge.enabled:
        judge_provider = build_provider(config.judge.provider, seed=config.run.seed)
        judge = RubricJudge(config.judge, judge_provider)

    evaluator = Evaluator(
        metrics=build_metrics(config.metrics.deterministic),
        judge=judge,
        judge_config=config.judge,
        evaluation_config=config.evaluation,
    )

    # -- execute ----------------------------------------------------------------
    run_id = run_id or store.new_run_id(config.run.system_version)
    logger.info(
        "run %s: system_version=%s cases=%d provider=%s judge=%s",
        run_id,
        config.run.system_version,
        len(cases),
        config.provider.name,
        "on" if judge else "off",
    )

    runner = EvaluationRunner(
        sut,
        max_workers=config.run.max_concurrency,
        retries=config.run.sut_retries,
        progress_callback=_progress_logger(cases, enabled=progress),
    )
    outcomes = runner.run(cases)

    try:
        results = _score_outcomes(
            outcomes,
            evaluator,
            run_id=run_id,
            system_version=config.run.system_version,
            max_workers=config.run.max_concurrency,
        )
    except JudgeAbortError as exc:
        raise PipelineError(str(exc)) from exc

    # -- aggregate + persist ------------------------------------------------------
    elapsed_s = time.perf_counter() - started
    summary = _build_summary(
        run_id=run_id,
        config=config,
        results=results,
        dataset_path=ds_path,
        fingerprint=fingerprint,
        sut_description=sut.describe(),
        judge_enabled=judge is not None,
        elapsed_s=elapsed_s,
    )

    run_dir = store.ensure_run_dir(run_id)
    results_path = store.save_results(run_id, results)
    summary_path = store.save_summary(run_id, summary)
    store.save_config_snapshot(run_id, config.model_dump(mode="json"))
    store.record_experiment(
        {
            "run_id": run_id,
            "system_version": config.run.system_version,
            "created_at": summary.created_at.isoformat(),
            "dataset_path": str(ds_path),
            "dataset_fingerprint": fingerprint,
            "n_cases": len(cases),
            "provider": config.provider.name,
            "provider_model": config.provider.model,
            "judge_enabled": judge is not None,
            "overall": summary.metrics.get("overall"),
            "pass_rate": summary.metrics.get("pass_rate"),
            "summary_path": str(summary_path),
        }
    )

    logger.info(
        "run %s finished in %.1fs | overall=%s pass_rate=%s | artifacts: %s",
        run_id,
        elapsed_s,
        _fmt(summary.metrics.get("overall")),
        _fmt(summary.metrics.get("pass_rate")),
        run_dir,
    )
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        results_path=results_path,
        summary_path=summary_path,
        summary=summary,
        results=results,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _score_outcomes(
    outcomes: list[CaseOutcome],
    evaluator: Evaluator,
    *,
    run_id: str,
    system_version: str,
    max_workers: int,
) -> list[EvaluationResult]:
    """Score outcomes (judge calls parallelized; mock judge is instant)."""

    def score(outcome: CaseOutcome) -> EvaluationResult:
        return evaluator.evaluate_case(
            outcome.case,
            outcome.output,
            run_id=run_id,
            system_version=system_version,
        )

    if max_workers <= 1 or len(outcomes) == 1:
        return [score(o) for o in outcomes]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(score, outcomes))


def _build_summary(
    *,
    run_id: str,
    config: LedgerConfig,
    results: list[EvaluationResult],
    dataset_path: Path,
    fingerprint: str,
    sut_description: dict[str, Any],
    judge_enabled: bool,
    elapsed_s: float,
) -> RunSummary:
    runtime: dict[str, Any] = runtime_fingerprint()
    runtime.update(
        {
            "elapsed_s": round(elapsed_s, 2),
            "seed": config.run.seed,
            "system": sut_description,
            "judge_on_failure": config.judge.on_failure if judge_enabled else None,
        }
    )
    prompt_version = next((r.prompt_version for r in results if r.prompt_version), None)
    return RunSummary(
        run_id=run_id,
        system_version=config.run.system_version,
        created_at=datetime.now(UTC),
        dataset=DatasetInfo(path=str(dataset_path), fingerprint=fingerprint, n_cases=len(results)),
        provider=ProviderInfo(
            name=config.provider.name,
            model=config.provider.model,
            temperature=config.provider.temperature,
            is_mock=config.provider.name == "mock",
        ),
        judge=(
            ProviderInfo(
                name=config.judge.provider.name,
                model=config.judge.provider.model,
                is_mock=config.judge.provider.name == "mock",
            )
            if judge_enabled
            else None
        ),
        judge_enabled=judge_enabled,
        prompt_version=prompt_version,
        metrics={k: round(v, 6) for k, v in metric_means(results).items()},
        categories=category_aggregates(results),
        counts=count_stats(results),
        latency=latency_stats(results),
        runtime=runtime,
    )


def _progress_logger(cases: list[TestCase], *, enabled: bool) -> Callable[[int, int], None] | None:
    if not enabled:
        return None
    total = len(cases)
    step = max(1, total // 5)

    def callback(completed: int, _total: int) -> None:
        if completed % step == 0 or completed == total:
            logger.info("progress: %d/%d cases", completed, total)

    return callback


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Small shared helper for writing JSON artifacts pretty-printed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return out
