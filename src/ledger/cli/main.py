"""Ledger command-line interface.

Exit codes (CI contract):
    0 - success / gate PASS
    1 - gate FAIL (or operational error)
    2 - gate WARNING (unless --fail-on-warning)
    3 - invalid usage / configuration / dataset errors
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from ledger import __version__
from ledger.config.settings import ConfigError, LedgerConfig, load_config
from ledger.core.models import RunSummary
from ledger.dataset.loader import DatasetError, load_dataset
from ledger.dataset.validator import validate_cases
from ledger.evaluation.compare import GateOutcome, compare_and_gate
from ledger.evaluation.pipeline import PipelineError, run_evaluation
from ledger.judges.calibration import (
    calibrate_judge,
    load_human_labels,
    render_calibration_markdown,
)
from ledger.judges.rubric import RubricJudge
from ledger.providers.registry import build_provider
from ledger.regression.policy import PolicyError, RegressionPolicy, load_policy
from ledger.reporting.console import render_console
from ledger.reporting.json_report import write_reports
from ledger.reporting.markdown import render_run_markdown
from ledger.reporting.report import build_report
from ledger.tracking.artifacts import ArtifactStore
from ledger.utils.env import load_dotenv
from ledger.utils.logging import get_logger, setup_logging

app = typer.Typer(
    name="ledger",
    help="Ledger - regression testing & evaluation infrastructure for LLM applications.",
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()
logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_WARNING = 2
EXIT_USAGE = 3


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
    version: Annotated[
        bool,
        typer.Option("--version", is_eager=True, help="Print the Ledger version and exit."),
    ] = False,
) -> None:
    """Global options."""
    if version:
        console.print(f"ledger {__version__}")
        raise typer.Exit(EXIT_OK)
    load_dotenv()
    setup_logging("DEBUG" if verbose else None, force=verbose)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _die(message: str, code: int = EXIT_USAGE) -> NoReturn:
    console.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code)


def _store_from_config(config: LedgerConfig | None, results_dir: str | None) -> ArtifactStore:
    output = config.output if config else None
    return ArtifactStore(
        results_dir=results_dir or (output.results_dir if output else "results"),
        reports_dir=output.reports_dir if output else "reports",
        experiments_dir=output.experiments_dir if output else "experiments",
    )


def _resolve_policy(policy_path: str | None, config: LedgerConfig | None) -> RegressionPolicy:
    path = policy_path or (config.policy_path if config else "configs/policy.yaml")
    inline = config.policy if config else None
    if path and Path(path).is_file():
        return load_policy(path, inline=inline)
    if inline:
        return load_policy(None, inline=inline)
    if path:
        logger.warning("policy file %s not found; using built-in defaults", path)
    return RegressionPolicy()


def _resolve_ref(
    store: ArtifactStore, ref: str | None, fallback_latest: bool = False
) -> str | Path:
    """Resolve ``latest``, ``latest:<system_version>`` or a concrete run ref."""
    if ref is None:
        if not fallback_latest:
            _die("a run reference is required (run id, path, or 'latest')")
        ref = "latest"
    if ref == "latest" or ref.startswith("latest:"):
        version = None if ref == "latest" else ref.split(":", 1)[1]
        run = store.latest_run(version)
        if run is None:
            _die(f"no stored run found (results_dir={store.results_dir}, version={version})")
        return run.run_id
    return ref


def _load_policy_safe(policy_path: str | None, config: LedgerConfig | None) -> RegressionPolicy:
    try:
        return _resolve_policy(policy_path, config)
    except PolicyError as exc:
        _die(str(exc))
        raise  # unreachable, keeps mypy happy


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="YAML config (system + provider + judge)."),
    ] = None,
    dataset: Annotated[str | None, typer.Option("--dataset", help="Override dataset path.")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Evaluate at most N cases.")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Force a specific run id.")] = None,
    mock: Annotated[
        bool, typer.Option("--mock", help="Force deterministic mock providers (offline).")
    ] = False,
    results_dir: Annotated[
        str | None, typer.Option("--results-dir", help="Override results dir.")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Only print the run id.")] = False,
) -> None:
    """Run a full evaluation of the configured system under test."""
    try:
        cfg = load_config(config, force_mock=True if mock else None)
        store = _store_from_config(cfg, results_dir)
        artifacts = run_evaluation(
            cfg,
            dataset_path=dataset,
            limit=limit,
            run_id=run_id,
            store=store,
            progress=not quiet,
        )
    except (ConfigError, DatasetError, PipelineError, FileNotFoundError, ValueError) as exc:
        _die(str(exc))
        return

    if quiet:
        console.print(artifacts.run_id)
        return
    summary = artifacts.summary
    table = Table(title=f"run {artifacts.run_id}", expand=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("system_version", summary.system_version)
    table.add_row("provider", f"{summary.provider.name} ({summary.provider.model})")
    table.add_row("judge", (summary.judge.model if summary.judge else "disabled") or "-")
    table.add_row("cases", str(summary.dataset.n_cases))
    table.add_row(
        "overall",
        f"{summary.metrics.get('overall', float('nan')):.4f}"
        if summary.metrics.get("overall") is not None
        else "n/a",
    )
    table.add_row(
        "pass_rate",
        f"{summary.metrics.get('pass_rate', float('nan')):.2%}"
        if summary.metrics.get("pass_rate") is not None
        else "n/a",
    )
    table.add_row("failed/errored", f"{summary.counts.failed}/{summary.counts.errored}")
    table.add_row("artifacts", str(artifacts.run_dir))
    console.print(table)


# ---------------------------------------------------------------------------
# compare / gate
# ---------------------------------------------------------------------------


def _comparison_flow(
    baseline_ref: str | None,
    candidate_ref: str | None,
    latest: bool,
    policy_path: str | None,
    config_path: str | None,
    out_dir: str | None,
    results_dir: str | None,
    write: bool,
) -> tuple[int, GateOutcome | None]:
    """Shared implementation of compare/gate. Returns (exit_code, outcome)."""
    try:
        cfg = load_config(config_path) if config_path else None
    except ConfigError as exc:
        _die(str(exc))
        return EXIT_USAGE, None
    store = _store_from_config(cfg, results_dir)
    policy = _load_policy_safe(policy_path, cfg)

    if latest and (baseline_ref or candidate_ref):
        _die("--latest cannot be combined with explicit --baseline/--candidate")
    if latest:
        runs = store.list_runs()
        if len(runs) < 2:
            _die(
                "--latest needs at least two stored runs (found "
                f"{len(runs)} in {store.results_dir})"
            )
        # runs are newest-first: candidate = newest, baseline = second newest
        candidate_ref, baseline_ref = runs[0].run_id, runs[1].run_id
        console.print(f"[dim]latest pair: baseline={baseline_ref} candidate={candidate_ref}[/dim]")

    base_ref = _resolve_ref(store, baseline_ref, fallback_latest=False)
    cand_ref = _resolve_ref(store, candidate_ref, fallback_latest=False)
    try:
        outcome = compare_and_gate(store, base_ref, cand_ref, policy)
    except (FileNotFoundError, ValueError) as exc:
        _die(str(exc))
        return EXIT_USAGE, None

    report = build_report(outcome.comparison, outcome.decision, outcome.baseline, outcome.candidate)
    if write:
        reports_dir = Path(out_dir) if out_dir else store.reports_dir
        paths = write_reports(report, reports_dir)
        report.artifacts = {k: str(v) for k, v in paths.items()}
        # Re-write JSON so the artifact list is self-referential.
        write_reports(report, reports_dir)
    render_console(report)
    return EXIT_OK, outcome


@app.command()
def compare(
    baseline: Annotated[
        str | None, typer.Option("--baseline", help="Baseline run id/path/'latest[:version]'.")
    ] = None,
    candidate: Annotated[
        str | None, typer.Option("--candidate", help="Candidate run id/path/'latest[:version]'.")
    ] = None,
    latest: Annotated[
        bool, typer.Option("--latest", help="Use the two most recent runs (older = baseline).")
    ] = False,
    policy: Annotated[str | None, typer.Option("--policy", help="Regression policy YAML.")] = None,
    config: Annotated[
        str | None, typer.Option("--config", "-c", help="Config for output dirs/policy path.")
    ] = None,
    out: Annotated[str | None, typer.Option("--out", help="Reports output directory.")] = None,
    no_write: Annotated[
        bool, typer.Option("--no-write", help="Print only; do not persist reports.")
    ] = False,
) -> None:
    """Compare two runs, apply the regression policy and write reports."""
    code, _ = _comparison_flow(baseline, candidate, latest, policy, config, out, None, not no_write)
    raise typer.Exit(code)


@app.command()
def gate(
    baseline: Annotated[str | None, typer.Option("--baseline", help="Baseline run ref.")] = None,
    candidate: Annotated[str | None, typer.Option("--candidate", help="Candidate run ref.")] = None,
    latest: Annotated[bool, typer.Option("--latest", help="Use the two most recent runs.")] = False,
    policy: Annotated[str | None, typer.Option("--policy", help="Regression policy YAML.")] = None,
    config: Annotated[
        str | None, typer.Option("--config", "-c", help="Config for output dirs/policy path.")
    ] = None,
    out: Annotated[str | None, typer.Option("--out", help="Reports output directory.")] = None,
    results_dir: Annotated[
        str | None, typer.Option("--results-dir", help="Override results dir.")
    ] = None,
    fail_on_warning: Annotated[
        bool, typer.Option("--fail-on-warning", help="Treat WARNING as failure.")
    ] = False,
    config_baseline: Annotated[
        str | None,
        typer.Option(
            "--config-baseline", help="One-shot mode: evaluate this config as baseline first."
        ),
    ] = None,
    config_candidate: Annotated[
        str | None,
        typer.Option(
            "--config-candidate", help="One-shot mode: evaluate this config as candidate first."
        ),
    ] = None,
    mock: Annotated[
        bool, typer.Option("--mock", help="Force mock providers in one-shot mode.")
    ] = False,
) -> None:
    """CI quality gate. Exit: 0=PASS, 1=FAIL, 2=WARNING (see docs/ci-cd.md)."""
    forced = config_baseline or config_candidate
    if forced:
        if not (config_baseline and config_candidate):
            _die("one-shot mode requires BOTH --config-baseline and --config-candidate")
        if baseline or candidate or latest:
            _die("one-shot mode cannot be combined with run references")
        try:
            base_cfg = load_config(config_baseline, force_mock=True if mock else None)
            cand_cfg = load_config(config_candidate, force_mock=True if mock else None)
            store = _store_from_config(base_cfg, results_dir)
            base_art = run_evaluation(base_cfg, store=store)
            cand_art = run_evaluation(cand_cfg, store=store)
        except (ConfigError, DatasetError, PipelineError, FileNotFoundError, ValueError) as exc:
            _die(str(exc))
            return
        baseline, candidate = base_art.run_id, cand_art.run_id
        # Reuse the baseline config so the comparison reads the same store.
        config = config_baseline

    code, outcome = _comparison_flow(
        baseline, candidate, latest, policy, config, out, results_dir, write=True
    )
    if outcome is None:
        raise typer.Exit(code)
    status = outcome.decision.status
    if status == "FAIL":
        raise typer.Exit(EXIT_FAIL)
    if status == "WARNING" and fail_on_warning:
        raise typer.Exit(EXIT_FAIL)
    if status == "WARNING":
        raise typer.Exit(EXIT_WARNING)
    raise typer.Exit(EXIT_OK)


# ---------------------------------------------------------------------------
# single-run report
# ---------------------------------------------------------------------------


@app.command()
def report(
    run: Annotated[
        str | None, typer.Option("--run", help="Run id/path/'latest[:version]'.")
    ] = None,
    latest: Annotated[bool, typer.Option("--latest", help="Use the most recent run.")] = False,
    config: Annotated[
        str | None, typer.Option("--config", "-c", help="Config for output dirs.")
    ] = None,
    out: Annotated[
        str | None, typer.Option("--out", help="Output directory (default: reports dir).")
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="md | json | both.")] = "md",
) -> None:
    """Render a single-run report (aggregate metrics, category breakdown)."""
    try:
        cfg = load_config(config) if config else None
    except ConfigError as exc:
        _die(str(exc))
        return
    store = _store_from_config(cfg, None)
    ref = "latest" if latest else run
    try:
        run_dir = store.resolve_run(_resolve_ref(store, ref, fallback_latest=True))
        summary: RunSummary = store.load_summary(run_dir)
    except (FileNotFoundError, ValueError) as exc:
        _die(str(exc))
        return

    out_dir = Path(out) if out else Path(store.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"run_{summary.run_id}"
    written: list[Path] = []
    if fmt in {"md", "both"}:
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(render_run_markdown(summary), encoding="utf-8")
        written.append(md_path)
    if fmt in {"json", "both"}:
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(json_path)
    if fmt not in {"md", "json", "both"}:
        _die(f"unknown --format '{fmt}' (expected md|json|both)")
        return
    console.print(render_run_markdown(summary))
    for path in written:
        console.print(f"[dim]written: {path}[/dim]")


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


@app.command()
def calibrate(
    labels: Annotated[
        str, typer.Option("--labels", help="Human labels JSONL.")
    ] = "datasets/human_labels.jsonl",
    config: Annotated[
        str | None, typer.Option("--config", "-c", help="Config supplying the judge.")
    ] = None,
    out: Annotated[
        str | None, typer.Option("--out", help="Output dir (default: reports dir).")
    ] = None,
    mock: Annotated[
        bool, typer.Option("--mock", help="Force mock judge (offline smoke test).")
    ] = False,
) -> None:
    """Measure judge-vs-human agreement and persist a calibration report."""
    try:
        cfg = load_config(config, force_mock=True if mock else None)
        human_labels = load_human_labels(labels)
        provider = build_provider(cfg.judge.provider, seed=cfg.run.seed)
        judge = RubricJudge(cfg.judge, provider)
        result = calibrate_judge(judge, human_labels, judge_config=cfg.judge)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        _die(str(exc))
        return

    out_dir = Path(out) if out else Path(cfg.output.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"calibration_{stamp}.json"
    md_path = out_dir / f"calibration_{stamp}.md"
    json_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_calibration_markdown(result), encoding="utf-8")

    table = Table(title="Judge calibration", expand=False)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("labels", str(result.n_labels))
    table.add_row("judged", str(result.n_judged))
    table.add_row("judge failures", str(result.n_judge_failures))
    for dim, dcal in result.dimensions.items():
        table.add_row(f"MAE {dim}", _opt(dcal.mae))
        table.add_row(f"within+-1 {dim}", _opt(dcal.agreement_within_1))
    table.add_row("pass agreement", _opt(result.pass_agreement_rate))
    table.add_row(
        "cohen kappa",
        f"{_opt(result.cohen_kappa)}"
        + (f" ({result.kappa_interpretation})" if result.kappa_interpretation else ""),
    )
    console.print(table)
    console.print(f"[dim]written: {json_path}[/dim]")
    console.print(f"[dim]written: {md_path}[/dim]")


def _opt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# ---------------------------------------------------------------------------
# dataset tools
# ---------------------------------------------------------------------------


@app.command("validate-dataset")
def validate_dataset(
    path: Annotated[str, typer.Argument(help="Golden dataset JSONL.")] = "datasets/golden.jsonl",
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit non-zero on warnings too.")
    ] = False,
) -> None:
    """Validate dataset schema and health rules."""
    try:
        cases = load_dataset(path)
    except (DatasetError, FileNotFoundError) as exc:
        _die(str(exc))
        return
    report_ = validate_cases(cases)
    console.print(f"cases: {report_.n_cases} | critical: {report_.critical_count}")
    table = Table(title="category coverage", expand=False)
    table.add_column("category")
    table.add_column("n", justify="right")
    for category, count in sorted(report_.category_counts.items()):
        table.add_row(category, str(count))
    console.print(table)
    for issue in report_.issues:
        style = "red" if issue.severity == "error" else "yellow"
        console.print(f"[{style}]{issue}[/{style}]")
    if report_.errors or (strict and report_.warnings):
        raise typer.Exit(EXIT_FAIL)
    console.print("[bold green]dataset OK[/bold green]")


# ---------------------------------------------------------------------------
# runs listing + demo
# ---------------------------------------------------------------------------


@app.command("list-runs")
def list_runs(
    results_dir: Annotated[str | None, typer.Option("--results-dir")] = None,
    config: Annotated[str | None, typer.Option("--config", "-c")] = None,
) -> None:
    """List stored evaluation runs (newest first)."""
    try:
        cfg = load_config(config) if config else None
    except ConfigError as exc:
        _die(str(exc))
        return
    store = _store_from_config(cfg, results_dir)
    runs = store.list_runs()
    if not runs:
        console.print(f"no runs found in {store.results_dir}")
        return
    table = Table(title=f"runs in {store.results_dir}", expand=False)
    for column in ("run_id", "system_version", "overall", "pass_rate", "created"):
        table.add_column(column)
    for run in runs[:50]:
        table.add_row(
            run.run_id,
            run.system_version,
            _opt(run.overall),
            _opt(run.pass_rate),
            run.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )
    console.print(table)


@app.command()
def demo(
    candidate_config: Annotated[
        str,
        typer.Option(
            "--candidate-config",
            help="Candidate config (use configs/candidate_degraded.yaml to see the gate FAIL).",
        ),
    ] = "configs/candidate.yaml",
    baseline_config: Annotated[
        str, typer.Option("--baseline-config", help="Baseline config.")
    ] = "configs/baseline.yaml",
    policy_path: Annotated[
        str, typer.Option("--policy", help="Regression policy YAML.")
    ] = "configs/policy.yaml",
    limit: Annotated[int | None, typer.Option("--limit", help="Evaluate at most N cases.")] = None,
    real: Annotated[
        bool, typer.Option("--real", help="Use configured real providers instead of forcing mock.")
    ] = False,
) -> None:
    """Full offline demo: evaluate baseline + candidate, compare, gate, report."""
    force = None if real else True
    try:
        base_cfg = load_config(baseline_config, force_mock=force)
        cand_cfg = load_config(candidate_config, force_mock=force)
        policy = load_policy(policy_path)
        store = _store_from_config(base_cfg, None)
        console.rule("Ledger demo: baseline evaluation")
        base_art = run_evaluation(base_cfg, limit=limit, store=store)
        console.rule("Ledger demo: candidate evaluation")
        cand_art = run_evaluation(cand_cfg, limit=limit, store=store)
        console.rule("Ledger demo: comparison + gate")
        outcome = compare_and_gate(store, base_art.run_id, cand_art.run_id, policy)
        report_obj = build_report(
            outcome.comparison, outcome.decision, outcome.baseline, outcome.candidate
        )
        paths = write_reports(report_obj, store.reports_dir)
        report_obj.artifacts = {k: str(v) for k, v in paths.items()}
        write_reports(report_obj, store.reports_dir)
        render_console(report_obj)
        console.print(f"[dim]reports: {paths['markdown']} | {paths['json']}[/dim]")
    except (
        ConfigError,
        DatasetError,
        PipelineError,
        PolicyError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        _die(str(exc))
        return
    status = outcome.decision.status
    raise typer.Exit(EXIT_FAIL if status == "FAIL" else EXIT_OK)


def main() -> None:
    """Console-script entry point with stable exit codes."""
    if "--version" in sys.argv[1:]:
        # Handled before Typer's group parsing so `ledger --version` works
        # without a subcommand.
        print(f"ledger {__version__}")
        sys.exit(EXIT_OK)
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":
    main()
