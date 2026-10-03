"""Console summary renderer (rich), matching the classic Ledger layout."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ledger.reporting.report import Report

_STATUS_STYLE = {"PASS": "bold green", "WARNING": "bold yellow", "FAIL": "bold red"}


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def render_console(report: Report, console: Console | None = None) -> None:
    """Print the human-facing evaluation summary."""
    console = console or Console()
    decision = report.decision
    overall = report.comparison.overall
    style = _STATUS_STYLE.get(decision.status, "bold")

    header = Text()
    header.append("LEDGER EVALUATION REPORT\n\n", style="bold cyan")
    header.append(f"Baseline:     {report.baseline.system_version}\n")
    header.append(f"Candidate:    {report.candidate.system_version}\n")
    header.append(f"Paired cases: {report.comparison.n_common}\n")
    console.print(Panel(header, title="comparison", expand=False))

    if overall:
        console.print(
            f"Overall:\n  {_fmt(overall.baseline_mean)} -> {_fmt(overall.candidate_mean)}   "
            f"delta: {_fmt(overall.delta)} ({_fmt_pct(overall.relative_delta)})   "
            f"CI: [{_fmt(overall.ci_low)}, {_fmt(overall.ci_high)}]   p={_fmt(overall.p_value)}"
        )

    if report.comparison.categories:
        table = Table(title="Categories (overall score)", expand=False)
        table.add_column("category")
        table.add_column("baseline", justify="right")
        table.add_column("candidate", justify="right")
        table.add_column("delta", justify="right")
        table.add_column("rel.", justify="right")
        for name in sorted(report.comparison.categories):
            comp = report.comparison.categories[name]
            display = "critical (all)" if name == "__critical__" else name
            delta_style = "red" if comp.delta < -1e-9 else ("green" if comp.delta > 1e-9 else "")
            table.add_row(
                display,
                _fmt(comp.baseline_mean),
                _fmt(comp.candidate_mean),
                Text(_fmt(comp.delta), style=delta_style),
                _fmt_pct(comp.relative_delta),
            )
        console.print(table)

    counts = report.comparison
    n_regressed = sum(1 for c in counts.cases if c.status == "regressed")
    n_improved = sum(1 for c in counts.cases if c.status == "improved")
    cand = report.candidate
    n_passed = cand.n_cases - cand.failed - cand.errored
    console.print(
        "Tests:\n"
        f"  Passed (candidate):   {n_passed}\n"
        f"  Failed (candidate):   {cand.failed} (+{cand.errored} errors)\n"
        f"  Regressed:            {n_regressed}\n"
        f"  Improved:             {n_improved}\n"
        f"  Newly failing:        {len(counts.newly_failing)}\n"
        f"  Critical regressions: {len(counts.critical_regressions)}"
    )

    if decision.violations:
        violation_table = Table(title="Policy violations", expand=False)
        violation_table.add_column("severity")
        violation_table.add_column("rule")
        violation_table.add_column("detail", overflow="fold")
        for violation in decision.violations[:12]:
            violation_table.add_row(
                Text(
                    violation.severity.upper(),
                    style="red" if violation.severity == "fail" else "yellow",
                ),
                violation.rule,
                violation.message,
            )
        console.print(violation_table)

    console.print()
    console.print(Text(f"Decision:\n  {decision.status}", style=style))
