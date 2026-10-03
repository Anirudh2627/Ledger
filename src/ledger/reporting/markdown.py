"""Markdown report renderer (human-readable, CI-step-summary friendly)."""

from __future__ import annotations

from ledger.core.models import RunSummary
from ledger.regression.comparator import MetricComparison
from ledger.reporting.report import Report

_STATUS_EMOJI = {"PASS": ":white_check_mark:", "WARNING": ":warning:", "FAIL": ":x:"}


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_signed(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def _metric_row(comp: MetricComparison) -> str:
    ci = (
        f"[{comp.ci_low:+.4f}, {comp.ci_high:+.4f}]"
        if comp.ci_low is not None and comp.ci_high is not None
        else "n/a"
    )
    flag = ""
    if comp.significant_regression:
        flag = " :red_circle:"
    elif comp.significant_improvement:
        flag = " :green_circle:"
    return (
        f"| `{comp.metric}` | {comp.n_paired} | {_fmt(comp.baseline_mean)} | "
        f"{_fmt(comp.candidate_mean)} | {_fmt_signed(comp.delta)} | "
        f"{_fmt_pct(comp.relative_delta)} | {ci} | {_fmt(comp.p_value, 3)} |{flag}"
    )


def _overall_row(overall: MetricComparison) -> str:
    base = (
        f"| **overall** | {_fmt(overall.baseline_mean)} | {_fmt(overall.candidate_mean)} | "
        f"**{_fmt_signed(overall.delta)}** | {_fmt_pct(overall.relative_delta)} | "
    )
    if overall.ci_low is not None and overall.ci_high is not None:
        return (
            base + f"[{overall.ci_low:+.4f}, {overall.ci_high:+.4f}] | {_fmt(overall.p_value, 3)} |"
        )
    return base + "n/a | n/a |"


def render_markdown(report: Report) -> str:
    """Render the full Markdown report."""
    decision = report.decision
    emoji = _STATUS_EMOJI.get(decision.status, "")
    overall = report.comparison.overall
    method = report.comparison.method

    lines: list[str] = [
        "# Ledger Evaluation Report",
        "",
        f"**Decision: {emoji} {decision.status}**",
        "",
        f"- Generated: {report.generated_at.isoformat(timespec='seconds')} "
        f"(ledger {report.ledger_version}, report schema {report.schema_version})",
        f"- Baseline: `{report.baseline.system_version}` "
        f"(run `{report.baseline.run_id}`, model `{report.baseline.model or '?'}`, "
        f"prompt `{report.baseline.prompt_version or '?'}`)",
        f"- Candidate: `{report.candidate.system_version}` "
        f"(run `{report.candidate.run_id}`, model `{report.candidate.model or '?'}`, "
        f"prompt `{report.candidate.prompt_version or '?'}`)",
        f"- Paired cases: {report.comparison.n_common} "
        f"(baseline {report.comparison.n_baseline}, candidate {report.comparison.n_candidate})",
        f"- Dataset fingerprint match: {report.comparison.dataset_fingerprint_match}",
        "",
        "## Overall",
        "",
    ]

    if overall:
        lines += [
            "| | baseline | candidate | delta | relative | 95% CI | p |",
            "|---|---|---|---|---|---|---|",
            _overall_row(overall),
            "",
        ]
    else:
        lines += ["No paired overall scores available.", ""]

    lines += [
        "## Metrics",
        "",
        "| metric | n | baseline | candidate | delta | rel. | CI | p | sig. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(report.comparison.metrics):
        lines.append(_metric_row(report.comparison.metrics[name]))

    lines += [
        "",
        "## Categories (overall score)",
        "",
        "| category | n | baseline | candidate | delta | rel. | CI | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(report.comparison.categories):
        comp = report.comparison.categories[name]
        display = "**critical cases**" if name == "__critical__" else f"`{name}`"
        ci = (
            f"[{comp.ci_low:+.4f}, {comp.ci_high:+.4f}]"
            if comp.ci_low is not None and comp.ci_high is not None
            else "n/a"
        )
        lines.append(
            f"| {display} | {comp.n_paired} | {_fmt(comp.baseline_mean)} | "
            f"{_fmt(comp.candidate_mean)} | {_fmt_signed(comp.delta)} | "
            f"{_fmt_pct(comp.relative_delta)} | {ci} | {_fmt(comp.p_value, 3)} |"
        )

    counts = report.comparison
    n_regressed = sum(1 for c in counts.cases if c.status == "regressed")
    n_improved = sum(1 for c in counts.cases if c.status == "improved")
    n_stable_pass = sum(1 for c in counts.cases if c.status == "stable_pass")
    n_stable_fail = sum(1 for c in counts.cases if c.status == "stable_fail")
    lines += [
        "",
        "## Test movement",
        "",
        f"- Regressed: **{n_regressed}**  |  Improved: **{n_improved}**  |  "
        f"Stable pass: {n_stable_pass}  |  Stable fail: {n_stable_fail}",
        f"- Newly failing tests: {len(counts.newly_failing)}"
        + (f" ({', '.join(counts.newly_failing[:10])})" if counts.newly_failing else ""),
        f"- Newly passing tests: {len(counts.newly_passing)}"
        + (f" ({', '.join(counts.newly_passing[:10])})" if counts.newly_passing else ""),
        f"- Critical regressions: **{len(counts.critical_regressions)}**"
        + (
            f" ({', '.join(counts.critical_regressions[:10])})"
            if counts.critical_regressions
            else ""
        ),
        f"- Pass/fail flip sign test: p="
        f"{_fmt(counts.sign_test_p_value, 3)} "
        f"(flips: {counts.sign_test_flips.get('n_flips', 0)})",
        "",
        "## Gate decision",
        "",
        f"**{decision.status}** - {len(decision.fails)} fail rule(s), "
        f"{len(decision.warnings)} warning(s).",
        "",
    ]
    if decision.violations:
        lines += ["| severity | rule | detail |", "|---|---|---|"]
        for violation in decision.violations:
            lines.append(
                f"| {violation.severity.upper()} | `{violation.rule}` | {violation.message} |"
            )
        lines.append("")

    if report.top_regressions:
        lines += ["## Top regressed cases", ""]
        for excerpt in report.top_regressions:
            crit = " **[CRITICAL]**" if excerpt.critical else ""
            lines += [
                f"### `{excerpt.test_id}`{crit} - {excerpt.category} "
                f"({_fmt_signed(excerpt.delta)})",
                "",
                f"- Question: {excerpt.question}",
                f"- Score: {_fmt(excerpt.baseline_overall)} -> "
                f"{_fmt(excerpt.candidate_overall)} | passed: "
                f"{excerpt.baseline_passed} -> {excerpt.candidate_passed}",
                f"- Baseline answer: {excerpt.baseline_answer}",
                f"- Candidate answer: {excerpt.candidate_answer}",
            ]
            if excerpt.candidate_judge_reason:
                lines.append(f"- Judge (candidate): {excerpt.candidate_judge_reason}")
            lines.append("")

    lines += [
        "## Method",
        "",
        f"- Comparison: paired per-case deltas over {report.comparison.n_common} common test ids.",
        f"- Statistics: {method.get('bootstrap', 'paired_percentile')} bootstrap, "
        f"{method.get('n_samples')} resamples, "
        f"confidence {method.get('confidence')}, seed {method.get('seed')}.",
        "- Judge overall/dimension scores normalized from 1-5 to 0-1; "
        "overall recomputed by Ledger from configured weights (judge-reported "
        "aggregates are diagnostics only).",
        "",
        "## Notes",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    if report.artifacts:
        lines += ["", "## Artifacts", ""]
        lines += [f"- `{name}`: `{path}`" for name, path in sorted(report.artifacts.items())]
    lines += ["", "---", "*Generated by Ledger.*", ""]
    return "\n".join(lines)


def render_run_markdown(summary: RunSummary) -> str:
    """Render a single-run report (no comparison)."""
    lines = [
        f"# Ledger Run Report - `{summary.system_version}`",
        "",
        f"- Run id: `{summary.run_id}`",
        f"- Created: {summary.created_at.isoformat(timespec='seconds')}",
        f"- Dataset: `{summary.dataset.path}` ({summary.dataset.n_cases} cases, "
        f"sha256 `{summary.dataset.fingerprint[:12]}...`)",
        f"- Provider: `{summary.provider.name}` model `{summary.provider.model}` "
        f"(mock={summary.provider.is_mock})",
        "- Judge: "
        + (
            f"`{summary.judge.name}` model `{summary.judge.model}`" if summary.judge else "disabled"
        ),
        f"- Prompt version: `{summary.prompt_version or '-'}`",
        "",
        "## Headline metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for name in sorted(summary.metrics):
        lines.append(f"| `{name}` | {summary.metrics[name]:.4f} |")
    counts = summary.counts
    lines += [
        "",
        "## Counts",
        "",
        f"- total: {counts.total} | passed: {counts.passed} | failed: {counts.failed} | "
        f"errored: {counts.errored} | judge failures: {counts.judge_failed}",
    ]
    latency = summary.latency
    if latency.mean_ms is not None:
        lines += [
            "",
            "## Latency (SUT generation)",
            "",
            f"- mean: {latency.mean_ms:.1f} ms | p50: {latency.p50_ms:.1f} ms | "
            f"p95: {latency.p95_ms:.1f} ms",
        ]
    if summary.categories:
        lines += [
            "",
            "## Categories",
            "",
            "| category | n | overall | pass rate |",
            "|---|---|---|---|",
        ]
        for name, agg in sorted(summary.categories.items()):
            lines.append(
                f"| `{name}` | {agg.n} | {_fmt(agg.overall_mean)} | {_fmt(agg.pass_rate)} |"
            )
    system_info = summary.runtime.get("system")
    if system_info:
        lines += ["", "## System under test", ""]
        lines += [f"- `{k}`: `{v}`" for k, v in sorted(system_info.items())]
    lines += ["", "---", "*Generated by Ledger.*", ""]
    return "\n".join(lines)
