#!/usr/bin/env python3
"""Compare two evaluation runs and apply the regression policy.

Usage:
    python scripts/compare.py --baseline <run-id> --candidate <run-id> \
        [--policy configs/policy.yaml]

Exit codes match the CLI gate contract: 0=PASS, 1=FAIL, 2=WARNING.
"""

from __future__ import annotations

import argparse
import sys

from ledger.config.settings import ConfigError, load_config
from ledger.evaluation.compare import compare_and_gate
from ledger.regression.policy import PolicyError, load_policy
from ledger.reporting.console import render_console
from ledger.reporting.json_report import write_reports
from ledger.reporting.report import build_report
from ledger.tracking.artifacts import ArtifactStore
from ledger.utils.env import load_dotenv
from ledger.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline run id / path")
    parser.add_argument("--candidate", required=True, help="Candidate run id / path")
    parser.add_argument("--policy", default="configs/policy.yaml", help="Policy YAML")
    parser.add_argument("--config", default=None, help="Config for output dirs")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv()
    setup_logging("DEBUG" if args.verbose else None)
    try:
        cfg = load_config(args.config) if args.config else None
        policy = load_policy(args.policy, inline=cfg.policy if cfg else None)
        store = ArtifactStore(
            results_dir=cfg.output.results_dir if cfg else "results",
            reports_dir=cfg.output.reports_dir if cfg else "reports",
            experiments_dir=cfg.output.experiments_dir if cfg else "experiments",
        )
        outcome = compare_and_gate(store, args.baseline, args.candidate, policy)
        report = build_report(
            outcome.comparison, outcome.decision, outcome.baseline, outcome.candidate
        )
        paths = write_reports(report, store.reports_dir)
        report.artifacts = {k: str(v) for k, v in paths.items()}
        write_reports(report, store.reports_dir)
    except (ConfigError, PolicyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    render_console(report)
    status = outcome.decision.status
    if status == "FAIL":
        return 1
    if status == "WARNING":
        return 1 if args.fail_on_warning else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
