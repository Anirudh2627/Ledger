#!/usr/bin/env python3
"""Run a full evaluation for one system configuration.

Thin wrapper around :func:`ledger.evaluation.pipeline.run_evaluation` for
environments that prefer plain scripts over the CLI (e.g. notebooks, cron).

Usage:
    python scripts/evaluate.py --config configs/baseline.yaml [--limit 10] [--mock]
"""

from __future__ import annotations

import argparse
import sys

from ledger.config.settings import ConfigError, load_config
from ledger.evaluation.pipeline import PipelineError, run_evaluation
from ledger.utils.env import load_dotenv
from ledger.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML config path")
    parser.add_argument("--dataset", default=None, help="Override dataset path")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N cases")
    parser.add_argument("--run-id", default=None, help="Force a specific run id")
    parser.add_argument("--mock", action="store_true", help="Force deterministic mock providers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv()
    setup_logging("DEBUG" if args.verbose else None)
    try:
        config = load_config(args.config, force_mock=True if args.mock else None)
        artifacts = run_evaluation(
            config, dataset_path=args.dataset, limit=args.limit, run_id=args.run_id
        )
    except (ConfigError, PipelineError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"run_id: {artifacts.run_id}")
    print(f"overall: {artifacts.summary.metrics.get('overall')}")
    print(f"pass_rate: {artifacts.summary.metrics.get('pass_rate')}")
    print(f"artifacts: {artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
