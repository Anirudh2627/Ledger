#!/usr/bin/env python3
"""Render a report for one stored evaluation run.

Usage:
    python scripts/generate_report.py --run <run-id|path|latest> [--format md|json|both]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ledger.config.settings import ConfigError, load_config
from ledger.reporting.markdown import render_run_markdown
from ledger.tracking.artifacts import ArtifactStore
from ledger.utils.env import load_dotenv
from ledger.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest", help="Run id, path, or 'latest'")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="Output dir (default: reports dir)")
    parser.add_argument("--format", default="md", choices=["md", "json", "both"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv()
    setup_logging("DEBUG" if args.verbose else None)
    try:
        cfg = load_config(args.config) if args.config else None
        store = ArtifactStore(
            results_dir=cfg.output.results_dir if cfg else "results",
            reports_dir=cfg.output.reports_dir if cfg else "reports",
            experiments_dir=cfg.output.experiments_dir if cfg else "experiments",
        )
        ref = args.run
        if ref == "latest":
            latest = store.latest_run()
            if latest is None:
                print("error: no stored runs found", file=sys.stderr)
                return 1
            ref = latest.run_id
        summary = store.load_summary(store.resolve_run(ref))
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else Path(store.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_run_markdown(summary)
    if args.format in {"md", "both"}:
        path = out_dir / f"run_{summary.run_id}.md"
        path.write_text(markdown, encoding="utf-8")
        print(f"written: {path}")
    if args.format in {"json", "both"}:
        path = out_dir / f"run_{summary.run_id}.json"
        path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"written: {path}")
    if args.format == "md":
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
