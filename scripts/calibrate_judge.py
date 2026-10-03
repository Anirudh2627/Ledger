#!/usr/bin/env python3
"""Run judge calibration against human labels.

Usage:
    python scripts/calibrate_judge.py [--labels datasets/human_labels.jsonl] \
        [--config configs/baseline.yaml] [--mock]
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from ledger.config.settings import ConfigError, load_config
from ledger.judges.calibration import (
    calibrate_judge,
    load_human_labels,
    render_calibration_markdown,
)
from ledger.judges.rubric import RubricJudge
from ledger.providers.registry import build_provider
from ledger.utils.env import load_dotenv
from ledger.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="datasets/human_labels.jsonl")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out", default=None, help="Output dir (default: reports dir)")
    parser.add_argument("--mock", action="store_true", help="Force the mock judge")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv()
    setup_logging("DEBUG" if args.verbose else None)
    try:
        cfg = load_config(args.config, force_mock=True if args.mock else None)
        labels = load_human_labels(args.labels)
        provider = build_provider(cfg.judge.provider, seed=cfg.run.seed)
        judge = RubricJudge(cfg.judge, provider)
        result = calibrate_judge(judge, labels, judge_config=cfg.judge)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else Path(cfg.output.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"calibration_{stamp}.json"
    md_path = out_dir / f"calibration_{stamp}.md"
    json_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_calibration_markdown(result), encoding="utf-8")

    print(
        f"labels: {result.n_labels} | judged: {result.n_judged} | "
        f"failures: {result.n_judge_failures}"
    )
    print(
        f"pass agreement: {result.pass_agreement_rate} | "
        f"cohen kappa: {result.cohen_kappa} ({result.kappa_interpretation})"
    )
    for dim, dcal in sorted(result.dimensions.items()):
        print(f"  {dim}: MAE={dcal.mae} within+-1={dcal.agreement_within_1} bias={dcal.bias}")
    print(f"written: {json_path}")
    print(f"written: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
