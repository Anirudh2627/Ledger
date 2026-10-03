"""Run artifact storage, experiment index and run resolution.

Layout (all paths configurable):

    results/<run_id>/
        results.jsonl        # one EvaluationResult per line
        summary.json         # RunSummary (aggregate metrics)
        config.snapshot.json # effective config, secrets redacted
    experiments/index.jsonl  # append-only registry of runs (ML-flow-lite)
    reports/                 # comparison & single-run reports
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ledger.core.models import EvaluationResult, RunSummary
from ledger.utils.env import redact_secrets
from ledger.utils.logging import get_logger

logger = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_len: int = 40) -> str:
    normalized = unicodedata.normalize("NFKD", text).lower()
    slug = _SLUG_RE.sub("-", normalized).strip("-")
    return slug[:max_len] or "run"


@dataclass
class RunInfo:
    run_id: str
    system_version: str
    created_at: datetime
    path: Path
    overall: float | None
    pass_rate: float | None


class ArtifactStore:
    """Filesystem-backed store for evaluation runs, experiments and reports."""

    def __init__(
        self,
        results_dir: str | Path = "results",
        reports_dir: str | Path = "reports",
        experiments_dir: str | Path = "experiments",
    ) -> None:
        self.results_dir = Path(results_dir)
        self.reports_dir = Path(reports_dir)
        self.experiments_dir = Path(experiments_dir)

    # -- run identity -----------------------------------------------------------

    def new_run_id(self, system_version: str) -> str:
        """Timestamped, collision-resistant, human-sortable run id."""
        import uuid

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        nonce = uuid.uuid4().hex[:6]
        return f"{stamp}-{_slug(system_version)}-{nonce}"

    def run_dir(self, run_id: str) -> Path:
        return self.results_dir / run_id

    def ensure_run_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- writing -----------------------------------------------------------

    def save_results(self, run_id: str, results: list[EvaluationResult]) -> Path:
        directory = self.ensure_run_dir(run_id)
        path = directory / "results.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for result in results:
                fh.write(result.model_dump_json() + "\n")
        logger.debug("wrote %d results to %s", len(results), path)
        return path

    def save_summary(self, run_id: str, summary: RunSummary) -> Path:
        directory = self.ensure_run_dir(run_id)
        path = directory / "summary.json"
        path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def save_config_snapshot(self, run_id: str, config_dump: dict[str, Any]) -> Path:
        directory = self.ensure_run_dir(run_id)
        path = directory / "config.snapshot.json"
        sanitized = redact_secrets(config_dump)
        path.write_text(json.dumps(sanitized, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    def record_experiment(self, entry: dict[str, Any]) -> Path:
        """Append one row to the append-only experiment index."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        path = self.experiments_dir / "index.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return path

    # -- reading -----------------------------------------------------------

    def load_results(self, run_ref: str | Path) -> list[EvaluationResult]:
        directory = self.resolve_run(run_ref)
        path = directory / "results.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"no results.jsonl in run directory {directory}")
        results: list[EvaluationResult] = []
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(EvaluationResult.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: corrupt result row: {exc}") from exc
        return results

    def load_summary(self, run_ref: str | Path) -> RunSummary:
        directory = self.resolve_run(run_ref)
        path = directory / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"no summary.json in run directory {directory}")
        return RunSummary.model_validate_json(path.read_text(encoding="utf-8"))

    def resolve_run(self, run_ref: str | Path) -> Path:
        """Resolve a run reference: a path, a run id, or a unique id prefix."""
        candidate = Path(run_ref)
        if candidate.is_dir() and (candidate / "summary.json").is_file():
            return candidate

        ref = str(run_ref)
        exact = self.results_dir / ref
        if exact.is_dir():
            return exact

        if not self.results_dir.is_dir():
            raise FileNotFoundError(f"results directory not found: {self.results_dir}")
        matches = sorted(
            (p for p in self.results_dir.iterdir() if p.is_dir() and p.name.startswith(ref)),
            key=lambda p: p.name,
            reverse=True,  # newest first (ids are timestamp-prefixed)
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(m.name for m in matches[:5])
            raise ValueError(f"run reference '{ref}' is ambiguous: {names}")
        raise FileNotFoundError(f"run not found: {run_ref}")

    def list_runs(self) -> list[RunInfo]:
        """All recorded runs, newest first."""
        if not self.results_dir.is_dir():
            return []
        runs: list[RunInfo] = []
        for directory in self.results_dir.iterdir():
            summary_path = directory / "summary.json"
            if not directory.is_dir() or not summary_path.is_file():
                continue
            try:
                summary = RunSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
            except ValueError:
                logger.warning("skipping run with unreadable summary: %s", directory)
                continue
            runs.append(
                RunInfo(
                    run_id=summary.run_id,
                    system_version=summary.system_version,
                    created_at=summary.created_at,
                    path=directory,
                    overall=summary.metrics.get("overall"),
                    pass_rate=summary.metrics.get("pass_rate"),
                )
            )
        runs.sort(key=lambda r: r.run_id, reverse=True)
        return runs

    def latest_run(self, system_version: str | None = None) -> RunInfo | None:
        for run in self.list_runs():
            if system_version is None or run.system_version == system_version:
                return run
        return None
