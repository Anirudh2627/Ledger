"""JSON report rendering and report file writing."""

from __future__ import annotations

from pathlib import Path

from ledger.reporting.report import Report
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


def render_json(report: Report, *, indent: int = 2) -> str:
    """Serialize the full report (machine-readable, schema-versioned)."""
    return report.model_dump_json(indent=indent)


def write_reports(
    report: Report,
    out_dir: str | Path,
    *,
    basename: str | None = None,
    markdown: str | None = None,
) -> dict[str, Path]:
    """Write ``<basename>.json`` and ``<basename>.md`` into ``out_dir``.

    Returns a mapping ``{"json": path, "markdown": path}``.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = basename or (
        f"comparison_{report.baseline.system_version}_vs_"
        f"{report.candidate.system_version}_{report.generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    json_path = directory / f"{stem}.json"
    md_path = directory / f"{stem}.md"
    json_path.write_text(render_json(report) + "\n", encoding="utf-8")
    if markdown is None:
        from ledger.reporting.markdown import render_markdown

        markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    logger.info("reports written: %s, %s", json_path, md_path)
    return {"json": json_path, "markdown": md_path}
