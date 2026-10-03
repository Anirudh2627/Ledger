"""Loading of golden datasets from JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from ledger.dataset.schema import TestCase
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetError(RuntimeError):
    """Raised when a dataset file cannot be loaded."""


def load_dataset(
    path: str | Path,
    *,
    limit: int | None = None,
    categories: Iterable[str] | None = None,
) -> list[TestCase]:
    """Load test cases from a JSONL file.

    Blank lines and ``#`` comment lines are ignored. Duplicate ids, malformed
    JSON and schema violations raise :class:`DatasetError` with the offending
    line number - a golden dataset must be strictly valid.

    Args:
        path: JSONL dataset file.
        limit: keep at most this many cases (after category filtering).
        categories: optional allow-list of categories.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetError(f"dataset file not found: {file_path}")

    category_filter = set(categories) if categories else None
    cases: list[TestCase] = []
    seen_ids: dict[str, int] = {}

    with file_path.open(encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{file_path}:{line_no}: invalid JSON: {exc}") from exc
            try:
                case = TestCase.model_validate(payload)
            except ValidationError as exc:
                raise DatasetError(f"{file_path}:{line_no}: schema violation: {exc}") from exc
            if case.id in seen_ids:
                raise DatasetError(
                    f"{file_path}:{line_no}: duplicate case id '{case.id}' "
                    f"(first seen on line {seen_ids[case.id]})"
                )
            seen_ids[case.id] = line_no
            if category_filter and case.category not in category_filter:
                continue
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break

    if not cases:
        raise DatasetError(f"dataset {file_path} produced zero cases (check filters)")
    logger.info("loaded %d test cases from %s", len(cases), file_path)
    return cases
