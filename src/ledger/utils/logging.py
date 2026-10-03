"""Logging helpers with a single, predictable setup entry point."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """Configure root logging for Ledger.

    Idempotent: repeated calls are ignored unless ``force=True``. The level is
    resolved from (in order) the explicit argument, ``LEDGER_LOG_LEVEL`` and the
    default ``INFO``.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    resolved = level or os.environ.get("LEDGER_LOG_LEVEL") or "INFO"
    if isinstance(resolved, str):
        resolved = resolved.upper()
    logging.basicConfig(
        level=resolved,
        format=_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=force,
    )
    # Third-party libraries are noisy at DEBUG; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger (configures defaults on first use)."""
    setup_logging()
    return logging.getLogger(name)
