"""Minimal .env loading and secret hygiene helpers.

We intentionally avoid a python-dotenv dependency: Ledger needs only the
``KEY=VALUE`` subset, and secrets must never be written into artifacts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ledger.utils.logging import get_logger

logger = get_logger(__name__)

_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")
_SECRET_KEY_RE = re.compile(r"(api[_-]?key|secret|token|password|authorization)", re.IGNORECASE)


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` pairs from a dotenv file into ``os.environ``.

    - Missing file is not an error (returns ``{}``).
    - Existing environment variables win unless ``override=True``.
    - Supports ``#`` comments, blank lines, optional ``export`` prefix and
      single/double quoted values.
    """
    path = Path(path)
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    if loaded:
        logger.debug("loaded %d variables from %s", len(loaded), path)
    return loaded


def looks_like_secret_key(key: str) -> bool:
    """Heuristic: does this config/env key name suggest it holds a secret?"""
    return bool(_SECRET_KEY_RE.search(key))


def redact_secrets(obj: object) -> object:
    """Recursively replace secret-looking values with ``***redacted***``.

    Used before persisting config snapshots so artifacts can never leak keys.
    """
    if isinstance(obj, dict):
        return {
            k: (
                "***redacted***"
                if (looks_like_secret_key(str(k)) and obj[k])
                else redact_secrets(v)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact_secrets(v) for v in obj]
    return obj


def resolve_api_key(env_var_name: str | None) -> str | None:
    """Resolve an API key from the process environment by variable name.

    Returns ``None`` when the name is unset or empty. Never logs the value.
    """
    if not env_var_name:
        return None
    value = os.environ.get(env_var_name, "").strip()
    return value or None
