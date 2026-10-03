"""Best-effort runtime provenance helpers (git commit, versions)."""

from __future__ import annotations

import platform
import subprocess
import sys

from ledger import __version__


def git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def git_dirty() -> bool | None:
    """Return whether the working tree is dirty, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def runtime_fingerprint() -> dict[str, str | bool | None]:
    """Collect environment metadata recorded with every evaluation run."""
    return {
        "ledger_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
