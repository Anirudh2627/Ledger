"""Robust JSON extraction from LLM output.

LLMs frequently wrap JSON in markdown fences, add prose before/after the
payload, or emit trailing commas. The helpers here recover a JSON object from
such responses deterministically, without external dependencies.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_LINE_COMMENT_RE = re.compile(r"^\s*//.*$", re.MULTILINE)


class JSONExtractionError(ValueError):
    """Raised when no valid JSON object can be recovered from the text."""


def _clean(candidate: str) -> str:
    candidate = _LINE_COMMENT_RE.sub("", candidate)
    candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    return candidate.strip()


def _balanced_object(text: str) -> str | None:
    """Return the first brace-balanced ``{...}`` substring, ignoring braces in strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(text: str) -> dict:
    """Best-effort extraction of a JSON object from raw LLM output.

    Strategy (in order):
      1. direct ``json.loads``
      2. content of a markdown code fence
      3. first brace-balanced substring
      4. the above after stripping trailing commas / line comments

    Raises:
        JSONExtractionError: if no JSON object can be recovered.
    """
    if not text or not text.strip():
        raise JSONExtractionError("empty response")

    candidates: list[str] = [text.strip()]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    balanced = _balanced_object(text)
    if balanced:
        candidates.append(balanced)
    candidates.extend(_clean(c) for c in list(candidates))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
        raise JSONExtractionError(f"expected a JSON object, got {type(parsed).__name__}")
    raise JSONExtractionError("no valid JSON object found in response")
