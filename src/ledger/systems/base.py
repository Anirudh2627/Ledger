"""Base class and registration for systems under test."""

from __future__ import annotations

from collections.abc import Callable

from ledger.core.interfaces import SystemUnderTest

__all__ = ["SystemUnderTest", "register_system"]

_REGISTRY: dict[str, type[SystemUnderTest]] = {}


def register_system(
    kind: str,
) -> Callable[[type[SystemUnderTest]], type[SystemUnderTest]]:
    """Class decorator registering a SUT implementation under a config ``kind``."""

    def decorator(cls: type[SystemUnderTest]) -> type[SystemUnderTest]:
        _REGISTRY[kind] = cls
        return cls

    return decorator


def get_system_class(kind: str) -> type[SystemUnderTest]:
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise KeyError(f"unknown system kind '{kind}'; registered: {sorted(_REGISTRY)}") from None


def registered_kinds() -> list[str]:
    return sorted(_REGISTRY)
