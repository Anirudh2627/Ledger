"""System-under-test factory: build a SUT from configuration."""

from __future__ import annotations

from ledger.config.settings import LedgerConfig, MockSystemConfig, RagSystemConfig
from ledger.core.interfaces import LLMProvider, SystemUnderTest
from ledger.providers.registry import build_provider
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


def build_system(config: LedgerConfig, provider: LLMProvider | None = None) -> SystemUnderTest:
    """Instantiate the configured system under test.

    The provider is injectable so tests can hand in fakes; by default it is
    built from ``config.provider``.
    """
    provider = provider or build_provider(config.provider, seed=config.run.seed)
    system_cfg = config.system

    if isinstance(system_cfg, RagSystemConfig):
        from ledger.systems.rag.system import RAGSystem

        return RAGSystem(
            system_cfg,
            provider,
            version=config.run.system_version,
            seed=config.run.seed,
        )
    if isinstance(system_cfg, MockSystemConfig):
        from ledger.systems.mock import MockSystem

        return MockSystem(system_cfg, version=config.run.system_version)

    raise TypeError(f"unsupported system config: {type(system_cfg).__name__}")  # pragma: no cover


__all__ = ["build_provider", "build_system"]
