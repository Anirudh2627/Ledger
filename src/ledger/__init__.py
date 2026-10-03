"""Ledger - regression testing and evaluation infrastructure for LLM applications.

Ledger evaluates a *System Under Test* (SUT) against a versioned golden dataset,
scores outputs with deterministic metrics and a calibrated LLM-as-a-judge, and
detects regressions between two system versions using paired bootstrap
statistics and a configurable policy. The final verdict (PASS / WARNING / FAIL)
is designed to be consumed by CI as a quality gate.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
