"""Statistical regression engine: bootstrap, comparator, policy, detector."""

from ledger.regression.bootstrap import (
    BootstrapResult,
    InsufficientDataError,
    SignTestResult,
    paired_bootstrap,
    sign_test_on_flips,
)
from ledger.regression.comparator import (
    CaseDelta,
    ComparisonResult,
    MetricComparison,
    RunData,
    compare_runs,
)
from ledger.regression.detector import GateDecision, Violation, detect_regressions
from ledger.regression.policy import (
    PolicyError,
    RegressionPolicy,
    load_policy,
)

__all__ = [
    "BootstrapResult",
    "CaseDelta",
    "ComparisonResult",
    "GateDecision",
    "InsufficientDataError",
    "MetricComparison",
    "PolicyError",
    "RegressionPolicy",
    "RunData",
    "SignTestResult",
    "Violation",
    "compare_runs",
    "detect_regressions",
    "load_policy",
    "paired_bootstrap",
    "sign_test_on_flips",
]
