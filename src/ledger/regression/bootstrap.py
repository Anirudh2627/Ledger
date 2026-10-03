"""Statistical primitives for regression detection.

Method: **paired percentile bootstrap**.

Baseline and candidate are evaluated on the *same* dataset, so per-case
scores form natural pairs. Resampling case indices (with replacement) and
recomputing the mean difference for each resample yields a distribution of
the delta under the empirical data-generating process; the percentile
interval of that distribution is a confidence interval for the true mean
delta, and the fraction of resamples on the other side of zero gives a
bootstrap p-value.

Also provided: an exact two-sided **sign test** on pass/fail flips, which
makes no distributional assumptions at all.

References for the approach (standard practice; no novel claims):
  * Efron & Tibshirani, *An Introduction to the Bootstrap* (1993), ch. 12-14.
  * Percentile method with paired resampling of unit-level differences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Deltas with magnitude below this are numerically zero.
EPS = 1e-12


class InsufficientDataError(ValueError):
    """Raised when there is not enough paired data to run statistics."""


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a paired bootstrap comparison for one metric."""

    n: int
    point_delta: float  # mean(candidate) - mean(baseline)
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    n_samples: int
    seed: int
    confidence: float
    method: str = "paired_percentile_bootstrap"

    @property
    def excludes_zero(self) -> bool:
        """True when the CI does not contain 0 (directionally conclusive)."""
        if self.ci_low is None or self.ci_high is None:
            return False
        return self.ci_high < 0.0 or self.ci_low > 0.0

    @property
    def significant_regression(self) -> bool:
        return self.excludes_zero and self.point_delta < 0 and (self.ci_high or 0) < 0

    @property
    def significant_improvement(self) -> bool:
        return self.excludes_zero and self.point_delta > 0 and (self.ci_low or 0) > 0


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 1337,
) -> BootstrapResult:
    """Paired percentile bootstrap of the mean difference.

    Args:
        baseline: per-case baseline scores.
        candidate: per-case candidate scores (same order/length: pairs).
        n_samples: bootstrap resamples (>= 200 recommended for stable CIs).
        confidence: CI level in (0, 1), e.g. 0.95.
        seed: RNG seed - results are exactly reproducible.

    Raises:
        InsufficientDataError: when inputs are empty, misaligned or invalid.
    """
    if len(baseline) != len(candidate):
        raise InsufficientDataError(
            f"paired bootstrap needs equal-length inputs, got {len(baseline)} vs {len(candidate)}"
        )
    n = len(baseline)
    if n == 0:
        raise InsufficientDataError("paired bootstrap needs at least one pair")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(cand)):
        raise InsufficientDataError("scores contain non-finite values")

    diffs = cand - base
    point_delta = float(diffs.mean())

    rng = np.random.default_rng(seed)
    # One (n_samples, n) index matrix; same indices for both arms => paired.
    idx = rng.integers(0, n, size=(n_samples, n))
    boot_deltas = diffs[idx].mean(axis=1)

    alpha = 1.0 - confidence
    ci_low, ci_high = (
        float(v) for v in np.percentile(boot_deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    )

    frac_le0 = float(np.mean(boot_deltas <= 0))
    frac_ge0 = float(np.mean(boot_deltas >= 0))
    p_value = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    # Report a floor instead of an impossible exact zero.
    p_value = max(p_value, 1.0 / n_samples)

    return BootstrapResult(
        n=n,
        point_delta=point_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_samples=n_samples,
        seed=seed,
        confidence=confidence,
    )


@dataclass(frozen=True)
class SignTestResult:
    """Exact two-sided sign test over pass/fail flips."""

    n_flips: int
    n_newly_failing: int  # baseline passed, candidate failed
    n_newly_passing: int  # baseline failed, candidate passed
    p_value: float | None
    method: str = "exact_sign_test"


def sign_test_on_flips(
    baseline_passed: Sequence[bool | None],
    candidate_passed: Sequence[bool | None],
) -> SignTestResult:
    """Exact binomial sign test on changed pass/fail decisions.

    Cases where either side has no decision are skipped. Under H0 (no
    difference), each flip direction is equally likely; the p-value is the
    two-sided binomial tail probability.
    """
    if len(baseline_passed) != len(candidate_passed):
        raise InsufficientDataError("flip test needs equal-length inputs")
    newly_failing = 0
    newly_passing = 0
    for b, c in zip(baseline_passed, candidate_passed, strict=True):
        if b is None or c is None or b == c:
            continue
        if b and not c:
            newly_failing += 1
        else:
            newly_passing += 1
    n = newly_failing + newly_passing
    if n == 0:
        return SignTestResult(0, 0, 0, None)
    k = min(newly_failing, newly_passing)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5**n)
    p_value = min(1.0, 2.0 * tail)
    return SignTestResult(n, newly_failing, newly_passing, p_value)


def means(
    baseline: Sequence[float], candidate: Sequence[float]
) -> tuple[float, float, float, float | None]:
    """(baseline_mean, candidate_mean, absolute delta, relative delta)."""
    if not baseline or not candidate:
        raise InsufficientDataError("cannot compute means of empty sequences")
    b = float(np.mean(baseline))
    c = float(np.mean(candidate))
    rel = (c - b) / abs(b) if abs(b) > EPS else None
    return b, c, c - b, rel
