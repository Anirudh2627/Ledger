"""Statistical primitives for regression detection.

Method: paired percentile bootstrap for confidence intervals and paired
sign-flip permutation for hypothesis-test p-values.

Baseline and candidate are evaluated on the *same* dataset, so per-case
scores form natural pairs. Resampling case indices (with replacement)
estimates the confidence interval for the mean difference. The p-value is
computed separately from the null distribution formed by sign-flipping paired
differences; this test is exact when pairwise differences are exchangeable
under the null.

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
from numbers import Real
from operator import index

import numpy as np

#: Deltas with magnitude below this are numerically zero.
EPS = 1e-12


class InsufficientDataError(ValueError):
    """Raised when there is not enough paired data to run statistics."""


def _finite_array(values: Sequence[float], label: str) -> np.ndarray:
    try:
        items = list(values)
    except TypeError as exc:
        raise InsufficientDataError(f"{label} must be a sequence of finite numbers") from exc
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in items):
        raise InsufficientDataError(f"{label} must contain finite numeric values")
    try:
        array = np.asarray(items, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise InsufficientDataError(f"{label} must contain finite numeric values") from exc
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise InsufficientDataError(f"{label} must contain finite numeric values")
    return array


def _stable_mean(values: np.ndarray, label: str) -> float:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if not math.isfinite(scale):
        raise InsufficientDataError(f"{label} must remain finite")
    if scale == 0.0:
        return 0.0
    mean = float(np.mean(values / scale)) * scale
    if not math.isfinite(mean):
        raise InsufficientDataError(f"{label} must remain finite")
    return mean


def _paired_permutation_p_value(differences: np.ndarray, *, n_samples: int, seed: int) -> float:
    """Two-sided sign-flip test, exact through 16 pairs and Monte Carlo beyond."""
    n = len(differences)
    observed = abs(float(np.mean(differences)))
    if n <= 16:
        masks = np.arange(1 << n, dtype=np.uint32)[:, None]
        bits = np.arange(n, dtype=np.uint32)
        signs = 1.0 - 2.0 * ((masks >> bits) & 1)
        permuted_means = signs @ differences / n
        extreme = np.count_nonzero(np.abs(permuted_means) >= observed - EPS)
        return float(extreme / len(permuted_means))

    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_samples, n), dtype=np.int8) * 2 - 1
    permuted_means = (signs * differences).mean(axis=1)
    extreme = np.count_nonzero(np.abs(permuted_means) >= observed - EPS)
    return float((extreme + 1) / (n_samples + 1))


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
    p_value_method: str = "paired_sign_flip_permutation"

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
    """Estimate a paired mean-difference interval and test by sign-flip permutation.

    Args:
        baseline: per-case baseline scores.
        candidate: per-case candidate scores (same order/length: pairs).
        n_samples: bootstrap and Monte Carlo permutation resamples.
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
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("confidence must be in (0, 1)")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if isinstance(n_samples, bool):
        raise ValueError("n_samples must be >= 1")
    try:
        n_samples = index(n_samples)
    except TypeError as exc:
        raise ValueError("n_samples must be a positive integer") from exc
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    base = _finite_array(baseline, "baseline scores")
    cand = _finite_array(candidate, "candidate scores")

    with np.errstate(over="ignore", invalid="ignore"):
        diffs = cand - base
    if not np.all(np.isfinite(diffs)):
        raise InsufficientDataError("paired differences must remain finite")
    scale = float(np.max(np.abs(diffs)))
    scaled_diffs = diffs / scale if scale else diffs
    mean_scaled = float(np.mean(scaled_diffs)) if scale else 0.0
    point_delta = mean_scaled * scale if scale else 0.0
    if not math.isfinite(point_delta):
        raise InsufficientDataError("paired mean difference must remain finite")

    rng = np.random.default_rng(seed)
    # One (n_samples, n) index matrix; same indices for both arms => paired.
    idx = rng.integers(0, n, size=(n_samples, n))
    boot_deltas = scaled_diffs[idx].mean(axis=1)

    alpha = 1.0 - confidence
    ci_low_scaled, ci_high_scaled = (
        float(v) for v in np.percentile(boot_deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    )
    ci_low, ci_high = ci_low_scaled * scale, ci_high_scaled * scale
    if not all(math.isfinite(value) for value in (ci_low, ci_high)):
        raise InsufficientDataError("bootstrap confidence interval must remain finite")
    p_value = _paired_permutation_p_value(scaled_diffs, n_samples=n_samples, seed=seed)

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
    if len(baseline) == 0 or len(candidate) == 0:
        raise InsufficientDataError("cannot compute means of empty sequences")
    base = _finite_array(baseline, "baseline scores")
    cand = _finite_array(candidate, "candidate scores")
    b = _stable_mean(base, "baseline mean")
    c = _stable_mean(cand, "candidate mean")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = c - b
        rel = delta / abs(b) if abs(b) > EPS else None
    if not math.isfinite(delta) or (rel is not None and not math.isfinite(rel)):
        raise InsufficientDataError("mean difference and relative change must remain finite")
    return b, c, delta, rel
