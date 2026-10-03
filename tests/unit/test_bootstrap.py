"""Unit tests: paired bootstrap and sign test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ledger.regression.bootstrap import (
    InsufficientDataError,
    means,
    paired_bootstrap,
    sign_test_on_flips,
)

pytestmark = pytest.mark.unit


def test_identical_inputs_zero_delta() -> None:
    values = [0.7] * 20
    result = paired_bootstrap(values, values, n_samples=500, seed=1)
    assert result.point_delta == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert not result.significant_regression
    assert not result.significant_improvement


def test_constant_shift_is_significant_improvement() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.uniform(0.4, 0.9, size=40).tolist()
    candidate = [b + 0.1 for b in baseline]
    result = paired_bootstrap(baseline, candidate, n_samples=1000, seed=42)
    assert result.point_delta == pytest.approx(0.1)
    assert result.ci_low == pytest.approx(0.1)
    assert result.ci_high == pytest.approx(0.1)
    assert result.significant_improvement
    assert not result.significant_regression
    # p-value floors at 1/n_samples instead of reporting impossible 0
    assert result.p_value == pytest.approx(1.0 / 1000)


def test_constant_regression_is_significant() -> None:
    baseline = [0.8] * 15
    candidate = [0.7] * 15
    result = paired_bootstrap(baseline, candidate, n_samples=500, seed=3)
    assert result.point_delta == pytest.approx(-0.1)
    assert result.significant_regression
    assert result.ci_high is not None and result.ci_high < 0


def test_ci_contains_point_estimate_and_widens_with_noise() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.uniform(0.3, 0.9, size=30).tolist()
    noise = rng.normal(0, 0.05, size=30)
    candidate = [b + 0.02 + n for b, n in zip(baseline, noise, strict=True)]
    result = paired_bootstrap(baseline, candidate, n_samples=2000, confidence=0.95, seed=7)
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low <= result.point_delta <= result.ci_high
    assert result.ci_low < result.point_delta < result.ci_high  # noise widens CI


def test_seed_reproducibility() -> None:
    rng = np.random.default_rng(1)
    baseline = rng.uniform(0, 1, size=25).tolist()
    candidate = rng.uniform(0, 1, size=25).tolist()
    a = paired_bootstrap(baseline, candidate, n_samples=800, seed=99)
    b = paired_bootstrap(baseline, candidate, n_samples=800, seed=99)
    c = paired_bootstrap(baseline, candidate, n_samples=800, seed=100)
    assert (a.ci_low, a.ci_high, a.p_value) == (b.ci_low, b.ci_high, b.p_value)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)


def test_higher_confidence_widens_interval() -> None:
    rng = np.random.default_rng(4)
    baseline = rng.uniform(0, 1, size=50).tolist()
    candidate = [b + rng.normal(0, 0.1) for b in baseline]
    narrow = paired_bootstrap(baseline, candidate, n_samples=2000, confidence=0.80, seed=5)
    wide = paired_bootstrap(baseline, candidate, n_samples=2000, confidence=0.99, seed=5)
    assert None not in (narrow.ci_low, narrow.ci_high, wide.ci_low, wide.ci_high)
    wide_width = (wide.ci_high or 0.0) - (wide.ci_low or 0.0)
    narrow_width = (narrow.ci_high or 0.0) - (narrow.ci_low or 0.0)
    assert wide_width > narrow_width


def test_input_validation() -> None:
    with pytest.raises(InsufficientDataError):
        paired_bootstrap([1.0], [1.0, 2.0])
    with pytest.raises(InsufficientDataError):
        paired_bootstrap([], [])
    with pytest.raises(InsufficientDataError):
        paired_bootstrap([float("nan")], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0], confidence=1.5)
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0], n_samples=0)


# ---------------------------------------------------------------------------
# sign test
# ---------------------------------------------------------------------------


def test_sign_test_no_flips() -> None:
    result = sign_test_on_flips([True, False, None], [True, True, False])
    # None side is skipped; (False -> True) is a single newly-passing flip
    assert result.n_flips == 1
    assert result.n_newly_passing == 1
    assert result.p_value == pytest.approx(1.0)


def test_sign_test_strong_asymmetry() -> None:
    base = [True] * 8
    cand = [False] * 8
    result = sign_test_on_flips(base, cand)
    assert result.n_newly_failing == 8
    assert result.p_value == pytest.approx(2 * 0.5**8)


def test_sign_test_balanced_flips_not_significant() -> None:
    base = [True] * 5 + [False] * 3
    cand = [False] * 5 + [True] * 3
    result = sign_test_on_flips(base, cand)
    assert result.n_flips == 8
    expected = min(1.0, 2 * sum(math.comb(8, i) for i in range(0, 4)) * 0.5**8)
    assert result.p_value == pytest.approx(expected)


def test_sign_test_length_mismatch() -> None:
    with pytest.raises(InsufficientDataError):
        sign_test_on_flips([True], [True, False])


# ---------------------------------------------------------------------------
# means helper
# ---------------------------------------------------------------------------


def test_means_helper() -> None:
    b, c, delta, rel = means([0.8, 0.6], [0.7, 0.5])
    assert b == pytest.approx(0.7)
    assert c == pytest.approx(0.6)
    assert delta == pytest.approx(-0.1)
    assert rel == pytest.approx(-0.1 / 0.7)


def test_means_zero_baseline_relative_is_none() -> None:
    _, _, _, rel = means([0.0], [0.1])
    assert rel is None


def test_means_empty_raises() -> None:
    with pytest.raises(InsufficientDataError):
        means([], [])
