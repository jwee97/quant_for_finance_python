"""Bootstrap utilities and the paired Sharpe-difference test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.robustness import (
    pairwise_sharpe_tests,
    paired_sharpe_test,
    stationary_bootstrap_indices,
)


def test_bootstrap_indices_are_valid_and_blocky():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(8000, 21, rng)
    assert idx.min() >= 0 and idx.max() < 8000
    continuing = np.diff(idx) % 8000 == 1
    mean_block = 1.0 / (1.0 - continuing.mean())
    assert mean_block == pytest.approx(21, rel=0.15)


def test_bootstrap_preserves_short_range_dependence():
    """A resample of an AR(1) series must keep most of its autocorrelation."""
    rng = np.random.default_rng(1)
    x = np.zeros(6000)
    for t in range(1, 6000):
        x[t] = 0.6 * x[t - 1] + rng.normal()
    idx = stationary_bootstrap_indices(6000, 40, rng)
    resampled = x[idx]
    shuffled = rng.permutation(x)
    ac = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]
    assert ac(resampled) > 0.4
    assert abs(ac(shuffled)) < 0.1


def _series(values, n):
    return pd.Series(values, index=pd.bdate_range("2010-01-01", periods=n))


def test_identical_strategies_are_not_distinguishable():
    rng = np.random.default_rng(2)
    x = _series(rng.normal(0.0004, 0.01, 2500), 2500)
    result = paired_sharpe_test(x, x)
    assert result["difference"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0, abs=0.02)


def test_pairing_detects_a_difference_that_marginal_intervals_hide():
    """Two 0.96-correlated books with a real Sharpe gap: pairing must find it."""
    rng = np.random.default_rng(3)
    n = 3000
    common = rng.normal(0.0, 0.01, n)
    a = _series(common + 0.0006 + rng.normal(0, 0.002, n), n)
    b = _series(common + 0.0001 + rng.normal(0, 0.002, n), n)
    result = paired_sharpe_test(a, b, n_samples=1000)
    assert result["return_correlation"] > 0.9
    assert result["difference"] > 0.3
    assert result["p_value"] < 0.01


def test_test_has_roughly_nominal_size_under_the_null():
    rejections, trials = 0, 80
    for seed in range(trials):
        g = np.random.default_rng(1000 + seed)
        a = _series(g.normal(0.0004, 0.01, 1200), 1200)
        b = _series(g.normal(0.0004, 0.01, 1200), 1200)
        rejections += paired_sharpe_test(a, b, n_samples=300, seed=seed)["p_value"] < 0.05
    assert rejections / trials < 0.12


def test_short_samples_return_nothing_rather_than_a_misleading_number():
    x = _series(np.random.default_rng(4).normal(0, 0.01, 50), 50)
    assert paired_sharpe_test(x, x) == {}


def test_pairwise_tests_apply_fdr_control():
    rng = np.random.default_rng(5)
    n = 2000
    common = rng.normal(0.0, 0.01, n)
    streams = {
        "good": _series(common + 0.0007 + rng.normal(0, 0.002, n), n),
        "bad": _series(common - 0.0003 + rng.normal(0, 0.002, n), n),
        "twin": _series(common - 0.0003 + rng.normal(0, 0.002, n), n),
    }
    table = pairwise_sharpe_tests(streams, n_samples=500)
    assert len(table) == 3
    assert {"bh_significant", "significant_raw_5pct", "p_value"} <= set(table.columns)
    twin_vs_bad = table[((table["a"] == "bad") & (table["b"] == "twin")) |
                        ((table["a"] == "twin") & (table["b"] == "bad"))]
    assert not bool(twin_vs_bad["bh_significant"].iloc[0])


def test_paired_volatility_test_finds_a_real_difference_and_not_a_fake_one():
    from src.validation.robustness import paired_volatility_test

    rng = np.random.default_rng(0)
    shocks = pd.Series(rng.standard_normal(2500) * 0.01, index=pd.bdate_range("2012-01-02", periods=2500))
    found = paired_volatility_test(shocks * 1.25, shocks, n_samples=600, seed=1)
    assert found["ratio"] == pytest.approx(1.25, rel=1e-9) and found["difference"] > 0
    assert found["p_value"] < 0.01
    noise = shocks + pd.Series(rng.standard_normal(2500) * 0.003, index=shocks.index)
    null = paired_volatility_test(noise, shocks + pd.Series(rng.standard_normal(2500) * 0.003, index=shocks.index),
                                  n_samples=600, seed=1)
    assert null["p_value"] > 0.05 or abs(null["ratio"] - 1.0) < 0.03


def test_paired_volatility_test_pairs_the_resampling():
    """Two nearly identical streams: the difference is estimated far more tightly than either volatility."""
    from src.validation.robustness import paired_volatility_test

    rng = np.random.default_rng(1)
    base = pd.Series(rng.standard_normal(2500) * 0.01, index=pd.bdate_range("2012-01-02", periods=2500))
    other = base * 1.05 + pd.Series(rng.standard_normal(2500) * 0.0005, index=base.index)
    result = paired_volatility_test(other, base, n_samples=600, seed=2)
    width = result["ci_upper_95pct"] - result["ci_lower_5pct"]
    assert width < 0.25 * result["volatility_b"] * 0.05 * 4          # far narrower than an unpaired interval
    assert result["return_correlation"] > 0.99
