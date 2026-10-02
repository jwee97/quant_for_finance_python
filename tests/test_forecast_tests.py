"""Forecast-comparison tests: Diebold-Mariano, Clark-West, OOS R-squared."""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.forecast_tests import (
    benjamini_hochberg,
    clark_west,
    diebold_mariano,
    oos_r2,
)


def _nested_forecasts(seed: int, signal: float, n: int = 260, split: int = 100):
    g = np.random.default_rng(seed)
    x, z = g.normal(size=n), g.normal(size=n)          # z is the extra predictor
    y = 0.5 * x + signal * z + g.normal(size=n)
    f1, f2 = np.zeros(n - split), np.zeros(n - split)
    for t in range(split, n):
        b1 = np.linalg.lstsq(x[:t, None], y[:t], rcond=None)[0]
        b2 = np.linalg.lstsq(np.column_stack([x[:t], z[:t]]), y[:t], rcond=None)[0]
        f1[t - split] = x[t] * b1[0]
        f2[t - split] = x[t] * b2[0] + z[t] * b2[1]
    return y[split:], f1, f2


def test_clark_west_is_close_to_nominal_size_when_the_extra_variable_is_useless():
    rejections = sum(clark_west(*_nested_forecasts(s, 0.0))["p_value"] < 0.05 for s in range(150))
    assert rejections / 150 < 0.10


def test_naive_diebold_mariano_is_badly_undersized_on_nested_models():
    """The reason Clark-West exists: DM almost never rejects here, even at 5%."""
    rejections = 0
    for s in range(150):
        y, f1, f2 = _nested_forecasts(s, 0.0)
        rejections += diebold_mariano((y - f1) ** 2, (y - f2) ** 2, alternative="greater")["p_value"] < 0.05
    assert rejections / 150 < 0.03


def test_clark_west_finds_a_genuine_predictor():
    rejections = sum(clark_west(*_nested_forecasts(2000 + s, 0.4))["p_value"] < 0.05 for s in range(60))
    assert rejections / 60 > 0.9


def test_oos_r2_is_zero_for_the_benchmark_and_positive_for_a_better_forecast():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    benchmark = np.zeros(500)
    assert oos_r2(y, benchmark, benchmark) == pytest.approx(0.0)
    assert oos_r2(y, benchmark, 0.8 * y) > 0.9
    assert oos_r2(y, benchmark, y + 3.0) < 0.0                 # a worse forecast scores negative


def test_diebold_mariano_detects_a_clearly_better_forecaster():
    rng = np.random.default_rng(1)
    y = rng.normal(size=600)
    good, bad = y + rng.normal(0, 0.3, 600), y + rng.normal(0, 1.0, 600)
    result = diebold_mariano((y - bad) ** 2, (y - good) ** 2, alternative="greater")
    assert result["p_value"] < 0.001
    assert result["statistic"] > 0                            # bad has the larger loss


def test_diebold_mariano_does_not_reject_identical_forecasts():
    rng = np.random.default_rng(2)
    loss = rng.normal(size=400) ** 2
    assert diebold_mariano(loss, loss)["p_value"] == pytest.approx(1.0)


def test_hac_lag_widens_the_interval_for_autocorrelated_loss_differentials():
    rng = np.random.default_rng(3)
    d = np.zeros(800)
    for t in range(1, 800):
        d[t] = 0.7 * d[t - 1] + rng.normal()
    d += 0.05
    naive = abs(diebold_mariano(d, np.zeros(800), lag=0)["statistic"])
    robust = abs(diebold_mariano(d, np.zeros(800), lag=10)["statistic"])
    assert robust < naive


def test_too_few_observations_return_nothing():
    assert clark_west(np.ones(5), np.ones(5), np.ones(5)) == {}
    assert diebold_mariano(np.ones(5), np.ones(5)) == {}


def test_benjamini_hochberg_rejects_a_prefix_of_the_sorted_p_values():
    mask = benjamini_hochberg([0.001, 0.8, 0.012, 0.2, 0.03], 0.10)
    assert mask.tolist() == [True, False, True, False, True]
    assert not benjamini_hochberg([0.5, 0.6, 0.9], 0.10).any()
