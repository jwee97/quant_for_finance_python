"""The path-dependent volatility forecaster (Guyon and Lekeufack 2023): on a market built from its own equation it finds the loadings that were planted, and its simulated paths feed their own volatility."""

from __future__ import annotations

import numpy as np
import pytest

from src.forecast.base import ForecastError
from src.forecast.pdv import PathDependentVol
from src.forecast.registry import METHODS, catalog
from src.references import references
from tests import forecast_worlds as W

HS = (1, 5, 21)


def pdv_world(n=2600, seed=0, b0=0.004, b_trend=-0.9, b_act=0.75, nu=6.0):
    """Returns whose volatility is ``b0 + b_trend * trend + b_act * activity`` with the model's own fast clocks: after falls and after busy days volatility is higher."""
    rng = np.random.default_rng(seed)
    at, aa = 0.5 ** (1 / 3.0), 0.5 ** (1 / 8.0)
    trend, act = 0.0, 0.01 ** 2
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    r = np.empty(n)
    for t in range(n):
        sigma = float(np.clip(b0 + b_trend * trend + b_act * np.sqrt(act), 0.002, 0.06))
        r[t] = sigma * z[t]
        trend, act = at * trend + (1 - at) * r[t], aa * act + (1 - aa) * r[t] ** 2
    return r


def forecast(history, n=4000, seed=0, horizons=HS):
    m = PathDependentVol()
    state = m.fit(history.tail(m.window), horizons)
    return m.predict(state, history.tail(m.lookback), horizons, n, np.random.default_rng(seed)), state


def test_it_is_in_the_catalogue_with_its_source():
    assert "pdv" in METHODS and METHODS["pdv"].family == "Volatility" and not METHODS["pdv"].mean_signal
    view = next(m for m in catalog() if m["id"] == "pdv")
    assert view["min_history"] == 600 and "falls" in view["text"]
    assert any("path-dependent" in r["cite"] for r in references("forecaster", "pdv"))


def test_on_its_own_market_it_finds_that_falls_and_busy_days_raise_volatility():
    f, state = forecast(W.to_history(pdv_world(seed=1)))
    b = state["beta"]
    assert b[1] + b[2] < -0.2 and b[3] + b[4] > 0.3                                       # the trend loads negatively, the activity positively
    assert f.info["falls_raise_volatility"] is True and f.info["share_of_proxy_explained"] > 0.03 and f.info["volatility_proxy"] == "returns"
    assert f.samples.shape == (4000, 21) and np.isfinite(f.samples).all()
    assert np.array_equal(forecast(W.to_history(pdv_world(seed=1)))[0].samples, f.samples)                # the same seed, the same forecast


def test_on_independent_returns_it_finds_no_reaction_to_falls():
    f, state = forecast(W.to_history(W.iid(2600, sigma=0.01, seed=3)))
    assert abs(state["beta"][1] + state["beta"][2]) < 0.25
    sd = f.at(21).std()
    assert sd == pytest.approx(0.01 * np.sqrt(21), rel=0.2)                               # and its 21-day spread is the square-root-of-time one


def test_a_simulated_path_that_falls_becomes_more_volatile_and_the_long_return_is_skewed_to_the_left():
    f, _ = forecast(W.to_history(pdv_world(seed=2)), n=20000)
    daily = np.diff(np.concatenate([np.zeros((len(f.samples), 1)), f.samples], axis=1), axis=1)
    first, later = daily[:, :5].sum(axis=1), np.sqrt((daily[:, 5:12] ** 2).mean(axis=1))
    fell, rose = first < np.quantile(first, 0.2), first > np.quantile(first, 0.8)
    assert later[fell].mean() > 1.15 * later[rose].mean()
    total = f.at(21)
    skew = float(np.mean(((total - total.mean()) / total.std()) ** 3))
    assert skew < -0.1                                                                    # falls feed on themselves: the left tail of the month is the longer one


def test_the_forecast_is_wider_after_a_crash_than_after_a_calm_month():
    base = pdv_world(seed=4)
    calm, crash = base.copy(), base.copy()
    calm[-21:] = 0.002 * np.where(np.arange(21) % 2 == 0, 1, -1)
    crash[-5:] = [-0.03, -0.04, 0.01, -0.035, -0.02]
    wide, narrow = forecast(W.to_history(crash))[0], forecast(W.to_history(calm))[0]
    assert wide.at(5).std() > 1.8 * narrow.at(5).std()
    assert wide.info["volatility_now_annual"] > 1.8 * narrow.info["volatility_now_annual"]


def test_the_high_low_range_is_used_when_the_data_have_one_and_a_short_history_is_refused():
    f, state = forecast(W.to_history(W.garch(1800, seed=11), high_low=True), n=500)
    assert state["kind"] == "range" and f.info["volatility_proxy"] == "range" and np.isfinite(f.samples).all()
    z = state["shocks"]
    assert np.sqrt(np.mean(z * z)) == pytest.approx(1.0, abs=1e-9)                        # the resampled shocks have unit variance, so the volatility carries the scale
    m = PathDependentVol()
    with pytest.raises(ForecastError):
        m.fit(W.to_history(W.iid(300, seed=1)), HS)


def test_its_intervals_cover_on_its_own_market():
    r = pdv_world(n=3400, seed=6)
    m, hits, total = PathDependentVol(), 0, 0
    for origin in range(2600, 3380, 20):
        h = W.to_history(r[:origin])
        state = m.fit(h.tail(m.window), (5,))
        draws = m.predict(state, h.tail(m.lookback), (5,), 1500, np.random.default_rng(origin)).at(5)
        lo, hi = np.quantile(draws, [0.1, 0.9])
        hits += lo <= r[origin:origin + 5].sum() <= hi
        total += 1
    assert 0.66 <= hits / total <= 0.93                                                   # an 80% interval over 39 forecasts
