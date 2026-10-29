"""The strategies the Futures tab offers, against calculations written from their definitions: trend and carry together on a bundle with planted signals, the catalogue of strategies and the settings each accepts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, bundle_from_prices, load_library
from src.framework.validate import check_causality

load_library()


def world(n: int = 900, seed: int = 4):
    """Four instruments: a riser with positive carry, a faller with negative carry, a riser with negative carry, and a drifter whose carry swings."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    drift = np.array([0.0007, -0.0006, 0.0006, 0.0])
    rets = drift + rng.normal(0.0, 0.009, (n, 4))
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=["UP", "DOWN", "MIXED", "SWING"])
    carry = pd.DataFrame({"UP": 0.05, "DOWN": -0.05, "MIXED": -0.04, "SWING": 0.06 * np.sin(np.arange(n) / 60.0)}, index=idx)
    return bundle_from_prices(prices, asset_class={c: "commodity" for c in prices.columns}, macro=carry.add_prefix("CARRY_"), name="planted")


@pytest.fixture(scope="module")
def b():
    return world()


def loop_daily_vol(returns: np.ndarray, halflife: float, min_periods: int = 21) -> np.ndarray:
    """The square root of the exponentially weighted mean of squared returns (``adjust=False``), written as a loop; NaN until ``min_periods`` returns have been seen."""
    alpha = 1.0 - math.exp(math.log(0.5) / halflife)
    out, level, seen = np.full(len(returns), np.nan), np.nan, 0
    for i, v in enumerate(returns):
        if not math.isfinite(v):
            continue
        level = v * v if seen == 0 else (1.0 - alpha) * level + alpha * v * v
        seen += 1
        if seen >= min_periods:
            out[i] = math.sqrt(level)
    return out


def expected_score(b, asset, lookback=252, skip=21, w=0.5, halflife=60.0, trend_scale=1.0, carry_scale=0.5, smooth=21):
    price = b.prices[asset].to_numpy()
    vol = loop_daily_vol(b.returns[asset].to_numpy(), halflife) * math.sqrt(252.0)
    carry = b.macro[f"CARRY_{asset}"].to_numpy()
    out = np.full(len(price), np.nan)
    for t in range(len(price)):
        if t < lookback or not math.isfinite(vol[t]) or vol[t] <= 0:
            continue
        mean_carry = np.mean(carry[max(0, t - smooth + 1):t + 1])
        past = math.log(price[t - skip] / price[t - lookback])
        trend = math.tanh(past / (vol[t] * math.sqrt((lookback - skip) / 252.0)) / trend_scale)
        out[t] = w * trend + (1.0 - w) * math.tanh(mean_carry / vol[t] / carry_scale)
    return out


@pytest.mark.parametrize("settings", [{}, {"lookback": 126, "skip": 5, "trend_weight": 0.7, "trend_scale": 0.8, "carry_scale": 0.3, "smooth": 10, "vol_halflife": 30.0}])
def test_momentum_carry_is_the_weighted_average_of_a_bounded_trend_part_and_a_bounded_carry_part(b, settings):
    got = MODELS.create("momentum_carry", **settings).score(b)
    kw = {"lookback": settings.get("lookback", 252), "skip": settings.get("skip", 21), "w": settings.get("trend_weight", 0.5), "halflife": settings.get("vol_halflife", 60.0),
          "trend_scale": settings.get("trend_scale", 1.0), "carry_scale": settings.get("carry_scale", 0.5), "smooth": settings.get("smooth", 21)}
    for asset in b.assets:
        want = expected_score(b, asset, **kw)
        assert np.isnan(got[asset].to_numpy()[:kw["lookback"]]).all()
        assert np.allclose(got[asset].to_numpy(), want, rtol=1e-9, atol=1e-12, equal_nan=True), asset
    assert got.abs().max().max() < 1.0


def test_the_two_parts_can_be_had_alone_and_oppose_each_other_where_trend_and_carry_disagree(b):
    only_trend, only_carry = MODELS.create("momentum_carry", trend_weight=1.0), MODELS.create("momentum_carry", trend_weight=0.0)
    trend, carry = only_trend.components(b)
    assert np.allclose(only_trend.score(b).to_numpy(), trend.where(carry.notna()).to_numpy(), equal_nan=True)                                  # a day on which either part is missing has no signal
    assert np.allclose(only_carry.score(b).to_numpy(), carry.where(trend.notna()).to_numpy(), equal_nan=True)
    both = MODELS.create("momentum_carry").score(b)
    last = both.iloc[-1]
    assert last["UP"] > 0.3 and last["DOWN"] < -0.3                                                    # a riser with positive carry is long; a faller with negative carry is short
    t, c = trend.iloc[-1]["MIXED"], carry.iloc[-1]["MIXED"]
    assert t > 0 > c and abs(last["MIXED"]) < max(abs(t), abs(c)) and last["MIXED"] == pytest.approx(0.5 * (t + c), rel=1e-12)       # disagreement shrinks the position


def test_momentum_carry_is_causal_refuses_a_bundle_without_carry_and_checks_its_settings(b):
    assert check_causality(MODELS.create("momentum_carry"), b, cutoff=b.index[600])["ok"]
    plain = bundle_from_prices(b.prices, name="no-signals")
    with pytest.raises(KeyError):
        MODELS.create("momentum_carry").score(plain)
    for bad in ({"lookback": 30, "skip": 21}, {"skip": -1}, {"trend_weight": 1.5}, {"trend_weight": -0.1}, {"vol_halflife": 0}, {"trend_scale": 0}, {"carry_scale": -1}, {"smooth": 0}):
        with pytest.raises(ValueError):
            MODELS.create("momentum_carry", **bad)
