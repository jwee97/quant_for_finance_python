"""Release 5, P1-18: matched benchmarks and causal exposures.

Strategies are built from known ingredients (a market exposure, an alpha, noise), so the correct answer to "what is the incremental return" is known in advance."""

import numpy as np
import pandas as pd
import pytest

from src.ml import benchmarks as bm

N = 1800
IDX = pd.bdate_range("2014-01-01", periods=N)


def world(seed=0, beta=0.7, alpha=0.0004, n_assets=6, drift=0.0003):
    rng = np.random.default_rng(seed)
    market = pd.Series(rng.normal(drift, 0.008, N), index=IDX)
    strat = beta * market + alpha + rng.normal(0, 0.004, N)
    rets = pd.DataFrame(rng.normal(0.0003, 0.01, (N, n_assets)), index=IDX, columns=[f"A{i}" for i in range(n_assets)])
    return market, pd.Series(strat, index=IDX), rets


def test_the_beta_used_on_a_day_never_depends_on_that_day_or_any_later_one():
    market, strat, _ = world(1)
    base = bm.causal_beta(strat, market, window=252, min_obs=60)
    k = 1000
    m2, s2 = market.copy(), strat.copy()
    m2.iloc[k:] = m2.iloc[k:] * 3 + 0.01                                                      # the future changes completely
    s2.iloc[k:] = -s2.iloc[k:]
    changed = bm.causal_beta(s2, m2, window=252, min_obs=60)
    pd.testing.assert_series_equal(base.iloc[: k + 1], changed.iloc[: k + 1])                  # up to and including day k, nothing moved: the beta for day k uses data to k-1
    assert not np.allclose(base.iloc[k + 1:k + 50], changed.iloc[k + 1:k + 50], equal_nan=True)
    assert base.iloc[:60].isna().all(), "no estimate before min_obs observations: nothing is filled"


def test_a_beta_matched_benchmark_recovers_the_alpha_that_an_unmatched_comparison_misattributes():
    market, strat, _ = world(2, beta=0.5, alpha=0.0004, drift=0.002)
    bench, beta = bm.beta_matched(strat, market)
    assert abs(float(beta.dropna().iloc[-500:].mean()) - 0.5) < 0.05
    matched = bm.incremental(strat, bench, kind="beta_matched", n_boot=200)
    unmatched = bm.incremental(strat, market, kind="descriptive_equal_weight", n_boot=200)
    a_m, a_u = matched["active_mean"]["value"], unmatched["active_mean"]["value"]
    assert a_m == pytest.approx(0.0004, abs=0.0002), "against the beta-matched benchmark the active return is the planted alpha"
    assert a_m - a_u > 0.0005 and a_u < a_m and matched["matched"] is True and unmatched["matched"] is False, "against the unmatched market the strategy looks like it has given away the half of the market premium it did not hold"
    assert matched["regression"]["alpha"]["value"] > 0 and matched["regression"]["beta"]["value"] == pytest.approx(1.0, abs=0.15)
    assert matched["n_aligned"] < N and matched["dropped"]["benchmark"] >= 0, "days without a causal beta have no matched benchmark and are dropped, not filled: the count says how many"


def test_a_dollar_neutral_book_is_matched_by_cash_and_a_levered_long_book_by_a_levered_market():
    market, _, rets = world(3)
    neutral = pd.DataFrame(0.0, index=IDX, columns=rets.columns)
    neutral.iloc[:, :3], neutral.iloc[:, 3:] = 0.1, -0.1
    assert bm.net_exposure_matched(neutral, market).dropna().abs().max() < 1e-12
    long_ = pd.DataFrame(0.0, index=IDX, columns=rets.columns)
    long_.iloc[:, :2] = 0.75
    m = bm.net_exposure_matched(long_, market)
    assert m.iloc[10] == pytest.approx(1.5 * market.iloc[10]) and np.isnan(m.iloc[0]), "the exposure of the PREVIOUS day: nothing on the first day"
    e = bm.book_exposures(neutral)
    assert e["gross"].iloc[0] == pytest.approx(0.6) and e["net"].iloc[0] == pytest.approx(0.0) and e["long"].iloc[0] == pytest.approx(0.3) and e["short"].iloc[0] == pytest.approx(0.3)


def test_the_sector_matched_benchmark_credits_a_bet_on_a_group_to_the_group_not_to_skill():
    _, _, rets = world(4)
    rets = rets.copy()
    rets[["A0", "A1", "A2"]] += 0.0008                                                         # group "x" simply does better
    group = {"A0": "x", "A1": "x", "A2": "x", "A3": "y", "A4": "y", "A5": "y"}
    held = pd.DataFrame(0.0, index=IDX, columns=rets.columns)
    held[["A0", "A1", "A2"]] = 0.15                                                           # long group x only; no stock picking within it
    strat = (held.shift(1) * rets).sum(axis=1).iloc[1:]
    bench = bm.sector_matched(held, rets, group).iloc[1:]
    inc = bm.incremental(strat, bench, kind="sector_matched", n_boot=100)
    assert abs(inc["active_mean"]["value"]) < 1e-12, "holding group x equally is exactly the group's equal-weight return: no skill within the group"


def test_the_risk_matched_benchmark_has_the_strategys_trailing_volatility_and_uses_only_the_past():
    market, strat, _ = world(5, beta=0.3)
    b = bm.risk_matched(strat, market, window=63)
    both = pd.concat([strat.rolling(63).std().shift(1), b.rolling(63).std()], axis=1).dropna()
    assert (both.iloc[:, 1] / both.iloc[:, 0]).between(0.6, 1.6).mean() > 0.9
    k = 900
    s2 = strat.copy(); s2.iloc[k:] *= 5
    pd.testing.assert_series_equal(b.iloc[: k + 1], bm.risk_matched(s2, market, window=63).iloc[: k + 1])


def test_a_benchmark_pays_the_same_trading_cost_per_unit_turnover_as_the_strategy():
    w = pd.DataFrame({"a": [0.5, 0.5, 0.2], "b": [0.5, 0.5, 0.8]}, index=IDX[:3])
    cost = bm.turnover_cost(w, 10.0)
    assert cost.iloc[0] == pytest.approx(0.001) and cost.iloc[1] == 0.0 and cost.iloc[2] == pytest.approx(0.0006), "first day builds the book; a constant book trades nothing; moving 0.3 each way costs 0.6 of weight"
