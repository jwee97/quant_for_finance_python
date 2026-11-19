"""Synthetic reference tests for the shared ML pipeline (positions -> engine -> returns -> benchmark).

Every number here is derived by hand from a world whose truth is known, so a shared defect (a timing slip, a cash leak, a compounding error, a benchmark that is not what it says) shows up as a
failure that no individual model could cause. A perfect-foresight signal is used ONLY as a diagnostic control: it must be hugely profitable at the right delay and worthless at the wrong one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.backtest import equal_weight_benchmark, run_backtest
from src.ml.signals import long_only_top_quantile, top_n_long_short
from src.ml.targets import forward_return

BT = {"rebalance": "daily", "cost_bps": 0.0, "slippage_bps": 0.0, "target_vol": None, "max_leverage": 1.5, "extra_lag": 0}


def world(n: int = 400, k: int = 6, seed: int = 0, vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0003, vol, (n, k)), pd.bdate_range("2020-01-01", periods=n), list("ABCDEF")[:k])


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252))


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_a_position_decided_at_t_earns_the_return_of_t_plus_lag_plus_one(lag):
    r = world()
    for k in range(0, 6):                                               # score_t = the return of day t+k (perfect foresight, a diagnostic control)
        w = top_n_long_short(r.shift(-k).fillna(0.0), n=2, gross=1.0, net=0.0)
        s = sharpe(run_backtest(w, r, None, BT, lag).net_returns)
        if k == lag + 1:
            assert s > 20, f"perfect foresight at the engine's own delay must be hugely profitable, got {s:.1f}"
        else:
            assert abs(s) < 3, f"foresight of the wrong day ({k}) must earn nothing, got {s:.1f}"


def test_the_label_window_starts_where_the_position_starts_earning():
    r = world(60)
    prices = (1 + r).cumprod()
    for lag, h in [(1, 5), (2, 3)]:
        y = forward_return(prices, h, lag)
        t = 10
        earned = (1 + r.iloc[t + lag + 1: t + lag + h + 1]).prod() - 1        # the days a position decided at t earns: t+lag+1 .. t+lag+h
        assert np.allclose(y.iloc[t].to_numpy(), earned.to_numpy())


def test_the_last_lag_plus_horizon_rows_have_no_label():
    r = world(60)
    y = forward_return((1 + r).cumprod(), 5, 1)
    assert y.iloc[-6:].isna().all().all() and y.iloc[-7].notna().all()


def test_cash_accounting_a_long_only_book_with_weights_below_one_earns_only_its_invested_share():
    r = world(300)
    w = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    w["A"] = 0.5                                                           # half in A, half in cash that earns nothing
    got = run_backtest(w, r, None, {**BT, "rebalance": "daily"}, 1).net_returns.dropna()
    assert np.allclose(got.iloc[5:].to_numpy(), (0.5 * r["A"]).shift(0).reindex(got.index).iloc[5:].to_numpy(), atol=1e-12)


def test_costs_are_charged_on_traded_notional_only():
    r = world(300)
    w = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    w["A"] = 1.0
    free = run_backtest(w, r, None, BT, 1).net_returns
    paid = run_backtest(w, r, None, {**BT, "cost_bps": 10.0}, 1).net_returns
    diff = (free.reindex(paid.index).fillna(0.0) - paid.fillna(0.0))                  # row 0 has neither a book nor a trade
    assert paid.iloc[0] == pytest.approx(-0.001, abs=1e-12), "the entry cost must reach the net series, not vanish into a NaN"
    assert free.iloc[0] == 0.0 and paid.index[0] == free.index[0]
    assert diff.iloc[0] == pytest.approx(0.001, abs=1e-12)                   # buying 100% costs 10 bp, once
    assert diff.iloc[1:].abs().max() < 1e-9                                # holding with daily rebalance to a constant target trades nothing (weights drift but the target is re-set only on rebalance days)


def test_compounding_the_equity_curve_is_the_product_of_one_plus_net_returns():
    r = world(300)
    w = long_only_top_quantile(r.shift(-2).fillna(0.0), 0.5, 1.0)
    res = run_backtest(w, r, None, BT, 1)
    net = res.net_returns.dropna()
    eq = (1 + net).cumprod()
    assert eq.iloc[-1] == pytest.approx(np.exp(np.log1p(net).sum()), rel=1e-12)


def test_equal_weight_benchmark_is_the_daily_cross_sectional_mean_of_investable_returns():
    r = world(100)
    inv = pd.DataFrame(True, r.index, r.columns)
    inv.loc[r.index[:20], "F"] = False                                      # an asset that is not yet investable must not enter the benchmark
    b = equal_weight_benchmark(r, inv)
    assert np.allclose(b.iloc[:20], r.iloc[:20][list("ABCDE")].mean(axis=1))
    assert np.allclose(b.iloc[20:], r.iloc[20:].mean(axis=1))


def test_a_dollar_neutral_book_has_zero_beta_to_a_rising_market_and_cannot_track_the_equal_weight_benchmark():
    """The default rule is market neutral: in a world where every asset drifts up it earns the cross-sectional spread, not the market. Comparing it with the equal-weight (beta-1) benchmark is
    comparing different bets; the audit's finding that most models "lose to equal weight" is partly this mismatch, so it must be visible to the researcher."""
    rng = np.random.default_rng(3)
    n = 1500
    common = rng.normal(0.0008, 0.008, n)
    r = pd.DataFrame(common[:, None] + rng.normal(0, 0.004, (n, 6)), pd.bdate_range("2015-01-01", periods=n), list("ABCDEF"))
    w = top_n_long_short(r.shift(-2).fillna(0.0), n=2, gross=1.0, net=0.0)    # perfect foresight, dollar neutral
    net = run_backtest(w, r, None, BT, 1).net_returns.dropna()
    bench = equal_weight_benchmark(r).reindex(net.index)
    beta = np.cov(net, bench)[0, 1] / bench.var()
    assert abs(beta) < 0.1
    lo = run_backtest(long_only_top_quantile(r.shift(-2).fillna(0.0), 0.34, 1.0), r, None, BT, 1).net_returns.dropna()
    assert np.cov(lo, bench.reindex(lo.index))[0, 1] / bench.reindex(lo.index).var() > 0.5   # the long-only book does carry the market
