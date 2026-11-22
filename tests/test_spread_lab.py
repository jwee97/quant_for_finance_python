"""The futures spread lab: hedge ratio, stationarity and cointegration, and a walk-forward backtest that must not look ahead."""
import numpy as np
import pandas as pd
import pytest

from src.assets import spread_lab as sl


def _pair(n=1500, phi=0.95, seed=0, cointegrated=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 0.01, n))
    dev = np.zeros(n)
    for i in range(1, n):
        dev[i] = phi * dev[i - 1] + rng.normal(0, 0.006)
    b = pd.Series(common, index=idx)
    a = pd.Series(common + dev, index=idx) if cointegrated else pd.Series(np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    return a, b


def test_hedge_ratio_recovers_the_true_beta():
    a, b = _pair()
    h = sl.hedge_ratio(a, b)
    assert h["ols"] == pytest.approx(1.0, abs=0.1) and h["tls"] == pytest.approx(1.0, abs=0.1) and h["rolling_ols"]


def test_cointegrated_pair_passes_and_independent_walks_do_not():
    a, b = _pair()
    d = sl.diagnostics(a, b, a - sl.hedge_ratio(a, b)["ols"] * b)
    assert d["stationary_at_5pct"] and d["cointegrated_at_5pct"] and 3 < d["half_life_days"] < 30
    a2, b2 = _pair(cointegrated=False, seed=3)
    d2 = sl.diagnostics(a2, b2, a2 - b2)
    assert "no statistical evidence" in d2["verdict"] and not d2["stationary_at_5pct"]


def test_zscore_excludes_the_current_day():
    s = pd.Series(np.r_[np.zeros(50), 10.0], index=pd.bdate_range("2020-01-01", periods=51))
    z = sl.zscore(s + np.random.default_rng(1).normal(0, 1e-3, 51), 30)
    assert abs(z.iloc[-1]) > 100          # the spike is measured against a window that does not contain it


def test_backtest_makes_money_on_mean_reversion_and_is_labelled_simplified():
    a, b = _pair()
    bt = sl.backtest(a, b)
    assert bt["summary"]["sharpe"] > 0.3 and bt["summary"]["trades"] > 5
    assert bt["validation"]["realism"]["label"] == "simplified"


def test_backtest_has_no_lookahead():
    a, b = _pair()
    full = sl.backtest(a, b)
    cut = sl.backtest(a.iloc[:1200], b.iloc[:1200])
    k = len(cut["equity"]) - 5
    assert full["equity"][:k] == cut["equity"][:k]        # the past of the path does not change when later data are added


def test_costs_reduce_the_result():
    a, b = _pair()
    assert sl.backtest(a, b, cost_bps=0)["summary"]["annual_return"] > sl.backtest(a, b, cost_bps=30)["summary"]["annual_return"]


def test_too_little_data_is_refused():
    a, b = _pair(n=100)
    with pytest.raises(ValueError):
        sl.backtest(a, b)


def test_basket_weights():
    a, b = _pair(n=200)
    s = sl.build_spread({"A": a, "B": b}, {"A": 1.0, "B": -1.0})
    assert len(s) == 200
    with pytest.raises(ValueError):
        sl.build_spread({"A": a}, {"C": 1.0})
