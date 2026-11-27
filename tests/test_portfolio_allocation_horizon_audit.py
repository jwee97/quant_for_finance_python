"""The multi-horizon assumptions, audited: which statements are identities and which are modelling assumptions, direct holding-horizon forecasts, and an independent units-and-cash ledger that reconciles overlapping
tranches against the authoritative backtest (cash conservation, trades, costs, P&L)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import alloc_horizon as hz
from src.ml import config as mlcfg
from src.ml import portfolio_alloc as pa
from src.ml.backtest import run_backtest
from tests.test_portfolio_allocation import panel
from tests.test_portfolio_allocation_v2 import BT, alloc_sig, informative


def _weights(r, s, **kw):
    w, log = pa.run(s, r, None, alloc_sig(h=5, opt_every=5, limits={"max_weight": 1.0}, cost_bps=0.0, **kw))
    return w, log


# ------------------------------------------------------------------------------------------------ identity versus assumption
def test_scale_invariance_is_an_identity_within_the_horizon_and_the_zero_beyond_assumption_shrinks_conviction_after_it():
    r = panel(vols=(0.01, 0.02), seed=101)
    s = informative(r, 101, strength=0.08, noise=0.03)                              # a weak forecast: an interior solution, so differences are visible
    base, _ = _weights(r, s, holding_days=5)
    for L in (1, 2, 3, 5):                                                                   # IDENTITY: mu scales with L and so does the covariance, so mean-variance weights do not depend on L
        w, _ = _weights(r, s, holding_days=L)
        assert np.allclose(w.to_numpy(), base.to_numpy(), atol=1e-6), L
    short, _ = _weights(r, s, holding_days=5)
    d = []
    for L in (10, 20, 40):                                                                   # ASSUMPTION: nothing is credited beyond the horizon while risk keeps growing, so the book moves toward minimum variance
        w, log = _weights(r, s, holding_days=L)
        rec = [x for x in log["records"] if x.get("horizon")][-1]["horizon"]
        assert rec["beyond_horizon"] == "zero" and any("nothing is credited beyond the forecast horizon" in a for a in rec["assumptions"])
        d.append(float((w.iloc[-1] - short.iloc[-1]).abs().sum()))
    assert d[0] > 1e-3 and d[0] <= d[1] + 1e-9 <= d[2] + 1e-9
    for L in (10, 20):                                                                       # persisting the rate restores invariance, and is labelled as the assumption it is
        w, log = _weights(r, s, holding_days=L, edge_beyond_horizon="persist")
        assert np.allclose(w.to_numpy(), base.to_numpy(), atol=1e-6), L
        assert "continue beyond the horizon" in " ".join([x for x in log["records"] if x.get("horizon")][-1]["horizon"]["assumptions"])


def test_linear_and_compound_scaling_agree_to_first_order_and_differ_at_second():
    mu = pd.Series({"A": 0.01, "B": 0.10, "C": -0.05})
    lin = hz.scale_expected_return(mu, 5, 10, "linear")
    comp = hz.scale_expected_return(mu, 5, 10, "compound")
    assert lin.to_dict() == pytest.approx({"A": 0.005, "B": 0.05, "C": -0.025})
    assert comp["B"] == pytest.approx(1.1 ** 0.5 - 1) and comp["B"] < lin["B"] and abs(comp["A"] - lin["A"]) < 1e-4
    assert hz.scale_expected_return(mu, 10, 10, "compound").to_dict() == pytest.approx(mu.to_dict()), "L = h needs no scaling under either model"
    two = hz.scale_expected_return(hz.scale_expected_return(mu, 5, 10, "compound"), 10, 10, "compound")
    assert two.to_dict() == pytest.approx(hz.scale_expected_return(mu, 5, 10, "compound").to_dict())
    assert hz.scale_expected_return(mu, 20, 10, "linear", "persist")["B"] == pytest.approx(0.2) and hz.scale_expected_return(mu, 20, 10, "linear", "zero")["B"] == pytest.approx(0.10)


def test_variance_scales_linearly_only_for_uncorrelated_increments():
    rng = np.random.default_rng(102)
    n, L = 40000, 5
    idx = pd.bdate_range("1990-01-01", periods=n)
    iid = pd.DataFrame(rng.standard_normal((n, 1)) * 0.01, index=idx, columns=["A"])
    ar = np.zeros((n, 1))
    for t in range(1, n):
        ar[t] = 0.3 * ar[t - 1] + rng.standard_normal(1) * 0.01
    ar = pd.DataFrame(ar, index=idx, columns=["A"])
    for name, x, expect_equal in (("iid", iid, True), ("ar", ar, False)):
        emp = float(x["A"].rolling(L).sum().dropna().var(ddof=1))
        lin = float(x["A"].var(ddof=1)) * L
        vr, _ = hz.horizon_covariance(x.cov(), x, L, "variance_ratio")
        assert (abs(emp / lin - 1.0) < 0.06) == expect_equal, (name, emp / lin)
        assert abs(vr.iloc[0, 0] / emp - 1.0) < 0.08, "the variance-ratio estimate recovers the empirical window variance in both cases"


# ------------------------------------------------------------------------------------------------ direct holding-horizon forecasts
def test_a_direct_forecast_is_used_as_it_is_and_refused_when_it_is_not_the_holding_window():
    r = panel(vols=(0.01, 0.02), seed=103)
    s = informative(r, 103, h=5)
    direct, log = _weights(r, s, forecast_source="direct")
    scaled, _ = _weights(r, s, holding_days=5)
    assert np.allclose(direct.to_numpy(), scaled.to_numpy(), atol=1e-12)
    rec = [x for x in log["records"] if x.get("horizon")][-1]["horizon"]
    assert rec["forecast_source"] == "direct" and "edge_scale" not in rec and any("direct 5-day forecast" in a for a in rec["assumptions"])
    with pytest.raises(pa.AllocationError, match="Train on fwd_ret_10"):
        _weights(r, s, forecast_source="direct", holding_days=10)
    with pytest.raises(mlcfg.MLConfigError, match="Train on fwd_ret_10"):
        mlcfg.experiment_config({"dataset": {"target": "fwd_ret_5", "horizon": 5}, "signal": {"method": "portfolio_allocation", "holding_period": 10, "allocation": {"forecast_source": "direct"}}})
    ok = mlcfg.experiment_config({"dataset": {"target": "fwd_ret_10", "horizon": 10}, "signal": {"method": "portfolio_allocation", "holding_period": 10, "allocation": {"forecast_source": "direct"}}})
    assert ok["signal"]["allocation"]["holding_days"] == 10


# ------------------------------------------------------------------------------------------------ overlapping tranches: cash, trades, costs, P&L
@pytest.mark.parametrize("life, hold, every", [("single_book", 5, 5), ("staggered_tranches", 10, 5), ("staggered_tranches", 20, 5), ("staggered_tranches", 20, 1)])
@pytest.mark.parametrize("freq", ["daily", "weekly", "monthly"])
@pytest.mark.parametrize("lag", [1, 3])
def test_an_independent_units_and_cash_ledger_reconciles_with_the_authoritative_backtest(life, hold, every, freq, lag):
    r = panel(vols=(0.01, 0.02, 0.015), corr=0.2, seed=63)
    s = informative(r, 63, h=5)
    w, log = pa.run(s, r, None, alloc_sig(h=5, holding_days=hold, opt_every=every, lifecycle=life, limits={"max_weight": 0.6}))
    res = run_backtest(w, r, None, {**BT, "rebalance": freq}, lag, "t")
    led = hz.ledger_replay(w, r, freq, lag, 7.0, cost_on="prior_equity")
    idx = res.net_returns.index
    assert np.abs(res.net_returns - led["net_return"].reindex(idx)).max() < 1e-12, "P&L"
    assert np.abs(res.costs.reindex(idx) - led["costs"].reindex(idx)).max() < 1e-12, "transaction costs"
    assert led["identity_error"] < 1e-12 and led["min_cash"] >= -1e-12, "cash conservation: cash + positions = equity, and no leverage is taken"
    traded = (res.trades.abs().sum(axis=1)).reindex(idx)
    eq_prev = led["equity"].shift(1).reindex(idx).fillna(1.0)
    pre_eq = led["equity"].reindex(idx) + led["costs"].reindex(idx) * eq_prev                # equity at the close before the trade's cost
    assert np.allclose(led["traded_notional"].reindex(idx), traded * pre_eq, atol=1e-9), "the units traded are the weights the engine says it traded"
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_the_physical_cost_convention_differs_from_the_engines_by_the_cost_times_the_days_return_only():
    r = panel(vols=(0.01, 0.02, 0.015), corr=0.2, seed=63)
    s = informative(r, 63, h=5)
    w, _ = pa.run(s, r, None, alloc_sig(h=5, holding_days=20, opt_every=5, lifecycle="staggered_tranches", limits={"max_weight": 0.6}))
    a = hz.ledger_replay(w, r, "weekly", 1, 7.0, cost_on="prior_equity")
    b = hz.ledger_replay(w, r, "weekly", 1, 7.0, cost_on="traded_notional")
    gap = (a["net_return"] - b["net_return"]).abs().max()
    assert 0 < gap < 0.01 * 7e-4 * 2 * 1.0 + 1e-6 and abs(a["equity"].iloc[-1] / b["equity"].iloc[-1] - 1) < 5e-4


def test_overlapping_tranches_hold_k_partial_positions_not_k_full_trades():
    r = panel(vols=(0.01, 0.02), seed=104)
    s = informative(r, 104, h=5)
    w, log = pa.run(s, r, None, alloc_sig(h=5, holding_days=20, opt_every=5, lifecycle="staggered_tranches"))
    books = [np.array(list(x["weights"].values())) for x in log["records"] if x.get("tranche")]
    held = [np.array(list(x["held_weights"].values())) for x in log["records"] if x.get("tranche")]
    assert all(abs(b.sum() - 1.0) < 1e-6 for b in books) and all(abs(h_.sum() - 1.0) < 1e-6 for h_ in held), "every tranche is a whole book and the account holds one book's worth, not four"
    for i in range(4, len(books)):
        assert np.allclose(np.mean(books[i - 3:i + 1], axis=0), held[i], atol=1e-9)
    turnover_tranche = np.mean([np.abs(held[i] - held[i - 1]).sum() / 2 for i in range(1, len(held))])
    turnover_single = np.mean([np.abs(books[i] - books[i - 1]).sum() / 2 for i in range(1, len(books))])
    assert turnover_tranche < turnover_single, "overlapping tranches trade 1/K of the change per decision"
