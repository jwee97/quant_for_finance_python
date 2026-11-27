"""Currency in the pipeline: exchange rates through the ticker store, conversion in the allocation and the backtest, FX costs inside the authoritative engine's cost model, an explicit hedge overlay with its own
cash flows and costs, and diagnostics. The provider is a local fake: no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import alloc_fx as fxm
from src.ml import experiments as ex
from src.ml import fx_data as fxd
from src.utils.config import load_config
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from src.webapp.universe import TickerStore, UniverseBuilder

N = 1500
IDX = pd.bdate_range("2015-01-05", periods=N)


def _frame(seed, drift=0.0003, vol=0.012, start=100.0):
    rng = np.random.default_rng(seed)
    p = start * np.cumprod(1.0 + drift + vol * rng.standard_normal(N))
    s = pd.Series(p, index=IDX)
    return pd.DataFrame({"close": s, "high": s * 1.005, "low": s * 0.995, "volume": 1e6})


def _fx(seed=9):
    rng = np.random.default_rng(seed)
    s = pd.Series(0.0067 * np.exp(np.cumsum(rng.normal(0, 0.004, N))), index=IDX)
    return pd.DataFrame({"close": s, "high": s, "low": s, "volume": 0.0})


def make_fetch(drop_fx=False, gap_after=None):
    def fetch(tickers, start):
        out = {}
        for t in tickers:
            if t == "JPYUSD=X":
                if drop_fx:
                    out[t] = "no data found: check the symbol"
                    continue
                f = _fx()
                out[t] = f if gap_after is None else f.iloc[:gap_after]
            elif t.startswith("NEW"):
                out[t] = _frame(abs(hash(t)) % 1000 + 1)
            else:
                out[t] = "no data found: check the symbol"
        return out
    return fetch


def builder(tmp_path, **kw):
    config = load_config()
    return config, UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=make_fetch(**kw)), default_loader=lambda c: make_bundle(n_assets=2, n=80, macro=False))


def cfg(fx=None, **sig):
    c = {"dataset": {"universe": ["NEWAAA", "NEWTOYO"], "feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5", "horizon": 5, "min_history": 60},
         "split": {"min_train_days": 300, "refit_every": 126, "validation_fraction": 0.2, "test_fraction": 0.2, "forward_fraction": 0.1}, "model": {"kind": "ridge"},
         "backtest": {"rebalance": "weekly", "target_vol": None}, "signal": {"method": "portfolio_allocation", "holding_period": 1, "allocation": {"allocator": "inverse_vol", "limits": {"max_weight": 0.9}, **sig}}}
    if fx is not None:
        c["backtest"]["fx"] = fx
    return c


FX = {"base": "USD", "currencies": {"NEWAAA": "USD", "NEWTOYO": "JPY"}}


def run(tmp_path, fx=None, **kw):
    config, b = builder(tmp_path, **kw.pop("fetch", {}))
    return ex.run_experiment(cfg(fx, **kw), config=config, builder=b, save=False, explain=False, final_model=False)


def test_exchange_rates_load_through_the_ticker_store_with_provenance_and_diagnostics(tmp_path):
    _, b = builder(tmp_path)
    table, rep = fxd.load_table(b.store, "USD", ["JPY", "USD"], "2015-01-01")
    assert fxd.fx_symbol("JPY", "USD") == "JPYUSD=X" and table.currencies() == ["JPY"]
    s = rep["series"]["JPY"]
    assert s["symbol"] == "JPYUSD=X" and s["observations"] == N and s["fetched_at"] and s["first"] == "2015-01-05" and "forward points" in " ".join(rep["assumptions"])
    cov = fxm.coverage(table, {"A": "JPY", "B": "USD"}, IDX)
    assert cov["JPY"]["coverage"] > 0.99 and cov["JPY"]["longest_gap_days"] <= 1
    _, b2 = builder(tmp_path / "x", drop_fx=True)
    with pytest.raises(fxm.FxError, match="JPYUSD=X.*no data found.*never guessed"):
        fxd.load_table(b2.store, "USD", ["JPY"], "2015-01-01")


def test_a_run_in_the_base_currency_reports_exposure_attribution_costs_and_data_diagnostics(tmp_path):
    local = run(tmp_path)
    o = run(tmp_path / "fx", {**FX, "cost_bps": 8.0})
    r = o.report["fx"]
    assert r["base"] == "USD" and r["foreign_assets"] == ["NEWTOYO"] and r["quote"].startswith("base currency per one unit") and r["exchange_accurate"] is False
    assert set(r["exposure"]["latest"]) == {"JPY", "USD"} and abs(sum(r["exposure"]["latest"].values()) - 1.0) < 1e-6
    assert r["data"]["coverage"]["JPY"]["coverage"] > 0.99 and r["data"]["series"]["JPY"]["symbol"] == "JPYUSD=X"
    assert r["forecast_conversion"]["assumption"] == "random_walk" and any("not exchange-accurate" in w for w in o.report["warnings"])
    # the valuation currency matters: the same forecasts through the same allocator earn differently once the yen's moves are in the P&L
    assert not np.allclose(local.net.reindex(o.net.index).fillna(0.0), o.net)
    assert abs(r["attribution"]["currency_return_total"]) > 1e-3 and set(r["attribution"]["by_currency_fx"]) == {"JPY", "USD"} and abs(r["attribution"]["by_currency_fx"]["USD"]) < 1e-12


def test_conversion_costs_are_inside_the_authoritative_backtest_cost_model(tmp_path):
    free = run(tmp_path, {**FX, "cost_bps": 0.0})
    dear = run(tmp_path / "d", {**FX, "cost_bps": 25.0})
    extra = (dear.costs - free.costs.reindex(dear.costs.index)).sum()
    assert extra > 0 and dear.net.sum() < free.net.sum()
    foreign_traded = dear.report["fx"]["costs"]["conversion_cost_total"]
    assert foreign_traded == pytest.approx(extra, rel=0.02, abs=1e-6), "the extra engine cost is exactly the conversion rate times the foreign notional traded"
    assert dear.report["fx"]["costs"]["note"].startswith("the conversion cost is inside the engine's costs")


def test_a_hedge_is_an_explicit_overlay_whose_cash_flows_reconcile_with_the_net_result(tmp_path):
    un = run(tmp_path, {**FX, "hedge": 0.0}, allocator="equal_weight")                           # a book that does not depend on the risk estimate, so that both runs hold the same weights
    hd = run(tmp_path / "h", {**FX, "hedge": 1.0, "hedge_cost_bps": 5.0}, allocator="equal_weight")
    h = hd.report["fx"]["hedge"]
    assert h["active"] and h["carry"] == "not modelled" and "settled daily" in h["instrument"] and hd.report["fx"]["costs"]["hedge_cost_total"] > 0
    idx = hd.net.index.intersection(un.net.index)
    diff = (hd.net.reindex(idx) - un.net.reindex(idx)).sum()
    assert diff == pytest.approx(h["pnl_total"] - h["cost_total"], abs=1e-9), "hedged net = unhedged net + hedge P&L - hedge cost, exactly"
    # hedging the whole yen exposure removes most of the currency's contribution to day-to-day variance
    assert hd.net.std() < un.net.std()
    assert abs(hd.report["fx"]["hedge"]["residual_exposure_latest"]["JPY"]) < 0.25


def test_hedge_overlay_arithmetic_on_a_known_book():
    idx = pd.bdate_range("2024-01-01", periods=6)
    t = fxm.FxTable("USD")
    t.set_series("JPY", pd.Series([0.0070, 0.0071, 0.0072, 0.00693, 0.0070, 0.0071], index=idx))
    ctx = fxm.FxContext(t, {"A": "JPY", "B": "USD"}, hedge={"A": 0.5}, hedge_cost_bps=10.0)
    held = pd.DataFrame({"A": 0.4, "B": 0.6}, index=idx)
    trades = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    trades.iloc[0] = [0.4, 0.6]
    fxr = pd.DataFrame({"A": t.returns("JPY", idx), "B": 0.0})
    o = fxm.hedge_overlay(held, trades, fxr, ctx)
    assert o["notional"]["JPY"].tolist() == pytest.approx([0.2] * 6), "half of the 40% yen exposure is hedged and struck once"
    assert o["pnl"].iloc[2] == pytest.approx(-0.2 * (0.0072 / 0.0071 - 1)) and o["pnl"].iloc[0] == 0.0
    assert o["cost"].iloc[0] == pytest.approx(0.2 * 10 / 1e4) and o["cost"].iloc[1:].sum() == 0.0
    assert o["residual_exposure"]["JPY"].tolist() == pytest.approx([0.2] * 6)


def test_missing_rates_stop_the_run_with_the_symbol_and_the_exclude_policy_is_explicit(tmp_path):
    with pytest.raises(ex.ExperimentError, match="JPYUSD=X.*never guessed"):
        run(tmp_path, FX, fetch={"drop_fx": True})
    with pytest.raises(ex.ExperimentError, match="exchange rates are missing"):
        run(tmp_path / "g", FX, fetch={"gap_after": 900})
    ok = run(tmp_path / "e", {**FX, "missing": "exclude", "max_stale_days": 2}, fetch={"gap_after": 900})
    m = ok.report["fx"]["data"]["missing"]
    assert m["policy"] == "exclude" and m["assets"]["NEWTOYO"]["days_missing_after_first_rate"] > 0 and ok.report["fx"]["data"]["coverage"]["JPY"]["coverage"] < 0.7


def test_undeclared_currencies_and_unsupported_methods_are_refused_with_what_to_change(tmp_path):
    config, b = builder(tmp_path)
    with pytest.raises(Exception, match="does not say which currency NEWTOYO trades in"):
        ex.run_experiment(cfg({"base": "USD", "currencies": {"NEWAAA": "USD"}}), config=config, builder=b, save=False, explain=False, final_model=False)
    bad = cfg(FX)
    bad["signal"] = {"method": "proportional", "holding_period": 1}
    with pytest.raises(Exception, match="portfolio-allocation method only"):
        ex.run_experiment(bad, config=config, builder=b, save=False, explain=False, final_model=False)


def test_without_fx_nothing_changes():
    from src.ml import config as mlcfg

    a = mlcfg.experiment_config({"dataset": {"target": "fwd_ret_5", "horizon": 5}})
    assert "fx" not in a["backtest"] and ex.config_hash(a) == ex.config_hash(mlcfg.experiment_config({"dataset": {"target": "fwd_ret_5", "horizon": 5}}))
    assert ex.config_hash(mlcfg.experiment_config({})) == "c1b756f10a81"
    with pytest.raises(ApiError if False else Exception):
        mlcfg.experiment_config({"backtest": {"fx": {"base": "usd"}}, "signal": {"method": "portfolio_allocation"}})


# ------------------------------------------------------------------------------------------------ the web API
def test_the_api_validates_the_fx_section_and_a_rebacktest_values_the_saved_forecasts_in_the_base_currency(tmp_path):
    from src.webapp.api import App
    from tests.app_roots import isolated_root
    from tests.test_webapp_assets_ml import finished

    config, b = builder(tmp_path)
    app = App(config, builder=b, root=isolated_root(tmp_path / "root", config))
    base = {**cfg(), "name": "two stocks"}
    base["signal"] = {"method": "proportional", "holding_period": 1}
    parent = finished(app, app.extra_post["/api/ml/train"]({**base, "mode": "development", "explain": False}))
    sig = {"method": "portfolio_allocation", "holding_period": 1, "allocation": {"allocator": "inverse_vol", "limits": {"max_weight": 0.9}}}
    for fx, text in (({"base": "dollars"}, "three-letter code"), ({"base": "USD"}, "fx.currencies maps every ticker"), ({"base": "USD", "currencies": {"A": "USD"}, "hedge": 2}, "between 0 and 1"),
                     ({"base": "USD", "currencies": {"A": "USD"}, "cost_bps": -1}, "cost_bps"), ({"base": "USD", "currencies": {"A": "USD"}, "bogus": 1}, "no setting"), ({"base": "USD", "currencies": {"A": "USD"}, "missing": "guess"}, "raise or exclude")):
        with pytest.raises(ApiError, match=text):
            app.extra_post["/api/ml/rebacktest"]({"id": parent["id"], "signal": sig, "backtest": {"fx": fx}})
    child = finished(app, app.extra_post["/api/ml/rebacktest"]({"id": parent["id"], "signal": sig, "backtest": {"fx": {**FX, "hedge": 0.5, "cost_bps": 6.0, "hedge_cost_bps": 2.0}}}))
    r = child["report"]["fx"]
    assert child["parent"] == parent["id"] and r["base"] == "USD" and r["hedge"]["active"] and r["hedge_ratio"] == {"NEWTOYO": 0.5} and r["data"]["series"]["JPY"]["symbol"] == "JPYUSD=X"
    assert child["report"]["config"]["backtest"]["fx"]["currencies"] == {"NEWAAA": "USD", "NEWTOYO": "JPY"}
    import json

    json.dumps(r, allow_nan=False)
