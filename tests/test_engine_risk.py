"""Constraints, first-class cost models, cost-aware optimisation, capacity, portfolio risk and the mixed-asset demo."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.engine import (CommissionModel, ConstraintInputs, ConstraintSet, CostSchedule, Engine, EngineConfig, FinancingModel, ImpactModel, LiquidityModel, Schedule, SlippageModel, SpreadModel,
                        Strategy, Target, calibrate_sqrt_impact, capacity_curve, capacity_estimate, cost_aware_weights, proportional_costs)
from src.engine.analysis import md_table
from src.engine.risk import PortfolioRisk
from src.instruments import Instrument, InstrumentRegistry, build_chain, fx_spot
from src.marketdata import events_from_prices

warnings.filterwarnings("ignore")
CAP = 1_000_000.0


def inputs(**kw):
    base = dict(capital=CAP, current={}, asset_class={"A": "equity", "B": "equity", "C": "fx"}, currency_exposure={"A": {"USD": 1.0}, "B": {"USD": 1.0}, "C": {"EUR": 1.0, "USD": -1.0}})
    base.update(kw)
    return ConstraintInputs(**base)


# ------------------------------------------------------------------------------------------------------------------------------ constraints
def test_instrument_and_gross_and_net_limits():
    cs = ConstraintSet(max_instrument_weight=0.5, max_gross=1.2, max_net=0.4)
    out, bound = cs.apply({"A": 900_000.0, "B": 900_000.0, "C": -600_000.0}, inputs())
    assert abs(out["A"]) <= 0.5 * CAP + 1e-6 and {"instrument_cap", "gross"} <= set(bound)
    assert sum(abs(v) for v in out.values()) <= 1.2 * CAP + 1e-6
    assert abs(sum(out.values())) <= 0.4 * CAP + 1e-6


def test_asset_class_currency_and_concentration_limits():
    out, bound = ConstraintSet(asset_class_caps={"equity": 0.5}).apply({"A": 400_000.0, "B": 400_000.0, "C": 100_000.0}, inputs())
    assert abs(out["A"]) + abs(out["B"]) <= 0.5 * CAP + 1e-6 and out["C"] == 100_000.0 and "asset_class:equity" in bound
    out, bound = ConstraintSet(currency_caps={"EUR": 0.1}).apply({"A": 100_000.0, "C": 500_000.0}, inputs())
    assert abs(out["C"]) <= 0.1 * CAP + 1e-6 and out["A"] == 100_000.0 and "currency:EUR" in bound
    out, bound = ConstraintSet(max_concentration=0.4).apply({"A": 800_000.0, "B": 100_000.0, "C": 100_000.0}, inputs())
    assert out["A"] <= 0.4 * sum(abs(v) for v in out.values()) + 1e-2 and "concentration" in bound
    assert out["B"] == 100_000.0 and out["C"] == 100_000.0                           # only the offending instrument is cut
    out, bound = ConstraintSet(max_concentration=0.4).apply({"A": 800_000.0, "B": 100_000.0}, inputs())
    assert out["A"] == 800_000.0 and "concentration" not in bound                      # two instruments cannot both be under 40%: infeasible, left alone


def test_hedges_are_not_cut_by_a_net_exposure_limit():
    out, _ = ConstraintSet(max_net=0.05).apply({"A": 600_000.0, "B": -500_000.0}, inputs())
    assert out["B"] == -500_000.0 and out["A"] < 600_000.0 and abs(out["A"] + out["B"]) <= 0.05 * CAP + 1e-6


def test_liquidity_vol_target_drawdown_margin_turnover():
    out, b = ConstraintSet(liquidity_participation=0.1).apply({"A": 500_000.0}, inputs(adv_notional={"A": 1_000_000.0}))
    assert out["A"] == 100_000.0 and "liquidity" in b
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["A", "B"], columns=["A", "B"])
    out, b = ConstraintSet(vol_target=0.05).apply({"A": 500_000.0, "B": 500_000.0}, inputs(cov=cov))
    w = np.array([out["A"], out["B"]]) / CAP
    assert abs(np.sqrt(w @ cov.to_numpy() @ w) - 0.05) < 1e-9 and "vol_target" in b
    out, b = ConstraintSet(drawdown_scaling=((0.1, 0.5), (0.2, 0.0))).apply({"A": 100_000.0}, inputs(drawdown=0.12))
    assert out["A"] == 50_000.0 and "drawdown" in b
    assert ConstraintSet(drawdown_scaling=((0.1, 0.5), (0.2, 0.0))).apply({"A": 100_000.0}, inputs(drawdown=0.25))[0]["A"] == 0.0
    out, b = ConstraintSet(max_margin_utilization=0.5).apply({"A": 1_000_000.0}, inputs(margin_rate={"A": 1.0}))
    assert out["A"] == 500_000.0 and "margin" in b
    out, b = ConstraintSet(max_turnover=0.1).apply({"A": 400_000.0}, inputs(current={"A": 100_000.0}))
    assert abs(out["A"] - 200_000.0) < 1e-6 and "turnover" in b


def test_no_capital_means_no_positions_and_unconstrained_targets_pass_through():
    out, b = ConstraintSet().apply({"A": 1.0}, inputs(capital=0.0))
    assert out["A"] == 0.0 and b == ["no_capital"]
    out, b = ConstraintSet().apply({"A": 123.0}, inputs())
    assert out["A"] == 123.0 and b == []


def test_constraints_bind_inside_a_run_and_are_counted():
    idx = pd.bdate_range("2024-01-02", periods=20)
    px = pd.DataFrame({"A": 100.0 * np.ones(20)}, index=idx)
    reg = InstrumentRegistry([Instrument(instrument_id="A", asset_class="equity", instrument_type="equity", currency="USD", calendar="WEEKDAY")])

    class Greedy(Strategy):
        name = "g"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            ctx.set_targets([Target("A", weight=3.0)])

    cfg = EngineConfig(start=idx[0], end=idx[-1], initial_cash={"USD": CAP})
    res = Engine(reg, events_from_prices(px, "bar", "16:00"), [Greedy()], cfg, CostSchedule().set("equity", spread=SpreadModel(0.0, 0.0)), FinancingModel(),
                 constraints=ConstraintSet(max_gross=1.0)).run()
    assert res.diagnostics["constraint:gross"] >= 10
    assert res.fills["quantity"].sum() * 100.0 <= CAP + 1.0


# ------------------------------------------------------------------------------------------------------------------------------------ costs
def test_cost_is_fixed_fee_plus_spread_plus_impact():
    inst = Instrument(instrument_id="X", asset_class="equity", instrument_type="equity", currency="USD", tick_size=0.01)
    cs = CostSchedule().set("equity", commission=CommissionModel(per_order=1.0, bps=1.0), spread=SpreadModel(fallback_bps=10.0), impact=ImpactModel("sqrt", 0.5), slippage=SlippageModel(2.0))
    c = cs.estimate(inst, 1000.0, 100.0, adv=100_000.0, sigma=0.02)
    assert abs(c.spread_price - 0.05) < 1e-12                                              # half of 10bp of 100
    assert abs(c.impact_price - 0.5 * 0.02 * np.sqrt(0.01) * 100.0) < 1e-12              # Y sigma sqrt(Q/ADV) x price
    assert abs(c.extra_price - 0.02) < 1e-12
    px = 100.0 + c.price_distance
    assert abs(c.fee - (1.0 + 1e-4 * 1000.0 * px)) < 1e-9
    sell = cs.estimate(inst, -1000.0, 100.0, adv=100_000.0, sigma=0.02)
    assert sell.price_distance == c.price_distance                                        # symmetric distance, opposite direction


def test_costs_use_the_observed_spread_and_respect_the_most_specific_model():
    inst = Instrument(instrument_id="X", asset_class="equity", instrument_type="equity", currency="USD", tick_size=0.01)
    cs = CostSchedule().set("equity", spread=SpreadModel(fallback_bps=100.0)).set("X", commission=CommissionModel(per_contract=0.5))
    assert abs(cs.estimate(inst, 10.0, 100.0, bid=99.99, ask=100.01).spread_price - 0.01) < 1e-12        # a quote beats the fallback
    assert abs(cs.estimate(inst, 10.0, 100.0).spread_price - 0.5) < 1e-12
    assert cs.estimate(inst, 10.0, 100.0).fee == 5.0
    assert cs.round_trip_bps(inst, 100.0, 100.0) > 100.0


def test_participation_limit_caps_the_fill_per_event():
    lm = LiquidityModel(max_participation=0.2)
    assert lm.fill_cap(None, 1000.0) == 200.0 and lm.fill_cap(0.5, 1000.0) == 500.0 and lm.fill_cap(None, float("nan")) == float("inf")


def test_roll_costs_are_attributed_to_roll_trades():
    from src.engine.demo import run_mixed_asset_demo

    d = run_mixed_asset_demo(n_days=130, with_swaps=False, risk=False)
    cs = d.result.cost_summary()
    assert cs["roll_costs"] > 0 and cs["roll_costs"] < cs["total_costs"] and cs["total_costs"] > 0


# -------------------------------------------------------------------------------------------------------------------------------- optimisation
def test_cost_aware_weights_create_a_no_trade_region():
    ids = list("ABCD")
    alpha = pd.Series([0.002, -0.001, 0.0005, 0.003], ids)
    cov = pd.DataFrame(np.diag([0.0004, 0.0004, 0.0004, 0.0009]), ids, ids)
    cur = pd.Series([0.3, 0.0, 0.5, 0.0], ids)
    free = cost_aware_weights(alpha, cov, cur, cost=0.0, risk_aversion=5.0)
    costly = cost_aware_weights(alpha, cov, cur, cost=0.02, risk_aversion=5.0)
    assert (free["trade"].abs() > 0.05).any()
    assert costly["trade"].abs().max() < 1e-6                                              # costs exceed any marginal gain: hold what you have
    mid = cost_aware_weights(alpha, cov, cur, cost=0.002, risk_aversion=5.0)
    assert 0 < mid["trade"].abs().sum() < free["trade"].abs().sum()
    assert mid["cost"].sum() > 0


def test_cost_aware_weights_respect_limits_and_convexity_of_impact():
    ids = list("AB")
    alpha = pd.Series([0.05, 0.05], ids)
    cov = pd.DataFrame(np.diag([0.01, 0.01]), ids, ids)
    r = cost_aware_weights(alpha, cov, None, risk_aversion=1.0, max_gross=1.0, max_weight=0.8)
    assert r["w"].abs().sum() <= 1.0 + 1e-6 and r["w"].abs().max() <= 0.8 + 1e-6
    cheap = cost_aware_weights(alpha, cov, None, impact=0.0, risk_aversion=1.0, max_gross=2.0)
    dear = cost_aware_weights(alpha, cov, None, impact=0.5, risk_aversion=1.0, max_gross=2.0)
    assert dear["w"].abs().sum() < cheap["w"].abs().sum()                                  # impact grows faster than linearly: trade less


def test_proportional_costs_by_instrument():
    reg = InstrumentRegistry([Instrument(instrument_id="X", asset_class="equity", instrument_type="equity", currency="USD", tick_size=0.01), fx_spot("EUR", "USD")])
    cs = CostSchedule().set("equity", commission=CommissionModel(bps=2.0), spread=SpreadModel(fallback_bps=4.0, min_half_ticks=0.0)).set("fx", spread=SpreadModel(fallback_bps=1.0, min_half_ticks=0.0))
    pc = proportional_costs(reg, cs, {"X": 50.0, "EURUSD": 1.1})
    assert abs(pc["X"] - (2e-4 + 2e-4)) < 1e-6 and abs(pc["EURUSD"] - 0.5e-4) < 1e-6


def test_impact_calibration_recovers_the_coefficient():
    rng = np.random.default_rng(1)
    n = 400
    adv, q, sig = rng.uniform(1e6, 1e7, n), rng.uniform(1e4, 1e6, n), rng.uniform(0.005, 0.02, n)
    slip = 0.7 * sig * 1e4 * np.sqrt(q / adv) + rng.normal(0, 1.0, n)
    fit = calibrate_sqrt_impact(pd.DataFrame({"slippage_bps": slip, "quantity": q, "adv": adv, "sigma": sig}))
    assert abs(fit["Y"] - 0.7) < 3 * fit["se"] + 0.01 and fit["r2"] > 0.9 and fit["n"] == n
    assert np.isnan(calibrate_sqrt_impact(pd.DataFrame({"slippage_bps": [1.0], "quantity": [1.0], "adv": [1.0], "sigma": [0.01]}))["Y"])


def test_capacity_curve_erodes_with_size_and_estimate_interpolates():
    from src.engine.demo import run_mixed_asset_demo

    d = run_mixed_asset_demo(n_days=130, with_swaps=False, risk=False)
    curve = capacity_curve(d.result, adv_notional=50_000_000.0, sigma_daily=0.01, multiples=(1, 5, 25, 125), registry=d.market.registry)
    assert curve["gross"].is_monotonic_increasing or curve["gross"].iloc[0] < 0
    assert (curve["impact"].diff().dropna() > 0).all()
    assert curve["edge_kept"].iloc[0] > curve["edge_kept"].iloc[-1]
    cap = capacity_estimate(pd.DataFrame({"edge_kept": [0.9, 0.7, 0.4, 0.1]}, index=[1, 2, 4, 8]), 0.5)
    assert 2 < cap < 4 and np.isnan(capacity_estimate(pd.DataFrame({"edge_kept": [0.2, 0.1]}, index=[1, 2]), 0.5))


# -------------------------------------------------------------------------------------------------------------------------------------- risk
def futures_book():
    idx = pd.bdate_range("2023-01-02", periods=150)
    chain = build_chain("ES", "2022-12-01", "2023-08-31", months=(3, 6, 9), multiplier=50.0, tick_size=0.25, calendar="WEEKDAY", initial_margin=10_000.0, maintenance_margin=9_000.0)
    reg = InstrumentRegistry([fx_spot("EUR", "USD")])
    reg.add_chain(chain)
    rng = np.random.default_rng(2)
    base = 4000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 150)))
    prices = pd.DataFrame({c.instrument_id: pd.Series(base, index=idx).where(idx <= c.expiry).round(2) for c in chain.contracts})
    eur = pd.DataFrame({"EURUSD": 1.1 * np.exp(np.cumsum(rng.normal(0, 0.005, 150)))}, index=idx)
    ev = pd.concat([events_from_prices(prices, "settlement", "16:00"), events_from_prices(eur, "bar", "16:00")], ignore_index=True)
    from src.marketdata import normalise_events

    return idx, reg, normalise_events(ev, lag="0s"), prices


def test_portfolio_risk_report_for_a_simple_book():
    idx, reg, ev, prices = futures_book()

    class Book(Strategy):
        name = "b"
        schedule = Schedule("weekly", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            ctx.set_targets([Target("ES", quantity=5.0), Target("EURUSD", notional=500_000.0)])

    cfg = EngineConfig(start=idx[0], end=idx[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 2_000_000.0}, managed_chains=())
    e = Engine(reg, ev, [Book()], cfg, CostSchedule(), FinancingModel(), risk_model=PortfolioRisk(lookback=100, alpha=0.95, scenarios={"es_down": {"returns": {"future": -0.10}}, "usd_up": {"fx_base": ("USD", 0.10)}}))
    res = e.run()
    rep = PortfolioRisk(lookback=100, alpha=0.95, scenarios={"es_down": {"returns": {"future": -0.10}}, "usd_up": {"fx_base": ("USD", 0.10)}}).report(e)
    assert rep.gross > 0 and abs(rep.net) <= rep.gross + 1e-6 and rep.equity == e.ledger.equity()
    assert rep.var > 0 and rep.cvar >= rep.var and rep.vol_annual > 0
    es_notional = abs(rep.positions.loc[[i for i in rep.positions.index if i.startswith("ES")][0], "delta_base"])
    assert abs(rep.scenarios["es_down"] + 0.10 * es_notional) < 1e-6 * es_notional           # a linear future: scenario P&L is delta x shock
    assert rep.scenarios["usd_up"] < 0                                                          # long EUR loses when the dollar rises
    assert rep.tail_contribution.index.isin(rep.positions.index).all() and abs(rep.tail_contribution.sum()) > 0
    assert "leverage" in rep.summary().index and res.reconciliation.ok


def test_risk_report_with_no_positions_is_empty_not_an_error():
    idx, reg, ev, _ = futures_book()
    cfg = EngineConfig(start=idx[0], end=idx[-1], initial_cash={"USD": 1_000_000.0})
    e = Engine(reg, ev, [], cfg, CostSchedule(), FinancingModel())
    e.run()
    rep = PortfolioRisk().report(e)
    assert rep.gross == 0.0 and rep.var == 0.0 or np.isnan(rep.var)


def test_md_table_renders_without_optional_dependencies():
    t = md_table(pd.DataFrame({"a": [1.5, 2.25], "b": ["x", "y"]}, index=pd.Index(["r1", "r2"], name="row")))
    lines = t.splitlines()
    assert lines[0] == "| row | a | b |" and lines[1] == "|---|---|---|" and lines[2] == "| r1 | 1.50 | x |"


# ----------------------------------------------------------------------------------------------------------------------------------- the demo
def test_the_mixed_asset_demo_meets_the_success_criteria():
    from src.engine.demo import run_mixed_asset_demo

    d = run_mixed_asset_demo(n_days=220, seed=2, with_swaps=True)
    r = d.result
    assert r.reconciliation.ok, r.reconciliation.failures()                                       # full cash and position reconciliation
    classes = set(r.attribution("asset_class").index)
    assert {"fx", "future", "crypto", "option", "swap", "commodity"} <= classes | {"future"}       # one portfolio, many asset classes
    assert {"crypto", "option", "swap"} <= classes
    cats = set(r.attribution("category").index)
    assert {"funding", "variation_margin", "spread", "fee"} <= cats                                # funding accrual, futures settlement, realistic costs
    assert r.diagnostics.get("rolls", 0) > 0
    pm = r.pnl_matrix("asset_class", "category")
    assert abs(pm.to_numpy().sum() - r.summary()["total_pnl"]) < 1e-3                              # asset-class attribution adds up
    assert r.last_risk is not None and r.last_risk.gross > 0
    txt = d.report()
    assert "## P&L by asset class and category" in txt and "reconciled: True" in txt


def test_the_demo_is_deterministic():
    from src.engine.demo import run_mixed_asset_demo

    a = run_mixed_asset_demo(n_days=110, seed=5, with_swaps=False, risk=False)
    b = run_mixed_asset_demo(n_days=110, seed=5, with_swaps=False, risk=False)
    c = run_mixed_asset_demo(n_days=110, seed=6, with_swaps=False, risk=False)
    assert a.result.digest == b.result.digest and a.result.digest != c.result.digest
