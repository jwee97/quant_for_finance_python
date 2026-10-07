"""Contract lifecycle: option expiry, exercise and assignment (physical, cash-settled, on futures), early exercise, futures delivery, with analytic expected P&L."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.engine import CommissionModel, CostSchedule, Engine, EngineConfig, FinancingModel, Schedule, SpreadModel, Strategy, Target
from src.instruments import Instrument, InstrumentRegistry, RollSpec, build_chain, make_option
from src.marketdata import events_from_prices

warnings.filterwarnings("ignore")
ZERO = CostSchedule().set("equity", spread=SpreadModel(0.0, 0.0)).set("option", spread=SpreadModel(0.0, 0.0)).set("future", spread=SpreadModel(0.0, 0.0)).set("index", spread=SpreadModel(0.0, 0.0))
START = 1_000_000.0


def market(path, underlying="SPY", extra=(), kind="equity"):
    idx = pd.bdate_range("2024-03-04", periods=len(path))
    reg = InstrumentRegistry([Instrument(instrument_id=underlying, asset_class=kind, instrument_type=kind, currency="USD", calendar="WEEKDAY")])
    for o in extra:
        reg.add(o)
    return idx, reg, pd.DataFrame({underlying: np.asarray(path, float)}, index=idx)


class Plan(Strategy):
    """Trades once on the first day, optionally exercises on a given day."""

    name = "plan"
    schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

    def __init__(self, idx, trades, exercise=None):
        self.idx, self.trades, self.exercise = idx, trades, exercise

    def on_schedule(self, ctx):
        d = ctx.ts.normalize()
        if d == self.idx[0]:
            ctx.set_targets([Target(i, quantity=q) for i, q in self.trades.items()])
        if self.exercise and d == self.idx[self.exercise[0]]:
            ctx.exercise(self.exercise[1])


def run(idx, reg, prices, strat, quotes=None, **cfgkw):
    frames = [events_from_prices(prices, "bar", "16:00")]
    if quotes is not None:
        frames.append(events_from_prices(quotes, "bar", "16:00"))
    from src.marketdata import concat_events

    cfg = EngineConfig(start=idx[0], end=idx[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": START}, **cfgkw)
    e = Engine(reg, concat_events(*frames), [strat], cfg, ZERO, FinancingModel())
    return e, e.run()


def flat_quotes(idx, opt, value):
    s = pd.Series(value, index=idx)
    return pd.DataFrame({opt.instrument_id: s.where(idx <= opt.expiry)})


def test_long_call_in_the_money_is_exercised_into_shares():
    expiry = pd.Timestamp("2024-03-15")
    call = make_option("SPY", expiry, "call", 100.0)
    path = np.r_[np.full(5, 100.0), np.linspace(101, 110, 10)]
    idx, reg, px = market(path, extra=[call])
    premium = 3.0
    e, res = run(idx, reg, px, Plan(idx, {call.instrument_id: 2.0}), flat_quotes(idx, call, premium))
    assert res.reconciliation.ok, res.reconciliation.failures()
    ST = float(px.loc[expiry, "SPY"])
    assert e.ledger.quantity(call.instrument_id) == 0.0 and abs(e.ledger.quantity("SPY") - 200.0) < 1e-9          # 2 contracts x 100 shares delivered
    last = float(px["SPY"].iloc[-1])
    expected = 200.0 * (last - 100.0) - 2 * 100 * premium                                                          # shares bought at the strike, marked at the last price; premium sunk
    assert abs(res.equity_curve.iloc[-1] - START - expected) < 1e-6
    assert any(ev.kind == "exercise" for ev in res.events)
    assert ST > 100.0


def test_out_of_the_money_option_expires_worthless_and_costs_the_premium():
    expiry = pd.Timestamp("2024-03-15")
    put = make_option("SPY", expiry, "put", 95.0)
    idx, reg, px = market(np.full(15, 100.0), extra=[put])
    e, res = run(idx, reg, px, Plan(idx, {put.instrument_id: 3.0}), flat_quotes(idx, put, 1.5))
    assert res.reconciliation.ok and e.ledger.quantity(put.instrument_id) == 0.0 and e.ledger.quantity("SPY") == 0.0
    assert abs(res.equity_curve.iloc[-1] - START + 3 * 100 * 1.5) < 1e-6
    assert any(ev.kind == "expiry" for ev in res.events)


def test_short_put_in_the_money_is_assigned_and_the_loss_is_the_intrinsic_value():
    expiry = pd.Timestamp("2024-03-15")
    put = make_option("SPY", expiry, "put", 100.0)
    path = np.r_[np.full(5, 100.0), np.linspace(99, 90, 10)]
    idx, reg, px = market(path, extra=[put])
    premium = 2.0
    e, res = run(idx, reg, px, Plan(idx, {put.instrument_id: -1.0}), flat_quotes(idx, put, premium))
    assert res.reconciliation.ok, res.reconciliation.failures()
    assert abs(e.ledger.quantity("SPY") - 100.0) < 1e-9                                                            # assigned: long 100 shares at 100
    last = float(px["SPY"].iloc[-1])
    expected = 100.0 * (last - 100.0) + 100 * premium
    assert abs(res.equity_curve.iloc[-1] - START - expected) < 1e-6
    assert any(ev.kind == "assignment" for ev in res.events)


def test_cash_settled_index_option_pays_the_intrinsic_value_in_cash():
    expiry = pd.Timestamp("2024-03-15")
    call = make_option("SPX", expiry, "call", 4000.0, underlying_kind="index", multiplier=100.0)
    path = np.r_[np.full(5, 4000.0), np.linspace(4010, 4100, 10)]
    idx, reg, px = market(path, "SPX", [call], kind="equity")
    e, res = run(idx, reg, px, Plan(idx, {call.instrument_id: 1.0}), flat_quotes(idx, call, 20.0))
    assert res.reconciliation.ok and e.ledger.quantity("SPX") == 0.0                                               # nothing delivered
    ST = float(px.loc[expiry, "SPX"])
    assert abs(res.equity_curve.iloc[-1] - START - (100.0 * (ST - 4000.0) - 100.0 * 20.0)) < 1e-6


def test_american_option_can_be_exercised_early_on_request():
    expiry = pd.Timestamp("2024-03-29")
    put = make_option("SPY", expiry, "put", 100.0, exercise_style="american")
    path = np.r_[np.full(3, 100.0), np.full(17, 90.0)]
    idx, reg, px = market(path, extra=[put])
    e, res = run(idx, reg, px, Plan(idx, {put.instrument_id: 1.0}, exercise=(6, put.instrument_id)), flat_quotes(idx, put, 10.5))
    assert res.reconciliation.ok, res.reconciliation.failures()
    assert e.ledger.quantity(put.instrument_id) == 0.0 and abs(e.ledger.quantity("SPY") + 100.0) < 1e-9          # exercised: short 100 shares at 100
    ex = [ev for ev in res.events if ev.kind == "exercise"]
    assert ex and ex[0].ts < expiry
    expected = 100.0 * (100.0 - 90.0) - 100 * 10.5
    assert abs(res.equity_curve.iloc[-1] - START - expected - 100.0 * (90.0 - 90.0)) < 1e-6


def test_european_option_cannot_be_exercised_early():
    expiry = pd.Timestamp("2024-03-29")
    put = make_option("SPY", expiry, "put", 100.0)
    idx, reg, px = market(np.full(20, 90.0), extra=[put])
    e, res = run(idx, reg, px, Plan(idx, {put.instrument_id: 1.0}, exercise=(5, put.instrument_id)), flat_quotes(idx, put, 10.0))
    early = [ev for ev in res.events if ev.kind == "exercise" and ev.ts < expiry]
    assert not early and res.reconciliation.ok


def test_option_on_a_future_delivers_a_futures_position_at_the_strike():
    chain = build_chain("ES", "2024-01-01", "2024-06-30", months=(3, 6), multiplier=50.0, tick_size=0.25, calendar="WEEKDAY", roll=RollSpec("calendar", 0))
    fut = chain.contracts[1]                                                     # the June contract: still alive when the option expires
    expiry = pd.Timestamp("2024-03-15")
    call = make_option(fut.instrument_id, expiry, "call", 4000.0, underlying_kind="future", multiplier=50.0)
    idx = pd.bdate_range("2024-03-04", periods=15)
    F = pd.Series(np.r_[np.full(5, 4000.0), np.linspace(4010, 4100, 10)], index=idx)
    reg = InstrumentRegistry()
    reg.add_chain(chain)
    reg.add(call)
    prices = pd.DataFrame({fut.instrument_id: F})
    e, res = run(idx, reg, prices, Plan(idx, {call.instrument_id: 1.0}), flat_quotes(idx, call, 30.0), managed_chains=())
    assert res.reconciliation.ok, res.reconciliation.failures()
    assert abs(e.ledger.quantity(fut.instrument_id) - 1.0) < 1e-9                  # exercised into one future at the strike
    last = float(F.iloc[-1])
    assert abs(res.equity_curve.iloc[-1] - START - (50.0 * (last - 4000.0) - 50.0 * 30.0)) < 1e-6


def test_every_lifecycle_run_conserves_cash_and_positions_with_costs_on():
    expiry = pd.Timestamp("2024-03-15")
    calls = [make_option("SPY", expiry, "call", k) for k in (95.0, 100.0, 105.0)]
    path = np.r_[np.full(5, 100.0), np.linspace(101, 108, 10)]
    idx, reg, px = market(path, extra=calls)
    quotes = pd.concat([flat_quotes(idx, c, 4.0) for c in calls], axis=1)
    from src.marketdata import concat_events

    cfg = EngineConfig(start=idx[0], end=idx[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": START})
    costs = CostSchedule().set("option", commission=CommissionModel(per_contract=0.65)).set("equity", commission=CommissionModel(bps=1.0))
    plan = Plan(idx, {calls[0].instrument_id: 3.0, calls[1].instrument_id: -2.0, calls[2].instrument_id: 4.0})
    res = Engine(reg, concat_events(events_from_prices(px, "bar", "16:00", spread_bps=2.0), events_from_prices(quotes, "bar", "16:00", spread_bps=100.0)), [plan], cfg, costs,
                 FinancingModel(deposit_rates={"USD": 0.03})).run()
    assert res.reconciliation.ok, res.reconciliation.failures()
    assert not res.reconciliation.expired_held
