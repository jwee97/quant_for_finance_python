"""The event-driven engine: independent P&L recomputation, conservation, determinism, no look-ahead, orders and fills, futures rolls, perpetual funding and liquidation, replay parity."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.engine import (CommissionModel, CostSchedule, Engine, EngineConfig, FinancingModel, LiquidityModel, Order, PaperSession, ReplayFeed, Schedule, SpreadModel, Strategy,
                        Target, run_session)
from src.instruments import Instrument, InstrumentRegistry, RollSpec, build_chain, crypto_perp, crypto_spot, fx_spot
from src.marketdata import concat_events, events_from_funding, events_from_prices

warnings.filterwarnings("ignore")
ZERO = CostSchedule().set("equity", spread=SpreadModel(0.0, 0.0)).set("future", spread=SpreadModel(0.0, 0.0)).set("fx", spread=SpreadModel(0.0, 0.0)).set("crypto", spread=SpreadModel(0.0, 0.0))


def cfg(index, **kw):
    base = dict(start=index[0], end=index[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 1_000_000.0}, pegs={"USDT": 1.0})
    base.update(kw)
    return EngineConfig(**base)


def stock_market(n=40, seed=0, drift=0.0, flat=False):
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(seed)
    px = pd.DataFrame({"AAA": 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), "BBB": 50.0 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))}, index=idx)
    if flat:
        px[:] = 100.0
    reg = InstrumentRegistry([Instrument(instrument_id=i, asset_class="equity", instrument_type="equity", currency="USD", calendar="WEEKDAY") for i in px.columns])
    return idx, px, reg


class Trader(Strategy):
    """Trades pseudo-randomly every day (to stress the accounting, not to make money)."""

    name = "random"
    schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

    def __init__(self, ids, seed=1, scale=200):
        self.ids, self.rng, self.scale = ids, np.random.default_rng(seed), scale

    def on_schedule(self, ctx):
        ctx.set_targets([Target(i, quantity=float(self.rng.integers(-self.scale, self.scale))) for i in self.ids])


def run(reg, events, strategies, config, costs=None, financing=None, **kw):
    return Engine(reg, events, strategies, config, costs or ZERO, financing or FinancingModel(), **kw).run()


# -------------------------------------------------------------------------------------------------------------- conservation and no self-generated P&L
def test_flat_prices_and_no_costs_mean_no_pnl_whatever_the_trading():
    idx, px, reg = stock_market(flat=True)
    res = run(reg, events_from_prices(px, "bar", "16:00"), [Trader(list(px.columns))], cfg(idx))
    assert res.reconciliation.ok, res.reconciliation.failures()
    assert len(res.fills) > 20
    assert abs(res.equity_curve.iloc[-1] - 1_000_000.0) < 1e-6                      # a ledger cannot make money out of trading at a constant price


def test_costs_are_the_only_thing_that_reduces_equity_at_constant_prices():
    idx, px, reg = stock_market(flat=True)
    costs = CostSchedule().set("equity", commission=CommissionModel(bps=2.0), spread=SpreadModel(fallback_bps=10.0, min_half_ticks=0.0))
    res = run(reg, events_from_prices(px, "bar", "16:00"), [Trader(list(px.columns))], cfg(idx), costs)
    traded = float((res.fills["quantity"].abs() * res.fills["price"]).sum())
    expected = traded * 0.5 * 10e-4 + traded * 2e-4                                         # half the 10bp spread plus 2bp commission on every dollar traded (price ~ mid)
    assert res.reconciliation.ok
    assert abs((1_000_000.0 - res.equity_curve.iloc[-1]) - expected) < 0.01 * expected
    cs = res.cost_summary()
    assert cs["spread"] > 0 and cs["fee"] > 0 and abs(cs["total_costs"] - (1_000_000.0 - res.equity_curve.iloc[-1])) < 1e-4


def test_attribution_adds_up_to_total_pnl_and_categories_are_exhaustive():
    idx, px, reg = stock_market()
    costs = CostSchedule().set("equity", commission=CommissionModel(bps=1.0), spread=SpreadModel(fallback_bps=4.0))
    res = run(reg, events_from_prices(px, "bar", "16:00"), [Trader(list(px.columns))], cfg(idx), costs, FinancingModel(deposit_rates={"USD": 0.04}))
    for key in ("category", "instrument_id", "asset_class", "strategy"):
        assert abs(res.attribution(key).sum() - res.summary()["total_pnl"]) < 1e-4, key


def test_buy_and_hold_futures_through_rolls_matches_an_independent_recomputation():
    idx = pd.bdate_range("2024-01-02", "2024-08-30")
    chain = build_chain("ES", "2024-01-01", "2024-08-31", months=(3, 6, 9, 12), multiplier=50.0, tick_size=0.25, roll=RollSpec("calendar", 5), calendar="WEEKDAY")
    reg = InstrumentRegistry()
    reg.add_chain(chain)
    rng = np.random.default_rng(5)
    base = 4500.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, len(idx))))
    cols = {}
    for c in chain.contracts:
        T = np.maximum((c.expiry - idx).days, 0) / 365.0
        s = pd.Series(base * (1.0 + 0.03 * T), index=idx)
        cols[c.instrument_id] = s.where(idx <= c.expiry).round(2)                            # prices on the tick grid, none after expiry
    prices = pd.DataFrame(cols)

    class Hold(Strategy):
        name = "hold"
        schedule = Schedule("weekly", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            ctx.set_targets([Target("ES", quantity=3.0)])

    res = run(reg, events_from_prices(prices, "settlement", "16:00"), [Hold()], cfg(idx))
    assert res.reconciliation.ok, res.reconciliation.failures()
    # independent recomputation: position per contract by day from the fills, pnl = position(d-1) x multiplier x price change
    f = res.fills.copy()
    f["day"] = pd.to_datetime(f["ts"]).dt.normalize()
    pos = f.pivot_table(index="day", columns="instrument_id", values="quantity", aggfunc="sum", fill_value=0.0).reindex(idx, fill_value=0.0).cumsum()
    pnl = 0.0
    for iid in pos.columns:
        p = prices[iid].ffill()
        pnl += float((pos[iid].shift(1).fillna(0.0) * 50.0 * p.diff().fillna(0.0)).where(pos[iid].shift(1).fillna(0.0) != 0).fillna(0.0).sum())
    assert abs(res.attribution("category").get("variation_margin", 0.0) - pnl) < 1e-6 * max(1.0, abs(pnl)) + 1e-6
    assert res.diagnostics.get("rolls", 0) >= 2 and not res.reconciliation.expired_held
    assert set(f.loc[f["tags"].str.contains("roll"), "instrument_id"]) <= set(prices.columns)


# --------------------------------------------------------------------------------------------------------------------------- determinism and leakage
def test_replay_is_deterministic_and_sensitive_to_data():
    idx, px, reg = stock_market()
    ev = events_from_prices(px, "bar", "16:00", spread_bps=2.0)
    a = run(reg, ev, [Trader(list(px.columns))], cfg(idx))
    b = run(reg, ev, [Trader(list(px.columns))], cfg(idx))
    assert a.digest == b.digest and a.equity_curve.equals(b.equity_curve)
    px2 = px.copy()
    px2.iloc[-3, 0] *= 1.01
    c = run(reg, events_from_prices(px2, "bar", "16:00", spread_bps=2.0), [Trader(list(px.columns))], cfg(idx))
    assert c.digest != a.digest


class MomentumLike(Strategy):
    name = "mom"
    schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

    def on_schedule(self, ctx):
        r = ctx.data.returns("AAA", 5)
        ctx.set_targets([Target("AAA", quantity=float(np.sign(r.mean()) * 100.0))] if len(r) else [])


def test_changing_the_future_does_not_change_the_past():
    idx, px, reg = stock_market(60)
    cut = idx[39]
    ev = events_from_prices(px, "bar", "16:00", spread_bps=2.0)
    a = run(reg, ev, [MomentumLike()], cfg(idx))
    px2 = px.copy()
    px2.loc[px2.index > cut] *= np.random.default_rng(9).uniform(0.7, 1.3, size=(int((px2.index > cut).sum()), px2.shape[1]))
    b = run(reg, events_from_prices(px2, "bar", "16:00", spread_bps=2.0), [MomentumLike()], cfg(idx))
    before = a.equity_curve.index <= cut + pd.Timedelta(hours=23)
    assert before.sum() > 30
    np.testing.assert_array_equal(a.equity_curve[before].to_numpy(), b.equity_curve[b.equity_curve.index <= cut + pd.Timedelta(hours=23)].to_numpy())
    fa, fb = a.fills[a.fills["ts"] <= cut + pd.Timedelta(hours=23)], b.fills[b.fills["ts"] <= cut + pd.Timedelta(hours=23)]
    assert fa.reset_index(drop=True).equals(fb.reset_index(drop=True))


def test_the_store_audit_sees_no_lookahead_in_a_full_run():
    idx, px, reg = stock_market()
    e = Engine(reg, events_from_prices(px, "bar", "16:00", lag="45min", spread_bps=2.0), [MomentumLike()], cfg(idx), ZERO)
    e.run()
    e.store.assert_no_lookahead()
    assert e.store.queries > 100 and e.store.max_available_returned <= e.end


def test_data_published_after_the_decision_is_not_used_for_it():
    idx, px, reg = stock_market(10)
    seen = []

    class Peek(Strategy):
        name = "peek"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            p = ctx.data.price("AAA")
            seen.append((ctx.ts, None if p is None else p.timestamp))

    # data stamped 16:00 but published at 18:00: invisible to the 16:30 decision of the same day
    run(reg, events_from_prices(px, "bar", "16:00", lag="2h"), [Peek()], cfg(idx))
    assert seen[0][1] is None and len(seen) > 5                                          # nothing is visible at the first decision
    assert all(obs_ts.normalize() < ts.normalize() for ts, obs_ts in seen[1:])


# ---------------------------------------------------------------------------------------------------------------------------------- orders
def test_market_order_fills_at_the_next_event_not_at_the_decision_price():
    idx, px, reg = stock_market(5)
    ev = events_from_prices(px, "bar", "16:00")

    class Once(Strategy):
        name = "once"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[1]:
                ctx.buy("AAA", 10)

    res = run(reg, ev, [Once()], cfg(idx))
    f = res.fills.iloc[0]
    assert pd.Timestamp(f["ts"]) == idx[2] + pd.Timedelta(hours=16) and abs(f["price"] - px["AAA"].iloc[2]) < 1e-9


def test_limit_and_stop_orders_rest_until_the_market_reaches_them():
    idx = pd.bdate_range("2024-01-02", periods=6)
    px = pd.DataFrame({"AAA": [100.0, 100.0, 98.0, 96.0, 104.0, 105.0]}, index=idx)
    reg = InstrumentRegistry([Instrument(instrument_id="AAA", asset_class="equity", instrument_type="equity", currency="USD", calendar="WEEKDAY")])
    ev = events_from_prices(px, "bar", "16:00", spread_bps=2.0)

    class Orders(Strategy):
        name = "o"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[0]:
                self.limit = ctx.submit(Order("AAA", 10, "limit", limit_price=97.0))
                self.stop = ctx.submit(Order("AAA", 5, "stop", stop_price=103.0))

    s = Orders()
    run(reg, ev, [s], cfg(idx))
    assert s.limit.status == "filled" and s.limit.avg_price <= 97.0 + 1e-9               # filled only once the ask reached 97 (day 4, 96)
    assert s.stop.status == "filled" and s.stop.avg_price >= 103.0 - 1e-9


def test_rejections_are_explained_not_silent():
    idx, px, reg = stock_market(5)
    orders = []

    class Bad(Strategy):
        name = "bad"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[0]:
                orders.extend([ctx.order("NOPE", 1), ctx.order("AAA", 0.4), ctx.order("AAA", 5, tags=("x",)), ctx.submit(Order("AAA", 5, "limit"))])

    res = run(reg, events_from_prices(px, "bar", "16:00"), [Bad()], cfg(idx))
    assert [o.status for o in orders][:2] == ["rejected", "rejected"] and "unknown" in orders[0].reason and "lot" in orders[1].reason
    assert orders[3].status == "rejected" and "limit_price" in orders[3].reason
    assert res.diagnostics["orders_rejected"] >= 3


def test_participation_cap_splits_a_large_order_over_several_events():
    idx, px, reg = stock_market(10, flat=True)
    vol = pd.DataFrame(1000.0, index=idx, columns=px.columns)
    ev = events_from_prices(px, "bar", "16:00", volume=vol)

    class Big(Strategy):
        name = "big"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[0]:
                self.o = ctx.buy("AAA", 1500)

    s = Big()
    res = run(reg, ev, [s], cfg(idx), liquidity=LiquidityModel(max_participation=0.25))
    fills = res.fills[res.fills["instrument_id"] == "AAA"]
    assert len(fills) >= 6 and (fills["quantity"] <= 250 + 1e-9).all() and abs(fills["quantity"].sum() - 1500) < 1e-9 and s.o.status == "filled"
    assert res.reconciliation.ok


def test_reduce_only_is_clipped_to_the_position_and_cannot_open_one():
    idx, px, reg = stock_market(8)
    got = []

    class R(Strategy):
        name = "r"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            d = ctx.ts.normalize()
            if d == idx[0]:
                ctx.buy("AAA", 10)
            if d == idx[3]:
                got.append(ctx.submit(Order("AAA", -20, reduce_only=True)))           # larger than the position: clipped, never a flip
            if d == idx[6]:
                got.append(ctx.submit(Order("AAA", -5, reduce_only=True)))            # nothing left to reduce

    res = run(reg, events_from_prices(px, "bar", "16:00"), [R()], cfg(idx))
    assert got[0].quantity == -10.0 and got[0].status == "filled"
    assert got[1].status == "rejected" and "reduce" in got[1].reason
    assert res.fills["quantity"].sum() == 0.0


# ------------------------------------------------------------------------------------------------------------------------------ lifecycle
def future_market(roll_methods=True):
    idx = pd.bdate_range("2024-02-01", "2024-04-30")
    chain = build_chain("ES", "2024-01-01", "2024-04-30", months=(3, 6), multiplier=50.0, tick_size=0.25, calendar="WEEKDAY", roll=RollSpec("calendar", 3))
    reg = InstrumentRegistry()
    reg.add_chain(chain)
    rng = np.random.default_rng(2)
    base = 4500.0 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
    cols = {c.instrument_id: pd.Series(base, index=idx).where(idx <= c.expiry).round(2) for c in chain.contracts}
    return idx, chain, reg, pd.DataFrame(cols)


def test_unrolled_future_expires_with_cash_settlement_and_is_not_held_afterwards():
    idx, chain, reg, prices = future_market()

    class Buy(Strategy):
        name = "b"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[0]:
                ctx.buy(chain.contracts[0].instrument_id, 2)

    e = Engine(reg, events_from_prices(prices, "settlement", "16:00"), [Buy()], cfg(idx, managed_chains=()), ZERO)
    res = e.run()
    h = chain.contracts[0]
    assert res.reconciliation.ok and e.ledger.quantity(h.instrument_id) == 0.0
    assert any(ev.kind == "expiry" and ev.instrument_id == h.instrument_id for ev in res.events)
    assert abs(res.equity_curve.iloc[-1] - 1_000_000.0 - 2 * 50.0 * (prices[h.instrument_id].dropna().iloc[-1] - prices[h.instrument_id].dropna().iloc[1])) < 1e-6


def test_roll_manager_rolls_chain_targets_and_tags_the_trades():
    idx, chain, reg, prices = future_market()

    class Chain(Strategy):
        name = "c"
        schedule = Schedule("weekly", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            ctx.set_targets([Target("ES", quantity=2.0)])

    res = run(reg, events_from_prices(prices, "settlement", "16:00"), [Chain()], cfg(idx))
    rolls = res.fills[res.fills["tags"].str.contains("roll")]
    assert len(rolls) >= 2 and res.diagnostics.get("rolls", 0) >= 1
    first, second = chain.contracts[0].instrument_id, chain.contracts[1].instrument_id
    assert set(rolls["instrument_id"]) == {first, second}
    assert res.reconciliation.ok and abs(res.cost_summary()["roll_costs"]) >= 0


def perp_market(prices, rate=0.0001, kind="linear"):
    idx = pd.bdate_range("2024-01-02", periods=len(prices))
    perp = crypto_perp("BINANCE", "BTC", "USDT", margin_mode="isolated") if kind == "linear" else crypto_perp("DERIBIT", "BTC", "USD", contract_type="inverse")
    spot = crypto_spot("BINANCE", "BTC", "USDT")
    reg = InstrumentRegistry([perp, spot, fx_spot("USDT", "USD", instrument_id="USDTUSD", lot_size=1.0)])
    ev = [events_from_prices(pd.DataFrame({perp.instrument_id: prices, "USDTUSD": np.ones(len(prices))}, index=idx), "bar", "16:00")]
    ft = pd.date_range(idx[0] + pd.Timedelta(hours=8), idx[-1] + pd.Timedelta(hours=23), freq="8h")
    ev.append(events_from_funding(pd.DataFrame({perp.instrument_id: np.full(len(ft), rate)}, index=ft)))
    return idx, perp, reg, concat_events(*ev), ft


def test_perp_funding_accrues_every_interval_with_the_right_sign():
    prices = np.full(10, 30_000.0)
    idx, perp, reg, ev, ft = perp_market(prices, rate=0.0001)

    class Short(Strategy):
        name = "s"
        schedule = Schedule("daily", "16:30", 4, "24x7")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[0]:
                ctx.sell(perp.instrument_id, 1.0)

    res = run(reg, ev, [Short()], cfg(idx, initial_cash={"USDT": 100_000.0, "USD": 0.0}))
    funding = res.attribution("category").get("funding", 0.0)
    paid = [e for e in res.events if e.kind == "funding"]
    expected = sum(perp.funding_payment(e.details["quantity"], e.details["mark"], e.details["rate"]) for e in paid)
    assert len(paid) >= 20 and funding > 0 and abs(funding - expected) < 1e-6          # shorts RECEIVE when the rate is positive
    assert res.reconciliation.ok


def test_isolated_perp_is_liquidated_when_the_mark_reaches_the_liquidation_price():
    prices = np.array([30_000, 30_000, 31_000, 33_000, 36_000, 40_000, 41_000, 41_000], float)         # a squeeze against a short
    idx, perp, reg, ev, _ = perp_market(prices, rate=0.0)

    class Short(Strategy):
        name = "s"
        schedule = Schedule("daily", "16:30", 4, "24x7")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[0]:
                ctx.sell(perp.instrument_id, 5.0)

    e = Engine(reg, ev, [Short()], cfg(idx, initial_cash={"USDT": 100_000.0, "USD": 0.0}, isolated_leverage=5.0), ZERO)
    res = e.run()
    assert res.diagnostics.get("liquidation_fills", 0) >= 1 and abs(e.ledger.quantity(perp.instrument_id)) < 5.0
    assert any(ev_.kind == "liquidation" for ev_ in res.events) and res.reconciliation.ok


def test_inverse_perp_pnl_is_in_the_coin():
    prices = np.array([20_000, 20_000, 25_000, 25_000, 25_000], float)
    idx, perp, reg, ev, _ = perp_market(prices, rate=0.0, kind="inverse")
    reg.add(fx_spot("BTC", "USD"))
    ev = concat_events(ev, events_from_prices(pd.DataFrame({"BTCUSD": prices}, index=idx), "bar", "16:00"))

    class Long(Strategy):
        name = "l"
        schedule = Schedule("daily", "16:30", 4, "24x7")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[0]:
                ctx.buy(perp.instrument_id, 1000.0)

    res = run(reg, ev, [Long()], cfg(idx, initial_cash={"BTC": 10.0, "USD": 0.0}, pegs={"BTC": 20_000.0}, liquidate=False, margin_check=False))
    assert res.reconciliation.ok
    expected_btc = perp.pnl(20_000.0, 25_000.0, 1000.0)
    assert expected_btc > 0 and abs(res.attribution("category").get("variation_margin", 0.0) - expected_btc * 25_000.0) < 1e-3 * expected_btc * 25_000.0


def test_interest_accrues_on_cash_and_borrow_on_negative_balances():
    idx, px, reg = stock_market(30, flat=True)
    res = run(reg, events_from_prices(px, "bar", "16:00"), [], cfg(idx), financing=FinancingModel(deposit_rates={"USD": 0.05}))
    days = (idx[-1] + pd.Timedelta(hours=23, minutes=59) - idx[0]).total_seconds() / 86400.0
    assert abs(res.attribution("category")["interest"] - 1_000_000.0 * 0.05 * days / 365.0) < 0.02 * 1_000_000.0 * 0.05 * days / 365.0


# ---------------------------------------------------------------------------------------------------------------------------- strategy API and paper
def test_strategy_hooks_are_called_with_the_right_events():
    prices = np.full(8, 30_000.0)
    idx, perp, reg, ev, _ = perp_market(prices, rate=0.0001)
    log = {"start": 0, "end": 0, "market": 0, "fills": 0, "funding": 0, "risk": 0}

    class Hooks(Strategy):
        name = "h"
        schedule = Schedule("daily", "16:30", 4, "24x7")
        subscriptions = (perp.instrument_id,)

        def on_start(self, ctx):
            log["start"] += 1

        def on_end(self, ctx):
            log["end"] += 1

        def on_market(self, ctx, obs):
            log["market"] += 1

        def on_portfolio_update(self, ctx, fill):
            log["fills"] += 1

        def on_instrument_event(self, ctx, event):
            log["funding"] += event.kind == "funding"

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[0]:
                ctx.buy(perp.instrument_id, 1.0)

    run(reg, ev, [Hooks()], cfg(idx, initial_cash={"USDT": 100_000.0, "USD": 0.0}))
    assert log["start"] == 1 and log["end"] == 1 and log["market"] >= 8 and log["fills"] == 1 and log["funding"] >= 10


def test_default_map_to_targets_treats_signals_as_weights():
    from src.engine import Signal

    idx, px, reg = stock_market(10)

    class Sig(Strategy):
        name = "sig"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def generate_signals(self, ctx):
            return [Signal("AAA", 0.5, 0.5)] if ctx.ts.normalize() == idx[1] else []

    res = run(reg, events_from_prices(px, "bar", "16:00"), [Sig()], cfg(idx))
    q = res.fills.iloc[0]["quantity"]
    assert abs(q * px["AAA"].iloc[1] - 0.25 * 1_000_000.0) < 0.01 * 1_000_000.0           # weight 0.5 x confidence 0.5 of the capital


def test_paper_session_fed_the_same_events_reproduces_the_backtest_digest():
    idx, px, reg = stock_market(40)
    ev = events_from_prices(px, "bar", "16:00", spread_bps=2.0)
    c = cfg(idx, session_days=idx)
    a = Engine(reg, ev, [MomentumLike()], c, ZERO).run()
    c2 = cfg(idx, session_days=idx)
    sess = PaperSession(reg, [MomentumLike()], c2, costs=ZERO)
    run_session(sess, ReplayFeed(ev, c2.start, c2.end, "1D"))
    b = sess.result()
    assert a.digest == b.digest and abs(a.equity_curve.iloc[-1] - b.equity_curve.iloc[-1]) < 1e-9 and b.reconciliation.ok


def test_paper_broker_submits_manual_orders_and_reports_positions():
    idx, px, reg = stock_market(10)
    ev = events_from_prices(px, "bar", "16:00", spread_bps=2.0)
    c = cfg(idx, session_days=idx)
    sess = PaperSession(reg, [], c, costs=ZERO)
    feed = iter(ReplayFeed(ev, c.start, c.end, "1D"))
    until, batch = next(feed)
    sess.feed(batch)
    sess.advance(until)
    o = sess.broker.submit("AAA", 25)
    assert o.status == "working" and sess.broker.open_orders() == [o]
    until, batch = next(feed)
    sess.feed(batch)
    sess.advance(until)
    assert o.status == "filled" and sess.broker.positions() == {"AAA": 25.0} and not sess.broker.fills().empty
    from src.engine import Broker, LiveBroker, BrokerUnavailable, reconcile_positions

    assert isinstance(sess.broker, Broker)
    assert reconcile_positions(sess.broker.positions(), {"AAA": 25.0}).empty and len(reconcile_positions(sess.broker.positions(), {"AAA": 20.0})) == 1
    with pytest.raises(BrokerUnavailable):
        LiveBroker()


def test_stale_marks_are_reported_not_hidden():
    idx, px, reg = stock_market(30)
    px = px.copy()
    px.loc[px.index[10]:, "BBB"] = np.nan                                                      # BBB stops publishing
    ev = events_from_prices(px, "bar", "16:00")

    class HoldB(Strategy):
        name = "hb"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

        def on_schedule(self, ctx):
            if ctx.ts.normalize() == idx[2]:
                ctx.buy("BBB", 10)

    res = run(reg, ev, [HoldB()], cfg(idx, max_mark_age="3D"))
    assert res.diagnostics.get("stale_marks", 0) > 0 or res.diagnostics.get("missing_marks", 0) > 0
