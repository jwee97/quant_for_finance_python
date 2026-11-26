"""Accounting conventions checked across the weight-based backtest engine, the ledger that the event engine posts to, and closed-form arithmetic worked out by hand.

``tests/test_engine_reconciliation.py`` already shows that the three engines are the same book when their assumptions are matched. This file adds what that left out: how a book DRIFTS when nothing is traded and it holds
cash or leverage (the weight engine's two drift conventions against the account's), dividends, splits, funding, partial fills, a futures roll, an option's expiry and a foreign-currency holding, each against a number
written down by hand. Where the engines intentionally differ the test states by how much and why.

Conventions (the ledger is the authority on accounting):

* a weight engine's day-t return is ``w_{t-1}' r_t`` of the equity at the start of day t; the ledger earns the same on the same equity, so they agree to rounding while the weights are the same;
* the weight engine charges ``cost x |traded weight|`` as a fraction of the PREVIOUS close's equity; the ledger charges ``cost x traded notional`` at the trade-time equity, which has the day's return in it. The two
  differ by ``cost x |traded weight| x day's return`` per trade, a second-order amount, and that is the only difference;
* weight drift: ``drift_basis="gross"`` keeps the gross exposure constant after every return (exact only for a fully invested long-only book), ``"equity"`` drifts as an untouched account does.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import LinearCostModel
from src.backtest.engine import BacktestEngine
from src.backtest.execution import drift_weights
from src.instruments import Instrument, InstrumentRegistry, build_chain, crypto_perp, make_option
from src.ledger import CurrencyConverter, Fill, Ledger
from src.ledger.reconcile import reconcile

T0 = pd.Timestamp("2024-01-02 16:00")
DAY = pd.Timedelta(days=1)


def ledger_with(*instruments, cash=1_000_000.0, pegs=None):
    reg = InstrumentRegistry()
    for i in instruments:
        reg.add(i)
    led = Ledger(reg, "USD", None, CurrencyConverter("USD", pegs), 1e-9)
    led.deposit(T0, "USD", cash)
    return reg, led


def stock(name="STK", ccy="USD"):
    return Instrument(instrument_id=name, asset_class="equity", instrument_type="equity", currency=ccy)


def rebalance_ledger(prices: pd.DataFrame, weights: pd.DataFrame, cost_bps=0.0, cash=1_000_000.0):
    """The same decisions as the weight engine, run through the ledger: at each close mark to market, then trade to the target weights of that close (decided on information through it). Returns the equity at each close,
    after costs."""
    led = ledger_with(*[stock(c) for c in prices.columns], cash=cash)[1]
    eq = []
    for i, (ts, row) in enumerate(prices.iterrows()):
        t = T0 + i * DAY
        led.revalue(t, row.to_dict())
        w = weights.iloc[i]
        if w.notna().all():
            eq_pre = led.equity()
            for c in prices.columns:
                target = w[c] * eq_pre / row[c]
                dq = target - led.quantity(c)
                if abs(dq) > 1e-12:
                    led.apply_fill(t, Fill(t, c, dq, row[c], fee=cost_bps / 1e4 * abs(dq) * row[c]), mid=row[c])
        eq.append(led.equity())
    return pd.Series(eq, index=prices.index), led


def market(n=40, seed=5, assets=("A", "B")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    px = pd.DataFrame({a: 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012 + 0.004 * k, n))) for k, a in enumerate(assets)}, index=idx)
    return px, px.pct_change().fillna(0.0)


# ------------------------------------------------------------------------------------------------------ the weight engine against the ledger
def test_a_daily_rebalanced_book_is_the_same_money_in_the_weight_engine_and_the_ledger_when_nothing_costs_anything():
    px, ret = market()
    w = pd.DataFrame({"A": 0.6, "B": 0.4}, index=px.index)
    eng = BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=True, drift_basis="equity", min_assets=1, cost_model=LinearCostModel(cost_bps=0.0))
    res = eng.run(w, ret)
    ledger_eq, led = rebalance_ledger(px, w)
    engine_eq = 1_000_000.0 * (1.0 + res.net_returns.reindex(px.index).fillna(0.0)).cumprod()
    assert np.allclose(engine_eq.to_numpy(), ledger_eq.to_numpy(), rtol=1e-12), "the same weights earn the same money"
    assert reconcile(led).ok


@pytest.mark.parametrize("w0,w1", [(0.5, 0.0), (0.6, 0.0), (1.5, 0.0), (0.8, -0.3), (1.0, -1.0)])
def test_drift_on_an_equity_basis_is_what_the_ledger_does_with_cash_leverage_or_shorts_and_the_gross_basis_is_not(w0, w1):
    px, ret = market(60, seed=8)
    targets = pd.DataFrame(np.nan, index=px.index, columns=["A", "B"])
    targets.iloc[0] = [w0, w1]                                                                  # set once, then left alone: no rebalance after day 0
    tw = targets.copy()
    # the ledger: trade once at the first close, then do nothing
    led = ledger_with(stock("A"), stock("B"))[1]
    path = []
    for i, (ts, row) in enumerate(px.iterrows()):
        t = T0 + i * DAY
        led.revalue(t, row.to_dict())
        if i == 0:
            for c, w in zip(("A", "B"), (w0, w1)):
                led.apply_fill(t, Fill(t, c, w * led.equity() / row[c], row[c]), mid=row[c])
        e = led.equity()
        path.append((led.quantity("A") * row["A"] / e, led.quantity("B") * row["B"] / e))
    true = pd.DataFrame(path, index=px.index, columns=["A", "B"])
    eq_basis = drift_weights(tw, ret, basis="equity")
    gross_basis = drift_weights(tw, ret, basis="gross")
    assert np.allclose(eq_basis.to_numpy(), true.to_numpy(), atol=1e-12), "equity-basis drift is the weights an untouched account has"
    fully_invested_long_only = abs(w0 + w1 - 1.0) < 1e-12 and w1 >= 0
    gap = float(np.abs(gross_basis.to_numpy() - true.to_numpy()).max())
    if fully_invested_long_only:
        assert gap < 1e-12
    else:
        assert gap > 1e-4, "with cash, leverage or shorts the gross-preserving convention is a different book (it re-trades to its gross every day for free)"


def test_the_gross_basis_drift_overstates_a_half_invested_books_exposure_after_a_rise_and_the_amount_is_the_one_derived_by_hand():
    r = 0.10
    targets = pd.DataFrame({"A": [0.5, np.nan, np.nan]}, index=pd.bdate_range("2024-01-02", periods=3))
    ret = pd.DataFrame({"A": [0.0, r, 0.0]}, index=targets.index)
    gross = drift_weights(targets, ret, basis="gross")["A"].iloc[1]
    equity = drift_weights(targets, ret, basis="equity")["A"].iloc[1]
    assert gross == pytest.approx(0.5) and equity == pytest.approx(0.5 * (1 + r) / (1 + 0.5 * r))           # 0.5238...: the half in cash did not grow
    assert equity - gross == pytest.approx(0.5 * (1 + r) / (1 + 0.5 * r) - 0.5)


def test_the_default_drift_basis_is_the_historical_one_so_saved_research_is_reproducible():
    idx = pd.bdate_range("2024-01-02", periods=5)
    assert BacktestEngine().drift_basis == "gross"
    res = BacktestEngine(min_assets=1, rebalance="daily", signal_lag=0).run(pd.DataFrame({"A": 1.0}, index=idx), pd.DataFrame({"A": 0.01}, index=idx))
    assert res.config["drift_basis"] == "gross"


def test_a_cost_per_dollar_traded_differs_between_the_engine_and_the_ledger_only_by_cost_times_turnover_times_the_days_return():
    px, ret = market(30, seed=2)
    w = pd.DataFrame({"A": 0.7, "B": 0.3}, index=px.index)
    c = 25.0                                                                                      # 25 bp: large enough to see
    eng = BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=True, drift_basis="equity", min_assets=1, cost_model=LinearCostModel(cost_bps=c))
    res = eng.run(w, ret)
    ledger_eq, led = rebalance_ledger(px, w, cost_bps=c)
    engine_eq = 1_000_000.0 * (1.0 + res.net_returns.reindex(px.index).fillna(0.0)).cumprod()
    # per day: ledger fee = c x TO x E_pre (E_pre carries the day's return g), engine fee = c x TO x E_prev. Cumulated, the gap is bounded by sum of c x TO x |g| plus terms of order c squared
    turnover = res.turnover.reindex(px.index).fillna(0.0)
    g = res.gross_returns.reindex(px.index).fillna(0.0)
    bound = float((c / 1e4 * turnover * g.abs()).sum())
    gap = abs(float(engine_eq.iloc[-1] / ledger_eq.iloc[-1]) - 1.0)
    assert gap <= 3.0 * bound + 1e-9, f"{gap:.3e} against a second-order bound of {bound:.3e}"
    assert gap < 0.1 * (c / 1e4 * float(turnover.sum())), "and the gap is an order of magnitude smaller than the costs themselves"
    assert reconcile(led).ok


# ------------------------------------------------------------------------------------------------------ cash events
def test_a_dividend_is_a_total_return_in_the_weight_engine_and_a_cashflow_in_the_ledger_and_both_are_the_same_money_when_reinvested():
    idx = pd.bdate_range("2024-01-02", periods=6)
    close = pd.Series([100.0, 101.0, 102.0, 100.0, 103.0, 104.0], index=idx)
    div = pd.Series(0.0, index=idx)
    div.iloc[3] = 2.0                                                                              # 2.00 paid on day 3 (ex-dividend: the close already dropped)
    total_return = (close + div) / close.shift(1) - 1.0
    eng = BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=True, drift_basis="equity", min_assets=1, cost_model=LinearCostModel(cost_bps=0.0))
    res = eng.run(pd.DataFrame({"A": 1.0}, index=idx), pd.DataFrame({"A": total_return.fillna(0.0)}))
    led = ledger_with(stock("A"))[1]
    path, paid = [], 0.0
    for i, ts in enumerate(idx):
        t = T0 + i * DAY
        led.revalue(t, {"A": float(close.iloc[i])})
        if div.iloc[i]:
            paid = led.quantity("A") * float(div.iloc[i])
            led.post_cashflow(t, "dividend", "USD", paid, "A")
        target = led.equity() / close.iloc[i]                                                      # fully invested again at each close: the dividend is reinvested
        dq = target - led.quantity("A")
        if abs(dq) > 1e-12:
            led.apply_fill(t, Fill(t, "A", dq, float(close.iloc[i])), mid=float(close.iloc[i]))
        path.append(led.equity())
    want = 1_000_000.0 * float(((close + div) / close.shift(1)).dropna().prod())
    assert path[-1] == pytest.approx(want, rel=1e-12)
    assert 1_000_000.0 * float((1 + res.net_returns).prod()) == pytest.approx(want, rel=1e-12)
    assert paid > 0 and led.pnl_by("category")["dividend"] == pytest.approx(paid), "the dividend is its own journal category, equal to shares held times the amount per share"
    assert reconcile(led).ok


def test_a_split_leaves_equity_alone_and_a_return_series_built_from_unadjusted_prices_would_show_a_false_loss():
    _, led = ledger_with(stock("A"))
    led.apply_fill(T0, Fill(T0, "A", 100.0, 100.0), mid=100.0)
    before = led.equity()
    led.split(T0 + DAY, "A", 2.0)
    led.revalue(T0 + DAY, {"A": 50.0})
    assert led.equity() == pytest.approx(before) and led.quantity("A") == 200.0 and reconcile(led).ok
    raw = pd.Series([100.0, 50.0])
    assert raw.pct_change().iloc[1] == pytest.approx(-0.5), "this is why a return series must be split-adjusted before a weight engine sees it"


# ------------------------------------------------------------------------------------------------------ funding and carry
def test_a_short_perpetual_that_collects_funding_earns_the_same_in_the_ledger_and_through_the_weight_engines_carry_input():
    idx = pd.bdate_range("2024-01-02", periods=8)
    mark = pd.Series([100.0, 101.0, 99.0, 100.0, 102.0, 101.0, 103.0, 102.0], index=idx)
    f = 0.0003                                                                                   # the short receives 3 bp of its notional each day
    perp = crypto_perp("OKX", "BTC", "USDT")
    reg, led = ledger_with(perp, cash=1_000_000.0, pegs={"USDT": 1.0})
    path = []
    for i in range(len(idx)):
        t = T0 + i * DAY
        led.revalue(t, {perp.instrument_id: float(mark.iloc[i])})
        if i == 0:
            led.apply_fill(t, Fill(t, perp.instrument_id, -led.equity() / (float(mark.iloc[0]) * perp.contract_multiplier), float(mark.iloc[0])), mid=float(mark.iloc[0]))
        else:
            led.post_cashflow(t, "funding", perp.currency, -led.quantity(perp.instrument_id) * perp.contract_multiplier * float(mark.iloc[i - 1]) * f, perp.instrument_id)
        path.append(led.equity())
    ret = mark.pct_change().fillna(0.0)
    carry = pd.DataFrame({"X": f}, index=idx)
    eng = BacktestEngine(signal_lag=0, rebalance="on_change", weight_drift=True, drift_basis="equity", min_assets=1, cost_model=LinearCostModel(cost_bps=0.0), short_carry=carry)
    w = pd.DataFrame({"X": -1.0}, index=idx)
    res = eng.run(w, pd.DataFrame({"X": ret}))
    engine_eq = 1_000_000.0 * (1.0 + res.net_returns.reindex(idx).fillna(0.0)).cumprod()
    funding_total = led.pnl_by("category").get("funding", 0.0)
    assert funding_total > 0, "a short with positive funding is paid"
    # first-order agreement: the short is re-sized to equity daily in the engine, held at its first size in the ledger; the gap is the compounding of the gains on the position size
    assert abs(engine_eq.iloc[-1] / path[-1] - 1.0) < 5e-3
    assert reconcile(led).ok


# ------------------------------------------------------------------------------------------------------ partial fills
def test_a_partial_fill_and_its_completion_leave_the_average_price_the_equity_and_the_journal_exactly_as_one_fill_at_the_average():
    _, one = ledger_with(stock("A"))
    one.apply_fill(T0, Fill(T0, "A", 100.0, 51.0, fee=2.0), mid=51.0)
    _, two = ledger_with(stock("A"))
    two.apply_fill(T0, Fill(T0, "A", 40.0, 50.0, fee=0.8), mid=50.0)
    two.apply_fill(T0 + DAY, Fill(T0 + DAY, "A", 60.0, 51.6667, fee=1.2), mid=51.6667)
    avg = (40 * 50.0 + 60 * 51.6667) / 100
    assert two.positions["A"].avg_price == pytest.approx(avg) and two.quantity("A") == 100.0
    two.revalue(T0 + 2 * DAY, {"A": 55.0})
    one.revalue(T0 + 2 * DAY, {"A": 55.0})
    # hand: equity = cash 1,000,000 - 40*50 - 60*51.6667 - fees 2.0 + 100*55
    assert two.equity() == pytest.approx(1_000_000.0 - 40 * 50.0 - 60 * 51.6667 - 2.0 + 100 * 55.0, rel=1e-12)
    assert one.equity() == pytest.approx(1_000_000.0 - 100 * 51.0 - 2.0 + 100 * 55.0, rel=1e-12)
    assert reconcile(two).ok and reconcile(one).ok


def test_an_order_that_fills_only_in_part_leaves_the_rest_unbooked_not_assumed():
    _, led = ledger_with(stock("A"))
    requested, filled = 100.0, 35.0
    led.apply_fill(T0, Fill(T0, "A", filled, 10.0), mid=10.0)
    assert led.quantity("A") == filled and led.cash["USD"] == pytest.approx(1_000_000.0 - 350.0), "nothing is booked for the 65 that did not trade"
    assert requested - led.quantity("A") == 65.0


# ------------------------------------------------------------------------------------------------------ futures roll and option expiry
def test_a_futures_roll_changes_equity_only_by_its_costs_and_never_by_the_calendar_spread():
    reg = InstrumentRegistry()
    reg.add_chain(build_chain("ES", "2024-01-01", "2024-12-31", months=(3, 6, 9, 12), multiplier=50.0, tick_size=0.25, initial_margin=12000.0, maintenance_margin=10000.0))
    led = Ledger(reg, "USD", None, CurrencyConverter("USD", None), 1e-9)
    led.deposit(T0, "USD", 1_000_000.0)
    near, far = (c.instrument_id for c in reg.chain("ES").contracts[:2])
    led.apply_fill(T0, Fill(T0, near, 3.0, 4000.0), mid=4000.0)
    led.revalue(T0 + DAY, {near: 4020.0})
    e0 = led.equity()
    # roll: sell the near at 4020 and buy the far at 4065 (a 45-point calendar spread, contango): that spread is not a loss
    led.apply_fill(T0 + DAY, Fill(T0 + DAY, near, -3.0, 4020.0, fee=4.5), mid=4020.0)
    led.apply_fill(T0 + DAY, Fill(T0 + DAY, far, 3.0, 4065.0, fee=4.5), mid=4065.0)
    assert led.equity() == pytest.approx(e0 - 9.0), "only the two fees"
    led.revalue(T0 + 2 * DAY, {far: 4075.0})
    assert led.equity() == pytest.approx(e0 - 9.0 + 3 * 50.0 * 10.0), "afterwards the position earns the far contract's own move"
    assert reconcile(led).ok


def test_a_long_option_that_expires_in_the_money_pays_its_intrinsic_value_less_the_premium_and_one_that_expires_worthless_loses_the_premium():
    opt_itm = make_option("SPX", "2024-03-15", "call", 100.0, multiplier=100.0)
    opt_otm = make_option("SPX", "2024-03-15", "call", 120.0, multiplier=100.0)
    _, led = ledger_with(opt_itm, opt_otm)
    led.apply_fill(T0, Fill(T0, opt_itm.instrument_id, 2.0, 5.0), mid=5.0)
    led.apply_fill(T0, Fill(T0, opt_otm.instrument_id, 3.0, 1.0), mid=1.0)
    start = 1_000_000.0
    assert led.equity() == pytest.approx(start)
    spot = 108.0
    led.close_position(T0 + 10 * DAY, opt_itm.instrument_id, max(spot - 100.0, 0.0), "lifecycle_settlement")
    led.close_position(T0 + 10 * DAY, opt_otm.instrument_id, max(spot - 120.0, 0.0), "lifecycle_settlement")
    want = start + 2 * 100.0 * ((spot - 100.0) - 5.0) + 3 * 100.0 * (0.0 - 1.0)
    assert led.equity() == pytest.approx(want) and led.cash["USD"] == pytest.approx(want)
    assert led.quantity(opt_itm.instrument_id) == 0.0 and reconcile(led).ok


# ------------------------------------------------------------------------------------------------------ currencies
def test_a_foreign_holding_splits_into_local_return_and_currency_translation_that_add_to_the_base_currency_return():
    _, led = ledger_with(stock("SAP", "EUR"), cash=0.0, pegs={"EUR": 1.10})
    led.deposit(T0, "EUR", 10_000.0)
    led.apply_fill(T0, Fill(T0, "SAP", 100.0, 100.0), mid=100.0)
    start = led.equity()
    led.converter.set_pair("EUR", "USD", 1.21)
    led.revalue(T0 + DAY, {"SAP": 105.0})
    cats = led.pnl_by("category")
    local = 100 * 5.0 * 1.21                                                                     # the stock gained five euros a share, valued at today's rate
    translation = 10_000.0 * 0.11                                                                 # the whole euro balance (stock at cost plus nothing in cash) gained eleven cents per euro
    assert cats["mtm"] == pytest.approx(local) and cats["fx_translation"] == pytest.approx(translation)
    assert led.equity() - start == pytest.approx(local + translation)
    assert (led.equity() / start - 1.0) == pytest.approx((1.05 * 1.21 / 1.10) - 1.0)
    assert reconcile(led).ok
