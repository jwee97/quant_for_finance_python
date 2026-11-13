"""The three backtesting engines on the same decisions: the vectorized engine (``src/backtest/engine.py``), the event-driven engine (``src/engine/engine.py``) and the realistic execution adapter of the
machine-learning pipeline (``src/ml/execution.py`` on ``src/ml/realism.py``).

They were written for different jobs and each has defaults of its own. What must hold is that, with the assumptions made the same, they are the same book: the decision is traded one session later, the
position earns from the close after, a cost is a cost of the dollars traded, and nothing else moves the equity. The tests state which assumption has to be matched for each pair, so the remaining differences
between the engines' defaults are a short, known list and not a surprise:

* sizing: the event engine and the adapter's ``decision_close`` fix the share count at the decision's prices; the vectorized engine and the adapter's ``fill_time`` re-size at the fill. The adapter does both,
  so it is the bridge between the other two;
* financing: the event engine charges a borrowing spread on a negative cash balance by default (0.5% a year); the other two do not unless told to;
* costs: the vectorized engine takes a cost out of the day's return; the other two pay it from cash, so their book is levered by the fee (a second-order difference of cost times the day's return).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import LinearCostModel
from src.backtest.engine import BacktestEngine
from src.engine import CommissionModel, CostSchedule, Engine, EngineConfig, FinancingModel, Schedule, SpreadModel, Strategy, Target
from src.instruments import Instrument, InstrumentRegistry
from src.marketdata import events_from_prices
from src.ml import execution as ex, realism as rz

warnings.filterwarnings("ignore")
CAPITAL = 1e9
LOT = 1e-6                    # shares in millionths, so that whole-share rounding is not what is being compared
WARM = 5                      # sessions before the first decision


def market(kind: str, n: int = 60, seed: int = 3):
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(seed)
    px = pd.DataFrame({c: 100.0 * np.exp(np.cumsum(rng.normal(0.0003, s, n))) for c, s in (("AAA", 0.01), ("BBB", 0.015), ("CCC", 0.02))}, index=idx)
    if kind == "long_only":
        w = pd.DataFrame(rng.dirichlet(np.ones(3), size=n), index=idx, columns=px.columns)
    else:
        w = pd.DataFrame(rng.normal(0.0, 0.4, size=(n, 3)), index=idx, columns=px.columns)
        w = w.sub(w.mean(axis=1), axis=0)                                                   # dollar neutral: long and short
    w.iloc[:WARM] = np.nan
    return px, w


def run_event(px, w, commission_bps=0.0, spread_bps=0.0, financing=None):
    reg = InstrumentRegistry([Instrument(instrument_id=i, asset_class="equity", instrument_type="equity", currency="USD", calendar="WEEKDAY", lot_size=LOT) for i in px.columns])

    class Follow(Strategy):
        name = "follow"
        schedule = Schedule("daily", "16:30", 4, "WEEKDAY")                                 # decides after the close, on that close

        def on_schedule(self, ctx):
            day = pd.Timestamp(ctx.ts).normalize()
            if day in w.index and w.loc[day].notna().all():
                ctx.set_targets([Target(i, weight=float(w.loc[day, i])) for i in w.columns])

    costs = CostSchedule().set("equity", commission=CommissionModel(bps=commission_bps), spread=SpreadModel(fallback_bps=spread_bps, min_half_ticks=0.0))
    cfg = EngineConfig(start=px.index[0], end=px.index[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": CAPITAL})
    res = Engine(reg, events_from_prices(px, "bar", "16:00"), [Follow()], cfg, costs, financing or FinancingModel(borrow_spread=0.0)).run()
    equity = res.equity_curve.copy()
    equity.index = pd.DatetimeIndex(equity.index).normalize()
    return res, equity.groupby(level=0).last()


def run_adapter(px, w, sizing, commission_bps=0.0, half_spread_bps=0.0, **over):
    spec = dict(capital=CAPITAL, delay=1, sizing=sizing, lot=LOT, max_participation=None, commission_bps=commission_bps, half_spread_bps=half_spread_bps, slippage_bps=0.0, impact=rz.ImpactSpec(coefficient=0.0),
                min_trade_weight=0.0, cash_rate=0.0, financing_spread=0.0)
    spec.update(over)
    return ex.execute_weights(w, px, ex.ExecutionSpec(**spec))


def run_vector(px, w, cost_bps=0.0):
    eng = BacktestEngine(signal_lag=1, rebalance="daily", weight_drift=True, annualisation=252, min_assets=1, cost_model=LinearCostModel(cost_bps=cost_bps), target_vol=None)
    return eng.run(w.fillna(0.0), px.pct_change(), "reference", None, apply_vol_target=False)


def window(px):
    return px.index[WARM + 3:]                                                              # from the first day a position decided on the first decision day has earned a whole return


def gap(a: pd.Series, b: pd.Series, days) -> float:
    return float((a.reindex(days) - b.reindex(days)).abs().max())


# ---------------------------------------------------------------------------------------------------------------------- the same book
@pytest.mark.parametrize("kind", ["long_only", "long_short"])
@pytest.mark.parametrize("bps", [0.0, 5.0])
def test_the_event_engine_and_the_adapter_are_the_same_book_when_both_size_at_the_decision(kind, bps):
    px, w = market(kind)
    res, equity = run_event(px, w, commission_bps=bps)
    adapter = run_adapter(px, w, "decision_close", commission_bps=bps)
    days = window(px)
    assert res.reconciliation.ok and adapter.reconciliation["reconciled"]
    assert gap(equity.pct_change(), adapter.returns, days) < 1e-12                          # two ledgers, written separately, to the last digit
    assert float(equity.iloc[-1]) == pytest.approx(float(adapter.equity.iloc[-1]), rel=1e-12)
    if bps:
        assert float(res.cost_summary()["fee"]) == pytest.approx(float(adapter.costs["fee"]), rel=1e-9)
        assert float(res.cost_summary()["fee"]) > 0


@pytest.mark.parametrize("kind", ["long_only", "long_short"])
def test_the_adapter_and_the_vectorized_engine_are_the_same_book_when_both_size_at_the_fill(kind):
    px, w = market(kind)
    adapter, vector = run_adapter(px, w, "fill_time"), run_vector(px, w)
    assert gap(adapter.returns, vector.net_returns, window(px)) < 1e-12


@pytest.mark.parametrize("kind", ["long_only", "long_short"])
def test_all_three_trade_the_first_decision_one_session_later_and_earn_from_the_close_after(kind):
    px, w = market(kind)
    res, equity = run_event(px, w)
    adapter, vector = run_adapter(px, w, "decision_close"), run_vector(px, w)
    decided, filled, earns = px.index[WARM], px.index[WARM + 1], px.index[WARM + 2]
    assert pd.Timestamp(res.fills["ts"].min()).normalize() == filled and adapter.fills["ts"].min() == filled
    held = vector.weights.abs().sum(axis=1)
    assert held.index[held > 0][0] == filled                                                # the vectorized engine's first book is in force at the same close
    for returns in (equity.pct_change(), adapter.returns, vector.net_returns):
        r = returns.reindex(px.index).fillna(0.0)
        assert (r.loc[:filled].abs() < 1e-15).all() and abs(r.loc[earns]) > 0               # nothing is earned on the decision day or the fill day
    assert decided < filled < earns


def test_a_cost_per_dollar_traded_is_the_same_money_in_all_three_up_to_how_it_is_paid():
    """The vectorized engine takes the cost out of the return, so its book stays fully invested; the other two pay it from cash, so theirs is levered by the fee. The gap is the cost times the day's return:
    a few per cent of a day's cost at most, and a fraction of a per cent of the total."""
    px, w = market("long_only")
    bps = 5.0
    adapter, vector, free = run_adapter(px, w, "fill_time", commission_bps=bps), run_vector(px, w, cost_bps=bps), run_vector(px, w)
    days = window(px)
    daily_cost = (free.net_returns - vector.net_returns).reindex(days)
    assert daily_cost.min() > 0 and gap(adapter.returns, vector.net_returns, days) < 0.05 * float(daily_cost.mean())
    paid_vector = float((daily_cost * adapter.equity.shift(1).reindex(days)).sum())          # the vectorized engine's cost in dollars, on the adapter's own equity
    paid_adapter = float(adapter.daily_costs["fee"].reindex(days).sum())
    assert paid_adapter == pytest.approx(paid_vector, rel=5e-3)
    expected = bps / 1e4 * float(adapter.turnover.reindex(days).sum()) if isinstance(adapter.turnover, pd.Series) else None
    if expected is not None:
        assert expected > 0


def test_a_half_spread_is_charged_once_per_dollar_traded_by_the_event_engine_and_by_the_adapter():
    px, w = market("long_only")
    half = 4.0
    res, equity = run_event(px, w, spread_bps=2 * half)                                     # the event engine is given the whole spread and crosses half of it
    adapter = run_adapter(px, w, "decision_close", half_spread_bps=half)
    free = run_adapter(px, w, "decision_close")
    days = window(px)
    assert gap(equity.pct_change(), adapter.returns, days) < 1e-9
    lost = float(free.equity.iloc[-1] - adapter.equity.iloc[-1])
    traded = float((adapter.fills["quantity"].abs() * adapter.fills["price"]).sum())
    assert lost == pytest.approx(half / 1e4 * traded, rel=0.02)                              # what the spread cost is the half-spread on the dollars traded (up to the returns earned on the difference)


def test_sizing_at_the_decision_and_at_the_fill_differ_only_by_the_move_between_the_two_closes():
    px, w = market("long_only")
    frozen = px.copy()
    frozen.iloc[1::2] = frozen.iloc[0::2].to_numpy()                                         # every second close repeats the one before: decisions on even days are filled at an unchanged price
    w2 = w.copy()
    w2.iloc[1::2] = np.nan
    w2 = w2.ffill()
    w2.iloc[:WARM + 1] = np.nan
    a, b = run_adapter(frozen, w2, "decision_close"), run_adapter(frozen, w2, "fill_time")
    assert gap(a.returns, b.returns, frozen.index[WARM + 4:]) < 1e-12                        # no move between decision and fill: the two conventions are one
    c, d = run_adapter(px, w, "decision_close"), run_adapter(px, w, "fill_time")
    assert 1e-5 < gap(c.returns, d.returns, window(px)) < 5e-3                               # with a move they differ, by the size of a day's move times the turnover


def test_the_event_engines_default_borrowing_spread_is_the_whole_of_its_difference_to_the_adapter():
    px, w = market("long_only")
    _, default = run_event(px, w, financing=FinancingModel())                                # 0.5% a year on a negative cash balance
    _, matched = run_event(px, w)
    adapter = run_adapter(px, w, "decision_close")
    days = window(px)
    assert gap(matched.pct_change(), adapter.returns, days) < 1e-12
    assert 0 < gap(default.pct_change(), adapter.returns, days) < 1e-5 and float(default.iloc[-1]) < float(matched.iloc[-1])


# ---------------------------------------------------------------------------------------------------------------------- the defect this found
def test_an_order_of_millions_of_fractional_shares_fills_whole_and_is_not_refused_for_rounding():
    """Before the fix the order book compared share counts with an absolute tolerance of 1e-12, so a fill of 6,109,110.123457 shares in lots of a millionth was refused as larger than itself."""
    book = rz.OrderBook()
    quantity = -6109110.123457
    order = book.submit("AAA", quantity, pd.Timestamp("2024-01-10"))
    noisy = float(np.floor(abs(quantity) / LOT) * LOT) * -1.0 * (1.0 + 2e-16)
    book.fill(order.order_id, "f1", noisy, pd.Timestamp("2024-01-11"))
    assert order.state == "filled" and order.remaining == 0.0
    with pytest.raises(rz.OrderError):
        again = book.submit("AAA", 100.0, pd.Timestamp("2024-01-10"))
        book.fill(again.order_id, "f2", 100.001, pd.Timestamp("2024-01-11"))                 # a real over-fill is still refused
    part = book.submit("AAA", 100.0, pd.Timestamp("2024-01-10"))
    book.fill(part.order_id, "f3", 40.0, pd.Timestamp("2024-01-11"))
    assert part.state == "partial" and part.remaining == 60.0


def test_a_whole_number_of_fractional_lots_is_not_rounded_down_by_one_lot():
    assert rz.fill_with_limits(0.3, float("inf"), None, 0.1)["filled"] == pytest.approx(0.3, abs=1e-12)       # 0.3 / 0.1 is 2.9999999999999996 in floating point
    assert rz.fill_with_limits(0.35, float("inf"), None, 0.1)["filled"] == pytest.approx(0.3, abs=1e-12)      # a real remainder is still rounded down
    assert rz.fill_with_limits(-7.0, float("inf"), None, 1)["filled"] == -7.0 and rz.fill_with_limits(7.9, float("inf"), None, 1)["filled"] == 7.0
    assert rz.fill_with_limits(1000.0, 5000.0, 0.1, 100)["filled"] == 500.0
