"""Futures contract specifications, continuous versus executable prices, and calendar spreads versus roll losses, with planted numbers worked out by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets.futures import continuous_series, excess_return_series
from src.assets import futures_validation as fv
from src.instruments import InstrumentRegistry, build_chain
from src.instruments.futures import Future


def curve(contango=5.0, n=90, seed=0, step=0.0):
    """Three contracts on a flat-drift random-walk spot, each ``contango`` points above the one before: expiries Mar 1, May 1, Jul 1."""
    d = pd.bdate_range("2024-01-02", periods=n)
    spot = 100 + np.cumsum(np.random.default_rng(seed).normal(step, 1.0, n))
    rows = []
    for k, (c, exp) in enumerate((("A", "2024-03-01"), ("B", "2024-05-01"), ("C", "2024-07-01"))):
        for i, dt in enumerate(d):
            if dt < pd.Timestamp(exp):
                rows.append({"date": dt, "contract": c, "expiry": pd.Timestamp(exp), "price": spot[i] + contango * k})
    return pd.DataFrame(rows), spot, d


# ------------------------------------------------------------------------------------------------------ the return of a difference-adjusted series (a defect fixed here)
def test_the_return_of_a_difference_adjusted_series_is_the_return_of_the_held_contract_not_a_percentage_of_a_shifted_level():
    long, _, _ = curve(contango=5.0, n=100)
    held = excess_return_series(long, 5)
    for adjust in ("ratio", "difference"):
        r = continuous_series(long, 5, adjust).returns.dropna()
        j = r.index.intersection(held.dropna().index)
        assert len(j) > 50 and np.allclose(r.loc[j], held.loc[j], atol=1e-12), adjust
    none = continuous_series(long, 5, "none").returns.dropna()
    j = none.index.intersection(held.dropna().index)
    assert np.abs(none.loc[j] - held.loc[j]).max() > 0.02, "the unadjusted splice keeps the roll gap in its return, on purpose"


def test_a_difference_adjusted_level_can_be_far_from_any_traded_price_and_the_report_says_so():
    long, _, _ = curve(contango=40.0, n=100)
    cs = continuous_series(long, 5, "difference")
    rep = fv.executability(cs)
    assert rep["rolls"] >= 1 and rep["max_abs_gap"] >= 40.0 and rep["share_executable"] < 1.0 and rep["executable_from"] is not None
    assert rep["min_adjusted_level"] > cs.raw.min() + 30, "back-adjusting through a contango lifts the old levels onto the new contract's"
    ratio = fv.executability(continuous_series(curve(contango=-12.0, n=100)[0], 5, "ratio"))
    assert ratio["non_positive_levels"] == 0 and ratio["share_executable"] < 1.0
    none = fv.executability(continuous_series(long, 5, "none"))
    assert none["share_executable"] == 1.0 and none["max_abs_gap"] == 0.0


def test_back_adjustment_through_backwardation_while_the_price_rises_can_take_old_levels_to_zero_or_below_which_is_not_a_price():
    d = pd.bdate_range("2024-01-02", periods=100)
    spot = np.linspace(40.0, 120.0, 100)
    rows = []
    for k, (c, exp) in enumerate((("A", "2024-03-01"), ("B", "2024-05-01"), ("C", "2024-07-01"))):
        for i, dt in enumerate(d):
            if dt < pd.Timestamp(exp):
                rows.append({"date": dt, "contract": c, "expiry": pd.Timestamp(exp), "price": spot[i] - 30.0 * k})
    long = pd.DataFrame(rows)
    long = long[long["price"] > 0]                                              # a contract is not quoted at a negative price: it lists once it is positive
    assert (long["price"] > 0).all() and len(long) > 100
    diff = fv.executability(continuous_series(long, 5, "difference"))
    assert diff["non_positive_levels"] > 0 and diff["min_adjusted_level"] <= 0
    ratio = fv.executability(continuous_series(long, 5, "ratio"))
    assert ratio["non_positive_levels"] == 0, "a ratio adjustment keeps every level positive"


# ------------------------------------------------------------------------------------------------------ calendar spreads and roll losses
def test_the_gap_at_a_roll_is_a_calendar_spread_whose_annualised_yield_is_stated_and_which_realises_only_the_execution_cost():
    long, _, _ = curve(contango=5.0, n=100)
    t = fv.roll_table(long, 5, exec_cost_points=0.25)
    assert len(t) >= 1
    r = t.iloc[0]
    assert r["structure"] == "contango" and r["spread_points"] == pytest.approx(5.0) and r["spread_pct"] == pytest.approx(5.0 / r["old_price"])
    days = (pd.Timestamp("2024-05-01") - pd.Timestamp("2024-03-01")).days
    assert r["annualised_roll_yield"] == pytest.approx(-(5.0 / r["old_price"]) * 365.0 / days)
    assert r["realised_by_roll"] == -0.25, "what the roll itself realises is the cost of crossing two markets, never the spread"
    back = fv.roll_table(curve(contango=-5.0, n=100)[0], 5)
    assert back.iloc[0]["structure"] == "backwardation" and back.iloc[0]["annualised_roll_yield"] > 0


def test_holding_a_contract_through_a_roll_earns_the_moves_of_that_contract_and_the_splice_gap_is_not_a_profit():
    long, spot, d = curve(contango=5.0, n=100, step=0.0)
    cs = continuous_series(long, 5, "none")
    roll = cs.roll_dates[0]
    i = list(cs.raw.index).index(roll)
    gap_return = cs.raw.iloc[i] / cs.raw.iloc[i - 1] - 1.0                              # what the spliced raw series calls the return of the roll day
    px = {c: g.set_index("date")["price"] for c, g in long.groupby("contract")}
    held_return = px["A"][roll] / px["A"][cs.raw.index[i - 1]] - 1.0                    # what the contract held at the previous close earned that day
    assert gap_return - held_return == pytest.approx(5.0 / cs.raw.iloc[i - 1], abs=1e-3 + abs(spot[1] - spot[0]) / 50), "the difference is the calendar spread, nothing the holder earned"
    ex = excess_return_series(long, 5)
    assert ex.loc[roll] == pytest.approx(held_return)


def test_a_roll_strategy_in_a_flat_market_with_contango_loses_the_spread_over_time_through_convergence_not_at_the_roll():
    d = pd.bdate_range("2024-01-02", periods=120)
    rows = []
    for k, (c, exp) in enumerate((("A", "2024-03-01"), ("B", "2024-05-01"), ("C", "2024-07-01"))):
        for dt in d:
            if dt < pd.Timestamp(exp):
                T = (pd.Timestamp(exp) - dt).days
                rows.append({"date": dt, "contract": c, "expiry": pd.Timestamp(exp), "price": 100.0 + 0.04 * T})          # flat spot of 100, a carry of 0.04 points a day to expiry
    long = pd.DataFrame(rows)
    ex = excess_return_series(long, 5).dropna()
    assert (ex < 0).all(), "every day the held contract converges down toward the unchanged spot"
    t = fv.roll_table(long, 5)
    assert (t["structure"] == "contango").all() and (t["realised_by_roll"] == 0.0).all()
    cum = float((1 + ex).prod() - 1)
    assert cum < -0.05, "the loss is real, and it is spread over the days of holding, not booked at the roll"


# ------------------------------------------------------------------------------------------------------ specifications
def good_registry():
    reg = InstrumentRegistry()
    reg.add_chain(build_chain("ES", "2024-01-01", "2024-12-31", months=(3, 6, 9, 12), multiplier=50.0, tick_size=0.25, initial_margin=12000.0, maintenance_margin=10000.0))
    return reg


def test_a_well_formed_chain_has_no_specification_findings_and_prices_on_the_tick_grid_pass():
    reg = good_registry()
    c0 = [i for i in reg if isinstance(i, Future)][0]
    assert fv.validate_specs(reg, {c0.instrument_id: pd.Series([4000.0, 4000.25, 4000.5])}) == []


def test_each_specification_rule_is_checked_and_named():
    reg = good_registry()
    c0, c1, c2 = [i for i in reg if isinstance(i, Future)][:3]
    object.__setattr__(c0, "contract_multiplier", 0.0)
    object.__setattr__(c1, "maintenance_margin", c1.initial_margin + 1.0)
    object.__setattr__(c2, "tick_size", -0.25)
    out = {(f["contract"], f["rule"]) for f in fv.validate_specs(reg)}
    assert (c0.instrument_id, "multiplier") in out and (c1.instrument_id, "margin_order") in out and (c2.instrument_id, "tick_size") in out


def test_a_price_off_the_tick_grid_and_a_listing_after_expiry_are_reported():
    reg = good_registry()
    c0, c1 = [i for i in reg if isinstance(i, Future)][:2]
    object.__setattr__(c1, "first_trade", pd.Timestamp(c1.expiry) + pd.Timedelta(days=1))
    out = fv.validate_specs(reg, {c0.instrument_id: pd.Series([4000.0, 4000.1, 4000.3])})
    by = {(f["contract"], f["rule"]): f for f in out}
    assert "2 prices" in by[(c0.instrument_id, "price_off_tick")]["detail"] and (c1.instrument_id, "listing_before_expiry") in by


def test_a_missing_margin_is_information_not_an_error():
    reg = InstrumentRegistry()
    reg.add_chain(build_chain("ZZ", "2024-01-01", "2024-12-31", months=(3, 6), multiplier=10.0, tick_size=0.01))
    out = fv.validate_specs(reg)
    assert out and {f["rule"] for f in out} == {"margin_missing"} and {f["severity"] for f in out} == {"info"}
