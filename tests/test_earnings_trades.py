"""The earnings statistics and the trade log, on small hand-made books whose answers can be worked out by hand."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.webapp.payload import _drawdown_summary, _longest_run, _profit_factor, clean, earnings_section, trades_section


# ----------------------------------------------------------------------------------------------------------------------------------- earnings
IDX = pd.bdate_range("2019-12-26", periods=8)             # Thu 26 Dec 2019 ... Mon 6 Jan 2020: two months, two calendar years
NET = pd.Series([0.10, -0.05, 0.02, 0.0, -0.10, 0.04, 0.0, 0.05], index=IDX)


def test_the_account_value_profit_and_averages_follow_from_compounding_the_daily_returns():
    e = earnings_section(NET, NET, pd.Series(0.0, index=IDX), 1000.0, {})
    value = 1000.0 * (1.0 + NET).cumprod()
    assert e["capital"] == 1000.0 and e["end_value"] == pytest.approx(value.iloc[-1]) and e["net_profit"] == pytest.approx(value.iloc[-1] - 1000.0)
    assert e["total_return"] == pytest.approx(value.iloc[-1] / 1000.0 - 1.0)
    assert e["years"] == pytest.approx(8 / 252) and e["profit_per_year"] == pytest.approx(e["net_profit"] / (8 / 252)) and e["profit_per_month"] == pytest.approx(e["net_profit"] / (8 / 252 * 12))
    assert e["profit_per_day"] == pytest.approx(e["net_profit"] / 8)
    assert e["gross_profit"] == pytest.approx(e["net_profit"]) and e["costs_paid"] == 0.0 and e["exposure"] is None


def test_profit_is_broken_down_by_month_and_by_calendar_year_and_adds_up():
    e = earnings_section(NET, NET, pd.Series(0.0, index=IDX), 1000.0, {})
    value = 1000.0 * (1.0 + NET).cumprod()
    december, january = value.iloc[3] - 1000.0, value.iloc[7] - value.iloc[3]
    assert [(m["year"], m["month"]) for m in e["monthly"]] == [(2019, 12), (2020, 1)]
    assert e["monthly"][0]["pnl"] == pytest.approx(december) and e["monthly"][1]["pnl"] == pytest.approx(january)
    assert e["monthly"][1]["ret"] == pytest.approx(value.iloc[7] / value.iloc[3] - 1.0) and e["monthly"][1]["end_value"] == pytest.approx(value.iloc[7])
    assert sum(m["pnl"] for m in e["monthly"]) == pytest.approx(e["net_profit"])
    assert [(a["year"], a["partial"], a["days"]) for a in e["annual"]] == [(2019, True, 4), (2020, True, 4)]
    assert sum(a["pnl"] for a in e["annual"]) == pytest.approx(e["net_profit"])
    assert e["annual"][1]["start_value"] == pytest.approx(value.iloc[3]) and e["annual"][1]["end_value"] == pytest.approx(value.iloc[7])
    assert e["winning_months"] == 1 and e["losing_months"] == 1 and e["months"] == 2 and e["winning_years"] + e["losing_years"] == 2


def test_best_and_worst_periods_and_how_often_it_made_money():
    e = earnings_section(NET, NET, pd.Series(0.0, index=IDX), 1000.0, {})
    value = 1000.0 * (1.0 + NET).cumprod()
    assert e["best_day"]["period"] == "2019-12-26" and e["best_day"]["pnl"] == pytest.approx(100.0) and e["best_day"]["ret"] == pytest.approx(0.10)
    assert e["worst_day"]["period"] == "2020-01-01" and e["worst_day"]["pnl"] == pytest.approx(value.iloc[4] - value.iloc[3])
    assert (e["winning_days"], e["losing_days"], e["flat_days"]) == (4, 2, 2)
    wins, losses = (value.diff().fillna(value.iloc[0] - 1000.0)).clip(lower=0).sum(), -(value.diff().fillna(0.0)).clip(upper=0).sum()
    assert e["profit_factor_daily"] == pytest.approx(wins / losses)
    assert e["longest_winning_streak_days"] == 1 and e["longest_losing_streak_days"] == 1            # a flat day ends a run


def test_the_deepest_fall_from_a_peak_is_measured_in_dollars_and_percent_and_not_yet_recovered():
    e = earnings_section(NET, NET, pd.Series(0.0, index=IDX), 1000.0, {})
    value = 1000.0 * (1.0 + NET).cumprod()
    d = e["drawdown"]
    assert d["max_dollars"] == pytest.approx(value.iloc[4] - 1100.0) and d["max_pct"] == pytest.approx(value.iloc[4] / 1100.0 - 1.0)
    assert d["peak_date"] == "2019-12-26" and d["trough_date"] == "2020-01-01" and d["recovered_on"] is None
    assert d["longest_calendar_days"] == (IDX[7] - IDX[0]).days


def test_a_peak_that_is_regained_is_reported_with_its_date():
    series = pd.Series([1000.0, 1100.0, 900.0, 1000.0, 1150.0], index=pd.bdate_range("2021-03-01", periods=5))
    d = _drawdown_summary(series)
    assert d["peak_date"] == "2021-03-02" and d["trough_date"] == "2021-03-03" and d["recovered_on"] == "2021-03-05" and d["max_dollars"] == pytest.approx(-200.0)
    assert d["longest_calendar_days"] == 3
    assert _drawdown_summary(pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2021-03-01", periods=3)))["max_dollars"] == 0.0


def test_costs_are_dollars_on_the_account_of_the_day_and_split_gross_from_net():
    costs = pd.Series(0.001, index=IDX)
    e = earnings_section(NET, NET + costs, costs, 1000.0, {})
    before = (1000.0 * (1.0 + NET).cumprod()).shift(1).fillna(1000.0)
    assert e["costs_paid"] == pytest.approx((before * 0.001).sum()) and e["gross_profit"] == pytest.approx(e["net_profit"] + e["costs_paid"])
    assert 0 < e["costs_share_of_gross"] < 1


def test_benchmarks_and_leverage_are_reported():
    bench = pd.Series(0.01, index=IDX)
    weights = pd.DataFrame({"A": np.linspace(0.5, 2.5, 8), "B": 0.0}, index=IDX)
    e = earnings_section(NET, NET, pd.Series(0.0, index=IDX), 1000.0, {"equal_weight": bench}, weights)
    assert e["benchmarks"]["equal_weight"]["end_value"] == pytest.approx(1000.0 * 1.01 ** 8) and e["benchmarks"]["equal_weight"]["ret"] == pytest.approx(1.01 ** 8 - 1.0)
    assert e["exposure"]["peak"] == pytest.approx(2.5) and e["exposure"]["average"] == pytest.approx(1.5) and e["exposure"]["share_of_days_above_one"] == pytest.approx(6 / 8)


def test_a_strategy_that_never_loses_has_no_profit_factor_and_never_falls():
    up = pd.Series(0.01, index=IDX)
    e = earnings_section(up, up, pd.Series(0.0, index=IDX), 500.0, {})
    assert e["profit_factor_daily"] is None and e["profit_factor_monthly"] is None and e["drawdown"]["max_dollars"] == 0.0 and e["drawdown"]["peak_date"] is None
    json.dumps(clean(e), allow_nan=False)


def test_a_flat_book_earns_nothing_and_produces_no_nan():
    flat = pd.Series(0.0, index=IDX)
    e = earnings_section(flat, flat, flat, 1000.0, {})
    assert e["net_profit"] == 0.0 and e["winning_days"] == 0 and e["flat_days"] == 8 and e["best_day"]["pnl"] == 0.0
    json.dumps(clean(e), allow_nan=False)


def test_the_small_helpers():
    assert _longest_run(np.array([1, 2, -1, 3, 4, 5, 0, 1]), True) == 3 and _longest_run(np.array([-1, -1, 0, -1]), False) == 2 and _longest_run(np.array([]), True) == 0
    assert _profit_factor(pd.Series([3.0, -1.0, 1.0])) == pytest.approx(4.0) and _profit_factor(pd.Series([1.0, 2.0])) is None


# ------------------------------------------------------------------------------------------------------------------------------------ trades
TIDX = pd.bdate_range("2021-01-04", periods=12)
W = pd.DataFrame({"A": [0, 0, .5, .5, .5, 0, 0, -.4, -.4, -.4, 0, .3]}, index=TIDX, dtype=float)
R = pd.DataFrame({"A": [0, .01, .02, -.01, .03, .01, -.02, -.01, .02, .01, .0, .05]}, index=TIDX, dtype=float)
P = 100.0 * (1.0 + R).cumprod()
T = W.diff().fillna(W.iloc[0])                                    # what was bought (+) or sold (-) on each day
ACCOUNT = pd.Series(1000.0, index=TIDX)                           # a constant account, so each profit is the sum of position x return x 1000


def test_a_round_trip_runs_from_the_day_a_position_appears_to_the_day_it_is_gone():
    out = trades_section(W, T, R, P, ACCOUNT, 1000.0)
    closed = [t for t in out["round_trips"] if not t["open"]]
    long, short = sorted(closed, key=lambda t: t["entered"])[0], sorted(closed, key=lambda t: t["entered"])[1]
    assert (long["side"], long["entered"], long["exited"], long["days"]) == ("long", str(TIDX[2].date()), str(TIDX[5].date()), 3)
    assert long["pnl"] == pytest.approx(1000.0 * (0.5 * -0.01 + 0.5 * 0.03 + 0.5 * 0.01))        # the position held at the close of each day earns the next day's return
    assert long["move"] == pytest.approx(P["A"].iloc[5] / P["A"].iloc[2] - 1.0) and long["entry_price"] == pytest.approx(P["A"].iloc[2]) and long["size"] == pytest.approx(0.5)
    assert (short["side"], short["entered"], short["exited"], short["days"]) == ("short", str(TIDX[7].date()), str(TIDX[10].date()), 3)
    assert short["pnl"] == pytest.approx(1000.0 * (-0.4 * 0.02 + -0.4 * 0.01 + -0.4 * 0.0)) and short["move"] == pytest.approx(-(P["A"].iloc[10] / P["A"].iloc[7] - 1.0))


def test_a_position_still_held_at_the_end_is_listed_first_and_marked_open():
    out = trades_section(W, T, R, P, ACCOUNT, 1000.0)
    first = out["round_trips"][0]
    assert first["open"] and first["exited"] is None and first["entered"] == str(TIDX[11].date()) and first["days"] == 0 and first["pnl"] == 0.0
    assert out["total"] == 3 and out["listed"] == 3 and out["stats"]["open_positions"] == 1 and out["stats"]["round_trips"] == 2


def test_the_statistics_of_the_closed_trips():
    s = trades_section(W, T, R, P, ACCOUNT, 1000.0)["stats"]
    win, loss = 1000.0 * (-0.005 + 0.015 + 0.005), -12.0
    assert (s["winners"], s["losers"], s["win_rate"]) == (1, 1, 0.5)
    assert s["average_win"] == pytest.approx(win) and s["average_loss"] == pytest.approx(loss) and s["payoff_ratio"] == pytest.approx(win / -loss)
    assert s["profit_factor"] == pytest.approx(win / -loss) and s["expectancy"] == pytest.approx((win + loss) / 2)
    assert (s["average_days_held"], s["longest_days_held"], s["shortest_days_held"]) == (3.0, 3, 3)
    assert s["best"]["pnl"] == pytest.approx(win) and s["worst"]["pnl"] == pytest.approx(loss) and s["best"]["side"] == "long" and s["worst"]["side"] == "short"
    assert (s["long_trips"], s["short_trips"]) == (1, 1) and s["long_pnl"] == pytest.approx(win) and s["short_pnl"] == pytest.approx(loss)
    assert s["days_in_market"] == pytest.approx(7 / 12) and s["average_exposure"] == pytest.approx((0.5 * 3 + 0.4 * 3 + 0.3) / 12)


def test_orders_are_the_changes_of_position_newest_first_with_a_floor_below_which_drift_is_ignored():
    out = trades_section(W, T, R, P, ACCOUNT, 1000.0)
    s = out["stats"]
    assert (s["orders"], s["buys"], s["sells"]) == (5, 3, 2)
    assert [(o["date"], o["side"]) for o in out["orders"]] == [(str(TIDX[i].date()), side) for i, side in [(11, "buy"), (10, "buy"), (7, "sell"), (5, "sell"), (2, "buy")]]
    assert out["orders"][0]["size"] == pytest.approx(0.3) and out["orders"][0]["amount"] == pytest.approx(300.0) and out["orders"][0]["price"] == pytest.approx(P["A"].iloc[11])
    tiny = T.copy()
    tiny.iloc[3, 0] = 0.001
    assert trades_section(W, tiny, R, P, ACCOUNT, 1000.0)["stats"]["orders"] == 5            # 0.1% of the account is drift, not a decision


def test_a_position_that_flips_side_closes_one_trip_and_opens_the_other():
    w = pd.DataFrame({"A": [.5, .5, -.5, -.5, 0.0]}, index=TIDX[:5])
    r = pd.DataFrame({"A": [0, .01, .01, .01, .01]}, index=TIDX[:5])
    out = trades_section(w, w.diff().fillna(w.iloc[0]), r, 100 * (1 + r).cumprod(), pd.Series(1000.0, index=TIDX[:5]), 1000.0)
    trips = sorted(out["round_trips"], key=lambda t: t["entered"])
    assert [(t["side"], t["entered"], t["exited"]) for t in trips] == [("long", str(TIDX[0].date()), str(TIDX[2].date())), ("short", str(TIDX[2].date()), str(TIDX[4].date()))]
    assert trips[0]["pnl"] == pytest.approx(1000 * (0.5 * 0.01 + 0.5 * 0.01)) and trips[1]["pnl"] == pytest.approx(1000 * (-0.5 * 0.01 - 0.5 * 0.01))


def test_several_assets_each_have_their_own_trips_and_the_total_matches_the_return_they_contributed():
    idx = pd.bdate_range("2021-01-04", periods=10)
    rng = np.random.default_rng(3)
    w = pd.DataFrame({"A": [0, .3, .3, .3, 0, 0, 0, .2, .2, .2], "B": [0, 0, -.2, -.2, -.2, -.2, 0, 0, 0, 0]}, index=idx, dtype=float)
    r = pd.DataFrame(rng.normal(0, 0.01, (10, 2)), index=idx, columns=["A", "B"])
    account = pd.Series(1000.0, index=idx)
    out = trades_section(w, w.diff().fillna(w.iloc[0]), r, 100 * (1 + r).cumprod(), account, 1000.0)
    assert sorted((t["ticker"], t["side"]) for t in out["round_trips"]) == [("A", "long"), ("A", "long"), ("B", "short")]
    total = (w.shift(1).fillna(0.0) * r).sum().sum() * 1000.0
    assert sum(t["pnl"] for t in out["round_trips"]) == pytest.approx(total)                  # every dollar of gross profit belongs to exactly one trip


def test_a_book_that_never_trades_has_an_empty_log_and_serialises():
    w = pd.DataFrame({"A": 0.0}, index=TIDX)
    out = trades_section(w, w.copy(), R, P, ACCOUNT, 1000.0)
    assert out["total"] == 0 and out["round_trips"] == [] and out["orders"] == [] and out["stats"]["win_rate"] is None and out["stats"]["round_trips"] == 0
    assert out["stats"]["days_in_market"] == 0.0 and out["stats"]["profit_factor"] is None and out["stats"]["best"] is None
    json.dumps(clean(out), allow_nan=False)


def test_a_missing_price_does_not_break_the_log():
    prices = P.copy()
    prices.iloc[2, 0] = np.nan
    out = trades_section(W, T, R, prices, ACCOUNT, 1000.0)
    long = next(t for t in out["round_trips"] if t["side"] == "long" and not t["open"])
    assert long["move"] is None and long["entry_price"] is None and long["pnl"] != 0.0
    json.dumps(clean(out), allow_nan=False)


def test_only_the_latest_trips_are_listed_but_the_statistics_cover_all_of_them():
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n)
    w = pd.DataFrame({"A": ([0.5, 0.5, 0.0] * (n // 3))}, index=idx)
    r = pd.DataFrame({"A": 0.001}, index=idx)
    out = trades_section(w, w.diff().fillna(w.iloc[0]), r, 100 * (1 + r).cumprod(), pd.Series(1000.0, index=idx), 1000.0, listed=5)
    assert out["listed"] == 5 and out["total"] == 20 and out["stats"]["round_trips"] == 20
    exits = [t["exited"] for t in out["round_trips"]]
    assert exits == sorted(exits, reverse=True)
