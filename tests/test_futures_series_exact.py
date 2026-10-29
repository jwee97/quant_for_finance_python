"""The futures series every real-curve backtest is built on, checked on tables small enough to work out by hand.

A futures price is a stitch of contracts that expire, and the stitch decides every statistic: when the book moves from one contract to the next, which price the day's return is measured on, how far the front
of a table is from the front of the market, and what a calendar says the expiry dates are. The tests that existed showed that the functions run and that a continuous series has no jump at a roll; these state
the days and the numbers, including the day a contract is exactly as close to expiry as the roll rule allows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets.futures import basis_momentum, contract_calendar, continuous_series, contract_returns, excess_return_series, front_month_dates, term_structure


def table() -> pd.DataFrame:
    """Contract A expires on Wednesday 17 March 2021 and contract B on Friday 16 April; B trades 10 above A."""
    days = pd.bdate_range("2021-03-08", "2021-03-19")
    rows = [{"date": d, "contract": "A", "expiry": pd.Timestamp("2021-03-17"), "price": 100.0 + i} for i, d in enumerate(days) if d <= pd.Timestamp("2021-03-17")]
    rows += [{"date": d, "contract": "B", "expiry": pd.Timestamp("2021-04-16"), "price": 110.0 + i} for i, d in enumerate(days)]
    return pd.DataFrame(rows)


def curve(days: pd.DatetimeIndex, contracts: int = 4) -> pd.DataFrame:
    """Monthly contracts (expiring 12 February, 12 March, 16 April, 14 May, 11 June and 16 July 2021), each listed from the first day to its expiry, 2 above the one before."""
    expiries = dict(list({"A": "2021-02-12", "B": "2021-03-12", "C": "2021-04-16", "D": "2021-05-14", "E": "2021-06-11", "F": "2021-07-16"}.items())[:contracts])
    return pd.DataFrame([{"date": d, "contract": c, "expiry": pd.Timestamp(e), "price": 100.0 + 2.0 * k + 0.1 * i} for k, (c, e) in enumerate(expiries.items()) for i, d in enumerate(days) if d <= pd.Timestamp(e)])


# ------------------------------------------------------------------------------------------------------------------ the calendar
def test_the_calendar_names_each_contract_by_its_month_and_year_and_sorts_them_by_expiry():
    calendar = contract_calendar("2021-03-10", "2021-03-10", months=(3, 6), rule="third_friday")
    assert list(calendar.columns) == ["contract", "delivery_month", "expiry"] and list(calendar.index) == list(range(len(calendar)))
    assert calendar["contract"].head(4).tolist() == ["H21", "M21", "H22", "M22"] and calendar["expiry"].is_monotonic_increasing
    assert calendar["expiry"].head(2).tolist() == [pd.Timestamp("2021-03-19"), pd.Timestamp("2021-06-18")], "the third Friday"
    assert calendar["delivery_month"].iloc[0] == pd.Timestamp("2021-03-01") and calendar["expiry"].iloc[-1] < pd.Timestamp("2022-09-10") + pd.Timedelta(days=45), "up to eighteen months past the end"
    assert contract_calendar("2021-03-10", "2021-03-10", months=(3,))["contract"].tolist() == ["H21", "H22"], "the months asked for, from the one the start falls in"
    every = contract_calendar("2021-03-10", "2021-03-10", months=range(1, 13))
    assert every["contract"].iloc[0] == "H21" and every["contract"].iloc[-1] == "U22" and len(every) == 19, "March 2021 to September 2022: the month of the start and eighteen more"
    assert list(contract_calendar("2021-03-10", "2021-03-10")["contract"]) == list(every["contract"]), "every month and the third Friday are the defaults"
    assert contract_calendar("2021-01-04", "2021-01-04", months=(1,)).iloc[0]["contract"] == "F21" and contract_calendar("2021-12-06", "2021-12-06", months=(12,)).iloc[0]["contract"] == "Z21"


def test_each_expiry_rule_puts_the_date_where_the_exchange_does():
    months = (2, 5, 6, 8)

    def expiries(rule):
        calendar = contract_calendar("2021-01-04", "2021-01-04", months=months, rule=rule)
        return {c: d for c, d in zip(calendar["contract"], calendar["expiry"]) if c.endswith("21")}

    weekend_starts = contract_calendar("2021-12-06", "2021-12-06", months=(1, 5), rule="third_friday")        # 1 January 2022 was a Saturday and 1 May a Sunday
    assert dict(zip(weekend_starts["contract"], weekend_starts["expiry"]))["F22"] == pd.Timestamp("2022-01-21") and dict(zip(weekend_starts["contract"], weekend_starts["expiry"]))["K22"] == pd.Timestamp("2022-05-20")
    assert expiries("last_business_day") == {"G21": pd.Timestamp("2021-02-26"), "K21": pd.Timestamp("2021-05-31"), "M21": pd.Timestamp("2021-06-30"), "Q21": pd.Timestamp("2021-08-31")}, "the last weekday of the month"
    assert expiries("15th") == {"G21": pd.Timestamp("2021-02-15"), "K21": pd.Timestamp("2021-05-14"), "M21": pd.Timestamp("2021-06-15"), "Q21": pd.Timestamp("2021-08-13")}, "the 15th, or the Friday before it"
    assert expiries("business_day_before_25th") == {"G21": pd.Timestamp("2021-02-22"), "K21": pd.Timestamp("2021-05-20"), "M21": pd.Timestamp("2021-06-22"), "Q21": pd.Timestamp("2021-08-20")}, "three business days before the 25th"
    with pytest.raises(ValueError, match="rule must be third_friday, business_day_before_25th, 15th or last_business_day"):
        contract_calendar("2021-01-04", "2021-12-31", rule="first_monday")


# ------------------------------------------------------------------------------------------------------------------ the continuous contract
def test_the_book_moves_to_the_next_contract_on_the_day_the_nearest_is_exactly_the_roll_days_from_its_expiry():
    series = continuous_series(table(), roll_days=5)
    roll = pd.Timestamp("2021-03-12")                              # Friday: contract A expires in exactly 5 days, so it is no longer one the book may hold; Thursday it was 6
    assert series.roll_dates == [roll]
    held = series.contract
    assert held.loc[:"2021-03-11"].eq("A").all() and held.loc[roll:].eq("B").all() and len(held) == 10
    assert series.raw.loc["2021-03-11"] == 103.0 and series.raw.loc[roll] == 114.0 and series.raw.loc["2021-03-19"] == 119.0, "the raw splice carries the 11 points gap"
    assert continuous_series(table()).roll_dates == [roll], "five days is the default"
    assert continuous_series(table(), roll_days=4).roll_dates == [pd.Timestamp("2021-03-15")] and continuous_series(table(), roll_days=6).roll_dates == [pd.Timestamp("2021-03-11")]


def test_the_adjustment_removes_the_gap_from_the_return_of_the_day_of_the_roll_and_none_leaves_it_in():
    roll = pd.Timestamp("2021-03-12")
    ratio = continuous_series(table(), adjust="ratio")
    assert ratio.price.loc["2021-03-11"] == pytest.approx(103.0 * 114.0 / 104.0) and ratio.price.loc[roll] == 114.0 and ratio.price.iloc[0] == pytest.approx(100.0 * 114.0 / 104.0)
    assert ratio.returns.loc[roll] == pytest.approx(104.0 / 103.0 - 1.0), "the return of the contract that was held, measured on that contract"
    assert np.isnan(ratio.returns.iloc[0]) and ratio.returns.loc["2021-03-15"] == pytest.approx(115.0 / 114.0 - 1.0)
    difference = continuous_series(table(), adjust="difference")
    assert difference.price.loc["2021-03-11"] == 113.0 and difference.price.iloc[0] == 110.0 and difference.price.loc[roll] == 114.0
    assert difference.returns.loc[roll] == pytest.approx(1.0 / 113.0) and difference.returns.loc["2021-03-09"] == pytest.approx(1.0 / 110.0), "a point move over the adjusted level before it"
    none = continuous_series(table(), adjust="none")
    assert none.price.equals(none.raw) and none.returns.loc[roll] == pytest.approx(114.0 / 103.0 - 1.0), "the gap is a 10.7% return that never happened"
    with pytest.raises(ValueError, match="adjust must be ratio, difference or none"):
        continuous_series(table(), adjust="panama")


def test_a_roll_with_no_price_on_one_side_is_not_adjusted_for_and_does_not_poison_the_history():
    missing = table()
    missing.loc[(missing["contract"] == "B") & (missing["date"] == pd.Timestamp("2021-03-12")), "price"] = np.nan
    series = continuous_series(missing)
    assert series.roll_dates == [] and series.price.loc[:"2021-03-11"].notna().all() and series.price.iloc[0] == 100.0 and series.contract.loc["2021-03-12"] == "B"
    assert np.isnan(series.raw.loc["2021-03-12"]) and series.raw.loc["2021-03-15"] == 115.0


# ------------------------------------------------------------------------------------------------------------------ the tradable excess return
def test_the_return_of_a_day_is_that_of_the_contract_held_at_the_previous_close():
    out = excess_return_series(table())
    assert out.name == "excess_return" and out.index[0] == pd.Timestamp("2021-03-09") and len(out) == 9, "no return before there is a book"
    assert out.loc["2021-03-12"] == pytest.approx(104.0 / 103.0 - 1.0), "the day A stops being the front it still earns the return of A"
    assert out.loc["2021-03-15"] == pytest.approx(115.0 / 114.0 - 1.0) and out.loc["2021-03-19"] == pytest.approx(119.0 / 118.0 - 1.0), "and from the next day that of B"
    assert out.loc["2021-03-09"] == pytest.approx(101.0 / 100.0 - 1.0) and out.loc["2021-03-11"] == pytest.approx(103.0 / 102.0 - 1.0)
    assert excess_return_series(table(), roll_days=4).loc["2021-03-15"] == pytest.approx(105.0 / 104.0 - 1.0), "rolled one trading day later, A is held through the 12th"
    assert excess_return_series(table(), roll_days=6).loc["2021-03-12"] == pytest.approx(114.0 / 113.0 - 1.0), "and one earlier, B is held from the 11th"
    assert excess_return_series(table()).equals(excess_return_series(table(), 5))


# ------------------------------------------------------------------------------------------------------------------ the front of the market
def test_a_date_is_a_front_month_date_when_the_nearest_contract_is_within_a_gap_between_expiries_and_the_roll_and_ten_days():
    early = pd.DataFrame([{"date": pd.Timestamp(d), "contract": "B", "expiry": pd.Timestamp("2021-04-16"), "price": 111.0} for d in ("2021-03-01", "2021-03-02", "2021-03-03")])
    long = pd.concat([early, table()], ignore_index=True)
    front = front_month_dates(long)                                  # the gap between the two expiries is 30 days, so the nearest contract may be 30 + 5 + 10 = 45 days out
    assert front.name == "front_month" and front.dtype == bool and list(front.index) == sorted(front.index)
    assert not front.loc["2021-03-01"] and front.loc["2021-03-02"] and front.loc["2021-03-03"], "46 days to expiry is too far, 45 is not"
    assert front.loc["2021-03-08"] and front.loc["2021-03-12"] and front.loc["2021-03-19"], "from the 8th contract A is 9 days out; on the roll day the nearest that may be held is B, 35 out"
    assert front_month_dates(long, roll_days=6).loc["2021-03-01"] and not front_month_dates(long, roll_days=4).loc["2021-03-02"], "the roll is part of the allowance: 46 <= 30 + 6 + 10 and 45 > 30 + 4 + 10"
    assert front_month_dates(long, slack=11).loc["2021-03-01"] and not front_month_dates(long, slack=9).loc["2021-03-02"]
    assert front_month_dates(long, roll_days=5, slack=10).equals(front) and len(front) == 13


def test_the_last_days_of_a_table_have_no_front_month_to_speak_of():
    long = table()
    late = pd.concat([long, pd.DataFrame([{"date": pd.Timestamp("2021-04-14"), "contract": "B", "expiry": pd.Timestamp("2021-04-16"), "price": 120.0}])], ignore_index=True)
    assert pd.Timestamp("2021-04-14") not in front_month_dates(late).index, "B is 2 days from expiry and nothing else is listed: no contract may be held"


# ------------------------------------------------------------------------------------------------------------------ basis momentum
def test_basis_momentum_is_the_annualised_trailing_mean_of_the_return_gap_and_waits_for_half_a_window():
    near, far = pd.Series(0.002, index=pd.bdate_range("2021-01-04", periods=8)), pd.Series(0.001, index=pd.bdate_range("2021-01-04", periods=8))
    frame = pd.DataFrame({"F1": near, "F2": far, "F3": far * 3})
    out = basis_momentum(frame, window=4)
    assert out.isna().tolist() == [True, False, False, False, False, False, False, False] and out.dropna().tolist() == pytest.approx([0.001 * 252.0] * 7), "two of the four days are enough"
    assert basis_momentum(frame, window=4, near="F3", far="F1").dropna().tolist() == pytest.approx([0.001 * 252.0] * 7)
    long = pd.DataFrame({"F1": 0.002, "F2": 0.001}, index=pd.bdate_range("2020-01-01", periods=300))
    default = basis_momentum(long)
    assert default.first_valid_index() == long.index[125] and default.iloc[-1] == pytest.approx(0.252), "a year, and half of it needed"
    ramp = pd.DataFrame({"F1": np.arange(300) * 1e-4, "F2": 0.0}, index=long.index)
    assert basis_momentum(ramp).iloc[-1] == pytest.approx(np.mean(np.arange(48, 300) * 1e-4) * 252.0), "the mean of the last 252 days, not 251 or 253"


# ------------------------------------------------------------------------------------------------------------------ the curve
def test_the_term_structure_ranks_the_unexpired_contracts_by_expiry_and_keeps_the_four_nearest_by_default():
    days = pd.bdate_range("2021-02-10", "2021-02-12")
    long = curve(days)
    ts = term_structure(long)
    assert list(ts["price"].columns) == ["F1", "F2", "F3", "F4"] and list(ts) == ["price", "contract", "ttm"], "four by default"
    assert ts["contract"].loc["2021-02-10"].tolist() == ["A", "B", "C", "D"] and ts["contract"].loc["2021-02-12"].tolist() == ["A", "B", "C", "D"], "a contract is still in the curve on the day it expires"
    assert ts["ttm"].at[pd.Timestamp("2021-02-12"), "F1"] == 0.0 and ts["ttm"].at[pd.Timestamp("2021-02-10"), "F2"] == pytest.approx((pd.Timestamp("2021-03-12") - pd.Timestamp("2021-02-10")).days / 365.0)
    after = term_structure(curve(pd.bdate_range("2021-02-10", "2021-02-16")))
    assert after["contract"].loc["2021-02-15"].tolist()[:3] == ["B", "C", "D"] and pd.isna(after["contract"].at[pd.Timestamp("2021-02-15"), "F4"]), "the day after, A has gone and only three contracts are left"
    assert list(term_structure(long, depth=2)["price"].columns) == ["F1", "F2"]
    six = term_structure(curve(days, contracts=6))
    assert list(six["price"].columns) == ["F1", "F2", "F3", "F4"] and list(term_structure(curve(days, contracts=6), depth=5)["price"].columns) == ["F1", "F2", "F3", "F4", "F5"], "four of six by default"


def test_the_return_of_a_rank_is_that_of_the_contract_that_held_the_rank_the_day_before():
    index = pd.bdate_range("2021-02-10", periods=3)
    contracts = pd.DataFrame({"F1": ["A", "B", "B"], "F2": ["B", "C", "C"]}, index=index)
    prices = pd.DataFrame({"F1": [100.0, 112.0, 113.0], "F2": [110.0, 120.0, 121.0]}, index=index)
    out = contract_returns(prices, contracts)
    assert np.isnan(out.loc[index[0]]).all(), "the first day has no yesterday"
    assert np.isnan(out.at[index[1], "F1"]), "A, first yesterday, is not in the curve today"
    assert out.at[index[1], "F2"] == pytest.approx(112.0 / 110.0 - 1.0), "B, second yesterday, is first today: its price today over its price yesterday"
    assert out.at[index[2], "F1"] == pytest.approx(113.0 / 112.0 - 1.0) and out.at[index[2], "F2"] == pytest.approx(121.0 / 120.0 - 1.0)
