"""The market bundle that a table of futures contracts becomes, checked against a hand-built oracle.

``futures_bundle`` turns each root's contract table into a total-return index of the excess return of holding the nearest contract, and into the signals a model reads: the roll yield, the slope of the
curve and the basis momentum. The oracle here is a plain loop over dates and contracts, so what the pandas code does with pivots and ranks is compared with what the definition says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets.bundles import _hold_through_closures, _index, futures_bundle

EXPIRIES = {"A": "2021-02-12", "B": "2021-03-12", "C": "2021-04-16", "D": "2021-05-14"}


def curve(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Four monthly contracts, each rising 0.1 a day from 100 + 2 a place further out (contango of 2 a contract), listed from the first day to its expiry."""
    rows = []
    for k, (name, expiry) in enumerate(EXPIRIES.items()):
        for i, d in enumerate(days):
            if d <= pd.Timestamp(expiry):
                rows.append({"date": d, "contract": name, "expiry": pd.Timestamp(expiry), "price": 100.0 + 2.0 * k + 0.1 * i})
    return pd.DataFrame(rows)


def oracle(long: pd.DataFrame, roll_days: int = 5) -> pd.Series:
    """Hold, at each close, the nearest contract with more than ``roll_days`` days left; the next day's return is that contract's price change."""
    price = {(r.date, r.contract): r.price for r in long.itertuples()}
    expiry = {r.contract: r.expiry for r in long.itertuples()}
    dates = sorted(long["date"].unique())
    out = {}
    for prev, today in zip(dates[:-1], dates[1:]):
        live = sorted((c for c in expiry if (expiry[c] - prev).days > roll_days and (prev, c) in price), key=lambda c: expiry[c])
        if live and (today, live[0]) in price:
            out[today] = price[(today, live[0])] / price[(prev, live[0])] - 1.0
    return pd.Series(out)


def test_the_index_starts_at_a_hundred_and_compounds_the_returns_with_a_missing_one_counted_as_nothing():
    returns = pd.Series([0.1, np.nan, -0.1, 0.0], index=pd.bdate_range("2021-01-04", periods=4))
    assert _index(returns).tolist() == pytest.approx([110.0, 110.0, 99.0, 99.0]) and _index(returns, start=50.0).iloc[0] == pytest.approx(55.0)


def test_a_closed_market_is_held_for_at_most_the_days_asked_and_never_before_its_first_price_or_after_its_last():
    index = pd.bdate_range("2021-01-04", periods=10)
    frame = pd.DataFrame({"A": np.arange(10.0) + 1.0, "B": np.nan}, index=index)
    frame.loc[index[2:5], "A"] = np.nan                                  # three closed days
    frame.loc[index[1:8], "B"] = [10.0, np.nan, np.nan, 13.0, np.nan, np.nan, 16.0]
    nan = np.nan
    held = _hold_through_closures(frame, 2)
    assert held["A"].equals(pd.Series([1.0, 2.0, 2.0, 2.0, nan, 6.0, 7.0, 8.0, 9.0, 10.0], index=index)), "two of the three closed days are held, the third is not"
    assert held["B"].equals(pd.Series([nan, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 16.0, nan, nan], index=index)), "nothing before its first price and nothing after its last, however few days have passed"
    assert _hold_through_closures(frame, 5)["A"].tolist() == [1.0, 2.0, 2.0, 2.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert frame["A"].isna().sum() == 3, "the table that was given is not changed"


def test_each_root_becomes_an_index_of_the_excess_return_of_the_contract_held_and_three_signals():
    days = pd.bdate_range("2021-01-04", "2021-04-02")
    long = curve(days)
    bundle = futures_bundle({"X": long})
    expected = oracle(long)
    assert bundle.assets == ["X"] and bundle.name == "futures" and bundle.asset_class == {"X": "commodity"}
    assert bundle.prices["X"].index.tolist() == expected.index.tolist(), "the first day has no return, so no price"
    assert bundle.prices["X"].tolist() == pytest.approx((100.0 * (1.0 + expected).cumprod()).tolist(), rel=1e-12)
    assert list(bundle.macro.columns) == ["CARRY_X", "SLOPE_X", "BASISMOM_X"] and bundle.macro.index.equals(bundle.prices.index)
    day = pd.Timestamp("2021-01-20")                                      # F1, F2, F3 are A, B, C: 100.9 + (0, 2, 4) on the 11th day (index 12 -> 0.1 * 12 = 1.2)
    i = days.get_loc(day)
    f1, f3 = 100.0 + 0.1 * i, 104.0 + 0.1 * i
    assert bundle.macro.at[day, "SLOPE_X"] == pytest.approx(np.log(f3 / f1))
    ttm1, ttm2 = (pd.Timestamp(EXPIRIES["A"]) - day).days / 365.0, (pd.Timestamp(EXPIRIES["B"]) - day).days / 365.0
    assert bundle.macro.at[day, "CARRY_X"] == pytest.approx(np.log(f1 / (102.0 + 0.1 * i)) / (ttm2 - ttm1)) and bundle.macro.at[day, "CARRY_X"] < 0.0, "contango: the nearest is the cheapest, so a long pays the roll"
    named = futures_bundle({"X": long}, asset_class={"X": "energy"}, roll_days=3, name="mine")
    assert named.name == "mine" and named.asset_class == {"X": "energy"} and not named.prices["X"].equals(bundle.prices["X"]), "a different roll rule is a different index"
    assert named.prices["X"].tolist() == pytest.approx((100.0 * (1.0 + oracle(long, 3)).cumprod()).tolist(), rel=1e-12)


def test_roots_on_different_calendars_are_joined_on_the_union_of_their_dates_and_carried_over_a_holiday_only_when_asked():
    days = pd.bdate_range("2021-01-04", "2021-04-02")
    holiday = pd.Timestamp("2021-02-15")
    closed = curve(days)
    closed = closed[closed["date"] != holiday]
    plain = futures_bundle({"X": curve(days), "Y": closed})
    assert holiday in plain.prices.index and np.isnan(plain.prices.at[holiday, "Y"]) and not np.isnan(plain.prices.at[holiday, "X"]), "the default carries nothing"
    held = futures_bundle({"X": curve(days), "Y": closed}, hold_days=1)
    before = holiday - pd.Timedelta(days=3)
    assert held.prices.at[holiday, "Y"] == held.prices.at[before, "Y"] and held.returns.at[holiday, "Y"] == 0.0, "a closed market earns nothing that day"
    assert held.macro.at[holiday, "CARRY_Y"] == held.macro.at[before, "CARRY_Y"] and held.prices["X"].equals(plain.prices["X"])
    assert futures_bundle({"X": curve(days), "Y": closed}, hold_days=0).prices.equals(plain.prices)
