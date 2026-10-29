"""Market impact, capacity and the no-trade band, checked against numbers worked out by hand.

A planted-bug audit (``experiments/plant_bugs.py``) found ``backtest/impact.py`` half pinned: of the places where one operator, comparison or number was changed the tests noticed 55% (51 of 93).
What the existing tests show is that the model reduces to the linear one when the assets under management go to zero; these show what it is when they do not: the square-root cost, its cap, what
fills a gap, the statistics of the participation, the capacity read off a curve, the band and the scheduling table.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.backtest.impact import (
    ANN,
    ImpactSettings,
    banded_execution,
    banded_gross_returns,
    capacity_curve,
    capacity_from_curve,
    fill_counts,
    impact_cost_frame,
    impact_net_returns,
    market_state,
    scheduling_table,
    sharpe,
)


def days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", periods=n)


def frame(values, columns=("A", "B")) -> pd.DataFrame:
    return pd.DataFrame(values, index=days(len(values)), columns=list(columns), dtype=float)


# ------------------------------------------------------------------------------------------------------------------ what the market looks like before each trade
def test_volatility_and_dollar_volume_are_measured_through_the_previous_close():
    rng = np.random.default_rng(5)
    n = 120
    returns = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), index=days(n), columns=["A", "B"])
    close = pd.DataFrame(rng.uniform(50, 60, (n, 2)), index=days(n), columns=["A", "B"])
    volume = pd.DataFrame(rng.uniform(1e5, 2e5, (n, 2)), index=days(n), columns=["A", "B"])
    sigma, adv = market_state(close, volume, returns)
    k = 80
    assert sigma["A"].iloc[k] == pytest.approx(returns["A"].iloc[k - 21:k].std(ddof=1)), "the 21 days before day k, not including it"
    assert adv["B"].iloc[k] == pytest.approx((close["B"] * volume["B"]).iloc[k - 63:k].mean()), "the 63 days before day k"
    assert sigma["A"].iloc[:10].isna().all() and sigma["A"].iloc[10:].notna().all(), "ten returns are the least that make a volatility (half the window, at least ten), and the first shifts a day"
    assert adv["A"].iloc[:31].isna().all() and adv["A"].iloc[31:].notna().all(), "31 days of dollar volume are the least for a 63-day average"
    shocked = returns.copy()
    shocked.iloc[k] = 0.5
    again, _ = market_state(close, volume, shocked)
    assert again["A"].iloc[k] == sigma["A"].iloc[k] and again["A"].iloc[k + 1] != sigma["A"].iloc[k + 1], "a day's own return reaches the volatility used for the next day"
    short_sigma, short_adv = market_state(close, volume, returns, vol_window=5, adv_window=20)
    assert short_sigma["A"].iloc[k] == pytest.approx(returns["A"].iloc[k - 5:k].std(ddof=1)) and short_adv["A"].iloc[k] == pytest.approx((close["A"] * volume["A"]).iloc[k - 20:k].mean())
    assert short_sigma["A"].iloc[:5].isna().all() and short_sigma["A"].iloc[5:].notna().all(), "a window of five days is as short as it gets: the whole window, not the ten that a longer one needs"
    assert short_adv["A"].iloc[:10].isna().all() and short_adv["A"].iloc[10:].notna().all(), "half of twenty days, and never fewer than ten"


# ------------------------------------------------------------------------------------------------------------------ what a trade costs
RATES = pd.Series({"A": 0.0005, "B": 0.001})


def test_a_trade_pays_its_linear_rate_plus_the_square_root_of_its_share_of_the_days_volume():
    trades = frame([[0.1, -0.2], [0.0, 0.05]])
    sigma = frame([[0.02, 0.03], [0.02, 0.03]])
    adv = frame([[1e9, 5e8], [1e9, 5e8]])
    cost, participation = impact_cost_frame(trades, 1e8, sigma, adv, RATES)
    part_a, part_b = 0.1 * 1e8 / 1e9, 0.2 * 1e8 / 5e8                                           # 0.01 and 0.04 of a day's volume
    assert participation.iloc[0].tolist() == pytest.approx([part_a, part_b]) and participation.iloc[1].tolist() == pytest.approx([0.0, 0.05 * 1e8 / 5e8])
    expected_a = 0.1 * 0.0005 + 1.0 * 0.02 * np.sqrt(part_a) * 0.1
    expected_b = 0.2 * 0.001 + 1.0 * 0.03 * np.sqrt(part_b) * 0.2
    assert cost.iloc[0].tolist() == pytest.approx([expected_a, expected_b]) and cost.iloc[1, 0] == 0.0, "no trade, no cost"
    doubled, _ = impact_cost_frame(trades, 1e8, sigma, adv, RATES, ImpactSettings(coefficient=2.0))
    assert doubled.iloc[0, 0] - 0.1 * 0.0005 == pytest.approx(2 * (cost.iloc[0, 0] - 0.1 * 0.0005)), "the coefficient scales the impact and not the linear cost"
    bigger, _ = impact_cost_frame(trades, 4e8, sigma, adv, RATES)
    assert bigger.iloc[0, 0] - 0.1 * 0.0005 == pytest.approx(2 * (cost.iloc[0, 0] - 0.1 * 0.0005)), "four times the capital is twice the impact per dollar: the square root"
    sell, _ = impact_cost_frame(-trades, 1e8, sigma, adv, RATES)
    assert sell.equals(cost), "selling costs what buying costs"
    assert ImpactSettings().coefficient == 1.0 and ImpactSettings().max_participation == 1.0 and ANN == 252.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        ImpactSettings().coefficient = 3.0                                                       # one setting is shared by every run that uses it


def test_with_no_capital_the_cost_is_the_linear_one_and_participation_above_the_cap_is_charged_at_the_cap():
    trades = frame([[0.5, 0.0]])
    sigma, adv = frame([[0.02, 0.02]]), frame([[1e6, 1e6]])
    for aum in (0.0, -5.0):
        cost, participation = impact_cost_frame(trades, aum, sigma, adv, RATES)
        assert cost.iloc[0].tolist() == pytest.approx([0.5 * 0.0005, 0.0]) and participation.iloc[0].tolist() == [0.0, 0.0]
    cost, participation = impact_cost_frame(trades, 1e7, sigma, adv, RATES)                      # 5 times the day's volume
    assert participation.iloc[0, 0] == pytest.approx(5.0), "the participation reported is the uncapped one"
    assert cost.iloc[0, 0] == pytest.approx(0.5 * 0.0005 + 0.02 * 1.0 * 0.5), "charged as if it were the whole day's volume"
    capped, _ = impact_cost_frame(trades, 1e7, sigma, adv, RATES, ImpactSettings(max_participation=0.25))
    assert capped.iloc[0, 0] == pytest.approx(0.5 * 0.0005 + 0.02 * np.sqrt(0.25) * 0.5)


def test_a_missing_volatility_or_volume_is_the_days_median_across_assets_and_zero_volume_is_not_free_or_infinite():
    trades = frame([[0.1, 0.1, 0.1]], ("A", "B", "C"))
    rates = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
    sigma = frame([[np.nan, 0.02, 0.06]], ("A", "B", "C"))                                      # the median of 0.02 and 0.06 is 0.04
    adv = frame([[1e9, 1e9, 1e9]], ("A", "B", "C"))
    cost, _ = impact_cost_frame(trades, 1e8, sigma, adv, rates)
    unit = np.sqrt(0.1 * 1e8 / 1e9) * 0.1
    assert cost.iloc[0].tolist() == pytest.approx([0.04 * unit, 0.02 * unit, 0.06 * unit]), "a gap in A's volatility is filled from the others, so its trade is not free"
    holes = frame([[1e9, np.nan, 3e9]], ("A", "B", "C"))
    cost, participation = impact_cost_frame(trades, 1e8, frame([[0.02, 0.02, 0.02]], ("A", "B", "C")), holes, rates)
    assert participation.iloc[0].tolist() == pytest.approx([0.1 * 1e8 / 1e9, 0.1 * 1e8 / 2e9, 0.1 * 1e8 / 3e9]), "the missing volume is the median of 1e9 and 3e9"
    empty_day = frame([[np.nan, np.nan, np.nan]], ("A", "B", "C"))
    cost, participation = impact_cost_frame(trades, 1e8, empty_day, empty_day, rates)
    assert (cost.to_numpy() == 0.0).all() and (participation.to_numpy() == 0.0).all(), "a day with nothing known has nothing to fill from"
    dead = frame([[0.0, 1e9, 1e9]], ("A", "B", "C"))
    cost, participation = impact_cost_frame(trades, 1e8, frame([[0.02] * 3], ("A", "B", "C")), dead, rates)
    assert participation.iloc[0, 0] == 0.0 and cost.iloc[0, 0] == 0.0 and np.isfinite(cost.to_numpy()).all(), "no volume is no participation, not an infinite one"


def test_filled_asset_days_are_counted_only_where_something_traded():
    trades = frame([[0.1, 1e-12, 0.0], [0.1, 2e-12, 0.1]], ("A", "B", "C"))
    sigma = frame([[np.nan, np.nan, np.nan], [0.02, np.nan, 0.02]], ("A", "B", "C"))
    adv = frame([[1e9, 1e9, 1e9], [np.nan, 1e9, 1e9]], ("A", "B", "C"))
    assert fill_counts(trades, sigma, adv) == 3, "A on day 1 (volatility), A on day 2 (volume) and B on day 2 (volatility); a trade of exactly 1e-12 is not a trade"
    assert fill_counts(frame([[1.2e-12, 0.0]]), frame([[np.nan, np.nan]]), frame([[1e9, 1e9]])) == 1, "and one of 1.2e-12 is"


def test_the_net_return_is_gross_less_the_days_cost_and_the_participation_statistics_are_over_traded_days():
    trades = frame([[0.1, 0.0], [0.0, 0.0], [-0.2, 0.4]])
    sigma, adv = frame([[0.01, 0.01]] * 3), frame([[1e9, 1e9]] * 3)
    gross = pd.Series([0.001, 0.002, -0.001, 0.0005], index=days(4))                           # a fourth day with no row of trades
    net, daily, stats = impact_net_returns(gross, trades, 1e8, sigma, adv, RATES)
    cost, participation = impact_cost_frame(trades, 1e8, sigma, adv, RATES)
    assert daily.iloc[:3].tolist() == pytest.approx(cost.sum(axis=1).tolist()) and daily.iloc[3] == 0.0 and net.tolist() == pytest.approx((gross - daily).tolist())
    cells = [0.1 * 1e8 / 1e9, 0.2 * 1e8 / 1e9, 0.4 * 1e8 / 1e9]                                 # the three asset-days that traded
    assert stats["traded_asset_days"] == 3 and stats["mean_participation"] == pytest.approx(np.mean(cells)) and stats["max_participation"] == pytest.approx(0.04)
    assert stats["p95_participation"] == pytest.approx(np.quantile(cells, 0.95)) and stats["share_capped"] == 0.0
    capped = impact_net_returns(gross, trades, 1e8, sigma, adv, RATES, ImpactSettings(max_participation=0.015))[2]
    assert capped["share_capped"] == pytest.approx(2 / 3), "two of the three trades are above a cap of 1.5% of the day's volume, and the empty cells are not counted"
    one = frame([[0.1, 0.0]])
    one_gross = pd.Series([0.001], index=days(1))
    assert impact_net_returns(one_gross, one, 1e8, frame([[0.01, 0.01]]), frame([[1e9, 1e9]]), RATES, ImpactSettings(max_participation=0.01))[2]["share_capped"] == 0.0, "a trade of exactly the cap is not over it"
    assert impact_net_returns(one_gross, one, 1e8, frame([[0.01, 0.01]]), frame([[1e9, 1e9]]), RATES, ImpactSettings(max_participation=0.0099))[2]["share_capped"] == 1.0
    assert impact_net_returns(one_gross, frame([[1.2e-12, 0.0]]), 1e8, frame([[0.01, 0.01]]), frame([[1e9, 1e9]]), RATES)[2]["traded_asset_days"] == 1, "a trade of 1.2e-12 is a trade"
    still = impact_net_returns(gross, trades * 0.0, 1e8, sigma, adv, RATES)[2]
    assert still == {"mean_participation": 0.0, "p95_participation": 0.0, "max_participation": 0.0, "share_capped": 0.0, "traded_asset_days": 0}, "no trades, no participation"


def test_the_sharpe_ratio_of_the_capacity_study_needs_three_observations_and_some_movement():
    r = pd.Series([0.01, -0.005, 0.02])
    assert sharpe(r) == pytest.approx(np.sqrt(252) * r.mean() / r.std(ddof=1))
    assert np.isnan(sharpe(r.iloc[:2])) and np.isnan(sharpe(pd.Series([0.01, np.nan, 0.02]))), "two observations are not enough"
    assert np.isnan(sharpe(pd.Series(0.001, index=range(30)))) and np.isnan(sharpe(pd.Series(dtype=float))), "a series that never moves has no Sharpe ratio, not 1e16"


# ------------------------------------------------------------------------------------------------------------------ capacity
def test_capacity_is_the_capital_at_which_the_sharpe_ratio_falls_to_half_its_linear_cost_value():
    curve = pd.Series({1e6: 1.0, 1e7: 0.8, 1e8: 0.2})
    assert capacity_from_curve(curve, 1.0) == pytest.approx(np.sqrt(1e7 * 1e8)), "half-way between 0.8 and 0.2 is 0.5, half-way in the logarithm of the capital"
    assert capacity_from_curve(pd.Series({1e6: 1.0, 1e7: 0.5, 1e8: 0.2}), 1.0) == pytest.approx(1e7), "a point exactly on the target is the answer"
    assert capacity_from_curve(pd.Series({1e6: 1.0, 1e7: 0.8, 1e8: 0.5}), 1.0) == pytest.approx(1e8), "even when it is the last point on the grid"
    assert capacity_from_curve(pd.Series({1e6: 1.0, 1e7: 0.8, 1e8: 0.6}), 1.0) == "above grid" and capacity_from_curve(pd.Series({1e6: 0.5, 1e7: 0.2}), 1.0) == "below grid", "a first point at the target is already below"
    assert capacity_from_curve(pd.Series({1e6: 0.6, 1e7: 0.4}), 1.0) == pytest.approx(10 ** (6 + (0.6 - 0.5) / (0.6 - 0.4)))
    assert capacity_from_curve(curve, 0.0) == "n/a" and capacity_from_curve(curve, -1.0) == "n/a" and capacity_from_curve(curve, float("nan")) == "n/a" and capacity_from_curve(curve, float("inf")) == "n/a"


def test_the_capacity_curve_is_the_net_sharpe_at_each_capital_and_falls_as_capital_grows():
    n = 250
    rng = np.random.default_rng(9)
    gross = pd.Series(rng.normal(0.001, 0.01, n), index=days(n))
    trades = pd.DataFrame({"A": np.where(np.arange(n) % 5 == 0, 0.3, 0.0), "B": np.where(np.arange(n) % 5 == 0, -0.3, 0.0)}, index=days(n))
    sigma, adv = pd.DataFrame(0.01, index=days(n), columns=["A", "B"]), pd.DataFrame(2e7, index=days(n), columns=["A", "B"])
    grid = [1e-3, 1e6, 1e7, 1e8]
    curve = capacity_curve(gross, trades, grid, sigma, adv, RATES)
    assert list(curve.index) == grid and curve.is_monotonic_decreasing
    for a in (1e-3, 1e8):
        assert curve[a] == pytest.approx(sharpe(impact_net_returns(gross, trades, a, sigma, adv, RATES)[0]))
    assert curve[1e-3] == pytest.approx(sharpe(gross - trades.abs().mul(RATES, axis=1).sum(axis=1)), abs=1e-3), "a tiny book pays the linear cost only"


# ------------------------------------------------------------------------------------------------------------------ the no-trade band
def test_a_band_holds_an_asset_unless_its_target_is_further_away_than_the_band():
    index = days(4)
    weights = pd.DataFrame({"A": [0.25, 0.5, 0.5, 0.5], "B": [0.25, 0.25, 0.25, 0.25]}, index=index)       # the engine's book: the target moves A from 0.25 to 0.5 on day 1
    trades = pd.DataFrame({"A": [0.25, 0.25, 0.0, 0.0], "B": [0.25, 0.0, 0.0, 0.0]}, index=index)
    returns = pd.DataFrame(0.0, index=index, columns=["A", "B"])
    held, traded = banded_execution(weights, trades, returns, band=0.0)
    assert held.equals(weights) and traded.equals(trades), "a band of zero is the engine's own execution"
    held, traded = banded_execution(weights, trades, returns, band=0.25)
    assert held["A"].tolist() == [0.25, 0.25, 0.25, 0.25] and traded["A"].tolist() == [0.25, 0.0, 0.0, 0.0], "a gap of exactly the band is not enough to trade"
    held, traded = banded_execution(weights, trades, returns, band=0.2)
    assert held["A"].tolist() == [0.25, 0.5, 0.5, 0.5] and traded["A"].tolist() == [0.25, 0.25, 0.0, 0.0], "a gap above the band trades all the way to the target"
    assert traded["B"].tolist() == [0.25, 0.0, 0.0, 0.0]


def test_between_rebalances_the_banded_book_drifts_with_returns_keeping_its_gross_exposure():
    index = days(3)
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]}, index=index)
    trades = pd.DataFrame({"A": [0.5, 0.0, 0.0], "B": [-0.5, 0.0, 0.0]}, index=index)
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.0], "B": [0.0, -0.05, 0.0]}, index=index)
    held, traded = banded_execution(weights, trades, returns, band=0.0)
    grown = np.array([0.5 * 1.10, -0.5 * 0.95])
    expected = grown * (1.0 / np.abs(grown).sum())
    assert held.iloc[0].tolist() == [0.5, -0.5] and held.iloc[1].tolist() == pytest.approx(expected.tolist()) and held.iloc[2].tolist() == pytest.approx(expected.tolist())
    assert np.abs(held.iloc[1]).sum() == pytest.approx(1.0) and (traded.iloc[1:].to_numpy() == 0.0).all(), "no trade on a day the engine did not trade"
    cash = banded_execution(pd.DataFrame(0.0, index=index, columns=["A", "B"]), pd.DataFrame(0.0, index=index, columns=["A", "B"]), returns, band=0.0)[0]
    assert (cash.to_numpy() == 0.0).all(), "an all-cash book is held as such"
    late = weights.copy()
    late.iloc[0] = np.nan
    held, _ = banded_execution(late, trades.assign(A=[0.0, 0.5, 0.0], B=[0.0, -0.5, 0.0]), returns, band=0.0)
    assert held.iloc[0].isna().all() and held.iloc[1].tolist() == [0.5, -0.5], "nothing is held before the engine holds anything"


def test_the_gross_return_of_a_banded_book_is_the_previous_weights_times_this_days_returns():
    index = days(4)
    held = pd.DataFrame({"A": [np.nan, 0.5, 0.5, 0.5], "B": [np.nan, 0.5, 0.5, 0.5]}, index=index)
    returns = pd.DataFrame({"A": [0.01, 0.02, np.nan, np.nan], "B": [0.01, -0.01, 0.03, np.nan]}, index=index)
    out = banded_gross_returns(held, returns)
    assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[1]), "nothing was held before the first row, so the first row earns nothing; the second earns what the first row's book earned: nothing held"
    assert out.iloc[2] == pytest.approx(0.5 * 0.03), "one asset has no return: the book earned what the other did"
    assert pd.isna(out.iloc[3]), "no asset has a return that day"
    full = banded_gross_returns(pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=days(3)), pd.DataFrame({"A": [0.0, 0.02, 0.04], "B": [0.0, 0.0, -0.04]}, index=days(3)))
    assert pd.isna(full.iloc[0]) and full.iloc[1] == pytest.approx(0.01) and full.iloc[2] == pytest.approx(0.0)


# ------------------------------------------------------------------------------------------------------------------ scheduling
def test_spreading_an_order_over_days_lowers_impact_by_the_root_and_adds_timing_risk():
    table = scheduling_table(0.1, 1e8, 0.02, 5e8, coefficient=1.0, linear_rate=0.0005)
    assert table.index.tolist() == [1, 2, 5, 10] and table.index.name == "days" and list(table.columns) == ["participation_per_day", "cost_bps_of_trade", "timing_risk_bps_of_trade"]
    for k in (1, 2, 5, 10):
        participation = 0.1 * 1e8 / 5e8 / k
        assert table.loc[k, "participation_per_day"] == pytest.approx(participation)
        assert table.loc[k, "cost_bps_of_trade"] == pytest.approx(1e4 * (0.0005 + 0.02 * np.sqrt(participation)))
        expected_risk = 0.0 if k == 1 else 1e4 * 0.02 * np.sqrt(sum(((k - j) / k) ** 2 for j in range(1, k + 1)))
        assert table.loc[k, "timing_risk_bps_of_trade"] == pytest.approx(expected_risk), "the standard deviation of the part still to execute, day by day"
    assert table.loc[2, "timing_risk_bps_of_trade"] == pytest.approx(1e4 * 0.02 * 0.5) and table.loc[2, "cost_bps_of_trade"] < table.loc[1, "cost_bps_of_trade"]
    assert scheduling_table(0.1, 1e8, 0.02, 5e8, days=(3,)).index.tolist() == [3]
    big = scheduling_table(-0.5, 1e9, 0.02, 1e8, days=(1, 100), coefficient=2.0)
    assert big.loc[1, "participation_per_day"] == pytest.approx(5.0) and big.loc[1, "cost_bps_of_trade"] == pytest.approx(1e4 * 2.0 * 0.02 * 1.0), "a trade of five days' volume is charged at the whole day's, though it is reported as it is"
    assert big.loc[100, "participation_per_day"] == pytest.approx(0.05) and big.loc[100, "cost_bps_of_trade"] == pytest.approx(1e4 * 2.0 * 0.02 * np.sqrt(0.05))
