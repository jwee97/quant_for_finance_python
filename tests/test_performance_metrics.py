"""The performance metrics against figures worked out by hand or by a second, plainer formula: volatility, the Sharpe, Sortino, Calmar and Omega ratios, value at risk and its tail mean, the tail
ratio, the summary block that every comparison table is made of, the rolling windows, calendar returns, and the deflated and probabilistic Sharpe ratios.

A planted-bug audit (``experiments/plant_bugs.py``) of ``src/backtest/metrics.py`` found that the tests named for it noticed 18 of 120 changes, and reading what survived found a real fault:
``deflated_sharpe_ratio`` returned exactly 1.0 for any Sharpe ratio, a negative one included, when the number of trials was one (the formula puts the expected maximum of one estimate at minus
infinity), so every strategy tried once was reported as certain to survive selection. The tests of the single-trial case below fail without the fix."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtest import metrics as m

R = [0.02, -0.01, 0.03, -0.02, 0.01, 0.0, 0.015, -0.005, 0.01, -0.01]
INDEX = pd.bdate_range("2021-01-04", periods=len(R))
LONG = pd.bdate_range("2021-01-04", periods=400)


def series(values=R, index=None) -> pd.Series:
    return pd.Series(values, index=LONG[:len(values)] if index is None else index, dtype=float)


# ------------------------------------------------------------------------------------------------------------------------ volatility and the ratios
def test_volatility_is_the_sample_standard_deviation_scaled_to_the_year_and_ignores_missing_days():
    expected = np.std(R, ddof=1)
    assert m.annualised_volatility(series()) == pytest.approx(expected * math.sqrt(252))
    assert m.annualised_volatility(series(), periods_per_year=12) == pytest.approx(expected * math.sqrt(12))
    holes = pd.concat([series(), pd.Series([np.nan, np.nan], index=pd.bdate_range("2021-01-18", periods=2))])
    assert m.annualised_volatility(holes) == pytest.approx(expected * math.sqrt(252))
    assert m.annualised_volatility(series([0.001] * 5)) == 0.0 and m.annualised_volatility(series([0.01] * 12)) == 0.0                  # rounding noise is not volatility
    assert math.isnan(m.annualised_volatility(series([np.nan, np.nan])))


def test_sharpe_is_the_mean_excess_return_over_its_volatility_scaled_to_the_year():
    mean, vol = np.mean(R), np.std(R, ddof=1)
    assert m.sharpe_ratio(series()) == pytest.approx(mean / vol * math.sqrt(252))
    assert m.sharpe_ratio(series(), periods_per_year=12) == pytest.approx(mean / vol * math.sqrt(12))
    assert m.sharpe_ratio(series(), risk_free=0.0252) == pytest.approx((mean - 0.0252 / 252) / vol * math.sqrt(252))            # an annual rate, per period
    rf = pd.Series([0.001] * 5, index=INDEX[:5])                                                              # a risk-free series covers part of the days; the rest count as zero
    excess = np.array(R) - np.r_[np.full(5, 0.001), np.zeros(5)]
    assert m.sharpe_ratio(series(), risk_free=rf) == pytest.approx(excess.mean() / excess.std(ddof=1) * math.sqrt(252))
    assert math.isnan(m.sharpe_ratio(series([np.nan, np.nan]))) and math.isnan(m.sharpe_ratio(series([0.01] * 5)))             # nothing to measure, and no volatility to divide by
    assert math.isnan(m.sharpe_ratio(series([0.001] * 50)))                                                    # a constant 0.1% has a variance of 1e-37 in floating point, not 0: it is still no volatility


def test_sortino_divides_by_the_root_mean_square_of_the_returns_below_the_target():
    # the four losses are -1%, -2%, -0.5% and -1%: their mean square is 0.000625 / 4, whose root is exactly 1.25%; the mean return is 0.4%
    assert m.sortino_ratio(series()) == pytest.approx(0.004 / 0.0125 * math.sqrt(252))
    assert m.sortino_ratio(series(), periods_per_year=12) == pytest.approx(0.004 / 0.0125 * math.sqrt(12))
    below = np.array([r for r in R if r < 0.01])                                                               # a higher target counts more days as shortfalls
    expected = (np.mean(R) - 0.01) / math.sqrt(np.mean((below - 0.01) ** 2)) * math.sqrt(252)
    assert m.sortino_ratio(series(), target=0.01) == pytest.approx(expected)
    assert math.isnan(m.sortino_ratio(series([0.01, -0.01, 0.02, 0.03])))                                      # one loss is not a deviation
    two = series([0.02, -0.01, -0.02, 0.03])                                                                    # two are: root mean square of 1% and 2% is sqrt(0.00025)
    assert m.sortino_ratio(two) == pytest.approx(0.005 / math.sqrt(0.00025) * math.sqrt(252))
    assert math.isnan(m.sortino_ratio(series([0.01, 0.02, 0.03])))


def test_calmar_is_the_annual_growth_over_the_worst_fall_and_omega_is_gains_over_losses():
    assert m.max_drawdown(m.cumulative_returns(series())) == pytest.approx(-0.02)                              # the 2% day straight after the high
    assert m.calmar_ratio(series()) == pytest.approx(m.cagr(series()) / 0.02)
    assert math.isnan(m.calmar_ratio(series([0.01, 0.02, 0.01])))                                              # never down: no fall to divide by
    assert m.omega_ratio(series()) == pytest.approx(0.085 / 0.045)
    t = 0.005
    assert m.omega_ratio(series(), threshold=t) == pytest.approx(sum(max(r - t, 0) for r in R) / sum(max(t - r, 0) for r in R))
    assert math.isnan(m.omega_ratio(series([0.01, 0.02])))


def test_cagr_compounds_over_the_calendar_length_and_falls_back_to_the_count_of_periods():
    two_years = series([0.10, 0.21], index=pd.DatetimeIndex(["2020-12-31", "2022-12-31"]))
    years = (pd.Timestamp("2022-12-31") - pd.Timestamp("2020-12-31")).days / 365.25
    assert m.cagr(two_years) == pytest.approx((1.10 * 1.21) ** (1.0 / years) - 1.0)
    single = series([0.10], index=pd.DatetimeIndex(["2021-06-30"]))                                           # a single day has no calendar length: 1 period of 252
    assert m.cagr(single) == pytest.approx(1.10 ** 252 - 1.0)
    same_day = series([0.10, 0.10], index=pd.DatetimeIndex(["2021-06-30", "2021-06-30"]))                      # no calendar length at all: counted in periods, not divided by zero
    assert m.cagr(same_day) == pytest.approx((1.10 * 1.10) ** (252 / 2) - 1.0)
    assert math.isnan(m.cagr(series([np.nan])))
    daily = series()
    assert m.cagr(daily) == pytest.approx(np.prod(1 + np.array(R)) ** (365.25 / (INDEX[-1] - INDEX[0]).days) - 1.0)


# ------------------------------------------------------------------------------------------------------------------------ the tail
def test_value_at_risk_and_its_tail_mean_are_losses_read_from_the_empirical_quantile():
    # sorted: -2%, -1%, -1%, -0.5%, 0, 1%, 1%, 1.5%, 2%, 3%; the 5% point is 45% of the way from -2% to -1%
    assert m.value_at_risk(series()) == pytest.approx(0.0155)
    assert m.value_at_risk(series(), 0.9) == pytest.approx(0.011)
    assert m.conditional_value_at_risk(series()) == pytest.approx(0.02)                                       # only the -2% is at or below -1.55%
    assert m.conditional_value_at_risk(series(), 0.8) == pytest.approx((0.02 + 0.01 + 0.01) / 3)             # at or below -1%
    assert m.conditional_value_at_risk(series(), 0.8) >= m.value_at_risk(series(), 0.8)
    assert math.isnan(m.value_at_risk(series([np.nan]))) and math.isnan(m.conditional_value_at_risk(series([np.nan])))


def test_the_tail_ratio_compares_the_best_five_percent_with_the_worst_five_percent():
    assert m.tail_ratio(series()) == pytest.approx(0.0255 / 0.0155)
    assert m.tail_ratio(series(), quantile=0.1) == pytest.approx(abs(np.quantile(R, 0.9)) / abs(np.quantile(R, 0.1)))
    assert math.isnan(m.tail_ratio(series([0.0, 0.0, 0.0, 0.0, 0.01, 0.02])))                                 # no losses in the tail to divide by


# ------------------------------------------------------------------------------------------------------------------------ the summary block
def test_the_summary_block_holds_each_figure_the_comparison_tables_show():
    s = m.performance_summary(series(), label="x")
    curve = np.cumprod(1 + np.array(R))
    assert (s["start"], s["end"], s["n_obs"], s["label"]) == ("2021-01-04", "2021-01-15", 10, "x")
    assert s["ann_return_arith"] == pytest.approx(0.004 * 252) and s["ann_vol"] == pytest.approx(np.std(R, ddof=1) * math.sqrt(252))
    assert s["sharpe"] == pytest.approx(m.sharpe_ratio(series())) and s["sortino"] == pytest.approx(0.32 * math.sqrt(252)) and s["omega"] == pytest.approx(0.085 / 0.045)
    assert s["max_drawdown"] == pytest.approx(-0.02) and s["max_drawdown_date"] == "2021-01-07" and s["calmar"] == pytest.approx(s["cagr"] / 0.02)
    assert s["time_underwater_share"] == pytest.approx(0.6) and s["hit_rate"] == pytest.approx(0.5)                # under water on days 2, 4, 5, 6, 8 and 10; up on five of ten days
    assert s["best_day"] == 0.03 and s["worst_day"] == -0.02 and s["total_return"] == pytest.approx(curve[-1] - 1.0)
    assert s["skew"] == pytest.approx(stats.skew(R, bias=False)) and s["excess_kurtosis"] == pytest.approx(stats.kurtosis(R, bias=False))
    assert s["var_95"] == pytest.approx(0.0155) and s["cvar_95"] == pytest.approx(0.02) and s["tail_ratio"] == pytest.approx(0.0255 / 0.0155)
    assert "label" not in m.performance_summary(series()) and m.performance_summary(series([np.nan, np.nan])) == {}


def test_a_series_that_never_moves_has_no_skew_and_the_summary_says_so_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = m.performance_summary(series([0.01] * 12))
    assert math.isnan(s["skew"]) and math.isnan(s["excess_kurtosis"]) and s["ann_vol"] == 0.0 and s["time_underwater_share"] == 0.0 and s["hit_rate"] == 1.0
    almost = m.performance_summary(series([0.01, 0.01 + 1e-13, 0.01, 0.01 + 1e-13, 0.01, 0.01]))                # a spread of 1e-13 is below what the third moment can be trusted with: also left undefined
    assert math.isnan(almost["skew"])
    spread = m.performance_summary(series([0.01, 0.02, 0.01, 0.02, 0.01, 0.015]))
    assert not math.isnan(spread["skew"])


def test_turnover_and_costs_are_annualised_over_the_days_of_the_series_with_missing_days_as_zero():
    turnover = pd.Series([1.0, 2.0], index=INDEX[:2])
    costs = pd.Series([0.0005] * 4, index=INDEX[2:6])
    s = m.performance_summary(series(), turnover=turnover, costs=costs)
    assert s["ann_turnover"] == pytest.approx(3.0 / 10 * 252) and s["ann_cost_drag"] == pytest.approx(0.002 / 10 * 252)
    assert "ann_turnover" not in m.performance_summary(series()) and "ann_cost_drag" not in m.performance_summary(series())


def test_against_a_benchmark_the_summary_gives_beta_alpha_tracking_error_and_the_information_ratio():
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2021-01-04", periods=120)
    bench = pd.Series(rng.normal(0.0004, 0.01, 120), index=index)
    fund = 0.001 + 1.5 * bench                                                                                 # a daily 10 bp of alpha on 1.5 times the market
    s = m.performance_summary(fund, benchmark=bench)
    active = fund - bench
    tracking = active.std(ddof=1) * math.sqrt(252)
    assert s["beta_to_benchmark"] == pytest.approx(1.5) and s["alpha_ann"] == pytest.approx(0.001 * 252)
    assert s["tracking_error"] == pytest.approx(tracking) and s["information_ratio"] == pytest.approx(active.mean() * 252 / tracking)
    short = m.performance_summary(fund.iloc[:30], benchmark=bench.iloc[:30])                                    # 30 days in common is not enough to say anything
    assert not {"beta_to_benchmark", "alpha_ann", "tracking_error", "information_ratio"} & set(short)
    enough = m.performance_summary(fund.iloc[:31], benchmark=bench.iloc[:31])
    assert "beta_to_benchmark" in enough
    flat = m.performance_summary(fund, benchmark=pd.Series(0.001, index=index))                                  # a benchmark that never moves has no beta (its variance is 1e-37 in floating point)
    assert math.isnan(flat["beta_to_benchmark"])
    offset = m.performance_summary(bench + 0.001, benchmark=bench)                                               # the same risk plus a constant: no active risk, so no information ratio
    assert offset["tracking_error"] == pytest.approx(0.0, abs=1e-12) and math.isnan(offset["information_ratio"]) and offset["beta_to_benchmark"] == pytest.approx(1.0)
    same = m.performance_summary(bench, benchmark=bench)                                                         # the benchmark itself: no active risk to divide by
    assert same["tracking_error"] == 0.0 and math.isnan(same["information_ratio"]) and same["beta_to_benchmark"] == pytest.approx(1.0) and same["alpha_ann"] == pytest.approx(0.0, abs=1e-12)


def test_the_comparison_table_has_one_row_per_strategy_and_leaves_out_one_with_no_returns():
    table = m.comparison_table({"a": series(), "b": series([x * 2 for x in R]), "empty": series([np.nan, np.nan])})
    assert list(table.index) == ["a", "b"] and table.loc["b", "worst_day"] == pytest.approx(-0.04) and table.loc["a", "n_obs"] == 10
    assert m.comparison_table({"a": series()}, periods_per_year=12).loc["a", "ann_vol"] == pytest.approx(np.std(R, ddof=1) * math.sqrt(12))


# ------------------------------------------------------------------------------------------------------------------------ rolling windows and calendars
def test_rolling_sharpe_and_volatility_start_when_half_a_window_is_known_and_are_scaled_to_the_year():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0005, 0.01, 300), index=pd.bdate_range("2020-01-01", periods=300))
    sharpe = m.rolling_sharpe(r, window=60)
    assert sharpe.iloc[:39].isna().all() and not np.isnan(sharpe.iloc[39])                                    # at least 40 observations (the floor), not 30 (half of 60)
    window = r.iloc[100:160]
    assert sharpe.iloc[159] == pytest.approx(window.mean() / window.std(ddof=1) * math.sqrt(252))
    assert m.rolling_sharpe(r, window=60, periods_per_year=12).iloc[159] == pytest.approx(window.mean() / window.std(ddof=1) * math.sqrt(12))
    long = m.rolling_sharpe(r, window=252)
    assert long.iloc[:125].isna().all() and not np.isnan(long.iloc[125])                                     # half of 252
    assert m.rolling_sharpe(r).dropna().equals(long.dropna()) and m.rolling_sharpe(r).iloc[125] == long.iloc[125]             # a year is the default window
    vol = m.rolling_volatility(r)
    assert vol.iloc[:30].isna().all() and not np.isnan(vol.iloc[30])                                          # half of 63, rounded down
    assert vol.iloc[100] == pytest.approx(r.iloc[38:101].std(ddof=1) * math.sqrt(252))                        # the 63 days ending on day 100
    short = m.rolling_volatility(r, window=20)
    assert short.iloc[:19].isna().all() and not np.isnan(short.iloc[19])                                      # never fewer than 20
    still = pd.Series(0.001, index=r.index)
    assert m.rolling_sharpe(still, window=60).isna().all()                                                    # no volatility: no ratio
    assert m.rolling_sharpe(pd.Series(0.0004, index=r.index), window=60).isna().all()


def test_calendar_returns_compound_within_each_year_and_month():
    index = pd.DatetimeIndex(["2020-11-30", "2020-12-31", "2021-01-29", "2021-01-31", "2021-02-26"])
    r = pd.Series([0.10, 0.05, -0.02, 0.03, 0.01], index=index)
    yearly = m.annual_returns(r)
    assert yearly.loc[2020] == pytest.approx(1.10 * 1.05 - 1.0) and yearly.loc[2021] == pytest.approx(0.98 * 1.03 * 1.01 - 1.0)
    monthly = m.monthly_return_table(r)
    assert monthly.loc[2020, 11] == pytest.approx(0.10) and monthly.loc[2021, 1] == pytest.approx(0.98 * 1.03 - 1.0) and monthly.loc[2021, 2] == pytest.approx(0.01)
    assert monthly.index.name == "year" and monthly.columns.name == "month" and math.isnan(monthly.loc[2021, 11])
    assert m.annual_returns(pd.Series([np.nan, 0.1], index=pd.DatetimeIndex(["2019-05-01", "2020-05-01"]))).to_dict() == {2020: pytest.approx(0.1)}


# ------------------------------------------------------------------------------------------------------------------------ the deflated and probabilistic Sharpe ratios
def expected_max(n: int) -> float:
    g = 0.5772156649015329
    return (1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e))


def test_one_trial_has_nothing_to_be_selected_from_so_the_deflated_ratio_is_the_probabilistic_one_against_zero():
    for sharpe in (-1.0, -0.2, 0.0, 0.5, 1.0, 2.0):
        assert m.deflated_sharpe_ratio(sharpe, 1, 1000) == pytest.approx(m.probabilistic_sharpe_ratio(sharpe, 0.0, 1000))
    assert m.deflated_sharpe_ratio(-1.0, 1, 1000) < 0.05 and m.deflated_sharpe_ratio(0.0, 1, 1000) == pytest.approx(0.5)      # a losing strategy tried once is not "certain" to survive
    assert m.deflated_sharpe_ratio(0.0, 1, 1000) < m.deflated_sharpe_ratio(0.5, 1, 1000) < m.deflated_sharpe_ratio(1.0, 1, 1000)


def test_the_benchmark_rises_with_the_number_of_trials_and_the_ratio_is_the_probability_of_clearing_it():
    sr, n_obs, ppy = 1.0, 1000, 252
    for trials in (2, 3, 10, 100):
        per_period = sr / math.sqrt(ppy)
        variance = (1.0 - 0.0 * per_period + (3.0 - 1.0) / 4.0 * per_period ** 2) / (n_obs - 1)
        expected = stats.norm.cdf((per_period - expected_max(trials) / math.sqrt(n_obs)) / math.sqrt(variance))
        assert m.deflated_sharpe_ratio(sr, trials, n_obs) == pytest.approx(expected)
    values = [m.deflated_sharpe_ratio(sr, n, n_obs) for n in (1, 2, 3, 10, 100, 1000)]
    assert values == sorted(values, reverse=True) and len(set(values)) == len(values)
    assert expected_max(2) > 0 and expected_max(100) > expected_max(10)


def test_skew_and_kurtosis_move_the_deflated_and_probabilistic_ratios_through_the_variance_of_the_estimate():
    base = m.deflated_sharpe_ratio(2.0, 3, 500)                                                                  # a Sharpe ratio well clear of what three trials would show by chance
    assert m.deflated_sharpe_ratio(2.0, 3, 500, skew=-1.0) < base < m.deflated_sharpe_ratio(2.0, 3, 500, skew=1.0)             # a negative skew widens the estimate's error, so surviving is less certain
    assert m.deflated_sharpe_ratio(2.0, 3, 500, kurtosis=9.0) < base                                            # fat tails the same
    below = m.deflated_sharpe_ratio(1.0, 10, 500)                                                                # and the other way round when the Sharpe ratio is short of the benchmark
    assert below < 0.5 and m.deflated_sharpe_ratio(1.0, 10, 500, skew=-1.0) > below
    sr, ppy = 1.0 / math.sqrt(252), 252
    assert m.probabilistic_sharpe_ratio(1.0, 0.0, 500, skew=-1.0) == pytest.approx(stats.norm.cdf(sr / math.sqrt((1 + sr + 0.5 * sr ** 2) / 499)))
    assert m.probabilistic_sharpe_ratio(1.0, 0.5, 500) == pytest.approx(stats.norm.cdf((sr - 0.5 / math.sqrt(252)) / math.sqrt((1 + 0.5 * sr ** 2) / 499)))
    assert m.probabilistic_sharpe_ratio(1.0, 0.0, 500, periods_per_year=12) != m.probabilistic_sharpe_ratio(1.0, 0.0, 500)
    assert ppy == 252


def test_the_deflated_and_probabilistic_ratios_say_nothing_where_they_cannot():
    assert math.isnan(m.deflated_sharpe_ratio(1.0, 0, 1000)) and math.isnan(m.deflated_sharpe_ratio(1.0, 5, 9)) and math.isnan(m.deflated_sharpe_ratio(float("nan"), 5, 1000))
    assert not math.isnan(m.deflated_sharpe_ratio(1.0, 5, 10))                                                  # ten observations are the least it accepts
    assert math.isnan(m.deflated_sharpe_ratio(1.0, 5, 1000, skew=2e3))                                           # an estimate with a negative variance is no estimate
    assert math.isnan(m.probabilistic_sharpe_ratio(1.0, 0.0, 9)) and math.isnan(m.probabilistic_sharpe_ratio(float("inf"), 0.0, 1000))
    assert math.isnan(m.probabilistic_sharpe_ratio(1.0, 0.0, 1000, skew=2e3))
    assert not math.isnan(m.probabilistic_sharpe_ratio(1.0, 0.0, 10))
