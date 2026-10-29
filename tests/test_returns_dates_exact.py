"""Return arithmetic, descriptive statistics and the calendar helpers, checked against numbers worked out by hand.

A planted-bug audit (``experiments/plant_bugs.py``) found these modules weakly pinned: of the places where one operator, comparison or number of ``features/returns.py`` was changed the tests noticed
13% (8 of 63), and of ``utils/dates.py`` 19% (6 of 32). Every window, fold and backtest is cut with ``slice_dates`` and ``rebalance_dates``, and every equity curve and worst fall goes through
``cumulative_returns`` and ``drawdown_table``, so what they do at their edges is stated here and not left to a review.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from src.features.returns import (
    FLAT,
    annualised_return,
    annualised_volatility,
    autocorrelation,
    average_pairwise_correlation,
    cumulative_returns,
    describe_returns,
    drawdown,
    drawdown_table,
    forward_returns,
    ljung_box,
    max_drawdown,
    rolling_correlation,
)
from src.utils.dates import DateWindow, ann_factor, ann_vol_scale, rebalance_dates, slice_dates, to_timestamp, windows_from_config, year_fraction


# ------------------------------------------------------------------------------------------------------------------ dates
def test_a_config_date_is_a_timestamp_and_nothing_or_null_means_open_ended():
    for empty in (None, "", "null", "NULL", "None", "none"):
        assert to_timestamp(empty) is None, empty
    assert to_timestamp("2020-03-05") == pd.Timestamp("2020-03-05") and to_timestamp(pd.Timestamp("2021-01-01")) == pd.Timestamp("2021-01-01")
    assert ann_factor() == 252.0 and ann_vol_scale() == pytest.approx(np.sqrt(252)) and ann_factor(12) == 12.0 and ann_vol_scale(12) == pytest.approx(np.sqrt(12))


def test_a_date_slice_includes_both_ends_and_tolerates_an_open_end():
    index = pd.bdate_range("2021-01-04", periods=10)                                          # Monday 4 January to Friday 15 January
    frame = pd.DataFrame({"a": range(10)}, index=index)
    both = slice_dates(frame, "2021-01-06", "2021-01-12")
    assert both.index[0] == pd.Timestamp("2021-01-06") and both.index[-1] == pd.Timestamp("2021-01-12") and len(both) == 5, "a bound that is a trading day is inside the slice"
    assert len(slice_dates(frame, "2021-01-06", None)) == 8 and len(slice_dates(frame, None, "2021-01-06")) == 3 and len(slice_dates(frame, None, None)) == 10
    assert len(slice_dates(frame["a"], "2021-01-09", "2021-01-10")) == 0, "a window of a weekend holds nothing"
    series = slice_dates(frame["a"], pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-05"))
    assert isinstance(series, pd.Series) and series.tolist() == [1], "a one-day window is that day"
    assert len(slice_dates(frame, "null", "none")) == 10


def test_the_rebalance_dates_are_the_last_day_observed_in_each_period():
    index = pd.DatetimeIndex(["2021-01-04", "2021-01-05", "2021-01-08", "2021-01-12", "2021-01-29", "2021-02-01", "2021-02-26", "2021-04-06", "2022-01-03"])
    assert list(rebalance_dates(index, "daily")) == list(index) and list(rebalance_dates(index, "d")) == list(index) and list(rebalance_dates(index, "DAILY")) == list(index)
    assert [d.date().isoformat() for d in rebalance_dates(index, "monthly")] == ["2021-01-29", "2021-02-26", "2021-04-06", "2022-01-03"], "the last observed day, not the calendar month end"
    assert [d.date().isoformat() for d in rebalance_dates(index, "weekly")] == ["2021-01-08", "2021-01-12", "2021-01-29", "2021-02-01", "2021-02-26", "2021-04-06", "2022-01-03"]
    assert [d.date().isoformat() for d in rebalance_dates(index, "quarterly")] == ["2021-02-26", "2021-04-06", "2022-01-03"]
    assert [d.date().isoformat() for d in rebalance_dates(index, "annual")] == ["2021-04-06", "2022-01-03"] == [d.date().isoformat() for d in rebalance_dates(index, "yearly")]
    assert list(rebalance_dates(index[::-1], "monthly")) == list(rebalance_dates(index, "monthly")), "the order of the index does not matter"
    with pytest.raises(ValueError, match="unsupported rebalance frequency 'hourly'"):
        rebalance_dates(index, "hourly")


def test_a_year_fraction_is_the_calendar_length_in_years_and_a_single_day_has_none():
    assert year_fraction(pd.DatetimeIndex(["2020-01-01", "2021-01-01"])) == pytest.approx(366 / 365.25)
    assert np.isnan(year_fraction(pd.DatetimeIndex(["2020-01-01"]))) and np.isnan(year_fraction(pd.DatetimeIndex([])))
    assert year_fraction(pd.DatetimeIndex(["2020-01-01", "2020-01-02"])) == pytest.approx(1 / 365.25)


def test_a_named_window_includes_its_ends_and_describes_itself():
    window = DateWindow.from_config("stress_2020", {"start": "2020-02-19", "end": "2020-03-23"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        window.name = "other"
    assert window.label == "Stress 2020" and window.describe() == "Stress 2020 [2020-02-19 .. 2020-03-23]"
    assert window.contains("2020-02-19") and window.contains("2020-03-23") and window.contains("2020-03-01")
    assert not window.contains("2020-02-18") and not window.contains("2020-03-24"), "one day outside either end is outside"
    open_start = DateWindow.from_config("early", {"end": "2010-12-31", "label": "The early years"})
    assert open_start.label == "The early years" and open_start.describe() == "The early years [-inf .. 2010-12-31]" and open_start.contains("1990-01-01") and not open_start.contains("2011-01-01")
    open_end = DateWindow.from_config("late", {"start": "2020-01-01"})
    assert open_end.describe() == "Late [2020-01-01 .. now]" and open_end.contains("2099-01-01") and not open_end.contains("2019-12-31")
    frame = pd.Series(range(5), index=pd.bdate_range("2020-03-18", periods=5))
    assert window.apply(frame).tolist() == [0, 1, 2, 3] and DateWindow.from_config("x", {"start": "2020-03-20"}).apply(frame).tolist() == [2, 3, 4]
    windows = windows_from_config({"a": {"start": "2020-01-01"}, "b": None})
    assert set(windows) == {"a", "b"} and windows["b"].start is None and windows["b"].end is None and windows_from_config(None) == {} and windows_from_config({}) == {}


# ------------------------------------------------------------------------------------------------------------------ cumulative returns, drawdowns
def test_cumulative_returns_start_from_the_initial_value_and_treat_a_missing_return_as_no_position():
    r = pd.Series([0.10, np.nan, -0.10], index=pd.bdate_range("2020-01-01", periods=3))
    assert cumulative_returns(r).tolist() == pytest.approx([1.1, 1.1, 0.99]) and cumulative_returns(r, initial=100.0).tolist() == pytest.approx([110.0, 110.0, 99.0])
    both = cumulative_returns(pd.DataFrame({"a": [0.1, 0.1], "b": [0.0, -0.5]}, index=pd.bdate_range("2020-01-01", periods=2)))
    assert both["a"].tolist() == pytest.approx([1.1, 1.21]) and both["b"].tolist() == pytest.approx([1.0, 0.5])


def test_forward_returns_compound_or_add_over_the_horizon_and_the_last_rows_are_not_known():
    index = pd.bdate_range("2020-01-01", periods=6)
    r = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.03, 0.0, 0.05]}, index=index)
    two = forward_returns(r, 2)["A"]
    assert two.iloc[0] == pytest.approx(1.02 * 0.99 - 1) and two.iloc[3] == pytest.approx(1.0 * 1.05 - 1) and two.iloc[4:].isna().all() and two.iloc[:4].notna().all()
    added = forward_returns(r, 2, compound=False)["A"]
    assert added.iloc[0] == pytest.approx(0.02 - 0.01) and added.iloc[1] == pytest.approx(-0.01 + 0.03)
    assert forward_returns(r, 1)["A"].iloc[:5].tolist() == pytest.approx(r["A"].iloc[1:].tolist()) and forward_returns(r, 1)["A"].iloc[5:].isna().all(), "a horizon of one day is allowed"
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        forward_returns(r, 0)


def test_drawdown_is_the_distance_below_the_running_high_and_max_drawdown_its_deepest_point():
    curve = pd.Series([100.0, 110.0, 99.0, 88.0, 100.0, 110.0, 121.0], index=pd.date_range("2020-01-01", periods=7))
    assert drawdown(curve).tolist() == pytest.approx([0.0, 0.0, -0.1, -0.2, 100 / 110 - 1.0, 0.0, 0.0])
    assert max_drawdown(curve) == pytest.approx(-0.2)
    frame = pd.DataFrame({"a": curve, "b": curve * 2.0})
    assert drawdown(frame)["b"].tolist() == pytest.approx(drawdown(curve).tolist())


def test_the_drawdown_table_lists_each_episode_with_its_peak_trough_recovery_and_lengths():
    curve = pd.Series([100, 110, 99, 88, 100, 110, 121, 100, 90, 95, 99, 90], index=pd.date_range("2020-01-01", periods=12), dtype=float)
    table = drawdown_table(curve)
    assert list(table.columns) == ["start", "trough", "end", "depth", "length_days", "recovery_days"] and len(table) == 2
    deep, first = table.iloc[0], table.iloc[1]
    assert deep["start"] == pd.Timestamp("2020-01-07") and deep["trough"] == pd.Timestamp("2020-01-09") and pd.isna(deep["end"]) and deep["depth"] == pytest.approx(90 / 121 - 1)
    assert pd.isna(deep["length_days"]) and pd.isna(deep["recovery_days"]), "an episode that has not recovered has no length yet"
    assert first["start"] == pd.Timestamp("2020-01-02") and first["trough"] == pd.Timestamp("2020-01-04") and first["end"] == pd.Timestamp("2020-01-06") and first["depth"] == pytest.approx(-0.2)
    assert first["length_days"] == 4 and first["recovery_days"] == 2, "from the peak on 2 January to the recovery on 6 January, and from the trough on 4 January"
    assert list(drawdown_table(curve, top=1)["depth"]) == pytest.approx([90 / 121 - 1]) and drawdown_table(curve, top=1).index.tolist() == [0]
    rising = drawdown_table(pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2020-01-01", periods=3)))
    assert rising.empty and list(rising.columns) == ["start", "trough", "end", "depth", "length_days", "recovery_days"]
    saw = []                                                                                     # seven separate falls, each a little deeper than the one before
    level = 100.0
    for depth in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07):
        saw += [level, level * (1 - depth)]
        level *= 1.0
    seven = pd.Series(saw + [saw[-2] + 1.0], index=pd.date_range("2020-01-01", periods=len(saw) + 1))
    assert len(drawdown_table(seven)) == 5 and drawdown_table(seven)["depth"].tolist() == pytest.approx([-0.07, -0.06, -0.05, -0.04, -0.03]), "the five deepest by default"
    assert len(drawdown_table(seven, top=7)) == 7
    starts_low = drawdown_table(pd.Series([90.0, 95.0, 100.0], index=pd.date_range("2020-01-01", periods=3)))
    assert starts_low.empty, "a curve that only rises has no drawdown"
    from_the_first_day = drawdown_table(pd.Series([100.0, 90.0, 100.0], index=pd.date_range("2020-01-01", periods=3)))
    assert from_the_first_day.iloc[0]["start"] == pd.Timestamp("2020-01-01") and from_the_first_day.iloc[0]["end"] == pd.Timestamp("2020-01-03")


# ------------------------------------------------------------------------------------------------------------------ annualising
def test_the_annual_return_compounds_the_whole_period_to_a_year_and_the_volatility_scales_by_the_root_of_the_year():
    two_halves = pd.Series([0.1, 0.1], index=pd.bdate_range("2020-01-01", periods=2))
    assert annualised_return(two_halves, 2) == pytest.approx(0.21) and annualised_return(two_halves, 4) == pytest.approx(1.21 ** 2 - 1)
    assert np.isnan(annualised_return(pd.Series([], dtype=float)))
    assert annualised_return(pd.Series([0.01, np.nan, 0.01]), 2) == pytest.approx(1.0201 - 1), "a missing return is not a period"
    r = pd.Series([0.01, -0.01, 0.01, -0.01])
    assert annualised_volatility(r) == pytest.approx(0.0115470054 * np.sqrt(252)) and annualised_volatility(r, 12) == pytest.approx(0.0115470054 * np.sqrt(12))
    assert annualised_volatility(pd.Series(0.0004, index=range(50))) == 0.0 and annualised_volatility(pd.Series(0.0, index=range(5))) == 0.0, "a series that never moves has no volatility, not rounding noise"
    assert FLAT == 1e-12


def make_returns(n: int = 80, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": rng.normal(0.0005, 0.01, n), "B": rng.normal(0.0, 0.02, n)}, index=pd.bdate_range("2020-01-01", periods=n))


def test_the_description_of_a_return_series_states_each_statistic_it_names():
    r = make_returns()
    table = describe_returns(r)
    a = r["A"]
    row = table.loc["A"]
    losses = a[a < 0.0]
    curve = (1.0 + a).cumprod()
    assert row["n_obs"] == 80 and row["mean_ann"] == pytest.approx(a.mean() * 252) and row["vol_ann"] == pytest.approx(a.std(ddof=1) * np.sqrt(252))
    assert row["cagr"] == pytest.approx((1.0 + a).prod() ** (252 / 80) - 1) and row["sharpe_naive"] == pytest.approx(row["cagr"] / row["vol_ann"])
    assert row["min"] == a.min() and row["max"] == a.max() and row["median"] == pytest.approx(np.median(a)) and row["q01"] == pytest.approx(np.quantile(a, 0.01)) and row["q99"] == pytest.approx(np.quantile(a, 0.99))
    assert row["hit_rate"] == pytest.approx((a > 0).mean()) and row["downside_vol_ann"] == pytest.approx(losses.std(ddof=1) * np.sqrt(252)) and row["max_drawdown"] == pytest.approx((curve / curve.cummax() - 1).min())
    assert row["autocorr_lag1"] == pytest.approx(np.corrcoef(a[:-1], a[1:])[0, 1])
    d = a.to_numpy() - a.mean()
    assert row["skew"] == pytest.approx(np.sqrt(80 * 79) / 78 * np.mean(d ** 3) / np.mean(d ** 2) ** 1.5), "the adjusted Fisher-Pearson skewness"
    assert 0 < row["jb_pvalue"] < 1 and row["jarque_bera"] >= 0
    assert row["excess_kurtosis"] == pytest.approx(a.kurt()), "the unbiased excess kurtosis, as pandas states it"
    assert describe_returns(r.iloc[:29]).empty and len(describe_returns(r.iloc[:30])) == 2, "fewer than thirty observations is not described"
    annual = describe_returns(r, periods_per_year=12).loc["A"]
    assert annual["mean_ann"] == pytest.approx(a.mean() * 12) and annual["vol_ann"] == pytest.approx(a.std(ddof=1) * np.sqrt(12))
    sparse = r.copy()
    sparse["B"] = np.where(np.arange(80) % 40 == 7, -0.01, 0.01)
    assert np.isnan(describe_returns(sparse).loc["B", "downside_vol_ann"]), "two losing days are too few for a downside deviation"
    sparse["B"] = np.where(np.arange(80) % 27 == 7, -0.01, 0.01)
    assert abs(describe_returns(sparse).loc["B", "downside_vol_ann"]) < 1e-9, "three losing days, all alike, are enough for a deviation, and it is nil"
    with_zeros = pd.DataFrame({"Z": [0.01, 0.0, -0.02, 0.03, 0.0, -0.01, 0.02, 0.0, -0.03, 0.01] * 4}, index=pd.bdate_range("2020-01-01", periods=40))
    z = describe_returns(with_zeros).loc["Z"]
    assert z["hit_rate"] == pytest.approx(16 / 40), "a day of exactly zero is not a gain"
    assert z["downside_vol_ann"] == pytest.approx(np.std([-0.02, -0.01, -0.03] * 4, ddof=1) * np.sqrt(252)), "nor is it a loss: the downside is the losing days only"
    flat = pd.DataFrame({"C": np.full(40, 0.0003)}, index=pd.bdate_range("2020-01-01", periods=40))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                                       # no complaint from scipy about moments of data that are all alike
        still = describe_returns(flat).loc["C"]
    assert still["vol_ann"] == 0.0 and np.isnan(still["sharpe_naive"]), "a series that never moves has no Sharpe ratio, not 1e16 from rounding noise"
    assert np.isnan(still["skew"]) and np.isnan(still["excess_kurtosis"]) and np.isnan(still["jarque_bera"]) and np.isnan(still["jb_pvalue"]) and still["hit_rate"] == 1.0
    assert np.isnan(still["autocorr_lag1"]), "and nothing to correlate with the day before"


def test_autocorrelation_is_a_table_by_lag_and_asset():
    r = make_returns(120)
    table = autocorrelation(r, lags=5)
    assert list(table.index) == [1, 2, 3, 4, 5] and table.index.name == "lag" and list(table.columns) == ["A", "B"]
    for lag in (1, 3, 5):
        assert table.loc[lag, "A"] == pytest.approx(r["A"].autocorr(lag)) and table.loc[lag, "B"] == pytest.approx(r["B"].autocorr(lag))
    assert len(autocorrelation(r)) == 21, "a month of lags by default"
    alternating = pd.DataFrame({"S": np.tile([0.01, -0.01], 30)}, index=pd.bdate_range("2020-01-01", periods=60))
    assert autocorrelation(alternating, 2).loc[1, "S"] == pytest.approx(-1.0) and autocorrelation(alternating, 2).loc[2, "S"] == pytest.approx(1.0)
    flat = pd.DataFrame({"S": np.tile([0.01, -0.01], 30), "F": np.full(60, 0.0003)}, index=pd.bdate_range("2020-01-01", periods=60))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                                       # a division by zero inside pandas is a warning on some versions
        table = autocorrelation(flat, 3)
    assert table["F"].isna().all() and table["S"].notna().all(), "a series that never moves has no autocorrelation, and the other column is not touched"


def test_the_ljung_box_statistic_is_n_times_n_plus_2_times_the_sum_of_squared_autocorrelations_over_n_minus_lag():
    rng = np.random.default_rng(6)
    n = 300
    shocks = rng.normal(0, 1, n)
    vol = np.ones(n)
    for t in range(1, n):                                                                        # volatility clustering: big days follow big days
        vol[t] = 0.2 + 0.8 * abs(shocks[t - 1]) * 1.5
    r = pd.DataFrame({"X": shocks * vol * 0.01, "Y": rng.normal(0, 0.01, n)}, index=pd.bdate_range("2015-01-01", periods=n))
    table = ljung_box(r, lags=10)
    x = r["X"].to_numpy()

    def q(series: np.ndarray, lags: int) -> float:
        d = series - series.mean()
        rho = [float(np.sum(d[k:] * d[:-k]) / np.sum(d ** 2)) for k in range(1, lags + 1)]
        return len(d) * (len(d) + 2) * sum(rho[k - 1] ** 2 / (len(d) - k) for k in range(1, lags + 1))

    assert table.loc["X", "lb_stat_returns"] == pytest.approx(q(x, 10)) and table.loc["X", "lb_stat_squared"] == pytest.approx(q(x ** 2, 10))
    assert table.loc["X", "lb_stat_squared"] > table.loc["X", "lb_stat_returns"] and table.loc["X", "lb_pvalue_squared"] < 0.05, "clustered volatility shows in the squares, not in the returns"
    assert list(table.columns) == ["lb_stat_returns", "lb_pvalue_returns", "lb_stat_squared", "lb_pvalue_squared"]
    assert ljung_box(r.iloc[:49]).empty and len(ljung_box(r.iloc[:50])) == 2, "fewer than fifty observations is not tested"
    assert ljung_box(r).equals(ljung_box(r, lags=10)) and not ljung_box(r).equals(ljung_box(r, lags=5)), "ten lags unless told otherwise"
    assert ljung_box(r, lags=5).loc["Y", "lb_stat_returns"] == pytest.approx(q(r["Y"].to_numpy(), 5))


def test_rolling_correlation_is_against_the_reference_and_leaves_the_reference_out():
    r = make_returns(200)
    r["C"] = 0.6 * r["A"] + 0.4 * np.random.default_rng(1).normal(0, 0.01, 200)
    out = rolling_correlation(r, "A", window=60)
    assert list(out.columns) == ["B", "C"] and out.iloc[:59].isna().all().all() and out.iloc[59:].notna().all().all()
    last = r.iloc[-60:]
    assert out["C"].iloc[-1] == pytest.approx(np.corrcoef(last["A"], last["C"])[0, 1]) and out["B"].iloc[-1] == pytest.approx(np.corrcoef(last["A"], last["B"])[0, 1])
    assert rolling_correlation(r, "A").iloc[:125].isna().all().all() and rolling_correlation(r, "A").iloc[125:].notna().all().all(), "the default window is 126 days"
    with pytest.raises(KeyError, match="Z not in returns"):
        rolling_correlation(r, "Z")


def test_the_average_pairwise_correlation_is_the_mean_of_the_off_diagonal():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.01, 150)
    r = pd.DataFrame({"A": base + rng.normal(0, 0.005, 150), "B": base + rng.normal(0, 0.005, 150), "C": rng.normal(0, 0.01, 150)}, index=pd.bdate_range("2020-01-01", periods=150))
    out = average_pairwise_correlation(r, window=50)
    assert out.name == "avg_pairwise_corr" and out.index[0] == r.index[49] and len(out) == 101
    c = np.corrcoef(r.iloc[-50:].to_numpy().T)
    assert out.iloc[-1] == pytest.approx((c[0, 1] + c[0, 2] + c[1, 2]) / 3), "the mean of the three distinct pairs"
    assert average_pairwise_correlation(r[["A"]]).empty and average_pairwise_correlation(r).index[0] == r.index[125], "one asset has no pairs; the default window is 126 days"
    assert len(average_pairwise_correlation(r[["A", "B"]], window=50)) == 101 and average_pairwise_correlation(r[["A", "B"]], window=50).iloc[-1] == pytest.approx(np.corrcoef(r["A"].iloc[-50:], r["B"].iloc[-50:])[0, 1])
