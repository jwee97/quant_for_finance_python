"""The risk and performance measures of the Backtest page against independent calculations (numpy loops, hand-worked paths, closed forms)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.risk import performance as rs


def series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


@pytest.fixture(scope="module")
def r():
    rng = np.random.default_rng(3)
    return series(rng.normal(0.0004, 0.01, 800) + np.where(rng.random(800) < 0.03, -0.04, 0.0))


def loop_downside(values):
    return math.sqrt(sum(min(v, 0.0) ** 2 for v in values) / len(values))


def test_sortino_is_annualised_mean_over_the_root_mean_square_of_losses(r):
    m = rs.risk_metrics(r)
    expected = math.sqrt(252) * r.mean() / loop_downside(r.tolist())
    assert m["sortino"] == pytest.approx(expected, rel=1e-12)
    assert m["daily_downside_deviation"] == pytest.approx(loop_downside(r.tolist()), rel=1e-12)


def test_calmar_martin_recovery_and_the_ulcer_index_follow_the_drawdown_path(r):
    m = rs.risk_metrics(r)
    equity, peak, worst, squares = 1.0, 1.0, 0.0, 0.0
    for v in r:
        equity *= 1.0 + v
        peak = max(peak, equity)
        d = equity / peak - 1.0
        worst, squares = min(worst, d), squares + (100.0 * d) ** 2
    cagr = equity ** (252.0 / len(r)) - 1.0
    ulcer = math.sqrt(squares / len(r))
    assert m["calmar"] == pytest.approx(cagr / -worst, rel=1e-12)
    assert m["ulcer_index"] == pytest.approx(ulcer, rel=1e-12) and m["martin_ratio"] == pytest.approx(100.0 * cagr / ulcer, rel=1e-12)
    assert m["recovery_factor"] == pytest.approx((equity - 1.0) / -worst, rel=1e-12)


def test_omega_is_total_gains_over_total_losses_and_none_without_a_loss(r):
    assert rs.risk_metrics(r)["omega"] == pytest.approx(sum(v for v in r if v > 0) / sum(-v for v in r if v < 0), rel=1e-12)
    up = series([0.001 * (1 + (i % 3)) for i in range(60)])
    m = rs.risk_metrics(up)
    assert m["omega"] is None and m["calmar"] is None and m["sortino"] is None and m["ulcer_index"] == 0.0


def test_tail_ratio_var_and_expected_shortfall_from_the_empirical_quantiles():
    values = np.linspace(-0.05, 0.05, 101)
    m = rs.risk_metrics(series(values))
    q05, q95, q01 = np.quantile(values, 0.05), np.quantile(values, 0.95), np.quantile(values, 0.01)
    assert m["tail_ratio"] == pytest.approx(abs(q95) / abs(q05), rel=1e-12) and m["tail_ratio"] == pytest.approx(1.0)
    assert m["var_95"] == pytest.approx(-q05) and m["var_99"] == pytest.approx(-q01)
    assert m["es_95"] == pytest.approx(-values[values <= q05].mean())
    assert m["es_95"] > m["var_95"]


def test_gain_to_pain_is_the_sum_of_monthly_returns_over_the_sum_of_the_losing_ones():
    totals = {"2020-01": 0.04, "2020-02": -0.02, "2020-03": 0.01, "2020-04": -0.01}           # months built to these returns exactly
    parts = []
    for month, total in totals.items():
        idx = pd.bdate_range(month + "-01", pd.Period(month).end_time.normalize())
        parts.append(pd.Series((1.0 + total) ** (1.0 / len(idx)) - 1.0, index=idx))
    net = pd.concat(parts)
    monthly = (1.0 + net).groupby([net.index.year, net.index.month]).prod() - 1.0
    assert monthly.to_numpy() == pytest.approx(list(totals.values()), abs=1e-12)
    assert rs.risk_metrics(net)["gain_to_pain"] == pytest.approx((0.04 - 0.02 + 0.01 - 0.01) / (0.02 + 0.01), rel=1e-9)


def test_cornish_fisher_equals_the_normal_value_without_skew_or_excess_kurtosis_and_the_formula_otherwise():
    z = norm.ppf(0.05)
    assert rs.cornish_fisher_var(0.0005, 0.01, 0.0, 0.0, 0.95) == pytest.approx(-(0.0005 + 0.01 * z), rel=1e-12)
    s, k = -0.8, 3.0
    zcf = z + (z * z - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s * s / 36
    assert rs.cornish_fisher_var(0.0005, 0.01, s, k, 0.95) == pytest.approx(-(0.0005 + 0.01 * zcf), rel=1e-12)
    assert rs.cornish_fisher_var(0.0, 0.01, -0.8, 3.0) > rs.cornish_fisher_var(0.0, 0.01, 0.0, 0.0)          # fat tails and negative skew make the loss larger


def test_kelly_leverage_is_mean_over_variance(r):
    assert rs.risk_metrics(r)["kelly_leverage"] == pytest.approx(r.mean() / r.var(ddof=1), rel=1e-12)


def test_minimum_track_record_length_matches_the_formula_and_is_none_for_a_non_positive_sharpe():
    sr, skew, kurt = 0.1, 0.0, 3.0
    obs = 1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) * (norm.ppf(0.95) / sr) ** 2
    assert rs.min_track_record_years(sr, skew, kurt) == pytest.approx(obs / 252.0, rel=1e-12)
    assert rs.min_track_record_years(0.0, 0.0, 3.0) is None and rs.min_track_record_years(-0.1, 0.0, 3.0) is None
    assert rs.min_track_record_years(0.1, 0.0, 3.0, benchmark=0.05) > rs.min_track_record_years(0.1, 0.0, 3.0)


def test_a_short_series_has_no_measures():
    assert rs.risk_metrics(series([0.01] * 10)) == {}


# ------------------------------------------------------------------------------------------------------------------ drawdowns
def returns_through(levels):
    levels = [1.0] + list(levels)
    return series([levels[i + 1] / levels[i] - 1.0 for i in range(len(levels) - 1)])


def test_drawdown_episodes_are_peak_to_recovery_deepest_first():
    net = returns_through([1.10, 1.20, 1.00, 0.90, 1.00, 1.25, 1.10, 1.15])
    ep = rs.drawdown_episodes(net)
    d = net.index
    assert [e["depth"] for e in ep] == pytest.approx([0.90 / 1.20 - 1.0, 1.10 / 1.25 - 1.0])
    first, second = ep
    assert (first["peak"], first["trough"], first["recovered"]) == (str(d[1].date()), str(d[3].date()), str(d[5].date()))
    assert first["days_to_trough"] == (d[3] - d[1]).days and first["days"] == (d[5] - d[1]).days
    assert (second["peak"], second["trough"], second["recovered"]) == (str(d[5].date()), str(d[6].date()), None)
    assert second["days"] == (d[7] - d[5]).days                                    # an open fall runs to the last day
    assert len(rs.drawdown_episodes(net, limit=1)) == 1


def test_a_loss_on_the_first_day_is_a_fall_from_the_starting_account():
    net = returns_through([0.95, 0.97, 1.02])
    ep = rs.drawdown_episodes(net)
    assert len(ep) == 1 and ep[0]["depth"] == pytest.approx(-0.05) and ep[0]["recovered"] == str(net.index[2].date())


def test_a_series_that_never_falls_has_no_episode_and_the_depth_agrees_with_the_drawdown_series(r):
    assert rs.drawdown_episodes(series([0.001] * 40)) == []
    ep = rs.drawdown_episodes(r, limit=3)
    assert ep[0]["depth"] == pytest.approx(rs.drawdown_series(r).min(), rel=1e-12)
    assert [e["depth"] for e in ep] == sorted(e["depth"] for e in ep)
    assert all(e["peak"] <= e["trough"] and (e["recovered"] is None or e["trough"] <= e["recovered"]) for e in ep)


# ------------------------------------------------------------------------------------------------------------------ rolling and calendar
def test_rolling_volatility_is_the_annualised_standard_deviation_of_the_window(r):
    v = rs.rolling_volatility(r, 63)
    assert v.iloc[:62].isna().all() and v.iloc[62] == pytest.approx(np.std(r.iloc[:63], ddof=1) * math.sqrt(252), rel=1e-12)
    assert v.iloc[-1] == pytest.approx(np.std(r.iloc[-63:], ddof=1) * math.sqrt(252), rel=1e-12)


def test_rolling_beta_of_twice_the_benchmark_is_two_and_matches_the_regression_slope(r):
    rng = np.random.default_rng(8)
    b = series(rng.normal(0.0003, 0.008, len(r)))
    assert rs.rolling_beta(2.0 * b, b, 100).dropna().to_numpy() == pytest.approx(2.0, rel=1e-9)
    mixed = 0.7 * b + series(rng.normal(0, 0.004, len(r)))
    beta = rs.rolling_beta(mixed, b, 252)
    window = slice(len(r) - 252, len(r))
    assert beta.iloc[-1] == pytest.approx(np.polyfit(b.iloc[window], mixed.iloc[window], 1)[0], rel=1e-9)
    assert beta.iloc[:251].isna().all()


def test_calendar_month_profile_collects_each_month_of_the_year():
    rows = []
    for year, (jan, feb) in {2020: (0.03, -0.01), 2021: (-0.01, -0.02), 2022: (0.05, 0.01)}.items():
        for month, total in ((1, jan), (2, feb)):
            idx = pd.bdate_range(f"{year}-{month:02d}-01", periods=20)
            rows.append(pd.Series((1.0 + total) ** (1.0 / 20) - 1.0, index=idx))
    net = pd.concat(rows)
    prof = {p["month"]: p for p in rs.calendar_month_profile(net)}
    assert sorted(prof) == [1, 2] and prof[1]["n"] == 3
    assert prof[1]["mean"] == pytest.approx((0.03 - 0.01 + 0.05) / 3, rel=1e-9) and prof[1]["best"] == pytest.approx(0.05) and prof[1]["worst"] == pytest.approx(-0.01)
    assert prof[1]["hit_rate"] == pytest.approx(2 / 3) and prof[2]["median"] == pytest.approx(-0.01) and prof[2]["hit_rate"] == pytest.approx(1 / 3)


# ------------------------------------------------------------------------------------------------------------------ against a benchmark
def test_benchmark_relative_measures_for_a_levered_copy_of_the_benchmark():
    rng = np.random.default_rng(5)
    b = series(rng.normal(0.0003, 0.01, 500))
    r = 1.5 * b + 0.0002
    out = rs.benchmark_relative(r, b)
    assert out["beta"] == pytest.approx(1.5, rel=1e-9) and out["correlation"] == pytest.approx(1.0, abs=1e-12)
    assert out["alpha_annual"] == pytest.approx(0.0002 * 252, rel=1e-9)
    active = r - b
    assert out["tracking_error"] == pytest.approx(active.std(ddof=1) * math.sqrt(252), rel=1e-12)
    assert out["information_ratio"] == pytest.approx(active.mean() * 252 / (active.std(ddof=1) * math.sqrt(252)), rel=1e-12)
    up, down = b > 0, b < 0
    assert out["up_capture"] == pytest.approx(r[up].mean() / b[up].mean(), rel=1e-12) and out["down_capture"] == pytest.approx(r[down].mean() / b[down].mean(), rel=1e-12)
    assert out["up_capture"] > 1.4 > 1.0 and out["n_days"] == 500


def test_benchmark_relative_handles_gaps_and_short_overlaps():
    rng = np.random.default_rng(6)
    b = series(rng.normal(0, 0.01, 200))
    r = b.copy()
    r.iloc[:120] = np.nan
    assert rs.benchmark_relative(r, b)["n_days"] == 80
    r.iloc[:190] = np.nan
    assert rs.benchmark_relative(r, b) == {}
    assert rs.benchmark_relative(b * 0.0, b)["correlation"] is None                # a flat strategy has no correlation


# ------------------------------------------------------------------------------------------------------------------ the monthly cells and the distribution of days
def test_monthly_returns_compound_each_calendar_month_and_skip_nothing():
    net = series([0.01, -0.02, 0.03] + [0.0] * 20 + [0.05, 0.05], start="2021-01-27")           # 27 January 2021 is a Wednesday
    got = rs.monthly_returns(net)
    by_month = {}
    for date, value in net.items():
        by_month.setdefault((date.year, date.month), []).append(value)
    expected = {key: math.prod(1.0 + v for v in values) - 1.0 for key, values in by_month.items()}
    assert [(c["year"], c["month"]) for c in got] == sorted(expected) and len(got) >= 2
    for cell in got:
        assert cell["value"] == pytest.approx(expected[(cell["year"], cell["month"])], abs=5e-6)         # rounded to five decimals
    assert rs.monthly_returns(series([])) == []


def test_the_return_distribution_bins_the_central_99_percent_and_counts_what_a_normal_would():
    rng = np.random.default_rng(11)
    values = rng.standard_t(4, 1500) * 0.008
    out = rs.return_distribution(series(values), bins=41)
    srt = sorted(values)
    n = len(values)
    lo, hi = (srt[int(math.floor(p * (n - 1) + 0.5))] for p in (0.005, 0.995))                         # the 0.5th and 99.5th percentiles, rounded half up
    step = (hi - lo) / 41
    assert out["edges"][0] == pytest.approx(lo, rel=1e-12) and out["edges"][-1] == pytest.approx(hi, rel=1e-12) and len(out["edges"]) == 42 and len(out["counts"]) == 41
    loop = [0] * 41
    for v in values:
        if lo <= v <= hi:
            loop[min(40, int((v - lo) // step))] += 1
    assert out["counts"] == loop and sum(loop) >= 0.98 * n
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    cdf = lambda x: 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))                           # noqa: E731
    for i, expected in enumerate(n * (cdf(out["edges"][i + 1]) - cdf(out["edges"][i])) for i in range(41)):
        assert out["normal"][i] == pytest.approx(expected, abs=1e-3)
    assert out["mean"] == pytest.approx(mean, rel=1e-12) and out["sd"] == pytest.approx(sd, rel=1e-12) and out["n"] == n
    assert out["counts"][20] > 1.1 * out["normal"][20]                                                    # fat tails: the centre holds more days than the bell curve with the same spread


def test_the_return_distribution_is_empty_for_a_short_history_and_survives_a_flat_one():
    assert rs.return_distribution(series(np.random.default_rng(1).normal(0, 0.01, 59))) == {}
    flat = rs.return_distribution(series([0.0] * 100))
    assert flat["n"] == 100 and sum(flat["counts"]) == 100 and flat["normal"] == [0.0] * 41
