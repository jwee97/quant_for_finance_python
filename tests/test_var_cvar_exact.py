"""Value at risk, conditional value at risk and their backtests against figures worked out by hand, or by a plainer second formula, one function at a time: the empirical quantile and the tail mean,
the closed forms (normal, Student-t, Cornish-Fisher), the simulation (its scale, its seed, its horizon, its joint draws), the rolling forecasts and the fact that they use only the past, and the
breach tests (Kupiec, Christoffersen) with the verdict they feed.

A planted-bug audit (``experiments/plant_bugs.py``) of ``src/risk/var.py`` found that the sixteen tests of ``tests/test_risk.py`` noticed only a quarter of the changes made to it; they check that the
methods agree with each other and behave sensibly, and several changes to a constant or an operator left every one of them passing. These check the numbers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from src.risk import cvar as C
from src.risk import var as V

LINE = pd.Series(np.arange(-50, 50) / 1000.0)                                      # 100 returns from -5.0% to +4.9%, a tenth of a percent apart
DAYS = pd.bdate_range("2010-01-01", periods=3000)


def noise(n=1500, seed=3, scale=0.01, loc=0.0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(loc, scale, n), index=DAYS[:n])


# ------------------------------------------------------------------------------------------------------------------------ historical VaR and CVaR
def test_historical_var_is_the_empirical_quantile_of_the_losses_scaled_by_the_root_of_the_horizon():
    # sorted, the 5% point is 95% of the way from -4.6% to -4.5%; the 1% point 99% of the way from -5.0% to -4.9%
    assert V.historical_var(LINE, 0.95) == pytest.approx(0.04505)
    assert V.historical_var(LINE, 0.99) == pytest.approx(0.04901)
    assert V.historical_var(LINE, 0.90) == pytest.approx(0.0401)
    assert V.historical_var(LINE, 0.95, horizon=4) == pytest.approx(2 * 0.04505) and V.historical_var(LINE, 0.95, horizon=9) == pytest.approx(3 * 0.04505)
    with_holes = pd.concat([LINE, pd.Series([np.nan] * 5)], ignore_index=True)
    assert V.historical_var(with_holes, 0.95) == pytest.approx(0.04505)
    assert math.isnan(V.historical_var(LINE.iloc[:19])) and not math.isnan(V.historical_var(LINE.iloc[:20]))      # twenty observations are the least it will quote


def test_a_return_that_is_itself_the_cutoff_belongs_to_the_tail():
    twenty_one = pd.Series(np.arange(-10, 11) / 100.0)                                  # the 5% point is the second smallest, -9%, exactly
    assert V.historical_var(twenty_one, 0.95) == pytest.approx(0.09)
    assert C.historical_cvar(twenty_one, 0.95) == pytest.approx(0.095)                  # the mean of -10% and -9%, not of -10% alone


def test_historical_cvar_is_the_mean_of_the_returns_at_or_below_the_cutoff():
    assert C.historical_cvar(LINE, 0.95) == pytest.approx(0.048)                        # the five worst: -5.0% to -4.6%
    assert C.historical_cvar(LINE, 0.90) == pytest.approx(0.0455)                       # the ten worst
    assert C.historical_cvar(LINE, 0.95, horizon=4) == pytest.approx(0.096)
    assert C.historical_cvar(LINE, 0.95) > V.historical_var(LINE, 0.95)
    assert math.isnan(C.historical_cvar(LINE.iloc[:19])) and not math.isnan(C.historical_cvar(LINE.iloc[:20]))
    assert C.historical_cvar(pd.concat([LINE, pd.Series([np.nan] * 3)], ignore_index=True), 0.95) == pytest.approx(0.048)


# ------------------------------------------------------------------------------------------------------------------------ closed forms
def test_normal_var_is_minus_the_mean_plus_the_normal_quantile_times_the_sample_deviation():
    r = pd.Series(np.tile([-0.01, 0.03], 50))
    expected = -(0.01 + stats.norm.ppf(0.05) * r.std(ddof=1))
    assert V.parametric_var(r, 0.95) == pytest.approx(expected) and V.parametric_var(r, 0.95, distribution="normal") == pytest.approx(expected)
    assert V.parametric_var(r, 0.99) == pytest.approx(-(0.01 + stats.norm.ppf(0.01) * r.std(ddof=1)))
    assert V.parametric_var(r, 0.95, horizon=9) == pytest.approx(3 * expected)
    assert math.isnan(V.parametric_var(r.iloc[:19])) and not math.isnan(V.parametric_var(r.iloc[:20]))
    with pytest.raises(ValueError, match="unknown distribution"):
        V.parametric_var(r, 0.95, distribution="laplace")
    with pytest.raises(ValueError, match="unknown distribution"):
        C.parametric_cvar(r, 0.95, distribution="laplace")


def test_student_t_var_uses_the_fitted_location_and_scale_and_is_larger_than_the_normal_one_in_the_tail_of_fat_tailed_data():
    r = pd.Series(stats.t.rvs(4, loc=0.0005, scale=0.01, size=6000, random_state=11))
    df, loc, scale = stats.t.fit(r.to_numpy())
    assert V.parametric_var(r, 0.99, distribution="t") == pytest.approx(-(loc + stats.t.ppf(0.01, df) * scale))
    assert V.parametric_var(r, 0.99, distribution="student_t") == V.parametric_var(r, 0.99, distribution="t")
    assert V.parametric_var(r, 0.99, 4, "t") == pytest.approx(2 * V.parametric_var(r, 0.99, 1, "t"))
    assert V.parametric_var(r, 0.99, distribution="t") > V.parametric_var(r, 0.99, distribution="normal")


def test_cornish_fisher_var_adjusts_the_normal_quantile_for_skew_and_excess_kurtosis_by_the_textbook_formula():
    rng = np.random.default_rng(5)
    r = pd.Series(-rng.gamma(2.0, 0.005, 3000) + 0.01)                                    # negatively skewed, thick on the left
    mu, sigma, s, k = r.mean(), r.std(ddof=1), stats.skew(r, bias=False), stats.kurtosis(r, bias=False)
    z = stats.norm.ppf(0.01)
    adjusted = z + (z * z - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s * s / 36
    assert V.cornish_fisher_var(r, 0.99) == pytest.approx(-(mu + adjusted * sigma))
    assert V.cornish_fisher_var(r, 0.99, horizon=4) == pytest.approx(2 * V.cornish_fisher_var(r, 0.99))
    assert V.cornish_fisher_var(r, 0.99) > V.parametric_var(r, 0.99)                       # the left tail is heavier than the normal one says
    assert math.isnan(V.cornish_fisher_var(r.iloc[:29])) and not math.isnan(V.cornish_fisher_var(r.iloc[:30]))


def test_normal_cvar_is_minus_the_mean_plus_sigma_times_the_density_at_the_quantile_over_the_tail_probability():
    r = pd.Series(np.tile([-0.01, 0.03], 50))
    z = stats.norm.ppf(0.05)
    expected = -0.01 + r.std(ddof=1) * stats.norm.pdf(z) / 0.05
    assert C.parametric_cvar(r, 0.95) == pytest.approx(expected) and C.parametric_cvar(r, 0.95, 4) == pytest.approx(2 * expected)
    assert math.isnan(C.parametric_cvar(r.iloc[:19])) and not math.isnan(C.parametric_cvar(r.iloc[:20]))
    assert C.parametric_cvar(r, 0.95) > V.parametric_var(r, 0.95)


def test_student_t_cvar_equals_the_tail_mean_found_by_numerical_integration_of_the_fitted_density():
    r = pd.Series(stats.t.rvs(5, loc=0.0003, scale=0.012, size=5000, random_state=2))
    df, loc, scale = stats.t.fit(r.to_numpy())
    cutoff = loc + stats.t.ppf(0.05, df) * scale
    tail_mean, _ = integrate.quad(lambda x: x * stats.t.pdf((x - loc) / scale, df) / scale, -np.inf, cutoff)
    assert C.parametric_cvar(r, 0.95, distribution="t") == pytest.approx(-tail_mean / 0.05, rel=1e-6)
    assert C.parametric_cvar(r, 0.95, 9, "student_t") == pytest.approx(3 * C.parametric_cvar(r, 0.95, 1, "t"))


# ------------------------------------------------------------------------------------------------------------------------ simulation
def test_the_simulation_reproduces_the_sample_mean_and_deviation_whatever_the_distribution_and_keeps_its_seed():
    r = noise(2000, seed=4, scale=0.01, loc=0.0004)
    mu, sigma = r.mean(), r.std(ddof=1)
    for distribution in ("normal", "student_t", "t"):
        var, draws = V.monte_carlo_var(r, None, 0.95, 1, 200_000, distribution, 5.0, seed=9)
        assert len(draws) == 200_000 and draws.mean() == pytest.approx(mu, abs=5 * sigma / math.sqrt(200_000)) and draws.std() == pytest.approx(sigma, rel=0.03)   # a t scaled to the sample's deviation
        assert var == pytest.approx(-np.quantile(draws, 0.05))
    assert V.monte_carlo_var(r, None, 0.95, 1, 5000, "normal", seed=1)[0] == V.monte_carlo_var(r, None, 0.95, 1, 5000, "normal", seed=1)[0]
    assert V.monte_carlo_var(r, None, 0.95, 1, 5000, "normal", seed=1)[0] != V.monte_carlo_var(r, None, 0.95, 1, 5000, "normal", seed=2)[0]
    normal_var = V.monte_carlo_var(r, None, 0.99, 1, 400_000, "normal", seed=3)[0]
    assert normal_var == pytest.approx(V.parametric_var(r, 0.99), rel=0.03)


def test_a_student_t_with_two_or_fewer_degrees_of_freedom_is_not_rescaled_and_a_bootstrap_draws_only_from_the_history():
    r = noise(500, seed=6)
    _, heavy = V.monte_carlo_var(r, None, 0.95, 1, 20_000, "student_t", 2.0, seed=1)
    assert np.isfinite(heavy).all() and heavy.std() > r.std()                              # the infinite variance of a t(2) is not cut down to the sample's
    _, boot = V.monte_carlo_var(r, None, 0.95, 1, 5000, "bootstrap", seed=1)
    assert set(np.round(boot, 12)) <= set(np.round(r.to_numpy(), 12)) and len(set(np.round(boot, 12))) > 400
    with pytest.raises(ValueError, match="unknown distribution"):
        V.monte_carlo_var(r, None, 0.95, 1, 100, "cauchy")


def test_the_horizon_scales_the_simulated_losses_by_the_root_of_the_number_of_days_and_a_table_of_returns_needs_weights():
    r = noise(800, seed=7)
    one, d1 = V.monte_carlo_var(r, None, 0.95, 1, 4000, "normal", seed=5)
    nine, d9 = V.monte_carlo_var(r, None, 0.95, 9, 4000, "normal", seed=5)
    assert np.allclose(d9, 3 * d1) and nine == pytest.approx(3 * one)
    with pytest.raises(ValueError, match="weights"):
        V.monte_carlo_var(pd.DataFrame({"A": r, "B": r}), None)


def test_a_simulated_book_keeps_the_correlation_between_its_assets_and_ignores_assets_with_no_weight():
    rng = np.random.default_rng(8)
    common = rng.normal(0.0, 0.01, 3000)
    frame = pd.DataFrame({"A": common, "B": common, "C": rng.normal(0.0, 0.01, 3000), "D": rng.normal(0.0, 0.01, 3000)})
    single = V.monte_carlo_var(frame["A"], None, 0.95, 1, 100_000, "normal", seed=3)[0]
    same = V.monte_carlo_var(frame, pd.Series({"A": 0.5, "B": 0.5}), 0.95, 1, 100_000, "normal", seed=3)[0]
    apart = V.monte_carlo_var(frame, pd.Series({"C": 0.5, "D": 0.5}), 0.95, 1, 100_000, "normal", seed=3)[0]
    assert same == pytest.approx(single, rel=0.04)                                         # two copies of the same asset do not diversify
    assert apart == pytest.approx(single / math.sqrt(2.0), rel=0.05)                       # two independent ones do, by the root of two
    only_a = V.monte_carlo_var(frame, pd.Series({"A": 1.0}), 0.95, 1, 100_000, "normal", seed=3)[0]
    assert only_a == pytest.approx(single, rel=0.04)                                       # a missing weight is no position
    boot = V.monte_carlo_var(frame, pd.Series({"A": 0.5, "B": 0.5}), 0.95, 1, 20_000, "bootstrap", seed=3)[1]
    assert set(np.round(boot, 12)) <= set(np.round(frame["A"].to_numpy(), 12))             # whole rows of history, so the two copies stay equal
    heavy = V.monte_carlo_var(frame, pd.Series({"A": 0.5, "B": 0.5}), 0.99, 1, 100_000, "student_t", 5.0, seed=3)[0]
    assert heavy > V.monte_carlo_var(frame, pd.Series({"A": 0.5, "B": 0.5}), 0.99, 1, 100_000, "normal", seed=3)[0]


def test_the_simulated_cvar_is_the_mean_loss_beyond_the_simulated_var_and_the_tables_hold_each_method_side_by_side():
    r = noise(1500, seed=9)
    var, draws = V.monte_carlo_var(r, None, 0.95, 1, 20_001, "student_t", 5.0, seed=2)             # 20,001 draws: the 5% point is the 1,001st smallest exactly, and it belongs to the tail
    expected = -draws[draws <= np.quantile(draws, 0.05)].mean()
    assert C.monte_carlo_cvar(r, None, 0.95, 1, 20_001, "student_t", 5.0, 2) == pytest.approx(expected) and expected > var
    assert (draws <= np.quantile(draws, 0.05)).sum() == 1001
    table = V.var_comparison(r, alphas=(0.95, 0.99), n_scenarios=20_000)
    assert list(table.index) == [0.95, 0.99] and list(table.columns) == ["historical", "parametric_normal", "parametric_t", "cornish_fisher", "monte_carlo_normal", "monte_carlo_t"]
    assert table.loc[0.99, "historical"] == pytest.approx(V.historical_var(r, 0.99)) and table.loc[0.95, "parametric_t"] == pytest.approx(V.parametric_var(r, 0.95, 1, "t"))
    assert table.loc[0.95, "monte_carlo_normal"] == pytest.approx(V.monte_carlo_var(r, None, 0.95, 1, 20_000, "normal", 5.0, 20240101)[0])
    assert table.loc[0.95, "monte_carlo_t"] == pytest.approx(V.monte_carlo_var(r, None, 0.95, 1, 20_000, "student_t", 5.0, 20240101)[0])
    wide = C.cvar_comparison(r, alphas=(0.95,), n_scenarios=20_000)
    assert wide.loc[0.95, "cvar_var_ratio"] == pytest.approx(C.historical_cvar(r, 0.95) / V.historical_var(r, 0.95)) and wide.loc[0.95, "cvar_normal"] == pytest.approx(C.parametric_cvar(r, 0.95))
    assert wide.loc[0.95, "cvar_t"] == pytest.approx(C.parametric_cvar(r, 0.95, 1, "t")) and wide.loc[0.95, "var_normal"] == pytest.approx(V.parametric_var(r, 0.95))


# ------------------------------------------------------------------------------------------------------------------------ rolling forecasts
def window_of(r: pd.Series, t: int, lookback: int) -> pd.Series:
    return r.iloc[max(0, t - lookback):t]                                                   # the days before day t: the forecast for t may not see t


@pytest.mark.parametrize("method", ["historical", "parametric_normal", "parametric_t", "ewma_normal", "cornish_fisher"])
def test_a_rolling_forecast_for_a_day_uses_only_the_days_before_it_and_waits_for_its_minimum_history(method):
    r = noise(700, seed=12)
    f = V.rolling_var(r, 0.95, 200, method, 120, 5.0)
    assert f.name == f"var_{method}_0.95" and f.iloc[:120].isna().all() and not np.isnan(f.iloc[120])         # 120 days known, so the first forecast is for day 120
    shocked = r.copy()
    shocked.iloc[500:] *= 30.0
    again = V.rolling_var(shocked, 0.95, 200, method, 120, 5.0)
    pd.testing.assert_series_equal(f.iloc[:501], again.iloc[:501])                          # nothing from day 500 on reaches the forecasts up to day 500


def test_each_rolling_method_is_its_closed_form_on_the_window_before_the_day():
    r = noise(700, seed=13, loc=0.0003)
    lookback, minimum, t = 200, 120, 450
    w = window_of(r, t, lookback)
    assert V.rolling_var(r, 0.95, lookback, "historical", minimum).iloc[t] == pytest.approx(-np.quantile(w, 0.05))
    assert V.rolling_var(r, 0.99, lookback, "parametric_normal", minimum).iloc[t] == pytest.approx(-(w.mean() + stats.norm.ppf(0.01) * w.std(ddof=1)))
    assert V.rolling_var(r, 0.99, lookback, "parametric_t", minimum, 6.0).iloc[t] == pytest.approx(-(w.mean() + stats.t.ppf(0.01, 6.0) / math.sqrt(6.0 / 4.0) * w.std(ddof=1)))
    assert V.rolling_var(r, 0.99, lookback, "cornish_fisher", minimum).iloc[t] == pytest.approx(V.cornish_fisher_var(w, 0.99))
    # RiskMetrics: a weighted mean of squared returns with weight (1 - a)^age, half-life a eighth of the look-back
    a = 1.0 - math.exp(-math.log(2.0) / (lookback / 8.0))
    ages = np.arange(t)[::-1]
    weights = (1.0 - a) ** ages
    variance = float((weights * r.iloc[:t].to_numpy() ** 2).sum() / weights.sum())
    assert V.rolling_var(r, 0.95, lookback, "ewma_normal", minimum).iloc[t] == pytest.approx(-stats.norm.ppf(0.05) * math.sqrt(variance))
    with pytest.raises(ValueError, match="unknown rolling VaR method"):
        V.rolling_var(r, 0.95, lookback, "garch", minimum)


def test_the_rolling_cvar_is_the_tail_mean_of_the_window_before_the_day_and_the_backtest_compares_it_with_what_happened():
    r = noise(700, seed=14)
    lookback, minimum, t = 200, 120, 450
    w = window_of(r, t, lookback).to_numpy()
    expected = -w[w <= np.quantile(w, 0.05)].mean()
    f = C.rolling_cvar(r, 0.95, lookback, minimum)
    assert f.name == "cvar_0.95" and f.iloc[t] == pytest.approx(expected) and f.iloc[:minimum].isna().all() and not np.isnan(f.iloc[minimum])
    calm = pd.Series(np.tile([0.01, -0.01], 150), index=DAYS[:300])
    crash = pd.concat([calm, pd.Series([-0.05, -0.04, -0.06], index=DAYS[300:303])])
    out = C.cvar_backtest(crash, 0.95, 100, 50)
    assert out["n_breaches"] == 3 and out["n_obs"] == 253 and out["realised_tail_loss"] == pytest.approx(0.05) and out["worst_realised_loss"] == pytest.approx(0.06)
    # the forecasts on the three crash days: the calm window's 1%, then windows holding one crash day (51 returns at or below -1%, -0.55 in all) and two (-0.58 in all)
    tail = (0.01 + 0.55 / 51 + 0.58 / 51) / 3
    assert out["predicted_cvar"] == pytest.approx(tail) and out["ratio_realised_to_predicted"] == pytest.approx(0.05 / tail)
    assert out["mean_cvar_forecast"] == pytest.approx((250 * 0.01 + 3 * tail) / 253)
    assert C.cvar_backtest(calm, 0.95, 100, 50) == {"n_breaches": 0} and C.cvar_backtest(calm.iloc[:30], 0.95, 100, 50) == {}


# ------------------------------------------------------------------------------------------------------------------------ the breach tests
def test_kupiec_is_the_likelihood_ratio_of_the_breach_count_against_the_stated_rate():
    stat, p = V.kupiec_test(50, 1000, 0.95)
    assert stat == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)                 # exactly the expected count: no evidence at all
    stat, p = V.kupiec_test(80, 1000, 0.95)
    rate, q = 0.08, 0.05
    ratio = -2 * (920 * math.log(1 - q) + 80 * math.log(q) - 920 * math.log(1 - rate) - 80 * math.log(rate))
    assert stat == pytest.approx(ratio) and p == pytest.approx(1 - stats.chi2.cdf(ratio, 1)) and p < 0.01
    fewer, p_fewer = V.kupiec_test(20, 1000, 0.95)                                           # too few is as suspect as too many
    assert fewer > 0 and p_fewer < 0.01
    stat, p = V.kupiec_test(0, 100, 0.95)                                                    # no breach at all: -2 n ln(1 - p)
    assert stat == pytest.approx(-200 * math.log(0.95)) and p == pytest.approx(1 - stats.chi2.cdf(stat, 1))
    assert all(math.isnan(x) for x in V.kupiec_test(0, 0, 0.95)) and all(math.isnan(x) for x in V.kupiec_test(100, 100, 0.95))
    assert V.kupiec_test(1, 100, 0.99)[0] == pytest.approx(0.0, abs=1e-9)


def test_christoffersen_is_the_likelihood_ratio_of_a_first_order_chain_against_independence():
    indicator = np.array([0] * 40 + [1, 1, 1] + [0] * 30 + [1] + [0] * 20 + [1, 1] + [0] * 3)
    previous, current = indicator[:-1], indicator[1:]
    n00, n01 = int(((previous == 0) & (current == 0)).sum()), int(((previous == 0) & (current == 1)).sum())
    n10, n11 = int(((previous == 1) & (current == 0)).sum()), int(((previous == 1) & (current == 1)).sum())
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / (n00 + n01 + n10 + n11)
    log_null = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
    log_alt = n00 * math.log(1 - pi01) + n01 * math.log(pi01) + n10 * math.log(1 - pi11) + n11 * math.log(pi11)
    stat, p = V.christoffersen_test(pd.Series(indicator))
    assert stat == pytest.approx(-2 * (log_null - log_alt)) and p == pytest.approx(1 - stats.chi2.cdf(stat, 1)) and stat > 0
    assert stat == V.christoffersen_test(pd.Series(indicator.astype(float)))[0]
    assert all(math.isnan(x) for x in V.christoffersen_test(pd.Series([0, 1, 0, 0, 1, 0, 1, 0, 0])))                # nine observations are too few
    assert not math.isnan(V.christoffersen_test(pd.Series([0, 1, 0, 0, 1, 0, 1, 0, 0, 1]))[0])                    # ten are enough
    for degenerate in ([0] * 50, [1] * 50, [0] * 49 + [1], [1] + [0] * 49 if False else [0] * 49 + [1]):
        assert all(math.isnan(x) for x in V.christoffersen_test(pd.Series(degenerate)))                           # nothing to test, or nothing after the only breach
    perfectly_clustered = V.christoffersen_test(pd.Series([0] * 90 + [1] * 10))
    assert np.isfinite(perfectly_clustered[0]) and perfectly_clustered[1] < 0.001                                  # the most extreme dependence still fires
    assert V.christoffersen_test(pd.Series([0, 1, np.nan] * 10 + [0, 0]))[0] == V.christoffersen_test(pd.Series([0, 1] * 10 + [0, 0]))[0]


def breach_report(**changes) -> V.VaRBacktest:
    base = dict(alpha=0.95, method="historical", n_obs=1000, n_breaches=50, expected_breaches=50.0, breach_rate=0.05, kupiec_stat=0.0, kupiec_pvalue=0.9, christoffersen_stat=0.1,
                christoffersen_pvalue=0.8, joint_stat=0.1, joint_pvalue=0.9, mean_breach_size=0.01, max_breach_size=0.02, breaches=pd.Series([0, 1]))
    return V.VaRBacktest(**{**base, **changes})


def test_the_verdict_names_what_is_wrong_with_a_var_model_and_waits_for_evidence():
    assert breach_report().verdict() == "pass"
    assert breach_report(christoffersen_pvalue=float("nan")).verdict() == "pass"                                    # independence untestable: the count alone decides
    assert breach_report(kupiec_pvalue=float("nan")).verdict() == "inconclusive"
    assert breach_report(kupiec_pvalue=0.01, breach_rate=0.08).verdict() == "fail: understates risk"
    assert breach_report(kupiec_pvalue=0.01, breach_rate=0.02).verdict() == "fail: overstates risk"
    assert breach_report(kupiec_pvalue=0.5, christoffersen_pvalue=0.01).verdict() == "fail: breaches are clustered"
    assert breach_report(kupiec_pvalue=0.04).verdict(significance=0.01) == "pass" and breach_report(kupiec_pvalue=0.04).verdict() != "pass"
    row = breach_report().to_row()
    assert row["expected_rate"] == pytest.approx(0.05) and row["verdict"] == "pass" and row["method"] == "historical" and row["n_breaches"] == 50
    assert set(row) >= {"alpha", "n_obs", "breach_rate", "kupiec_pvalue", "christoffersen_pvalue", "joint_pvalue", "mean_breach_size", "max_breach_size"}


def test_a_backtest_counts_the_days_the_loss_was_beyond_the_forecast_and_measures_by_how_much():
    calm = pd.Series(np.tile([0.01, -0.01], 150), index=DAYS[:300])                                                  # the 5% point of a 50-50 split is -1%
    crash = pd.concat([calm, pd.Series([-0.05, 0.0, -0.03], index=DAYS[300:303])])
    result = V.rolling_var_backtest(crash, 0.95, 100, "historical", 50)
    assert result.n_obs == 253 and result.n_breaches == 2 and result.breach_rate == pytest.approx(2 / 253) and result.expected_breaches == pytest.approx(253 * 0.05)
    assert result.mean_breach_size == pytest.approx(0.03) and result.max_breach_size == pytest.approx(0.04)           # losses of 5% and 3% against a 1% forecast
    assert list(result.breaches[result.breaches == 1].index) == [DAYS[300], DAYS[302]]
    assert result.kupiec_pvalue == pytest.approx(V.kupiec_test(2, 253, 0.95)[1]) and result.christoffersen_pvalue == pytest.approx(V.christoffersen_test(result.breaches)[1])
    assert result.joint_stat == pytest.approx(result.kupiec_stat + result.christoffersen_stat) and result.joint_pvalue == pytest.approx(1 - stats.chi2.cdf(result.joint_stat, 2))
    quiet = V.rolling_var_backtest(calm, 0.95, 100, "historical", 50)                                                # a loss equal to the forecast is not a breach
    assert quiet.n_breaches == 0 and quiet.mean_breach_size == 0.0 and quiet.max_breach_size == 0.0
    with pytest.raises(ValueError, match="no overlapping"):
        V.rolling_var_backtest(calm.iloc[:30], 0.95, 100, "historical", 50)


def test_the_comparison_runs_every_method_at_every_level_and_skips_a_method_with_nothing_to_say():
    r = noise(900, seed=15)
    table = V.compare_var_methods(r, alphas=(0.95, 0.99), methods=("historical", "parametric_normal"), lookback=300, min_lookback=150)
    assert len(table) == 4 and set(table["method"]) == {"historical", "parametric_normal"} and set(table["alpha"]) == {0.95, 0.99}
    assert (table["n_obs"] == 750).all()
    assert V.compare_var_methods(r.iloc[:100], lookback=300, min_lookback=150).empty


def test_the_draws_are_the_generators_own_for_the_seed_scaled_to_the_samples_mean_and_deviation():
    r = noise(900, seed=16, loc=0.0003)
    mu, sigma = r.mean(), r.std(ddof=1)
    n, seed = 3000, 11
    _, normal = V.monte_carlo_var(r, None, 0.95, 1, n, "normal", seed=seed)
    assert np.array_equal(normal, np.random.default_rng(seed).normal(mu, sigma, n))
    _, heavy = V.monte_carlo_var(r, None, 0.95, 1, n, "student_t", 5.0, seed=seed)
    assert np.allclose(heavy, mu + sigma / math.sqrt(5.0 / 3.0) * np.random.default_rng(seed).standard_t(5.0, n))     # a t(5) has variance 5/3: divided by its root it has the sample's deviation
    _, boot = V.monte_carlo_var(r, None, 0.95, 1, n, "bootstrap", seed=seed)
    assert np.array_equal(boot, np.random.default_rng(seed).choice(r.to_numpy(), size=n, replace=True))
    _, wide = V.monte_carlo_var(r, None, 0.95, 1, n, "student_t", 2.0, seed=seed)
    assert np.allclose(wide, mu + sigma * np.random.default_rng(seed).standard_t(2.0, n))                                # with two degrees of freedom the variance is infinite: no rescaling


def test_a_simulated_book_under_the_t_has_the_variance_of_its_weights_and_the_covariance_even_with_two_degrees_of_freedom_that_have_none():
    rng = np.random.default_rng(17)
    common = rng.normal(0.0, 0.01, 4000)
    frame = pd.DataFrame({"A": common + rng.normal(0.0, 0.005, 4000), "B": 0.5 * common + rng.normal(0.0, 0.008, 4000)})
    w = pd.Series({"A": 0.6, "B": 0.4})
    book_sd = math.sqrt(w.to_numpy() @ frame.cov().to_numpy() @ w.to_numpy())
    for distribution in ("normal", "student_t"):
        _, draws = V.monte_carlo_var(frame, w, 0.95, 1, 300_000, distribution, 8.0, seed=4)
        assert draws.std() == pytest.approx(book_sd, rel=0.03) and draws.mean() == pytest.approx(float(frame.mean() @ w), abs=5 * book_sd / math.sqrt(300_000))
    _, two = V.monte_carlo_var(frame, w, 0.95, 1, 5000, "student_t", 2.0, seed=4)
    assert two.std() > book_sd                                                                  # not rescaled to nothing: a t(2) is wilder than the normal it replaces


def test_a_chain_that_breaches_as_often_after_a_breach_as_after_a_calm_day_has_a_ratio_of_zero_and_not_an_untestable_one():
    independent = pd.Series([1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0])                           # two thirds both ways; floating point gives -2e-15 for it
    stat, p = V.christoffersen_test(independent)
    assert stat == 0.0 and p == 1.0
    assert V.christoffersen_test(pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 1, 0]))[0] == pytest.approx(0.0, abs=1e-12)


def test_the_chain_test_runs_when_there_are_as_many_calm_days_after_calm_days_as_breaches_after_them():
    indicator = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])                                         # after a calm day: three calm, three breaches; after a breach: two calm, one breach
    previous, current = indicator[:-1], indicator[1:]
    n = {(a, b): int(((previous == a) & (current == b)).sum()) for a in (0, 1) for b in (0, 1)}
    assert n[(0, 0)] == n[(0, 1)] == 3 and n[(1, 0)] == 2 and n[(1, 1)] == 1
    pi01, pi11, pi = 3 / 6, 1 / 3, 4 / 9
    log_null = 5 * math.log(1 - pi) + 4 * math.log(pi)
    log_alt = 3 * math.log(1 - pi01) + 3 * math.log(pi01) + 2 * math.log(1 - pi11) + 1 * math.log(pi11)
    stat, p = V.christoffersen_test(pd.Series(indicator))
    assert stat == pytest.approx(-2 * (log_null - log_alt)) and p == pytest.approx(1 - stats.chi2.cdf(stat, 1))
    equal_after_breach = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1])                           # one calm day and one breach follow a breach
    previous, current = equal_after_breach[:-1], equal_after_breach[1:]
    assert int(((previous == 1) & (current == 0)).sum()) == int(((previous == 1) & (current == 1)).sum()) == 1
    pi01, pi11, pi = 2 / 9, 1 / 2, 3 / 11
    log_null = 8 * math.log(1 - pi) + 3 * math.log(pi)
    log_alt = 7 * math.log(1 - pi01) + 2 * math.log(pi01) + 1 * math.log(1 - pi11) + 1 * math.log(pi11)
    assert V.christoffersen_test(pd.Series(equal_after_breach))[0] == pytest.approx(-2 * (log_null - log_alt))


def test_the_coverage_statistic_is_never_below_zero_even_where_rounding_would_make_it_so():
    for breaches, observations in ((50, 1000), (5, 100), (25, 500), (10, 200), (1, 20)):
        assert V.kupiec_test(breaches, observations, 0.95)[0] >= 0.0
    assert V.kupiec_test(50, 1000, 0.95) == (0.0, 1.0) or V.kupiec_test(50, 1000, 0.95)[0] < 1e-9


def test_the_documented_defaults_are_the_ones_used_95_percent_one_day_a_500_day_window_with_250_days_needed_20000_scenarios_5_degrees_of_freedom():
    r = noise(900, seed=21)
    assert V.historical_var(r) == V.historical_var(r, 0.95, 1) and V.parametric_var(r) == V.parametric_var(r, 0.95, 1, "normal") and V.cornish_fisher_var(r) == V.cornish_fisher_var(r, 0.95, 1)
    assert V.monte_carlo_var(r)[0] == V.monte_carlo_var(r, None, 0.95, 1, 20000, "student_t", 5.0, 20240101)[0]
    pd.testing.assert_frame_equal(V.var_comparison(r), V.var_comparison(r, (0.95, 0.99), 1, 20000, 5.0, 20240101))
    for method in ("historical", "parametric_t"):
        pd.testing.assert_series_equal(V.rolling_var(r, method=method), V.rolling_var(r, 0.95, 500, method, 250, 5.0))
    a, b = V.rolling_var_backtest(r, method="parametric_t"), V.rolling_var_backtest(r, 0.95, 500, "parametric_t", 250, 5.0)
    assert (a.n_obs, a.n_breaches, a.kupiec_stat) == (b.n_obs, b.n_breaches, b.kupiec_stat) and a.n_obs == 650
    pd.testing.assert_frame_equal(V.compare_var_methods(r), V.compare_var_methods(r, (0.95, 0.99), ("historical", "parametric_normal", "parametric_t", "ewma_normal"), 500, 250, 5.0))
    assert C.historical_cvar(r) == C.historical_cvar(r, 0.95, 1) and C.parametric_cvar(r) == C.parametric_cvar(r, 0.95, 1, "normal")
    assert C.monte_carlo_cvar(r) == C.monte_carlo_cvar(r, None, 0.95, 1, 20000, "student_t", 5.0, 20240101)
    pd.testing.assert_frame_equal(C.cvar_comparison(r), C.cvar_comparison(r, (0.95, 0.99), 1, 20000, 5.0, 20240101))
    pd.testing.assert_series_equal(C.rolling_cvar(r), C.rolling_cvar(r, 0.95, 500, 250))
    assert C.cvar_backtest(r) == C.cvar_backtest(r, 0.95, 500, 250)
