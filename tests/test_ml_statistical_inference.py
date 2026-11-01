"""Release 5, P1-16: dependence-aware inference and the statistical output contract.

References are independent of the code under test: statsmodels' HAC covariance, the closed-form AR(1) variance inflation, the textbook Mertens (iid-with-moments) standard error, and simulation of interval coverage
whose own uncertainty (a Wilson interval over the replications) is part of what is asserted."""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.ml import calibration as cal, inference as inf
from src.stats.inference import sharpe_standard_error


def ar1(n, phi, mu=0.0, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return pd.Series(x + mu, index=pd.bdate_range("2018-01-01", periods=n))


def test_the_hac_standard_error_of_the_mean_is_statsmodels_bartlett_hac():
    x = ar1(1200, 0.3, mu=0.0004, seed=1)
    for lags in (0, 3, 6, 11):
        mine = inf.hac_mean(x, lags=lags)
        ref = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False, "kernel": "bartlett"})
        assert mine.extras["se"] == pytest.approx(float(ref.bse[0]), rel=1e-12, abs=1e-15), lags
        assert mine.value == pytest.approx(float(ref.params[0]), abs=1e-15) and mine.settings["lags"] == lags


def test_overlapping_horizons_force_at_least_h_minus_one_lags_and_widen_the_interval():
    rng = np.random.default_rng(3)
    daily = pd.Series(rng.normal(0.0005, 0.01, 3000), index=pd.bdate_range("2010-01-01", periods=3000))
    h = 5
    overlapped = daily.rolling(h).sum().dropna()                                               # five-day returns observed every day
    assert inf.overlap_lags(len(overlapped), horizon=h, lags=1) == h - 1, "a caller's smaller lag cannot undercut the overlap"
    naive = inf.hac_mean(overlapped, lags=0)
    aware = inf.hac_mean(overlapped, horizon=h)
    assert aware.settings["lags"] >= h - 1 and "overlapping horizon 5" in aware.dependence
    ratio = aware.extras["se"] / naive.extras["se"]
    assert ratio == pytest.approx(np.sqrt(h), rel=0.2), "for iid daily returns the standard error of the mean of overlapping h-day sums is sqrt(h) times the naive one"


def test_the_hac_delta_method_for_the_sharpe_ratio_reduces_to_the_mertens_formula_without_lags():
    rng = np.random.default_rng(5)
    x = rng.standard_t(6, 20000) * 0.01 + 0.0006                                              # fat tails: the moments matter
    sr, se = inf.sharpe_delta_hac(x, 0)
    ref = sharpe_standard_error(x, "mertens")
    assert se == pytest.approx(ref, rel=2e-3) and sr == pytest.approx(x.mean() / x.std(ddof=0), rel=1e-12)


def test_serial_dependence_widens_the_sharpe_interval_by_about_the_ar1_factor_and_the_naive_one_says_what_it_ignores():
    phi = 0.35
    x = ar1(6000, phi, mu=0.0006, seed=7)
    out = inf.sharpe_inference(x, lags=12, n_boot=200, seed=4)
    ratio = out["hac"].extras["se_ratio_to_naive"]
    assert ratio == pytest.approx(np.sqrt((1 + phi) / (1 - phi)), rel=0.15), "the dependence-aware standard error is the naive one times the AR(1) inflation factor, to estimation accuracy"
    assert "NONE assumed" in out["naive"].dependence and any("too narrow" in x for x in out["naive"].limitations)
    assert out["naive"].value == pytest.approx(out["hac"].value) == pytest.approx(out["bootstrap"].value), "the point estimate is one number; the three differ in uncertainty only"
    iid = inf.sharpe_inference(pd.Series(np.random.default_rng(8).normal(0.0006, 0.01, 6000)), lags=12, n_boot=100)
    assert iid["hac"].extras["se_ratio_to_naive"] == pytest.approx(1.0, abs=0.12)
    assert out["bootstrap"].replications > 100 and out["bootstrap"].seed == 4 and out["bootstrap"].settings["mean_block"] >= 1.0


def test_interval_coverage_by_simulation_is_better_for_the_dependence_aware_interval_and_its_own_uncertainty_is_stated():
    phi, n, reps = 0.3, 500, 400
    sigma_marg = 0.01 / np.sqrt(1 - phi ** 2)
    true_sr = 0.0006 / sigma_marg
    hits = {"naive": 0, "hac": 0}
    for r in range(reps):
        x = ar1(n, phi, mu=0.0006, seed=1000 + r)
        out = inf.sharpe_inference(x, lags=None, n_boot=0 + 50, seed=r)
        for k in hits:
            lo, hi = out[k].interval
            hits[k] += int(lo / np.sqrt(252) <= true_sr <= hi / np.sqrt(252))
    cov = {k: v / reps for k, v in hits.items()}
    wl = {k: cal.wilson(v, reps) for k, v in hits.items()}
    assert wl["naive"][1] < 0.95, f"the naive interval under-covers: {cov['naive']:.3f}, Wilson {wl['naive']}"
    assert cov["hac"] > cov["naive"] + 0.02 and cov["hac"] > 0.90, f"HAC {cov['hac']:.3f} {wl['hac']} against naive {cov['naive']:.3f} {wl['naive']}"


def test_a_paired_difference_uses_the_same_resampled_blocks_and_records_what_it_aligned():
    rng = np.random.default_rng(11)
    base = pd.Series(rng.normal(0.0004, 0.01, 1500), index=pd.bdate_range("2015-01-01", periods=1500))
    other = base + rng.normal(0.0003, 0.002, 1500)                                             # nearly the same stream, a little better
    d = inf.paired_difference(other, base, statistic="mean", n_boot=800, seed=3)
    assert d.eligibility == "applied" and d.value > 0 and d.interval[0] > 0 and d.p_value < 0.05
    independent = np.sqrt(inf.hac_mean(other).extras["se"] ** 2 + inf.hac_mean(base).extras["se"] ** 2)
    assert d.extras["bootstrap_se"] < 0.3 * independent, "joint resampling keeps the dependence between the two streams: the difference is far more precise than two independent estimates"
    assert inf.economic_effect(d)["excludes_zero"] is True
    same = inf.paired_difference(base, base, statistic="mean", n_boot=200, seed=1)
    assert same.value == 0.0 and same.p_value == 1.0
    short = base.iloc[:700].set_axis(base.index[:700]); long_ = base.iloc[300:].set_axis(base.index[300:])
    al = inf.paired_difference(short, long_, statistic="sharpe", n_boot=200, seed=1)
    assert al.sample["aligned"] == 400 and al.sample["dropped"] == {"only_in_a": 300, "only_in_b": 800}


def test_a_result_that_was_not_applied_carries_no_number_and_every_result_carries_the_whole_contract():
    short = inf.hac_mean(pd.Series(np.arange(5.0)))
    assert short.eligibility == "insufficient" and short.value is None and short.interval is None and short.p_value is None and short.reason
    const = inf.sharpe_inference(pd.Series(np.full(300, 0.001)))
    assert {r.eligibility for r in const.values()} == {"insufficient"} and all(r.value is None for r in const.values())
    bad = inf.paired_difference(pd.Series(np.arange(50.0)), pd.Series(np.arange(50.0)), statistic="median")
    assert bad.eligibility == "unsupported" and bad.value is None
    with pytest.raises(ValueError):
        inf.StatResult("x", "v", "h", {}, "daily", "u", "r", "e", "d", {}, 0.95, None, None, "s", eligibility="maybe")
    with pytest.raises(ValueError, match="annualisation factor is not guessed"):
        inf.periods_per_year("fortnightly")
    full = inf.sharpe_inference(ar1(800, 0.2, mu=0.0005, seed=2), n_boot=100)["hac"].as_dict()
    for field in ("name", "version", "hypothesis", "sample", "frequency", "units", "risk_free", "estimator", "dependence", "settings", "level", "seed", "replications", "scope", "value", "interval",
                  "p_value", "eligibility", "reason", "limitations"):
        assert field in full, field
    assert full["sample"]["n"] == 800 and full["sample"]["values_hash"] and full["level"] == 0.95 and full["scope"]
    json.dumps(full, allow_nan=False)


def test_a_bootstrap_with_too_few_replications_is_insufficient_and_carries_no_interval():
    x = np.random.default_rng(5).normal(0.0005, 0.01, 300)
    few = inf.sharpe_inference(x, n_boot=100, label="few")
    assert few["bootstrap"].eligibility == "insufficient" and few["bootstrap"].interval is None and "fewer than 200" in few["bootstrap"].reason
    assert few["hac"].eligibility == "applied" and few["naive"].eligibility == "applied", "the other two estimates do not depend on the replications"
    enough = inf.sharpe_inference(x, n_boot=300, label="enough")["bootstrap"]
    assert enough.eligibility == "applied" and np.isfinite(enough.interval).all() and enough.replications == 300
    paired = inf.paired_difference(x, x[::-1], n_boot=50)
    assert paired.eligibility == "insufficient" and paired.value is None and paired.sample["aligned"] == 300
