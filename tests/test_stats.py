"""The statistics package against statsmodels and arch (reference implementations), plus properties that must hold by construction."""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from arch.unitroot import PhillipsPerron, VarianceRatio
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests, kpss

from src.probability.resampling import (bootstrap_indices, circular_block_indices, moving_block_indices, optimal_block_length, stationary_indices,
                                        subsample_ci, subsample_statistics, wild_bootstrap_multipliers)
from src.stats import (acf, adf_test, arch_lm, bootstrap_ci, deflated_sharpe_ratio, diagnostics, engle_granger, event_study, expected_max_sharpe,
                       fama_macbeth, fama_macbeth_two_pass, feasible_gls_ar1, gls, granger_causality, grs_test, haircut_sharpe, half_life, hausman_test,
                       hurst_exponent, kpss_test, ljung_box, min_track_record_length, ols, p_adjust, panel_ols, phillips_perron,
                       probabilistic_sharpe_ratio, random_effects, rolling_ols, romano_wolf, sharpe_standard_error, stationarity_summary, storey_qvalues,
                       variance_inflation, variance_ratio, wls)
from src.stats.event_study import buy_and_hold_abnormal_return

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def reg_data():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    u = rng.normal(size=n) * (1 + abs(X.a))
    return pd.Series(1 + 0.5 * X.a - 0.2 * X.b + u), X


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(3)
    n = 800
    rw = np.cumsum(rng.normal(size=n))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    return rw, ar


# --------------------------------------------------------------------------------------------------------------- regression
@pytest.mark.parametrize("cov", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def test_ols_matches_statsmodels(reg_data, cov):
    y, X = reg_data
    mine, ref = ols(y, X, cov), sm.OLS(y, sm.add_constant(X)).fit(**({} if cov == "nonrobust" else {"cov_type": cov}))
    np.testing.assert_allclose(mine.params, ref.params, atol=1e-10)
    np.testing.assert_allclose(mine.bse, ref.bse, atol=1e-10)
    np.testing.assert_allclose(mine.pvalues, ref.pvalues, atol=1e-10)
    assert mine.r2 == pytest.approx(ref.rsquared) and mine.adj_r2 == pytest.approx(ref.rsquared_adj)


def test_hac_and_cluster_match_statsmodels(reg_data):
    y, X = reg_data
    ref = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": True})
    np.testing.assert_allclose(ols(y, X, "HAC", lags=5).bse, ref.bse, atol=1e-10)
    g = np.repeat(np.arange(40), 10)
    ref = sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": g})
    np.testing.assert_allclose(ols(y, X, "cluster", groups=g).bse, ref.bse, atol=1e-10)


def test_two_way_cluster_is_psd_and_wider_than_naive_with_common_shock():
    rng = np.random.default_rng(1)
    n_e, n_t = 25, 40
    e, t = np.repeat(np.arange(n_e), n_t), np.tile(np.arange(n_t), n_e)
    shock = rng.normal(size=n_t)[t]
    x = rng.normal(size=n_e * n_t) + shock
    y = pd.Series(0.2 * x + shock + rng.normal(size=n_e * n_t))
    X = pd.DataFrame({"x": x})
    naive = ols(y, X, "HC1").bse["x"]
    twoway = ols(y, X, "cluster2", groups=np.column_stack([e, t])).bse["x"]
    assert twoway > naive
    assert np.all(np.linalg.eigvalsh(ols(y, X, "cluster2", groups=np.column_stack([e, t])).cov.to_numpy()) >= -1e-12)


def test_missing_rows_are_dropped_and_groups_stay_aligned(reg_data):
    y, X = reg_data
    X2 = X.copy()
    X2.iloc[5, 0] = np.nan
    g = np.repeat(np.arange(40), 10)
    r = ols(y, X2, "cluster", groups=g)
    assert r.nobs == len(y) - 1 and r.detail["n_clusters"] == 40


def test_wls_gls_and_ar1(reg_data):
    y, X = reg_data
    w = 1 / (1 + abs(X.a)) ** 2
    mine, ref = wls(y, X, w), sm.WLS(y, sm.add_constant(X), weights=w).fit()
    np.testing.assert_allclose(mine.params, ref.params, atol=1e-10)
    np.testing.assert_allclose(mine.bse, ref.bse, atol=1e-10)
    assert mine.r2 == pytest.approx(ref.rsquared)
    # GLS with identity covariance is OLS
    np.testing.assert_allclose(gls(y.iloc[:60], X.iloc[:60], np.eye(60)).params, ols(y.iloc[:60], X.iloc[:60]).params, atol=1e-9)
    rng = np.random.default_rng(2)
    e = np.zeros(400)
    for t in range(1, 400):
        e[t] = 0.6 * e[t - 1] + rng.normal()
    r = feasible_gls_ar1(pd.Series(1 + 0.5 * X.a + e), X[["a"]])
    assert 0.45 < r.detail["rho"] < 0.75


def test_wls_rejects_nonpositive_weights(reg_data):
    y, X = reg_data
    with pytest.raises(ValueError):
        wls(y, X, np.zeros(len(y)))


def test_rolling_ols_equals_window_ols_and_is_causal(reg_data):
    y, X = reg_data
    roll = rolling_ols(y, X, 100)
    last = ols(y.iloc[-100:], X.iloc[-100:])
    np.testing.assert_allclose(roll["params"].iloc[-1], last.params, atol=1e-8)
    np.testing.assert_allclose(roll["bse"].iloc[-1], last.bse, atol=1e-8)
    assert roll["params"].iloc[:99].isna().all().all()
    changed = y.copy()
    changed.iloc[300:] += 100
    assert np.allclose(rolling_ols(changed, X, 100)["params"].iloc[:300].dropna(), roll["params"].iloc[:300].dropna())
    expanding = rolling_ols(y, X, 100, expanding=True, min_obs=30)
    np.testing.assert_allclose(expanding["params"].iloc[-1], ols(y, X).params, atol=1e-8)


def test_diagnostics_detect_heteroskedasticity_and_autocorrelation(reg_data):
    y, X = reg_data
    d = diagnostics(ols(y, X))
    assert d.loc["homoskedastic (Breusch-Pagan, Koenker)", "pvalue"] < 0.01 and d.loc["homoskedastic (White)", "pvalue"] < 0.01
    rng = np.random.default_rng(5)
    clean = pd.Series(rng.normal(size=500))
    xc = pd.DataFrame({"x": rng.normal(size=500)})
    c = diagnostics(ols(clean, xc))
    assert (c["pvalue"].dropna() > 0.001).all()
    e = np.zeros(500)
    for t in range(1, 500):
        e[t] = 0.7 * e[t - 1] + rng.normal()
    bad = diagnostics(ols(pd.Series(e), xc))
    assert bad.loc["no serial correlation (Breusch-Godfrey, 4 lags)", "pvalue"] < 1e-6
    assert bad.iloc[-1, 0] < 1.0


def test_variance_inflation_flags_collinearity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    X = pd.DataFrame({"a": a, "b": a + 0.01 * rng.normal(size=300), "c": rng.normal(size=300)})
    v = variance_inflation(X)
    assert v["a"] > 100 and v["c"] < 1.5


def test_wald_test_and_conf_int(reg_data):
    y, X = reg_data
    r = ols(y, X)
    w = r.wald_test(np.array([[0, 1, 0, 0]]), [0.5])
    assert 0 <= w["chi2_pvalue"] <= 1
    ci = r.conf_int()
    assert (ci["lower"] < r.params).all() and (r.params < ci["upper"]).all()
    assert r.f_test_all()["F_pvalue"] < 1e-6
    with pytest.raises(ValueError):
        ols(y.iloc[:3], X.iloc[:3])


# ------------------------------------------------------------------------------------------------------------------ panel
@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    N, T = 30, 80
    idx = pd.MultiIndex.from_product([range(N), pd.date_range("2020", periods=T)], names=["e", "t"])
    alpha = np.repeat(rng.normal(size=N), T)
    X = pd.DataFrame(rng.normal(size=(N * T, 2)), index=idx, columns=["a", "b"])
    X["a"] += alpha * 0.5
    y = pd.Series(alpha + X.a * 0.7 - X.b * 0.3 + rng.normal(size=N * T), index=idx)
    return y, X, idx, N, T


def test_entity_fixed_effects_equal_lsdv(panel):
    y, X, idx, N, T = panel
    r = panel_ols(y, X, entity_effects=True)
    D = pd.get_dummies(idx.get_level_values(0)).astype(float)
    D.index = idx
    ref = sm.OLS(y, pd.concat([X, D], axis=1)).fit()
    np.testing.assert_allclose(r.params, ref.params[["a", "b"]], atol=1e-9)
    np.testing.assert_allclose(r.bse, ref.bse[["a", "b"]], atol=1e-9)


def test_two_way_fixed_effects_equal_lsdv(panel):
    y, X, idx, N, T = panel
    r = panel_ols(y, X, entity_effects=True, time_effects=True)
    D = pd.get_dummies(idx.get_level_values(0)).astype(float)
    Dt = pd.get_dummies(idx.get_level_values(1), drop_first=True).astype(float)
    D.index, Dt.index = idx, idx
    ref = sm.OLS(y, pd.concat([X, D, Dt], axis=1)).fit()
    np.testing.assert_allclose(r.params, ref.params[["a", "b"]], atol=1e-8)
    np.testing.assert_allclose(r.bse, ref.bse[["a", "b"]], atol=1e-8)


def test_clustered_se_entity_nested_matches_formula(panel):
    y, X, idx, N, T = panel
    r = panel_ols(y, X, entity_effects=True, cov="cluster_entity")
    D = pd.get_dummies(idx.get_level_values(0)).astype(float)
    D.index = idx
    ref = sm.OLS(y, pd.concat([X, D], axis=1)).fit(cov_type="cluster", cov_kwds={"groups": idx.get_level_values(0)})
    n, k = N * T, 2 + N
    np.testing.assert_allclose(r.bse, ref.bse[["a", "b"]] * np.sqrt((n - k) / (n - 1)), atol=1e-9)


def test_panel_covariances_run_and_driscoll_kraay_is_robust_to_time_effects(panel):
    y, X, *_ = panel
    for cov in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_two", "driscoll_kraay"]:
        r = panel_ols(y, X, entity_effects=True, cov=cov)
        assert np.all(np.isfinite(r.bse)) and np.all(r.bse > 0)
    assert panel_ols(y, X).params.index[0] == "const"
    with pytest.raises(ValueError):
        panel_ols(y, X, cov="nope")


def test_random_effects_and_hausman(panel):
    y, X, idx, N, T = panel
    h = hausman_test(y, X)
    assert h["pvalue"] < 0.01                                   # regressor a is built from the effect: RE is inconsistent
    rng = np.random.default_rng(1)
    X2 = pd.DataFrame(rng.normal(size=(N * T, 2)), index=idx, columns=["a", "b"])
    alpha = np.repeat(rng.normal(size=N), T)
    y2 = pd.Series(alpha + 0.7 * X2.a + rng.normal(size=N * T), index=idx)
    assert hausman_test(y2, X2)["chi2"] >= 0
    re = random_effects(y2, X2)
    assert abs(re["params"]["a"] - 0.7) < 0.1 and re["sigma_u2"] > 0.3


def test_fama_macbeth_recovers_premium_and_is_predictive():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010", periods=300)
    A = [f"s{i}" for i in range(60)]
    c = pd.DataFrame(rng.normal(size=(300, 60)), index=dates, columns=A)
    r = 0.01 * c.shift(1) + pd.DataFrame(rng.normal(size=(300, 60)), index=dates, columns=A) * 0.05
    fm = fama_macbeth(r, {"c": c})
    assert fm.mean["c"] == pytest.approx(0.01, abs=0.003) and fm.tvalues["c"] > 5
    # contemporaneous characteristic (lag=0) must not be what recovers the premium
    assert abs(fama_macbeth(r, {"c": c}, lag=0).mean["c"]) < 0.004
    assert fm.n_periods == 299 and fm.avg_assets == 60
    assert list(fm.summary().columns) == ["premium", "se", "t", "p"]


def test_two_pass_shanken_widens_standard_errors_and_grs_accepts_true_model():
    rng = np.random.default_rng(4)
    F = pd.DataFrame(rng.normal(0.005, 0.04, size=(600, 2)), columns=["m", "s"])
    beta = rng.uniform(0.5, 1.5, size=(2, 20))
    R = pd.DataFrame(F.values @ beta + rng.normal(0, 0.03, size=(600, 20)))
    tp = fama_macbeth_two_pass(R, F)
    assert (tp["se_shanken"] >= tp["se_naive"]).all() and tp["shanken_factor"] > 1.0
    np.testing.assert_allclose(tp["lambda"], [0.005, 0.005], atol=0.01)
    assert grs_test(R, F)["pvalue"] > 0.01
    R2 = R.copy()
    R2.iloc[:, :10] += 0.01                                       # a large pricing error in half the assets
    assert grs_test(R2, F)["pvalue"] < 1e-6


# ------------------------------------------------------------------------------------------------------------- time series
@pytest.mark.parametrize("reg", ["n", "c", "ct"])
@pytest.mark.parametrize("which", [0, 1])
def test_adf_matches_statsmodels(series, reg, which):
    x = series[which]
    mine = adf_test(x, reg, maxlag=8, autolag="aic")
    ref = adfuller(x, maxlag=8, regression=reg, autolag="AIC")
    assert mine.statistic == pytest.approx(ref[0], abs=1e-8) and mine.lags == ref[2] and mine.pvalue == pytest.approx(ref[1], abs=1e-8)


@pytest.mark.parametrize("auto", ["bic", "tstat"])
def test_adf_lag_selection_matches_statsmodels(series, auto):
    mine = adf_test(series[1], "c", maxlag=8, autolag=auto)
    ref = adfuller(series[1], maxlag=8, autolag={"bic": "BIC", "tstat": "t-stat"}[auto])
    assert mine.statistic == pytest.approx(ref[0], abs=1e-8) and mine.lags == ref[2]


def test_adf_separates_random_walk_from_ar(series):
    assert not adf_test(series[0]).reject() and adf_test(series[1]).reject()
    with pytest.raises(ValueError):
        adf_test(series[0], autolag="bad")


@pytest.mark.parametrize("reg", ["c", "ct"])
def test_kpss_matches_statsmodels(series, reg):
    for x in series:
        assert kpss_test(x, reg, lags=10).statistic == pytest.approx(kpss(x, regression=reg, nlags=10)[0], abs=1e-9)
    assert kpss_test(series[0]).reject() and not kpss_test(series[1]).reject()


@pytest.mark.parametrize("reg", ["n", "c", "ct"])
def test_phillips_perron_matches_arch(series, reg):
    for x in series:
        mine, ref = phillips_perron(x, reg, lags=8), PhillipsPerron(x, lags=8, trend=reg)
        assert mine.statistic == pytest.approx(ref.stat, abs=1e-8) and mine.pvalue == pytest.approx(ref.pvalue, abs=1e-8)
    assert phillips_perron(series[1], "c", lags=8, test_type="rho").statistic == pytest.approx(PhillipsPerron(series[1], lags=8, test_type="rho").stat, abs=1e-6)


def test_stationarity_summary_verdicts(series):
    assert stationarity_summary(series[0])["verdict"] == "unit root"
    assert stationarity_summary(series[1])["verdict"] == "stationary"


def test_variance_ratio_matches_arch(series):
    for x in series:
        r = variance_ratio(np.exp(x * 0.01), q=5)
        a = VarianceRatio(x * 0.01, lags=5, robust=True, debiased=True, overlap=True)
        assert r["vr"] == pytest.approx(a.vr, abs=1e-9) and r["z_robust"] == pytest.approx(a.stat, abs=1e-8)
        assert variance_ratio(np.exp(x * 0.01), q=5)["z"] == pytest.approx(VarianceRatio(x * 0.01, lags=5, robust=False, debiased=True, overlap=True).stat, abs=1e-8)
    assert variance_ratio(np.exp(series[1] * 0.01), q=5)["vr"] < 1.0


def test_hurst_and_half_life(series):
    rng = np.random.default_rng(0)
    assert hurst_exponent(rng.normal(size=4000)) == pytest.approx(0.5, abs=0.08)
    assert hurst_exponent(np.diff(series[1])) < 0.4
    assert hurst_exponent(rng.normal(size=4000), "dfa") == pytest.approx(0.5, abs=0.08)
    assert half_life(series[1]) == pytest.approx(np.log(2) / -np.log(0.9), rel=0.35)
    assert half_life(series[0]) > 50
    with pytest.raises(ValueError):
        hurst_exponent(rng.normal(size=500), "x")


def test_ljung_box_acf_and_arch_lm():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    assert ljung_box(x, 10)["pvalue"] > 0.01 and acf(x, 3)[0] == 1.0
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    ref = acorr_ljungbox(x, lags=[10], return_df=True)
    assert ljung_box(x, 10)["statistic"] == pytest.approx(float(ref["lb_stat"].iloc[0]), rel=1e-9)
    h = np.zeros(2000)
    e = rng.normal(size=2000)
    for t in range(1, 2000):
        h[t] = np.sqrt(0.2 + 0.7 * h[t - 1] ** 2) * e[t]
    assert arch_lm(h, 5)["pvalue"] < 1e-6
    assert arch_lm(x, 5)["statistic"] == pytest.approx(het_arch(x - x.mean(), nlags=5)[0], rel=1e-6)


def test_engle_granger_matches_statsmodels_and_detects_cointegration(series):
    rng = np.random.default_rng(2)
    w = np.cumsum(rng.normal(size=800))
    y = 1.5 * w + rng.normal(size=800)
    eg = engle_granger(y, w, maxlag=4, autolag=None)
    ref = coint(y, w, trend="c", maxlag=4, autolag=None)
    assert eg["statistic"] == pytest.approx(ref[0], abs=1e-8) and eg["pvalue"] == pytest.approx(ref[1], abs=1e-6)
    assert eg["pvalue"] < 0.01 and eg["hedge_ratio"]["x"] == pytest.approx(1.5, abs=0.1)
    independent = engle_granger(np.cumsum(rng.normal(size=800)), np.cumsum(rng.normal(size=800)))
    assert independent["pvalue"] > 0.01 or independent["statistic"] > -4


def test_granger_matches_statsmodels_and_is_directional():
    rng = np.random.default_rng(3)
    y = rng.normal(size=600)
    x = np.roll(y, 1) * 0.5 + rng.normal(size=600)                # x depends on lagged y: y Granger-causes x, not the reverse
    g = granger_causality(x, y, 3)
    ref = grangercausalitytests(np.column_stack([x, y]), 3)
    for lag in (1, 2, 3):
        assert g.loc[lag, "F"] == pytest.approx(ref[lag][0]["ssr_ftest"][0], rel=1e-9)
        assert g.loc[lag, "chi2"] == pytest.approx(ref[lag][0]["ssr_chi2test"][0], rel=1e-9)
    assert g["pvalue"].max() < 1e-6 and granger_causality(y, x, 3)["pvalue"].min() > 0.001


# ------------------------------------------------------------------------------------------------------------ event study
@pytest.fixture(scope="module")
def events():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015", periods=1500)
    mk = pd.Series(rng.normal(0, 0.01, 1500), index=dates)
    rets = pd.DataFrame({f"a{i}": 1.0 * mk + rng.normal(0, 0.01, 1500) for i in range(30)}, index=dates)
    evd = [dates[600 + 10 * i] for i in range(30)]
    for i, d in enumerate(evd):
        rets.loc[d, f"a{i}"] += 0.03
    return rets, mk, pd.DataFrame({"asset": [f"a{i}" for i in range(30)], "date": evd}), dates


@pytest.mark.parametrize("model", ["market", "mean", "market_adjusted"])
def test_event_study_detects_a_planted_jump(events, model):
    rets, mk, ev, _ = events
    es = event_study(rets, ev, mk, model=model)
    assert es.summary.loc[0, "AAR"] == pytest.approx(0.03, abs=0.006) and es.summary.loc[0, "t"] > 5 and es.summary.loc[0, "z_corrado"] > 3
    assert abs(es.summary.loc[-3, "t"]) < 3.5
    t = es.test_car(0, 1)
    assert t["n_events"] == 30 and t["p_bmp"] < 0.001 and t["share_positive"] > 0.8


def test_event_study_estimation_window_never_overlaps_event_window(events):
    rets, mk, ev, dates = events
    # a huge shock inside the event window must not move the normal-return model, so AR on the shock day is the shock itself
    shocked = rets.copy()
    shocked.loc[dates[600], "a0"] += 1.0
    base = event_study(rets, ev, mk).ar.iloc[:, 0]
    big = event_study(shocked, ev, mk).ar.iloc[:, 0]
    assert big.loc[0] - base.loc[0] == pytest.approx(1.0, abs=1e-9) and np.allclose(big.drop(0), base.drop(0))


def test_event_study_future_data_does_not_change_estimation(events):
    rets, mk, ev, dates = events
    base = event_study(rets, ev, mk).ar.iloc[:, 0]
    changed = rets.copy()
    changed.iloc[700:] = np.random.default_rng(9).normal(size=changed.iloc[700:].shape)
    assert np.allclose(event_study(changed, ev, mk).ar.iloc[:, 0], base)


def test_event_study_validates_and_drops_unusable(events):
    rets, mk, ev, dates = events
    with pytest.raises(ValueError):
        event_study(rets, ev, None, model="market")
    with pytest.raises(ValueError):
        event_study(rets, ev, mk, model="nope")
    early = pd.DataFrame({"asset": ["a0", "zzz"], "date": [dates[10], dates[700]]})
    with pytest.raises(ValueError):
        event_study(rets, early, mk)
    mix = pd.concat([ev.head(5), early])
    es = event_study(rets, mix, mk)
    assert es.events["used"].sum() == 5
    assert event_study(rets, ev, mk, cluster_by_date=True).ar.shape[1] == 30
    assert len(buy_and_hold_abnormal_return(rets, mk, ev, 10)) == 30


# --------------------------------------------------------------------------------------------------------------- inference
@pytest.mark.parametrize("method,ref", [("bonferroni", "bonferroni"), ("holm", "holm"), ("hochberg", "simes-hochberg"), ("bh", "fdr_bh"), ("by", "fdr_by")])
def test_p_adjust_matches_statsmodels(method, ref):
    p = np.random.default_rng(1).uniform(size=60) ** 2
    np.testing.assert_allclose(p_adjust(p, method), multipletests(p, method=ref)[1], atol=1e-12)
    with pytest.raises(ValueError):
        p_adjust(p, "x")


def test_storey_qvalues_estimate_null_share():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=800), rng.uniform(size=200) ** 6])
    q = storey_qvalues(p)
    assert 0.65 < q["pi0"] <= 1.0 and (q["qvalues"] >= 0).all() and (q["qvalues"] <= 1).all()
    assert (q["qvalues"] <= p_adjust(p, "bh") + 1e-12).mean() > 0.95


def test_romano_wolf_controls_familywise_error_and_finds_the_true_edge():
    rng = np.random.default_rng(1)
    R = pd.DataFrame(rng.normal(0, 0.01, size=(1500, 25)), columns=[f"s{i}" for i in range(25)])
    R["s0"] += 0.0015
    out = romano_wolf(R, n_boot=300, seed=2)
    assert out.loc["s0", "p_romano_wolf"] < 0.01
    assert (out.drop("s0")["p_romano_wolf"] > 0.05).mean() > 0.85
    assert (out["p_romano_wolf"] >= out["p_raw"] - 1e-12).all()


def test_sharpe_inference_properties():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 1000)
    assert sharpe_standard_error(r, "iid") == pytest.approx(sharpe_standard_error(r, "mertens"), rel=0.05)
    assert 0.0 <= probabilistic_sharpe_ratio(r) <= 1.0 and probabilistic_sharpe_ratio(r, 5.0, 252) < 0.5
    assert min_track_record_length(r, 0.0, 0.95) > 20 and min_track_record_length(-r) == np.inf
    ar = np.zeros(2000)
    e = rng.normal(0.0005, 0.01, 2000)
    for t in range(1, 2000):
        ar[t] = 0.4 * ar[t - 1] + e[t]
    assert sharpe_standard_error(ar, "lo") > sharpe_standard_error(ar, "iid") * 1.3
    with pytest.raises(ValueError):
        sharpe_standard_error(r, "x")


def test_deflated_sharpe_falls_with_more_trials_and_wider_spread():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0012, 0.01, 1000)
    few, many = deflated_sharpe_ratio(r, n_trials=5), deflated_sharpe_ratio(r, n_trials=500)
    assert many["benchmark"] > few["benchmark"] and many["probability"] < few["probability"]
    spread = deflated_sharpe_ratio(r, trial_sharpes=rng.normal(0, 0.05, 100))
    assert spread["n_trials"] == 100 and spread["benchmark"] > 0
    assert expected_max_sharpe(1) == 0.0 and expected_max_sharpe(100, 1.0) == pytest.approx(2.5, abs=0.15)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(r)


def test_haircut_sharpe_is_monotone_in_number_of_tests():
    a, b = haircut_sharpe(1.0, 240, 10), haircut_sharpe(1.0, 240, 1000)
    assert 0 <= a["haircut"] < b["haircut"] <= 1 and b["sharpe_adjusted"] < a["sharpe_adjusted"] < 1.0
    assert haircut_sharpe(1.0, 240, 100, rho=0.9)["haircut"] < haircut_sharpe(1.0, 240, 100, rho=0.0)["haircut"]


@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_bootstrap_ci_covers_the_true_mean(method):
    rng = np.random.default_rng(7)
    hits = 0
    for i in range(60):
        x = rng.normal(0.5, 1.0, 200)
        ci = bootstrap_ci(x, np.mean, method=method, n_boot=300, seed=i)
        hits += ci["lower"] <= 0.5 <= ci["upper"]
    assert hits >= 50


def test_studentized_and_block_bootstrap_ci_widen_for_dependent_data():
    rng = np.random.default_rng(1)
    x = np.zeros(1500)
    e = rng.normal(size=1500)
    for t in range(1, 1500):
        x[t] = 0.6 * x[t - 1] + e[t]
    iid = bootstrap_ci(x, np.mean, method="percentile", n_boot=400, resample="iid")
    blk = bootstrap_ci(x, np.mean, method="percentile", n_boot=400, resample="stationary")
    assert blk["upper"] - blk["lower"] > 1.4 * (iid["upper"] - iid["lower"])
    st = bootstrap_ci(x[:300], lambda d: (d.mean(), d.std(ddof=1) / np.sqrt(len(d))), method="studentized", n_boot=300)
    assert st["lower"] < st["estimate"] < st["upper"]
    with pytest.raises(ValueError):
        bootstrap_ci(x, np.mean, method="x")


# --------------------------------------------------------------------------------------------------------------- resampling
def test_resampling_indices_are_valid_and_reproducible():
    for kind in ("iid", "moving", "circular", "stationary"):
        i = bootstrap_indices(500, kind, 12, 3)
        assert len(i) == 500 and i.min() >= 0 and i.max() < 500
        assert np.array_equal(i, bootstrap_indices(500, kind, 12, 3))
    assert np.all(np.diff(moving_block_indices(100, 10, 1)[:10]) == 1)
    assert set(np.diff(circular_block_indices(100, 10, 1)[:10]) % 100) == {1}
    runs = np.diff(stationary_indices(100000, 20.0, 1)) != 1
    assert 1 / (runs.mean() + 1e-12) == pytest.approx(20.0, rel=0.1)
    with pytest.raises(ValueError):
        bootstrap_indices(10, "x")


def test_optimal_block_length_grows_with_dependence():
    rng = np.random.default_rng(1)
    e = rng.normal(size=3000)
    x = np.zeros(3000)
    for t in range(1, 3000):
        x[t] = 0.7 * x[t - 1] + e[t]
    assert optimal_block_length(x) > 3 * optimal_block_length(e) and optimal_block_length(e) < 3.0
    assert optimal_block_length(x, "circular") > 1.0 and optimal_block_length(np.arange(10.0)) >= 1.0


def test_block_bootstrap_standard_error_matches_theory():
    rng = np.random.default_rng(1)
    n, rho = 4000, 0.5
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    from src.probability.resampling import bootstrap_statistic
    se = bootstrap_statistic(x, np.mean, 600, "stationary", seed=3).std()
    theory = x.std() / np.sqrt(n) * np.sqrt((1 + rho) / (1 - rho))
    assert se == pytest.approx(theory, rel=0.2)


def test_wild_multipliers_and_subsampling():
    for kind in ("rademacher", "mammen", "gaussian"):
        m = wild_bootstrap_multipliers(200000, kind, 1)
        assert abs(m.mean()) < 0.02 and m.var() == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError):
        wild_bootstrap_multipliers(5, "x")
    x = np.random.default_rng(0).normal(1.0, 1.0, 2000)
    res = subsample_statistics(x, np.mean, 200)
    lo, hi = subsample_ci(res)
    assert lo < 1.0 < hi and res["n"] == 2000
