"""Econometric models against statsmodels / arch (reference implementations) and against simulated data with known parameters."""

import warnings

import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from src.econometrics import (bai_ng_criteria, error_correction_model, ewma_volatility, fit_arima, fit_bvar, fit_dynamic_factor, fit_garch, fit_var, fit_vecm,
                              johansen, kalman_hedge_ratio, local_level, local_linear_trend, qlike, rolling_forecasts, select_arima, select_var_order,
                              time_varying_regression, walk_forward_volatility)
from src.econometrics.arima import ar_ols, ar_to_pacf, arma_loglik, arma_state_space, pacf_to_ar, yule_walker
from src.econometrics.factor import nowcast
from src.econometrics.statespace import StateSpace
from src.probability.montecarlo import simulate_garch

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------------------------------------- state space
@pytest.fixture(scope="module")
def local_level_series():
    rng = np.random.default_rng(0)
    n = 400
    return pd.Series(np.cumsum(rng.normal(0, 0.3, n)) + rng.normal(0, 1, n), index=pd.bdate_range("2020", periods=n))


def test_local_level_matches_statsmodels(local_level_series):
    r = local_level(local_level_series)
    ref = UnobservedComponents(local_level_series.values, "llevel").fit(disp=False)
    assert r.loglik == pytest.approx(ref.llf, abs=1e-3)
    assert r.params["sigma2_eps"] == pytest.approx(ref.params[0], rel=0.01) and r.params["sigma2_eta"] == pytest.approx(ref.params[1], rel=0.02)
    assert r.signal_to_noise["q"] == pytest.approx(ref.params[1] / ref.params[0], rel=0.03)
    assert r.forecast(3).shape == (3,)


def test_local_linear_trend_recovers_slope_and_filter_is_causal():
    rng = np.random.default_rng(0)
    n = 400
    slope = np.cumsum(rng.normal(0, 0.02, n)) + 0.05
    y = pd.Series(np.cumsum(slope) + rng.normal(0, 0.5, n), index=pd.bdate_range("2020", periods=n))
    r = local_linear_trend(y)
    ref = UnobservedComponents(y.values, "lltrend").fit(disp=False)
    assert r.loglik == pytest.approx(ref.llf, abs=1e-2)
    assert np.corrcoef(r.filtered["slope"].iloc[50:], slope[50:])[0, 1] > 0.9
    changed = y.copy()
    changed.iloc[300:] += np.random.default_rng(1).normal(0, 50, 100)
    f2 = r.model.filter(changed.to_numpy(), 2)
    assert np.allclose(f2.filtered_state[:300], r.model.filter(y.to_numpy(), 2).filtered_state[:300])         # filtered states never use the future
    assert not np.allclose(r.smoothed["slope"].iloc[:300].to_numpy(), r.filtered["slope"].iloc[:300].to_numpy())  # smoothed ones do


def test_kalman_filter_handles_missing_observations_and_matches_dense_case():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100)) + rng.normal(size=100)
    m = StateSpace([[1.0]], [[1.0]], [[1.0]], [[1.0]], P0=[[1e7]])
    full = m.filter(y, 1)
    holes = y.copy()
    holes[40:45] = np.nan
    f = m.filter(holes, 1)
    assert np.all(np.isfinite(f.filtered_state)) and f.filtered_cov[44, 0, 0] > f.filtered_cov[39, 0, 0]       # uncertainty grows without data
    assert f.filtered_state[:40] == pytest.approx(full.filtered_state[:40])
    sm = m.smooth(y, 1)
    assert sm["smoothed_cov"][50, 0, 0] <= full.filtered_cov[50, 0, 0] + 1e-9


def test_time_varying_regression_tracks_a_moving_beta_and_errors_are_white():
    rng = np.random.default_rng(1)
    n = 800
    x = pd.Series(rng.normal(size=n), index=pd.bdate_range("2018", periods=n))
    b = 1.0 + np.cumsum(rng.normal(0, 0.05, n))
    y = pd.Series(b * x + rng.normal(0, 0.5, n), index=x.index)
    tv = time_varying_regression(y, x.to_frame("x"), add_constant=False)
    assert np.corrcoef(tv["filtered"]["x"].iloc[100:], b[100:])[0, 1] > 0.85
    assert tv["standardised_errors"].std() == pytest.approx(1.0, abs=0.1)
    from src.stats import ljung_box
    assert ljung_box(tv["standardised_errors"].iloc[50:], 10)["pvalue"] > 0.001
    fixed = time_varying_regression(y, x.to_frame("x"), q=1e-6, fit_q=False, add_constant=False)
    assert fixed["filtered"]["x"].std() < tv["filtered"]["x"].std()


def test_kalman_hedge_ratio_for_a_cointegrated_pair_and_causal_spread():
    rng = np.random.default_rng(2)
    n = 1000
    x = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=pd.bdate_range("2018", periods=n))
    y = 5 + 1.5 * x + pd.Series(rng.normal(0, 1.0, n), index=x.index)
    h = kalman_hedge_ratio(y, x)
    assert h["beta"].iloc[-1] == pytest.approx(1.5, abs=0.05) and abs(h["zscore"].iloc[100:].std() - 1.0) < 0.2
    y2 = y.copy()
    y2.iloc[600:] += 500
    f1 = time_varying_regression(y, x.to_frame("x"), q=0.01, fit_q=False)["filtered"]["x"]
    f2 = time_varying_regression(y2, x.to_frame("x"), q=0.01, fit_q=False)["filtered"]["x"]
    assert np.allclose(f1.iloc[:600], f2.iloc[:600])


# ---------------------------------------------------------------------------------------------------------------------- ARIMA
@pytest.fixture(scope="module")
def arma_series():
    rng = np.random.default_rng(1)
    n = 600
    e = rng.normal(size=n + 50)
    y = np.zeros(n + 50)
    for t in range(2, n + 50):
        y[t] = 0.5 + 0.6 * y[t - 1] - 0.2 * y[t - 2] + e[t] + 0.4 * e[t - 1]
    return pd.Series(y[50:], index=pd.bdate_range("2018", periods=n))


def test_pacf_maps_are_inverses_and_always_stationary():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pacf = np.tanh(rng.normal(size=4) * 2)
        phi = pacf_to_ar(pacf)
        assert ar_to_pacf(phi) == pytest.approx(pacf, abs=1e-9)
        roots = np.roots(np.r_[1.0, -phi])
        assert np.all(np.abs(roots) < 1.0)


@pytest.mark.parametrize("ar,ma", [([0.6, -0.2], [0.4]), ([0.5], []), ([], [0.5, 0.2]), ([0.9], [-0.6]), ([], [])])
def test_fast_arma_likelihood_equals_the_full_kalman_filter(ar, ma):
    rng = np.random.default_rng(1)
    ar, ma = np.array(ar), np.array(ma)
    e = rng.normal(size=500)
    w = np.zeros(500)
    for t in range(500):
        w[t] = e[t] + sum(ar[i] * w[t - 1 - i] for i in range(len(ar)) if t - 1 - i >= 0) + sum(ma[j] * e[t - 1 - j] for j in range(len(ma)) if t - 1 - j >= 0)
    assert arma_loglik(w, ar, ma, 1.0) == pytest.approx(arma_state_space(ar, ma, 1.0).filter(w).loglik, abs=1e-8)


def test_arima_matches_statsmodels_params_loglik_and_forecasts(arma_series):
    r = fit_arima(arma_series, (2, 0, 1))
    ref = ARIMA(arma_series.values, order=(2, 0, 1), trend="c").fit()
    assert r.loglik == pytest.approx(ref.llf, abs=1e-4) and r.aic == pytest.approx(ref.aic, abs=1e-3)
    assert np.r_[r.mean, r.ar, r.ma, r.sigma2] == pytest.approx(ref.params, abs=2e-3)
    f = r.forecast(3)
    sf = ref.get_forecast(3).summary_frame()
    assert f["mean"].to_numpy() == pytest.approx(sf["mean"].to_numpy(), abs=1e-3) and f["se"].to_numpy() == pytest.approx(sf["mean_se"].to_numpy(), abs=1e-3)
    assert r.ljung_box(10)["pvalue"] > 0.01 and abs(r.residuals.std() - 1.0) < 0.1


def test_arima_with_differencing_integrates_forecasts_and_variance(arma_series):
    level = arma_series.cumsum()
    r = fit_arima(level, (1, 1, 1))
    ref = ARIMA(level.values, order=(1, 1, 1)).fit()
    assert r.loglik == pytest.approx(ref.llf, abs=1e-3)
    f, sf = r.forecast(4), ref.get_forecast(4).summary_frame()
    assert f["mean"].to_numpy() == pytest.approx(sf["mean"].to_numpy(), abs=5e-3) and f["se"].to_numpy() == pytest.approx(sf["mean_se"].to_numpy(), rel=5e-3)
    assert (np.diff(f["se"]) > 0).all()


def test_arima_selection_and_simple_ar_estimators(arma_series):
    table = select_arima(arma_series, 2, 2, criterion="bic")
    assert table.iloc[0]["bic"] <= table["bic"].min() and len(table) == 9
    top = table.iloc[0]
    assert (top["p"], top["q"]) != (0, 0)
    ols = ar_ols(arma_series, 2)
    yw = yule_walker(arma_series, 2)
    assert ols["ar"] == pytest.approx(yw["ar"], abs=0.05) and ols["sigma2"] > 0
    white = pd.Series(np.random.default_rng(3).normal(size=800), index=pd.bdate_range("2018", periods=800))
    assert select_arima(white, 2, 2, criterion="bic").iloc[0][["p", "q"]].tolist() == [0, 0]


def test_rolling_forecasts_use_only_the_past():
    rng = np.random.default_rng(2)
    n = 900
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + rng.normal()
    s = pd.Series(y, index=pd.bdate_range("2015", periods=n))
    rf = rolling_forecasts(s, (1, 0, 0), min_train=500, refit_every=50)
    r2 = 1 - ((rf["actual"] - rf["forecast"]) ** 2).sum() / ((rf["actual"] - rf["actual"].mean()) ** 2).sum()
    assert r2 > 0.3 and len(rf) == n - 1 - 500
    changed = s.copy()
    changed.iloc[700:] = rng.normal(0, 30, n - 700)
    rf2 = rolling_forecasts(changed, (1, 0, 0), min_train=500, refit_every=50)
    cut = s.index[699]
    assert np.allclose(rf.loc[:cut, "forecast"].astype(float), rf2.loc[:cut, "forecast"].astype(float))


# ------------------------------------------------------------------------------------------------------------------------- VAR
@pytest.fixture(scope="module")
def var_data():
    rng = np.random.default_rng(0)
    n = 500
    A = np.array([[0.5, 0.1], [0.2, 0.3]])
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = A @ y[t - 1] + rng.normal(size=2)
    return pd.DataFrame(y, columns=["a", "b"], index=pd.bdate_range("2019", periods=n))


def test_var_matches_statsmodels_everywhere(var_data):
    r, s = fit_var(var_data, 2), VAR(var_data).fit(2, trend="c")
    assert np.concatenate(list(r.coefs), axis=1) == pytest.approx(np.hstack([s.coefs[0], s.coefs[1]]), abs=1e-10)
    assert r.sigma_u == pytest.approx(s.sigma_u, abs=1e-10) and r.loglik == pytest.approx(s.llf, abs=1e-8)
    assert r.forecast(3).to_numpy() == pytest.approx(s.forecast(var_data.values[-2:], 3), abs=1e-10)
    assert r.irf(6) == pytest.approx(s.irf(5).orth_irfs, abs=1e-10) and r.ma_representation(6) == pytest.approx(s.irf(5).irfs, abs=1e-10)
    assert r.fevd(5) == pytest.approx(s.fevd(5).decomp.transpose(1, 0, 2), abs=1e-10)
    g = r.granger("a", "b")
    assert g["chi2_pvalue"] == pytest.approx(s.test_causality("a", "b", kind="wald").pvalue, abs=1e-9)
    assert r.is_stable() and np.allclose(r.fevd(5).sum(axis=2), 1.0) and len(r.forecast_cov(3)) == 3


def test_var_order_selection_picks_the_true_order_and_se_shape(var_data):
    t = select_var_order(var_data, 5)
    assert t["bic"].idxmin() == 1 and set(t.columns) == {"aic", "bic", "hqic"}
    assert fit_var(var_data, 1).se.shape == (3, 2)


def test_unstable_var_is_flagged():
    rng = np.random.default_rng(1)
    y = np.zeros((300, 2))
    for t in range(1, 300):
        y[t] = np.array([[1.02, 0.0], [0.0, 0.5]]) @ y[t - 1] + rng.normal(size=2)
    assert not fit_var(pd.DataFrame(y, columns=["a", "b"]), 1).is_stable()


# ------------------------------------------------------------------------------------------------------------ cointegration
@pytest.fixture(scope="module")
def coint_data():
    rng = np.random.default_rng(0)
    n = 500
    w = np.cumsum(rng.normal(size=n))
    return pd.DataFrame({"x": w + rng.normal(size=n) * 0.5, "y": 1.5 * w + rng.normal(size=n) * 0.5, "z": np.cumsum(rng.normal(size=n))}, index=pd.bdate_range("2019", periods=n))


@pytest.mark.parametrize("det", [-1, 0])
def test_johansen_matches_statsmodels_and_finds_one_relation(coint_data, det):
    j, cj = johansen(coint_data, 1, det), coint_johansen(coint_data, det, 1)
    assert j["trace"] == pytest.approx(cj.lr1, abs=1e-6) and j["max_eig"] == pytest.approx(cj.lr2, abs=1e-6)
    assert j["trace_cv"] == pytest.approx(cj.cvt) and j["eigenvalues"] == pytest.approx(cj.eig, abs=1e-9)
    assert j["rank"] == 1
    assert johansen(coint_data[["x", "z"]], 1, 0)["rank"] == 0
    with pytest.raises(ValueError):
        johansen(coint_data, 1, 1)


@pytest.mark.parametrize("det", ["ci", "co", "n"])
def test_vecm_matches_statsmodels(coint_data, det):
    v = fit_vecm(coint_data[["x", "y"]], 1, 1, det)
    s = VECM(coint_data[["x", "y"]], k_ar_diff=1, coint_rank=1, deterministic=det).fit()
    assert v.beta.values[:2].ravel() == pytest.approx(s.beta.ravel(), abs=1e-8) and v.alpha.values.ravel() == pytest.approx(s.alpha.ravel(), abs=1e-8)
    assert v.loglik == pytest.approx(s.llf, abs=1e-6) and v.gamma[0] == pytest.approx(s.gamma, abs=1e-8)


def test_vecm_error_correction_term_is_stationary_and_adjusts_toward_equilibrium(coint_data):
    from src.stats import adf_test
    v = fit_vecm(coint_data[["x", "y"]], 1, 1, "ci")
    ect = v.error_correction_term(coint_data[["x", "y"]])["beta1"]
    assert adf_test(ect.to_numpy()).reject() and not adf_test(coint_data["x"].to_numpy()).reject()
    assert (v.alpha.abs() > 0.05).any().any()                                                                    # at least one variable error-corrects
    assert np.isfinite(v.to_var()).all() and v.to_var().shape[0] == 2


def test_error_correction_model_speed_is_negative_with_half_life(coint_data):
    e = error_correction_model(coint_data["y"], coint_data["x"])
    assert e["speed"] < 0 and e["t_speed"] < -4 and 0 < e["half_life"] < 5 and e["hedge_ratio"] == pytest.approx(1.5, abs=0.15)


# ----------------------------------------------------------------------------------------------------------------------- GARCH
@pytest.fixture(scope="module")
def garch_returns():
    return pd.Series(simulate_garch(2500, 1, 2e-6, 0.08, 0.9, mu=0.0004, df=6, seed=3)[0], index=pd.bdate_range("2010", periods=2500))


@pytest.mark.parametrize("model,vol,kw", [("garch", "GARCH", {}), ("gjr", "GARCH", {"o": 1}), ("egarch", "EGARCH", {"o": 1})])
@pytest.mark.parametrize("dist", ["normal", "t"])
def test_garch_family_matches_arch(garch_returns, model, vol, kw, dist):
    m = fit_garch(garch_returns, model, dist)
    a = arch_model(garch_returns * 100, mean="Constant", vol=vol, p=1, q=1, dist=dist, **kw).fit(disp="off")
    assert m.loglik == pytest.approx(a.loglikelihood + len(garch_returns) * np.log(100), abs=0.05)
    assert m.params["alpha"] == pytest.approx(float(a.params["alpha[1]"]), abs=2e-3) and m.params["beta"] == pytest.approx(float(a.params["beta[1]"]), abs=2e-3)
    if "gamma" in m.params:
        assert m.params["gamma"] == pytest.approx(float(a.params["gamma[1]"]), abs=3e-3)
    if dist == "t":
        assert m.params["nu"] == pytest.approx(float(a.params["nu"]), rel=0.02)
    if model != "egarch":
        assert m.params["omega"] * 1e4 == pytest.approx(float(a.params["omega"]), rel=0.05)
    assert m.volatility.iloc[-1] * 100 == pytest.approx(float(a.conditional_volatility.iloc[-1]), rel=0.01)


def test_garch_recovers_true_parameters_and_diagnostics(garch_returns):
    m = fit_garch(garch_returns, "garch", "t")
    assert m.params["alpha"] == pytest.approx(0.08, abs=0.03) and m.params["beta"] == pytest.approx(0.90, abs=0.04) and m.params["nu"] == pytest.approx(6.0, abs=2.0)
    assert m.persistence == pytest.approx(0.98, abs=0.02) and m.half_life > 15 and m.unconditional_variance == pytest.approx(1e-4, rel=0.4)
    from src.stats import arch_lm
    assert arch_lm(garch_returns.to_numpy(), 5)["pvalue"] < 1e-6 and arch_lm(m.std_resid.to_numpy(), 5)["pvalue"] > 0.001
    assert (m.volatility > 0).all() and m.bic > m.aic - 1e9


def test_garch_forecasts_match_arch_and_leverage_is_detected():
    a = np.random.default_rng(1)
    n = 3000
    e, s2, r = a.standard_normal(n), np.full(n, 1e-4), np.zeros(n)
    for t in range(1, n):
        s2[t] = 2e-6 + (0.03 + 0.12 * (r[t - 1] < 0)) * r[t - 1] ** 2 + 0.9 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * e[t]
    ret = pd.Series(r, index=pd.bdate_range("2008", periods=n))
    g = fit_garch(ret, "gjr", "normal")
    assert g.params["gamma"] > 0.05
    ni = g.news_impact()
    assert ni.iloc[0] > ni.iloc[-1]                                     # a large fall raises volatility more than an equal rise
    am = arch_model(ret * 100, vol="GARCH", p=1, o=1, q=1).fit(disp="off")
    assert g.forecast_variance(5) == pytest.approx(am.forecast(horizon=5).variance.iloc[-1].to_numpy() / 1e4, rel=0.01)
    eg = fit_garch(ret, "egarch", "normal")
    assert eg.params["gamma"] < 0 and eg.forecast_variance(3, 20000).shape == (3,)


def test_garch_input_validation_and_zero_mean(garch_returns):
    with pytest.raises(ValueError):
        fit_garch(garch_returns, "x")
    z = fit_garch(garch_returns, "garch", "normal", "zero")
    assert z.params["mu"] == 0.0


def test_walk_forward_volatility_is_causal_and_competitive_with_ewma(garch_returns):
    r = garch_returns.iloc[:1100]
    wf = walk_forward_volatility(r, "garch", "normal", min_train=700, refit_every=100)
    assert len(wf) == 400 and (wf["forecast"] > 0).all()
    q_g, q_e = qlike(wf["realised_sq"], wf["forecast"] ** 2), qlike(wf["realised_sq"], wf["ewma"] ** 2)
    assert q_g < q_e * 1.15
    changed = r.copy()
    changed.iloc[900:] = np.random.default_rng(1).normal(0, 0.05, 200)
    wf2 = walk_forward_volatility(changed, "garch", "normal", min_train=700, refit_every=100)
    assert np.allclose(wf["forecast"].iloc[:200], wf2["forecast"].iloc[:200])
    e = ewma_volatility(r)
    assert np.isnan(e.iloc[0]) and (e.dropna() > 0).all()
    assert qlike([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------------------------- factors, BVAR
@pytest.fixture(scope="module")
def factor_panel():
    rng = np.random.default_rng(0)
    T, N, k = 400, 30, 2
    F = np.zeros((T, k))
    A = np.array([[0.7, 0.1], [0.0, 0.5]])
    for t in range(1, T):
        F[t] = A @ F[t - 1] + rng.normal(size=k)
    L = rng.normal(size=(N, k))
    return F, pd.DataFrame(F @ L.T + rng.normal(size=(T, N)) * 0.7, index=pd.bdate_range("2010", periods=T), columns=[f"s{i}" for i in range(N)])


def test_bai_ng_selects_two_factors(factor_panel):
    t = bai_ng_criteria(factor_panel[1], 5)
    assert t["ic1"].idxmin() == 2 and t["ic3"].idxmin() == 2 and t["variance_share"].is_monotonic_increasing


def test_dynamic_factor_model_recovers_factors_handles_gaps_and_is_causal(factor_panel):
    F, X = factor_panel
    r = fit_dynamic_factor(X, 2, 1)
    C = np.abs(np.corrcoef(np.column_stack([r.filtered_factors.to_numpy(), F]).T)[:2, 2:])
    assert (C.max(axis=1) > 0.95).all() and r.transition.shape == (2, 2) and (r.idio_var > 0).all()
    changed = X.copy()
    changed.iloc[300:] = np.random.default_rng(1).normal(size=changed.iloc[300:].shape) * 10
    f_alt = r.model.filter(((changed - r.mean) / r.std).to_numpy()).filtered_state[:300, :2]
    assert np.allclose(f_alt, r.filtered_factors.iloc[:300].to_numpy())
    gappy = X.copy()
    gappy.iloc[-10:, :10] = np.nan
    r2 = fit_dynamic_factor(gappy, 2, 1)
    assert np.isfinite(r2.filtered_factors.to_numpy()).all()
    nc = nowcast(r2, "s0", gappy)
    assert nc.notna().all() and nc.iloc[-1] != 0
    assert r.forecast(2).shape == (2, 30) and r.forecast_factors(2).shape == (2, 2) and r.common_component().shape == X.shape


def test_bvar_recovers_ols_without_shrinkage_and_shrinks_with_a_tight_prior(factor_panel):
    X = factor_panel[1].iloc[:, :5]
    ols, loose = fit_var(X, 2), fit_bvar(X, 2, lam=1e8, random_walk_mean=0.0)
    assert loose.coefs == pytest.approx(ols.coefs, abs=1e-5)
    tight = fit_bvar(X, 2, lam=0.05, random_walk_mean=0.0)
    assert np.abs(tight.coefs).max() < np.abs(ols.coefs).max() and tight.is_stable()
    assert np.abs(tight.coefs).sum() < np.abs(loose.coefs).sum()
    assert fit_bvar(X, 2, lam=0.2, sum_coef_mu=1.0, co_persistence=1.0).coefs.shape == (2, 5, 5)


def test_tgarch_moments_leverage_and_forecasts():
    from src.econometrics.garch import mean_abs_innovation, tgarch_moments

    assert mean_abs_innovation("normal") == pytest.approx(np.sqrt(2 / np.pi)) and mean_abs_innovation("t", 1e6) == pytest.approx(np.sqrt(2 / np.pi), rel=1e-4)
    z = np.random.default_rng(0).standard_t(7, 2_000_000) * np.sqrt(5 / 7)
    assert mean_abs_innovation("t", 7.0) == pytest.approx(np.abs(z).mean(), rel=2e-3)
    ec, ec2 = tgarch_moments(0.05, 0.08, 0.9, np.sqrt(2 / np.pi))
    zz = np.random.default_rng(1).standard_normal(2_000_000)
    cc = (0.05 + 0.08 * (zz < 0)) * np.abs(zz) + 0.9
    assert ec == pytest.approx(cc.mean(), rel=2e-3) and ec2 == pytest.approx((cc ** 2).mean(), rel=5e-3)
    rng = np.random.default_rng(2)
    n = 3000
    e = rng.standard_normal(n)
    sig, r = np.full(n, 0.01), np.zeros(n)
    for t in range(1, n):
        sig[t] = 4e-4 + (0.03 + 0.10 * (r[t - 1] < 0)) * abs(r[t - 1]) + 0.9 * sig[t - 1]
        r[t] = sig[t] * e[t]
    ret = pd.Series(r, index=pd.bdate_range("2008", periods=n))
    g = fit_garch(ret, "tgarch", "normal")
    assert g.params["gamma"] > 0.04 and 0.8 < g.params["beta"] < 0.97 and g.persistence < 1 and np.isfinite(g.unconditional_variance)
    ni = g.news_impact()
    assert ni.iloc[0] > ni.iloc[-1]                                        # a fall raises volatility more than an equal rise
    f = g.forecast_variance(5, 40000)
    e_last, p = float(ret.iloc[-1] - g.params["mu"]), g.params
    one_step = (p["omega"] + (p["alpha"] + p["gamma"] * (e_last < 0)) * abs(e_last) + p["beta"] * g.volatility.iloc[-1]) ** 2
    assert f.shape == (5,) and np.all(f > 0) and f[0] == pytest.approx(one_step, rel=1e-6)
    wf = walk_forward_volatility(ret.iloc[:1300], "tgarch", "normal", min_train=800, refit_every=250)
    assert wf["forecast"].notna().all() and (wf["forecast"] > 0).all()


@pytest.mark.parametrize("dist", ["normal", "t"])
def test_tgarch_likelihood_and_volatility_equal_arch_at_the_same_parameters(garch_returns, dist):
    """``arch`` (power = 1) constrains alpha + gamma/2 + beta < 1, which excludes some admissible TGARCH solutions, so its own optimum can be lower; evaluated at OUR parameters its
    likelihood must equal ours to within the pre-sample convention (arch starts a power-one recursion from a slightly different initial value; the difference decays geometrically and is below 1e-5 after 100 observations), and so must the conditional volatility from then on."""
    m = fit_garch(garch_returns, "tgarch", dist)
    p = m.params
    vec = [p["mu"] * 100, p["omega"] * 100, p["alpha"], p["gamma"], p["beta"]] + ([p["nu"]] if dist == "t" else [])
    fixed = arch_model(garch_returns * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, power=1.0, dist=dist).fix(vec)
    assert m.loglik == pytest.approx(fixed.loglikelihood + len(garch_returns) * np.log(100), abs=0.1)
    assert m.volatility.to_numpy()[100:] * 100 == pytest.approx(fixed.conditional_volatility.to_numpy()[100:], rel=1e-5)
    assert m.volatility.to_numpy()[0] * 100 == pytest.approx(fixed.conditional_volatility.to_numpy()[0], rel=0.01)
    assert m.params["alpha"] == pytest.approx(0.08, abs=0.05) and m.params["beta"] > 0.8
