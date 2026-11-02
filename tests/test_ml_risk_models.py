"""Release 6, P2-07: covariance estimators fitted on prior eligible returns, repaired when they must be, compared on the risk a portfolio realised.

References are written here from the definitions: the EWMA weights, scikit-learn's Ledoit-Wolf for the identity target, the closed-form global-minimum-variance book, and the eigen-decomposition of the sample
matrix for the factor model. The comparison fixture has a known factor structure, so what it shows (shrinkage and factor structure beat the sample matrix when assets are many and observations few) is the
textbook result, asserted with a margin."""

import numpy as np
import pandas as pd
import pytest

from src.ml import risk_models as rm

IDX = pd.bdate_range("2019-01-01", periods=800)


def factor_returns(n_assets=20, t=800, k=2, seed=0, idio=0.008):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 1.0, (n_assets, k)) * 0.5
    factors = rng.normal(0.0, 0.01, (t, k))
    eps = rng.normal(0.0, idio, (t, n_assets))
    return pd.DataFrame(factors @ loadings.T + eps, index=pd.bdate_range("2019-01-01", periods=t), columns=[f"a{i:02d}" for i in range(n_assets)])


R = factor_returns()


def test_ewma_is_the_exponentially_weighted_covariance_with_bias_correction():
    spec = rm.CovSpec("ewma", window=252, halflife=30.0)
    fit = rm.fit_asof(R, R.index[400], spec)
    w = R.loc[:R.index[400]].tail(252)
    age = np.arange(len(w))[::-1]
    wt = 0.5 ** (age / 30.0)
    wt = wt / wt.sum()
    centred = w.to_numpy() - wt @ w.to_numpy()
    ref = (centred * wt[:, None]).T @ centred / (1.0 - np.sum(wt ** 2))
    assert np.allclose(fit.matrix.to_numpy(), ref, rtol=1e-10, atol=1e-14) and fit.record()["units"] == "per day"
    assert fit.n_obs == 252 and fit.window_end == R.index[400] and fit.window_start == R.index[400 - 251]


def test_the_identity_target_shrinkage_is_scikit_learns_and_the_constant_correlation_one_is_between_its_poles():
    from sklearn.covariance import LedoitWolf

    window = R.loc[:R.index[500]].tail(120)
    fit = rm.fit_asof(R, R.index[500], rm.CovSpec("lw_identity", window=120))
    sk = LedoitWolf().fit(window.to_numpy())
    assert np.allclose(fit.matrix.to_numpy(), sk.covariance_, rtol=1e-10, atol=1e-14) and fit.shrinkage == pytest.approx(sk.shrinkage_, abs=1e-12)
    cc = rm.fit_asof(R, R.index[500], rm.CovSpec("lw_constant_correlation", window=120))
    sample = rm.fit_asof(R, R.index[500], rm.CovSpec("sample", window=120))
    assert 0.0 <= cc.shrinkage <= 1.0
    off = ~np.eye(len(cc.matrix), dtype=bool)
    assert np.abs(cc.matrix.to_numpy()[off]).mean() < np.abs(sample.matrix.to_numpy()[off]).mean() * 1.05 or cc.shrinkage < 0.05, "shrinking towards one common correlation does not inflate the pairwise covariances"
    assert np.allclose(np.diag(cc.matrix), np.diag(sample.matrix), rtol=0.2)


def test_the_factor_model_restores_every_variance_and_has_the_declared_rank_in_its_common_part():
    window = R.loc[:R.index[500]].tail(252)
    fit = rm.fit_asof(R, R.index[500], rm.CovSpec("factor", window=252, n_factors=2))
    s = window.cov(ddof=1).to_numpy()
    assert fit.n_factors_used == 2 and np.allclose(np.diag(fit.matrix), np.diag(s), rtol=1e-10), "the residual variances restore each asset's own variance exactly"
    values, vectors = np.linalg.eigh(s)
    top = np.argsort(values)[::-1][:2]
    common = (vectors[:, top] * values[top]) @ vectors[:, top].T
    off = ~np.eye(len(s), dtype=bool)
    assert np.allclose(fit.matrix.to_numpy()[off], common[off], rtol=1e-8, atol=1e-14), "off the diagonal the matrix is the rank-2 common part"
    auto = rm.fit_asof(R, R.index[500], rm.CovSpec("factor", window=252))
    assert 1 <= auto.n_factors_used <= 4, "the Marchenko-Pastur bound picks the few eigenvalues that stand out of the noise"


def test_a_fit_uses_only_returns_up_to_and_including_the_decision_date():
    d = R.index[450]
    base = rm.fit_asof(R, d, rm.CovSpec("ewma", window=200, halflife=40.0))
    later = R.copy()
    later.loc[later.index > d] = 10.0 * later.loc[later.index > d]
    assert np.array_equal(base.matrix.to_numpy(), rm.fit_asof(later, d, rm.CovSpec("ewma", window=200, halflife=40.0)).matrix.to_numpy()), "nothing after the decision date can change the fit"
    earlier = R.copy()
    earlier.loc[d - pd.Timedelta(days=20)] = 5.0 * earlier.loc[d - pd.Timedelta(days=20)] if (d - pd.Timedelta(days=20)) in earlier.index else earlier.iloc[430]
    changed = R.copy()
    changed.iloc[440] = changed.iloc[440] * 5.0
    assert not np.allclose(base.matrix.to_numpy(), rm.fit_asof(changed, d, rm.CovSpec("ewma", window=200, halflife=40.0)).matrix.to_numpy()), "a return inside the window does"


def test_ineligible_and_thin_assets_are_dropped_with_their_reasons_and_too_few_assets_refuse():
    r = R.iloc[:, :8].copy()
    r.iloc[:360, 3] = np.nan                                                                       # a late listing: 40 observations in a 252-day window ending at 400
    r.iloc[:300, 6] = np.nan                                                                       # a shorter history that is still long enough: the complete cases are the rows where it exists
    elig = pd.DataFrame(True, index=r.index, columns=r.columns)
    elig.iloc[:, 5] = False
    fit = rm.fit_asof(r, r.index[400], rm.CovSpec("sample", window=252), elig)
    assert fit.dropped == {"a03": "fewer than 60 observations in the window", "a05": "not eligible at the decision date"} and fit.assets == ["a00", "a01", "a02", "a04", "a06", "a07"]
    assert fit.n_obs == 101 and fit.window_start == r.index[300] and fit.window_end == r.index[400], "the estimate uses the rows on which every kept asset has a return"
    with pytest.raises(ValueError, match="fewer than two assets"):
        rm.fit_asof(r.iloc[:, :2].assign(a01=np.nan), r.index[400], rm.CovSpec("sample"))
    with pytest.raises(ValueError, match="unknown covariance method"):
        rm.CovSpec("magic")
    with pytest.raises(ValueError, match="at least min_obs"):
        rm.CovSpec("sample", window=30, min_obs=60)


def test_a_singular_estimate_is_symmetrised_and_repaired_and_the_record_says_so():
    wide = factor_returns(n_assets=30, t=300, seed=3)
    fit = rm.fit_asof(wide, wide.index[200], rm.CovSpec("sample", window=60, min_obs=60))              # 60 observations for 30 assets: the sample matrix is not positive definite once rounding is counted
    rec = fit.record()
    f = fit.matrix.to_numpy()
    assert rec["repaired"] is True or rec["min_eigenvalue_raw"] > 1e-10
    assert np.allclose(f, f.T, atol=0) and np.linalg.eigvalsh(f).min() > 0 and fit.min_eig_final >= 1e-10 - 1e-18
    raw = wide.loc[:wide.index[200]].tail(60).cov(ddof=1).to_numpy()
    assert np.allclose(np.diag(f), np.diag(raw), rtol=1e-6), "the repair restores the diagonal"
    tiny = factor_returns(n_assets=70, t=200, seed=4)                                                  # 70 assets, 60 observations: the sample matrix has rank 59
    lw = rm.fit_asof(tiny, tiny.index[150], rm.CovSpec("sample", window=60, min_obs=60))
    assert lw.repaired and lw.condition_raw == float("inf") and lw.record()["condition_raw"] is None and lw.condition_final < 1e12 and lw.repair_size > 0


def test_risk_contributions_sum_to_one_and_follow_the_definition():
    cov = pd.DataFrame([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.01]], index=list("abc"), columns=list("abc"))
    w = pd.Series([0.5, 0.2, 0.3], index=list("abc"))
    rc = rm.risk_contributions(w, cov)
    var = w.to_numpy() @ cov.to_numpy() @ w.to_numpy()
    assert rc["share"].sum() == pytest.approx(1.0) and rc["contribution"].sum() == pytest.approx(np.sqrt(var))
    assert rc.loc["a", "share"] == pytest.approx(0.5 * (0.04 * 0.5 + 0.01 * 0.2 + 0.0 * 0.3) / var)
    assert rm.risk_contributions(pd.Series(0.0, index=list("abc")), cov)["share"].isna().all(), "a flat book has no risk shares, not zeros"


def test_gmv_weights_are_the_closed_form():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=list("ab"), columns=list("ab"))
    w = rm.gmv_weights(cov)
    assert w["a"] == pytest.approx((0.09 - 0.01) / (0.04 + 0.09 - 2 * 0.01)) and w.sum() == pytest.approx(1.0)


def test_shrinkage_and_factor_structure_beat_the_sample_matrix_when_observations_are_few():
    wide = factor_returns(n_assets=30, t=900, k=2, seed=7)
    dates = wide.index[120:800:21]
    specs = [rm.CovSpec("sample", window=60), rm.CovSpec("lw_constant_correlation", window=60), rm.CovSpec("factor", window=60, n_factors=2), rm.CovSpec("ewma", window=60, halflife=20.0)]
    out = rm.compare(wide, dates, specs, horizon=21)
    est = out["estimators"]
    sample, lw, fac = est["sample/w60"], est["lw_constant_correlation/w60"], est["factor/k2/w60"]
    assert sample["windows"] == lw["windows"] == fac["windows"] >= 25 and sample["failed"] == 0
    assert lw["gmv_realised_risk"] < 0.95 * sample["gmv_realised_risk"] and fac["gmv_realised_risk"] < 0.95 * sample["gmv_realised_risk"], "the minimum-variance book of a noisy matrix realises more risk"
    assert lw["gmv_turnover"] < sample["gmv_turnover"] and fac["gmv_turnover"] < sample["gmv_turnover"], "and moves more from one date to the next"
    assert sample["worst_condition"] is None or sample["worst_condition"] > 2.0 * lw["worst_condition"], "shrinkage conditions the matrix"
    assert sample["repair_rate"] is not None and 0.0 <= sample["repair_rate"] <= 1.0 and out["horizon"] == 21 and "mean squared daily return" in out["units"]


def test_an_estimator_with_ample_data_forecasts_the_risk_of_an_equal_weight_book_without_bias():
    few = factor_returns(n_assets=6, t=1200, k=2, seed=11)
    dates = few.index[300:1100:21]
    out = rm.compare(few, dates, [rm.CovSpec("sample", window=252)], horizon=21)["estimators"]["sample/w252"]
    assert 0.85 < out["risk_ratio_forecast_to_realised"] < 1.15 and out["qlike"] < 0.35 and out["risk_correlation"] is not None
    with_book = rm.compare(few, dates, [rm.CovSpec("sample", window=252)], weights=pd.DataFrame(0.0, index=few.index, columns=few.columns).assign(a00=1.0), horizon=21)["estimators"]["sample/w252"]
    assert with_book["windows"] == out["windows"] and with_book["risk_ratio_forecast_to_realised"] != out["risk_ratio_forecast_to_realised"], "a supplied book is the one whose risk is forecast"
    flat = rm.compare(few, dates[:3], [rm.CovSpec("sample", window=252)], weights=pd.DataFrame(0.0, index=few.index, columns=few.columns))["estimators"]["sample/w252"]
    assert flat["windows"] == 0 and flat["failed"] == 3, "a flat book has no risk to forecast: counted, not hidden"
