"""Statistical, macroeconomic, cross-sectional and machine-learning factor models, each against a world with the factor structure planted."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity import alpha_model as am
from src.equity.synthetic import simulate_fundamental_world
from src.framework import MODELS, bundle_from_prices, load_library
from src.strategies import factor_models as fm
from src.utils.dates import rebalance_dates

load_library()


def monthly_ic(bundle, score, skip=36):
    grid = rebalance_dates(bundle.index, "monthly")
    px = bundle.prices.loc[grid]
    fwd = px.shift(-1) / px - 1.0
    ic = am.information_coefficients({"x": score.loc[grid]}, fwd, "spearman", 15)["x"].iloc[skip:]
    return ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic))


@pytest.fixture(scope="module")
def momentum_world():
    w = simulate_fundamental_world(n_assets=60, n_years=12, seed=1, premia={"value": 0.0, "quality": 0.0, "investment": 0.0, "momentum": 0.008, "reversal": 0.006, "revision": 0.0})
    return bundle_from_prices(w.prices, min_history=60, name="characteristic-world")


# ------------------------------------------------------------------------------------------------------------------ statistical APT
def factor_returns(seed, T=3000, n=40, k=3, alpha_size=0.0004, idio=0.01):
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 1.0, (n, k))
    f = 0.0004 + 0.008 * rng.normal(size=(T, k))
    alpha = alpha_size * rng.normal(size=n)
    r = alpha + f @ B.T + idio * rng.normal(size=(T, n))
    idx = pd.bdate_range("2005-01-03", periods=T)
    return pd.DataFrame(r, index=idx, columns=[f"S{i:02d}" for i in range(n)]), alpha, B, f


def test_the_pca_residual_alpha_recovers_the_mispricing_planted_beyond_the_factors():
    r, alpha, B, f = factor_returns(0)
    est, resid_sd = fm.pca_residual_scores(r.to_numpy(), 3)
    assert np.corrcoef(est, alpha)[0, 1] > 0.8 and np.abs(est - alpha).max() < 0.0007                  # the standard error of a mean over 3000 days of 1% noise is 0.00018, against alphas of 0.0004
    assert np.abs(resid_sd - 0.01).max() < 0.0012 and resid_sd.mean() < 0.01                            # a little low: fitting three components to 40 noisy series absorbs some of the noise
    wrong = fm.pca_residual_scores(r.to_numpy(), 1)[0]                                                 # leaving out two factors leaves their premia in the "alpha"
    assert np.corrcoef(wrong, alpha)[0, 1] < np.corrcoef(est, alpha)[0, 1]


def test_apt_alpha_ranks_assets_by_a_persistent_mispricing_and_not_by_factor_exposure():
    r, alpha, _, _ = factor_returns(1, T=2500, alpha_size=0.0006)
    bundle = bundle_from_prices(100 * (1 + r).cumprod(), min_history=60, name="apt-world")
    score = MODELS.create("apt_alpha", factors=3, window=504, refit=21).score(bundle)
    ic, t = monthly_ic(bundle, score, skip=24)
    assert ic > 0.1 and t > 4
    last = score.iloc[-1].dropna()
    assert len(last) == 40 and np.corrcoef(last, alpha)[0, 1] > 0.7
    market_only = MODELS.create("apt_alpha", factors=1).score(bundle)
    assert monthly_ic(bundle, market_only, skip=24)[1] < t                                            # one factor leaves the other two, and their premia, in the score: the mean IC can be higher, the IC is less steady
    assert score.iloc[:520].isna().all().all()                                                         # nothing until the window has filled


def test_apt_alpha_uses_nothing_after_the_decision_date_and_validates_its_inputs():
    r, _, _, _ = factor_returns(2, T=1200, n=12)
    bundle = bundle_from_prices(100 * (1 + r).cumprod(), min_history=60, name="x")
    noisy = bundle.perturbed_after(bundle.index[900])
    a, b = MODELS.create("apt_alpha", window=252).score(bundle), MODELS.create("apt_alpha", window=252).score(noisy)
    assert np.allclose(a.loc[:bundle.index[900]].to_numpy(), b.loc[:bundle.index[900]].to_numpy(), equal_nan=True)
    for bad in ({"factors": 0}, {"window": 50}, {"skip": -1}, {"refit": 0}):
        with pytest.raises(ValueError):
            MODELS.create("apt_alpha", **bad)


# ------------------------------------------------------------------------------------------------------------------ macroeconomic factors
def test_rolling_betas_equal_the_regression_on_each_window():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=400)
    F = pd.DataFrame(rng.normal(size=(400, 3)), index=idx, columns=list("abc"))
    B = rng.normal(size=(3, 5))
    R = pd.DataFrame(F.to_numpy() @ B + 0.5 * rng.normal(size=(400, 5)), index=idx, columns=list("vwxyz"))
    out = fm.rolling_betas(R, F, window=120, min_periods=60)
    assert np.isnan(out[:59]).all() and np.isfinite(out[60:]).all()
    for t in (80, 150, 399):
        lo = max(0, t - 119)
        X = np.column_stack([np.ones(t - lo + 1), F.iloc[lo:t + 1].to_numpy()])
        ref, *_ = np.linalg.lstsq(X, R.iloc[lo:t + 1].to_numpy(), rcond=None)
        assert np.abs(out[t] - ref[1:]).max() < 1e-9, t


def test_macro_changes_use_differences_for_rates_percent_changes_for_indices_and_are_causal():
    idx = pd.bdate_range("2010-01-01", periods=1500)
    rng = np.random.default_rng(1)
    macro = pd.DataFrame({"DGS10": 3 + np.cumsum(rng.normal(0, 0.02, 1500)), "VIX": 20 + np.cumsum(rng.normal(0, 0.3, 1500))}, index=idx)
    raw = fm.macro_changes(macro, ["DGS10", "VIX"], innovations=False, window=252)
    assert abs(raw["DGS10"].iloc[600] * macro["DGS10"].diff().rolling(252, min_periods=126).std().shift(1).iloc[600] - macro["DGS10"].diff().iloc[600]) < 1e-12
    assert abs(raw["VIX"].iloc[600] * macro["VIX"].pct_change().rolling(252, min_periods=126).std().shift(1).iloc[600] - macro["VIX"].pct_change().iloc[600]) < 1e-12
    ar = pd.DataFrame({"x": np.zeros(1500)}, index=idx)
    e = rng.normal(size=1500)
    for t in range(1, 1500):
        ar.iloc[t, 0] = ar.iloc[t - 1, 0] + 0.6 * (ar.iloc[t - 1, 0] - ar.iloc[t - 2, 0] if t > 1 else 0.0) + e[t]       # increments with a first-order autoregression of 0.6
    innov = fm.macro_changes(ar, ["x"], innovations=True, window=504)["x"].iloc[800:]
    plain = fm.macro_changes(ar, ["x"], innovations=False, window=504)["x"].iloc[800:]
    assert abs(innov.autocorr(1)) < 0.15 < abs(plain.autocorr(1))                                      # taking out the predictable part leaves no autocorrelation
    changed = macro.copy()
    changed.iloc[1000:] = 0.0
    assert np.allclose(fm.macro_changes(changed, ["DGS10", "VIX"], True, 252).iloc[:1000].to_numpy(), fm.macro_changes(macro, ["DGS10", "VIX"], True, 252).iloc[:1000].to_numpy(), equal_nan=True)


def macro_world(seed=0, T=3600, n=40, k=2, premium=0.0007):
    """Daily returns with loadings on k macro changes and a mean return proportional to the loadings (a priced risk); the macro series are levels whose changes are the factors."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-01-05", periods=T)
    f = rng.normal(size=(T, k))
    B = rng.normal(size=(n, k))
    mean = B @ np.full(k, premium)
    r = mean + 0.004 * f @ B.T + 0.012 * rng.normal(size=(T, n))
    macro = pd.DataFrame({"DGS10": 3 + np.cumsum(f[:, 0] * 0.03), "T10Y3M": 1 + np.cumsum(f[:, 1] * 0.03)}, index=idx)
    return bundle_from_prices(100 * (1 + pd.DataFrame(r, index=idx, columns=[f"S{i:02d}" for i in range(n)])).cumprod(), macro=macro, min_history=60, name="macro-world"), B


def test_the_macro_model_finds_the_priced_exposures_and_ranks_by_them():
    bundle, B = macro_world(0)
    score = MODELS.create("macro_factor_model", series="DGS10,T10Y3M", window=504, premium_window=60, min_months=24, innovations=False).score(bundle)
    ic, t = monthly_ic(bundle, score, skip=48)
    assert ic > 0.08 and t > 3
    expected = B @ np.array([1.0, 1.0])                                                                # the true ranking: exposure to the priced factors
    assert np.corrcoef(score.iloc[-1].dropna(), expected)[0, 1] > 0.6
    flat, _ = macro_world(1, premium=0.0)
    null = MODELS.create("macro_factor_model", series="DGS10,T10Y3M", innovations=False).score(flat)
    assert abs(monthly_ic(flat, null, skip=48)[1]) < 3                                                 # no premium planted: no skill claimed


def test_the_macro_model_says_what_it_needs_and_uses_only_the_past():
    bundle, _ = macro_world(2, T=2600, n=20)
    with pytest.raises(KeyError, match="macro series"):
        MODELS.create("macro_factor_model", series="NOPE,NADA").score(bundle)
    with pytest.raises(KeyError, match="macro series"):
        MODELS.create("macro_factor_model", series="DGS10").score(bundle)                              # one series is not a factor model
    noisy = bundle.perturbed_after(bundle.index[2000])
    kw = dict(series="DGS10,T10Y3M", window=252, premium_window=24, min_months=12)
    a, b = MODELS.create("macro_factor_model", **kw).score(bundle), MODELS.create("macro_factor_model", **kw).score(noisy)
    assert np.allclose(a.loc[:bundle.index[2000]].to_numpy(), b.loc[:bundle.index[2000]].to_numpy(), equal_nan=True)
    for bad in ({"window": 50}, {"premium_window": 3}, {"min_months": 1}):
        with pytest.raises(ValueError):
            MODELS.create("macro_factor_model", **bad)


# ------------------------------------------------------------------------------------------------------------------ characteristics
def test_the_price_characteristics_are_what_they_say_and_oriented_so_more_is_better(momentum_world):
    c = fm.price_characteristics(momentum_world)
    px, r = momentum_world.prices, momentum_world.returns
    t = 800
    assert set(c) == set(fm.CHARACTERISTICS)
    assert c["mom"].iloc[t].tolist() == pytest.approx((px.iloc[t - 21] / px.iloc[t - 252] - 1).tolist())
    assert c["rev"].iloc[t].tolist() == pytest.approx((-(px.iloc[t] / px.iloc[t - 21] - 1)).tolist())
    assert c["lowvol"].iloc[t].tolist() == pytest.approx((-r.iloc[t - 59:t + 1].std()).tolist())
    assert c["nomax"].iloc[t].tolist() == pytest.approx((-r.iloc[t - 20:t + 1].max()).tolist())
    assert c["high"].iloc[t].tolist() == pytest.approx((px.iloc[t] / px.iloc[t - 251:t + 1].max() - 1).tolist())
    market = r.mean(axis=1)
    beta = r.iloc[t - 251:t + 1].apply(lambda col: col.cov(market.iloc[t - 251:t + 1]) / market.iloc[t - 251:t + 1].var())
    assert c["lowbeta"].iloc[t].tolist() == pytest.approx((-beta).tolist())


@pytest.mark.parametrize("kind", ["ols", "ridge", "lasso", "enet", "pcr", "pls"])
def test_the_linear_learners_find_the_planted_momentum_and_reversal(momentum_world, kind):
    ic, t = monthly_ic(momentum_world, MODELS.create("ml_factor_model", kind=kind).score(momentum_world))
    assert ic > 0.08 and t > 4, (kind, ic, t)


def test_the_cross_sectional_regression_model_finds_them_too_and_both_are_causal(momentum_world):
    model = MODELS.create("characteristic_regression")
    ic, t = monthly_ic(momentum_world, model.score(momentum_world))
    assert ic > 0.08 and t > 4
    cut = momentum_world.index[1900]
    noisy = momentum_world.perturbed_after(cut)
    for name in ("characteristic_regression", "ml_factor_model"):
        a, b = MODELS.create(name).score(momentum_world), MODELS.create(name).score(noisy)
        assert np.allclose(a.loc[:cut].to_numpy(), b.loc[:cut].to_numpy(), equal_nan=True), name


def test_trees_and_the_network_are_slower_to_refit_by_default_and_the_choice_is_honoured(momentum_world, monkeypatch):
    assert MODELS.create("ml_factor_model", kind="ols").refit_every == 1 and MODELS.create("ml_factor_model", kind="forest").refit_every == 3
    assert MODELS.create("ml_factor_model", kind="forest", refit_every=6).refit_every == 6
    calls = []
    real = fm.fit_learner
    monkeypatch.setattr(fm, "fit_learner", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    short = momentum_world.prices.iloc[:1500]
    bundle = bundle_from_prices(short, min_history=60, name="short")
    MODELS.create("ml_factor_model", kind="ols", min_months=12, train_months=24).score(bundle)
    monthly = len(calls)
    calls.clear()
    MODELS.create("ml_factor_model", kind="ols", min_months=12, train_months=24, refit_every=4).score(bundle)
    assert abs(len(calls) * 4 - monthly) <= 8 and len(calls) < monthly / 2                                # refitting every fourth month is about a quarter of the fits


# ------------------------------------------------------------------------------------------------------------------ the learners themselves
def design(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    return X, rng


def test_the_lasso_zeroes_the_features_that_do_not_matter_where_ridge_keeps_them_all():
    X, rng = design()
    y = 1.0 * X[:, 0] - 0.8 * X[:, 1] + rng.normal(size=len(X))
    _, lasso = fm.fit_learner("lasso", X, y, alpha=0.1)
    _, ridge = fm.fit_learner("ridge", X, y, alpha=1.0)
    _, enet = fm.fit_learner("enet", X, y, alpha=0.1)
    assert lasso["selected"] == 2 and np.abs(lasso["coef"][2:]).max() == 0.0 and lasso["coef"][0] > 0.8 > 0 > lasso["coef"][1]
    assert ridge["selected"] == 6 and np.abs(ridge["coef"][2:]).max() < 0.1
    assert 2 <= enet["selected"] <= 6
    _, ols = fm.fit_learner("ols", X, y)
    assert np.abs(ols["coef"][:2] - [1.0, -0.8]).max() < 0.06


def test_pls_keeps_the_direction_that_matters_and_pcr_the_direction_that_varies_most():
    X, rng = design(seed=1)
    target = np.array([1.0, 1.0, 0, 0, 0, 0]) / np.sqrt(2)
    y = X @ target * np.sqrt(2) + 0.5 * rng.normal(size=len(X))
    align = lambda info: abs(info["coef"] @ target / np.linalg.norm(info["coef"]))
    pls1, pcr1 = fm.fit_learner("pls", X, y, components=1)[1], fm.fit_learner("pcr", X, y, components=1)[1]
    assert pls1["components"] == 1 and pcr1["components"] == 1
    assert align(pls1) > 0.95 and align(pcr1) < align(pls1) - 0.3                                     # features of equal variance: the first principal component is arbitrary, the PLS component is the useful one
    common = X.copy()
    shared = 2.0 * rng.normal(size=len(X))
    common[:, 0] += shared
    common[:, 1] += shared                                                                            # the first two features share a large common part, so the direction (1, 1) is the one that varies most
    y2 = 0.3 * (common[:, 0] + common[:, 1]) + 0.5 * rng.normal(size=len(X))
    assert align(fm.fit_learner("pcr", common, y2, components=1)[1]) > 0.9
    pcr_many = fm.fit_learner("pcr", X, y, components=6)[1]["coef"]
    ols = fm.fit_learner("ols", X, y)[1]["coef"]
    assert np.abs(pcr_many - ols).max() < 1e-8                                                           # with every component kept PCR is OLS


def test_trees_and_the_network_find_a_u_shape_and_an_interaction_that_no_linear_learner_can():
    X, rng = design(seed=2)
    Xt, _ = design(2000, seed=3)
    r2 = lambda yt, p: 1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    for name, target in (("u-shape", lambda Z: Z[:, 0] ** 2 - 1.0), ("interaction", lambda Z: Z[:, 0] ** 2 - 1.0 + Z[:, 0] * Z[:, 1])):
        y = target(X) + 0.3 * rng.normal(size=len(X))
        yt = target(Xt)
        scores = {k: r2(yt, fm.fit_learner(k, X, y, depth=6, trees=60)[0](Xt)) for k in ("ols", "lasso", "cart", "forest", "mlp")}
        assert scores["ols"] < 0.02 and scores["lasso"] < 0.02, (name, scores)                        # a straight line sees nothing in a square
        assert scores["forest"] > 0.4 and scores["cart"] > 0.3 and scores["mlp"] > 0.3, (name, scores)
    importances = fm.fit_learner("forest", X, y, depth=6, trees=60)[1]["importances"]
    assert importances[:2].sum() > 0.8 and importances.argmax() == 0


def test_unknown_learners_and_characteristics_are_refused(momentum_world):
    with pytest.raises(ValueError, match="kind"):
        fm.fit_learner("svm", np.zeros((5, 2)), np.zeros(5))
    for bad in ({"kind": "svm"}, {"alpha": -1.0}, {"components": 0}, {"depth": 0}, {"trees": 2}, {"train_months": 3}, {"min_months": 1}, {"refit_every": -1}, {"characteristics": "mom,beauty"}, {"characteristics": ""}):
        with pytest.raises(ValueError):
            MODELS.create("ml_factor_model", **bad)
    for bad in ({"window": 3}, {"min_obs": 1}, {"characteristics": "nope"}):
        with pytest.raises(ValueError):
            MODELS.create("characteristic_regression", **bad)
