"""Gaussian process regression: the same answer as scikit-learn for the same kernel, analytic gradients that match numerical ones, hyperparameters that recover what was planted, honest uncertainty, and
the factor model built on it."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, Matern, WhiteKernel

from src.equity import alpha_model as am
from src.equity.synthetic import simulate_fundamental_world
from src.framework import MODELS, bundle_from_prices, load_library
from src.models.gaussian_process import GaussianProcess, Linear, Stationary, make_kernel
from src.utils.dates import rebalance_dates

load_library()


def data(n=60, d=3, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1] + noise * rng.normal(size=n)
    return X, y, rng.normal(size=(15, d))


CASES = [
    (lambda: Stationary("rbf", [0.8, 1.5, 3.0], 1.3, ard=True), lambda: ConstantKernel(1.3) * RBF([0.8, 1.5, 3.0])),
    (lambda: Stationary("matern32", [0.8, 1.5, 3.0], 0.7, ard=True), lambda: ConstantKernel(0.7) * Matern([0.8, 1.5, 3.0], nu=1.5)),
    (lambda: Stationary("matern52", 1.2, 2.0), lambda: ConstantKernel(2.0) * Matern(1.2, nu=2.5)),
    (lambda: Stationary("rbf", [1.0, 2.0, 2.0], 1.0, ard=True) + Linear(0.4), lambda: ConstantKernel(1.0) * RBF([1.0, 2.0, 2.0]) + ConstantKernel(0.4) * DotProduct(sigma_0=0.0)),
]


@pytest.mark.parametrize("ours,theirs", CASES, ids=["rbf-ard", "matern32-ard", "matern52", "rbf+linear"])
def test_predictions_and_likelihood_match_scikit_learn(ours, theirs):
    X, y, Xs = data()
    gp = GaussianProcess(ours(), noise=0.05, normalize_y=False, optimize=False).fit(X, y)
    sk = GaussianProcessRegressor(theirs() + WhiteKernel(0.05), optimizer=None, normalize_y=False).fit(X, y)
    mean, std = gp.predict(Xs, return_std=True)
    sk_mean, sk_std = sk.predict(Xs, return_std=True)
    assert np.allclose(mean, sk_mean, atol=1e-6) and np.allclose(np.sqrt(std ** 2 + 0.05), sk_std, atol=1e-6)         # scikit-learn's spread includes the noise
    assert gp.log_marginal_likelihood() == pytest.approx(sk.log_marginal_likelihood_value_, abs=1e-4)
    m2, cov = gp.predict(Xs, return_cov=True)
    assert np.allclose(np.sqrt(np.diag(cov)), std, atol=1e-8) and np.allclose(cov, cov.T, atol=1e-10)


@pytest.mark.parametrize("ours,_", CASES, ids=["rbf-ard", "matern32-ard", "matern52", "rbf+linear"])
def test_analytic_gradients_of_the_marginal_likelihood_match_numerical_ones(ours, _):
    X, y, _xs = data(n=40)
    gp = GaussianProcess(ours(), noise=0.05, normalize_y=False, optimize=False)
    params = np.concatenate([gp.kernel.theta, [np.log(0.05)]])
    _, grad = gp._objective(params, X, y)
    num = []
    for i in range(len(params)):
        e = np.zeros_like(params)
        e[i] = 1e-6
        num.append((gp._objective(params + e, X, y)[0] - gp._objective(params - e, X, y)[0]) / 2e-6)
    assert np.allclose(grad, num, atol=1e-5, rtol=1e-4)


def test_optimisation_improves_the_likelihood_and_finds_the_relevant_inputs_and_the_noise():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 3))
    y = np.sin(1.5 * X[:, 0]) + 0.1 * rng.normal(size=150)                                                 # only the first input matters
    start = GaussianProcess(make_kernel("rbf", 3), noise=0.5, optimize=False).fit(X, y)
    gp = GaussianProcess(make_kernel("rbf", 3), noise=0.5, optimize=True, restarts=2).fit(X, y)
    assert gp.log_marginal_likelihood() > start.log_marginal_likelihood() + 5
    ls = gp.length_scales
    assert ls[1] > 5 * ls[0] and ls[2] > 5 * ls[0]                                                         # automatic relevance determination: the irrelevant inputs get long length scales
    assert gp.noise * gp.y_scale ** 2 == pytest.approx(0.01, abs=0.01)                                     # the noise variance (planted 0.01)
    Xt = rng.normal(size=(300, 3))
    assert np.sqrt(np.mean((gp.predict(Xt) - np.sin(1.5 * Xt[:, 0])) ** 2)) < 0.12


def test_the_spread_is_honest_calibrated_wider_far_from_the_data_and_collapsing_on_noiseless_points():
    rng = np.random.default_rng(2)
    kernel = Stationary("rbf", 0.7, 1.0)
    grid = rng.uniform(-3, 3, (400, 1))
    K = kernel(grid) + 0.04 * np.eye(400)
    f = np.linalg.cholesky(K) @ rng.normal(size=400)                                                      # a draw from the prior plus noise
    train, test = np.arange(200), np.arange(200, 400)
    gp = GaussianProcess(Stationary("rbf", 0.7, 1.0), noise=0.04, normalize_y=False, optimize=False).fit(grid[train], f[train])
    mean, std = gp.predict(grid[test], return_std=True, include_noise=True)
    coverage = np.mean(np.abs(f[test] - mean) <= 1.96 * std)
    assert 0.90 <= coverage <= 0.99                                                                       # about 95% of new observations fall inside the 95% interval
    far = gp.predict(np.array([[50.0]]), return_std=True)[1][0]
    assert far == pytest.approx(1.0, abs=1e-3)                                                            # no data nearby: the prior's own spread
    pts = np.linspace(-3, 3, 12)[:, None]
    exact = GaussianProcess(Stationary("rbf", 1.0, 1.0), noise=1e-6, normalize_y=False, optimize=False).fit(pts, np.sin(pts[:, 0]))
    assert np.allclose(exact.predict(pts), np.sin(pts[:, 0]), atol=1e-3) and exact.predict(pts, return_std=True)[1].max() < 1e-2      # with no noise it interpolates and is certain at the data
    draws = gp.sample_y(grid[test][:30], n=4000, seed=1)
    m, cov = gp.predict(grid[test][:30], return_cov=True)
    assert np.abs(draws.mean(axis=0) - m).max() < 0.08 and np.abs(np.cov(draws.T) - cov).max() < 0.08


def test_a_gaussian_process_learns_what_a_straight_line_cannot():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2))
    f = lambda Z: Z[:, 0] ** 2 + Z[:, 0] * Z[:, 1]
    y = f(X) + 0.3 * rng.normal(size=300)
    Xt = rng.normal(size=(500, 2))
    gp = GaussianProcess(make_kernel("rbf+linear", 2), optimize=True).fit(X, y)
    lin = GaussianProcess(make_kernel("linear", 2), optimize=True).fit(X, y)
    r2 = lambda p: 1 - ((f(Xt) - p) ** 2).sum() / ((f(Xt) - f(Xt).mean()) ** 2).sum()
    assert r2(gp.predict(Xt)) > 0.8 and r2(lin.predict(Xt)) < 0.1


def test_kernel_names_and_inputs_are_checked():
    for bad in ("", "cubic", "rbf+nope"):
        with pytest.raises(ValueError):
            make_kernel(bad, 2)
    for ctor in (lambda: Stationary("sinc"), lambda: Stationary("rbf", -1.0), lambda: Linear(0.0), lambda: GaussianProcess(noise=0.0)):
        with pytest.raises(ValueError):
            ctor()
    with pytest.raises(RuntimeError):
        GaussianProcess(make_kernel("rbf", 2)).predict(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        GaussianProcess(make_kernel("rbf", 2)).fit(np.zeros((3, 2)), np.zeros(2))


# ------------------------------------------------------------------------------------------------------------------ the factor model
@pytest.fixture(scope="module")
def momentum_world():
    w = simulate_fundamental_world(n_assets=40, n_years=10, seed=1, premia={"value": 0.0, "quality": 0.0, "investment": 0.0, "momentum": 0.008, "reversal": 0.006, "revision": 0.0})
    return bundle_from_prices(w.prices, min_history=60, name="gp-world")


def test_the_factor_model_finds_a_linear_effect_forecasts_with_a_spread_and_looks_only_back(momentum_world):
    model = MODELS.create("gp_factor_model", kernel="rbf+linear", max_points=300, optimize_every=24)
    grid = rebalance_dates(momentum_world.index, "monthly")
    px = momentum_world.prices.loc[grid]
    ic = am.information_coefficients({"x": model.score(momentum_world).loc[grid]}, px.shift(-1) / px - 1.0, "spearman", 15)["x"].iloc[36:]
    assert ic.mean() > 0.05 and ic.mean() / ic.std() * np.sqrt(len(ic)) > 2.5
    unc = model.uncertainty(momentum_world)
    vals = unc.to_numpy()
    assert np.isfinite(vals).any() and (vals[np.isfinite(vals)] > 0).all() and unc.shape == momentum_world.prices.shape
    cut = momentum_world.index[1700]
    a, b = model.score(momentum_world), model.score(momentum_world.perturbed_after(cut))
    assert np.allclose(a.loc[:cut].to_numpy(), b.loc[:cut].to_numpy(), equal_nan=True)


def test_the_factor_model_refuses_bad_settings():
    for bad in ({"kernel": "cubic"}, {"train_months": 3}, {"max_points": 10}, {"optimize_every": 0}, {"restarts": -1}, {"characteristics": "mom,beauty"}, {"min_months": 1}):
        with pytest.raises(ValueError):
            MODELS.create("gp_factor_model", **bad)
