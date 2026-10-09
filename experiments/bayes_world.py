"""Hierarchical Bayesian expected returns, Bayesian decisions and Gaussian process regression on simulated worlds with a known truth, and what they find.

    python -m experiments.bayes_world          # about two minutes
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from src.equity import alpha_model as am
from src.equity.synthetic import simulate_fundamental_world, simulate_nonlinear_world
from src.framework import MODELS, bundle_from_prices, load_library
from src.models.gaussian_process import GaussianProcess, Stationary, make_kernel
from src.portfolio import decision_theory as dt
from src.portfolio import hierarchical_bayes as hb
from src.portfolio.mean_variance import mean_variance_weights
from src.utils.dates import rebalance_dates


def _world(seed, tau_g, tau_a, n_groups=4, per=5, T=60, vol=0.05, m=0.08):
    rng = np.random.default_rng(seed)
    n = n_groups * per
    groups = {f"A{i}": f"G{i // per}" for i in range(n)}
    mu = np.repeat(m + tau_g * rng.normal(size=n_groups), per) + tau_a * rng.normal(size=n)
    F = rng.normal(size=(n, 2))
    S = vol ** 2 * (0.3 * F @ F.T / 2 + np.eye(n))
    return mu, rng.multivariate_normal(mu, S / T), S, groups, T


def hierarchical_table(worlds: int = 60) -> pd.DataFrame:
    """Mean squared error of the estimates of 20 expected returns from 60 periods, and the true expected utility of the long-only mean-variance portfolio built on each estimate."""
    rows = {}
    for label, tg, ta in (("groups differ (tau_g 0.06, tau_a 0.01)", 0.06, 0.01), ("groups alike (tau_g 0, tau_a 0.01)", 0.0, 0.01), ("everything alike (tau_g 0, tau_a 0)", 0.0, 0.0)):
        mse = {"sample mean": [], "one common mean (Bayes-Stein style)": [], "hierarchical": []}
        util = {k: [] for k in mse}
        for seed in range(worlds):
            mu, x, S, groups, T = _world(seed, tg, ta)
            assets = list(groups)
            est = {"sample mean": x, "one common mean (Bayes-Stein style)": hb.hierarchical_posterior(x, S, T, {a: "ALL" for a in assets}, assets, 5).mean, "hierarchical": hb.hierarchical_posterior(x, S, T, groups, assets, 5).mean}
            for k, e in est.items():
                mse[k].append(np.mean((e - mu) ** 2))
                from src.portfolio.constraints import Constraints
                w = mean_variance_weights(pd.Series(e, index=assets), pd.DataFrame(S, index=assets, columns=assets), 5.0 / 12.0, Constraints(max_weight=0.3)).weights.reindex(assets).to_numpy()
                util[k].append(float(w @ mu - 0.5 * (5.0 / 12.0) * w @ S @ w))
        for k in mse:
            rows[(label, k)] = {"MSE of the means (x1e5)": 1e5 * np.mean(mse[k]), "true utility (bp a year)": 1e4 * np.mean(util[k]), "worlds": worlds}
    out = pd.DataFrame(rows).T
    out.index.names = ["world", "estimate"]
    return out


def decision_table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    mu, sig = 0.06 / 12, 0.16 / np.sqrt(12)
    R = rng.normal(mu, sig, (300000, 1))
    rows = {}
    for gamma in (2.0, 5.0, 10.0):
        w = dt.bayes_weights(R, dt.crra(gamma), lb=0.0, ub=4.0, budget=None, w0=[0.3])[0]
        rows[f"relative risk aversion {gamma:g}"] = {"expected-utility weight": w, "Merton mu/(gamma sigma^2)": mu / (gamma * sig ** 2)}
    return pd.DataFrame(rows).T


def uncertainty_table() -> pd.DataFrame:
    """Two assets, the expected return of the first uncertain (sd 1% a month against its mean of 1%): the Bayes decision against the plug-in decision, with the value of perfect information."""
    rng = np.random.default_rng(1)
    mus = rng.normal([0.01, 0.006], [0.01, 0.001], (300, 2))
    covs = np.array([np.diag([0.05 ** 2, 0.03 ** 2])] * 300)
    R = dt.predictive_draws(mus, covs, 60, 3)
    rows = {}
    for name, u in (("power utility, risk aversion 5", dt.crra(5.0)), ("loss-averse (shortfall x3)", dt.shortfall(0.0, 3.0)), ("mean - 0.95 CVaR", dt.cvar(0.95, 1.0))):
        bayes = dt.bayes_weights(R, u)
        plug = dt.bayes_weights(rng.multivariate_normal(mus.mean(axis=0), covs[0], 20000), u)
        info = dt.value_of_information(mus, covs, u, per_parameter=200, limit=20)
        rows[name] = {"weight on asset 1, Bayes": bayes[0], "weight on asset 1, plug-in": plug[0], "expected utility, Bayes": dt.expected_utility(bayes, R, u), "expected utility, plug-in": dt.expected_utility(plug, R, u), "value of perfect information": info["evpi"]}
    return pd.DataFrame(rows).T


def gp_table() -> pd.DataFrame:
    rows = {}
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 3))
    y = np.sin(1.5 * X[:, 0]) + 0.1 * rng.normal(size=150)
    gp = GaussianProcess(make_kernel("rbf", 3), noise=0.5, restarts=2).fit(X, y)
    ls = gp.length_scales
    Xt = rng.normal(size=(300, 3))
    rows["learning which input matters"] = {"length scale, input 1 (matters)": ls[0], "length scale, input 2": ls[1], "length scale, input 3": ls[2], "noise variance (planted 0.01)": gp.noise * gp.y_scale ** 2,
                                            "test RMSE": float(np.sqrt(np.mean((gp.predict(Xt) - np.sin(1.5 * Xt[:, 0])) ** 2)))}
    grid = rng.uniform(-3, 3, (400, 1))
    f = np.linalg.cholesky(Stationary("rbf", 0.7, 1.0)(grid) + 0.04 * np.eye(400)) @ rng.normal(size=400)
    g = GaussianProcess(Stationary("rbf", 0.7, 1.0), noise=0.04, normalize_y=False, optimize=False).fit(grid[:200], f[:200])
    m, s = g.predict(grid[200:], return_std=True, include_noise=True)
    rows["coverage of 95% intervals"] = {"share of new observations inside": float(np.mean(np.abs(f[200:] - m) <= 1.96 * s))}
    X2 = rng.normal(size=(300, 2))
    f2 = lambda Z: Z[:, 0] ** 2 + Z[:, 0] * Z[:, 1]
    y2 = f2(X2) + 0.3 * rng.normal(size=300)
    X2t = rng.normal(size=(500, 2))
    r2 = lambda p: 1 - ((f2(X2t) - p) ** 2).sum() / ((f2(X2t) - f2(X2t).mean()) ** 2).sum()
    rows["a square and a product (out-of-sample R2)"] = {"GP rbf+linear": r2(GaussianProcess(make_kernel("rbf+linear", 2)).fit(X2, y2).predict(X2t)), "GP linear only": r2(GaussianProcess(make_kernel("linear", 2)).fit(X2, y2).predict(X2t))}
    return pd.DataFrame(rows).T


def factor_model_table() -> pd.DataFrame:
    """The Gaussian process model against ridge and a forest on a world with a straight-line effect and on one with a U shape, 60 stocks for 14 years."""
    load_library()
    worlds = {"straight-line momentum and reversal": ("fund", None),
              "U-shaped momentum plus a straight-line reversal": ("nonlinear", lambda zm, zr: 0.012 * (np.minimum(zm ** 2, 4.0) - 1.0) + 0.004 * zr)}
    rows = {}
    for label, (kind, effect) in worlds.items():
        if kind == "fund":
            w = simulate_fundamental_world(n_assets=60, n_years=12, seed=1, premia={"value": 0.0, "quality": 0.0, "investment": 0.0, "momentum": 0.008, "reversal": 0.006, "revision": 0.0})
            bundle = bundle_from_prices(w.prices, min_history=60, name="gp")
        else:
            idx, prices = simulate_nonlinear_world(0, years=14, n=60, effect=effect)
            bundle = bundle_from_prices(pd.DataFrame(prices, index=idx, columns=[f"S{i:02d}" for i in range(prices.shape[1])]), min_history=60, name="gp")
        grid = rebalance_dates(bundle.index, "monthly")
        px = bundle.prices.loc[grid]
        fwd = px.shift(-1) / px - 1.0
        for name, model in (("ridge", MODELS.create("ml_factor_model", kind="ridge")), ("random forest", MODELS.create("ml_factor_model", kind="forest")),
                            ("Gaussian process, rbf + linear", MODELS.create("gp_factor_model", kernel="rbf+linear")), ("Gaussian process, linear", MODELS.create("gp_factor_model", kernel="linear"))):
            t0 = time.time()
            ic = am.information_coefficients({"x": model.score(bundle).loc[grid]}, fwd, "spearman", 15)["x"].iloc[48:]
            rows[(label, name)] = {"mean IC": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(len(ic)), "seconds": time.time() - t0}
    out = pd.DataFrame(rows).T
    out.index.names = ["world", "model"]
    return out


def run() -> dict:
    warnings.simplefilter("ignore")
    return {"hierarchical": hierarchical_table(), "merton": decision_table(), "uncertainty": uncertainty_table(), "gp": gp_table(), "gp_factor_model": factor_model_table()}


def main() -> None:
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 220
    t0 = time.time()
    for name, table in run().items():
        print(f"== {name} ==")
        print(table.to_string(), "\n")
    print(f"{time.time() - t0:.0f} seconds")


if __name__ == "__main__":
    main()
