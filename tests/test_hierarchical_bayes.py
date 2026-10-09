"""Hierarchical Bayesian expected returns and Bayesian decision theory: the algebra, recovery of planted structure, the Merton fraction from expected utility, and the two allocators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import ALLOCATORS, bundle_from_prices, load_library
from src.framework.allocation import Context
from src.portfolio import decision_theory as dt
from src.portfolio import hierarchical_bayes as hb
from src.utils.config import load_config

load_library()


def world(seed, n_groups=4, per=5, tau_g=0.06, tau_a=0.01, T=60, vol=0.05, m=0.08):
    rng = np.random.default_rng(seed)
    n = n_groups * per
    groups = {f"A{i}": f"G{i // per}" for i in range(n)}
    theta = m + tau_g * rng.normal(size=n_groups)
    mu = np.repeat(theta, per) + tau_a * rng.normal(size=n)
    F = rng.normal(size=(n, 2))
    S = vol ** 2 * (0.3 * F @ F.T / 2 + np.eye(n))
    x = rng.multivariate_normal(mu, S / T)
    return mu, x, S, groups, T


def test_the_posterior_mean_is_the_conjugate_formula_at_every_scale():
    mu, x, S, groups, T = world(0)
    post = hb.hierarchical_posterior(x, S, T, groups, list(groups), points=4)
    n = len(x)
    G = hb.membership(list(groups), groups)
    V = S / T
    k = 7
    ta, tg = post.grid[k]
    s0 = 10.0 * np.std(x) + 10.0 * np.sqrt(np.mean(np.diag(V)))
    Omega = ta ** 2 * np.eye(n) + tg ** 2 * G @ G.T + s0 ** 2 * np.ones((n, n))
    P = np.linalg.inv(Omega) + np.linalg.inv(V)
    direct = np.linalg.solve(P, np.linalg.inv(Omega) @ (x.mean() * np.ones(n)) + np.linalg.inv(V) @ x)
    assert np.allclose(post.means[k], direct, atol=1e-6) and np.allclose(post.covs[k], np.linalg.inv(P), atol=1e-8)
    assert post.weights.sum() == pytest.approx(1.0) and (post.weights >= 0).all()
    assert np.allclose(post.mean, post.weights @ post.means) and np.all(np.linalg.eigvalsh(post.covariance()) > -1e-12)


def test_the_hierarchical_means_beat_the_sample_and_the_flat_shrinker_when_groups_differ_and_match_it_when_they_do_not():
    errs = {"sample": [], "flat": [], "hier": []}
    for seed in range(40):
        mu, x, S, groups, T = world(seed)
        post = hb.hierarchical_posterior(x, S, T, groups, list(groups), points=5)
        flat = hb.hierarchical_posterior(x, S, T, {a: "ALL" for a in groups}, list(groups), points=5)             # one group: ordinary shrinkage to a common mean
        errs["sample"].append(np.mean((x - mu) ** 2))
        errs["flat"].append(np.mean((flat.mean - mu) ** 2))
        errs["hier"].append(np.mean((post.mean - mu) ** 2))
    assert np.mean(errs["hier"]) < 0.8 * np.mean(errs["flat"]) < np.mean(errs["sample"])                # groups that really differ: pool within them
    same = [world(s, tau_g=0.0) for s in range(40)]
    h = np.mean([np.mean((hb.hierarchical_posterior(x, S, T, g, list(g), points=5).mean - mu) ** 2) for mu, x, S, g, T in same])
    f = np.mean([np.mean((hb.hierarchical_posterior(x, S, T, {a: "ALL" for a in g}, list(g), points=5).mean - mu) ** 2) for mu, x, S, g, T in same])
    assert h < 1.15 * f                                                                                  # groups that do not differ: no worse than ordinary shrinkage


def test_the_scales_are_inferred_and_pooling_follows_them():
    def fit(seed, **kw):
        mu, x, S, g, T = world(seed, **kw)
        return hb.hierarchical_posterior(x, S, T, g, list(g), points=5)

    big = [fit(s, tau_g=0.08, tau_a=0.002) for s in range(10)]
    none = [fit(s, tau_g=0.0, tau_a=0.0) for s in range(10)]
    assert np.mean([p.tau_g for p in big]) > 2 * np.mean([p.tau_g for p in none])
    mu, x, S, groups, T = world(3, tau_a=0.0)
    post = hb.hierarchical_posterior(x, S, T, groups, list(groups), points=5)
    within = np.mean([post.mean[5 * g:5 * g + 5].std() for g in range(4)])
    assert within < np.mean([x[5 * g:5 * g + 5].std() for g in range(4)])                              # means in a group are pulled together
    draws = post.draw(4000, np.random.default_rng(1))
    assert np.abs(draws.mean(axis=0) - post.mean).max() < 0.01 and np.abs(np.cov(draws.T) - post.covariance()).max() < 5e-4
    with pytest.raises(ValueError):
        hb.hierarchical_posterior(x[:3], S, T, groups, list(groups)[:3])


# ------------------------------------------------------------------------------------------------------------------ decision theory
def test_expected_power_utility_gives_the_merton_fraction_and_cvar_cuts_the_tail():
    rng = np.random.default_rng(0)
    mu, sig = 0.06 / 12, 0.16 / np.sqrt(12)
    R = rng.normal(mu, sig, (300000, 1))
    for gamma in (2.0, 5.0):
        w = dt.bayes_weights(R, dt.crra(gamma), lb=0.0, ub=4.0, budget=None, w0=[0.3])[0]
        assert w == pytest.approx(mu / (gamma * sig ** 2), rel=0.06)
    fat = np.hstack([rng.normal(0.004, 0.03, (20000, 1)), rng.standard_t(3, (20000, 1)) * 0.02 + 0.006])
    w_mv = dt.bayes_weights(fat, dt.mean_variance(5.0))
    w_cv = dt.bayes_weights(fat, dt.cvar(0.95, 2.0))
    tail = lambda w: np.partition(fat @ w, 1000)[:1000].mean()
    assert tail(w_cv) > tail(w_mv) and w_cv[1] < w_mv[1]                                                  # caring about the tail means less of the fat-tailed asset


def test_parameter_uncertainty_changes_the_decision_and_perfect_information_has_a_value():
    rng = np.random.default_rng(1)
    mus = rng.normal([0.01, 0.006], [0.01, 0.001], (200, 2))
    covs = np.array([np.diag([0.05 ** 2, 0.03 ** 2])] * 200)
    R = dt.predictive_draws(mus, covs, 50, 3)
    u = dt.crra(5.0)
    bayes = dt.bayes_weights(R, u)
    plug = dt.bayes_weights(rng.multivariate_normal(mus.mean(axis=0), covs[0], 10000), u)
    assert dt.expected_utility(bayes, R, u) >= dt.expected_utility(plug, R, u) - 1e-12                    # the Bayes decision is best under the predictive distribution, by construction
    assert abs(bayes[0] - plug[0]) > 0.005
    info = dt.value_of_information(mus, covs, u, per_parameter=200, limit=15)
    assert info["evpi"] > 0 and dt.regret(bayes, mus, covs, u, per_parameter=200, limit=15)["regret"] >= 0
    for bad in (lambda: dt.crra(0.0), lambda: dt.cara(-1.0), lambda: dt.cvar(0.2)):
        with pytest.raises(ValueError):
            bad()


# ------------------------------------------------------------------------------------------------------------------ the allocators
@pytest.fixture(scope="module")
def env():
    rng = np.random.default_rng(5)
    T, n = 1900, 9
    idx = pd.bdate_range("2010-01-04", periods=T)
    drift = np.repeat([0.0004, 0.0001, 0.0002], 3) + 0.00005 * rng.normal(size=n)
    r = pd.DataFrame(drift + 0.01 * rng.normal(size=(T, n)), index=idx, columns=[f"A{i}" for i in range(n)])
    bundle = bundle_from_prices(100 * (1 + r).cumprod(), min_history=60, name="bayes", asset_class={f"A{i}": f"C{i // 3}" for i in range(n)})
    return {"bundle": bundle, "ctx": Context(bundle, load_config(), forecasts=None, models=[])}


def test_the_hierarchical_allocator_is_a_capped_long_only_book_that_looks_only_back(env):
    w = ALLOCATORS.create("hierarchical_bayes", lookback=504, points=5).build(env["ctx"])
    live = w.loc[w.abs().sum(axis=1) > 0]
    assert len(live) > 500 and np.allclose(live.sum(axis=1), 1.0, atol=1e-5) and (live >= -1e-8).all().all() and live.max().max() <= env["ctx"].constraints().max_weight + 1e-6
    cut = env["bundle"].index[1500]
    b = ALLOCATORS.create("hierarchical_bayes", lookback=504, points=5).build(Context(env["bundle"].perturbed_after(cut), env["ctx"].config, forecasts=None, models=[]))
    assert np.allclose(w.loc[:cut].to_numpy(), b.loc[:cut].to_numpy(), atol=1e-9)


def test_the_expected_utility_allocator_runs_for_each_utility_and_refuses_bad_settings(env):
    for utility in ("crra", "cvar"):
        w = ALLOCATORS.create("bayes_expected_utility", utility=utility, lookback=504, draws=40).build(env["ctx"])
        live = w.loc[w.abs().sum(axis=1) > 0]
        assert len(live) > 300 and np.allclose(live.sum(axis=1), 1.0, atol=1e-4) and (live >= -1e-6).all().all()
    for name, bad in (("hierarchical_bayes", {"lookback": 20}), ("hierarchical_bayes", {"points": 2}), ("bayes_expected_utility", {"utility": "log"}), ("bayes_expected_utility", {"draws": 2}), ("bayes_expected_utility", {"risk": 0})):
        with pytest.raises(ValueError):
            ALLOCATORS.create(name, **bad)
