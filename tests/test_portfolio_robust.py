"""Wasserstein-robust CVaR, neighbourhood caps on a graph of the assets, and Black-Litterman with factor views: each against an explicit construction, a brute-force search, or the property that defines it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio import augmented_bl as abl
from src.portfolio import graph_constrained as gc
from src.portfolio import robust_wasserstein as rw
from src.portfolio.black_litterman import black_litterman_posterior
from src.portfolio.cvar_optimize import mean_cvar_weights


@pytest.fixture(scope="module")
def scenarios():
    rng = np.random.default_rng(3)
    T = 400
    common = rng.standard_normal((T, 1)) * 0.008
    R = common + rng.standard_normal((T, 4)) * np.array([0.004, 0.008, 0.012, 0.02]) + np.array([0.0004, 0.0003, 0.0006, 0.0009])
    return pd.DataFrame(R, columns=list("ABCD"))


def test_the_worst_case_is_attained_by_moving_the_worst_scenarios_against_the_portfolio(scenarios):
    """The bound CVaR + eps ||w|| / beta is not just an upper bound: move the worst beta share of scenarios by eps/beta against the portfolio (total transport cost eps) and the CVaR rises by exactly that."""
    w = np.array([0.4, 0.3, 0.2, 0.1])
    R, alpha, eps = scenarios.to_numpy(), 0.95, 0.002
    beta = 1 - alpha
    k = int(round(beta * len(R)))
    assert k == 20
    base = rw.empirical_cvar(-(R @ w), beta)
    worst = np.argsort(-(-(R @ w)))[:k]                                           # the k largest losses
    moved = R.copy()
    moved[worst] -= (eps / beta) * w / np.linalg.norm(w)
    cost = (k / len(R)) * (eps / beta)                                           # mass moved x distance moved
    assert cost == pytest.approx(eps)
    assert rw.empirical_cvar(-(moved @ w), beta) == pytest.approx(base + eps * np.linalg.norm(w) / beta, rel=1e-9)
    assert rw.worst_case_cvar(w, R, eps, alpha) == pytest.approx(base + eps * np.linalg.norm(w) / beta)
    rng = np.random.default_rng(0)                                               # and no other way of spending the same budget does better
    for _ in range(50):
        d = rng.standard_normal(R.shape)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        share = rng.dirichlet(np.ones(len(R)) * 0.2)                              # how the cost eps is spread over scenarios: scenario i moves by eps * share_i * T
        trial = R - (eps * share * len(R))[:, None] * d
        assert rw.empirical_cvar(-(trial @ w), beta) <= rw.worst_case_cvar(w, R, eps, alpha) + 1e-12


def test_with_no_ball_it_is_the_ordinary_minimum_cvar_portfolio(scenarios):
    r = rw.wasserstein_cvar_weights(scenarios, eps=0.0)
    ref = mean_cvar_weights(scenarios, alpha=0.95)
    assert r["ok"] and r["weights"].sum() == pytest.approx(1.0) and (r["weights"] >= -1e-9).all()
    assert r["empirical_cvar"] == pytest.approx(rw.empirical_cvar(-(scenarios.to_numpy() @ ref.to_numpy()), 0.05), rel=1e-6), "the same minimum CVaR as the existing linear programme"


def test_the_solver_finds_the_best_portfolio_on_a_grid_and_a_larger_ball_spreads_the_weights(scenarios):
    two = scenarios[["A", "D"]]
    eps = 0.003
    res = rw.wasserstein_cvar_weights(two, eps=eps)
    grid = np.linspace(0, 1, 2001)
    vals = [rw.worst_case_cvar(np.array([x, 1 - x]), two.to_numpy(), eps) for x in grid]
    assert res["worst_case_cvar"] <= min(vals) + 1e-7 and res["gap"] < 1e-6 and res["ok"]
    assert res["weights"]["A"] == pytest.approx(grid[int(np.argmin(vals))], abs=2e-3)
    norms = [np.linalg.norm(rw.wasserstein_cvar_weights(scenarios, eps=e)["weights"]) for e in (0.0, 0.001, 0.003, 0.01, 0.05)]
    assert all(b <= a + 1e-9 for a, b in zip(norms, norms[1:])), norms
    assert norms[-1] == pytest.approx(0.5, abs=0.02), "a large enough ball gives equal weights, whose norm is 1/sqrt(4)"


def test_the_wasserstein_solver_refuses_what_it_cannot_do(scenarios):
    for kw, word in (({"alpha": 1.2}, "alpha"), ({"eps": -1}, "eps"), ({"ub": 0.1}, "no fully invested")):
        with pytest.raises(ValueError, match=word):
            rw.wasserstein_cvar_weights(scenarios, **kw)
    with pytest.raises(ValueError, match="at least 50"):
        rw.wasserstein_cvar_weights(scenarios.iloc[:20])


# ------------------------------------------------------------------------------------------------------------------ graph neighbourhood caps
def chain_cov(n=6, rho=0.6):
    idx = np.arange(n)
    corr = rho ** np.abs(idx[:, None] - idx[None, :])
    vol = np.linspace(0.15, 0.25, n)
    cols = [f"X{i}" for i in range(n)]
    return pd.DataFrame(corr * np.outer(vol, vol), index=cols, columns=cols)


def test_every_neighbourhood_stays_under_the_cap_and_the_cost_in_variance_is_reported():
    cov = chain_cov()
    r = gc.graph_capped_weights(cov, "mst", cap=0.5, hops=1)
    assert r["weights"].sum() == pytest.approx(1.0) and (r["weights"] >= -1e-9).all()
    assert r["neighbourhood_weight"].max() <= 0.5 + 1e-6 and r["binding"], "the cap is a real constraint here"
    assert r["variance_cost"] >= 1.0 - 1e-9 and r["unconstrained"].sum() == pytest.approx(1.0)
    assert len(r["edges"]) == 5, "the minimum spanning tree of six assets has five edges"
    for v, members in enumerate(gc.neighbourhoods(gc.mst(np.corrcoef(np.random.default_rng(0).standard_normal((50, 6)), rowvar=False)), 1)):
        assert v in members


def test_a_loose_cap_changes_nothing_and_wider_neighbourhoods_bind_harder():
    cov = chain_cov(10)
    loose = gc.graph_capped_weights(cov, "mst", cap=1.0)
    assert np.allclose(loose["weights"], loose["unconstrained"], atol=1e-6) and loose["binding"] == [] and loose["variance_cost"] == pytest.approx(1.0, abs=1e-6)
    one, two = gc.graph_capped_weights(cov, "mst", cap=0.7, hops=1), gc.graph_capped_weights(cov, "mst", cap=0.7, hops=2)
    assert two["variance_cost"] >= one["variance_cost"] - 1e-9
    t = gc.graph_capped_weights(cov, "tmfg", cap=0.8, hops=1)
    assert t["neighbourhood_weight"].max() <= 0.8 + 1e-6 and len(t["edges"]) == 3 * (10 - 2)
    m = gc.graph_capped_weights(cov, "mst", cap=0.6, mu=pd.Series(np.linspace(0.0, 0.1, 10), index=cov.index), risk_aversion=3.0)
    assert m["weights"].sum() == pytest.approx(1.0) and m["neighbourhood_weight"].max() <= 0.6 + 1e-6


def test_a_hub_or_an_impossible_cap_is_refused_with_the_reason():
    n = 5
    corr = np.full((n, n), 0.1)
    corr[0, :] = corr[:, 0] = 0.6
    np.fill_diagonal(corr, 1.0)
    star = pd.DataFrame(corr * 0.04, index=list("HABCD"), columns=list("HABCD"))
    with pytest.raises(ValueError, match="joined to every other"):
        gc.graph_capped_weights(star, "mst", cap=0.5)
    with pytest.raises(ValueError, match="no portfolio meets"):
        gc.graph_capped_weights(chain_cov(), "mst", cap=0.1)
    for kw, word in (({"graph": "knn"}, "graph must"), ({"cap": 0.0}, "cap must")):
        with pytest.raises(ValueError, match=word):
            gc.graph_capped_weights(chain_cov(), **{"graph": "mst", "cap": 0.5, **kw})
    with pytest.raises(ValueError, match="at least 4"):
        gc.graph_capped_weights(chain_cov(3), "tmfg", cap=0.9)


# ------------------------------------------------------------------------------------------------------------------ Black-Litterman with factor views
@pytest.fixture(scope="module")
def factor_world():
    assets = list("ABCDE")
    B = pd.DataFrame({"value": [0.8, 0.4, 0.0, -0.4, -0.8], "size": [0.2, -0.1, 0.5, 0.1, -0.3]}, index=assets)
    F = pd.DataFrame(np.diag([0.0004, 0.0002]), index=B.columns, columns=B.columns)
    D = pd.Series([0.0006, 0.0005, 0.0007, 0.0006, 0.0005], index=assets)
    pi = pd.Series([0.05, 0.04, 0.045, 0.035, 0.03], index=assets)
    return B, F, D, pi


def test_with_views_only_on_assets_it_is_the_ordinary_black_litterman_posterior(factor_world):
    B, F, D, pi = factor_world
    views = pd.DataFrame([[1.0, -1.0, 0, 0, 0], [0, 0, 1.0, 0, 0]], columns=list("ABCDE"), index=["a-b", "c"])
    q = pd.Series([0.02, 0.06], index=views.index)
    sigma_rr = abl.joint_covariance(B, F, D).loc[B.index, B.index]
    ref = black_litterman_posterior(sigma_rr, pi, views, q, tau=0.05, view_confidence=0.3)
    got = abl.augmented_posterior(B, F, D, pi, views, q, tau=0.05, view_confidence=0.3)
    assert got["assets"].to_numpy() == pytest.approx(ref.reindex(B.index).to_numpy(), rel=1e-6, abs=1e-9)


def test_a_view_on_a_factor_moves_each_asset_in_proportion_to_its_loading_and_leaves_an_unloaded_one_alone(factor_world):
    B, F, D, pi = factor_world
    views = pd.DataFrame([[0, 0, 0, 0, 0, 1.0, 0.0]], columns=list("ABCDE") + ["value", "size"], index=["value up"])
    got = abl.augmented_posterior(B, F, D, pi, views, pd.Series([0.05], index=views.index))
    shift = got["assets"] - pi
    assert shift["C"] == pytest.approx(0.0, abs=1e-12), "an asset with no loading on the factor does not move"
    assert shift["A"] > 0 > shift["E"] and got["factors"]["value"] > got["prior_factors"]["value"]
    assert (shift / B["value"]).drop("C").round(10).nunique() == 1, "the shifts are proportional to the loadings"
    assert got["factors"]["size"] == pytest.approx(got["prior_factors"]["size"]), "a factor uncorrelated with the viewed one is unchanged"


def test_a_view_can_mix_assets_and_factors_and_bad_views_are_refused(factor_world):
    B, F, D, pi = factor_world
    cols = list("ABCDE") + ["value", "size"]
    mixed = pd.DataFrame([[1.0, 0, 0, 0, 0, -1.0, 0]], columns=cols, index=["A beats the value factor"])
    got = abl.augmented_posterior(B, F, D, pi, mixed, pd.Series([0.01], index=mixed.index))
    assert np.isfinite(got["assets"]).all() and got["sigma"].shape == (7, 7)
    with pytest.raises(ValueError, match="unknown names"):
        abl.augmented_posterior(B, F, D, pi, pd.DataFrame([[1.0]], columns=["Z"], index=["v"]), pd.Series([0.01], index=["v"]))
    with pytest.raises(ValueError, match="at least one view"):
        abl.augmented_posterior(B, F, D, pi, pd.DataFrame(), pd.Series(dtype=float))
    prior = abl.factor_prior(B, abl.joint_covariance(B, F, D).loc[B.index, B.index], pi)
    assert list(prior.index) == ["value", "size"] and np.isfinite(prior).all()
