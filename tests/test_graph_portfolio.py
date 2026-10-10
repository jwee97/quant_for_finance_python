"""Filtered graphs, centrality and hierarchical sensitivity parity: the graphs have the properties that define them, the centrality tilts point the right way, and the weights obey their own equations."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src.features.sleeves import month_end_dates
from src.framework import ALLOCATORS, bundle_from_prices, load_library
from src.framework.allocation import Context
from src.portfolio import graph_portfolio as gp
from src.portfolio.hierarchical import hrp_weights
from src.utils.config import load_config

load_library()


def block_corr(blocks=(4, 4, 4), within=0.7, between=0.1, seed=0, noise=0.0):
    n = sum(blocks)
    C = np.full((n, n), between)
    s = 0
    for b in blocks:
        C[s:s + b, s:s + b] = within
        s += b
    np.fill_diagonal(C, 1.0)
    if noise:
        rng = np.random.default_rng(seed)
        E = rng.normal(scale=noise, size=(n, n))
        C = np.clip(C + (E + E.T) / 2, -0.95, 0.95)
        np.fill_diagonal(C, 1.0)
    return C


def random_corr(n=10, seed=0, k=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, k)) @ rng.normal(size=(k, n)) + rng.normal(size=(400, n))
    return np.corrcoef(X.T)


def connected(n, edges):
    seen, stack = {0}, [0]
    while stack:
        v = stack.pop()
        for a, b in edges:
            for u, w in ((a, b), (b, a)):
                if u == v and w not in seen:
                    seen.add(w)
                    stack.append(w)
    return len(seen) == n


# ------------------------------------------------------------------------------------------------------------------ MST and TMFG
def test_mst_is_the_cheapest_spanning_tree_found_by_brute_force():
    C = random_corr(6, 1)
    D = gp.correlation_distance(C)
    g = gp.mst(C)
    cost = lambda es: sum(D[i, j] for i, j in es)                                                        # noqa: E731
    best = min((es for es in combinations(list(combinations(range(6), 2)), 5) if connected(6, es)), key=cost)
    assert len(g.edges) == 5 and connected(6, g.edges) and cost(g.edges) == pytest.approx(cost(best))


def test_tmfg_has_the_edges_cliques_and_degeneracy_of_a_planar_triangulation():
    C = random_corr(14, 2)
    g = gp.tmfg(C)
    n = 14
    assert len(g.edges) == 3 * (n - 2) and len(g.cliques) == n - 3 and connected(n, g.edges)
    A = g.adjacency
    for q in g.cliques:
        assert all(A[a, b] for a, b in combinations(q, 2))                                               # every clique is complete
    assert set(g.order) == set(range(n))
    _, core, degeneracy = gp.degeneracy_ordering(g)
    assert degeneracy == 3 and core.max() == 3
    inserted = {v: i for i, v in enumerate(g.order)}
    for v in g.order[4:]:                                                                                # each later vertex joined three earlier ones
        assert sum(inserted[u] < inserted[v] for u in g.neighbours(v)) == 3


def test_tmfg_keeps_the_strong_within_block_links():
    C = block_corr(noise=0.03, seed=1)
    g = gp.tmfg(C)
    block = np.repeat([0, 1, 2], 4)
    within = np.mean([block[i] == block[j] for i, j in g.edges])
    assert within > 0.5                                                                                   # chance is 18/66 = 0.27; a triangulation cannot avoid some links between blocks
    m = gp.mst(C)
    assert np.mean([block[i] == block[j] for i, j in m.edges]) > 0.7


def test_tmfg_needs_four_assets_and_the_mst_works_on_three():
    with pytest.raises(ValueError):
        gp.tmfg(np.eye(3))
    assert len(gp.mst(random_corr(3, 4)).edges) == 2


def test_degeneracy_of_known_graphs():
    path = gp.Graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)], np.zeros((5, 5)), [], [], [])
    assert gp.degeneracy_ordering(path)[2] == 1
    cycle = gp.Graph(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)], np.zeros((5, 5)), [], [], [])
    assert gp.degeneracy_ordering(cycle)[2] == 2
    k4 = gp.Graph(4, list(combinations(range(4), 2)), np.zeros((4, 4)), [], [], [])
    order, core, d = gp.degeneracy_ordering(k4)
    assert d == 3 and sorted(order) == [0, 1, 2, 3] and (core == 3).all()


# ------------------------------------------------------------------------------------------------------------------ centrality
@pytest.mark.parametrize("kind", ["degree", "strength", "eigenvector", "clique"])
def test_centrality_is_positive_and_a_hub_is_more_central_than_a_leaf(kind):
    n = 8
    C = np.full((n, n), 0.05)
    C[0, 1:] = C[1:, 0] = 0.8                                                                          # asset 0 correlates strongly with everything
    np.fill_diagonal(C, 1.0)
    g = gp.mst(C)
    c = gp.centrality(g, kind)
    assert (c > 0).all() and c[0] == c.max()
    t = gp.tmfg(C)
    ct = gp.centrality(t, kind)
    assert (ct > 0).all() and ct[0] == ct.max()


def test_peripheral_and_central_tilts_are_opposites_and_weights_sum_to_one():
    C = random_corr(12, 3)
    p = gp.centrality_weights(C, "tmfg", "clique", "peripheral")
    c = gp.centrality_weights(C, "tmfg", "clique", "central")
    cen = gp.centrality(gp.tmfg(C), "clique")
    assert p.sum() == pytest.approx(1) and c.sum() == pytest.approx(1) and (p > 0).all()
    assert np.corrcoef(p, cen)[0, 1] < 0 < np.corrcoef(c, cen)[0, 1]
    with pytest.raises(ValueError):
        gp.centrality_weights(C, "tmfg", "clique", "sideways")
    with pytest.raises(ValueError):
        gp.centrality(gp.mst(C), "betweenness")


# ------------------------------------------------------------------------------------------------------------------ HSP
def frame(cov):
    names = [f"A{i}" for i in range(len(cov))]
    return pd.DataFrame(cov, index=names, columns=names)


def test_hsp_weights_are_a_long_only_budget_and_a_pair_splits_by_inverse_volatility():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 9)) * rng.uniform(0.5, 2.0, 9)
    for kind in ("risk", "factor"):
        w = gp.hsp_weights(frame(np.cov(X.T)), "ward", kind)
        assert w.sum() == pytest.approx(1) and (w > 0).all()
    w = gp.hsp_weights(frame(np.diag([1.0, 4.0])), "ward", "risk")                                    # uncorrelated: equal risk contributions are inverse-volatility weights
    assert w.iloc[0] == pytest.approx(2 / 3) and w.iloc[1] == pytest.approx(1 / 3)
    assert gp.hsp_weights(frame(np.array([[2.0, 1.0], [1.0, 2.0]])), "ward", "risk").tolist() == pytest.approx([0.5, 0.5])
    assert gp.hsp_weights(frame(np.eye(1) * 3.0)).tolist() == [1.0]


def test_hsp_equalises_the_risk_contributions_of_the_two_top_branches():
    C = block_corr((3, 5), within=0.6, between=0.2)
    sd = np.linspace(0.5, 1.5, 8)
    cov = C * np.outer(sd, sd)
    w = gp.hsp_weights(frame(cov), "ward", "risk").to_numpy()
    sigma = np.sqrt(w @ cov @ w)
    rc = w * (cov @ w) / sigma ** 2
    left = np.arange(8) < 3
    assert rc[left].sum() == pytest.approx(rc[~left].sum(), abs=0.02)                                    # the two blocks carry the same share of risk


def test_hsp_differs_from_hrp_when_branches_are_correlated():
    C = block_corr((3, 3, 3), within=0.8, between=0.5)
    sd = np.array([1, 1.2, 0.8, 1.5, 1.0, 0.7, 2.0, 1.1, 0.9])
    cov = frame(C * np.outer(sd, sd))
    h, r = gp.hsp_weights(cov, "ward", "risk"), hrp_weights(cov, "single")
    assert np.abs(h - r).max() > 0.01


def test_hsp_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        gp.hsp_weights(frame(np.eye(3)), "ward", "nonsense")


# ------------------------------------------------------------------------------------------------------------------ allocators
@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2014-01-02", periods=900)
    f = rng.normal(0, 0.008, size=(len(idx), 3))
    B = rng.normal(size=(3, 10))
    rets = f @ B + rng.normal(0, 0.006, size=(len(idx), 10))
    return bundle_from_prices(pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=list("ABCDEFGHIJ")), min_history=60, name="graph")


@pytest.mark.parametrize("name,params", [("graph_centrality", {}), ("graph_centrality", {"graph": "mst", "centrality": "strength"}), ("hierarchical_sensitivity_parity", {}),
                                         ("hierarchical_sensitivity_parity", {"sensitivity": "factor"})])
def test_graph_allocators_give_a_capped_long_only_book_without_looking_ahead(bundle, name, params):
    alloc = ALLOCATORS.create(name, **params)
    cfg = load_config()
    w = alloc.build(Context(bundle, cfg))
    rows = w.loc[[d for d in month_end_dates(bundle.index) if w.loc[d].abs().sum() > 1e-9]]
    assert len(rows) > 10 and (rows >= -1e-9).all().all() and np.allclose(rows.sum(axis=1), 1.0, atol=1e-6)
    cut = bundle.index[700]
    prices = bundle.prices.copy()
    prices.loc[prices.index > cut] *= 1.0 + np.linspace(0, 1, int((prices.index > cut).sum()))[:, None] * np.arange(10)[None, :] / 10
    other = bundle_from_prices(prices, min_history=60, name="graph2")
    w2 = alloc.build(Context(other, cfg))
    before = w.index[w.index <= cut]
    assert np.allclose(w.loc[before].to_numpy(), w2.loc[before].to_numpy(), atol=1e-12)


def test_graph_allocators_validate_parameters():
    for name, bad in (("graph_centrality", {"graph": "knn"}), ("graph_centrality", {"tilt": "x"}), ("graph_centrality", {"lookback": 10}), ("hierarchical_sensitivity_parity", {"sensitivity": "x"})):
        with pytest.raises(ValueError):
            ALLOCATORS.create(name, **bad)
