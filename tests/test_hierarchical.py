"""Hierarchical risk parity (Generation 2, Priority 11).

HRP and HERC have no closed-form answer to check against, so these tests pin
the exact invariants the algorithms must satisfy. Each one would fail on a
plausible implementation bug: a wrong leaf order, a bisection that ignores
cluster variance, a label-dependent result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster, leaves_list

from src.portfolio.constraints import Constraints
from src.portfolio.hierarchical import (
    choose_number_of_clusters,
    cluster_variance,
    correlation_distance,
    correlation_matrix,
    herc_weights,
    hierarchical_book,
    hierarchical_linkage,
    hrp_weights,
    quasi_diagonal_order,
    weight_stability,
)
from src.portfolio.inverse_vol import inverse_volatility_weights


def _names(n: int) -> list[str]:
    return [f"A{i}" for i in range(n)]


@pytest.fixture(scope="module")
def random_cov():
    rng = np.random.default_rng(0)
    n = 9
    panel = rng.normal(size=(600, n)) @ rng.normal(size=(n, n))
    return pd.DataFrame(np.cov(panel.T), index=_names(n), columns=_names(n))


def test_correlation_distance_is_a_metric(random_cov):
    d = correlation_distance(correlation_matrix(random_cov)).to_numpy()
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)
    assert (d >= 0).all() and (d <= 1.0 + 1e-12).all()
    # Triangle inequality on every triple: what makes Ward linkage valid.
    n = len(d)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert d[i, j] <= d[i, k] + d[k, j] + 1e-12


def test_perfect_correlation_has_zero_distance_and_perfect_anticorrelation_one():
    corr = pd.DataFrame([[1, 1, -1], [1, 1, -1], [-1, -1, 1]], dtype=float)
    d = correlation_distance(corr).to_numpy()
    assert d[0, 1] == pytest.approx(0.0)
    assert d[0, 2] == pytest.approx(1.0)


def test_quasi_diagonal_order_matches_scipy_leaf_order(random_cov):
    link = hierarchical_linkage(random_cov, "single")
    assert quasi_diagonal_order(link) == list(leaves_list(link))


def test_quasi_diagonal_order_is_a_permutation(random_cov):
    order = quasi_diagonal_order(hierarchical_linkage(random_cov, "single"))
    assert sorted(order) == list(range(len(random_cov)))


def test_hrp_weights_are_a_valid_long_only_book(random_cov):
    w = hrp_weights(random_cov)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_hrp_with_uncorrelated_assets_is_exactly_inverse_variance():
    """With zero correlation, recursive bisection collapses to 1/sigma^2 weights."""
    vols = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.12, 0.18])
    cov = pd.DataFrame(np.diag(vols ** 2), index=_names(8), columns=_names(8))
    expected = (1 / vols ** 2) / (1 / vols ** 2).sum()
    assert hrp_weights(cov).to_numpy() == pytest.approx(expected, abs=1e-12)


def test_hrp_with_identical_assets_is_equal_weight():
    n = 8
    corr = np.full((n, n), 0.4)
    np.fill_diagonal(corr, 1.0)
    cov = pd.DataFrame(corr * 0.04, index=_names(n), columns=_names(n))
    assert hrp_weights(cov).to_numpy() == pytest.approx(np.full(n, 1 / n), abs=1e-12)


def test_hrp_is_invariant_to_asset_order(random_cov):
    rng = np.random.default_rng(3)
    shuffled = random_cov.iloc[list(rng.permutation(len(random_cov)))]
    shuffled = shuffled[shuffled.index]
    a = hrp_weights(random_cov).sort_index()
    b = hrp_weights(shuffled).sort_index()
    assert a.to_numpy() == pytest.approx(b.to_numpy(), abs=1e-10)


def test_hrp_gives_the_quietest_isolated_asset_the_most_weight():
    """Two clusters, one far quieter: the quiet one must receive more capital."""
    n = 6
    corr = np.eye(n)
    for block in ([0, 1, 2], [3, 4, 5]):
        for i in block:
            for j in block:
                if i != j:
                    corr[i, j] = 0.8
    vols = np.array([0.30, 0.30, 0.30, 0.03, 0.03, 0.03])
    cov = pd.DataFrame(corr * np.outer(vols, vols), index=_names(n), columns=_names(n))
    w = hrp_weights(cov)
    assert w.iloc[3:].sum() > 0.8


def test_cluster_variance_of_a_single_asset_is_its_variance(random_cov):
    cov = random_cov.to_numpy()
    assert cluster_variance(cov, [2]) == pytest.approx(cov[2, 2])


def test_herc_weights_are_valid_and_order_invariant(random_cov):
    w = herc_weights(random_cov)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    rng = np.random.default_rng(5)
    shuffled = random_cov.iloc[list(rng.permutation(len(random_cov)))]
    shuffled = shuffled[shuffled.index]
    assert herc_weights(shuffled).sort_index().to_numpy() == pytest.approx(
        w.sort_index().to_numpy(), abs=1e-9)


def test_herc_finds_the_planted_clusters():
    """Three well-separated blocks must be recovered as three clusters."""
    n_per, blocks = 3, 3
    n = n_per * blocks
    corr = np.full((n, n), 0.05)
    for b in range(blocks):
        sl = slice(b * n_per, (b + 1) * n_per)
        corr[sl, sl] = 0.85
    np.fill_diagonal(corr, 1.0)
    cov = pd.DataFrame(corr * 0.04, index=_names(n), columns=_names(n))
    link = hierarchical_linkage(cov, "ward")
    distance = correlation_distance(correlation_matrix(cov)).to_numpy()
    k = choose_number_of_clusters(link, distance, (2, 6))
    labels = fcluster(link, t=k, criterion="maxclust")
    assert k == 3
    for b in range(blocks):
        assert len(set(labels[b * n_per:(b + 1) * n_per])) == 1


def _block_cov(blocks: int, n_per: int = 3) -> pd.DataFrame:
    n = n_per * blocks
    corr = np.zeros((n, n))
    for b in range(blocks):
        sl = slice(b * n_per, (b + 1) * n_per)
        corr[sl, sl] = 0.7
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr * 0.04, index=_names(n), columns=_names(n))


def test_herc_splits_two_identical_clusters_evenly():
    n_per = 3
    w = herc_weights(_block_cov(2, n_per), k=2)
    totals = [w.iloc[b * n_per:(b + 1) * n_per].sum() for b in range(2)]
    assert totals == pytest.approx([0.5, 0.5], abs=1e-6)


def test_herc_splits_identical_clusters_evenly_when_the_tree_is_genuinely_balanced():
    """((B0, B1), (B2, B3)) with identical blocks: each cluster must get 25%.

    Four mutually *equidistant* clusters would not do. Under Ward linkage their
    merge costs tie exactly (1.5 e^2 to join two blocks, and 2 x 0.75 e^2 =
    1.5 e^2 to add a third), so the tree shape is decided by tie-breaking on
    index and comes out as a chain. The structure has to be hierarchical:
    related pairs of blocks, unrelated across pairs.
    """
    n_per, blocks = 3, 4
    n = n_per * blocks
    corr = np.zeros((n, n))
    for b in range(blocks):
        sl = slice(b * n_per, (b + 1) * n_per)
        corr[sl, sl] = 0.7
    for first, second in ((0, 1), (2, 3)):                  # sibling blocks
        corr[first * n_per:(first + 1) * n_per, second * n_per:(second + 1) * n_per] = 0.3
        corr[second * n_per:(second + 1) * n_per, first * n_per:(first + 1) * n_per] = 0.3
    np.fill_diagonal(corr, 1.0)
    cov = pd.DataFrame(corr * 0.04, index=_names(n), columns=_names(n))
    w = herc_weights(cov, k=blocks)
    totals = [w.iloc[b * n_per:(b + 1) * n_per].sum() for b in range(blocks)]
    assert totals == pytest.approx([0.25] * 4, abs=1e-6)


def test_herc_three_identical_clusters_follow_the_tree_not_equal_thirds():
    """With three clusters the dendrogram is unbalanced, and the allocation says so.

    Ward merges two of the identical clusters first. The root then splits
    1/sigma against sqrt(2)/sigma, because the merged pair is diversified and
    so has volatility sigma/sqrt(2). The lone cluster therefore receives
    1/(1+sqrt(2)) and the pair splits what is left. This is a known property
    of tree-based allocation, not an implementation error, and it is why HERC
    is "equal risk contribution" in the hierarchical sense only.
    """
    n_per = 3
    w = herc_weights(_block_cov(3, n_per), k=3)
    totals = sorted(w.iloc[b * n_per:(b + 1) * n_per].sum() for b in range(3))
    lone = 1.0 / (1.0 + np.sqrt(2.0))
    pair = (1.0 - lone) / 2.0
    assert totals == pytest.approx(sorted([pair, pair, lone]), abs=1e-6)


def test_hierarchical_book_is_causal_and_respects_constraints(synthetic_market):
    constraints = Constraints(min_weight=0.0, max_weight=0.45, group_limits={},
                              group_map={}, net_exposure=1.0)
    returns = synthetic_market.returns()
    book = hierarchical_book(returns, synthetic_market.investable, "hrp", lookback=120,
                             constraints=constraints, min_assets=3)
    live = book.loc[book.abs().sum(axis=1) > 0]
    assert len(live) > 100
    assert np.allclose(live.sum(axis=1), 1.0)
    assert live.max().max() <= 0.45 + 1e-9

    altered = returns.copy()
    altered.iloc[900:] *= 25.0
    changed = hierarchical_book(altered, synthetic_market.investable, "hrp", lookback=120,
                                constraints=constraints, min_assets=3)
    pd.testing.assert_frame_equal(book.iloc[:900], changed.iloc[:900])


def test_weight_stability_reports_every_method(synthetic_returns):
    methods = {
        "inverse_vol": lambda cov: inverse_volatility_weights(pd.Series(np.sqrt(np.diag(cov)), index=cov.columns)),
        "hrp": hrp_weights,
    }
    table = weight_stability(synthetic_returns.tail(500), methods, n_bootstrap=20, block_length=21)
    assert set(table.index) == {"inverse_vol", "hrp"}
    assert (table["mean_effective_n"] >= 1.0).all()
    assert (table["mean_pairwise_l1"] >= 0.0).all()
