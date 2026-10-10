"""Portfolios from graphs: filtered networks of the correlation matrix (minimum spanning tree, triangulated maximally filtered graph), centrality on them, and hierarchical sensitivity parity.

**Filtering.** A correlation matrix of ``N`` assets has ``N(N-1)/2`` numbers, most of them noise. A *filtered graph* keeps only the strongest structure: the *minimum spanning tree* (MST) keeps ``N - 1`` edges, the
shortest connected skeleton under the distance ``d = sqrt(2 (1 - rho))``; the *triangulated maximally filtered graph* (TMFG, Massara, Di Matteo and Aste 2016) keeps ``3 (N - 2)`` edges, built by starting with the four
assets that are most correlated overall and repeatedly inserting the remaining asset into the triangle where its total similarity is largest. The TMFG is planar, contains the MST's information plus loops, and is a
chain of four-cliques glued along triangles.

**Centrality.** How embedded in the network is each asset? Degree, strength (sum of similarity over edges), eigenvector centrality, and *clique centrality*: the total similarity of the four-cliques an asset belongs to
(on the MST, which has no cliques, its edges). Assets at the centre are hubs whose returns move with many others; those at the periphery are the diversifiers. A *peripheral* portfolio holds assets in inverse
proportion to centrality (Pozzi, Di Matteo and Aste 2013 find the periphery pays better per unit of risk, in their data).

**Degeneracy ordering.** Repeatedly removing a vertex of smallest degree gives a *degeneracy ordering* and the *core number* of every vertex; the TMFG is 3-degenerate and the reverse of its insertion order is such an
ordering. It is the order in which a clique tree is built and a cheap measure of how deep inside the graph an asset sits.

**Hierarchical sensitivity parity (HSP).** The source this repository follows names the method but its definition was not available when this was written, so the version here is the author's, built on the same
skeleton as hierarchical risk parity: cluster the assets (a dendrogram on the correlation distance), and at every split of the tree give the two branches weights ``a`` and ``1 - a`` so that the *sensitivity* of
the portfolio to each branch is equal. Two sensitivities are provided: ``risk``, the contribution of each branch to the variance of the two-branch portfolio (including their covariance, which HRP's split ignores),
so ``a (a s_A^2 + b s_AB) = b (b s_B^2 + a s_AB)``; and ``factor``, the exposure of each branch to a small set of principal-component factor shocks, so ``a s_A = b s_B`` with ``s`` the norm of the branch's
factor exposure weighted by the factor variances. With uncorrelated branches the ``risk`` rule gives inverse-*volatility* splits (HRP uses inverse variance); with identical branches both give one half. Treat the
method as a documented generalisation of HRP, not a reproduction of a published algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import brentq
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

from .constraints import Constraints, project_frame
from .covariance import estimate_covariance, nearest_positive_definite


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(2.0 * (1.0 - np.asarray(corr, dtype=float)), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def to_correlation(cov: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    c = cov / np.outer(s, s)
    np.fill_diagonal(c, 1.0)
    return np.clip(c, -1.0, 1.0)


# ------------------------------------------------------------------------------------------------------------------ graphs
@dataclass
class Graph:
    n: int
    edges: list                 # (i, j) with i < j
    weight: np.ndarray          # (n, n) similarity on edges, 0 elsewhere (symmetric)
    cliques: list               # four-cliques (TMFG) in insertion order; empty for a tree
    separators: list            # the triangle each clique was glued on
    order: list                 # vertices in insertion order (TMFG)

    @property
    def adjacency(self) -> np.ndarray:
        a = np.zeros((self.n, self.n))
        for i, j in self.edges:
            a[i, j] = a[j, i] = 1.0
        return a

    def neighbours(self, v: int) -> list:
        return [int(j) for j in np.flatnonzero(self.adjacency[v])]


def mst(similarity: np.ndarray) -> Graph:
    """The minimum spanning tree of the distance ``sqrt(2 (1 - rho))`` when ``similarity`` is a correlation matrix (any similarity in [-1, 1] is treated that way)."""
    S = np.asarray(similarity, dtype=float)
    n = len(S)
    D = correlation_distance(S)
    tree = minimum_spanning_tree(D + np.where(np.eye(n) > 0, 0.0, 1e-12)).toarray()                 # a zero distance would read as "no edge"
    edges = [(int(i), int(j)) if i < j else (int(j), int(i)) for i, j in zip(*np.nonzero(tree))]
    W = np.zeros((n, n))
    for i, j in edges:
        W[i, j] = W[j, i] = S[i, j]
    return Graph(n, sorted(edges), W, [], [], [])


def tmfg(similarity: np.ndarray) -> Graph:
    """The triangulated maximally filtered graph of a symmetric similarity matrix (``n >= 4``)."""
    S = np.asarray(similarity, dtype=float)
    n = len(S)
    if n < 4:
        raise ValueError("the TMFG needs at least four assets")
    W = S - np.diag(np.diag(S))
    strength = np.where(W > W.mean(), W, 0.0).sum(axis=1)
    seed = [int(i) for i in np.argsort(-strength)[:4]]
    edges = {(min(a, b), max(a, b)) for a in seed for b in seed if a < b}
    faces = [tuple(sorted(f)) for f in ((seed[0], seed[1], seed[2]), (seed[0], seed[1], seed[3]), (seed[0], seed[2], seed[3]), (seed[1], seed[2], seed[3]))]
    remaining = [v for v in range(n) if v not in seed]
    cliques, separators, order = [tuple(sorted(seed))], [()], list(seed)
    while remaining:
        gains = np.array([[W[v, f[0]] + W[v, f[1]] + W[v, f[2]] for f in faces] for v in remaining])
        vi, fi = np.unravel_index(np.argmax(gains), gains.shape)
        v, f = remaining.pop(int(vi)), faces.pop(int(fi))
        for a in f:
            edges.add((min(a, v), max(a, v)))
        faces += [tuple(sorted((v, f[0], f[1]))), tuple(sorted((v, f[1], f[2]))), tuple(sorted((v, f[0], f[2])))]
        cliques.append(tuple(sorted((v,) + f)))
        separators.append(f)
        order.append(v)
    Wm = np.zeros((n, n))
    for i, j in edges:
        Wm[i, j] = Wm[j, i] = S[i, j]
    return Graph(n, sorted(edges), Wm, cliques, separators, order)


def degeneracy_ordering(graph: Graph) -> tuple[list, np.ndarray, int]:
    """Smallest-last ordering: repeatedly remove a vertex of minimum degree in the remaining graph. Returns ``(removal order, core number of each vertex, degeneracy)``; the degeneracy is the largest
    degree at removal, and the core number of a vertex is the largest degree-at-removal seen up to and including its own removal."""
    adj = graph.adjacency.astype(bool)
    alive = np.ones(graph.n, dtype=bool)
    order, core = [], np.zeros(graph.n, dtype=int)
    current = 0
    for _ in range(graph.n):
        deg = np.where(alive, adj[:, alive].sum(axis=1), 10 ** 9)
        v = int(np.argmin(deg))
        current = max(current, int(deg[v]))
        core[v] = current
        order.append(v)
        alive[v] = False
    return order, core, int(current)


# ------------------------------------------------------------------------------------------------------------------ centrality
def centrality(graph: Graph, kind: str = "clique") -> np.ndarray:
    """``degree``, ``strength`` (sum of similarity over edges, shifted to be positive), ``eigenvector`` (Perron vector of the similarity-weighted adjacency, shifted positive) or ``clique`` (the total of
    the similarity over the edges of every four-clique containing the vertex; on a tree, the strength)."""
    n = graph.n
    shift = 1.0 + 1e-9                                                                                  # similarities live in [-1, 1]: shift to (0, 2] so centrality is positive
    A = graph.adjacency
    Wp = (graph.weight + shift * A)
    if kind == "degree":
        return A.sum(axis=1)
    if kind == "strength":
        return Wp.sum(axis=1)
    if kind == "eigenvector":
        vals, vecs = np.linalg.eigh(Wp)
        v = np.abs(vecs[:, -1])
        return v / v.sum() * n
    if kind == "clique":
        if not graph.cliques:
            return Wp.sum(axis=1)
        c = np.zeros(n)
        for q in graph.cliques:
            members = list(q)
            total = sum(Wp[a, b] for i, a in enumerate(members) for b in members[i + 1:] if A[a, b])
            for v in members:
                c[v] += total
        return c
    raise ValueError("kind must be degree, strength, eigenvector or clique")


def centrality_weights(similarity: np.ndarray, graph: str = "tmfg", kind: str = "clique", tilt: str = "peripheral") -> np.ndarray:
    """Weights inversely proportional to centrality (``peripheral``) or proportional to it (``central``)."""
    g = tmfg(similarity) if graph == "tmfg" else mst(similarity)
    c = centrality(g, kind)
    if tilt not in ("peripheral", "central"):
        raise ValueError("tilt must be peripheral or central")
    w = 1.0 / c if tilt == "peripheral" else c
    return w / w.sum()


# ------------------------------------------------------------------------------------------------------------------ hierarchical sensitivity parity
def _split_share(cov: np.ndarray, wa: np.ndarray, ia: list, wb: np.ndarray, ib: list, kind: str, factor: tuple | None) -> float:
    """The share ``a`` of the left branch such that the two branches have equal sensitivity."""
    if kind == "factor":
        B, fvar = factor
        sa = np.sqrt(np.sum((B[ia].T @ wa) ** 2 * fvar))
        sb = np.sqrt(np.sum((B[ib].T @ wb) ** 2 * fvar))
        return float(sb / (sa + sb)) if sa + sb > 0 else 0.5
    sA = float(wa @ cov[np.ix_(ia, ia)] @ wa)
    sB = float(wb @ cov[np.ix_(ib, ib)] @ wb)
    sAB = float(wa @ cov[np.ix_(ia, ib)] @ wb)

    def f(a):
        b = 1.0 - a
        return a * (a * sA + b * sAB) - b * (b * sB + a * sAB)

    return float(brentq(f, 0.0, 1.0)) if f(0.0) * f(1.0) < 0 else 0.5


def hsp_weights(covariance: pd.DataFrame, linkage_method: str = "ward", kind: str = "risk", factors: int = 3) -> pd.Series:
    """Hierarchical sensitivity parity weights, long only and fully invested. See the module docstring: this is a documented generalisation of hierarchical risk parity, not a published algorithm."""
    if kind not in ("risk", "factor"):
        raise ValueError("kind must be risk or factor")
    assets = list(covariance.columns)
    n = len(assets)
    if n == 1:
        return pd.Series(1.0, index=assets)
    cov = nearest_positive_definite(covariance).to_numpy(dtype=float)
    corr = to_correlation(cov)
    if n == 2:
        link = np.array([[0, 1, 1.0, 2]])
    else:
        link = linkage(squareform(correlation_distance(corr), checks=False), method=linkage_method)
    factor = None
    if kind == "factor":
        vals, vecs = np.linalg.eigh(cov)
        K = min(factors, n)
        factor = (vecs[:, -K:] * np.sqrt(vals[-K:]), np.ones(K))                                        # loadings scaled by factor volatility, unit shocks
    root = to_tree(link)

    def build(node):
        if node.is_leaf():
            return [node.id], np.ones(1)
        ia, wa = build(node.get_left())
        ib, wb = build(node.get_right())
        a = _split_share(cov, wa, ia, wb, ib, kind, factor)
        return ia + ib, np.concatenate([a * wa, (1.0 - a) * wb])

    ids, w = build(root)
    out = np.zeros(n)
    out[ids] = w
    return pd.Series(out / out.sum(), index=assets)


# ------------------------------------------------------------------------------------------------------------------ through time
def graph_book(returns: pd.DataFrame, investable: pd.DataFrame, method: str = "graph", graph: str = "tmfg", centrality_kind: str = "clique", tilt: str = "peripheral", sensitivity: str = "risk",
               lookback: int = 252, rebalance_index: pd.DatetimeIndex | None = None, covariance_method: str = "shrinkage", constraints: Constraints | None = None, min_assets: int = 5) -> pd.DataFrame:
    """``method='graph'`` (centrality weights on the MST or TMFG) or ``'hsp'`` through time, re-solved at each rebalance date from the trailing ``lookback`` window only."""
    index = pd.DatetimeIndex(returns.index)
    dates = rebalance_index if rebalance_index is not None else index
    book = pd.DataFrame(np.nan, index=index, columns=returns.columns)
    for stamp in dates:
        if stamp not in index:
            continue
        pos = index.get_loc(stamp)
        if pos < lookback:
            continue
        window = returns.iloc[pos - lookback + 1:pos + 1]
        live = [c for c in returns.columns if bool(investable.loc[stamp, c]) and window[c].notna().sum() > lookback * 0.8]
        if len(live) < max(min_assets, 4 if method == "graph" and graph == "tmfg" else 2):
            continue
        sample = window[live].dropna(how="any")
        if len(sample) < lookback * 0.6:
            continue
        try:
            cov = estimate_covariance(sample, covariance_method, lookback, annualise=True)
            if method == "hsp":
                w = hsp_weights(cov, "ward", sensitivity)
            else:
                w = pd.Series(centrality_weights(to_correlation(cov.to_numpy(dtype=float)), graph, centrality_kind, tilt), index=live)
        except (ValueError, np.linalg.LinAlgError):
            continue
        book.loc[stamp, live] = w.reindex(live).to_numpy()
    book = book.ffill().fillna(0.0)
    return project_frame(book, constraints) if constraints is not None else book
