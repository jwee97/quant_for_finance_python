"""Concentration limits on a network of the assets (Cajas, *Advanced Portfolio Optimization*, ch. 13: graph-based portfolios).

A cap on each asset does not stop a portfolio piling into a group of assets that move together. A filtered graph of the correlation matrix (the minimum spanning tree or the TMFG, see
:mod:`src.portfolio.graph_portfolio`) says which assets those are: each vertex is joined to the assets it is most like. This module limits the **weight inside every neighbourhood**: for each asset ``v``,
the weights of ``v`` and of every asset within ``hops`` edges of it sum to at most ``cap``. Minimise variance (or maximise ``mu'w - (lambda/2) w'Sigma w``) subject to that, to the budget, and to bounds.

A cap that no portfolio can meet (the neighbourhood of a hub contains almost everything, or ``cap`` is below the weight the other neighbourhoods force) is refused with the reason, not softened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..equity.qp import solve_qp_ipm
from .covariance import nearest_positive_definite
from .graph_portfolio import Graph, mst, tmfg, to_correlation


def neighbourhoods(graph: Graph, hops: int = 1) -> list[list[int]]:
    """For each vertex, itself and everything within ``hops`` edges, sorted."""
    adj = graph.adjacency > 0
    reach = np.eye(graph.n, dtype=bool)
    for _ in range(int(hops)):
        reach = reach | (reach.astype(int) @ adj.astype(int) > 0)
    return [sorted(int(j) for j in np.flatnonzero(reach[v])) for v in range(graph.n)]


def graph_capped_weights(cov: pd.DataFrame, graph: str = "mst", cap: float = 0.4, hops: int = 1, mu: pd.Series | None = None, risk_aversion: float = 1.0, lb: float = 0.0, ub: float = 1.0) -> dict:
    """Weights that minimise ``(lambda/2) w'Sigma w - mu'w`` (``mu = 0``: minimum variance) with the weight inside every neighbourhood of ``graph`` (``mst`` or ``tmfg``) at most ``cap``.

    Returns the ``weights``, the ``neighbourhood_weight`` of each asset's neighbourhood, the ``binding`` ones, the ``unconstrained`` solution with the same bounds, and the ``variance_cost`` of the limits (the
    ratio of their variances)."""
    assets = list(cov.columns)
    n = len(assets)
    if graph not in ("mst", "tmfg"):
        raise ValueError("graph must be 'mst' or 'tmfg'")
    if n < (4 if graph == "tmfg" else 3):
        raise ValueError(f"the {graph} needs at least {4 if graph == 'tmfg' else 3} assets")
    if not 0.0 < cap <= 1.0 or hops < 1 or risk_aversion <= 0:
        raise ValueError("cap must be in (0, 1], hops at least 1 and risk_aversion positive")
    S = nearest_positive_definite(cov).to_numpy(dtype=float)
    corr = to_correlation(S)
    g = mst(corr) if graph == "mst" else tmfg(corr)
    hoods = neighbourhoods(g, hops)
    m = np.zeros(n) if mu is None else np.asarray(pd.Series(mu).reindex(assets).fillna(0.0), dtype=float)
    A_hood = np.zeros((n, n))
    for v, members in enumerate(hoods):
        A_hood[v, members] = 1.0
    keep = [v for v in range(n) if len(hoods[v]) < n or cap < 1.0]                              # a neighbourhood holding every asset is the budget: it can only say cap >= 1
    if any(len(hoods[v]) == n for v in range(n)) and cap < 1.0 - 1e-12:
        hub = next(v for v in range(n) if len(hoods[v]) == n)
        raise ValueError(f"asset {assets[hub]} is joined to every other within {hops} hop(s) of the {graph}, so its neighbourhood is the whole portfolio and a cap of {cap} cannot be met: use fewer hops, more assets, or the other graph")
    A = np.vstack([np.ones((1, n)), A_hood[keep]])
    lo = np.concatenate([[1.0], np.full(len(keep), -np.inf)])
    hi = np.concatenate([[1.0], np.full(len(keep), cap)])
    free = solve_qp_ipm(risk_aversion * S, -m, A=np.ones((1, n)), l=[1.0], u=[1.0], lb=lb, ub=ub)
    r = solve_qp_ipm(risk_aversion * S, -m, A=A, l=lo, u=hi, lb=lb, ub=ub)
    if not r.ok or not np.isfinite(r.x).all():
        raise ValueError(f"no portfolio meets a neighbourhood cap of {cap} with these bounds ({r.status}): raise the cap or lower the hops")
    w = np.asarray(r.x, dtype=float)
    w = w / w.sum()
    hood_w = A_hood @ w
    wf = np.asarray(free.x, dtype=float)
    return {"weights": pd.Series(w, index=assets), "neighbourhood_weight": pd.Series(hood_w, index=assets), "binding": [assets[v] for v in range(n) if hood_w[v] >= cap - 1e-6], "unconstrained": pd.Series(wf / wf.sum(), index=assets),
            "variance_cost": float(w @ S @ w) / float(wf @ S @ wf) if float(wf @ S @ wf) > 0 else float("nan"), "edges": [(assets[i], assets[j]) for i, j in g.edges], "graph": graph, "hops": int(hops)}
