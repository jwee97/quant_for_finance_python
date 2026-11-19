"""Nested Clustered Optimisation (López de Prado 2016, *A Robust Estimator of the Efficient Frontier*).

Mean-variance fails on a large, correlated universe because the optimiser puts the noise of a badly conditioned covariance matrix into the weights (Markowitz's curse). Clustering confines the
instability: solve each cluster on its own, where the assets resemble each other and the problem is small, collapse each cluster to one synthetic asset, and solve the small problem between the clusters.

1. cluster the assets on the correlation distance (Ward linkage; the number of clusters by silhouette unless given; ``detone`` removes the market component of the correlation first so that clusters are
   not just 'the market');
2. inside each cluster, solve the chosen problem (``min_variance``, ``max_sharpe`` or ``risk_parity``), long-only;
3. the cluster weights matrix ``W`` (assets x clusters) gives the reduced moments ``W' Sigma W`` and ``W' mu``;
4. solve the same problem between clusters, and the final weights are ``W`` times the cluster weights.

The weights are long-only and sum to one by construction (each stage's are).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..equity.qp import solve_qp
from .covariance import nearest_positive_definite
from .hierarchical import choose_number_of_clusters, correlation_distance, correlation_matrix
from .risk_parity import risk_parity_weights

OBJECTIVES = ("min_variance", "max_sharpe", "risk_parity")


def long_only(cov: np.ndarray, mu: np.ndarray | None, objective: str) -> np.ndarray:
    """Long-only, fully invested weights for one stage. Maximum Sharpe is the QP ``min w'Sigma w  s.t.  mu'w = 1, w >= 0`` rescaled to sum to one (Cornuejols and Tutuncu), and falls back to minimum
    variance when no asset has a positive expected return."""
    n = len(cov)
    if n == 1:
        return np.ones(1)
    if objective == "risk_parity":
        return risk_parity_weights(pd.DataFrame(cov)).to_numpy(dtype=float)
    if objective == "max_sharpe" and mu is not None and np.max(mu) > 0:
        r = solve_qp(cov, np.zeros(n), A=np.asarray(mu, dtype=float)[None, :], l=[1.0], u=[1.0], lb=0.0)
        w = np.clip(np.asarray(r.x, dtype=float), 0.0, None)
        if np.isfinite(w).all() and w.sum() > 0:
            return w / w.sum()
    r = solve_qp(cov, np.zeros(n), A=np.ones((1, n)), l=[1.0], u=[1.0], lb=0.0)
    w = np.clip(np.asarray(r.x, dtype=float), 0.0, None)
    return w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else np.full(n, 1.0 / n)


def clusters(cov: pd.DataFrame, k: int | None = None, k_range: tuple[int, int] = (2, 8), detone: int = 0, method: str = "ward") -> np.ndarray:
    """Cluster labels (1..k) of the assets of ``cov`` by Ward linkage on the correlation distance, with ``k`` chosen by silhouette when ``None``."""
    corr = correlation_matrix(cov)
    if detone:
        from ..riskengine.covariance import detone_correlation

        corr = detone_correlation(corr, detone)
    dist = correlation_distance(corr).to_numpy(dtype=float)
    link = linkage(squareform(dist, checks=False), method=method)
    k = int(k) if k else choose_number_of_clusters(link, dist, k_range)
    return fcluster(link, t=k, criterion="maxclust")


def nco_weights(cov: pd.DataFrame, mu: pd.Series | None = None, objective: str = "min_variance", k: int | None = None, k_range: tuple[int, int] = (2, 8), detone: int = 0, details: bool = False):
    """The NCO weights (a Series indexed like ``cov``); with ``details=True`` also the labels, the within-cluster weights and the weights between clusters."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}")
    assets = list(cov.columns)
    n = len(assets)
    S = nearest_positive_definite(cov).to_numpy(dtype=float)
    m = None if mu is None else mu.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    if n <= 3:
        w = long_only(S, m, objective)
        out = pd.Series(w, index=assets)
        return (out, {"labels": np.ones(n, dtype=int), "inner": out, "outer": pd.Series([1.0])}) if details else out
    labels = clusters(pd.DataFrame(S, index=assets, columns=assets), k, k_range, detone)
    ids = sorted(set(labels))
    W = np.zeros((n, len(ids)))
    for j, c in enumerate(ids):
        members = np.where(labels == c)[0]
        W[members, j] = long_only(S[np.ix_(members, members)], None if m is None else m[members], objective)
    S_red = W.T @ S @ W
    m_red = None if m is None else W.T @ m
    outer = long_only(0.5 * (S_red + S_red.T), m_red, objective)
    w = pd.Series(W @ outer, index=assets)
    w = w / w.sum()
    if details:
        return w, {"labels": labels, "inner": pd.Series(W.sum(axis=1), index=assets), "outer": pd.Series(outer, index=[f"cluster {c}" for c in ids])}
    return w
