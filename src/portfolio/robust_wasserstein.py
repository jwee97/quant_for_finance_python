"""Portfolios that are robust to the data being a sample: the worst case over a Wasserstein ball of distributions around the empirical one (Mohajerin Esfahani and Kuhn 2018; Cajas, *Advanced Portfolio Optimization*, robust optimisation).

The empirical scenarios are one guess at the distribution of returns. Allow the true distribution to be any whose type-1 Wasserstein distance from it is at most ``eps`` (the cost of moving probability mass, with
the Euclidean distance between return vectors as the cost per unit): the **worst-case** expected loss of a portfolio ``w`` is then, for a loss that is linear in the returns, the empirical loss plus
``eps ||w||_2``, and for the CVaR at tail probability ``beta`` (``beta = 1 - alpha``) the empirical CVaR plus ``eps ||w||_2 / beta``. The bound is attained: move every scenario ``eps`` against the portfolio.

The penalty is a norm of the weights, so the robust portfolio is a *less concentrated* one: ``||w||_2`` is smallest for equal weights. ``eps = 0`` is the ordinary minimum-CVaR portfolio.

It is a second-order cone programme. This module solves it with linear programmes only: the norm is replaced by the largest of many linear lower bounds ``v'w`` with ``||v||_2 = 1`` and a new bound is added at the
current weights after each solve (Kelley's cutting planes), which converges to the exact optimum; the result carries the gap that is left.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog


def empirical_cvar(losses: np.ndarray, beta: float) -> float:
    """The mean of the worst ``beta`` share of ``losses`` (Rockafellar-Uryasev form, so a fractional tail is weighted correctly)."""
    x = np.asarray(losses, dtype=float)
    tau = np.quantile(x, 1.0 - beta, method="higher")
    return float(tau + np.maximum(x - tau, 0.0).sum() / (beta * len(x)))


def worst_case_cvar(weights, scenarios, eps: float, alpha: float = 0.95) -> float:
    """The worst-case CVaR of a portfolio over the Wasserstein ball: ``CVaR_emp + eps ||w||_2 / (1 - alpha)``. ``scenarios`` are returns, so the loss is ``-r'w``."""
    w = np.asarray(weights, dtype=float)
    R = np.asarray(scenarios, dtype=float)
    return empirical_cvar(-(R @ w), 1.0 - alpha) + eps * float(np.linalg.norm(w)) / (1.0 - alpha)


def wasserstein_cvar_weights(scenarios: pd.DataFrame, eps: float = 0.0, alpha: float = 0.95, lb: float = 0.0, ub: float = 1.0, max_iter: int = 80, tol: float = 1e-9) -> dict:
    """Weights (summing to one, within ``[lb, ub]``) that minimise the worst-case CVaR over a Wasserstein ball of radius ``eps`` (in the units of the scenarios' returns: ``0.001`` is ten basis points of
    movement per scenario). Returns the ``weights``, the worst-case and empirical CVaR, the cutting-plane ``gap`` left (zero up to ``tol`` when converged) and ``ok``."""
    clean = scenarios.dropna(how="any")
    if len(clean) < 50:
        raise ValueError(f"need at least 50 complete scenarios, got {len(clean)}")
    if not 0.0 < alpha < 1.0 or eps < 0:
        raise ValueError("alpha must be in (0, 1) and eps must not be negative")
    assets, R = list(clean.columns), clean.to_numpy(dtype=float)
    T, n = R.shape
    if lb * n > 1.0 + 1e-12 or ub * n < 1.0 - 1e-12:
        raise ValueError("the bounds leave no fully invested portfolio")
    beta = 1.0 - alpha
    # variables: w (n), tau (1), u_t >= 0 (T), s >= 0 (the norm); minimise tau + sum(u)/(beta T) + eps s / beta, with u_t >= -r_t'w - tau and s >= v_k'w for every cut v_k
    nv = n + 1 + T + 1
    c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / (beta * T)), [eps / beta]])
    A0 = sparse.hstack([-sparse.csr_matrix(R), -sparse.csr_matrix(np.ones((T, 1))), -sparse.identity(T, format="csr"), sparse.csr_matrix((T, 1))]).tocsr()     # -r'w - tau - u <= 0
    A_eq = sparse.csr_matrix(np.concatenate([np.ones(n), np.zeros(nv - n)])[None, :])
    bounds = [(lb, ub)] * n + [(None, None)] + [(0, None)] * T + [(0, None)]
    cuts = [np.full(n, 1.0 / np.sqrt(n))]
    w = np.full(n, 1.0 / n)
    gap = float("inf")
    for _ in range(max_iter):
        rows = [sparse.csr_matrix(np.concatenate([v, np.zeros(T + 1), [-1.0]])[None, :]) for v in cuts]                  # v'w - s <= 0
        A = sparse.vstack([A0] + rows).tocsr()
        res = linprog(c, A_ub=A, b_ub=np.zeros(A.shape[0]), A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
        if not res.success:
            return {"weights": pd.Series(np.full(n, 1.0 / n), index=assets), "worst_case_cvar": float("nan"), "empirical_cvar": float("nan"), "gap": float("nan"), "ok": False, "message": str(res.message)}
        w = np.clip(res.x[:n], lb, ub)
        s, norm = float(res.x[-1]), float(np.linalg.norm(w))
        gap = (norm - s) * eps / beta                                   # the penalty the cuts so far under-count
        if eps == 0.0 or norm - s <= tol * max(norm, 1.0):
            break
        cuts.append(w / norm)
    w = w / w.sum()
    return {"weights": pd.Series(w, index=assets), "worst_case_cvar": worst_case_cvar(w, R, eps, alpha), "empirical_cvar": empirical_cvar(-(R @ w), beta), "gap": max(float(gap), 0.0), "ok": True, "message": "optimal"}
