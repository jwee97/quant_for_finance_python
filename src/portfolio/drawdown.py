"""Drawdown-based and exponential-tail risk, and the portfolios that minimise them.

**Drawdown** at ``t`` is how far the cumulative return has fallen from its running peak. These measures use the *uncompounded* cumulative return ``y_t = sum_(s<=t) r_s`` (the formulation of
Chekhlov, Uryasev and Zabarankin 2005), which is what makes the problem linear in the weights; the compounded drawdown of a finished portfolio is in :mod:`src.risk.performance`.

* :func:`max_drawdown_weights`: minimise the largest drawdown, a linear programme.
* :func:`cdar_weights`: minimise the **Conditional Drawdown at Risk**, the mean of the worst ``1 - alpha`` share of drawdowns. ``alpha -> 1`` is the maximum drawdown, ``alpha = 0`` the average drawdown. Linear programme.
* :func:`evar_weights`: minimise the **Entropic Value at Risk** of losses (Ahmadi-Javid 2012), ``EVaR_alpha(L) = inf_(z>0) z ln( E[e^(L/z)] / (1 - alpha) )``, a coherent measure that bounds both VaR and CVaR from
  above and takes all moments of the loss into account. Smooth convex programme in ``(w, z)``.

The scenarios are rows in TIME ORDER: shuffling them changes every drawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp


def drawdown_path(returns) -> pd.Series | np.ndarray:
    """``max_(s<=t) y_s - y_t`` with ``y`` the cumulative sum of returns starting at zero (so the first peak is the starting equity)."""
    r = np.asarray(returns, dtype=float)
    y = np.cumsum(r)
    d = np.maximum.accumulate(np.maximum(y, 0.0)) - y
    return pd.Series(d, index=returns.index) if isinstance(returns, pd.Series) else d


def max_drawdown(returns) -> float:
    return float(np.max(drawdown_path(returns)))


def cdar(returns, alpha: float = 0.95) -> float:
    """The mean of the worst ``1 - alpha`` share of the drawdown path (``alpha = 0.95`` is the average of the worst 5%). The Rockafellar-Uryasev form is used, so that a fractional tail is weighted correctly."""
    d = np.asarray(drawdown_path(returns), dtype=float)
    t = len(d)
    zeta = np.quantile(d, alpha, method="higher")
    return float(zeta + np.maximum(d - zeta, 0.0).sum() / ((1.0 - alpha) * t))


def evar(returns, alpha: float = 0.95) -> float:
    """The Entropic Value at Risk of the LOSS ``-r`` at level ``alpha``, in the units of ``returns``. Minimised over ``z`` numerically; at least ``CVaR``."""
    loss = -np.asarray(returns, dtype=float)
    scale = float(np.std(loss)) or 1.0
    x = loss / scale
    c = np.log(len(x)) + np.log(1.0 - alpha)

    def f(logz):
        z = np.exp(logz)
        return z * (logsumexp(x / z) - c)

    from scipy.optimize import minimize_scalar

    r = minimize_scalar(f, bounds=(-8.0, 8.0), method="bounded", options={"xatol": 1e-10})
    return float(r.fun * scale)


def _bounds(n: int, lb: float, ub: float, long_only: bool) -> list:
    return [((0.0 if long_only and lb < 0 else lb), ub)] * n


def _prep(scenarios: pd.DataFrame, min_rows: int = 60):
    clean = scenarios.dropna(how="any")
    if len(clean) < min_rows:
        raise ValueError(f"need at least {min_rows} complete scenario rows in time order, got {len(clean)}")
    return list(clean.columns), clean.to_numpy(dtype=float)


def _return_constraint(R: np.ndarray, mu, target_return, n: int):
    if target_return is None:
        return None, None
    m = np.asarray(mu if mu is not None else R.mean(axis=0), dtype=float)
    return -m, -float(target_return)


def cdar_weights(scenarios: pd.DataFrame, alpha: float = 0.95, target_return: float | None = None, mu=None, lb: float = 0.0, ub: float = 1.0) -> dict:
    """Weights (summing to one, within ``[lb, ub]``) that minimise CDaR on the scenario path, optionally subject to an expected return of at least ``target_return`` per period (``mu`` per period; the
    mean of the scenarios when omitted). Returns ``weights``, the achieved ``cdar`` and ``max_drawdown``, and ``ok``."""
    assets, R = _prep(scenarios)
    T, n = R.shape
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    C = np.cumsum(R, axis=0)                                           # y_t = C_t w
    # variables: w (n), zeta (1), z_t (T) >= 0, u_t (T) >= 0
    nv = n + 1 + 2 * T
    I = sparse.identity(T, format="csr")
    D = sparse.diags([np.ones(T - 1)], [-1], shape=(T, T), format="csr")      # u_(t-1)
    rows = []
    # z_t >= u_t - C_t w - zeta        ->  -z_t + u_t - C_t w - zeta <= 0
    rows.append(sparse.hstack([-sparse.csr_matrix(C), -sparse.csr_matrix(np.ones((T, 1))), -I, I]))
    # u_t >= C_t w                      ->  C_t w - u_t <= 0
    rows.append(sparse.hstack([sparse.csr_matrix(C), sparse.csr_matrix((T, 1)), sparse.csr_matrix((T, T)), -I]))
    # u_t >= u_(t-1)                    ->  u_(t-1) - u_t <= 0
    rows.append(sparse.hstack([sparse.csr_matrix((T, n + 1)), sparse.csr_matrix((T, T)), D - I]))
    A_ub = sparse.vstack(rows).tocsr()
    b_ub = np.zeros(A_ub.shape[0])
    g, rhs = _return_constraint(R, mu, target_return, n)
    if g is not None:
        A_ub = sparse.vstack([A_ub, sparse.csr_matrix(np.concatenate([g, np.zeros(nv - n)])[None, :])]).tocsr()
        b_ub = np.append(b_ub, rhs)
    c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / ((1.0 - alpha) * T)), np.zeros(T)])
    A_eq = sparse.csr_matrix(np.concatenate([np.ones(n), np.zeros(nv - n)])[None, :])
    bounds = [(lb, ub)] * n + [(None, None)] + [(0, None)] * T + [(0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        return {"weights": pd.Series(np.full(n, 1.0 / n), index=assets), "cdar": float("nan"), "max_drawdown": float("nan"), "ok": False, "message": str(res.message)}
    w = np.clip(res.x[:n], lb, ub)
    w = w / w.sum()
    pr = R @ w
    return {"weights": pd.Series(w, index=assets), "cdar": cdar(pr, alpha), "max_drawdown": max_drawdown(pr), "ok": True, "message": "optimal"}


def max_drawdown_weights(scenarios: pd.DataFrame, target_return: float | None = None, mu=None, lb: float = 0.0, ub: float = 1.0) -> dict:
    """Weights that minimise the largest drawdown of the scenario path (a linear programme)."""
    assets, R = _prep(scenarios)
    T, n = R.shape
    C = np.cumsum(R, axis=0)
    nv = n + 1 + T                                                     # w, Delta, u
    I = sparse.identity(T, format="csr")
    D = sparse.diags([np.ones(T - 1)], [-1], shape=(T, T), format="csr")
    rows = [sparse.hstack([-sparse.csr_matrix(C), -sparse.csr_matrix(np.ones((T, 1))), I]),         # u_t - C_t w - Delta <= 0
            sparse.hstack([sparse.csr_matrix(C), sparse.csr_matrix((T, 1)), -I]),                    # C_t w - u_t <= 0
            sparse.hstack([sparse.csr_matrix((T, n + 1)), D - I])]                                   # u_(t-1) - u_t <= 0
    A_ub = sparse.vstack(rows).tocsr()
    b_ub = np.zeros(A_ub.shape[0])
    g, rhs = _return_constraint(R, mu, target_return, n)
    if g is not None:
        A_ub = sparse.vstack([A_ub, sparse.csr_matrix(np.concatenate([g, np.zeros(nv - n)])[None, :])]).tocsr()
        b_ub = np.append(b_ub, rhs)
    c = np.concatenate([np.zeros(n), [1.0], np.zeros(T)])
    A_eq = sparse.csr_matrix(np.concatenate([np.ones(n), np.zeros(nv - n)])[None, :])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(lb, ub)] * n + [(0, None)] + [(0, None)] * T, method="highs")
    if not res.success:
        return {"weights": pd.Series(np.full(n, 1.0 / n), index=assets), "max_drawdown": float("nan"), "ok": False, "message": str(res.message)}
    w = np.clip(res.x[:n], lb, ub)
    w = w / w.sum()
    return {"weights": pd.Series(w, index=assets), "max_drawdown": max_drawdown(R @ w), "ok": True, "message": "optimal"}


def evar_weights(scenarios: pd.DataFrame, alpha: float = 0.95, target_return: float | None = None, mu=None, lb: float = 0.0, ub: float = 1.0) -> dict:
    """Weights that minimise the Entropic VaR of losses at level ``alpha`` (jointly over the weights and the entropic parameter ``z``; the objective is convex in both). The scenarios are rescaled to
    percent internally for conditioning."""
    assets, R = _prep(scenarios, 30)
    T, n = R.shape
    X = R * 100.0
    c = np.log(T) + np.log(1.0 - alpha)

    def f(v):
        w, z = v[:n], v[n]
        a = -(X @ w) / z
        return float(z * (logsumexp(a) - c))

    def grad(v):
        w, z = v[:n], v[n]
        a = -(X @ w) / z
        lse = logsumexp(a)
        p = np.exp(a - lse)
        return np.concatenate([-(X.T @ p), [lse - c + float(p @ (X @ w)) / z]])

    cons = [{"type": "eq", "fun": lambda v: float(v[:n].sum() - 1.0), "jac": lambda v: np.concatenate([np.ones(n), [0.0]])}]
    if target_return is not None:
        m = np.asarray(mu if mu is not None else R.mean(axis=0), dtype=float) * 100.0
        cons.append({"type": "ineq", "fun": lambda v: float(m @ v[:n] - target_return * 100.0), "jac": lambda v: np.concatenate([m, [0.0]])})
    w0 = np.full(n, 1.0 / n)
    z0 = float(np.std(-(X @ w0))) or 1.0
    res = minimize(f, np.append(w0, z0), jac=grad, method="SLSQP", bounds=[(lb, ub)] * n + [(1e-4, None)], constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    w = np.clip(res.x[:n], lb, ub)
    w = w / w.sum()
    return {"weights": pd.Series(w, index=assets), "evar": evar(R @ w, alpha), "ok": bool(res.success), "message": str(res.message)}
