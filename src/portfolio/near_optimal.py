"""Near-optimal centering (Cajas 2024).

The optimiser's answer is a corner: with estimated inputs, a long-only mean-variance optimum puts most of the weight in a few assets and zero in the rest, and moves a long way when the inputs move a
little, because the objective is almost flat around its maximum. Many portfolios are nearly as good. Instead of the single best one, take the *centre* of the region of nearly-best portfolios.

Define the utility ``U(w) = mu'w - (lambda/2) w'Sigma w`` (``mu = 0`` is minimum variance), the optimum ``U*`` and the value ``U_ew`` of the equal-weight portfolio. The near-optimal region is
``U(w) >= U* - eps (U* - U_ew)``: ``eps = 0`` is the optimum alone, ``eps = 1`` every portfolio at least as good as equal weight. Its analytic centre maximises the log-barrier
``sum_i log(w_i - lb) + sum_i log(ub - w_i) + log(U(w) - U_eps)`` over the budget plane: a point in the interior of every constraint, as far from each as the region allows. It is unique, interior
(every asset held when the bounds are not binding), and moves smoothly with the inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..equity.qp import solve_qp


def _utility(w, mu, S, lam):
    return float(mu @ w - 0.5 * lam * w @ S @ w)


def optimum(mu, cov, risk_aversion: float = 1.0, lb: float = 0.0, ub: float = 1.0) -> np.ndarray:
    n = len(cov)
    r = solve_qp(risk_aversion * np.asarray(cov, dtype=float), -np.asarray(mu, dtype=float), A=np.ones((1, n)), l=[1.0], u=[1.0], lb=lb, ub=ub)
    w = np.asarray(r.x, dtype=float)
    return w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else np.full(n, 1.0 / n)


def effective_n(w) -> float:
    w = np.asarray(w, dtype=float)
    return float(1.0 / (w ** 2).sum()) if (w ** 2).sum() > 0 else float("nan")


def near_optimal_center(mu, cov: pd.DataFrame, risk_aversion: float = 1.0, tolerance: float = 0.1, lb: float = 0.0, ub: float = 1.0) -> dict:
    """The centre of the near-optimal region (see the module note). ``mu`` may be ``None`` for minimum variance. Returns the centred ``weights``, the ``optimum`` and the utilities of both and of equal weight, how much of the
    optimal utility advantage over equal weight was given up (``loss_share``), the effective number of assets of each, and whether the barrier solve converged."""
    assets = list(cov.columns)
    n = len(assets)
    S = 0.5 * (cov.to_numpy(dtype=float) + cov.to_numpy(dtype=float).T)
    m = np.zeros(n) if mu is None else np.asarray(pd.Series(mu).reindex(assets).fillna(0.0), dtype=float)
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("tolerance must be between 0 and 1")
    if lb * n > 1.0 + 1e-12 or ub * n < 1.0 - 1e-12:
        raise ValueError("the bounds leave no fully invested portfolio")
    w_star = optimum(m, S, risk_aversion, lb, ub)
    w_eq = np.full(n, 1.0 / n)
    u_star, u_eq = _utility(w_star, m, S, risk_aversion), _utility(w_eq, m, S, risk_aversion)
    gap = max(u_star - u_eq, 0.0)
    out = {"assets": assets, "optimum": pd.Series(w_star, index=assets), "u_optimum": u_star, "u_equal": u_eq}
    if tolerance == 0.0 or gap < 1e-14:
        return {**out, "weights": pd.Series(w_star if tolerance == 0.0 else w_eq, index=assets), "u_center": u_star if tolerance == 0.0 else u_eq, "loss_share": 0.0 if tolerance == 0.0 else 1.0, "n_eff_optimum": effective_n(w_star),
                "n_eff_center": effective_n(w_star if tolerance == 0.0 else w_eq), "converged": True}
    floor = u_star - tolerance * gap
    tiny = 1e-9

    def barrier(w):
        s = [w - lb, ub - w, np.array([_utility(w, m, S, risk_aversion) - floor])]
        v = np.concatenate(s)
        return float(-np.log(np.maximum(v, tiny)).sum())

    def grad(w):
        u = _utility(w, m, S, risk_aversion)
        gu = m - risk_aversion * S @ w
        return -(1.0 / np.maximum(w - lb, tiny)) + 1.0 / np.maximum(ub - w, tiny) - gu / max(u - floor, tiny)

    # a strictly interior start: slide from the optimum toward equal weight until every bound is slack and the utility is above the floor
    theta = 0.5
    for theta in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9):
        w0 = (1 - theta) * w_star + theta * w_eq
        if (w0 > lb + 1e-7).all() and (w0 < ub - 1e-7).all() and _utility(w0, m, S, risk_aversion) > floor + 1e-12:
            break
    res = minimize(barrier, w0, jac=grad, method="SLSQP", bounds=[(lb + tiny, ub - tiny)] * n, constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0), "jac": lambda w: np.ones(n)}],
                   options={"maxiter": 500, "ftol": 1e-14})
    w = np.asarray(res.x, dtype=float)
    ok = bool(res.success) and _utility(w, m, S, risk_aversion) >= floor - 1e-9 and np.isfinite(w).all()
    if not ok:
        w = w0
    w = w / w.sum()
    u_c = _utility(w, m, S, risk_aversion)
    return {**out, "weights": pd.Series(w, index=assets), "u_center": u_c, "loss_share": float((u_star - u_c) / gap), "n_eff_optimum": effective_n(w_star), "n_eff_center": effective_n(w), "converged": ok}


def weight_dispersion(returns: pd.DataFrame, fit, n_boot: int = 30, seed: int = 0) -> dict:
    """How much the weights move when the data are resampled: ``fit(returns) -> weights`` (a Series or an array in column order) is run on the sample and on ``n_boot`` bootstrap resamples of its rows; the dispersion is the mean one-way
    turnover (``sum |w_b - w| / 2``) between each resample's weights and the sample's, and the spread of the weight of each asset. Lower is more stable."""
    rng = np.random.default_rng(seed)

    def weights(frame) -> pd.Series:
        w = fit(frame)
        return (w if isinstance(w, pd.Series) else pd.Series(np.asarray(w, dtype=float), index=returns.columns)).reindex(returns.columns).fillna(0.0)

    base = weights(returns)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(returns), len(returns))
        draws.append(weights(returns.iloc[idx].reset_index(drop=True)))
    D = pd.DataFrame(draws)
    return {"turnover": float((D.sub(base, axis=1).abs().sum(axis=1) / 2).mean()), "weight_std": D.std(ddof=1), "mean_weights": D.mean(), "base": base}
