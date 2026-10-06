"""Four causal estimators, each small enough to read and each validated against simulated truth (Stage 37).

Every function returns plain dictionaries of numbers. None of them knows anything about finance; the stage script supplies the data.

* ``dml_plr``  - double/debiased machine learning for the partially linear model y = theta d + g(X) + e (Chernozhukov et al. 2018): out-of-fold
  residuals of y and d on X, then theta = sum(ry rd) / sum(rd^2) with the robust ("sandwich") standard error.
* ``r_learner`` - Nie and Wager (2021): the same cross-fitted residuals, then a ridge regression of ry on rd * [effect modifiers], so the
  coefficients describe how the effect varies.
* ``iv_2sls``  - two-stage least squares with a heteroskedasticity-robust standard error.
* ``did_2x2``  - difference in differences from the saturated regression, standard errors clustered by unit.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _learner(name: str, seed: int):
    if name == "ridge":
        from sklearn.linear_model import RidgeCV

        return RidgeCV(alphas=np.logspace(-3, 3, 13))
    if name == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_depth=3, max_iter=80, learning_rate=0.08, random_state=seed)
    if name == "gbm_strong":                      # added post-hoc (Stage 37) as a diagnostic: more capacity than the declared learner
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_depth=4, max_iter=600, learning_rate=0.05, min_samples_leaf=10, random_state=seed)
    raise ValueError(f"unknown learner '{name}'")


def cross_fit_residuals(y: np.ndarray, d: np.ndarray, X: np.ndarray, learner: str = "gbm", folds: int = 5, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold residuals of y and of d after predicting each from X."""
    from sklearn.model_selection import KFold

    ry, rd = np.zeros(len(y)), np.zeros(len(y))
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(X):
        ry[test] = y[test] - _learner(learner, seed).fit(X[train], y[train]).predict(X[test])
        rd[test] = d[test] - _learner(learner, seed).fit(X[train], d[train]).predict(X[test])
    return ry, rd


def dml_plr(y, d, X, learner: str = "gbm", folds: int = 5, seed: int = 7) -> dict:
    y, d, X = (np.asarray(a, dtype=float) for a in (y, d, X))
    ry, rd = cross_fit_residuals(y, d, X, learner, folds, seed)
    return dml_from_residuals(ry, rd)


def dml_from_residuals(ry, rd) -> dict:
    """The final DML step from residuals of y and d. With the TRUE nuisance functions this is the oracle estimator, which isolates estimator code from learner error."""
    ry, rd = np.asarray(ry, dtype=float), np.asarray(rd, dtype=float)
    y = ry
    theta = float((rd * ry).sum() / (rd ** 2).sum())
    psi = rd * (ry - theta * rd)
    se = float(np.sqrt(np.mean(psi ** 2) / np.mean(rd ** 2) ** 2 / len(y)))
    return {"theta": theta, "se": se, "p_value": float(2 * norm.sf(abs(theta / se))), "n": len(y)}


def naive_ols(y, d, X=None) -> dict:
    """OLS of y on d (and linear controls if ``X`` is given); robust standard error for the coefficient on d."""
    y, d = np.asarray(y, dtype=float), np.asarray(d, dtype=float)
    cols = [np.ones(len(y)), d] + ([] if X is None else [np.asarray(X, dtype=float)])
    M = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    u = y - M @ beta
    bread = np.linalg.inv(M.T @ M)
    cov = bread @ (M.T * u ** 2) @ M @ bread
    se = float(np.sqrt(cov[1, 1]))
    return {"theta": float(beta[1]), "se": se, "p_value": float(2 * norm.sf(abs(beta[1] / se))), "n": len(y)}


def r_learner(y, d, X, modifiers, learner: str = "gbm", folds: int = 5, seed: int = 7, ridge: float = 1e-3) -> dict:
    """Effect model tau(w) = w . gamma on ``modifiers`` (an (n, m) matrix that should include a constant column if a mean effect is wanted)."""
    y, d, X, W = (np.asarray(a, dtype=float) for a in (y, d, X, modifiers))
    ry, rd = cross_fit_residuals(y, d, X, learner, folds, seed)
    Z = W * rd[:, None]
    gamma = np.linalg.solve(Z.T @ Z + ridge * len(y) * np.eye(Z.shape[1]), Z.T @ ry)
    return {"gamma": gamma, "tau": W @ gamma}


def iv_2sls(y, d, z, controls=None) -> dict:
    y, d = np.asarray(y, dtype=float), np.asarray(d, dtype=float)
    z = np.asarray(z, dtype=float).reshape(len(y), -1)
    C = np.ones((len(y), 1)) if controls is None else np.column_stack([np.ones(len(y)), np.asarray(controls, dtype=float)])
    Zfull = np.column_stack([C, z])
    Xfull = np.column_stack([C, d])
    first, *_ = np.linalg.lstsq(Zfull, Xfull, rcond=None)
    Xhat = Zfull @ first
    beta = np.linalg.solve(Xhat.T @ Xfull, Xhat.T @ y)
    u = y - Xfull @ beta
    bread = np.linalg.inv(Xhat.T @ Xfull)
    cov = bread @ (Xhat.T * u ** 2) @ Xhat @ bread.T
    se = float(np.sqrt(cov[-1, -1]))
    # first-stage strength: F statistic for the instruments
    rss_r = float(((Xfull[:, -1] - C @ np.linalg.lstsq(C, Xfull[:, -1], rcond=None)[0]) ** 2).sum())
    rss_u = float(((Xfull[:, -1] - Zfull @ np.linalg.lstsq(Zfull, Xfull[:, -1], rcond=None)[0]) ** 2).sum())
    q = z.shape[1]
    f_stat = ((rss_r - rss_u) / q) / (rss_u / (len(y) - Zfull.shape[1]))
    return {"theta": float(beta[-1]), "se": se, "p_value": float(2 * norm.sf(abs(beta[-1] / se))), "first_stage_F": float(f_stat), "n": len(y)}


def did_2x2(y, treated, post, unit) -> dict:
    """Coefficient on treated*post in y ~ 1 + treated + post + treated*post, standard error clustered by ``unit``."""
    y, treated, post = (np.asarray(a, dtype=float) for a in (y, treated, post))
    unit = np.asarray(unit)
    M = np.column_stack([np.ones(len(y)), treated, post, treated * post])
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    u = y - M @ beta
    bread = np.linalg.inv(M.T @ M)
    meat = np.zeros((4, 4))
    for g in np.unique(unit):
        s = M[unit == g].T @ u[unit == g]
        meat += np.outer(s, s)
    n_g = len(np.unique(unit))
    k = len(y) - 4
    cov = bread @ meat @ bread * (n_g / (n_g - 1)) * ((len(y) - 1) / k)
    se = float(np.sqrt(cov[3, 3]))
    return {"theta": float(beta[3]), "se": se, "p_value": float(2 * norm.sf(abs(beta[3] / se))), "n": len(y), "clusters": int(n_g)}
