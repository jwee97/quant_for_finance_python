"""Commodity curves: estimating the Schwartz-Smith two-factor model by Kalman filter, seasonality, and curve-shape factors.

The Schwartz-Smith (2000) model gives every point on a futures curve in closed form from two unobserved factors, ``ln F(T) = e^{-kappa T} chi + xi + A(T)`` (see ``futures``). With the
factors as the state and the log futures prices of the nearest ``N`` contracts as noisy observations, that is a linear Gaussian state-space model with a measurement matrix that changes
daily as contracts age, which ``econometrics.statespace`` handles directly; ``fit_schwartz_smith`` maximises the exact Gaussian likelihood. The filtered factors are the objects a
trader wants: ``chi`` (the short-term deviation: how tight the market is) and ``xi`` (the long-run price level), and the fitted risk premium explains the contango or backwardation.

``seasonal_factors`` measures calendar-month return seasonality with HAC inference (natural gas and agricultural contracts have real ones); ``curve_factors`` gives the level, slope and
curvature of the observed curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from ..econometrics.statespace import StateSpace
from ..stats.regression import ols
from .futures import SchwartzSmithParams, schwartz_smith_A, term_structure


def _ss_model(theta, ttm: np.ndarray, dt: float, y_dim: int, p0_xi: float) -> StateSpace:
    kappa, sig_chi, sig_xi, rho, mu_xi, lam, mu_star, sig_eps = theta
    p = SchwartzSmithParams(kappa=kappa, sigma_chi=sig_chi, mu_xi=mu_xi, sigma_xi=sig_xi, rho=rho, lambda_chi=lam, mu_xi_star=mu_star)
    a = np.exp(-kappa * dt)
    T = np.array([[a, 0.0], [0.0, 1.0]])
    sd_chi2 = sig_chi ** 2 * (1 - a ** 2) / (2 * kappa)
    cov_cx = rho * sig_chi * sig_xi * (1 - a) / kappa
    Q = np.array([[sd_chi2, cov_cx], [cov_cx, sig_xi ** 2 * dt]])
    c = np.array([0.0, (mu_xi - 0.5 * sig_xi ** 2) * dt])
    Z = np.stack([np.column_stack([np.exp(-kappa * ttm[t]), np.ones(y_dim)]) for t in range(len(ttm))])
    d = np.stack([schwartz_smith_A(ttm[t], p) for t in range(len(ttm))])
    H = np.eye(y_dim) * sig_eps ** 2
    return StateSpace(T, Z, H, Q, c=c, d=d, a0=np.array([0.0, p0_xi]), P0=np.diag([sd_chi2 / max(1 - a ** 2, 1e-9) * (1 - a ** 2), 10.0]))


def fit_schwartz_smith(long: pd.DataFrame, depth: int = 6, x0: tuple | None = None, max_iter: int = 80, n_starts: int = 1) -> dict:
    """Maximum-likelihood Schwartz-Smith fit to the nearest ``depth`` contracts. ``x0`` = starting ``(kappa, sigma_chi, sigma_xi, rho, mu_xi, lambda_chi, mu_xi_star, sigma_eps)``.
    The likelihood is rugged in the mean-reversion speed, the short-term volatility and the measurement noise (different optimisers stop at different local maxima), so the
    likelihood is first evaluated on a grid over those three parameters around ``x0`` and the ``n_starts`` best points, with ``x0`` itself, are optimised; the best result is kept.
    Returns the parameters (as ``SchwartzSmithParams`` plus the measurement noise), the log-likelihood and the FILTERED factors (causal)."""
    ts = term_structure(long, depth)
    price, ttm = ts["price"].dropna(), ts["ttm"].reindex(ts["price"].dropna().index)
    y = np.log(price.to_numpy())
    tau = ttm.to_numpy()
    dt = 1.0 / 252.0
    n, p = y.shape
    start = list(x0) if x0 is not None else [1.0, 0.3, 0.15, 0.0, 0.0, 0.05, 0.0, 0.005]
    lower = [0.05, 0.02, 0.02, -0.95, -0.5, -1.0, -0.5, 1e-4]
    upper = [10.0, 2.0, 1.0, 0.95, 0.5, 1.0, 0.5, 0.05]
    p0_xi = float(y[0].mean())

    def nll(theta):
        try:
            return -_ss_model(theta, tau, dt, p, p0_xi).filter(y, n_diffuse=1).loglik
        except (np.linalg.LinAlgError, ValueError):
            return 1e12

    candidates = [list(start)]
    for kappa in (0.5, 1.0, 1.5, 2.5):
        for sig_chi in (0.2, 0.35, 0.5):
            for sig_eps in (0.002, 0.005):
                c = list(start)
                c[0], c[1], c[7] = kappa, sig_chi, sig_eps
                candidates.append(c)
    scored = sorted(((nll(np.clip(c, lower, upper)), i) for i, c in enumerate(candidates)))
    starts = [candidates[0]] + [candidates[i] for _, i in scored if i != 0][: max(int(n_starts), 1)]
    res = None
    for s in starts:
        r = optimize.minimize(nll, np.clip(s, lower, upper), method="L-BFGS-B", bounds=list(zip(lower, upper)), options={"maxiter": max_iter, "maxfun": 4000, "ftol": 1e-10})
        if res is None or r.fun < res.fun:
            res = r
    model = _ss_model(res.x, tau, dt, p, p0_xi)
    f = model.filter(y, n_diffuse=1)
    kappa, sig_chi, sig_xi, rho, mu_xi, lam, mu_star, sig_eps = res.x
    return {"params": SchwartzSmithParams(kappa, sig_chi, mu_xi, sig_xi, rho, lam, mu_star), "sigma_eps": float(sig_eps), "loglik": float(f.loglik), "converged": bool(res.success),
            "chi": pd.Series(f.filtered_state[:, 0], index=price.index), "xi": pd.Series(f.filtered_state[:, 1], index=price.index), "theta": res.x}


def curve_factors(price: pd.DataFrame) -> pd.DataFrame:
    """Level (``ln F1``), slope (``ln F_last - ln F1``, negative = backwardation) and curvature (``ln F1 - 2 ln F_mid + ln F_last``) of each day's curve."""
    logp = np.log(price.dropna(how="any"))
    cols = list(logp.columns)
    mid = cols[len(cols) // 2]
    return pd.DataFrame({"level": logp[cols[0]], "slope": logp[cols[-1]] - logp[cols[0]], "curvature": logp[cols[0]] - 2 * logp[mid] + logp[cols[-1]]})


def seasonal_factors(returns: pd.Series) -> pd.DataFrame:
    """Average return by calendar month (annualised) relative to the overall mean, with Newey-West t statistics from a regression on month dummies. A seasonal premium is
    credible only if it is large for the month AND its t statistic survives the 12-month multiple-testing penalty (|t| above ~2.9)."""
    r = returns.dropna()
    dummies = pd.get_dummies(r.index.month, prefix="m").astype(float)
    dummies.index = r.index
    X = dummies.iloc[:, 1:]
    res = ols(r, X, "HAC", lags=5)
    base = float(res.params["const"])
    months = {}
    for m in range(1, 13):
        if m == 1 and "m_1" not in X.columns:
            effect, t = base, res.tvalues["const"]
        else:
            effect = base + float(res.params.get(f"m_{m}", 0.0))
            t = res.tvalues.get(f"m_{m}", np.nan)
        months[m] = {"mean_daily": effect, "annualised": effect * 252.0, "excess_vs_average": (effect - r.mean()) * 252.0, "t_vs_january": t}
    return pd.DataFrame(months).T.rename_axis("month")
