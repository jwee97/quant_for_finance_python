"""Linear-Gaussian state-space models and the Kalman filter, written from the recursions.

Model (Durbin & Koopman 2012)::

    y_t     = Z_t a_t + d_t + eps_t,       eps_t ~ N(0, H_t)         observation
    a_{t+1} = T a_t + c + R eta_t,         eta_t ~ N(0, Q)            state

``kalman_filter`` returns the one-step-ahead predictions, the FILTERED states (which use data up to ``t`` only, so they can be traded), the innovations and
the exact Gaussian log-likelihood; missing observations (NaN) simply skip the update. ``kalman_smoother`` (Rauch-Tung-Striebel) uses the whole sample and is for
description only. Unknown variances are estimated by maximum likelihood.

Ready-made models:

* ``local_level`` / ``local_linear_trend``: a stochastic level (and slope). The filtered slope is a smooth, adaptive trend signal; the signal-to-noise ratio ``q``
  sets how quickly it adapts and is estimated from the data rather than chosen as a window length.
* ``time_varying_regression``: ``y_t = x_t' beta_t + e_t`` with ``beta`` a random walk: a rolling beta that needs no window.
* ``kalman_hedge_ratio``: the time-varying hedge ratio and spread of a pairs trade (Elliott, van der Hoek & Malcolm 2005).

Initialisation: stationary states use the unconditional covariance (``solve_discrete_lyapunov``); non-stationary states use a large diagonal ``P0`` (approximate
diffuse), and the first ``n_diffuse`` innovations are dropped from the likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, optimize


@dataclass
class FilterResult:
    loglik: float
    predicted_state: np.ndarray       # a_{t|t-1}, shape (n, k)
    predicted_cov: np.ndarray         # P_{t|t-1}, shape (n, k, k)
    filtered_state: np.ndarray        # a_{t|t}
    filtered_cov: np.ndarray          # P_{t|t}
    innovations: np.ndarray           # v_t, shape (n, p)
    innovation_var: np.ndarray        # F_t, shape (n, p, p)
    n_obs_used: int


class StateSpace:
    """``Z`` may be ``(p, k)`` or ``(n, p, k)`` (time-varying); ``H`` is ``(p, p)`` or ``(n, p, p)``."""

    def __init__(self, T, Z, H, Q, R=None, c=None, d=None, a0=None, P0=None):
        self.T = np.atleast_2d(np.asarray(T, dtype=float))
        self.Z = np.asarray(Z, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        k = self.T.shape[0]
        self.R = np.eye(k) if R is None else np.atleast_2d(np.asarray(R, dtype=float))
        self.c = np.zeros(k) if c is None else np.asarray(c, dtype=float)
        self.d = d
        self.a0 = np.zeros(k) if a0 is None else np.asarray(a0, dtype=float)
        self.P0 = P0
        self.k = k

    def _initial_cov(self) -> np.ndarray:
        if self.P0 is not None:
            return np.atleast_2d(np.asarray(self.P0, dtype=float))
        RQR = self.R @ self.Q @ self.R.T
        if np.max(np.abs(np.linalg.eigvals(self.T))) < 1.0 - 1e-9:
            return linalg.solve_discrete_lyapunov(self.T, RQR)
        return np.eye(self.k) * 1e7

    def filter(self, y, n_diffuse: int = 0) -> FilterResult:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        n, p = y.shape
        k = self.k
        a, P = self.a0.copy(), self._initial_cov()
        pa, pP = np.empty((n, k)), np.empty((n, k, k))
        fa, fP = np.empty((n, k)), np.empty((n, k, k))
        vs, Fs = np.full((n, p), np.nan), np.full((n, p, p), np.nan)
        ll, used = 0.0, 0
        RQR = self.R @ self.Q @ self.R.T
        for t in range(n):
            Zt = self.Z[t] if self.Z.ndim == 3 else self.Z
            Zt = np.atleast_2d(Zt)
            Ht = self.H[t] if self.H.ndim == 3 else np.atleast_2d(self.H)
            dt = np.zeros(p) if self.d is None else np.asarray(self.d if np.ndim(self.d) == 1 else np.asarray(self.d)[t], dtype=float)
            pa[t], pP[t] = a, P
            obs = np.isfinite(y[t])
            if obs.all():
                v = y[t] - Zt @ a - dt
                F = Zt @ P @ Zt.T + Ht
                K = P @ Zt.T @ np.linalg.inv(F)
                a_f, P_f = a + K @ v, P - K @ Zt @ P
                vs[t], Fs[t] = v, F
                if t >= n_diffuse:
                    sign, logdet = np.linalg.slogdet(F)
                    ll += -0.5 * (p * np.log(2 * np.pi) + logdet + v @ np.linalg.solve(F, v))
                    used += 1
            elif obs.any():
                idx = np.flatnonzero(obs)
                Zo, Ho, yo, do = Zt[idx], Ht[np.ix_(idx, idx)], y[t][idx], dt[idx]
                v = yo - Zo @ a - do
                F = Zo @ P @ Zo.T + Ho
                K = P @ Zo.T @ np.linalg.inv(F)
                a_f, P_f = a + K @ v, P - K @ Zo @ P
                if t >= n_diffuse:
                    ll += -0.5 * (len(idx) * np.log(2 * np.pi) + np.linalg.slogdet(F)[1] + v @ np.linalg.solve(F, v))
                    used += 1
            else:
                a_f, P_f = a, P
            P_f = 0.5 * (P_f + P_f.T)
            fa[t], fP[t] = a_f, P_f
            a = self.T @ a_f + self.c
            P = self.T @ P_f @ self.T.T + RQR
        return FilterResult(float(ll), pa, pP, fa, fP, vs, Fs, used)

    def smooth(self, y, n_diffuse: int = 0) -> dict:
        """Rauch-Tung-Striebel smoother: state estimates using ALL the data (not causal)."""
        f = self.filter(y, n_diffuse)
        n = len(f.filtered_state)
        sa, sP = f.filtered_state.copy(), f.filtered_cov.copy()
        for t in range(n - 2, -1, -1):
            Pp = f.predicted_cov[t + 1]
            J = f.filtered_cov[t] @ self.T.T @ np.linalg.inv(Pp)
            sa[t] = f.filtered_state[t] + J @ (sa[t + 1] - f.predicted_state[t + 1])
            sP[t] = f.filtered_cov[t] + J @ (sP[t + 1] - Pp) @ J.T
        return {"smoothed_state": sa, "smoothed_cov": sP, "filter": f}

    def forecast(self, filter_result: FilterResult, steps: int) -> dict:
        """Forecast the state and the observation ``steps`` ahead from the last filtered state (time-invariant ``Z``)."""
        a, P = filter_result.filtered_state[-1], filter_result.filtered_cov[-1]
        RQR = self.R @ self.Q @ self.R.T
        Z = np.atleast_2d(self.Z[-1] if self.Z.ndim == 3 else self.Z)
        H = np.atleast_2d(self.H[-1] if self.H.ndim == 3 else self.H)
        means, variances = [], []
        for _ in range(steps):
            a = self.T @ a + self.c
            P = self.T @ P @ self.T.T + RQR
            means.append(Z @ a + (0 if self.d is None else np.asarray(self.d)))
            variances.append(Z @ P @ Z.T + H)
        return {"mean": np.array(means).squeeze(), "var": np.array(variances).squeeze()}


# ---------------------------------------------------------------------------------------------------------------------- models
def _fit(builder, y, x0, n_diffuse=0, bounds=None):
    y = np.asarray(y, dtype=float)

    def nll(theta):
        try:
            return -builder(theta).filter(y, n_diffuse).loglik
        except np.linalg.LinAlgError:
            return 1e12

    res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    return res, builder(res.x)


@dataclass
class UCResult:
    params: dict
    loglik: float
    model: StateSpace
    filtered: pd.DataFrame
    smoothed: pd.DataFrame
    signal_to_noise: dict
    index: pd.Index

    def forecast(self, steps: int) -> np.ndarray:
        f = self.model.filter(self._y)
        return self.model.forecast(f, steps)["mean"]


def local_level(y: pd.Series) -> UCResult:
    """Local level model ``y_t = mu_t + eps_t``, ``mu_{t+1} = mu_t + eta_t``. Estimates ``sigma_eps^2`` and ``sigma_eta^2`` by ML; the filtered level is an adaptive
    exponential smoother whose smoothing constant is implied by ``q = sigma_eta^2 / sigma_eps^2``."""
    yv = y.dropna()
    v = float(yv.var())

    def build(th):
        return StateSpace([[1.0]], [[1.0]], [[np.exp(th[0])]], [[np.exp(th[1])]], P0=[[1e7]])

    res, m = _fit(build, yv.to_numpy(), [np.log(v * 0.5), np.log(v * 0.1)], n_diffuse=1)
    f = m.filter(yv.to_numpy(), 1)
    s = m.smooth(yv.to_numpy(), 1)
    out = UCResult({"sigma2_eps": float(np.exp(res.x[0])), "sigma2_eta": float(np.exp(res.x[1]))}, f.loglik, m,
                   pd.DataFrame({"level": f.filtered_state[:, 0]}, index=yv.index), pd.DataFrame({"level": s["smoothed_state"][:, 0]}, index=yv.index),
                   {"q": float(np.exp(res.x[1] - res.x[0]))}, yv.index)
    out._y = yv.to_numpy()
    return out


def local_linear_trend(y: pd.Series) -> UCResult:
    """Local linear trend: level ``mu_t`` and slope ``beta_t``, both random walks. ``filtered['slope']`` is a causal trend estimate in units of ``y`` per step."""
    yv = y.dropna()
    v = float(yv.diff().var())
    T = [[1.0, 1.0], [0.0, 1.0]]

    def build(th):
        return StateSpace(T, [[1.0, 0.0]], [[np.exp(th[0])]], np.diag([np.exp(th[1]), np.exp(th[2])]), P0=np.eye(2) * 1e7)

    res, m = _fit(build, yv.to_numpy(), [np.log(v * 0.5), np.log(v * 0.1), np.log(v * 0.001)], n_diffuse=2)
    f = m.filter(yv.to_numpy(), 2)
    s = m.smooth(yv.to_numpy(), 2)
    out = UCResult({"sigma2_eps": float(np.exp(res.x[0])), "sigma2_level": float(np.exp(res.x[1])), "sigma2_slope": float(np.exp(res.x[2]))}, f.loglik, m,
                   pd.DataFrame(f.filtered_state, index=yv.index, columns=["level", "slope"]), pd.DataFrame(s["smoothed_state"], index=yv.index, columns=["level", "slope"]),
                   {"q_level": float(np.exp(res.x[1] - res.x[0])), "q_slope": float(np.exp(res.x[2] - res.x[0]))}, yv.index)
    out._y = yv.to_numpy()
    return out


def time_varying_regression(y: pd.Series, X: pd.DataFrame, q: float | None = None, add_constant: bool = True, fit_q: bool = True) -> dict:
    """``y_t = x_t' beta_t + e_t``, ``beta_{t+1} = beta_t + eta_t`` with ``Q = q^2 sigma_e^2 I`` (one signal-to-noise ratio for all coefficients).

    ``q`` is estimated by ML when ``fit_q`` (or fixed at the given value). Returns the FILTERED coefficients (causal, using data to ``t``), the smoothed ones and the
    one-step-ahead prediction errors ``v_t`` (standardised, which should be white if the model is right). Because ``beta_t`` at date ``t`` multiplies ``x_t`` already
    known at ``t - 1`` in a forecasting regression, pass lagged regressors if the coefficient is to predict the next return.
    """
    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    yv = frame["__y__"].to_numpy()
    Xv = frame[list(X.columns)].to_numpy()
    names = list(X.columns)
    if add_constant:
        Xv = np.column_stack([np.ones(len(Xv)), Xv])
        names = ["const"] + names
    n, k = Xv.shape
    Z = Xv[:, None, :]
    s2 = float(np.var(yv - Xv @ np.linalg.lstsq(Xv, yv, rcond=None)[0]))

    def build(th):
        se2, qq = np.exp(th[0]), np.exp(th[1]) if q is None or fit_q else q
        return StateSpace(np.eye(k), Z, [[se2]], np.eye(k) * se2 * qq ** 2, P0=np.eye(k) * 1e3 * max(s2, 1e-12) / max(Xv.var(axis=0).mean(), 1e-12))

    x0 = [np.log(s2), np.log(q if q else 0.05)]
    if fit_q or q is None:
        res, m = _fit(build, yv, x0, n_diffuse=k)
        se2, qq = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    else:
        m = build(x0)
        se2, qq = s2, q
    f = m.filter(yv, k)
    sm = m.smooth(yv, k)
    idx = frame.index
    return {"filtered": pd.DataFrame(f.filtered_state, index=idx, columns=names), "smoothed": pd.DataFrame(sm["smoothed_state"], index=idx, columns=names),
            "standardised_errors": pd.Series(f.innovations[:, 0] / np.sqrt(f.innovation_var[:, 0, 0]), index=idx), "sigma2": se2, "q": qq, "loglik": f.loglik,
            "filtered_std": pd.DataFrame(np.sqrt(np.einsum("tii->ti", f.filtered_cov)), index=idx, columns=names)}


def kalman_hedge_ratio(y: pd.Series, x: pd.Series, q: float | None = None) -> dict:
    """Time-varying hedge ratio of ``y`` on ``x`` (with an intercept): returns ``beta`` (filtered, causal), the ``spread = y - alpha_t - beta_t x`` computed
    from the PREVIOUS date's coefficients (the spread you could have known at the time) and its standardised version."""
    res = time_varying_regression(y, x.to_frame("x"), q=q, fit_q=q is None)
    f = res["filtered"]
    yy, xx = y.reindex(f.index), x.reindex(f.index)
    prev = f.shift(1)
    spread = yy - prev["const"] - prev["x"] * xx
    z = res["standardised_errors"]
    return {"beta": f["x"], "alpha": f["const"], "spread": spread, "zscore": z, "q": res["q"], "sigma2": res["sigma2"]}
