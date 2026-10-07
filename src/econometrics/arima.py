"""ARMA / ARIMA models by exact Gaussian maximum likelihood through the Kalman filter.

``y_t - mu = sum_i phi_i (y_{t-i} - mu) + e_t + sum_j theta_j e_{t-j}``, ``e ~ N(0, sigma^2)``, written in the Harvey state-space form and evaluated with the
filter in ``statespace``. Parameters are optimised in an unconstrained space: the autoregressive and moving-average coefficients are mapped from partial
autocorrelations through ``tanh`` (Jones 1980; Monahan 1984), so every candidate is stationary and invertible.

* ``fit_arima(y, order=(p, d, q))`` differences ``d`` times (pure differencing: the forecast is integrated back to levels).
* ``select_arima`` searches a grid and ranks by AIC / BIC (BIC is the better guard against over-fitting daily financial data).
* ``rolling_forecasts`` refits on an expanding (or rolling) window and forecasts one step ahead, using only past data at each origin.
* ``ar_ols`` / ``yule_walker`` are the simple AR(p) estimators; ``ljung_box`` on the residuals tests whether any linear structure is left.

Daily returns are close to white noise: expect the selected order to be (0, 0, 0) and any out-of-sample R^2 to be about zero. The module exists so that this
is a measured conclusion, and so that the same machinery works on spreads, yields and volatility series where AR structure is real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats as sps

from ..stats.timeseries import ljung_box
from .statespace import StateSpace


def pacf_to_ar(pacf: np.ndarray) -> np.ndarray:
    """Durbin-Levinson map from partial autocorrelations in (-1, 1) to stationary AR coefficients."""
    phi = np.zeros(0)
    for k, r in enumerate(pacf, start=1):
        new = np.empty(k)
        new[: k - 1] = phi - r * phi[::-1]
        new[k - 1] = r
        phi = new
    return phi


def ar_to_pacf(phi: np.ndarray) -> np.ndarray:
    """Inverse of ``pacf_to_ar`` (the reverse Durbin-Levinson recursion)."""
    phi = np.array(phi, dtype=float)
    out = np.zeros(len(phi))
    for k in range(len(phi), 0, -1):
        out[k - 1] = phi[k - 1]
        if k > 1:
            phi = (phi[: k - 1] + out[k - 1] * phi[: k - 1][::-1]) / (1.0 - out[k - 1] ** 2)
    return out


def _unpack(theta, p, q, mean):
    i = 0
    mu = theta[i] if mean else 0.0
    i += int(mean)
    ar = pacf_to_ar(np.tanh(theta[i: i + p])) if p else np.zeros(0)
    i += p
    ma = -pacf_to_ar(np.tanh(theta[i: i + q])) if q else np.zeros(0)     # invertible MA(q): the AR map of the negated polynomial
    i += q
    return mu, ar, ma, float(np.exp(theta[i]))


def arma_state_space(ar: np.ndarray, ma: np.ndarray, sigma2: float) -> StateSpace:
    """Harvey representation: state dimension ``r = max(p, q + 1)``, ``y_t = a_{1,t}``, innovation loading ``(1, theta_1, ..., theta_{r-1})``."""
    p, q = len(ar), len(ma)
    r = max(p, q + 1)
    T = np.zeros((r, r))
    T[: p, 0] = ar
    if r > 1:
        T[:-1, 1:] += np.eye(r - 1)
    R = np.zeros((r, 1))
    R[0, 0] = 1.0
    R[1: q + 1, 0] = ma
    Z = np.zeros((1, r))
    Z[0, 0] = 1.0
    return StateSpace(T, Z, [[0.0]], [[sigma2]], R=R)


def arma_loglik(w: np.ndarray, ar: np.ndarray, ma: np.ndarray, sigma2: float, tol: float = 1e-11) -> float:
    """Exact Gaussian log-likelihood of a zero-mean ARMA series, fast.

    The Kalman filter is run step by step until the prediction variance has reached its steady state ``sigma2`` for ``q + 1`` consecutive steps (geometrically fast
    for an invertible process); from there the innovations ARE the ARMA innovations and are computed for the rest of the sample in one vectorised recursion
    (``scipy.signal.lfilter``), seeded with the filter's own last innovations. Identical to the full filter to rounding error (checked in the tests).
    """
    p, q = len(ar), len(ma)
    model = arma_state_space(ar, ma, sigma2)
    T, R = model.T, model.R
    RQR = sigma2 * (R @ R.T)
    a = np.zeros(T.shape[0])
    P = model._initial_cov()
    n = len(w)
    ll = 0.0
    run = 0
    v_hist = []
    t = 0
    while t < n:
        F = P[0, 0]
        v = w[t] - a[0]
        K = P[:, 0] / F
        ll -= 0.5 * (np.log(2 * np.pi) + np.log(F) + v * v / F)
        v_hist.append(v)
        a = T @ (a + K * v)
        P = T @ (P - np.outer(K, P[0])) @ T.T + RQR
        t += 1
        run = run + 1 if abs(F - sigma2) < tol * sigma2 else 0
        if run > q and t >= max(p, q) + 1 and t < n:
            b_ = np.r_[1.0, -ar]
            a_ = np.r_[1.0, ma]
            zi = signal.lfiltic(b_, a_, y=v_hist[::-1][: max(len(a_) - 1, 1)], x=list(w[:t][::-1][: max(len(b_) - 1, 1)]))
            e, _ = signal.lfilter(b_, a_, w[t:], zi=zi)
            ll += -0.5 * (len(e) * np.log(2 * np.pi * sigma2) + float(e @ e) / sigma2)
            return float(ll)
    return float(ll)


@dataclass
class ARIMAResult:
    order: tuple
    mean: float
    ar: np.ndarray
    ma: np.ndarray
    sigma2: float
    loglik: float
    aic: float
    bic: float
    nobs: int
    residuals: pd.Series
    fitted: pd.Series
    _y: pd.Series
    _model: StateSpace

    def forecast(self, steps: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        """Point forecasts and ``1 - alpha`` intervals. For ``d > 0`` the differenced forecast is integrated and the variance accumulates through the MA(infinity) weights."""
        p, d, q = self.order
        w = self._y.to_numpy()
        for _ in range(d):
            w = np.diff(w)
        filt = self._model.filter(w - self.mean)
        fc = self._model.forecast(filt, steps)
        mean = np.atleast_1d(fc["mean"]) + self.mean
        var = np.atleast_1d(fc["var"]).astype(float)
        if d == 0:
            m, v = mean, var
        else:
            psi = self.psi_weights(steps)
            last = self._y.to_numpy()[-d:]
            m = mean.copy()
            # integrate d times
            levels = [self._y.to_numpy()]
            for _ in range(d - 1):
                levels.append(np.diff(levels[-1]))
            tails = [lv[-1] for lv in levels]
            inc = mean.copy()
            for j in range(d - 1, -1, -1):
                inc = tails[j] + np.cumsum(inc)
            m = inc
            cum = np.cumsum(psi)
            for _ in range(d - 1):
                cum = np.cumsum(cum)
            v = self.sigma2 * np.cumsum(cum ** 2)
            del last
        z = sps.norm.ppf(1 - alpha / 2)
        return pd.DataFrame({"mean": m, "lower": m - z * np.sqrt(v), "upper": m + z * np.sqrt(v), "se": np.sqrt(v)}, index=range(1, steps + 1))

    def psi_weights(self, n: int) -> np.ndarray:
        """MA(infinity) weights of the differenced process: the impulse response of ``y`` to a one-unit shock."""
        psi = np.zeros(n)
        psi[0] = 1.0
        for j in range(1, n):
            v = self.ma[j - 1] if j - 1 < len(self.ma) else 0.0
            for i in range(1, min(j, len(self.ar)) + 1):
                v += self.ar[i - 1] * psi[j - i]
            psi[j] = v
        return psi

    def summary(self) -> pd.Series:
        names = (["mean"] if self.order[1] == 0 else []) + [f"ar{i+1}" for i in range(len(self.ar))] + [f"ma{j+1}" for j in range(len(self.ma))] + ["sigma2"]
        vals = ([self.mean] if self.order[1] == 0 else []) + list(self.ar) + list(self.ma) + [self.sigma2]
        return pd.Series(vals, index=names)

    def ljung_box(self, lags: int = 10) -> dict:
        return ljung_box(self.residuals.dropna(), lags, model_df=len(self.ar) + len(self.ma))


def fit_arima(y: pd.Series, order=(1, 0, 0), trend: str = "c") -> ARIMAResult:
    """Exact ML estimate of an ARIMA(p, d, q). ``trend='c'`` includes a mean (only meaningful when ``d = 0``), ``'n'`` none."""
    p, d, q = order
    yy = y.dropna().astype(float)
    w = yy.to_numpy()
    for _ in range(d):
        w = np.diff(w)
    mean = trend == "c" and d == 0
    n = len(w)

    def nll(theta):
        mu, ar, ma, s2 = _unpack(theta, p, q, mean)
        if s2 <= 0 or not np.isfinite(s2):
            return 1e12
        try:
            return -arma_loglik(w - mu, ar, ma, s2)
        except (np.linalg.LinAlgError, ValueError):
            return 1e12

    x0 = np.zeros(int(mean) + p + q + 1)
    if mean:
        x0[0] = w.mean()
    x0[-1] = np.log(max(w.var(), 1e-12))
    best = None
    for start in (x0, x0 + np.r_[np.zeros(int(mean)), 0.3 * np.ones(p + q), 0.0]):
        res = optimize.minimize(nll, start, method="BFGS", options={"gtol": 1e-6, "maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res
    mu, ar, ma, s2 = _unpack(best.x, p, q, mean)
    model = arma_state_space(ar, ma, s2)
    filt = model.filter(w - mu)
    k = int(mean) + p + q + 1
    ll = filt.loglik
    resid = pd.Series(filt.innovations[:, 0] / np.sqrt(filt.innovation_var[:, 0, 0]), index=yy.index[d:])
    fitted = pd.Series(w - mu - filt.innovations[:, 0] + mu, index=yy.index[d:])
    return ARIMAResult(order, float(mu), ar, ma, s2, ll, -2 * ll + 2 * k, -2 * ll + np.log(n) * k, n, resid, fitted, yy, model)


def select_arima(y: pd.Series, max_p: int = 3, max_q: int = 3, d: int = 0, criterion: str = "bic", trend: str = "c") -> pd.DataFrame:
    """Fit every ARMA(p, q) up to the limits and rank by ``criterion`` ('aic' or 'bic'). Returns the table, best first."""
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                r = fit_arima(y, (p, d, q), trend)
            except Exception:
                continue
            rows.append({"p": p, "d": d, "q": q, "loglik": r.loglik, "aic": r.aic, "bic": r.bic})
    return pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)


def ar_ols(y, p: int = 1, trend: str = "c") -> dict:
    """AR(p) by least squares on the lagged values (conditional on the first ``p`` observations)."""
    x = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    rows = np.column_stack([x[p - j: len(x) - j] for j in range(1, p + 1)])
    Z = np.column_stack([np.ones(len(rows)), rows]) if trend == "c" else rows
    target = x[p:]
    beta = np.linalg.lstsq(Z, target, rcond=None)[0]
    resid = target - Z @ beta
    s2 = float(resid @ resid) / (len(target) - Z.shape[1])
    cov = s2 * np.linalg.inv(Z.T @ Z)
    return {"const": float(beta[0]) if trend == "c" else 0.0, "ar": beta[1:] if trend == "c" else beta, "sigma2": s2, "se": np.sqrt(np.diag(cov)), "resid": resid}


def yule_walker(y, p: int = 1) -> dict:
    """AR(p) from the Yule-Walker equations ``R phi = r`` using the sample autocovariances; always stationary."""
    x = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    n = len(x)
    gamma = np.array([x[: n - k] @ x[k:] / n for k in range(p + 1)])
    R = np.array([[gamma[abs(i - j)] for j in range(p)] for i in range(p)])
    phi = np.linalg.solve(R, gamma[1: p + 1])
    return {"ar": phi, "sigma2": float(gamma[0] - phi @ gamma[1: p + 1])}


def rolling_forecasts(y: pd.Series, order=(1, 0, 0), min_train: int = 500, refit_every: int = 21, window: int | None = None, trend: str = "c") -> pd.DataFrame:
    """One-step-ahead forecasts at every date after ``min_train``, refitting every ``refit_every`` days on data up to the forecast origin only.

    The forecast stored at date ``t`` is for ``y_{t+1}`` and is indexed by ``t + 1`` (the day it applies to). Returns ``forecast``, ``actual`` and ``se``.
    """
    yy = y.dropna()
    n = len(yy)
    rows = {}
    model = None
    for t in range(min_train, n - 1):
        if model is None or (t - min_train) % refit_every == 0:
            start = 0 if window is None else max(0, t - window)
            model = fit_arima(yy.iloc[start: t + 1], order, trend)
        else:
            model._y = yy.iloc[: t + 1] if window is None else yy.iloc[max(0, t - window): t + 1]
        f = model.forecast(1)
        rows[yy.index[t + 1]] = {"forecast": float(f["mean"].iloc[0]), "se": float(f["se"].iloc[0]), "actual": float(yy.iloc[t + 1])}
    return pd.DataFrame(rows).T
