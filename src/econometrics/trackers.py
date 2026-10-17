r"""Two scalar Kalman filters that track what a fixed-window estimate cannot: how fast a spread is reverting right now, and how volatile a return series is right now.

``kalman_ar1``     a time-varying AR(1) coefficient. For a demeaned series ``z`` the model is ``z_t = phi_t z_{t-1} + e_t`` with ``phi_t`` a random walk: the filter's ``phi_t`` is the
                   reversion speed of a spread (``theta_t = -ln(phi_t) / dt``, half-life ``ln 2 / theta_t``) estimated from the data through ``t`` only, moving when the pair's behaviour
                   changes. The two noise variances are fitted by maximum likelihood of the innovations.
``sv_tracker``     a stochastic-volatility model of returns by the quasi-maximum-likelihood method of Harvey, Ruiz and Shephard (1994). With ``h_t = ln(sigma_t^2)`` following an AR(1),
                   ``h_t = mu + phi (h_{t-1} - mu) + eta_t``, and a return ``r_t = sigma_t z_t``, the log square ``y_t = ln(r_t^2) = h_t + ln(z_t^2)`` is the log variance plus noise whose
                   mean is ``-1.2704`` and variance ``pi^2 / 2`` (the log of a chi-square with one degree of freedom): treat the noise as Gaussian with those moments and the Kalman filter gives
                   the best linear estimate of ``h_t``. The three parameters are fitted by maximising the Gaussian likelihood of the innovations. The filter is causal: ``sigma_t`` uses the
                   returns through ``t``. Compared with GARCH and exponentially weighted volatility it has no leverage term but an explicit measure of its own uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

LOG_CHI2_MEAN = -1.2703628454614782                        # E[ln chi2_1] = -gamma - ln 2
LOG_CHI2_VAR = np.pi ** 2 / 2.0


# ------------------------------------------------------------------------------------------------------------------------------ AR(1) tracker
@dataclass
class AR1Track:
    """The filtered AR(1) coefficient through time: ``phi`` and its standard deviation ``phi_sd``, the one-step prediction of each observation and its standard deviation, and the standardised
    innovations (which should look like noise when the model holds). ``q`` is the variance of the random walk of ``phi`` per step, ``r`` the observation noise variance."""

    phi: np.ndarray
    phi_sd: np.ndarray
    prediction: np.ndarray
    prediction_sd: np.ndarray
    innovation: np.ndarray
    q: float
    r: float
    loglik: float

    def theta(self, dt: float = 1.0 / 252.0) -> np.ndarray:
        """The reversion speed ``-ln(phi) / dt`` (NaN where ``phi`` is outside ``(0, 1)``: no reversion then)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((self.phi > 0) & (self.phi < 1), -np.log(self.phi) / dt, np.nan)

    def half_life(self, dt: float = 1.0 / 252.0) -> np.ndarray:
        return np.log(2.0) / self.theta(dt)


def _ar1_filter(z: np.ndarray, q: float, r: float, phi0: float, p0: float):
    n = len(z)
    phi, p = phi0, p0
    out_phi, out_sd = np.full(n, np.nan), np.full(n, np.nan)
    pred, pred_sd, innov = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    ll = 0.0
    for t in range(1, n):
        x = z[t - 1]
        if not (np.isfinite(x) and np.isfinite(z[t])):
            p = p + q
            out_phi[t], out_sd[t] = phi, np.sqrt(p)
            continue
        p_pred = p + q
        f = x * x * p_pred + r
        v = z[t] - phi * x
        pred[t], pred_sd[t], innov[t] = phi * x, np.sqrt(f), v / np.sqrt(f)
        ll += -0.5 * (np.log(2.0 * np.pi * f) + v * v / f)
        k = p_pred * x / f
        phi = phi + k * v
        p = (1.0 - k * x) * p_pred
        out_phi[t], out_sd[t] = phi, np.sqrt(max(p, 0.0))
    return out_phi, out_sd, pred, pred_sd, innov, ll


def kalman_ar1(series, q: float | None = None, r: float | None = None, phi0: float = 0.95, p0: float = 0.05, demean_window: int | None = None) -> AR1Track:
    """Track the AR(1) coefficient of ``series`` (a spread). ``series`` is demeaned first: by its expanding mean through each date (causal) unless ``demean_window`` asks for a rolling mean.
    ``q`` (the variance of the random walk of the coefficient) and ``r`` (the noise variance) are fitted by maximum likelihood unless given."""
    x = np.asarray(series, dtype=float)
    if len(x) < 30:
        raise ValueError("at least 30 observations are needed")
    s = pd.Series(x)
    mean = s.expanding(min_periods=5).mean() if demean_window is None else s.rolling(demean_window, min_periods=5).mean()
    z = (s - mean).to_numpy()
    scale = float(np.nanstd(z)) or 1.0
    zs = z / scale                                                         # unit scale, so the same starting values suit any series
    if q is None or r is None:
        def nll(p):
            qq, rr = np.exp(p)
            return -_ar1_filter(zs, qq, rr, phi0, p0)[-1]

        start = np.log([1e-5 if q is None else q, 0.05 if r is None else r / scale ** 2])
        res = optimize.minimize(nll, start, method="L-BFGS-B", bounds=[(np.log(1e-10), np.log(1e-1)), (np.log(1e-4), np.log(10.0))])
        q_hat, r_hat = np.exp(res.x)
        q_use = q_hat if q is None else q
        r_use = r_hat if r is None else r / scale ** 2
    else:
        q_use, r_use = q, r / scale ** 2
    phi, sd, pred, pred_sd, innov, ll = _ar1_filter(zs, q_use, r_use, phi0, p0)
    return AR1Track(phi, sd, pred * scale, pred_sd * scale, innov, float(q_use), float(r_use * scale ** 2), float(ll - (len(zs) - 1) * np.log(scale)))


# ------------------------------------------------------------------------------------------------------------------------------- SV tracker
@dataclass
class SVTrack:
    """A fitted stochastic-volatility model and the filter's output: ``mu`` (the mean log variance), ``phi`` (its persistence), ``sigma_eta`` (the volatility of log variance), the filtered log
    variance ``h`` and its variance ``p`` at each date, the daily volatility ``vol`` (the filtered ``exp(h/2)``, corrected for the uncertainty in ``h``) and the log-likelihood."""

    mu: float
    phi: float
    sigma_eta: float
    h: np.ndarray
    p: np.ndarray
    vol: np.ndarray
    loglik: float
    n: int
    converged: bool
    offset: float = field(default=0.0)

    def params(self) -> dict:
        return {"mu": self.mu, "phi": self.phi, "sigma_eta": self.sigma_eta}

    def forecast_variance(self, steps: int) -> np.ndarray:
        """The expected daily variance ``E[sigma^2_{T+k} | returns through T]`` for ``k = 1 .. steps``: the log variance reverts to ``mu`` at rate ``phi`` and the lognormal convexity
        ``exp(mean + variance / 2)`` is applied, with the variance of the state forecast (the filter's uncertainty carried forward plus the shocks to come)."""
        h, p = self.h[-1], self.p[-1]
        k = np.arange(1, steps + 1)
        mean = self.mu + self.phi ** k * (h - self.mu)
        var = self.phi ** (2 * k) * p + self.sigma_eta ** 2 * (1.0 - self.phi ** (2 * k)) / (1.0 - self.phi ** 2)
        return np.exp(mean + 0.5 * var)

    def simulate_log_variance(self, steps: int, n_paths: int, seed=0) -> np.ndarray:
        """Draws of the future log variance ``h_{T+1..T+steps}`` given the data: the state's uncertainty at ``T``, then the AR(1) shocks. Shape ``(n_paths, steps)``."""
        rng = np.random.default_rng(seed)
        h = rng.normal(self.h[-1], np.sqrt(max(self.p[-1], 0.0)), n_paths)
        out = np.empty((n_paths, steps))
        for i in range(steps):
            h = self.mu + self.phi * (h - self.mu) + self.sigma_eta * rng.standard_normal(n_paths)
            out[:, i] = h
        return out


def _sv_filter(y: np.ndarray, mu: float, phi: float, s_eta: float):
    n = len(y)
    h_f, p_f = np.empty(n), np.empty(n)
    h, p = mu, s_eta ** 2 / max(1.0 - phi ** 2, 1e-8)                        # the stationary distribution of the state
    ll = 0.0
    for t in range(n):
        h_pred = mu + phi * (h - mu) if t else h
        p_pred = phi ** 2 * p + s_eta ** 2 if t else p
        f = p_pred + LOG_CHI2_VAR
        v = y[t] - LOG_CHI2_MEAN - h_pred
        ll += -0.5 * (np.log(2.0 * np.pi * f) + v * v / f)
        k = p_pred / f
        h = h_pred + k * v
        p = (1.0 - k) * p_pred
        h_f[t], p_f[t] = h, p
    return h_f, p_f, ll


def sv_tracker(returns, start: tuple[float, float, float] | None = None) -> SVTrack:
    """Fit the stochastic-volatility model to daily ``returns`` by quasi-maximum likelihood and filter the log variance. Returns that are exactly zero (a closed market) are given the log square of
    a small fraction of the sample variance so that the logarithm exists (Fuller's offset)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        raise ValueError("at least 100 returns are needed")
    var = float(np.var(r))
    offset = 1e-4 * var                                                    # Fuller's offset: ln(r^2 + offset)
    y = np.log(r ** 2 + offset)
    mu0 = float(np.log(var))
    if start is None:
        start = (mu0, 0.95, 0.2)

    def nll(p):
        mu, phi_raw, s_log = p
        phi = np.tanh(phi_raw)
        return -_sv_filter(y, mu, phi, np.exp(s_log))[2]

    x0 = [start[0], np.arctanh(np.clip(start[1], -0.999, 0.999)), np.log(start[2])]
    res = optimize.minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 800})
    mu, phi, s_eta = float(res.x[0]), float(np.tanh(res.x[1])), float(np.exp(res.x[2]))
    h, p, ll = _sv_filter(y, mu, phi, s_eta)
    vol = np.exp(0.5 * h + 0.125 * p)                                      # E[exp(h/2)] under the filtered normal law of h
    return SVTrack(mu, phi, s_eta, h, p, vol, float(ll), len(r), bool(res.success), offset)
