"""Conditional-volatility models by maximum likelihood: GARCH, GJR-GARCH, TGARCH (Zakoian) and EGARCH with Gaussian or Student-t innovations.

``r_t = mu + e_t``, ``e_t = sigma_t z_t``, ``z_t`` unit-variance Gaussian or Student-t(nu). The variance equations (all first order here):

``garch``  ``sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}``                                       (Bollerslev 1986)
``gjr``    ``sigma2_t = omega + (alpha + gamma 1[e_{t-1} < 0]) e_{t-1}^2 + beta sigma2_{t-1}``               (Glosten, Jagannathan & Runkle 1993)
``tgarch`` ``sigma_t = omega + (alpha + gamma 1[e_{t-1} < 0]) |e_{t-1}| + beta sigma_{t-1}``                (Zakoian 1994: the threshold model on the standard deviation, GJR's power-one cousin)
``egarch`` ``ln sigma2_t = omega + alpha (|z_{t-1}| - sqrt(2/pi)) + gamma z_{t-1} + beta ln sigma2_{t-1}``  (Nelson 1991; no positivity constraints)

Leverage: ``gamma > 0`` in GJR and TGARCH (or ``gamma < 0`` in EGARCH) means a fall raises volatility more than an equal rise. The recursions are started from the same
exponentially weighted back-cast of ``e^2`` (``|e|`` for TGARCH) that ``arch`` uses, so the fitted likelihoods agree with that library to optimiser tolerance (checked in the tests).

Everything that goes into a forecast for date ``t + 1`` uses returns up to ``t``: ``walk_forward_volatility`` refits on an expanding window and compares the forecasts
with the EWMA baseline on QLIKE, the loss that ranks variance forecasts consistently (Patton 2011).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special

SQRT2PI = math.sqrt(2.0 / math.pi)
MODELS = ("garch", "gjr", "tgarch", "egarch")


def mean_abs_innovation(dist: str, nu: float = float("inf")) -> float:
    """``E|z|`` for a unit-variance innovation: ``sqrt(2/pi)`` for the normal, ``sqrt((nu - 2)/pi) Gamma((nu - 1)/2) / Gamma(nu/2)`` for the Student-t."""
    if dist == "normal" or not np.isfinite(nu):
        return SQRT2PI
    return float(math.sqrt((nu - 2.0) / math.pi) * math.exp(special.gammaln((nu - 1.0) / 2.0) - special.gammaln(nu / 2.0)))


def tgarch_moments(alpha: float, gamma: float, beta: float, m1: float) -> tuple[float, float]:
    """``E[c]`` and ``E[c^2]`` of the random multiplier ``c = (alpha + gamma 1[z < 0]) |z| + beta`` in ``sigma_t = omega + c_{t-1} sigma_{t-1}``, for symmetric ``z``
    with ``E|z| = m1`` and ``E z^2 = 1``. The variance of the returns is finite when ``E[c^2] < 1``."""
    ec = beta + (alpha + 0.5 * gamma) * m1
    ec2 = beta ** 2 + 2.0 * beta * (alpha + 0.5 * gamma) * m1 + alpha ** 2 + alpha * gamma + 0.5 * gamma ** 2
    return ec, ec2


def backcast(e2: np.ndarray, tau: int = 75, decay: float = 0.94) -> float:
    """Exponentially weighted average of the first ``tau`` squared residuals: the pre-sample variance used to start the recursion."""
    m = min(tau, len(e2))
    w = decay ** np.arange(m)
    return float(np.dot(w / w.sum(), e2[:m]))


def _variance_path(model: str, e: np.ndarray, omega: float, alpha: float, gamma: float, beta: float, bc: float) -> np.ndarray:
    n = len(e)
    s2 = np.empty(n)
    if model == "egarch":
        ls2 = math.log(bc)
        z_prev = 0.0
        abs_prev = SQRT2PI                                       # the pre-sample innovation contributes nothing: |z| - E|z| = 0
        for t in range(n):
            ls2 = omega + alpha * (abs_prev - SQRT2PI) + gamma * z_prev + beta * ls2 if t else omega + beta * ls2
            ls2 = min(max(ls2, -50.0), 50.0)
            s2[t] = math.exp(ls2)
            z_prev = e[t] / math.sqrt(s2[t])
            abs_prev = abs(z_prev)
        return s2
    if model == "tgarch":                                         # recursion on sigma; ``bc`` is the back-cast of |e|
        prev_sig, prev_abs, prev_neg = bc, bc, 0.5
        for t in range(n):
            sig = omega + (alpha + gamma * prev_neg) * prev_abs + beta * prev_sig
            sig = max(sig, 1e-12)
            s2[t] = sig * sig
            prev_sig, prev_abs, prev_neg = sig, abs(e[t]), 1.0 if e[t] < 0 else 0.0
        return s2
    prev_s2, prev_e2, prev_neg = bc, bc, 0.5                      # pre-sample: back-cast variance, half of it counted as "negative" (arch convention)
    for t in range(n):
        s2[t] = omega + (alpha + gamma * prev_neg) * prev_e2 + beta * prev_s2 if model == "gjr" else omega + alpha * prev_e2 + beta * prev_s2
        prev_s2 = s2[t]
        prev_e2 = e[t] * e[t]
        prev_neg = 1.0 if e[t] < 0 else 0.0
    return s2


def _backcast_for(model: str, e: np.ndarray) -> float:
    return backcast(np.abs(e)) if model == "tgarch" else backcast(e * e)


def _loglik_terms(e: np.ndarray, s2: np.ndarray, dist: str, nu: float) -> np.ndarray:
    if dist == "normal":
        return -0.5 * (np.log(2 * np.pi) + np.log(s2) + e * e / s2)
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(e * e / (s2 * (nu - 2)))


@dataclass
class GARCHResult:
    model: str
    dist: str
    params: dict
    loglik: float
    aic: float
    bic: float
    nobs: int
    volatility: pd.Series              # conditional standard deviation sigma_t (per period), causal: sigma_t depends on returns up to t-1
    std_resid: pd.Series
    scale: float
    _r: pd.Series
    _bc: float

    @property
    def persistence(self) -> float:
        p = self.params
        if self.model == "garch":
            return p["alpha"] + p["beta"]
        if self.model == "gjr":
            return p["alpha"] + 0.5 * p["gamma"] + p["beta"]
        if self.model == "tgarch":                                  # decay of the expected standard deviation
            return tgarch_moments(p["alpha"], p["gamma"], p["beta"], mean_abs_innovation(self.dist, p.get("nu", float("inf"))))[0]
        return p["beta"]

    @property
    def unconditional_variance(self) -> float:
        p = self.params
        if self.model == "egarch":
            return float("nan")
        if self.model == "tgarch":
            ec, ec2 = tgarch_moments(p["alpha"], p["gamma"], p["beta"], mean_abs_innovation(self.dist, p.get("nu", float("inf"))))
            return (p["omega"] ** 2 * (1.0 + 2.0 * ec / (1.0 - ec))) / (1.0 - ec2) if ec2 < 1 and ec < 1 else float("inf")
        return p["omega"] / (1.0 - self.persistence) if self.persistence < 1 else float("inf")

    @property
    def half_life(self) -> float:
        """Periods for a volatility shock to decay by half: ``ln 0.5 / ln persistence``."""
        return float(math.log(0.5) / math.log(self.persistence)) if 0 < self.persistence < 1 else float("inf")

    def forecast_variance(self, steps: int = 1, n_sim: int = 5000, seed: int = 0) -> np.ndarray:
        """Forecast of ``sigma2_{T+1} .. sigma2_{T+steps}`` given returns up to ``T`` (in the units of the input). GARCH and GJR use the analytical recursion,
        EGARCH simulation."""
        p = self.params
        e = (self._r.to_numpy() - p["mu"]) / self.scale
        pr = self._params_scaled()
        s2 = _variance_path(self.model, e, pr["omega"], pr["alpha"], pr.get("gamma", 0.0), pr["beta"], self._bc)
        last_e2, last_neg, last_s2 = e[-1] ** 2, float(e[-1] < 0), s2[-1]
        if self.model == "tgarch":
            rng = np.random.default_rng(seed)
            sig = np.full(n_sim, pr["omega"] + (pr["alpha"] + pr["gamma"] * last_neg) * abs(e[-1]) + pr["beta"] * math.sqrt(last_s2))
            out = np.zeros(steps)
            for h in range(steps):
                out[h] = float(np.mean(sig ** 2))
                z = rng.standard_normal(n_sim) if self.dist == "normal" else rng.standard_t(p["nu"], n_sim) * math.sqrt((p["nu"] - 2) / p["nu"])
                sig = pr["omega"] + (pr["alpha"] + pr["gamma"] * (z < 0)) * np.abs(z) * sig + pr["beta"] * sig
            return out * self.scale ** 2
        if self.model == "egarch":
            rng = np.random.default_rng(seed)
            out = np.zeros(steps)
            ls2 = np.full(n_sim, math.log(last_s2))
            z = np.full(n_sim, e[-1] / math.sqrt(last_s2))
            for h in range(steps):
                ls2 = pr["omega"] + pr["alpha"] * (np.abs(z) - SQRT2PI) + pr["gamma"] * z + pr["beta"] * ls2
                out[h] = np.exp(ls2).mean()
                z = rng.standard_normal(n_sim) if self.dist == "normal" else rng.standard_t(p["nu"], n_sim) * math.sqrt((p["nu"] - 2) / p["nu"])
            return out * self.scale ** 2
        nxt = pr["omega"] + (pr["alpha"] + (pr.get("gamma", 0.0) * last_neg)) * last_e2 + pr["beta"] * last_s2
        persist = pr["alpha"] + 0.5 * pr.get("gamma", 0.0) + pr["beta"]
        sbar = pr["omega"] / (1 - persist) if persist < 1 else nxt
        out = np.array([sbar + persist ** h * (nxt - sbar) for h in range(steps)])
        return out * self.scale ** 2

    def _params_scaled(self) -> dict:
        s = self.scale
        out = dict(self.params)
        if self.model == "tgarch":
            out["omega"] = out["omega"] / s
        else:
            out["omega"] = out["omega"] / s ** 2 if self.model != "egarch" else out["omega"] - 2 * math.log(s) * (1 - out["beta"])
        return out

    def news_impact(self, shocks: np.ndarray | None = None) -> pd.Series:
        """Next-period variance as a function of today's shock ``e`` with yesterday's variance at its unconditional (or last) level: the asymmetry curve."""
        e = np.linspace(-4, 4, 81) * float(np.sqrt(self.volatility.var() + self.volatility.mean() ** 2)) if shocks is None else np.asarray(shocks, float)
        p = self.params
        s2bar = float(self.volatility.mean() ** 2)
        if self.model == "garch":
            v = p["omega"] + p["alpha"] * e ** 2 + p["beta"] * s2bar
        elif self.model == "gjr":
            v = p["omega"] + (p["alpha"] + p["gamma"] * (e < 0)) * e ** 2 + p["beta"] * s2bar
        elif self.model == "tgarch":
            v = (p["omega"] + (p["alpha"] + p["gamma"] * (e < 0)) * np.abs(e) + p["beta"] * math.sqrt(s2bar)) ** 2
        else:
            z = e / math.sqrt(s2bar)
            v = np.exp(p["omega"] + p["alpha"] * (np.abs(z) - SQRT2PI) + p["gamma"] * z + p["beta"] * math.log(s2bar))
        return pd.Series(v, index=pd.Index(e, name="shock"))


def fit_garch(returns: pd.Series, model: str = "garch", dist: str = "normal", mean: str = "constant") -> GARCHResult:
    """Maximum-likelihood fit. ``model``: ``garch``, ``gjr``, ``tgarch`` or ``egarch``; ``dist``: ``normal`` or ``t``; ``mean``: ``constant`` or ``zero``.

    Returns are rescaled internally to unit standard deviation (the optimiser works far better there) and the parameters are reported in the INPUT units.
    """
    if model not in MODELS or dist not in ("normal", "t") or mean not in ("constant", "zero"):
        raise ValueError("model in {garch, gjr, tgarch, egarch}, dist in {normal, t}, mean in {constant, zero}")
    r = returns.dropna().astype(float)
    scale = float(r.std())
    x = r.to_numpy() / scale
    n = len(x)
    has_gamma = model in ("gjr", "tgarch", "egarch")

    def split(th):
        i = 0
        mu = th[i] if mean == "constant" else 0.0
        i += int(mean == "constant")
        omega, alpha = th[i], th[i + 1]
        i += 2
        gamma = th[i] if has_gamma else 0.0
        i += int(has_gamma)
        beta = th[i]
        i += 1
        nu = th[i] if dist == "t" else np.inf
        return mu, omega, alpha, gamma, beta, nu

    def nll(th):
        mu, omega, alpha, gamma, beta, nu = split(th)
        e = x - mu
        bc = _backcast_for(model, e)
        s2 = _variance_path(model, e, omega, alpha, gamma, beta, bc)
        if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
            return 1e12
        return -float(np.sum(_loglik_terms(e, s2, dist, nu)))

    # starting values and bounds
    mu0 = float(x.mean()) if mean == "constant" else None
    if model == "egarch":
        th0 = [0.0, 0.1, -0.05, 0.95]
        bounds = [(-1.0, 1.0), (-0.5, 1.0), (-1.0, 1.0), (0.0, 0.9999)]
    elif model == "tgarch":
        th0 = [0.03, 0.03, 0.05, 0.9]
        bounds = [(1e-8, 2.0), (0.0, 1.0), (-0.5, 1.0), (0.0, 0.9999)]
    elif model == "gjr":
        th0 = [0.05, 0.03, 0.05, 0.9]
        bounds = [(1e-8, 5.0), (0.0, 1.0), (-0.5, 1.0), (0.0, 0.9999)]
    else:
        th0 = [0.05, 0.05, 0.9]
        bounds = [(1e-8, 5.0), (0.0, 1.0), (0.0, 0.9999)]
    if mean == "constant":
        th0, bounds = [mu0] + th0, [(-1.0, 1.0)] + bounds
    if dist == "t":
        th0, bounds = th0 + [8.0], bounds + [(2.05, 200.0)]

    def persistence_ok(th):
        _, _, alpha, gamma, beta, nu_ = split(th)
        if model == "egarch":
            return 0.9999 - beta
        if model == "tgarch":
            return 0.9999 - tgarch_moments(alpha, gamma, beta, mean_abs_innovation(dist, nu_))[1]
        return 0.9999 - (alpha + max(gamma, 0.0) * 0.5 + beta) if model == "gjr" else 0.9999 - (alpha + beta)

    cons = [{"type": "ineq", "fun": persistence_ok}]
    best = None
    n_tail = int(dist == "t")
    for beta_start, alpha_start in ((None, None), (0.8, 0.1)):                     # two starts guard against a flat or ridge-shaped likelihood
        st = list(th0)
        if beta_start is not None:
            st[len(st) - 1 - n_tail] = beta_start
            st[len(st) - 2 - n_tail - int(has_gamma)] = alpha_start if model != "egarch" else 0.2
        res = optimize.minimize(nll, st, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    mu, omega, alpha, gamma, beta, nu = split(best.x)
    e = x - mu
    bc = _backcast_for(model, e)
    s2 = _variance_path(model, e, omega, alpha, gamma, beta, bc)
    ll = -best.fun - n * math.log(scale)                         # likelihood of the ORIGINAL series (Jacobian of the rescaling)
    k = len(best.x)
    out_omega = omega * scale if model == "tgarch" else omega * scale ** 2 if model != "egarch" else omega + 2 * math.log(scale) * (1 - beta)
    params = {"mu": mu * scale, "omega": out_omega, "alpha": alpha, "beta": beta}
    if has_gamma:
        params["gamma"] = gamma
    if dist == "t":
        params["nu"] = nu
    vol = pd.Series(np.sqrt(s2) * scale, index=r.index, name="volatility")
    z = pd.Series(e / np.sqrt(s2), index=r.index, name="std_resid")
    return GARCHResult(model, dist, params, float(ll), float(-2 * ll + 2 * k), float(-2 * ll + math.log(n) * k), n, vol, z, scale, r, bc)


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility for date ``t`` from returns up to ``t - 1`` (a causal one-step forecast)."""
    r = returns.dropna()
    var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var).shift(1)


def qlike(realised_var, forecast_var) -> float:
    """Mean QLIKE loss ``rv / f - ln(rv / f) - 1``. A consistent loss for ranking variance forecasts when the proxy (here ``r^2``) is noisy; lower is better."""
    rv, f = np.asarray(realised_var, float), np.asarray(forecast_var, float)
    ok = np.isfinite(rv) & np.isfinite(f) & (rv > 0) & (f > 0)
    ratio = rv[ok] / f[ok]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def walk_forward_volatility(returns: pd.Series, model: str = "garch", dist: str = "t", min_train: int = 750, refit_every: int = 63) -> pd.DataFrame:
    """Causal one-step volatility forecasts. Every ``refit_every`` days the model is refitted on all data up to the origin; between refits the variance recursion is
    updated daily with the fixed parameters. The forecast for date ``t + 1`` uses returns up to ``t`` only (the recursion is evaluated once per block, so the cost is
    linear in the sample length). Returns ``forecast`` (sigma for the date in the index), the EWMA baseline and the squared return."""
    r = returns.dropna()
    n = len(r)
    x = r.to_numpy()
    fc = np.full(n, np.nan)
    for start in range(min_train, n, refit_every):
        fit = fit_garch(r.iloc[:start], model, dist)
        pr = fit._params_scaled()
        e = (x - fit.params["mu"]) / fit.scale
        s2 = _variance_path(model, e, pr["omega"], pr["alpha"], pr.get("gamma", 0.0), pr["beta"], fit._bc)      # sigma2 for date t from returns up to t - 1
        end = min(start + refit_every, n)
        fc[start:end] = np.sqrt(s2[start:end]) * fit.scale
    out = pd.DataFrame({"forecast": fc, "ewma": ewma_volatility(r), "realised_sq": r ** 2}, index=r.index)
    return out.iloc[min_train:]
