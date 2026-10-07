"""Extreme value theory for tail risk (McNeil, Frey & Embrechts, *Quantitative Risk Management*, ch. 5).

Value-at-risk at 99% or 99.9% sits where there are almost no observations, so the empirical quantile is noisy and cannot extrapolate beyond the sample.
EVT replaces the unknowable body of the distribution with a theorem about its tail:

* **Peaks over threshold** (Pickands-Balkema-de Haan): excesses over a high threshold ``u`` converge to a generalised Pareto distribution
  ``G(y) = 1 - (1 + xi y / beta)^(-1/xi)``. ``xi > 0`` is a heavy (power-law) tail; equity-index losses typically give ``xi`` around 0.1-0.3.
* **Block maxima** (Fisher-Tippett-Gnedenko): the maxima of blocks converge to a generalised extreme value distribution (``fit_gev``).
* **Hill estimator**: the tail index from the top order statistics, for power-law tails.

``fit_gpd`` is maximum likelihood (Grimshaw's one-dimensional reduction, with probability-weighted moments as the start). ``gpd_var_es`` turns the fit into
VaR and expected shortfall; ``dynamic_evt_var`` filters volatility first (McNeil-Frey 2000) so the tail is fitted to roughly i.i.d. residuals;
``threshold_diagnostics`` shows how stable the shape is across thresholds, which is how a threshold should be chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps


@dataclass
class GPDFit:
    xi: float
    beta: float
    threshold: float
    n_exceed: int
    n_total: int
    xi_se: float
    beta_se: float
    loglik: float

    @property
    def tail_fraction(self) -> float:
        return self.n_exceed / self.n_total

    def quantile(self, p: float) -> float:
        """The p-quantile (p > 1 - tail_fraction) of the loss distribution."""
        return gpd_quantile(p, self)

    def var(self, p: float = 0.99) -> float:
        return gpd_quantile(p, self)

    def es(self, p: float = 0.99) -> float:
        return gpd_var_es(self, p)["es"]


def _gpd_nll(params, y):
    xi, beta = params
    if beta <= 0:
        return np.inf
    z = 1.0 + xi * y / beta
    if np.any(z <= 0):
        return np.inf
    if abs(xi) < 1e-8:
        return len(y) * np.log(beta) + y.sum() / beta
    return len(y) * np.log(beta) + (1.0 + 1.0 / xi) * np.log(z).sum()


def fit_gpd_excesses(y: np.ndarray) -> tuple[float, float, float, float, float]:
    """MLE of the GPD to positive excesses ``y``. Returns ``(xi, beta, xi_se, beta_se, loglik)``. Start from probability-weighted moments (Hosking & Wallis 1987)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    ys = np.sort(y)
    b0 = ys.mean()
    b1 = np.sum((np.arange(n) / (n - 1.0)) * ys) / n
    xi0 = 2.0 - b0 / (b0 - 2.0 * b1)
    beta0 = 2.0 * b0 * b1 / (b0 - 2.0 * b1)
    xi0 = float(np.clip(xi0, -0.4, 0.9))
    beta0 = float(max(beta0, 1e-8))
    best = None
    for start in ([xi0, beta0], [0.1, y.mean()], [0.3, y.mean()]):
        res = optimize.minimize(_gpd_nll, start, args=(y,), method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    xi, beta = best.x
    # observed-information standard errors from a numerical Hessian
    h = np.array([1e-4, 1e-4 * max(beta, 1e-6)])
    hess = np.zeros((2, 2))
    f0 = _gpd_nll(best.x, y)
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
            hess[i, j] = (_gpd_nll(best.x + ei + ej, y) - _gpd_nll(best.x + ei - ej, y) - _gpd_nll(best.x - ei + ej, y) + _gpd_nll(best.x - ei - ej, y)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(hess)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    return float(xi), float(beta), float(se[0]), float(se[1]), float(-f0)


def fit_gpd(losses, threshold: float | None = None, quantile: float = 0.95) -> GPDFit:
    """Peaks-over-threshold fit to LOSSES (positive = loss). The threshold is the ``quantile`` of the losses unless given."""
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    u = float(np.quantile(x, quantile)) if threshold is None else float(threshold)
    exc = x[x > u] - u
    if len(exc) < 20:
        raise ValueError(f"only {len(exc)} exceedances: lower the threshold or use more data")
    xi, beta, se_xi, se_beta, ll = fit_gpd_excesses(exc)
    return GPDFit(xi, beta, u, len(exc), len(x), se_xi, se_beta, ll)


def gpd_quantile(p: float, fit: GPDFit) -> float:
    """``VaR_p = u + (beta/xi) [ ((1-p)/(Nu/N))^(-xi) - 1 ]`` (xi = 0: ``u - beta ln((1-p)/(Nu/N))``)."""
    ratio = (1.0 - p) / fit.tail_fraction
    if abs(fit.xi) < 1e-8:
        return fit.threshold - fit.beta * np.log(ratio)
    return fit.threshold + fit.beta / fit.xi * (ratio ** (-fit.xi) - 1.0)


def gpd_var_es(fit: GPDFit, p: float = 0.99) -> dict:
    """VaR and expected shortfall at ``p``: ``ES = (VaR + beta - xi u) / (1 - xi)`` for ``xi < 1`` (the mean is infinite otherwise)."""
    var = gpd_quantile(p, fit)
    es = (var + fit.beta - fit.xi * fit.threshold) / (1.0 - fit.xi) if fit.xi < 1 else float("inf")
    return {"var": float(var), "es": float(es), "xi": fit.xi, "p": p}


def profile_var_ci(losses, p: float = 0.99, threshold_quantile: float = 0.95, alpha: float = 0.05, n_boot: int = 300, seed: int = 0) -> dict:
    """Parametric-bootstrap interval for EVT VaR and ES: refit the GPD to excesses simulated from the fitted model (and resampled exceedance counts)."""
    fit = fit_gpd(losses, quantile=threshold_quantile)
    rng = np.random.default_rng(seed)
    vs, es = [], []
    for _ in range(n_boot):
        k = rng.binomial(fit.n_total, fit.tail_fraction)
        if k < 20:
            continue
        exc = sps.genpareto.rvs(fit.xi, scale=fit.beta, size=k, random_state=rng)
        xi, beta, *_ = fit_gpd_excesses(exc)
        f = GPDFit(xi, beta, fit.threshold, k, fit.n_total, 0, 0, 0)
        out = gpd_var_es(f, p)
        vs.append(out["var"])
        es.append(out["es"])
    lo, hi = np.quantile(vs, [alpha / 2, 1 - alpha / 2])
    elo, ehi = np.quantile(es, [alpha / 2, 1 - alpha / 2])
    point = gpd_var_es(fit, p)
    return {"var": point["var"], "var_ci": (float(lo), float(hi)), "es": point["es"], "es_ci": (float(elo), float(ehi)), "xi": fit.xi, "xi_se": fit.xi_se}


def hill_estimator(losses, k: int | None = None) -> dict:
    """Hill (1975) tail index from the ``k`` largest losses: ``xi = mean(ln x_(i)) - ln x_(k+1)``. Only for positive heavy tails (``xi > 0``)."""
    x = np.sort(np.asarray(losses, dtype=float))[::-1]
    x = x[x > 0]
    k = int(k or max(20, len(x) * 0.05))
    k = min(k, len(x) - 1)
    xi = float(np.mean(np.log(x[:k])) - np.log(x[k]))
    return {"xi": xi, "alpha": 1.0 / xi if xi > 0 else float("inf"), "k": k, "se": xi / np.sqrt(k)}


def hill_plot(losses, ks=None) -> pd.DataFrame:
    """The Hill estimate for a range of ``k``: a flat stretch signals a usable ``k``."""
    n = int((np.asarray(losses) > 0).sum())
    ks = ks if ks is not None else np.unique(np.linspace(10, max(20, n // 5), 30).astype(int))
    return pd.DataFrame([hill_estimator(losses, k) for k in ks]).set_index("k")


def threshold_diagnostics(losses, quantiles=(0.85, 0.90, 0.925, 0.95, 0.975)) -> pd.DataFrame:
    """GPD shape and scale (modified scale ``beta - xi u``, which should be flat above a good threshold), exceedance counts and Kolmogorov-Smirnov p-values."""
    rows = []
    x = np.asarray(losses, dtype=float)
    for q in quantiles:
        try:
            f = fit_gpd(x, quantile=q)
        except ValueError:
            continue
        exc = x[x > f.threshold] - f.threshold
        ks = sps.kstest(exc, "genpareto", args=(f.xi, 0, f.beta))
        rows.append({"quantile": q, "threshold": f.threshold, "n_exceed": f.n_exceed, "xi": f.xi, "xi_se": f.xi_se, "beta_modified": f.beta - f.xi * f.threshold,
                     "ks_pvalue": float(ks.pvalue)})
    return pd.DataFrame(rows).set_index("quantile")


def mean_excess(losses, n_points: int = 40) -> pd.DataFrame:
    """Mean-excess function ``e(u) = E[X - u | X > u]``. Linear and increasing in ``u`` for a heavy (GPD, ``xi > 0``) tail."""
    x = np.sort(np.asarray(losses, dtype=float))
    us = np.quantile(x, np.linspace(0.5, 0.99, n_points))
    return pd.DataFrame({"threshold": us, "mean_excess": [x[x > u].mean() - u for u in us], "n": [(x > u).sum() for u in us]}).set_index("threshold")


def fit_gev(block_maxima) -> dict:
    """GEV fit to block maxima by maximum likelihood (scipy parameterisation ``c = -xi``). Returns ``xi, mu, sigma``."""
    m = np.asarray(block_maxima, dtype=float)
    c, loc, scale = sps.genextreme.fit(m)
    return {"xi": float(-c), "mu": float(loc), "sigma": float(scale), "loglik": float(np.sum(sps.genextreme.logpdf(m, c, loc, scale)))}


def gev_return_level(fit: dict, period: float) -> float:
    """The level exceeded on average once every ``period`` blocks."""
    return float(sps.genextreme.isf(1.0 / period, -fit["xi"], fit["mu"], fit["sigma"]))


def block_maxima(losses: pd.Series, freq: str = "YE") -> pd.Series:
    """Maximum loss per calendar block (default year)."""
    return losses.resample(freq).max().dropna()


def extremal_index(losses, threshold: float | None = None, quantile: float = 0.95) -> float:
    """Intervals estimator of the extremal index (Ferro & Segers 2003): ``theta = 1`` means extremes arrive alone, ``theta < 1`` means they cluster
    (``1/theta`` is the mean cluster size). Volatility clustering gives ``theta`` well below one for daily returns."""
    x = np.asarray(losses, dtype=float)
    u = np.quantile(x, quantile) if threshold is None else threshold
    t = np.flatnonzero(x > u)
    if len(t) < 3:
        return float("nan")
    gaps = np.diff(t)
    n = len(gaps)
    if gaps.max() <= 2:
        theta = min(1.0, 2.0 * gaps.sum() ** 2 / (n * (gaps ** 2).sum()))
    else:
        theta = min(1.0, 2.0 * (gaps - 1).sum() ** 2 / (n * ((gaps - 1) * (gaps - 2)).sum()))
    return float(theta)


def dynamic_evt_var(returns: pd.Series, p: float = 0.99, halflife: float = 40.0, threshold_quantile: float = 0.90, min_obs: int = 750, refit_every: int = 21) -> pd.DataFrame:
    """McNeil-Frey (2000) conditional EVT. Each ``refit_every`` days, using only the past:

    1. EWMA volatility ``sigma_t`` (causal) and standardised losses ``z = -r / sigma``;
    2. a GPD fitted to the upper tail of ``z`` up to the previous day, giving the ``p``-quantile ``q_z``;
    3. forecast ``VaR_{t+1} = sigma_{t+1} q_z`` and ``ES_{t+1}`` likewise (``sigma_{t+1}`` from today's EWMA).

    Returns the forecasts indexed by the day they APPLY to, with the realised loss, so breaches can be counted (Kupiec/Christoffersen in ``src.risk.var``).
    """
    r = returns.dropna()
    sigma = np.sqrt((r ** 2).ewm(halflife=halflife, adjust=False).mean())
    z = -r / sigma.shift(1)
    out = {}
    q_z = es_z = np.nan
    for i in range(min_obs, len(r) - 1):
        if (i - min_obs) % refit_every == 0:
            try:
                f = fit_gpd(z.iloc[1: i + 1].dropna().to_numpy(), quantile=threshold_quantile)
                res = gpd_var_es(f, p)
                q_z, es_z = res["var"], res["es"]
            except ValueError:
                pass
        out[r.index[i + 1]] = {"var": sigma.iloc[i] * q_z, "es": sigma.iloc[i] * es_z, "loss": -r.iloc[i + 1]}
    frame = pd.DataFrame(out).T
    frame["breach"] = frame["loss"] > frame["var"]
    return frame
