"""Dynamic covariance forecasting (Generation 2, Priority 4).

Static estimators (a trailing sample covariance, an EWMA, a shrinkage target)
treat the covariance matrix as a number to be estimated. Volatility and
correlation are processes, and they move most when it matters. Two models
that let the matrix move:

**DCC-GARCH** (Engle 2002). Each asset gets its own GARCH(1,1) variance; the
standardised residuals z_t = e_t / sigma_t share a slowly moving correlation
matrix R_t driven by

    Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1},   R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2,

and the covariance is D_t R_t D_t. Estimation is in two steps (Gaussian
quasi-maximum likelihood): the N univariate GARCH models first, then (a, b)
from the correlation likelihood given the standardised residuals. That keeps a
15-asset problem to 15 one-dimensional fits and one two-parameter fit.

**Orthogonal GARCH** (Alexander 2001). Principal components of the
standardised returns are uncorrelated by construction, so only k univariate
GARCH models are needed; the covariance is rebuilt from the k factor variances
plus a constant diagonal for what the retained factors do not explain.

Both forecast the AVERAGE daily covariance over the coming ``horizon`` days,
not the next day's, because the covariance is used to size a month-long book.
Volatility forecasts follow the GARCH recursion (and its IGARCH limit without
exploding); correlation forecasts mean-revert towards Qbar at rate (a + b).

Not implemented, and not needed to answer the question asked here: Wishart /
matrix-variate stochastic-volatility models and factor stochastic volatility.
They are the natural next step if a 15-asset DCC proves too rigid, but each is
a considerably heavier estimation problem.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from ..features.volatility import fit_garch


# ---------------------------------------------------------------------------
# Univariate GARCH(1,1) building blocks (everything in the scaled units of the fit)
# ---------------------------------------------------------------------------
@dataclass
class Garch11:
    mu: float
    omega: float
    alpha: float
    beta: float
    scale: float                      # x = scale * r
    sigma2_last: float                # conditional variance of the last training day (scaled units)
    eps_last: float                   # its residual
    std_resid: np.ndarray             # in-sample standardised residuals
    sigma2: np.ndarray                # in-sample conditional variances (scaled units)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


def fit_garch11(returns: np.ndarray) -> Garch11:
    """Gaussian QMLE GARCH(1,1) with a constant mean, via the existing univariate fitter."""
    series = pd.Series(np.asarray(returns, dtype=float))
    fit = fit_garch(series, 1, 1, dist="normal")
    omega, alpha, beta, mu = fit.unpack()
    result = fit.result
    sigma = np.asarray(result.conditional_volatility, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    return Garch11(mu=mu, omega=omega, alpha=alpha, beta=beta, scale=float(fit.scale),
                   sigma2_last=float(sigma[-1] ** 2), eps_last=float(resid[-1]),
                   std_resid=np.asarray(result.std_resid, dtype=float), sigma2=sigma ** 2)


def garch_filter(returns_scaled: np.ndarray, g: Garch11, sigma2_last: float | None = None,
                 eps_last: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Roll the GARCH recursion forward over new observations.

    ``returns_scaled`` are new returns already multiplied by ``g.scale``.
    Returns the conditional variance FOR each new day (known the evening
    before) and the standardised residual of each. The recursion is linear in
    sigma^2, so it is a one-pole filter and needs no Python loop.
    """
    x = np.asarray(returns_scaled, dtype=float)
    eps = x - g.mu
    s2 = g.sigma2_last if sigma2_last is None else sigma2_last
    e0 = g.eps_last if eps_last is None else eps_last
    drive = np.empty_like(x)
    drive[0] = g.omega + g.alpha * e0 ** 2 + g.beta * s2
    drive[1:] = g.omega + g.alpha * eps[:-1] ** 2
    sigma2 = lfilter([1.0], [1.0, -g.beta], drive)
    return sigma2, eps / np.sqrt(sigma2)


def garch_variance_path(g: Garch11, sigma2_next: float, horizon: int) -> np.ndarray:
    """E[sigma^2_{t+h}] for h = 1..horizon (scaled units), starting from the next day's variance.

    Below unit persistence the path decays geometrically towards the long-run
    variance; at or above it (an integrated fit) there is no long-run variance
    and the path grows linearly in omega instead of diverging.
    """
    steps = np.arange(horizon, dtype=float)
    if g.persistence < 1.0 - 1e-8:
        long_run = g.omega / (1.0 - g.persistence)
        return long_run + g.persistence ** steps * (sigma2_next - long_run)
    return sigma2_next + steps * g.omega


# ---------------------------------------------------------------------------
# DCC
# ---------------------------------------------------------------------------
@dataclass
class DCC:
    a: float
    b: float
    Qbar: np.ndarray
    loglik: float
    success: bool


def dcc_q_path(z: np.ndarray, a: float, b: float, Qbar: np.ndarray,
               Q0: np.ndarray | None = None) -> np.ndarray:
    """Q_0..Q_T for standardised residuals z (T x N): Q_t is the pseudo-correlation FOR day t.

    ``Q_0`` is ``Q0`` (default ``Qbar``); ``Q_T`` is the one-day-ahead matrix
    for the first unseen day. The recursion is linear, so every entry is a
    one-pole filter in time.
    """
    T, N = z.shape
    start = Qbar if Q0 is None else Q0
    drive = (1.0 - a - b) * Qbar[None, :, :] + a * z[:, :, None] * z[:, None, :]      # input at step t+1 uses z_t
    out = np.empty((T + 1, N, N))
    out[0] = start
    flat = drive.reshape(T, N * N)
    filtered = lfilter([1.0], [1.0, -b], flat, axis=0, zi=(b * start.reshape(1, -1)))[0]
    out[1:] = filtered.reshape(T, N, N)
    return out


def _normalise(Q: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.einsum("...ii->...i", Q))
    return Q / (d[..., :, None] * d[..., None, :])


def dcc_negative_loglik(params: np.ndarray, z: np.ndarray, Qbar: np.ndarray) -> float:
    """Minus the correlation part of the Gaussian log-likelihood, summed over t = 1..T-1."""
    a, b = float(params[0]), float(params[1])
    if a < 0 or b < 0 or a + b >= 1.0:
        return 1e12
    Q = dcc_q_path(z[:-1], a, b, Qbar)[1:]                 # Q_1 .. Q_{T-1}, each using z up to t-1
    R = _normalise(Q)
    target = z[1:]
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return 1e12
    solved = np.linalg.solve(R, target[..., None])[..., 0]
    quad = np.einsum("ti,ti->t", target, solved) - np.einsum("ti,ti->t", target, target)
    return float(0.5 * np.sum(logdet + quad))


def fit_dcc(z: np.ndarray) -> DCC:
    """Second step of the two-step QMLE: (a, b) given standardised residuals."""
    z = np.asarray(z, dtype=float)
    Qbar = np.cov(z, rowvar=False, ddof=1)
    best = None
    for start in ((0.02, 0.95), (0.05, 0.90), (0.01, 0.98)):
        result = minimize(dcc_negative_loglik, np.array(start), args=(z, Qbar), method="SLSQP",
                          bounds=[(1e-6, 0.5), (1e-6, 0.9999)],
                          constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[0] - p[1]}],
                          options={"maxiter": 100, "ftol": 1e-9})
        if best is None or result.fun < best.fun:
            best = result
        interior = 1e-4 < result.x[0] < 0.49 and result.x[1] < 0.9998
        if result.success and interior:                    # a well-behaved first start needs no second opinion
            break
    return DCC(a=float(best.x[0]), b=float(best.x[1]), Qbar=Qbar, loglik=float(-best.fun),
               success=bool(best.success))


def dcc_correlation_forecast(Q_next: np.ndarray, dcc: DCC, horizon: int) -> np.ndarray:
    """R_{t+h|t}, h = 1..horizon, from E[Q_{t+h}] = Qbar + (a+b)^(h-1) (Q_{t+1} - Qbar)."""
    persistence = dcc.a + dcc.b
    decay = persistence ** np.arange(horizon, dtype=float)
    Q = dcc.Qbar[None, :, :] + decay[:, None, None] * (Q_next - dcc.Qbar)[None, :, :]
    return _normalise(Q)


def average_covariance(variance_paths: np.ndarray, correlations: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(1/H) sum_h D_h R_h D_h in RETURN units.

    ``variance_paths`` is (H, N) in scaled units, ``correlations`` is (H, N, N),
    ``scale`` converts back (variance in return units = scaled variance / scale^2).
    """
    sd = np.sqrt(variance_paths) / scale[None, :]
    return np.mean(sd[:, :, None] * correlations * sd[:, None, :], axis=0)


# ---------------------------------------------------------------------------
# Orthogonal GARCH
# ---------------------------------------------------------------------------
@dataclass
class OGarch:
    sigma: np.ndarray                 # training standard deviation of each asset
    center: np.ndarray                # training mean of each asset
    weights: np.ndarray               # N x k loadings of the retained components
    eigenvalues: np.ndarray           # variance of each retained component
    psi: np.ndarray                   # constant residual variance of each standardised asset
    garch: list[Garch11] = field(default_factory=list)


def fit_ogarch(returns: np.ndarray, n_factors: int = 3) -> OGarch:
    """PCA of the standardised returns, then a GARCH(1,1) on each retained component."""
    r = np.asarray(returns, dtype=float)
    center = r.mean(axis=0)
    sigma = r.std(axis=0, ddof=1)
    y = (r - center) / sigma
    eigenvalues, vectors = np.linalg.eigh(np.corrcoef(y, rowvar=False))
    order = np.argsort(eigenvalues)[::-1][:n_factors]
    lam, W = eigenvalues[order], vectors[:, order]
    explained = (W ** 2) @ lam
    psi = np.maximum(1.0 - explained, 1e-6)
    scores = y @ W
    garch = [fit_garch11(scores[:, j]) for j in range(W.shape[1])]
    return OGarch(sigma=sigma, center=center, weights=W, eigenvalues=lam, psi=psi, garch=garch)


def ogarch_forecast(model: OGarch, factor_sigma2_next: np.ndarray, horizon: int) -> np.ndarray:
    """Average daily covariance (return units) over the next ``horizon`` days."""
    paths = np.column_stack([garch_variance_path(g, float(factor_sigma2_next[j]), horizon) / g.scale ** 2
                             for j, g in enumerate(model.garch)])                     # (H, k) in score units
    mean_factor_variance = paths.mean(axis=0)
    sigma_y = (model.weights * mean_factor_variance) @ model.weights.T + np.diag(model.psi)
    return sigma_y * np.outer(model.sigma, model.sigma)


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------
@dataclass
class DynamicForecasts:
    """Daily-covariance forecasts at each origin, and what each refit estimated."""

    origins: pd.DatetimeIndex
    columns: list[str]
    forecasts: dict[str, np.ndarray]            # model -> (n_origins, N, N)
    diagnostics: pd.DataFrame


def walk_forward_dynamic_covariance(returns: pd.DataFrame, origins: pd.DatetimeIndex,
                                    min_train: int = 750, refit_every: int = 252, horizon: int = 21,
                                    n_factors: int = 3, models: tuple[str, ...] = ("dcc", "ogarch")
                                    ) -> DynamicForecasts:
    """Refit on rows before each window; filter through it; forecast at each origin.

    The refit that starts at row ``r`` is estimated on rows ``0..r-1``. The GARCH
    recursions and the DCC matrix then run over every row up to the origin
    ``o`` (inclusive) with those frozen parameters, so a forecast made at
    ``o`` uses parameters that never saw ``o`` and returns through ``o``.
    Origins before the first refit have no forecast.
    """
    values = np.ascontiguousarray(returns.to_numpy(dtype=float))
    index = pd.DatetimeIndex(returns.index)
    T, N = values.shape
    positions = np.array([index.get_loc(d) for d in pd.DatetimeIndex(origins)])
    keep = positions >= min_train
    positions, kept_origins = positions[keep], pd.DatetimeIndex(origins)[keep]
    out = {m: np.full((len(positions), N, N), np.nan) for m in models}
    rows = []
    for j, start in enumerate(range(int(min_train), T, int(refit_every))):
        end = min(start + int(refit_every), T)
        window = np.flatnonzero((positions >= start) & (positions < end))
        train = values[:start]
        entry = {"refit_date": index[start].date().isoformat(), "train_days": start}
        if "dcc" in models:
            fits = [fit_garch11(train[:, i]) for i in range(N)]
            z_train = np.column_stack([g.std_resid for g in fits])
            dcc = fit_dcc(z_train)
            sigma2_new = np.empty((end - start, N))
            z_new = np.empty((end - start, N))
            for i, g in enumerate(fits):
                sigma2_new[:, i], z_new[:, i] = garch_filter(values[start:end, i] * g.scale, g)
            z_all = np.vstack([z_train, z_new])
            Q = dcc_q_path(z_all, dcc.a, dcc.b, dcc.Qbar)
            scale = np.array([g.scale for g in fits])
            for k in window:
                o = positions[k]
                row = o - start                                     # last observed new row
                # one-day-ahead variance for each asset, from the state at the origin
                nxt = np.array([g.omega + g.alpha * (values[o, i] * g.scale - g.mu) ** 2 + g.beta * sigma2_new[row, i]
                                for i, g in enumerate(fits)])
                paths = np.column_stack([garch_variance_path(g, nxt[i], horizon) for i, g in enumerate(fits)])
                R = dcc_correlation_forecast(Q[o + 1], dcc, horizon)
                out["dcc"][k] = average_covariance(paths, R, scale)
            entry.update({"dcc_a": dcc.a, "dcc_b": dcc.b, "dcc_persistence": dcc.a + dcc.b,
                          "dcc_loglik": dcc.loglik, "dcc_converged": dcc.success,
                          "garch_persistence_mean": float(np.mean([g.persistence for g in fits])),
                          "garch_persistence_max": float(np.max([g.persistence for g in fits]))})
        if "ogarch" in models:
            og = fit_ogarch(train, n_factors)
            y_new = (values[start:end] - og.center) / og.sigma
            scores_new = y_new @ og.weights
            factor_s2 = np.empty((end - start, n_factors))
            for f, g in enumerate(og.garch):
                factor_s2[:, f], _ = garch_filter(scores_new[:, f] * g.scale, g)
            for k in window:
                o = positions[k]
                row = o - start
                nxt = np.array([g.omega + g.alpha * (scores_new[row, f] * g.scale - g.mu) ** 2 + g.beta * factor_s2[row, f]
                                for f, g in enumerate(og.garch)])
                out["ogarch"][k] = ogarch_forecast(og, nxt, horizon)
            entry.update({"ogarch_explained": float(og.eigenvalues.sum() / N),
                          "ogarch_persistence_mean": float(np.mean([g.persistence for g in og.garch]))})
        rows.append(entry)
    return DynamicForecasts(origins=kept_origins, columns=list(returns.columns), forecasts=out,
                            diagnostics=pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Static estimators and evaluation
# ---------------------------------------------------------------------------
def static_forecasts(returns: pd.DataFrame, origins: pd.DatetimeIndex, spec: dict) -> dict[str, np.ndarray]:
    """Daily-covariance forecasts from the static estimators, through and including each origin."""
    from .covariance import estimate_covariance

    index = pd.DatetimeIndex(returns.index)
    out = {name: np.full((len(origins), returns.shape[1], returns.shape[1]), np.nan) for name in spec}
    for k, origin in enumerate(origins):
        pos = index.get_loc(origin)
        history = returns.iloc[:pos + 1]
        for name, params in spec.items():
            lookback = int(params.get("lookback", 252))
            cov = estimate_covariance(history, name, lookback, float(params.get("halflife", 60.0)),
                                      target=str(params.get("target", "constant_correlation")),
                                      annualise=False)
            out[name][k] = cov.to_numpy(dtype=float)
    return out


def realised_second_moment(returns: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """(1/H) sum_{h=1..H} r_{o+h} r_{o+h}' for each origin o (NaN when the future is too short)."""
    values = returns.to_numpy(dtype=float)
    index = pd.DatetimeIndex(returns.index)
    out = np.full((len(origins), values.shape[1], values.shape[1]), np.nan)
    for k, origin in enumerate(origins):
        pos = index.get_loc(origin)
        future = values[pos + 1: pos + 1 + horizon]
        if len(future) == horizon:
            out[k] = future.T @ future / horizon
    return out


def qlike_loss(forecast: np.ndarray, realised: np.ndarray) -> np.ndarray:
    """log|S| + tr(S^-1 R) per origin: the Gaussian deviance, up to a constant that does not depend on S."""
    sign, logdet = np.linalg.slogdet(forecast)
    trace = np.einsum("tij,tji->t", np.linalg.inv(forecast), realised)
    loss = logdet + trace
    return np.where((sign > 0) & np.isfinite(realised).all(axis=(1, 2)), loss, np.nan)


def gmv_realised_variance(forecast: np.ndarray, realised: np.ndarray) -> np.ndarray:
    """Realised variance of the unconstrained global-minimum-variance portfolio built from each forecast."""
    ones = np.ones(forecast.shape[1])
    solved = np.linalg.solve(forecast, ones)
    w = solved / (ones @ solved)
    return np.einsum("ti,tij,tj->t", w, realised, w)
