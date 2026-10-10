"""Scenario generators: from the simplest (replay history) to the learned (GAN, VAE, diffusion).

Every generator takes the history ``returns`` (a DataFrame of daily simple returns, one column per asset, no missing values), a number of scenarios ``n``, a ``horizon`` in days and a ``seed``, and returns an
array ``(n, horizon, assets)``. They differ in what they preserve:

    historical     random windows of ``horizon`` consecutive days of the actual history: everything is preserved (dependence across assets, clustering, tails) but only what has already happened can occur
    bootstrap      days resampled jointly (whole rows, so cross-sectional dependence survives) as i.i.d. rows, moving blocks, circular blocks or stationary blocks (blocks keep serial dependence, e.g. volatility clustering)
    copula         each asset's own marginal distribution joined by a fitted copula (Gaussian, Student, ...) and drawn i.i.d. day by day: heavy tails and tail dependence, no clustering
    risk_factor    the first ``factors`` principal components bootstrapped in blocks (they carry the dependence and the clustering) plus each asset's residual resampled on its own
    arima_garch    per asset an AR(1) mean and a GJR-GARCH(1,1) volatility, driven by jointly resampled standardised residuals: clustering, leverage and cross-asset dependence, volatility that can exceed anything seen
    wgan_gp        a Wasserstein GAN with gradient penalty on windows of ``window`` days (learned dependence, i.i.d. between windows)
    factor_vae     a variational autoencoder with a factor structure and Student-t noise on windows of ``window`` days
    diffusion      the denoising diffusion model of :mod:`src.models.diffusion` on windows of ``window`` days

The learned generators are fitted on the whole history that is passed in: to use them in a test, pass only the data available at the time. A generator makes no claim about the *future*; it samples a model of the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..econometrics.garch import fit_garch
from ..probability import copulas as cop
from ..probability.resampling import bootstrap_indices


def _clean(returns: pd.DataFrame) -> np.ndarray:
    r = returns.dropna()
    if len(r) < 60 or r.shape[1] < 1:
        raise ValueError("need at least 60 complete days of returns")
    return r.to_numpy(dtype=float)


def historical(returns: pd.DataFrame, n: int, horizon: int, seed: int = 0) -> np.ndarray:
    R = _clean(returns)
    if horizon > len(R):
        raise ValueError("horizon longer than the history")
    starts = np.random.default_rng(seed).integers(0, len(R) - horizon + 1, size=n)
    return np.stack([R[s:s + horizon] for s in starts])


def bootstrap(returns: pd.DataFrame, n: int, horizon: int, seed: int = 0, kind: str = "stationary", block: float = 20.0) -> np.ndarray:
    R = _clean(returns)
    out = np.empty((n, horizon, R.shape[1]))
    for i in range(n):
        out[i] = R[bootstrap_indices(len(R), kind, block, seed=seed * 1000003 + i)[:horizon]]
    return out


def copula(returns: pd.DataFrame, n: int, horizon: int, seed: int = 0, family: str = "student", marginal: str = "empirical") -> np.ndarray:
    R = pd.DataFrame(_clean(returns), columns=returns.columns)
    flat = cop.simulate_joint(R, n * horizon, family, marginal, seed)
    return flat.to_numpy(dtype=float).reshape(n, horizon, R.shape[1])


def risk_factor(returns: pd.DataFrame, n: int, horizon: int, seed: int = 0, factors: int = 3, block: float = 20.0) -> np.ndarray:
    R = _clean(returns)
    mean = R.mean(axis=0)
    X = R - mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    K = min(factors, len(S))
    F, B = U[:, :K] * S[:K], Vt[:K]                                                                  # factor returns (T, K) and loadings (K, N)
    resid = X - F @ B
    out = np.empty((n, horizon, R.shape[1]))
    for i in range(n):
        idx = bootstrap_indices(len(R), "stationary", block, seed=seed * 1000003 + i)[:horizon]
        rng = np.random.default_rng(seed * 7919 + i)
        eps = resid[rng.integers(0, len(R), size=(horizon, R.shape[1])), np.arange(R.shape[1])]    # each asset's residual drawn on its own
        out[i] = mean + F[idx] @ B + eps
    return out


def arima_garch(returns: pd.DataFrame, n: int, horizon: int, seed: int = 0, model: str = "gjr") -> np.ndarray:
    R = _clean(returns)
    T, N = R.shape
    params, Z = [], np.empty((T, N))
    mu, phi = np.empty(N), np.empty(N)
    for j in range(N):
        y = R[:, j]
        x = np.column_stack([np.ones(T - 1), y[:-1]])
        coef = np.linalg.lstsq(x, y[1:], rcond=None)[0]
        mu[j], phi[j] = coef
        e = np.r_[y[0] - y.mean(), y[1:] - x @ coef]
        fit = fit_garch(pd.Series(e), model, "normal", "zero")
        params.append(fit.params)
        Z[:, j] = fit.std_resid.to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty((n, horizon, N))
    last_y = np.tile(R[-1], (n, 1))
    s2 = np.empty((n, N))
    prev_e = np.empty((n, N))
    for j, p in enumerate(params):
        s2[:, j] = p["omega"] / max(1.0 - p["alpha"] - 0.5 * p.get("gamma", 0.0) - p["beta"], 1e-3)
        prev_e[:, j] = 0.0
    prev_s2 = s2.copy()
    for t in range(horizon):
        z = Z[rng.integers(0, T, size=n)]                                                          # jointly resampled standardised residual rows
        for j, p in enumerate(params):
            neg = (prev_e[:, j] < 0).astype(float)
            s2[:, j] = p["omega"] + (p["alpha"] + p.get("gamma", 0.0) * neg) * prev_e[:, j] ** 2 + p["beta"] * prev_s2[:, j]
        e = np.sqrt(s2) * z
        y = mu + phi * last_y + e
        out[:, t] = y
        last_y, prev_e, prev_s2 = y, e, s2.copy()
    return out


def _windows(R: np.ndarray, window: int) -> np.ndarray:
    n = (len(R) // window) * window
    return R[len(R) - n:].reshape(-1, window * R.shape[1])


def _learned(model, returns: pd.DataFrame, n: int, horizon: int, seed: int, window: int) -> np.ndarray:
    R = _clean(returns)
    N = R.shape[1]
    model.fit(_windows(R, window))
    parts = -(-horizon // window)                                                                   # windows needed per scenario
    flat = model.sample(n * parts, seed).reshape(n, parts * window, N)
    return flat[:, :horizon]


def wgan_gp(returns: pd.DataFrame, n: int, horizon: int, seed: int = 0, window: int = 1, **kw) -> np.ndarray:
    from ..models.generative import WGANGP

    return _learned(WGANGP(seed=seed, **kw), returns, n, horizon, seed, window)


def factor_vae(returns: pd.DataFrame, n: int, horizon: int, seed: int = 0, window: int = 1, **kw) -> np.ndarray:
    from ..models.generative import FactorVAE

    return _learned(FactorVAE(seed=seed, **kw), returns, n, horizon, seed, window)


def diffusion(returns: pd.DataFrame, n: int, horizon: int, seed: int = 0, window: int = 1, **kw) -> np.ndarray:
    from ..models.diffusion import Diffusion

    return _learned(Diffusion(seed=seed, **kw), returns, n, horizon, seed, window)


GENERATORS = {"historical": historical, "bootstrap": bootstrap, "copula": copula, "risk_factor": risk_factor, "arima_garch": arima_garch, "wgan_gp": wgan_gp, "factor_vae": factor_vae, "diffusion": diffusion}


def generate(name: str, returns: pd.DataFrame, n: int = 1000, horizon: int = 21, seed: int = 0, **params) -> np.ndarray:
    if name not in GENERATORS:
        raise ValueError(f"unknown generator '{name}'; choose from {sorted(GENERATORS)}")
    if n < 1 or horizon < 1:
        raise ValueError("n >= 1 and horizon >= 1")
    return GENERATORS[name](returns, n, horizon, seed, **params)
