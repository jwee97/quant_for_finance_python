"""Covariance estimation (Ch. 20 §20.2.7-§20.2.11).

With N = 15 assets and a 252-day window the sample covariance matrix is
estimated from 252 observations for 120 free parameters. It is invertible,
but its smallest eigenvalues are almost pure noise -- and mean-variance
optimisation loads precisely on those directions, because they look like
free diversification. Every estimator here exists to manage that.

``sample``                 the unbiased baseline
``rolling`` / ``ewma``     time-varying versions
``shrinkage``              Ledoit-Wolf towards a structured target
``pca_denoised``           replace sub-noise-bound eigenvalues (RMT)

Plus ``nearest_positive_definite`` for the case the book raises directly:
what to do when an estimated matrix has negative eigenvalues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.dates import TRADING_DAYS_PER_YEAR

ANN = TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Point estimators
# ---------------------------------------------------------------------------
def sample_covariance(returns: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    cov = returns.dropna(how="any").cov(ddof=1)
    return cov * ANN if annualise else cov


def ewma_covariance(returns: pd.DataFrame, halflife: float = 60.0, annualise: bool = True,
                    min_periods: int = 60) -> pd.DataFrame:
    """Exponentially weighted covariance, evaluated at the last date."""
    clean = returns.dropna(how="any")
    if len(clean) < min_periods:
        raise ValueError(f"need at least {min_periods} complete observations")
    weights = 0.5 ** (np.arange(len(clean))[::-1] / halflife)
    weights = weights / weights.sum()
    centred = clean - np.average(clean, axis=0, weights=weights)
    cov = np.einsum("t,ti,tj->ij", weights, centred.to_numpy(), centred.to_numpy())
    # Bias correction for the effective sample size of a weighted mean.
    cov = cov / (1.0 - np.sum(weights ** 2))
    frame = pd.DataFrame(cov, index=clean.columns, columns=clean.columns)
    return frame * ANN if annualise else frame


def constant_correlation_target(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf's structured target: each asset's own variance, and one
    common correlation for every pair.

    Two parameters instead of 120. It is certainly wrong, and that is the
    point: it is wrong in a stable way, so blending towards it trades a little
    bias for a large reduction in variance.
    """
    clean = returns.dropna(how="any")
    cov = clean.cov(ddof=1)
    vol = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(vol, vol)
    n = len(vol)
    off_diagonal = corr[~np.eye(n, dtype=bool)]
    average = float(np.mean(off_diagonal))
    target_corr = np.full((n, n), average)
    np.fill_diagonal(target_corr, 1.0)
    target = target_corr * np.outer(vol, vol)
    return pd.DataFrame(target, index=cov.index, columns=cov.columns)


def ledoit_wolf_shrinkage(returns: pd.DataFrame, target: str = "constant_correlation",
                          delta: float | None = None, annualise: bool = True) -> tuple[pd.DataFrame, float]:
    """``Sigma* = delta * F + (1 - delta) * S``.

    With ``delta=None`` the intensity is estimated analytically: the optimal
    delta trades the estimation error of the sample matrix against the
    specification error of the target, and scikit-learn's implementation is
    used for the identity target while the constant-correlation target uses
    the closed form of Ledoit and Wolf (2004).
    """
    clean = returns.dropna(how="any")
    if len(clean) < 30:
        raise ValueError("not enough complete observations for a covariance estimate")
    sample = clean.cov(ddof=1)

    if target == "identity":
        from sklearn.covariance import LedoitWolf

        estimator = LedoitWolf().fit(clean.to_numpy())
        shrunk = pd.DataFrame(estimator.covariance_, index=clean.columns, columns=clean.columns)
        intensity = float(estimator.shrinkage_)
        return (shrunk * ANN if annualise else shrunk), intensity

    if target == "diagonal":
        structured = pd.DataFrame(np.diag(np.diag(sample.to_numpy())),
                                  index=sample.index, columns=sample.columns)
    elif target == "constant_correlation":
        structured = constant_correlation_target(clean)
    else:
        raise ValueError(f"unknown shrinkage target '{target}'")

    if delta is None:
        delta = _shrinkage_intensity(clean, sample, structured)
    delta = float(np.clip(delta, 0.0, 1.0))
    shrunk = delta * structured + (1.0 - delta) * sample
    return (shrunk * ANN if annualise else shrunk), delta


def _shrinkage_intensity(returns: pd.DataFrame, sample: pd.DataFrame,
                         target: pd.DataFrame) -> float:
    """Analytic shrinkage intensity ``delta* = (pi - rho) / gamma / n``.

    ``pi``    sum of asymptotic variances of the sample covariance entries
    ``gamma`` squared distance between the sample matrix and the target
    ``rho``   covariance between the estimation errors of the two, which is
              approximated by its diagonal term (Ledoit-Wolf's own
              simplification for the constant-correlation target).
    """
    x = returns.to_numpy(dtype=float)
    n, p = x.shape
    centred = x - x.mean(axis=0)
    s = sample.to_numpy(dtype=float)

    # pi: variance of each entry of the sample covariance matrix
    squared = centred ** 2
    pi_matrix = (squared.T @ squared) / n - 2.0 * (centred.T @ centred) / n * s + s ** 2
    pi = float(pi_matrix.sum())

    gamma = float(np.sum((target.to_numpy(dtype=float) - s) ** 2))
    rho = float(np.trace(pi_matrix))  # diagonal terms only

    if gamma <= 1e-18:
        return 1.0
    kappa = (pi - rho) / gamma
    return float(np.clip(kappa / n, 0.0, 1.0))


def pca_denoised_covariance(returns: pd.DataFrame, n_components: int | None = None,
                            annualise: bool = True) -> pd.DataFrame:
    """Random-matrix denoising: keep the informative eigenvalues, average the rest.

    ``n_components=None`` uses the Marchenko-Pastur bound to decide how many
    eigenvalues are distinguishable from noise.
    """
    from ..features.pca import marchenko_pastur_bounds

    clean = returns.dropna(how="any")
    cov = clean.cov(ddof=1)
    vol = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(vol, vol)

    values, vectors = np.linalg.eigh(corr)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    if n_components is None:
        _, upper = marchenko_pastur_bounds(len(clean), clean.shape[1])
        n_components = max(int((values > upper).sum()), 1)

    denoised = values.copy()
    if n_components < len(values):
        denoised[n_components:] = denoised[n_components:].mean()
    rebuilt = vectors @ np.diag(denoised) @ vectors.T
    # Restore unit diagonal, then rescale by the original volatilities.
    diag = np.sqrt(np.diag(rebuilt))
    rebuilt = rebuilt / np.outer(diag, diag)
    out = pd.DataFrame(rebuilt * np.outer(vol, vol), index=cov.index, columns=cov.columns)
    return out * ANN if annualise else out


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
def is_positive_definite(matrix: pd.DataFrame | np.ndarray, tolerance: float = 0.0) -> bool:
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    return bool(values.min() > tolerance)


def nearest_positive_definite(matrix: pd.DataFrame, min_eigenvalue: float = 1e-10) -> pd.DataFrame:
    """Clip negative eigenvalues and rescale to preserve the diagonal.

    Ch. 20 §20.2.9's problem: an estimated covariance matrix with a negative
    eigenvalue implies a portfolio with negative variance. Clipping the
    spectrum is the standard repair; rescaling afterwards preserves each
    asset's variance, which matters because those variances are the one part
    of the matrix that is estimated well.
    """
    array = np.asarray(matrix, dtype=float)
    symmetric = (array + array.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    if values.min() > min_eigenvalue:
        return pd.DataFrame(symmetric, index=matrix.index, columns=matrix.columns)

    values = np.clip(values, min_eigenvalue, None)
    repaired = vectors @ np.diag(values) @ vectors.T
    original_diag = np.diag(symmetric).copy()
    new_diag = np.diag(repaired).copy()
    scale = np.sqrt(np.where(new_diag > 0, original_diag / new_diag, 1.0))
    repaired = repaired * np.outer(scale, scale)
    return pd.DataFrame(repaired, index=matrix.index, columns=matrix.columns)


def condition_number(matrix: pd.DataFrame) -> float:
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    smallest = values.min()
    return float(values.max() / smallest) if smallest > 1e-18 else float("inf")


# ---------------------------------------------------------------------------
# Dispatch and evaluation
# ---------------------------------------------------------------------------
def estimate_covariance(returns: pd.DataFrame, method: str = "shrinkage", lookback: int = 252,
                        halflife: float = 60.0, target: str = "constant_correlation",
                        delta: float | None = None, annualise: bool = True,
                        min_eigenvalue: float = 1e-10) -> pd.DataFrame:
    """Single entry point used by the optimisers and the backtest loop."""
    window = returns.tail(lookback) if lookback else returns
    if method == "sample":
        cov = sample_covariance(window, annualise)
    elif method == "rolling":
        cov = sample_covariance(window, annualise)
    elif method == "ewma":
        cov = ewma_covariance(returns, halflife, annualise)
    elif method == "shrinkage":
        cov, _ = ledoit_wolf_shrinkage(window, target, delta, annualise)
    elif method == "pca_denoised":
        cov = pca_denoised_covariance(window, annualise=annualise)
    else:
        raise ValueError(f"unknown covariance method '{method}'")
    return nearest_positive_definite(cov, min_eigenvalue)


def portfolio_volatility(weights, covariance) -> float:
    """``sqrt(w' Sigma w)``."""
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def evaluate_covariance_forecasts(returns: pd.DataFrame, methods=("sample", "ewma", "shrinkage", "pca_denoised"),
                                  lookback: int = 252, horizon: int = 21, step: int = 21,
                                  weights: pd.DataFrame | None = None,
                                  halflife: float = 60.0) -> pd.DataFrame:
    """Score covariance estimators by out-of-sample portfolio-risk forecasts.

    Spec §33: do not judge a covariance estimator by its mathematical
    properties alone. Build the forecast ``sqrt(w' Sigma_t w)``, compare it
    against the risk the portfolio subsequently realises, and report the
    error. An equally weighted book is used when no weights are supplied, so
    the comparison is not contaminated by an optimiser's own behaviour.
    """
    clean = returns.dropna(how="any")
    rows = []
    for method in methods:
        forecasts, realised, deltas, conditions = [], [], [], []
        for end in range(lookback, len(clean) - horizon, step):
            window = clean.iloc[end - lookback:end]
            future = clean.iloc[end:end + horizon]
            try:
                cov = estimate_covariance(window, method, lookback, halflife, annualise=True)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if weights is not None and clean.index[end - 1] in weights.index:
                w = weights.loc[clean.index[end - 1]].reindex(clean.columns).fillna(0.0).to_numpy()
                if np.abs(w).sum() < 1e-12:
                    continue
            else:
                w = np.full(clean.shape[1], 1.0 / clean.shape[1])
            forecasts.append(portfolio_volatility(w, cov))
            realised.append(float((future @ w).std(ddof=1) * np.sqrt(ANN)))
            conditions.append(condition_number(cov))
            if method == "shrinkage":
                _, intensity = ledoit_wolf_shrinkage(window, "constant_correlation", None, True)
                deltas.append(intensity)
        if not forecasts:
            continue
        f, r = np.array(forecasts), np.array(realised)
        valid = np.isfinite(f) & np.isfinite(r) & (f > 0) & (r > 0)
        f, r = f[valid], r[valid]
        ratio = (r ** 2) / (f ** 2)
        rows.append(
            {
                "method": method,
                "n_windows": int(len(f)),
                "corr": float(np.corrcoef(f, r)[0, 1]) if len(f) > 2 else np.nan,
                "rmse": float(np.sqrt(np.mean((f - r) ** 2))),
                "mean_bias": float(np.mean(f - r)),
                "mean_ratio": float(np.mean(f / r)),
                "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
                "forecast_stability": float(np.std(np.diff(f))),
                "mean_condition_number": float(np.mean(conditions)),
                "mean_shrinkage_delta": float(np.mean(deltas)) if deltas else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("method").sort_values("qlike") if rows else pd.DataFrame()
