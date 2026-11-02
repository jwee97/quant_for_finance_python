"""Covariance estimators compared on the risk a portfolio actually realised (spec v3, Release 6, P2-07).

The platform already has the estimators (``src/portfolio/covariance.py``: sample, EWMA, Ledoit-Wolf shrinkage, random-matrix denoising, a positive-definite repair). What an ML strategy needs on top of them is:

* **a fit that can only use what was known.** ``fit_asof`` takes the returns that end at the decision date (inclusive: the close of ``t`` is known at ``t``) over a declared window, drops assets that were not
  eligible at that date or have too few observations (and says which and why), and uses complete cases. Nothing after the date can change the fit (tested by changing it);
* **a matrix that can be used.** Every fit is symmetrised and, if its smallest eigenvalue is below a floor, repaired by clipping the spectrum and restoring the diagonal (the platform's documented repair). The record
  keeps the smallest eigenvalue and condition number before and after, whether a repair was needed and how large it was;
* **a supported factor model.** ``factor`` is a statistical (principal-component) model: ``Sigma = B F B' + D`` with the number of factors fixed by the spec or by the Marchenko-Pastur bound, and ``D`` the residual
  variances, so that every asset's variance is reproduced exactly. No fundamental or macro factor model is claimed: the platform has no point-in-time factor returns;
* **a comparison on realised risk, not on the algebra.** ``compare`` walks decision dates, fits each estimator as of the date, forecasts the risk of a book (equal weight, or the book the strategy held) and
  compares it with the risk realised over the next ``horizon`` days (forecast bias, correlation, QLIKE), and builds the global-minimum-variance book from each matrix to report its realised risk and how much
  it turns over from one date to the next (weight stability).

All covariances here are per-day; nothing is annualised, so that they combine with per-day costs and per-horizon scalings without a hidden factor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..portfolio import covariance as pc

VERSION = "risk-models-v1"
METHODS = ("sample", "ewma", "lw_constant_correlation", "lw_identity", "pca_denoised", "factor")


@dataclass(frozen=True)
class CovSpec:
    method: str
    window: int = 252
    halflife: float = 60.0
    n_factors: int | None = None
    min_obs: int = 60
    min_eigenvalue: float = 1e-10

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown covariance method '{self.method}'; the methods are {list(METHODS)}")
        if self.window < self.min_obs or self.min_obs < 20:
            raise ValueError("the window must hold at least min_obs observations, and min_obs at least 20")
        if self.halflife <= 0 or (self.n_factors is not None and self.n_factors < 1):
            raise ValueError("halflife must be positive and n_factors at least 1")

    def label(self) -> str:
        extra = f"/hl{self.halflife:g}" if self.method == "ewma" else f"/k{self.n_factors or 'mp'}" if self.method == "factor" else ""
        return f"{self.method}{extra}/w{self.window}"

    def content(self) -> dict:
        return {"schema": VERSION, **asdict(self)}

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.content(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class CovFit:
    spec: CovSpec
    asof: pd.Timestamp
    matrix: pd.DataFrame
    dropped: dict
    n_obs: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    min_eig_raw: float
    min_eig_final: float
    condition_raw: float
    condition_final: float
    repaired: bool
    repair_size: float
    symmetry_error: float
    shrinkage: float | None = None
    n_factors_used: int | None = None

    @property
    def assets(self) -> list[str]:
        return list(self.matrix.columns)

    def record(self) -> dict:
        return {"method": self.spec.label(), "spec_hash": self.spec.content_hash(), "asof": str(self.asof.date()), "window": [str(self.window_start.date()), str(self.window_end.date())], "n_obs": int(self.n_obs),
                "n_assets": len(self.matrix), "dropped": self.dropped, "min_eigenvalue_raw": float(self.min_eig_raw), "min_eigenvalue": float(self.min_eig_final),
                "condition_raw": _fin(self.condition_raw), "condition": _fin(self.condition_final), "repaired": bool(self.repaired), "repair_size": float(self.repair_size),
                "symmetry_error": float(self.symmetry_error), "shrinkage": self.shrinkage, "n_factors": self.n_factors_used, "units": "per day"}


def _fin(x: float):
    return float(x) if np.isfinite(x) else None


def _cond(a: np.ndarray) -> float:
    v = np.linalg.eigvalsh(a)
    return float(v.max() / v.min()) if v.min() > 1e-18 else float("inf")


def factor_covariance(window: pd.DataFrame, n_factors: int | None = None) -> tuple[pd.DataFrame, int]:
    """``B F B' + D`` from the principal components of the sample covariance: the leading ``k`` components (``k`` given, or the eigenvalues of the correlation matrix above the Marchenko-Pastur upper bound,
    at least one) and a diagonal of residual variances that restores each asset's own variance exactly."""
    from ..features.pca import marchenko_pastur_bounds

    s = window.cov(ddof=1)
    a = s.to_numpy()
    vol = np.sqrt(np.diag(a))
    corr = a / np.outer(vol, vol)
    values, vectors = np.linalg.eigh(a)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if n_factors is None:
        _, upper = marchenko_pastur_bounds(len(window), window.shape[1])
        n_factors = max(int((np.linalg.eigvalsh(corr) > upper).sum()), 1)
    k = int(min(n_factors, len(values) - 1)) if len(values) > 1 else 1
    common = (vectors[:, :k] * values[:k]) @ vectors[:, :k].T
    resid = np.clip(np.diag(a) - np.diag(common), 0.0, None)
    return pd.DataFrame(common + np.diag(resid), index=s.index, columns=s.columns), k


def _is_eligible(eligible: pd.DataFrame | None, asof: pd.Timestamp, asset) -> bool:
    if eligible is None:
        return True
    return bool(asof in eligible.index and asset in eligible.columns and eligible.loc[asof, asset])


def fit_asof(returns: pd.DataFrame, asof, spec: CovSpec, eligible: pd.DataFrame | None = None) -> CovFit:
    """The covariance as of the close of ``asof``: the last ``spec.window`` rows of ``returns`` up to and including that date, over the assets that were eligible then and have enough observations."""
    asof = pd.Timestamp(asof)
    past = returns.loc[:asof].tail(spec.window)
    dropped: dict = {}
    keep = []
    for a in past.columns:
        if not _is_eligible(eligible, asof, a):
            dropped[str(a)] = "not eligible at the decision date"
        elif past[a].notna().sum() < spec.min_obs:
            dropped[str(a)] = f"fewer than {spec.min_obs} observations in the window"
        else:
            keep.append(a)
    if len(keep) < 2:
        raise ValueError(f"fewer than two assets can be estimated as of {asof.date()} ({len(keep)} kept)")
    window = past[keep].dropna(how="any")
    if len(window) < spec.min_obs:
        raise ValueError(f"only {len(window)} complete observations as of {asof.date()}: fewer than the {spec.min_obs} required")
    shrinkage, k = None, None
    if spec.method == "sample":
        raw = window.cov(ddof=1)
    elif spec.method == "ewma":
        raw = pc.ewma_covariance(window, spec.halflife, annualise=False, min_periods=spec.min_obs)
    elif spec.method == "lw_constant_correlation":
        raw, shrinkage = pc.ledoit_wolf_shrinkage(window, "constant_correlation", annualise=False)
    elif spec.method == "lw_identity":
        raw, shrinkage = pc.ledoit_wolf_shrinkage(window, "identity", annualise=False)
    elif spec.method == "pca_denoised":
        raw = pc.pca_denoised_covariance(window, annualise=False)
    else:
        raw, k = factor_covariance(window, spec.n_factors)
    arr = raw.to_numpy(dtype=float)
    sym_err = float(np.abs(arr - arr.T).max())
    sym = (arr + arr.T) / 2.0
    min_raw = float(np.linalg.eigvalsh(sym).min())
    cond_raw = _cond(sym) if min_raw > 0 else float("inf")
    repaired = min_raw <= spec.min_eigenvalue
    final = pc.nearest_positive_definite(pd.DataFrame(sym, index=raw.index, columns=raw.columns), spec.min_eigenvalue) if repaired else pd.DataFrame(sym, index=raw.index, columns=raw.columns)
    f = final.to_numpy()
    return CovFit(spec, asof, final, dropped, len(window), window.index[0], window.index[-1], min_raw, float(np.linalg.eigvalsh(f).min()), cond_raw, _cond(f), repaired,
                  float(np.linalg.norm(f - sym) / max(np.linalg.norm(sym), 1e-300)), sym_err, None if shrinkage is None else float(shrinkage), k)


def gmv_weights(cov: pd.DataFrame) -> pd.Series:
    """The unconstrained global-minimum-variance book that sums to one: ``S^-1 1 / (1' S^-1 1)``."""
    inv1 = np.linalg.solve(cov.to_numpy(), np.ones(len(cov)))
    return pd.Series(inv1 / inv1.sum(), index=cov.index)


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Each asset's share of the book's variance: ``w_i (S w)_i / w'Sw`` (they sum to one) and its marginal risk ``(S w)_i / sigma``."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    s = cov.to_numpy()
    var = float(w @ s @ w)
    sw = s @ w
    if var <= 0:
        return pd.DataFrame({"weight": w, "marginal": np.nan, "contribution": np.nan, "share": np.nan}, index=cov.index)
    sigma = np.sqrt(var)
    return pd.DataFrame({"weight": w, "marginal": sw / sigma, "contribution": w * sw / sigma, "share": w * sw / var}, index=cov.index)


def compare(returns: pd.DataFrame, decision_dates, specs: list[CovSpec], *, weights: pd.DataFrame | None = None, horizon: int = 21, eligible: pd.DataFrame | None = None) -> dict:
    """Walk ``decision_dates``: fit every estimator as of the date, forecast the daily risk of the book (``weights`` at that date, or equal weight over the fitted assets), and compare it with the risk realised over
    the next ``horizon`` days. Also builds the global-minimum-variance book of each matrix to report its realised risk and its turnover between consecutive dates.

    Per estimator: the number of windows, the mean ratio of forecast to realised risk (1 is unbiased), the correlation of forecast with realised risk, QLIKE of the variance forecast (lower is better), the mean
    and worst condition number, how often a repair was needed, the realised risk of the GMV book and its mean one-way turnover. A window that cannot be estimated is counted, not hidden."""
    dates = [pd.Timestamp(d) for d in decision_dates]
    rows = {s.label(): {"spec": s, "forecast": [], "realised": [], "gmv_risk": [], "turnover": [], "cond": [], "repairs": 0, "windows": 0, "failed": 0, "prev": None} for s in specs}
    for d in dates:
        i = returns.index.searchsorted(d, side="right")
        ahead = returns.iloc[i:i + horizon]
        if len(ahead) < horizon:
            continue
        for label, row in rows.items():
            try:
                fit = fit_asof(returns, d, row["spec"], eligible)
            except ValueError:
                row["failed"] += 1
                continue
            cov = fit.matrix
            w = (weights.loc[d].reindex(cov.index).fillna(0.0) if weights is not None and d in weights.index else pd.Series(1.0 / len(cov), index=cov.index))
            if w.abs().sum() < 1e-12:
                row["failed"] += 1
                continue
            f_var = float(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
            book = ahead.reindex(columns=cov.index).fillna(0.0) @ w
            g = gmv_weights(cov)
            g_book = ahead.reindex(columns=cov.index).fillna(0.0) @ g
            row["windows"] += 1
            row["forecast"].append(f_var)
            row["realised"].append(float((book ** 2).mean()))
            row["gmv_risk"].append(float(np.sqrt((g_book ** 2).mean())))
            row["cond"].append(fit.condition_final)
            row["repairs"] += int(fit.repaired)
            if row["prev"] is not None:
                u = row["prev"].index.union(g.index)
                row["turnover"].append(float(0.5 * (g.reindex(u).fillna(0.0) - row["prev"].reindex(u).fillna(0.0)).abs().sum()))
            row["prev"] = g
    out = {}
    for label, row in rows.items():
        f, r = np.array(row["forecast"]), np.array(row["realised"])
        ok = (f > 0) & (r > 0)
        f, r = f[ok], r[ok]
        ratio = r / f if len(f) else np.array([])
        out[label] = {"spec": row["spec"].content(), "windows": int(row["windows"]), "failed": int(row["failed"]),
                      "risk_ratio_forecast_to_realised": float(np.mean(np.sqrt(f / r))) if len(f) else None, "risk_correlation": float(np.corrcoef(np.sqrt(f), np.sqrt(r))[0, 1]) if len(f) > 2 else None,
                      "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)) if len(f) else None, "mean_condition": _fin(float(np.mean(row["cond"]))) if row["cond"] else None,
                      "worst_condition": _fin(float(np.max(row["cond"]))) if row["cond"] else None, "repair_rate": float(row["repairs"] / row["windows"]) if row["windows"] else None,
                      "gmv_realised_risk": float(np.mean(row["gmv_risk"])) if row["gmv_risk"] else None, "gmv_turnover": float(np.mean(row["turnover"])) if row["turnover"] else None}
    return {"version": VERSION, "horizon": int(horizon), "decision_dates": len(dates), "estimators": out, "units": "per day; realised variance is the mean squared daily return of the book over the horizon",
            "notes": ["every fit uses only returns up to and including its decision date", "QLIKE of the variance forecast: mean(r - ln r - 1) with r = realised / forecast", "a window that could not be estimated is counted in `failed`"]}
