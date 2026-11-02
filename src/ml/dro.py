"""Distributionally robust allocation with a documented uncertainty set (spec v3, Release 8, P3-04). Optional: the base workflow does not use it, and it needs nothing beyond the solver the allocation
module already has.

The nominal problem is ``max_w  mu'w - (lambda/2) w'Sigma w - costs`` over the book's limits. Here the inputs are not taken as true: they may be wrong by an amount that is declared, and calibrated, up
front, and the book is chosen for the worst case inside that set.

* **Expected returns: a box.** ``|mu_i - mu_hat_i| <= kappa * se_i`` for every name, with ``se_i`` the standard error of the sample mean. The worst case of ``mu'w`` over the box is exactly
  ``mu_hat'w - kappa * se'|w|``: a per-name penalty on the size of each position, which the allocation solver takes as ``holding_penalty``. ``kappa`` is **calibrated** as the ``level`` quantile of the largest
  standardised deviation of a block-bootstrap mean from the sample mean over all names, so the box contains the true mean vector with about that probability simultaneously.
* **Covariance: a scale.** ``Sigma' = (1 + rho) Sigma`` with ``rho >= 0``. ``rho`` is calibrated from history: the ``level`` quantile of the ratio of realised to forecast variance of probe portfolios, less
  one, never below zero.
* **Limiting cases.** With ``kappa = 0`` and ``rho = 0`` the problem is the nominal one and the solution is the nominal solution. As ``kappa`` grows the book shrinks, and once every ``|mu_i|`` is inside the box
  it is flat. As ``rho`` grows the book takes less risk.
* **Reported.** The solver's status, residuals, utilisation and fallback (from the allocation module), the nominal and the worst-case expected return and variance of the optimum, and an adversarial check
  of any book against the same set. The robust book is optimal for the worst-case objective, so it cannot be beaten on that measure; it gives up nominal utility to buy it, and the report shows both.

It is a model of ignorance with a stated size, not a forecast: a radius that was calibrated on the past says what the past would have contained, and the report says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..probability.resampling import stationary_indices
from . import allocation as alloc

VERSION = "dro-v1"


@dataclass(frozen=True)
class UncertaintySet:
    kappa: float = 0.0                       # the half-width of the mean box, in standard errors
    rho: float = 0.0                         # the proportional inflation of the covariance
    mean_level: float | None = None          # the simultaneous coverage kappa was calibrated for (None: set by hand)
    cov_level: float | None = None

    def __post_init__(self):
        if self.kappa < 0 or self.rho < 0:
            raise ValueError("kappa and rho are not negative")

    def content(self) -> dict:
        return {"schema": VERSION, **asdict(self), "mean_set": "box |mu_i - mu_hat_i| <= kappa * se_i", "covariance_set": "Sigma' = (1 + rho) * Sigma"}


def standard_errors(returns: pd.DataFrame, block: float = 21.0, n_boot: int = 400, seed: int = 0) -> pd.Series:
    """The standard error of each name's mean return by the stationary bootstrap (serial dependence is kept in the resampled blocks)."""
    R = returns.dropna().to_numpy(dtype=float)
    n = len(R)
    if n < 60:
        raise ValueError(f"{n} complete rows of returns; at least 60 are needed to size an uncertainty set")
    means = np.empty((n_boot, R.shape[1]))
    for b in range(n_boot):
        means[b] = R[stationary_indices(n, block, seed=seed + b)].mean(axis=0)
    return pd.Series(means.std(axis=0, ddof=1), index=returns.columns)


def calibrate_mean_radius(returns: pd.DataFrame, level: float = 0.9, n_boot: int = 500, block: float = 21.0, seed: int = 0) -> dict:
    """``kappa``: the ``level`` quantile, over block-bootstrap resamples, of ``max_i |mean*_i - mean_i| / se_i``. The box ``kappa * se`` then contains the sample-mean error of every name at once with that
    probability (in the bootstrap world); it is a statement about estimation error, not about a regime change."""
    if not 0 < level < 1:
        raise ValueError("level is in (0, 1)")
    R = returns.dropna().to_numpy(dtype=float)
    n = len(R)
    if n < 60:
        raise ValueError(f"{n} complete rows of returns; at least 60 are needed to calibrate a radius")
    mean = R.mean(axis=0)
    draws = np.empty((n_boot, R.shape[1]))
    for b in range(n_boot):
        draws[b] = R[stationary_indices(n, block, seed=seed + b)].mean(axis=0)
    se = draws.std(axis=0, ddof=1)
    worst = np.max(np.abs(draws - mean) / np.where(se > 0, se, np.nan), axis=1)
    kappa = float(np.nanquantile(worst, level))
    return {"kappa": kappa, "level": level, "se": pd.Series(se, index=returns.columns), "n": n, "n_boot": n_boot, "block": block,
            "note": "the simultaneous box for the sample means over all names in the bootstrap world: estimation error only, not regime change"}


def calibrate_inflation(returns: pd.DataFrame, window: int = 63, horizon: int = 21, level: float = 0.9, probes: int = 50, seed: int = 0) -> dict:
    """``rho``: the ``level`` quantile of the ratio of realised to forecast variance, less one (never below zero). The forecast variance of a probe portfolio is its trailing ``window`` variance times
    ``horizon``; the realised one is the sum of its squared returns over the next ``horizon`` days; probes are the equal-weight book and random books."""
    R = returns.dropna().to_numpy(dtype=float)
    n, m = R.shape
    if n < window + 2 * horizon:
        raise ValueError(f"{n} rows; at least {window + 2 * horizon} are needed to compare forecast and realised variance")
    rng = np.random.default_rng(seed)
    W = np.vstack([np.full(m, 1.0 / m)] + [(lambda v: v / np.abs(v).sum())(rng.normal(size=m)) for _ in range(probes - 1)])
    P = R @ W.T
    ratios = []
    for t in range(window, n - horizon, horizon):
        fc = P[t - window:t].var(axis=0, ddof=1) * horizon
        real = (P[t:t + horizon] ** 2).sum(axis=0)
        ratios.extend(real[fc > 0] / fc[fc > 0])
    q = float(np.quantile(ratios, level))
    return {"rho": max(q - 1.0, 0.0), "quantile_ratio": q, "level": level, "windows": len(ratios), "note": "realised over forecast variance of probe portfolios; below one means the forecasts were not too optimistic, and rho is then zero"}


@dataclass
class RobustAllocation:
    allocation: alloc.Allocation
    uncertainty: UncertaintySet
    nominal: dict
    worst_case: dict
    units: str = "returns over one holding window, as fractions of capital"

    @property
    def weights(self) -> pd.Series | None:
        return self.allocation.weights


def worst_case(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, se: pd.Series, uset: UncertaintySet, risk_aversion: float) -> dict:
    """The nominal and the adversarial value of a book against the uncertainty set: the worst mean in the box (each forecast pushed against the position's sign), the inflated variance and the
    resulting utility. Costs are not in this figure; it compares books on the forecast alone."""
    a = w.reindex(mu.index).fillna(0.0).to_numpy()
    m, s = mu.to_numpy(dtype=float), cov.reindex(index=mu.index, columns=mu.index).to_numpy(dtype=float)
    e = se.reindex(mu.index).to_numpy(dtype=float)
    var = float(a @ s @ a)
    nominal_u = float(m @ a - 0.5 * risk_aversion * var)
    mu_wc = float(m @ a - uset.kappa * (e * np.abs(a)).sum())
    var_wc = (1.0 + uset.rho) * var
    return {"nominal_return": float(m @ a), "nominal_variance": var, "nominal_utility": nominal_u, "worst_case_return": mu_wc, "worst_case_variance": var_wc,
            "worst_case_utility": float(mu_wc - 0.5 * risk_aversion * var_wc), "gross": float(np.abs(a).sum())}


def solve(mu: pd.Series, cov: pd.DataFrame, se: pd.Series, uset: UncertaintySet, spec: alloc.AllocationSpec | None = None, current: pd.Series | None = None, **kw) -> RobustAllocation:
    """The book that maximises the worst-case utility over the uncertainty set, under the limits of ``spec``. ``se`` is the standard error of each name's forecast (``standard_errors`` of the returns the
    forecast was estimated from, scaled to the same window)."""
    spec = spec or alloc.AllocationSpec()
    inflated = cov * (1.0 + uset.rho)
    penalty = (uset.kappa * se.reindex(mu.index)).clip(lower=0.0)
    res = alloc.solve(mu, inflated, current, spec, holding_penalty=penalty if uset.kappa > 0 else None, **kw)
    nominal = worst_case(res.weights, mu, cov, se, UncertaintySet(), spec.risk_aversion) if res.weights is not None else {}
    wc = worst_case(res.weights, mu, cov, se, uset, spec.risk_aversion) if res.weights is not None else {}
    return RobustAllocation(res, uset, nominal, wc)


def baselines(mu: pd.Series, cov: pd.DataFrame, spec: alloc.AllocationSpec, shrink: float = 0.5) -> dict[str, pd.Series]:
    """The simpler robust alternatives a robust optimiser is compared with, under the same limits: the nominal book, the book on forecasts shrunk toward zero, the minimum-variance book, and equal weights."""
    out = {}
    nom = alloc.solve(mu, cov, None, spec)
    if nom.weights is not None:
        out["nominal"] = nom.weights
    sh = alloc.solve(mu * shrink, cov, None, spec)
    if sh.weights is not None:
        out["shrunk_forecast"] = sh.weights
    budget = lambda v: v if v is not None else (spec.gross_max if spec.long_only else None)                    # noqa: E731  (a long-only book is fully invested; otherwise the net exposure is as free as the spec leaves it)
    mv = alloc.solve(mu * 0.0, cov, None, alloc.AllocationSpec(**{**asdict(spec), "net_min": budget(spec.net_min), "net_max": budget(spec.net_max)}))
    if mv.weights is not None:
        out["min_variance"] = mv.weights
    n = len(mu)
    out["equal_weight"] = pd.Series(min(spec.gross_max / n, spec.position_max), index=mu.index)
    return out
