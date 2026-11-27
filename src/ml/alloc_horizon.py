"""Horizon arithmetic for the portfolio allocation: four quantities that are easy to confuse and are kept apart here.

``h``  the prediction horizon: the window the model's number is a forecast of (``dataset.horizon``); outcomes resolve ``lag + h`` sessions after the decision.
``L``  the intended holding window: how long a position the allocation takes is meant to be held before it is reconsidered; the window the optimisation is posed over (``allocation.holding_days``; ``h`` when not set).
``r``  the rebalancing frequency: how often a new decision is made (``allocation.opt_every``, in sessions), and, separately, how often the engine is allowed to trade (``backtest.rebalance``).
``lag`` the execution lag between a decision and the first return it earns (``dataset.lag`` + ``backtest.extra_lag``).

Expected return over L. A forecast is an expected return over h. Over a window L it is taken to accrue evenly within the horizon and not at all beyond it: ``mu_L = mu_h * min(L, h) / h``. This is an assumption (a model
trained on an h-day target says nothing about days after h), and it is conservative: it never extrapolates an edge past its horizon.

Covariance over L. Three estimators, recorded with every decision:

``linear``          the daily covariance times L (returns uncorrelated through time: the legacy rule when L = h).
``variance_ratio``  the daily correlation, each asset's variance multiplied by its Lo-MacKinlay variance ratio at L (Bartlett-weighted autocorrelations of its past daily returns, clipped to [0.25, 4]): a
                    positively autocorrelated asset is riskier over L days than the square-root rule says, a mean-reverting one less so.
``multiperiod``     the covariance of overlapping L-day compounded returns over the same past window, blended with the linear estimate by ``n_eff / (n_eff + n0)`` where ``n_eff = scenarios / L`` is the number of
                    independent windows the past holds (overlapping windows share days, so they carry far less information than their count): with few independent windows the answer is the linear estimate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import risk_models as rm

COV_HORIZON_METHODS = ("linear", "variance_ratio", "multiperiod")
N0 = 12.0                                   # independent windows at which the direct estimate and the linear one carry equal weight


def edge_scale(holding: int, horizon: int) -> float:
    """Share of the h-day expected return earned over an L-day holding window (evenly within the horizon, nothing beyond it)."""
    return min(int(holding), int(horizon)) / float(horizon)


def variance_ratio(window: pd.DataFrame, L: int) -> pd.Series:
    """Per-asset variance ratio at L: ``1 + 2 * sum_{k=1}^{L-1} (1 - k/L) rho_k`` from the window's daily returns (Bartlett weights keep it non-negative). L = 1 gives 1."""
    out = {}
    for c in window.columns:
        x = window[c].dropna().to_numpy(dtype=float)
        n = len(x)
        if L <= 1 or n < 4 * L or n < 40:
            out[c] = 1.0
            continue
        x = x - x.mean()
        v = float((x ** 2).sum() / n)
        if v <= 0:
            out[c] = 1.0
            continue
        s = 0.0
        for k in range(1, min(L, n // 4)):
            s += (1.0 - k / L) * float((x[k:] * x[:-k]).sum() / n) / v
        out[c] = float(np.clip(1.0 + 2.0 * s, 0.25, 4.0))
    return pd.Series(out)


def horizon_covariance(daily: pd.DataFrame, window: pd.DataFrame, L: int, method: str) -> tuple[pd.DataFrame, dict]:
    """The covariance over an L-day window from the daily one and the past daily returns ``window`` it was fitted on, and what was done."""
    L = int(L)
    if method not in COV_HORIZON_METHODS:
        raise ValueError(f"cov_horizon must be one of {list(COV_HORIZON_METHODS)}")
    lin = daily * float(L)
    info: dict = {"method": method, "window_days": L}
    if method == "linear" or L <= 1:
        info["method"] = "linear" if method == "linear" else f"{method} (L = 1 is the daily covariance)"
        return lin, info
    names = list(daily.index)
    if method == "variance_ratio":
        vr = variance_ratio(window[names], L)
        d = np.sqrt(vr.reindex(names).fillna(1.0).to_numpy())
        out = lin.to_numpy() * np.outer(d, d)
        info["variance_ratio"] = {str(k): float(v) for k, v in vr.items()}
        return pd.DataFrame(out, index=daily.index, columns=daily.columns), info
    w = window[names].dropna(how="any")
    n = len(w)
    n_scen = n - L + 1
    if n_scen < max(3 * L, 40):
        info.update(method="linear", fallback=f"{n_scen} overlapping {L}-day windows in the past window: fewer than {max(3 * L, 40)} needed for a direct estimate, so the linear estimate is used")
        return lin, info
    comp = np.expm1(np.log1p(w.clip(lower=-0.999999)).rolling(L).sum()).dropna().to_numpy(dtype=float)
    direct = np.cov(comp, rowvar=False, ddof=1)
    direct = np.atleast_2d(direct)
    n_eff = n_scen / float(L)
    lam = n_eff / (n_eff + N0)
    blend = lam * direct + (1.0 - lam) * lin.to_numpy()
    sym = (blend + blend.T) / 2.0
    fixed = rm.pc.nearest_positive_definite(pd.DataFrame(sym, index=daily.index, columns=daily.columns), 1e-10) if np.linalg.eigvalsh(sym).min() <= 1e-10 else pd.DataFrame(sym, index=daily.index, columns=daily.columns)
    info.update(direct_weight=float(lam), independent_windows=float(n_eff), overlapping_windows=int(n_scen), repaired=bool(np.linalg.eigvalsh(sym).min() <= 1e-10))
    return fixed, info


def tranche_count(holding: int, every: int) -> int:
    """Staggered tranches alive at once: a tranche is held for L sessions and a new one is opened every ``every`` sessions, so ``ceil(L / every)`` overlap."""
    return max(1, int(math.ceil(int(holding) / max(int(every), 1))))
