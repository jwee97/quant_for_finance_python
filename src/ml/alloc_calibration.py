"""Calibrating a model's forecast against past outcomes for the portfolio allocation: how much of the forecast to believe, for which asset, and how sure that belief is.

The baseline is one slope through the origin pooled over all assets (``pooled``), kept as the small-sample default. The alternatives, all estimated from resolved forecasts only (a forecast whose outcome had ended
at or before the decision date):

``asset``            an own slope per asset, used only for an asset with at least ``calib_asset_min_eff`` independent outcome windows; every other asset takes the pooled slope.
``partial_pooling``  hierarchical (empirical-Bayes) shrinkage: each asset's slope is pulled toward the pooled slope by ``tau^2 / (tau^2 + se_a^2)``, where ``tau^2`` is the between-asset variance of the true slopes estimated
                     from the dispersion of the asset slopes beyond their sampling error (DerSimonian-Laird). ``tau^2`` needs at least three assets: with two the model cannot tell a real difference between the stocks
                     from noise, so it falls back to the pooled slope and says so.
``sector_pooling``   the same with a sector level between: assets shrink toward their sector's slope, sectors toward the pooled slope (needs three sectors for the second step, and a group map).
``regularized``      the pooled slope shrunk toward zero (no edge) by a normal prior of standard deviation ``calib_prior_sd``: ``b * tau0^2 / (tau0^2 + se^2)``. A noisy slope loses more of its size than a precise one.

Standard errors. The legacy rule inflates the iid error by the square root of the horizon (``sqrt_horizon``). The dependence-aware alternatives work on the per-date score ``u_t = sum_a x_ta e_ta`` (one number per date,
so the correlation between assets on a date is inside it):

``hac``              the Newey-West long-run variance of ``u_t`` with at least ``horizon - 1`` lags (outcomes that overlap share days; consecutive residuals are serially correlated by construction);
``block_bootstrap``  the standard deviation of the slope over stationary block resamples of the dates (block length at least the horizon), which keeps serial dependence of any form.

The effective sample size is the number of dates times ``gamma_0 / LRV(u)``, capped at the number of dates: how many independent observations the overlapping, cross-correlated pairs are worth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..probability.resampling import bootstrap_indices
from .inference import long_run_variance, overlap_lags

METHODS = ("pooled", "asset", "partial_pooling", "sector_pooling", "regularized")
INFERENCE = ("sqrt_horizon", "hac", "block_bootstrap")
N_BOOT = 300


@dataclass
class Calibration:
    slopes: pd.Series                  # the slope applied to each asset's forecast (0 where there is no demonstrated edge)
    se: pd.Series
    t: pd.Series
    confidence: pd.Series              # probability that the asset's true slope is positive given the estimate: Phi(b / se), 0 where there is no edge
    pooled_slope: float
    pooled_se: float
    pooled_t: float
    n_pairs: int
    n_dates: int
    n_eff: float
    method: str                        # the method asked for
    used: dict = field(default_factory=dict)         # asset -> what was actually used (pooled / asset / shrunk / sector)
    inference: str = "sqrt_horizon"
    tau: float | None = None
    fallback: str = ""
    note: str = ""


def _score_series(x: np.ndarray, y: np.ndarray, b: float) -> np.ndarray:
    """Per-date score ``sum_a x_ta (y_ta - b x_ta)`` of a (dates x assets) panel with NaN where an asset has no outcome."""
    e = np.where(np.isfinite(x) & np.isfinite(y), x * (y - b * x), 0.0)
    return e.sum(axis=1)


def slope_inference(x: np.ndarray, y: np.ndarray, h: int, inference: str, seed: int = 0) -> tuple[float, float, float]:
    """Slope through the origin of a (dates x assets) panel, its standard error under ``inference`` and the effective number of independent observations."""
    ok = np.isfinite(x) & np.isfinite(y)
    xv, yv = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    sxx = float((xv ** 2).sum())
    n = int(ok.sum())
    dates = int(ok.any(axis=1).sum())
    if sxx <= 0 or n < 3:
        return 0.0, float("nan"), 0.0
    b = float((xv * yv).sum() / sxx)
    if inference == "sqrt_horizon":
        resid = (yv - b * xv)[ok]
        se = float(math.sqrt(resid.var(ddof=1) / sxx * h)) if n > 2 else float("nan")
        return b, se, n / float(h)
    u = _score_series(x, y, b)
    T = len(u)
    if inference == "hac":
        lags = overlap_lags(T, h)
        lrv = long_run_variance(u - 0.0, lags)                                     # the score has mean ~ 0 at the least-squares slope
        var_b = T * max(lrv, 0.0) / sxx ** 2
        g0 = float(u @ u) / T
        n_eff = float(np.clip(dates * (g0 / lrv) if lrv > 0 else dates, 1.0, dates))
        return b, float(math.sqrt(var_b)), n_eff
    if inference == "block_bootstrap":
        rng = np.random.default_rng(seed)
        block = max(float(h), 5.0)
        draws = []
        for _ in range(N_BOOT):
            idx = bootstrap_indices(T, "stationary", block, int(rng.integers(0, 2 ** 31 - 1)))
            xs, ys = xv[idx], yv[idx]
            s2 = float((xs ** 2).sum())
            if s2 > 0:
                draws.append(float((xs * ys).sum() / s2))
        se = float(np.std(draws, ddof=1)) if len(draws) > 10 else float("nan")
        lrv_se = se ** 2 * sxx ** 2 / T if np.isfinite(se) else float("nan")
        g0 = float(u @ u) / T
        n_eff = float(np.clip(dates * (g0 / lrv_se) if np.isfinite(lrv_se) and lrv_se > 0 else dates, 1.0, dates))
        return b, se, n_eff
    raise ValueError(f"inference must be one of {list(INFERENCE)}")


def _eb_tau2(b: np.ndarray, se: np.ndarray) -> tuple[float, float]:
    """Between-group variance of true slopes (DerSimonian-Laird) and the precision-weighted mean slope."""
    w = 1.0 / se ** 2
    bar = float((w * b).sum() / w.sum())
    q = float((w * (b - bar) ** 2).sum())
    denom = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (len(b) - 1)) / denom) if denom > 0 else 0.0
    return tau2, bar


def calibrate(xs: pd.DataFrame, ys: pd.DataFrame, cfg, h: int, groups: dict | None = None) -> Calibration:
    """Calibration of forecasts ``xs`` against realised window returns ``ys`` (dates x assets, resolved outcomes only) by the method and inference in ``cfg``."""
    method = getattr(cfg, "calib_method", "pooled")
    inference = getattr(cfg, "calib_inference", "sqrt_horizon")
    min_eff = float(getattr(cfg, "calib_asset_min_eff", 30.0))
    min_t = float(cfg.calib_min_t)
    prior_sd = float(getattr(cfg, "calib_prior_sd", 0.5))
    cols = list(xs.columns)
    X, Y = xs.to_numpy(dtype=float), ys.reindex(index=xs.index, columns=cols).to_numpy(dtype=float)
    ok = np.isfinite(X) & np.isfinite(Y)
    b0, se0, neff0 = slope_inference(X, Y, h, inference)
    t0 = b0 / se0 if np.isfinite(se0) and se0 > 0 else float("nan")
    n_pairs, n_dates = int(ok.sum()), int(ok.any(axis=1).sum())
    fallback = ""
    used = {c: "pooled" for c in cols}
    slope_a = {c: b0 for c in cols}
    se_a = {c: se0 for c in cols}
    tau = None
    # per-asset estimates, where there is enough to make one
    own: dict = {}
    if method in ("asset", "partial_pooling", "sector_pooling"):
        for j, c in enumerate(cols):
            ba, sea, na = slope_inference(X[:, [j]], Y[:, [j]], h, inference, seed=j + 1)
            if np.isfinite(sea) and sea > 0 and na >= min_eff:
                own[c] = (ba, sea, na)
        short = [c for c in cols if c not in own]
        if short:
            fallback = f"{', '.join(map(str, short))} had fewer than {min_eff:g} independent outcome windows for a slope of its own, so the pooled slope is used for {'it' if len(short) == 1 else 'them'}"
    if method == "asset":
        for c, (ba, sea, _) in own.items():
            slope_a[c], se_a[c], used[c] = ba, sea, "asset"
    elif method == "partial_pooling":
        if len(own) < 3:
            fallback = (fallback + "; " if fallback else "") + f"partial pooling needs at least three assets with enough history to tell real differences between assets from noise; {len(own)} qualify, so the pooled slope is used for all"
        else:
            b = np.array([own[c][0] for c in own])
            s = np.array([own[c][1] for c in own])
            tau2, bar = _eb_tau2(b, s)
            tau = math.sqrt(tau2)
            for c, (ba, sea, _) in own.items():
                B = tau2 / (tau2 + sea ** 2)
                slope_a[c] = b0 + B * (ba - b0)
                se_a[c] = math.sqrt(B * sea ** 2 + (1.0 - B) ** 2 * se0 ** 2) if np.isfinite(se0) else sea
                used[c] = "shrunk"
    elif method == "sector_pooling":
        if not groups:
            fallback = (fallback + "; " if fallback else "") + "sector pooling needs a sector map for the universe and none is available: the pooled slope is used"
        else:
            sectors: dict = {}
            for j, c in enumerate(cols):
                sectors.setdefault(groups.get(c, "_none"), []).append(j)
            sec_est = {}
            for g, js in sectors.items():
                bs, ses, ns = slope_inference(X[:, js], Y[:, js], h, inference, seed=100 + len(sec_est))
                if np.isfinite(ses) and ses > 0 and ns >= min_eff:
                    sec_est[g] = (bs, ses)
            if len(sec_est) >= 3:
                bb = np.array([v[0] for v in sec_est.values()])
                ss = np.array([v[1] for v in sec_est.values()])
                tau2s, _ = _eb_tau2(bb, ss)
                tau = math.sqrt(tau2s)
                for g, (bs, ses) in sec_est.items():
                    B = tau2s / (tau2s + ses ** 2)
                    sec_est[g] = (b0 + B * (bs - b0), math.sqrt(B * ses ** 2 + (1 - B) ** 2 * se0 ** 2) if np.isfinite(se0) else ses)
                for g, js in sectors.items():
                    if g in sec_est:
                        for j in js:
                            slope_a[cols[j]], se_a[cols[j]], used[cols[j]] = sec_est[g][0], sec_est[g][1], "sector"
            else:
                fallback = (fallback + "; " if fallback else "") + f"sector pooling needs at least three sectors with enough history; {len(sec_est)} qualify, so the pooled slope is used"
    elif method == "regularized":
        if np.isfinite(se0) and se0 > 0:
            f = prior_sd ** 2 / (prior_sd ** 2 + se0 ** 2)
            slope_a = {c: b0 * f for c in cols}
            se_a = {c: se0 * math.sqrt(f) for c in cols}
            used = {c: "regularized" for c in cols}
    slopes, ses_out, ts, conf = {}, {}, {}, {}
    for c in cols:
        b, se = slope_a[c], se_a[c]
        t = b / se if np.isfinite(se) and se > 0 else float("nan")
        ts[c], ses_out[c] = t, se
        edge = np.isfinite(t) and b > 0 and t >= min_t
        slopes[c] = float(min(b, 1.0)) if edge else 0.0
        conf[c] = float(norm.cdf(t)) if edge else 0.0
    note = f"{method} calibration, {inference} standard errors; {n_pairs} resolved forecasts on {n_dates} dates are worth {neff0:.0f} independent observations"
    return Calibration(pd.Series(slopes), pd.Series(ses_out), pd.Series(ts), pd.Series(conf), float(b0), float(se0), float(t0), n_pairs, n_dates, float(neff0), method, used, inference, tau, fallback, note)
