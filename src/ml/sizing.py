"""Sizing from calibrated forecasts, with the uncertainty of the calibration (spec v3, Release 7, P2-06).

A model's score is not a return. A cross-sectional z-score has no units, and a ridge prediction is a return only if the model happens to be calibrated. Sizing a book in proportion to either, as if it were an
expected return, says nothing about how much risk the edge deserves. Here the score is **calibrated walk-forward** against what followed it, and the book is sized from the calibrated forecast, its
uncertainty, the asset's risk and how stable the signal has been:

1. *Calibration.* On each decision date ``t`` the cross-sectional slope ``b_s`` of the demeaned forward return on the demeaned score is computed for every earlier decision ``s`` whose outcome had settled
   (``s + label_offset <= t``); the calibration ``b_t`` is the mean of the last ``calib_window`` of them and its standard error is Newey-West (the labels overlap, so the lag is at least the horizon less one).
   The calibrated forecast of asset ``i`` is ``b_t * x_i`` in return units over the holding window, with ``x`` the demeaned score.
2. *Edge.* If ``b_t`` is not positive, or its t-statistic is below ``calib_min_t``, the model has not shown an edge and the book is **flat**. Otherwise the forecast is shrunk by
   ``g = max(0, 1 - 1 / t^2)``: the larger the uncertainty of the calibration, the smaller the bet, and a large enough uncertainty brings it to zero.
3. *Size.* ``w_i = g * b_t * x_i / (lambda * sigma_i^2) * stability``, with ``sigma_i^2`` the trailing variance of the asset scaled to the window and ``stability`` the mean cross-sectional rank autocorrelation of
   consecutive scores over the last ``stability_window`` dates, clipped to [0, 1] (a signal that reshuffles every day is trusted less).
4. *Limits.* The book is shifted to the configured net exposure with each position held inside ``max_weight`` (``project``), and scaled down, net included, if its gross exceeds ``gross``; **a book is never scaled up to
   fill the gross**: how much risk to take is the result of the calibration, not an input.

Everything on date ``t`` uses information available at ``t`` (the scores up to ``t``, the returns up to ``t``, and outcomes settled by ``t``); the tests change the future and check that the past does not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import inference as inf
from .targets import forward_return

VERSION = "sizing-v1"
EPS = 1e-12


def cross_sectional_slopes(score: pd.DataFrame, fwd: pd.DataFrame, min_assets: int = 5) -> pd.Series:
    """Per decision date, the slope of the demeaned forward return on the demeaned score (return per unit of score); NaN where fewer than ``min_assets`` assets have both or the score is constant."""
    both = score.notna() & fwd.notna()
    n = both.sum(axis=1)
    x, y = score.where(both), fwd.where(both)
    xc, yc = x.sub(x.mean(axis=1), axis=0), y.sub(y.mean(axis=1), axis=0)
    sxx = (xc ** 2).sum(axis=1)
    slope = (xc * yc).sum(axis=1) / sxx.where(sxx > EPS)
    return slope.where(n >= min_assets)


def calibration(score: pd.DataFrame, returns: pd.DataFrame, horizon: int, lag: int, window: int = 504, min_dates: int = 60) -> pd.DataFrame:
    """The walk-forward calibration by decision date: ``slope`` (mean of the settled slopes in the window), ``se`` (Newey-West, lag at least ``horizon - 1``), ``t`` and the number of settled dates used."""
    prices = (1.0 + returns.fillna(0.0)).cumprod()
    fwd = forward_return(prices, horizon, lag).reindex(index=score.index, columns=score.columns)
    slopes = cross_sectional_slopes(score, fwd)
    offset = int(lag) + int(horizon)
    settled = slopes.shift(offset)                                                                # decision s is known to have settled `offset` sessions later
    vals = settled.to_numpy()
    out = np.full((len(score), 4), np.nan)
    L = max(int(horizon) - 1, 0)
    for i in range(len(score)):
        w = vals[max(0, i - int(window) + 1): i + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_dates:
            continue
        mu = float(w.mean())
        lags = inf.overlap_lags(len(w), int(horizon), None)
        lrv = inf.long_run_variance(w - mu, max(lags, L))
        se = float(np.sqrt(max(lrv, 0.0) / len(w)))
        out[i] = (mu, se, mu / se if se > 0 else np.nan, len(w))
    return pd.DataFrame(out, index=score.index, columns=["slope", "se", "t", "n"])


def project(w: np.ndarray, net: float, cap: float) -> np.ndarray:
    """The book closest to ``w`` (shifting every free name by the same amount) with the sum ``net`` and every position within ``[-cap, cap]``: shift, clip, and shift the names that are not at a limit again until the
    sum is right. If the limits make ``net`` unreachable the closest reachable book is returned."""
    out = np.asarray(w, dtype=float) - (float(np.sum(w)) - net) / len(w)
    out = np.clip(out, -cap, cap)
    for _ in range(100):
        excess = float(out.sum()) - net
        if abs(excess) < 1e-12:
            break
        free = (out > -cap + 1e-12) if excess > 0 else (out < cap - 1e-12)                       # a name can give up weight if it is above its lower limit, and take weight if below its upper one
        if not free.any():
            break
        out = out.copy()
        out[free] -= excess / free.sum()
        out = np.clip(out, -cap, cap)
    return out


def rank_stability(score: pd.DataFrame, window: int = 21) -> pd.Series:
    """The mean over the last ``window`` dates of the cross-sectional Spearman correlation between a date's score and the previous date's, clipped to [0, 1]."""
    r = score.rank(axis=1)
    prev = r.shift(1)
    both = r.notna() & prev.notna()
    a, b = r.where(both), prev.where(both)
    ac, bc = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    den = np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
    rho = (ac * bc).sum(axis=1) / den.where(den > EPS)
    return rho.rolling(int(window), min_periods=max(5, int(window) // 3)).mean().clip(lower=0.0, upper=1.0)


def apply(score: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, dict]:
    days = int(cfg["window_days"])
    lag = int(cfg["label_offset"]) - days
    lam = float(cfg.get("risk_aversion", 5.0))
    min_t = float(cfg.get("calib_min_t", 2.0))
    cal = calibration(score, returns, days, lag, int(cfg.get("calib_window", 504)))
    stab = rank_stability(score, int(cfg.get("stability_window", 21)))
    var = returns.rolling(63, min_periods=40).var(ddof=1) * days
    gross_cap, pos_cap, net = float(cfg.get("gross", 1.0)), float(cfg.get("max_weight") or 0.35), float(cfg.get("net", 0.0))
    out = np.zeros(score.shape)
    cols = score.columns
    log = {"version": VERSION, "dates": 0, "sized": 0, "flat_no_edge": 0, "flat_not_calibrated": 0, "flat_zero_size": 0, "slopes": [], "t": [], "shrink": [], "stability": [], "gross": [], "capped": 0,
           "settings": {"calib_window": int(cfg.get("calib_window", 504)), "calib_min_t": min_t, "risk_aversion": lam, "window_days": days, "stability_window": int(cfg.get("stability_window", 21))}}
    for i, date in enumerate(score.index):
        row = score.iloc[i]
        ok = row.notna()
        if investable is not None:
            ok &= investable.loc[date].reindex(cols).fillna(False).to_numpy()
        if ok.sum() < 4:
            continue
        log["dates"] += 1
        b, t = cal.iloc[i]["slope"], cal.iloc[i]["t"]
        if not np.isfinite(t):
            log["flat_not_calibrated"] += 1                                                       # too few settled outcomes yet: nothing is sized on a guess
            continue
        if b <= 0 or t < min_t:
            log["flat_no_edge"] += 1
            continue
        g = max(0.0, 1.0 - 1.0 / (t * t))
        x = row[ok] - row[ok].mean()
        sig2 = var.iloc[i].reindex(x.index)
        sig2 = sig2.where(sig2 > EPS)
        s = stab.iloc[i]
        w = (g * b * x / (lam * sig2)).fillna(0.0) * (float(s) if np.isfinite(s) else 0.0)
        raw = w.to_numpy()
        if np.abs(raw).sum() <= 1e-9:                                                             # the stability or the risk reduced the bet to dust: the book is flat, and no net exposure is forced onto an empty book
            log["flat_zero_size"] += 1
            continue
        fit = project(raw, net, pos_cap)                                                          # the configured net exposure with every position inside its limit
        log["capped"] += int(np.any(np.abs(fit - (raw - (raw.sum() - net) / len(raw))) > 1e-12))
        w = pd.Series(fit, index=w.index)
        gross = float(w.abs().sum())
        if gross > gross_cap:
            w = w * (gross_cap / gross)
            gross = gross_cap
        out[i, [cols.get_loc(a) for a in w.index]] = w.to_numpy()
        log["sized"] += 1
        log["slopes"].append(float(b))
        log["t"].append(float(t))
        log["shrink"].append(float(g))
        log["stability"].append(float(s) if np.isfinite(s) else 0.0)
        log["gross"].append(gross)
    for key in ("slopes", "t", "shrink", "stability", "gross"):
        v = log.pop(key)
        log["mean_" + {"slopes": "slope", "t": "t", "shrink": "shrink", "stability": "stability", "gross": "gross"}[key]] = float(np.mean(v)) if v else None
    log["note"] = ("the score is calibrated against settled outcomes; with no demonstrated edge the book is flat, and the gross is whatever the calibrated edge and risk imply (never scaled up to a target)")
    return pd.DataFrame(out, index=score.index, columns=cols), log
