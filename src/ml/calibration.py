"""Probability and interval forecast evaluation, calibration and time-series conformal prediction (spec v3, Release 5, P1-22).

* Proper scoring rules and reliability for class probabilities (``brier``, ``log_loss``, ``reliability``) and for quantile / interval forecasts (``pinball``, ``interval_scores``: coverage, width and the
  interval score), reported by *eligible segment*; a segment with too few observations is ``insufficient``, not a pass.
* Calibrators (``PlattCalibrator``, ``IsotonicCalibrator``) are fitted on a calibration set that is *earlier in time* than the data they are applied to (``walk_forward_calibrate`` refits at predeclared
  dates on rows whose labels had already ended), never on the evaluation data.
* Conformal intervals for time series. ``rolling_split_conformal`` is the textbook split conformal on the most recent calibration residuals; its finite-sample coverage guarantee needs the calibration and
  test residuals to be exchangeable, which a market series does not give, so the result is labelled *without a coverage guarantee* and its empirical coverage is reported with an interval.
  ``adaptive_conformal`` is Adaptive Conformal Inference (Gibbs & Candes 2021): the miscoverage level is nudged after every outcome, which guarantees the LONG-RUN average miscoverage is within
  ``(max(a1, 1 - a1) + gamma) / (gamma T)`` of ``alpha`` for ANY sequence. That is a guarantee about the average over time, not about coverage in every regime or conditional on the features, and
  nothing here promises more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

VERSION = "calibration-v1"
MIN_SEGMENT = 30


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """The Wilson score interval for a proportion (used for empirical coverage and for reliability bins)."""
    if n == 0:
        return float("nan"), float("nan")
    z = float(sps.norm.ppf(0.5 + level / 2.0))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


# ------------------------------------------------------------------------------------------------------------------------------------ class probabilities
def brier(p, y) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-12) -> float:
    p, y = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps), np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def reliability(p, y, bins: int = 10) -> dict:
    """Equal-width reliability bins: the mean forecast, the observed frequency (with a Wilson interval) and the count in each bin, and the expected calibration error (count-weighted mean absolute gap)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, gap = [], 0.0
    for b in range(bins):
        m = which == b
        n = int(m.sum())
        if n == 0:
            continue
        k, mp = int(y[m].sum()), float(p[m].mean())
        lo, hi = wilson(k, n)
        rows.append({"bin": b, "mean_forecast": mp, "frequency": k / n, "frequency_interval": [lo, hi], "n": n})
        gap += n * abs(mp - k / n)
    return {"bins": rows, "ece": float(gap / len(p)) if len(p) else float("nan"), "n": int(len(p))}


def probability_report(p: pd.Series, y: pd.Series, segments: pd.Series, *, min_segment: int = MIN_SEGMENT, bins: int = 10) -> dict:
    """Brier, log loss and reliability per eligible segment, and the base-rate forecast as the reference (a forecaster that predicts the training base rate everywhere)."""
    out: dict = {"version": VERSION, "segments": {}}
    for seg in pd.unique(segments.dropna()):
        m = (segments == seg).to_numpy()
        pp, yy = p.to_numpy(dtype=float)[m], y.to_numpy(dtype=float)[m]
        ok = np.isfinite(pp) & np.isfinite(yy)
        pp, yy = pp[ok], yy[ok]
        if len(pp) < min_segment:
            out["segments"][str(seg)] = {"eligibility": "insufficient", "n": int(len(pp)), "reason": f"{len(pp)} forecasts: fewer than {min_segment}"}
            continue
        base = float(yy.mean())
        out["segments"][str(seg)] = {"eligibility": "applied", "n": int(len(pp)), "brier": brier(pp, yy), "brier_base_rate": brier(np.full_like(pp, base), yy), "log_loss": log_loss(pp, yy),
                                     "log_loss_base_rate": log_loss(np.full_like(pp, np.clip(base, 1e-6, 1 - 1e-6)), yy), "reliability": reliability(pp, yy, bins)}
    return out


class PlattCalibrator:
    """Logistic recalibration of a probability: ``sigmoid(a * logit(p) + b)``, fitted by maximum likelihood on the calibration set."""

    def fit(self, p, y) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        z = _logit(np.asarray(p, dtype=float)).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("the calibration set holds one class only: a calibrator cannot be fitted")
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
        self.a_, self.b_ = float(m.coef_[0, 0]), float(m.intercept_[0])
        return self

    def predict(self, p) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a_ * _logit(np.asarray(p, dtype=float)) + self.b_)))


class IsotonicCalibrator:
    """Monotone recalibration (pool-adjacent-violators), clipped to the open interval so a log loss stays finite."""

    def fit(self, p, y) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        y = np.asarray(y, dtype=float)
        if len(np.unique(y)) < 2:
            raise ValueError("the calibration set holds one class only: a calibrator cannot be fitted")
        self.m_ = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(np.asarray(p, dtype=float), y)
        return self

    def predict(self, p) -> np.ndarray:
        return self.m_.predict(np.asarray(p, dtype=float))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def walk_forward_calibrate(p: pd.Series, y: pd.Series, label_end: pd.Series, refit_dates: list, *, method: str = "platt", min_calibration: int = 200) -> pd.Series:
    """Recalibrated probabilities where each block is calibrated by a model fitted ONLY on forecasts whose labels had ended before the block began (``label_end < refit date``). Blocks with too little
    calibration data are left uncalibrated (``NaN``): a calibrator is never fitted on what it is then scored on. ``p``, ``y`` and ``label_end`` share an index of dates."""
    out = pd.Series(np.nan, index=p.index)
    dates = list(pd.DatetimeIndex(refit_dates))
    bounds = dates + [p.index.max() + pd.Timedelta(days=1)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        cal = (label_end < a) & p.notna() & y.notna()
        block = (p.index >= a) & (p.index < b)
        if cal.sum() < min_calibration or not block.any():
            continue
        try:
            cal_model = (PlattCalibrator() if method == "platt" else IsotonicCalibrator()).fit(p[cal], y[cal])
        except ValueError:
            continue
        out[block] = cal_model.predict(p[block].to_numpy())
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ quantiles and intervals
def pinball(pred, y, tau: float) -> float:
    """The pinball (quantile) loss of a ``tau``-quantile forecast."""
    pred, y = np.asarray(pred, dtype=float), np.asarray(y, dtype=float)
    d = y - pred
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def interval_scores(lower, upper, y, alpha: float) -> dict:
    """Empirical coverage of ``[lower, upper]`` (with a Wilson interval, to be read against the nominal ``1 - alpha``), the mean width, and the interval score (Gneiting & Raftery)."""
    lo, hi, y = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
    lo, hi, y = lo[ok], hi[ok], y[ok]
    n = int(len(y))
    if n == 0:
        return {"eligibility": "insufficient", "n": 0, "reason": "no forecast with an interval"}
    inside = (y >= lo) & (y <= hi)
    score = (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)
    cl, ch = wilson(int(inside.sum()), n)
    return {"eligibility": "applied", "n": n, "nominal": 1.0 - alpha, "coverage": float(inside.mean()), "coverage_interval": [cl, ch], "width": float(np.mean(hi - lo)), "interval_score": float(score.mean())}


# ------------------------------------------------------------------------------------------------------------------------------------ conformal
def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def rolling_split_conformal(pred: pd.Series, y: pd.Series, *, alpha: float = 0.1, window: int = 250, min_calibration: int = 50, delay: int = 1) -> dict:
    """Intervals ``pred +- q`` where ``q`` is the conformal quantile of the absolute residuals of the last ``window`` forecasts whose outcomes were KNOWN when the forecast was made (outcome ``delay`` rows
    old at least). The finite-sample guarantee of split conformal needs exchangeability, which time series do not provide: the result says so and gives the empirical coverage."""
    resid = (y - pred).abs()
    q = pd.Series(np.nan, index=pred.index)
    for i in range(len(pred)):
        hist = resid.iloc[max(0, i - delay - window + 1): max(0, i - delay + 1)].dropna().to_numpy()
        if len(hist) >= min_calibration:
            q.iloc[i] = _conformal_quantile(hist, alpha)
    lo, hi = pred - q, pred + q
    return {"method": "rolling split conformal", "guarantee": "none for time series: finite-sample coverage needs exchangeable calibration and test residuals", "alpha": alpha, "window": window, "lower": lo, "upper": hi,
            "scores": interval_scores(lo.to_numpy(), hi.to_numpy(), y.to_numpy(), alpha)}


def adaptive_conformal(pred: pd.Series, y: pd.Series, *, alpha: float = 0.1, gamma: float = 0.01, window: int = 250, min_calibration: int = 50, delay: int = 1) -> dict:
    """Adaptive Conformal Inference. The level used on day ``t`` is ``alpha_t``; after the outcome, ``alpha_{t+1} = alpha_t + gamma (alpha - err_t)`` where ``err_t`` is 1 if the interval missed. The interval is
    ``pred +- q_t`` with ``q_t`` the ``1 - alpha_t`` conformal quantile of the recent residuals (infinite when ``alpha_t <= 0``, empty when ``alpha_t >= 1``). Guarantee: over ``T`` forecasts the average
    miscoverage differs from ``alpha`` by at most ``(max(alpha_1, 1 - alpha_1) + gamma) / (gamma T)``, for any sequence of outcomes."""
    resid = (y - pred).abs()
    a_t = alpha
    lo = pd.Series(np.nan, index=pred.index)
    hi = pd.Series(np.nan, index=pred.index)
    err = pd.Series(np.nan, index=pred.index)
    levels = pd.Series(np.nan, index=pred.index)
    for i in range(len(pred)):
        if not np.isfinite(pred.iloc[i]):
            continue
        hist = resid.iloc[max(0, i - delay - window + 1): max(0, i - delay + 1)].dropna().to_numpy()
        if len(hist) < min_calibration:
            continue
        levels.iloc[i] = a_t
        if a_t <= 0:
            q = float("inf")
        elif a_t >= 1:
            q = 0.0
        else:
            q = _conformal_quantile(hist, a_t)
        lo.iloc[i], hi.iloc[i] = pred.iloc[i] - q, pred.iloc[i] + q
        if np.isfinite(y.iloc[i]):
            miss = float(not (lo.iloc[i] <= y.iloc[i] <= hi.iloc[i]))
            err.iloc[i] = miss
            a_t = a_t + gamma * (alpha - miss)
    used = err.dropna()
    T = len(used)
    bound = (max(alpha, 1.0 - alpha) + gamma) / (gamma * T) if T else float("nan")
    return {"method": "adaptive conformal inference", "guarantee": f"long-run average miscoverage within {bound:.4f} of alpha over {T} forecasts, for any sequence; not conditional coverage",
            "alpha": alpha, "gamma": gamma, "window": window, "lower": lo, "upper": hi, "levels": levels, "average_miscoverage": float(used.mean()) if T else float("nan"), "bound": bound, "n": int(T),
            "scores": interval_scores(lo.to_numpy(), hi.to_numpy(), y.to_numpy(), alpha)}
