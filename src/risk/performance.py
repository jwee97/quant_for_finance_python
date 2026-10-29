"""Risk and performance measures of a return series that the Backtest page shows beside the Sharpe ratio: pure functions over a daily net-return series (and a benchmark's).

Sortino (Sortino and Price 1994), Calmar (Young 1991), Omega (Keating and Shadwick 2002), the Ulcer index and Martin ratio (Martin and McCann 1989), the tail ratio, Schwager's gain-to-pain ratio, historical and
Cornish-Fisher value at risk (Favre and Galeano 2002) with expected shortfall, the Kelly leverage (Thorp), the probabilistic Sharpe ratio and the minimum track record length (Bailey and Lopez de Prado 2012),
the worst drawdown episodes, rolling volatility and beta, the average return of each calendar month, and a benchmark's beta, alpha, capture ratios, tracking error and information ratio.

Conventions. Returns are simple daily returns, 252 days to a year, a risk-free rate of zero (the page reports excess-of-nothing Sharpe ratios too). Losses (``var``, ``es``) are positive numbers. A ratio whose
denominator is zero is ``None`` (no losing day, no drawdown), not infinity: the page prints "n/a".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

ANN = 252.0
Z95 = float(norm.ppf(0.95))


def _clean(net: pd.Series) -> pd.Series:
    return net.dropna().astype(float)


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 and math.isfinite(num) and math.isfinite(den) else None


def drawdown_series(net: pd.Series) -> pd.Series:
    equity = (1.0 + _clean(net)).cumprod()
    return equity / equity.cummax() - 1.0


def cornish_fisher_var(mean: float, sd: float, skew: float, excess_kurtosis: float, level: float = 0.95) -> float:
    """The loss not exceeded with probability ``level`` when the distribution has this skewness and excess kurtosis: the normal quantile corrected by the first terms of the Cornish-Fisher expansion
    (Favre and Galeano 2002), ``z + (z^2 - 1) S / 6 + (z^3 - 3z) K / 24 - (2 z^3 - 5 z) S^2 / 36`` with ``z`` the normal quantile of ``1 - level``."""
    z = float(norm.ppf(1.0 - level))
    z_cf = z + (z ** 2 - 1.0) * skew / 6.0 + (z ** 3 - 3.0 * z) * excess_kurtosis / 24.0 - (2.0 * z ** 3 - 5.0 * z) * skew ** 2 / 36.0
    return float(-(mean + sd * z_cf))


def min_track_record_years(daily_sharpe: float, skew: float, kurtosis: float, confidence: float = 0.95, benchmark: float = 0.0) -> float | None:
    """How many years of daily returns the observed Sharpe ratio needs before it is above ``benchmark`` (a per-day Sharpe) at the given confidence (Bailey and Lopez de Prado 2012):
    ``1 + (1 - S sr + (K - 1) sr^2 / 4) (z / (sr - sr*))^2`` observations, with ``K`` the raw (not excess) kurtosis. ``None`` when the Sharpe ratio is not above the benchmark."""
    if not daily_sharpe > benchmark:
        return None
    z = float(norm.ppf(confidence))
    factor = 1.0 - skew * daily_sharpe + (kurtosis - 1.0) / 4.0 * daily_sharpe ** 2
    if factor <= 0:
        return None
    return float((1.0 + factor * (z / (daily_sharpe - benchmark)) ** 2) / ANN)


def risk_metrics(net: pd.Series) -> dict:
    """The measures above for one return series."""
    r = _clean(net)
    n = len(r)
    if n < 30:
        return {}
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    cagr = float(equity.iloc[-1] ** (ANN / n) - 1.0)
    downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)))
    gains, losses = float(r[r > 0].sum()), float(-r[r < 0].sum())
    monthly = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    q05, q95 = float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95))
    tail = r[r <= q05]
    q01 = float(np.quantile(r, 0.01))
    skew, excess = float(r.skew()), float(r.kurt())
    ulcer = float(np.sqrt(np.mean((100.0 * dd.to_numpy()) ** 2)))
    sharpe_d = mean / sd if sd > 0 else float("nan")
    out = {
        "sortino": _ratio(math.sqrt(ANN) * mean, downside), "calmar": _ratio(cagr, -float(dd.min())), "omega": _ratio(gains, losses),
        "ulcer_index": ulcer, "martin_ratio": _ratio(100.0 * cagr, ulcer), "tail_ratio": _ratio(abs(q95), abs(q05)),
        "gain_to_pain": _ratio(float(monthly.sum()), float(-monthly[monthly < 0].sum())),
        "var_95": -q05, "es_95": -float(tail.mean()) if len(tail) else None, "var_99": -q01, "cornish_fisher_var_95": cornish_fisher_var(mean, sd, skew, excess, 0.95),
        "kelly_leverage": _ratio(mean, sd ** 2), "recovery_factor": _ratio(float(equity.iloc[-1] - 1.0), -float(dd.min())),
        "average_drawdown": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0, "pain_index": float(-dd.mean()),
        "daily_downside_deviation": downside, "positive_days": float((r > 0).mean()), "skew": skew, "excess_kurtosis": excess,
        "min_track_record_years": min_track_record_years(sharpe_d, skew, excess + 3.0) if math.isfinite(sharpe_d) else None,
    }
    return out


def drawdown_episodes(net: pd.Series, limit: int = 5) -> list[dict]:
    """The ``limit`` deepest falls from a peak, each from the peak to the day the peak was regained (``recovered`` is ``None`` for a fall still open). Episodes do not overlap: a new one starts only after
    the previous peak is regained. ``days`` count calendar days; ``depth`` is negative."""
    r = _clean(net)
    if r.empty:
        return []
    equity = (1.0 + r).cumprod()
    peak_value, peak_date, episodes = 1.0, r.index[0], []
    trough_value, trough_date, below = 1.0, None, False
    start_value = 1.0                                            # the account before the first return counts as a peak of 1
    peak_value, peak_date = start_value, r.index[0] - pd.Timedelta(days=1)
    for date, value in equity.items():
        if value >= peak_value:
            if below:
                episodes.append({"peak": peak_date, "trough": trough_date, "recovered": date, "depth": trough_value / peak_value - 1.0})
                below = False
            peak_value, peak_date = value, date
        else:
            if not below:
                below, trough_value, trough_date = True, value, date
            elif value < trough_value:
                trough_value, trough_date = value, date
    if below:
        episodes.append({"peak": peak_date, "trough": trough_date, "recovered": None, "depth": trough_value / peak_value - 1.0})
    episodes.sort(key=lambda e: e["depth"])
    out = []
    for e in episodes[:limit]:
        end = e["recovered"] if e["recovered"] is not None else r.index[-1]
        out.append({"peak": str(e["peak"].date()), "trough": str(e["trough"].date()), "recovered": None if e["recovered"] is None else str(e["recovered"].date()), "depth": float(e["depth"]),
                    "days_to_trough": int((e["trough"] - e["peak"]).days), "days": int((end - e["peak"]).days)})
    return out


def rolling_volatility(net: pd.Series, window: int = 63) -> pd.Series:
    """Annualised standard deviation of the last ``window`` daily returns."""
    return _clean(net).rolling(window, min_periods=window).std(ddof=1) * math.sqrt(ANN)


def rolling_beta(net: pd.Series, bench: pd.Series, window: int = 252) -> pd.Series:
    """Beta of the strategy to a benchmark over the last ``window`` days: covariance over the benchmark's variance."""
    both = pd.concat([_clean(net), _clean(bench)], axis=1, join="inner").dropna()
    both.columns = ["r", "b"]
    return both["r"].rolling(window, min_periods=window).cov(both["b"]) / both["b"].rolling(window, min_periods=window).var()


def calendar_month_profile(net: pd.Series) -> list[dict]:
    """For each calendar month: how many Januaries (and so on) the backtest holds, their average and best and worst return, and the share that were gains."""
    r = _clean(net)
    monthly = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    out = []
    for month in range(1, 13):
        v = monthly[monthly.index.get_level_values(1) == month]
        if len(v):
            out.append({"month": month, "n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()), "best": float(v.max()), "worst": float(v.min()), "hit_rate": float((v > 0).mean())})
    return out


def benchmark_relative(net: pd.Series, bench: pd.Series) -> dict:
    """The strategy against a benchmark on the days both have a return: correlation, beta, annual alpha (the strategy's mean less beta times the benchmark's, times 252), tracking error and information
    ratio of the difference, and the capture ratios (the strategy's average return on the days the benchmark rose or fell, over the benchmark's)."""
    both = pd.concat([_clean(net), _clean(bench)], axis=1, join="inner").dropna()
    if len(both) < 30:
        return {}
    r, b = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(r, b, ddof=1)[0, 1] / var_b) if var_b > 0 else None
    active = r - b
    te = float(np.std(active, ddof=1) * math.sqrt(ANN))
    up, down = b > 0, b < 0
    return {
        "n_days": int(len(both)), "correlation": float(np.corrcoef(r, b)[0, 1]) if np.std(r) > 0 and np.std(b) > 0 else None, "beta": beta,
        "alpha_annual": None if beta is None else float((r.mean() - beta * b.mean()) * ANN),
        "tracking_error": te, "information_ratio": _ratio(float(active.mean() * ANN), te),
        "up_capture": _ratio(float(r[up].mean()), float(b[up].mean())) if up.any() else None,
        "down_capture": float(r[down].mean() / b[down].mean()) if down.any() and b[down].mean() < 0 else None,
    }
