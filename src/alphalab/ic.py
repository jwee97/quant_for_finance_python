"""Information coefficients: how good a signal is, how steadily, in which regimes, and how fast it goes stale.

``IC_t`` is the cross-sectional correlation, on date ``t``, between the signal known at ``t`` and the return that follows. A forward window of ``h`` periods is only known ``h`` periods later: every function
here that turns ICs into something used at date ``t`` must use ICs of dates ``s <= t - h`` (see :func:`known_at`).

**Decay.** The IC of the signal against the return of the single period ``k`` periods after it is known, ``IC(k)``, is the cleanest measure of how long information lasts (the cumulative-return IC mixes the
decay with the horizon). Under an exponential model ``IC(k) = IC(1) exp(-(k-1)/tau)``: the half-life of the information is ``tau ln 2``, and the share of the information a trade placed ``d`` periods late
still captures is ``exp(-d/tau)``. :func:`fit_decay` fits it by weighted least squares with the sampling error of each lag as the weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..models.regression import cross_sectional_ic


def ic_series(signal: pd.DataFrame, forward: pd.DataFrame, method: str = "spearman", min_assets: int = 5) -> pd.Series:
    """The IC of every date: ``signal[t]`` against ``forward[t]`` (the return AFTER ``t``)."""
    return cross_sectional_ic(signal, forward, method=method, min_assets=min_assets)


def known_at(ic: pd.Series, horizon: int) -> pd.Series:
    """The IC series as it was KNOWN: the IC of date ``t`` needs the return of ``t+1 .. t+horizon``, so it is available at ``t + horizon``. Shifted so that row ``t`` holds only what was known at ``t``."""
    return ic.shift(int(horizon))


def _newey_west_se(x: np.ndarray, lags: int) -> float:
    """Standard error of the mean with a Bartlett kernel of ``lags`` lags (zero lags is the ordinary one)."""
    n = len(x)
    e = x - x.mean()
    var = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(var, 0.0) / n))


def ic_summary(ic: pd.Series, horizon: int = 1, periods_per_year: float = 252.0) -> dict:
    """Mean IC, its dispersion, the t-statistic of the mean (Newey-West with ``horizon - 1`` lags, because forward windows that overlap make neighbouring ICs dependent), the information ratio of the IC
    series, the hit rate, and how steady the sign is."""
    clean = ic.dropna()
    n = int(len(clean))
    if n < 3:
        return {"n": n, "mean": float("nan"), "std": float("nan"), "t": float("nan"), "ir": float("nan"), "ir_annual": float("nan"), "hit_rate": float("nan"), "stability": float("nan")}
    x = clean.to_numpy(dtype=float)
    mean, std = float(x.mean()), float(x.std(ddof=1))
    se = _newey_west_se(x, max(int(horizon) - 1, 0))
    return {"n": n, "mean": mean, "std": std, "t": mean / se if se > 0 else float("nan"), "ir": mean / std if std > 0 else float("nan"),
            "ir_annual": mean / std * np.sqrt(periods_per_year / max(horizon, 1)) if std > 0 else float("nan"), "hit_rate": float((x * np.sign(mean) > 0).mean()), "stability": stability(clean, horizon)}


def stability(ic: pd.Series, horizon: int = 1, blocks: int = 6) -> float:
    """The share of ``blocks`` equal consecutive blocks of the history in which the mean IC has the sign of the whole-sample mean (1 = it never flips; 0.5 = a coin). Blocks are not shorter than
    ``horizon`` periods, so each is independent of its neighbours."""
    x = ic.dropna().to_numpy(dtype=float)
    k = int(min(blocks, max(len(x) // max(int(horizon), 5), 1)))
    if len(x) < 2 or k < 2:
        return float("nan")
    sign = np.sign(x.mean())
    return float(np.mean([np.sign(b.mean()) == sign for b in np.array_split(x, k)])) if sign != 0 else float("nan")


def rolling_ic(ic: pd.Series, window: int = 63, min_periods: int | None = None) -> pd.DataFrame:
    """The rolling mean IC and its rolling information ratio."""
    r = ic.rolling(window, min_periods=min_periods or max(window // 2, 5))
    mean, std = r.mean(), r.std(ddof=1)
    return pd.DataFrame({"mean": mean, "ir": mean / std.where(std > 0)})


def ic_by_regime(ic: pd.Series, regime: pd.Series, horizon: int = 1) -> pd.DataFrame:
    """The IC summarised separately in each state of ``regime`` (labels on dates; a date without a label is dropped). One row per regime: count, mean, t, hit rate."""
    labels = regime.reindex(ic.index)
    rows = {}
    for state, part in ic.groupby(labels, dropna=True):
        s = ic_summary(part, horizon)
        rows[state] = {"n": s["n"], "mean": s["mean"], "t": s["t"], "ir": s["ir"], "hit_rate": s["hit_rate"]}
    return pd.DataFrame(rows).T.rename_axis("regime")


def lagged_ic(signal: pd.DataFrame, returns: pd.DataFrame, lags=(1, 2, 3, 5, 10, 21, 42, 63), method: str = "spearman", min_assets: int = 5) -> pd.DataFrame:
    """``IC(k)``: the signal of date ``t`` against the return of the single period ``t+k``, for each lag. Columns: mean IC, its standard error (ICs of different dates are weakly dependent here, the lag
    windows being single periods), t-statistic, and the number of dates."""
    rows = {}
    for k in lags:
        series = cross_sectional_ic(signal, returns.shift(-int(k)), method=method, min_assets=min_assets).dropna()
        n = len(series)
        se = float(series.std(ddof=1) / np.sqrt(n)) if n > 2 else float("nan")
        rows[int(k)] = {"ic": float(series.mean()) if n else float("nan"), "se": se, "t": float(series.mean() / se) if se and se > 0 else float("nan"), "n": n}
    return pd.DataFrame(rows).T.rename_axis("lag")


def fit_decay(curve: pd.DataFrame) -> dict:
    """Fit ``IC(k) = ic1 exp(-(k-1)/tau)`` to a :func:`lagged_ic` curve.

    Returns ``ic1`` (the IC at the first lag), ``tau`` and the ``half_life`` (``tau ln 2``, in periods), the R-squared of the fit, whether the first-lag IC is statistically distinguishable from zero (|t| > 2),
    and the number of periods after which ``90%`` of the information is gone. ``half_life`` is ``inf`` when the IC does not fall and ``nan`` when nothing can be fitted (fewer than three lags, or no
    significant IC at the first lag: a decay rate of noise is noise)."""
    c = curve.dropna(subset=["ic"])
    out = {"ic1": float("nan"), "tau": float("nan"), "half_life": float("nan"), "r2": float("nan"), "significant": False, "periods_to_90pct": float("nan"), "fitted": pd.Series(dtype=float)}
    if len(c) < 3:
        return out
    k = c.index.to_numpy(dtype=float)
    y = c["ic"].to_numpy(dtype=float)
    se = c["se"].to_numpy(dtype=float)
    se = np.where(np.isfinite(se) & (se > 0), se, np.nanmedian(se[np.isfinite(se) & (se > 0)]) if np.isfinite(se).any() else 1.0)
    out["ic1"] = float(y[0])
    out["significant"] = bool(abs(c["t"].iloc[0]) > 2) if np.isfinite(c["t"].iloc[0]) else False
    if not out["significant"]:
        return out
    k0 = k - k[0]

    def model(x, ic1, tau):
        return ic1 * np.exp(-x / tau)

    try:
        (ic1, tau), _ = curve_fit(model, k0, y, p0=(y[0], max(k0[-1] / 2.0, 1.0)), sigma=se, bounds=([-1.0, 1e-3], [1.0, 1e6]), maxfev=2000)
    except (RuntimeError, ValueError):
        return out
    fit = model(k0, ic1, tau)
    ss_res, ss_tot = float(((y - fit) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    out.update(ic1=float(ic1), tau=float(tau), half_life=float(tau * np.log(2.0)) if tau < 1e5 else float("inf"), r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
               periods_to_90pct=float(-tau * np.log(0.1)) if tau < 1e5 else float("inf"), fitted=pd.Series(fit, index=c.index))
    return out


def delay_cost(tau: float, delay: float) -> float:
    """The share of a signal's information lost by acting ``delay`` periods late under exponential decay with time constant ``tau``: ``1 - exp(-delay/tau)``."""
    return float(1.0 - np.exp(-float(delay) / float(tau))) if tau and np.isfinite(tau) and tau > 0 else 0.0
