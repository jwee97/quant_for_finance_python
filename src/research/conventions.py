"""How a saved return stream is measured: its calendar, its frequency and the annualisation that follows from them, stated rather than assumed.

A study's equity curve is a set of dated levels. What a number like "annual volatility" means depends on how often those levels were observed and on which calendar, so each series carries (or is given, by inference
from its dates, flagged as inferred):

* ``frequency``     daily, weekly, monthly, quarterly, or irregular;
* ``calendar``      ``business_days`` (no weekend observations: equities, futures), ``continuous`` (observations on weekends: crypto), or ``custom``;
* ``periods_per_year``  the convention for per-period statistics: 252 for business days, 365 for a continuous daily calendar, 52, 12, 4, or ``None`` for irregular series;
* ``return_type``, ``currency``, ``timezone`` and the ``risk_free`` convention, which are recorded and compared even where they do not enter the arithmetic.

Two ways to annualise, chosen by the series and never mixed:

``periods``  per-period statistics scaled by the number of periods in a year: volatility ``sd x sqrt(ppy)``, Sharpe ``sqrt(ppy) x mean(excess) / sd(excess)``, Sortino with the downside deviation. The scaling assumes the periods
             are alike and the returns are not strongly autocorrelated; with autocorrelation ``sqrt(ppy)`` mis-states the annual volatility, and the page says so.
``elapsed``  for irregular spacing: variance and mean rates per unit of elapsed time, ``sum(e_i^2) / sum(dt_i)`` and ``sum(excess_i) / sum(dt_i)``, so a long gap contributes in proportion to its length. It assumes returns scale with elapsed time.

CAGR always uses the actual elapsed time between the first and last observation (365.25-day years). The risk-free rate is an annual effective rate accrued by elapsed calendar time between observations, so a Friday-to-Monday
return accrues three days of it. No statistic multiplies by a constant that the series did not declare.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

YEAR_DAYS = 365.25
STANDARD = {"business_days": 252.0, "continuous": 365.0}
FREQ_PPY = {"daily": None, "weekly": 52.0, "monthly": 12.0, "quarterly": 4.0}
RF_CONVENTION = "annual effective rate, accrued by elapsed calendar time between observations"


class ConventionError(ValueError):
    """The series cannot be measured under a stated convention (and why)."""


def infer_meta(dates) -> dict:
    """The convention a series' own dates imply, marked ``source: inferred``. Weekend observations make a daily series ``continuous``; a median spacing near 7, 30 or 91 days makes it weekly, monthly or quarterly; anything
    else, or a very uneven spacing, is ``irregular`` and is annualised by elapsed time."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values().unique()
    meta = {"frequency": "irregular", "calendar": "custom", "periods_per_year": None, "return_type": "simple", "currency": "unspecified", "timezone": "date-level (no time of day)", "risk_free": RF_CONVENTION, "source": "inferred"}
    if len(idx) < 4:
        return meta
    gaps = np.diff(idx.values).astype("timedelta64[h]").astype(float) / 24.0
    med = float(np.median(gaps))
    weekend = float((idx.dayofweek >= 5).mean())
    if med <= 1.5:
        cont = weekend > 0.15
        meta.update(frequency="daily", calendar="continuous" if cont else "business_days", periods_per_year=STANDARD["continuous" if cont else "business_days"])
        if (gaps > 10).mean() > 0.02:                                                                  # long holes: spacing is too uneven to call it regular
            meta.update(frequency="irregular", calendar="custom", periods_per_year=None)
    elif 5.5 <= med <= 8.5:
        meta.update(frequency="weekly", calendar="custom", periods_per_year=52.0)
    elif 26 <= med <= 33:
        meta.update(frequency="monthly", calendar="custom", periods_per_year=12.0)
    elif 85 <= med <= 95:
        meta.update(frequency="quarterly", calendar="custom", periods_per_year=4.0)
    return meta


def meta_of(series: dict | None) -> dict:
    """A saved series' convention: what it declared, with anything missing filled by inference (and ``source`` saying which)."""
    series = series or {}
    inferred = infer_meta(series.get("dates") or [])
    declared = dict(series.get("meta") or {})
    if not declared:
        return inferred
    out = {**inferred, **{k: v for k, v in declared.items() if v not in (None, "")}}
    missing = [k for k in ("frequency", "calendar", "periods_per_year") if k not in declared]
    out["source"] = "declared" if not missing else "declared (partly inferred: " + ", ".join(missing) + ")"
    return out


def grid_annualisation(grid: pd.DatetimeIndex, metas: list[dict]) -> dict:
    """The annualisation for statistics measured on the common grid of ``grid``. If every series declared the same per-period convention and the grid is their own, that convention is used; otherwise the
    grid's own spacing decides (a business-day grid is 252, a continuous one 365, a regular weekly or monthly one 52 or 12, anything else elapsed-time)."""
    g = infer_meta(grid)
    declared = {(m.get("calendar"), m.get("periods_per_year")) for m in metas if str(m.get("source", "")).startswith("declared") and m.get("periods_per_year")}
    if len(declared) == 1 and all(str(m.get("source", "")).startswith("declared") for m in metas) and g["frequency"] == metas[0].get("frequency"):
        cal, ppy = next(iter(declared))
        return {"basis": "periods", "periods_per_year": float(ppy), "calendar": cal, "frequency": g["frequency"], "source": "declared by every series"}
    if g["periods_per_year"] is None:
        return {"basis": "elapsed", "periods_per_year": None, "calendar": g["calendar"], "frequency": g["frequency"], "source": "inferred from the shared dates: irregular spacing is annualised by elapsed time"}
    return {"basis": "periods", "periods_per_year": float(g["periods_per_year"]), "calendar": g["calendar"], "frequency": g["frequency"], "source": "inferred from the shared dates"}


def step_years(index: pd.DatetimeIndex) -> np.ndarray:
    """Elapsed calendar time, in years, that each return after the first level covers."""
    return np.diff(index.values).astype("timedelta64[s]").astype(float) / (86400.0 * YEAR_DAYS)


def rf_accrual(rf: float, dt: np.ndarray) -> np.ndarray:
    return (1.0 + rf) ** dt - 1.0


def measure_levels(levels: pd.Series, ann: dict, rf: float = 0.0) -> dict:
    """The headline numbers of a series of dated levels under ``ann`` (see :func:`grid_annualisation`). ``levels`` has no gaps by construction: every value is an observed level, and a return is the change between two
    consecutive observed levels (so a missing day is a longer return, never a zero)."""
    lv = levels.dropna().astype(float)
    if len(lv) < 3 or (lv <= 0).any():
        return {"n_days": max(len(lv) - 1, 0)}
    r = lv.pct_change().dropna()
    dt = step_years(lv.index)
    n = len(r)
    elapsed = float(dt.sum())
    excess = r.to_numpy() - rf_accrual(rf, dt)
    cagr = float((lv.iloc[-1] / lv.iloc[0]) ** (1.0 / elapsed) - 1.0) if elapsed > 0 else None
    eq = lv / lv.iloc[0]
    dd = eq / eq.cummax() - 1.0
    out = {"n_days": int(n), "elapsed_years": elapsed, "cagr": cagr, "total_return": float(eq.iloc[-1] - 1.0), "max_drawdown": float(dd.min()), "calmar": None if not cagr or dd.min() == 0 else float(cagr / -dd.min())}
    if ann["basis"] == "periods":
        ppy = ann["periods_per_year"]
        sd_r, sd_e = float(np.std(r.to_numpy(), ddof=1)), float(np.std(excess, ddof=1))
        down = float(math.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
        out.update(ann_vol=sd_r * math.sqrt(ppy), sharpe=None if sd_e <= 0 else float(math.sqrt(ppy) * excess.mean() / sd_e), sortino=None if down <= 0 else float(math.sqrt(ppy) * excess.mean() / down))
    else:
        mu = float(excess.sum() / elapsed)
        e = excess - mu * dt
        var_rate = float((e ** 2).sum() / elapsed * n / max(n - 1, 1))
        down = float(math.sqrt((np.minimum(excess, 0.0) ** 2).sum() / elapsed))
        out.update(ann_vol=math.sqrt(var_rate), sharpe=None if var_rate <= 0 else float(mu / math.sqrt(var_rate)), sortino=None if down <= 0 else float(mu / down))
    out["annualisation"] = {k: ann[k] for k in ("basis", "periods_per_year", "calendar", "frequency", "source")}
    return out


def measure_returns(r: pd.Series, rf: float = 0.0, ann: dict | None = None) -> dict:
    """:func:`measure_levels` for a series of returns: the levels are rebuilt from a starting level of 1 placed one typical step before the first return, so ``n`` returns span ``n`` steps."""
    r = r.dropna().astype(float)
    if len(r) < 2:
        return {"n_days": len(r)}
    idx = pd.DatetimeIndex(r.index)
    step = pd.Series(idx).diff().median()
    start = idx[0] - (step if pd.notna(step) else pd.Timedelta(days=1))
    lv = pd.Series(np.concatenate([[1.0], (1.0 + r.to_numpy()).cumprod()]), index=pd.DatetimeIndex([start, *idx]))
    return measure_levels(lv, ann or grid_annualisation(lv.index, [infer_meta(lv.index)]), rf)


def rolling_measures(levels: pd.Series, ann: dict, window: int, rf: float = 0.0) -> dict:
    """Rolling volatility and Sharpe over ``window`` observations, aligned to the levels' dates after the first (None before a full window)."""
    lv = levels.dropna().astype(float)
    r = lv.pct_change().dropna()
    dt = pd.Series(step_years(lv.index), index=r.index)
    ex = r - rf_accrual(rf, dt.to_numpy())
    if ann["basis"] == "periods":
        ppy = ann["periods_per_year"]
        sd = r.rolling(window, min_periods=window).std(ddof=1)
        sh = math.sqrt(ppy) * ex.rolling(window, min_periods=window).mean() / ex.rolling(window, min_periods=window).std(ddof=1)
        return {"vol": sd * math.sqrt(ppy), "sharpe": sh}
    T = dt.rolling(window, min_periods=window).sum()
    mu = ex.rolling(window, min_periods=window).sum() / T
    e = ex - (mu * dt)
    var = (e ** 2).rolling(window, min_periods=window).sum() / T
    return {"vol": np.sqrt(var), "sharpe": mu / np.sqrt(var)}


def relative_to(r: pd.Series, b: pd.Series, ann: dict) -> dict:
    """The strategy against a reference on the same dates, annualised with the same convention as everything else."""
    both = pd.concat([r, b], axis=1, join="inner").dropna()
    if len(both) < 30:
        return {}
    x, y = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    act = x - y
    var_b = float(np.var(y, ddof=1))
    beta = float(np.cov(x, y, ddof=1)[0, 1] / var_b) if var_b > 0 else None
    if ann["basis"] == "periods":
        k, ks = ann["periods_per_year"], math.sqrt(ann["periods_per_year"])
    else:
        T = step_years(pd.DatetimeIndex([both.index[0] - (both.index[1] - both.index[0])] + list(both.index))).sum()
        k, ks = len(both) / T, math.sqrt(len(both) / T)                         # observations per elapsed year of this pair
    te = float(np.std(act, ddof=1) * ks)
    up, down = y > 0, y < 0
    return {"n_days": int(len(both)), "correlation": float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None, "beta": beta,
            "alpha_annual": None if beta is None else float((x.mean() - beta * y.mean()) * k), "tracking_error": te, "information_ratio": None if te == 0 else float(act.mean() * k / te),
            "up_capture": float(x[up].mean() / y[up].mean()) if up.any() and y[up].mean() != 0 else None, "down_capture": float(x[down].mean() / y[down].mean()) if down.any() and y[down].mean() < 0 else None}
