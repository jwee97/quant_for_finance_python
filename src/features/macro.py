"""Macro features (Generation 2, Priority 5).

Two rules that are easy to get wrong:

**Transform at the native frequency, then apply the publication lag.** Year-
over-year CPI inflation needs the CPI level twelve *reference months* ago, not
twelve trading days ago. So monthly series are transformed while still indexed
by reference month, and only then mapped onto trading days through their
availability dates. A derived series inherits the publication lag of the series
it was built from.

**Standardise with expanding statistics only.** A full-sample z-score gives
every past feature knowledge of the future mean and variance of the series. An
expanding z-score at date t uses only data up to t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.macro import MacroSeriesSpec, asof_series

# Derived monthly series and the published series each inherits its lag from.
DERIVED_SOURCE = {
    "CPI_YOY": "CPIAUCNS",
    "SAHM": "UNRATE",
    "PHILLY_CHG3": "GACDFSA066MSFRBPHI",
}

FEATURE_COLUMNS = [
    "curve_10y2y", "curve_10y3m", "curve_change_63",
    "fed_funds", "fed_funds_change_126",
    "breakeven", "cpi_yoy",
    "unemployment", "sahm_gap",
    "philly", "philly_change_3m",
    "vix", "move",
]


def monthly_grid(series: pd.Series) -> pd.Series:
    """Re-index a monthly series to an unbroken run of month-start stamps.

    A month that was never published (the October 2025 CPI and unemployment
    prints were cancelled by the government shutdown) becomes NaN instead of
    vanishing from the index. Position-based operations such as ``pct_change(12)``
    would otherwise compare against thirteen months back for the next year, a
    silent error that no look-ahead test can see because it only misstates
    the arithmetic.
    """
    stamps = series.index.to_period("M").to_timestamp()
    if stamps.has_duplicates:
        raise ValueError("monthly series has two observations in one month")
    stamped = pd.Series(series.to_numpy(dtype=float), index=stamps).sort_index()
    return stamped.reindex(pd.date_range(stamped.index.min(), stamped.index.max(), freq="MS"))


def monthly_transforms(raw: dict[str, pd.Series], short_months: int = 3,
                       long_months: int = 12) -> dict[str, pd.Series]:
    """Monthly transforms, still indexed by REFERENCE month.

    ``SAHM`` is the Sahm-rule gap: the 3-month average unemployment rate minus
    the minimum of that average over the preceding 12 months. It rises above
    ~0.5 pp at the onset of recessions.

    Transforms run on the date-aligned monthly grid and are defined only for
    months in which the underlying series was actually observed, so a missing
    publication is carried as a gap rather than back-filled or invented.
    """
    out: dict[str, pd.Series] = {}
    if "CPIAUCNS" in raw:
        grid = monthly_grid(raw["CPIAUCNS"])
        out["CPI_YOY"] = grid / grid.shift(12) - 1.0
    if "UNRATE" in raw:
        grid = monthly_grid(raw["UNRATE"])
        average = grid.rolling(short_months, min_periods=short_months - 1).mean().where(grid.notna())
        floor = average.shift(1).rolling(long_months, min_periods=long_months - 1).min()
        out["SAHM"] = (average - floor).where(grid.notna())
    if "GACDFSA066MSFRBPHI" in raw:
        grid = monthly_grid(raw["GACDFSA066MSFRBPHI"])
        out["PHILLY_CHG3"] = grid.diff(3)
    return out


def build_macro_levels(raw: dict[str, pd.Series], specs: list[MacroSeriesSpec],
                       calendar: pd.DatetimeIndex, short_months: int = 3,
                       long_months: int = 12) -> pd.DataFrame:
    """Everything known on each date, before any standardisation."""
    by_id = {s.id: s for s in specs}
    series: dict[str, pd.Series] = {}
    for key, values in raw.items():
        series[key] = asof_series(values, by_id[key], calendar)
    for key, values in monthly_transforms(raw, short_months, long_months).items():
        series[key] = asof_series(values.dropna(), by_id[DERIVED_SOURCE[key]], calendar)
    return pd.DataFrame(series, index=pd.DatetimeIndex(calendar))


def build_macro_features(levels: pd.DataFrame, change_windows=(63, 126)) -> pd.DataFrame:
    """Named features, unstandardised. Daily changes are over TRADING days of
    the already-published panel, so they are causal."""
    f = pd.DataFrame(index=levels.index)
    f["curve_10y2y"] = levels.get("T10Y2Y")
    f["curve_10y3m"] = levels.get("T10Y3M")
    f["curve_change_63"] = levels["T10Y2Y"].diff(change_windows[0]) if "T10Y2Y" in levels else np.nan
    f["fed_funds"] = levels.get("DFF")
    f["fed_funds_change_126"] = levels["DFF"].diff(change_windows[1]) if "DFF" in levels else np.nan
    f["breakeven"] = levels.get("T10YIE")
    f["cpi_yoy"] = levels.get("CPI_YOY")
    f["unemployment"] = levels.get("UNRATE")
    f["sahm_gap"] = levels.get("SAHM")
    f["philly"] = levels.get("GACDFSA066MSFRBPHI")
    f["philly_change_3m"] = levels.get("PHILLY_CHG3")
    f["vix"] = np.log(levels["VIX"]) if "VIX" in levels else np.nan        # log level
    f["move"] = np.log(levels["MOVE"]) if "MOVE" in levels else np.nan
    return f[FEATURE_COLUMNS]


def expanding_zscore(frame: pd.DataFrame, min_periods: int = 756, winsorize: float = 4.0) -> pd.DataFrame:
    """Causal standardisation: the mean and deviation at t use data up to t only."""
    mean = frame.expanding(min_periods=min_periods).mean()
    std = frame.expanding(min_periods=min_periods).std(ddof=1).replace(0.0, np.nan)
    z = (frame - mean) / std
    return z.clip(lower=-winsorize, upper=winsorize) if winsorize else z


def expanding_percentile(series: pd.Series, min_periods: int = 252) -> pd.Series:
    """Rank of today's value within its own history to date, in [0, 1].

    Computed as the share of PAST-AND-CURRENT observations at or below today's
    value, using only data up to t.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    sorted_history: list[float] = []
    import bisect

    for i, v in enumerate(values):
        if np.isfinite(v):
            bisect.insort(sorted_history, v)
            if len(sorted_history) >= min_periods:
                out[i] = bisect.bisect_right(sorted_history, v) / len(sorted_history)
    return pd.Series(out, index=series.index)


def macro_feature_panel(raw: dict[str, pd.Series], specs: list[MacroSeriesSpec],
                        target_index: pd.DatetimeIndex, config=None) -> dict[str, pd.DataFrame]:
    """Levels, features and standardised features on the trading calendar.

    The panel is built on a business-day calendar that starts years before the
    first ETF observation, so the expanding standardisation has history to work
    with from day one of the ETF sample instead of burning in on it.
    """
    node = (config.get("macro.features", {}) if config is not None else {}) or {}
    std = node.get("standardise", {}) or {}
    sahm = node.get("sahm_rule", {}) or {}
    windows = tuple(node.get("change_windows", [63, 126]))

    start = min(s.index.min() for s in raw.values())
    start = max(start, pd.Timestamp("1995-01-01"))
    calendar = pd.bdate_range(start, pd.DatetimeIndex(target_index).max())

    levels = build_macro_levels(raw, specs, calendar, int(sahm.get("short_months", 3)),
                                int(sahm.get("long_months", 12)))
    features = build_macro_features(levels, windows)
    z = expanding_zscore(features, int(std.get("min_periods", 756)), float(std.get("winsorize", 4.0)))
    target = pd.DatetimeIndex(target_index)
    return {"levels": levels.reindex(target), "features": features.reindex(target),
            "z": z.reindex(target)}
