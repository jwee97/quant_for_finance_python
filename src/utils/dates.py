"""Date, calendar and annualisation helpers.

The platform works on the *observed* trading calendar of the data (the union
of dates returned by the provider) rather than an exchange calendar library.
That keeps the dependency surface small and, more importantly, means the
backtest never holds a position on a date for which no price exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12


def ann_factor(periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(periods_per_year)


def ann_vol_scale(periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(np.sqrt(periods_per_year))


def to_timestamp(value) -> pd.Timestamp | None:
    """Parse a config date. ``None``/``"null"`` mean 'open ended'."""
    if value is None or (isinstance(value, str) and value.lower() in {"", "null", "none"}):
        return None
    return pd.Timestamp(value)


def slice_dates(
    frame: pd.DataFrame | pd.Series,
    start=None,
    end=None,
) -> pd.DataFrame | pd.Series:
    """Inclusive date slice that tolerates ``None`` on either end."""
    start_ts, end_ts = to_timestamp(start), to_timestamp(end)
    out = frame
    if start_ts is not None:
        out = out.loc[out.index >= start_ts]
    if end_ts is not None:
        out = out.loc[out.index <= end_ts]
    return out


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "monthly") -> pd.DatetimeIndex:
    """Last observed trading day of each period in ``index``.

    Using the last *observed* day (rather than a calendar month end) means the
    rebalance date is always a date on which a price exists.
    """
    index = pd.DatetimeIndex(index).sort_values()
    freq = frequency.lower()
    if freq in {"daily", "d"}:
        return index
    key = {
        "weekly": index.to_period("W"),
        "monthly": index.to_period("M"),
        "quarterly": index.to_period("Q"),
        "annual": index.to_period("Y"),
        "yearly": index.to_period("Y"),
    }.get(freq)
    if key is None:
        raise ValueError(f"unsupported rebalance frequency '{frequency}'")
    marks = pd.Series(index, index=key).groupby(level=0).max()
    return pd.DatetimeIndex(marks.to_numpy())


def year_fraction(index: pd.DatetimeIndex) -> float:
    """Calendar length of a series in years (used for CAGR)."""
    index = pd.DatetimeIndex(index)
    if len(index) < 2:
        return float("nan")
    return (index[-1] - index[0]).days / 365.25


@dataclass(frozen=True)
class DateWindow:
    """A named, inclusive date window (samples, stress regimes)."""

    name: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    label: str = ""

    @classmethod
    def from_config(cls, name: str, payload: dict) -> "DateWindow":
        return cls(
            name=name,
            start=to_timestamp(payload.get("start")),
            end=to_timestamp(payload.get("end")),
            label=payload.get("label", name.replace("_", " ").title()),
        )

    def apply(self, frame: pd.DataFrame | pd.Series):
        return slice_dates(frame, self.start, self.end)

    def contains(self, ts) -> bool:
        ts = pd.Timestamp(ts)
        if self.start is not None and ts < self.start:
            return False
        if self.end is not None and ts > self.end:
            return False
        return True

    def describe(self) -> str:
        s = "-inf" if self.start is None else self.start.date().isoformat()
        e = "now" if self.end is None else self.end.date().isoformat()
        return f"{self.label} [{s} .. {e}]"


def windows_from_config(payload: dict) -> dict[str, DateWindow]:
    return {name: DateWindow.from_config(name, spec or {}) for name, spec in (payload or {}).items()}
