"""Turning results into plain JSON: numpy and pandas values become Python numbers, strings and lists; NaN and infinity become ``None`` (JSON has no such numbers)."""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd


def clean(obj, digits: int | None = 8):
    """A JSON-safe copy of ``obj``. Floats are rounded to ``digits`` decimals (``None`` keeps them whole)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return round(v, digits) if digits is not None else v
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return None if pd.isna(obj) else pd.Timestamp(obj).strftime("%Y-%m-%d")
    if isinstance(obj, pd.Series):
        return {str(k): clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return {str(c): {str(i): clean(v, digits) for i, v in obj[c].items()} for c in obj.columns}
    if isinstance(obj, np.ndarray):
        return [clean(v, digits) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v, digits) for v in obj]
    if obj is pd.NaT:
        return None
    return str(obj)


def series_points(series: pd.Series, digits: int = 6) -> list:
    """``[[date, value], ...]`` with missing values left out: the shape the charts read."""
    s = series.dropna()
    return [[pd.Timestamp(i).strftime("%Y-%m-%d"), round(float(v), digits)] for i, v in s.items() if math.isfinite(float(v))]
