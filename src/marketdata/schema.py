"""Normalisation and validation of market-data events.

``normalise_events`` turns any frame with (a subset of) the standard columns into the canonical one: types, ordering, and an ``available_at`` where one is missing. A missing
``available_at`` is the dangerous case, so it is filled only by an explicit, recorded rule (``lag``: a fixed delay or a per-event-type mapping) and the number of rows filled that way is
reported, never silently assumed to be zero.

``validate_events`` returns a table of problems rather than raising on the first: available-before-observed rows (look-ahead), crossed or non-positive prices, duplicated keys, events
missing the fields their type requires, unknown event types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .events import COLUMNS, EVENT_TYPES, NUMERIC_COLUMNS, OBJECT_COLUMNS, PRICE_COLUMNS, REQUIRED

ERROR_KINDS = ("missing_column", "bad_event_type", "available_before_observed", "crossed_quote", "non_positive_price", "duplicate_key", "missing_field", "non_finite")


def _to_ns(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def normalise_events(frame: pd.DataFrame, lag=None, source: str | None = None) -> pd.DataFrame:
    """The canonical event frame, sorted by ``(available_at, timestamp, instrument_id, event_type, revision)``.

    ``lag`` fills ``available_at`` where it is missing (or the column is absent): a ``pd.Timedelta``/string for one delay, or ``{event_type: delay}``. Rows filled this way are counted in
    ``frame.attrs['assumed_available_at']`` so that the assumption stays visible."""
    df = frame.copy()
    for needed in ("timestamp", "instrument_id", "event_type"):
        if needed not in df.columns:
            raise KeyError(f"events need a '{needed}' column")
    df["timestamp"] = _to_ns(df["timestamp"])
    if "available_at" not in df.columns:
        df["available_at"] = pd.NaT
    df["available_at"] = _to_ns(df["available_at"])
    if "vendor_timestamp" not in df.columns:
        df["vendor_timestamp"] = pd.NaT
    df["vendor_timestamp"] = _to_ns(df["vendor_timestamp"])
    missing = df["available_at"].isna()
    if missing.any():
        if lag is None:
            raise ValueError(f"{int(missing.sum())} events have no available_at and no lag rule was given: availability must be stated, not guessed")
        if isinstance(lag, dict):
            delays = pd.to_timedelta(df.loc[missing, "event_type"].map(lambda e: pd.Timedelta(lag.get(e, 0))))
        else:
            delays = pd.Timedelta(lag)
        df.loc[missing, "available_at"] = df.loc[missing, "timestamp"] + delays
    df.attrs["assumed_available_at"] = int(missing.sum())
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64") if col in df.columns else np.nan
    for col in OBJECT_COLUMNS:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)
    if source is not None:
        df["source"] = df["source"].where(df["source"].notna(), source)
    df["revision"] = pd.to_numeric(df["revision"], errors="coerce").fillna(0).astype("int64") if "revision" in df.columns else 0
    df["instrument_id"] = df["instrument_id"].astype(str)
    df["event_type"] = df["event_type"].astype(str)
    df = df[COLUMNS].sort_values(["available_at", "timestamp", "instrument_id", "event_type", "revision"], kind="stable").reset_index(drop=True)
    df.attrs["assumed_available_at"] = int(missing.sum())
    return df


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    """A table of ``kind``, ``count`` and an ``example`` row index for every problem found (empty means clean)."""
    problems: list[dict] = []

    def add(kind, mask, note=""):
        n = int(mask.sum())
        if n:
            problems.append({"kind": kind, "count": n, "first_row": int(np.flatnonzero(np.asarray(mask))[0]), "note": note})

    absent = [c for c in COLUMNS if c not in df.columns]
    if absent:
        return pd.DataFrame([{"kind": "missing_column", "count": len(absent), "first_row": -1, "note": ", ".join(absent)}])
    add("bad_event_type", ~df["event_type"].isin(EVENT_TYPES), "unknown event_type")
    add("available_before_observed", df["available_at"] < df["timestamp"], "available_at precedes timestamp: look-ahead")
    add("crossed_quote", (df["bid"] > df["ask"]) & df["bid"].notna() & df["ask"].notna(), "bid above ask")
    others = [c for c in PRICE_COLUMNS if c not in ("bid",)]
    non_positive = ((df[others] <= 0).any(axis=1) | (df["bid"] < 0)) & ~df["event_type"].isin(["curve", "fixing", "reference"])
    add("non_positive_price", non_positive, "price <= 0 (a bid of exactly zero is allowed; use 'value' for series that can be negative)")
    add("non_finite", np.isinf(df[NUMERIC_COLUMNS]).any(axis=1), "infinite value")
    dup = df.duplicated(["instrument_id", "event_type", "timestamp", "revision"], keep=False)
    add("duplicate_key", dup, "same instrument, type, timestamp and revision more than once")
    for etype, alternatives in REQUIRED.items():
        sub = df["event_type"] == etype
        if not sub.any():
            continue
        ok = pd.Series(False, index=df.index)
        for group in alternatives:
            group_ok = pd.Series(True, index=df.index)
            for col in group:
                group_ok &= df[col].notna() if col not in OBJECT_COLUMNS else df[col].map(lambda v: v is not None and v == v)
            ok |= group_ok
        add("missing_field", sub & ~ok, f"{etype} events without {alternatives}")
    return pd.DataFrame(problems, columns=["kind", "count", "first_row", "note"])


def assert_valid(df: pd.DataFrame) -> pd.DataFrame:
    problems = validate_events(df)
    if len(problems):
        raise ValueError("invalid market data:\n" + problems.to_string(index=False))
    return df
