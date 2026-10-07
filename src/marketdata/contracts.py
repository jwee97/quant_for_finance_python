"""Data contracts: the promises a feed makes about time, sessions and completeness, enforced rather than assumed.

A :class:`DataContract` states, for one source, the vendor's time zone, the exchange calendar and trading session, how long after an observation each kind of value is published
(``publication_lag``), how old a quote may be before it counts as stale, what to do about missing data and about revised values. :func:`apply_contract` converts a raw frame to UTC
under that contract and reports what it did, so every backtest says which promises it relied on.

* **Timezones**: vendor timestamps are interpreted in ``timezone`` and stored in UTC; ambiguous or non-existent local times (daylight-saving changes) are an error unless the contract
  says to shift them.
* **Sessions and calendars**: events outside the session or on a non-business day are flagged and optionally dropped.
* **Publication timestamps**: ``available_at = max(existing available_at, timestamp + publication_lag[event_type])``.
* **Stale quotes**: a quote identical to its predecessor for ``max_repeat`` consecutive updates, or a gap larger than ``max_gap`` inside a session, is flagged (:func:`stale_flags`).
* **Missing data**: ``error``, ``drop`` or ``ffill`` with a limit, applied to the bar series (:func:`fill_missing_bars`); the engine never invents a price older than ``max_age``.
* **Revisions**: ``first_release`` (what was known at the time) or ``latest_known`` (the corrected value once published).
* **Survivorship**: :class:`UniverseHistory` records when each instrument was listed and delisted, so dead instruments stay in a historical universe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..instruments.calendars import get_calendar
from .schema import normalise_events

REVISION_POLICIES = ("first_release", "latest_known")
MISSING_POLICIES = ("error", "drop", "ffill")


@dataclass
class DataContract:
    name: str = "default"
    timezone: str = "UTC"
    calendar: str = "24x7"
    session: tuple | None = None                       # ("09:30", "16:00") local times; None = all day
    publication_lag: dict = field(default_factory=dict)  # event_type -> pandas Timedelta string
    max_repeat: int = 5
    max_gap: str | None = None
    max_age: str | None = None                         # the engine does not use an observation older than this
    missing_policy: str = "ffill"
    ffill_limit: int = 3
    revision_policy: str = "latest_known"
    outside_session: str = "flag"                      # flag or drop
    dst_policy: str = "error"                          # error or shift_forward
    survivorship: str = "dead instruments are kept: pass a UniverseHistory with delisting dates"

    def __post_init__(self):
        if self.missing_policy not in MISSING_POLICIES or self.revision_policy not in REVISION_POLICIES or self.outside_session not in ("flag", "drop"):
            raise ValueError("invalid contract policy")
        get_calendar(self.calendar)

    def lag_for(self, event_type: str) -> pd.Timedelta:
        return pd.Timedelta(self.publication_lag.get(event_type, 0))


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    shifted_dst: int = 0
    outside_session: int = 0
    non_business_day: int = 0
    stale: int = 0
    availability_raised: int = 0
    dropped: int = 0
    notes: list = field(default_factory=list)

    def summary(self) -> pd.Series:
        return pd.Series({k: v for k, v in self.__dict__.items() if k != "notes"})


def _localize(ts: pd.Series, tz: str, dst_policy: str, report: QualityReport) -> pd.Series:
    if tz == "UTC":
        return ts
    shift = "shift_forward" if dst_policy == "shift_forward" else "raise"
    try:
        local = ts.dt.tz_localize(tz, ambiguous="raise", nonexistent=shift)
    except Exception as exc:
        raise ValueError(f"timestamps are ambiguous or non-existent in {tz} (daylight-saving change); set dst_policy='shift_forward' to accept: {exc}") from exc
    if shift == "shift_forward":
        report.shifted_dst = int((local.dt.tz_convert("UTC").dt.tz_localize(None) != ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)).sum())
    return local.dt.tz_convert("UTC").dt.tz_localize(None)


def apply_contract(frame: pd.DataFrame, contract: DataContract) -> tuple[pd.DataFrame, QualityReport]:
    """Convert a raw event frame to UTC and apply the contract; returns the canonical events and a :class:`QualityReport`."""
    report = QualityReport(rows_in=len(frame))
    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if getattr(df["timestamp"].dt, "tz", None) is None:
        df["timestamp"] = _localize(df["timestamp"], contract.timezone, contract.dst_policy, report)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    if "available_at" in df.columns and df["available_at"].notna().any():
        av = pd.to_datetime(df["available_at"])
        if getattr(av.dt, "tz", None) is None:
            av = _localize(av, contract.timezone, contract.dst_policy, report)
        else:
            av = av.dt.tz_convert("UTC").dt.tz_localize(None)
        df["available_at"] = av
    else:
        df["available_at"] = pd.NaT
    lag = df["event_type"].map(lambda e: contract.lag_for(e))
    required = df["timestamp"] + pd.to_timedelta(lag)
    raised = df["available_at"].isna() | (df["available_at"] < required)
    report.availability_raised = int(raised.sum())
    df["available_at"] = df["available_at"].where(~raised, required)
    cal = get_calendar(contract.calendar)
    local_days = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(contract.timezone).dt.tz_localize(None).dt.normalize() if contract.timezone != "UTC" else df["timestamp"].dt.normalize()
    bad_day = ~local_days.map(cal.is_business_day).astype(bool)
    report.non_business_day = int(bad_day.sum())
    outside = pd.Series(False, index=df.index)
    if contract.session is not None:
        local = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(contract.timezone).dt.tz_localize(None)
        open_t, close_t = (pd.Timedelta(hours=int(s.split(":")[0]), minutes=int(s.split(":")[1])) for s in contract.session)
        tod = local - local.dt.normalize()
        is_bar_or_settlement = df["event_type"].isin(["bar", "settlement", "curve", "fixing", "reference", "funding", "mark"])
        outside = ((tod < open_t) | (tod > close_t)) & ~is_bar_or_settlement
    report.outside_session = int(outside.sum())
    flagged = bad_day | outside
    if contract.outside_session == "drop":
        report.dropped = int(flagged.sum())
        df = df[~flagged]
    out = normalise_events(df, lag=pd.Timedelta(0))
    out["stale"] = stale_flags(out, contract.max_repeat, contract.max_gap)
    report.stale = int(out["stale"].sum())
    report.rows_out = len(out)
    return out.drop(columns="stale"), report


def stale_flags(events: pd.DataFrame, max_repeat: int = 5, max_gap=None) -> pd.Series:
    """True for quotes that repeat the same bid/ask for more than ``max_repeat`` consecutive updates, and for any event that follows a silent gap longer than ``max_gap``."""
    flags = pd.Series(False, index=events.index)
    q = events[events["event_type"] == "quote"]
    if len(q):
        key = q["bid"].astype(str) + "|" + q["ask"].astype(str)
        for _, g in key.groupby(q["instrument_id"]):
            run = (g != g.shift()).cumsum()
            count = g.groupby(run).cumcount() + 1
            flags.loc[g.index[count > max_repeat]] = True
    if max_gap is not None:
        gap = pd.Timedelta(max_gap)
        for _, g in events.groupby(["instrument_id", "event_type"]):
            late = g["timestamp"].diff() > gap
            flags.loc[g.index[late.to_numpy()]] = True
    return flags


def fill_missing_bars(bars: pd.DataFrame, calendar: str, policy: str = "ffill", limit: int = 3) -> pd.DataFrame:
    """Reindex a daily price frame to the calendar's business days and apply the missing-data policy: ``error`` raises if any day is missing, ``drop`` removes incomplete days,
    ``ffill`` carries the last value forward for at most ``limit`` days (longer gaps stay missing: a stale price is not a price)."""
    if policy not in MISSING_POLICIES:
        raise ValueError(f"policy must be one of {MISSING_POLICIES}")
    days = get_calendar(calendar).business_days(bars.index.min(), bars.index.max())
    out = bars.reindex(days)
    gaps = out.isna().all(axis=1)
    if policy == "error" and gaps.any():
        raise ValueError(f"{int(gaps.sum())} business days have no data, first {gaps.idxmax().date()}")
    if policy == "ffill":
        return out.ffill(limit=limit)
    if policy == "drop":
        return out.dropna(how="any")
    return out


@dataclass
class UniverseHistory:
    """When each instrument was a member of the investable universe: ``listed`` and ``delisted`` dates per instrument. Using it keeps dead instruments in historical backtests
    (survivorship bias is what you get when the universe is built from today's list)."""

    table: pd.DataFrame

    def __post_init__(self):
        t = self.table.copy()
        t["listed"] = pd.to_datetime(t["listed"])
        t["delisted"] = pd.to_datetime(t["delisted"]) if "delisted" in t.columns else pd.NaT
        if "instrument_id" in t.columns:
            t = t.set_index("instrument_id")
        self.table = t

    def members(self, ts) -> list[str]:
        ts = pd.Timestamp(ts)
        t = self.table
        return sorted(t.index[(t["listed"] <= ts) & (t["delisted"].isna() | (t["delisted"] > ts))])

    def survivors_only_bias(self, ts) -> float:
        """The share of instruments alive at ``ts`` that are missing from today's list (those delisted since): the size of the survivorship bias a naive universe would have."""
        alive = self.members(ts)
        today_alive = set(self.members(self.table["delisted"].max() + pd.Timedelta(days=1))) if self.table["delisted"].notna().any() else set(alive)
        return float(1.0 - len([i for i in alive if i in today_alive]) / max(len(alive), 1))
