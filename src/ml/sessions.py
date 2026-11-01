"""Session-aware timestamps: when each market's daily bar is final, what a decision at one market's close may know, and when a release is public (P0-16).

A daily bar carries a DATE label, and two bars with the same label are not simultaneous. Tokyo's bar of a day is final at 06:30 UTC, London's at 16:30, New York's at 21:00 UTC (20:00 in summer), and a crypto
"daily" bar covers the UTC day and is final at the next midnight. A decision taken at the New York close may use Tokyo's and London's bars of that date but not a crypto bar with the same label, which closes
hours later; a decision at the Tokyo close may use none of the others' bars of that date. Joining panels on the date label gives every market the same information set and so lets the later-closing one into the
earlier one's decision.

``Session`` says when a bar becomes final, on the real clock (zone rules, so daylight-saving changes move the UTC instant and nothing else). ``asof_panel`` gives, for each decision date, the latest bar of each
asset that was final at the decision instant, with its age. ``information_lag`` says how many bars a market is ahead of a decision, ``entry_check`` refuses a label convention whose entry price would be in the past
of the decision, and ``release_instant`` / ``release_known`` do the same for a scheduled release (US CPI at 08:30 New York is public to a New York close the same day and to a Tokyo close only the next day).

Holidays need no calendar here: a market that was shut has no bar, so the latest final bar is an older one and its age says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd


def ns(index) -> np.ndarray:
    """UTC nanoseconds since the epoch of each instant. ``DatetimeIndex.asi8`` counts in the index's own unit, which pandas 3 chooses by what it parsed (seconds, microseconds ...), so two indexes can disagree about
    what an integer means; everything here is compared in nanoseconds."""
    return pd.DatetimeIndex(index).as_unit("ns").asi8


class SessionError(ValueError):
    """A session setting, or a combination of sessions, that cannot mean what it says."""


@dataclass(frozen=True)
class Session:
    name: str
    tz: str                      # IANA zone of the exchange
    close: time                  # local wall-clock time at which the daily bar is final (after any closing auction)
    day_offset: int = 0          # a bar labelled d is final on local date d + day_offset (a crypto bar is final at the next midnight)
    note: str = ""

    def close_utc(self, dates) -> pd.DatetimeIndex:
        """The UTC instant at which the bar labelled each date is final, from the zone's rules for that date."""
        idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        local = idx + pd.Timedelta(days=self.day_offset) + pd.Timedelta(hours=self.close.hour, minutes=self.close.minute)
        try:
            return local.tz_localize(self.tz).tz_convert("UTC")
        except Exception as exc:                                                                    # noqa: BLE001 - an ambiguous or missing local time is a setting error, with the zone named
            raise SessionError(f"{self.name}: the close time {self.close} does not exist exactly once on every date in {self.tz} ({exc})") from None


SESSIONS: dict[str, Session] = {s.name: s for s in (
    Session("us_equity", "America/New_York", time(16, 0), note="NYSE and Nasdaq close"),
    Session("uk_equity", "Europe/London", time(16, 35), note="London Stock Exchange, after the closing auction"),
    Session("eu_equity", "Europe/Paris", time(17, 35), note="Euronext and Xetra, after the closing auction"),
    Session("tokyo_equity", "Asia/Tokyo", time(15, 30), note="Tokyo Stock Exchange (15:00 before 5 November 2024: the later time is the conservative one)"),
    Session("hk_equity", "Asia/Hong_Kong", time(16, 10), note="Hong Kong Exchange, after the closing auction"),
    Session("au_equity", "Australia/Sydney", time(16, 10), note="ASX, after the closing single-price auction"),
    Session("fx", "America/New_York", time(17, 0), note="the foreign-exchange day ends at 17:00 New York"),
    Session("crypto", "UTC", time(0, 0), 1, note="the bar of a UTC day is final at the next midnight UTC"),
)}

SUFFIXES = {".T": "tokyo_equity", ".L": "uk_equity", ".DE": "eu_equity", ".PA": "eu_equity", ".AS": "eu_equity", ".MI": "eu_equity", ".MC": "eu_equity", ".HK": "hk_equity", ".AX": "au_equity", "=X": "fx"}
INDEXES = {"^N225": "tokyo_equity", "^FTSE": "uk_equity", "^GDAXI": "eu_equity", "^FCHI": "eu_equity", "^STOXX50E": "eu_equity", "^HSI": "hk_equity", "^AXJO": "au_equity",
           "^GSPC": "us_equity", "^IXIC": "us_equity", "^DJI": "us_equity", "^RUT": "us_equity"}
CRYPTO_BASES = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BNB", "DOT", "AVAX", "LINK", "MATIC"}


def get(session: str | Session) -> Session:
    if isinstance(session, Session):
        return session
    if session not in SESSIONS:
        raise SessionError(f"unknown session '{session}'; choose from {sorted(SESSIONS)}")
    return SESSIONS[session]


def session_for(ticker: str, asset_class: str | None = None) -> tuple[str, str]:
    """``(session name, basis)`` for a ticker: from its exchange suffix, its index symbol, its asset class or, failing all of those, the platform's default (US equity hours), with the basis saying which."""
    t = str(ticker).upper()
    if t in INDEXES:
        return INDEXES[t], "index symbol"
    for suffix, name in SUFFIXES.items():
        if t.endswith(suffix):
            return name, f"suffix {suffix}"
    if asset_class == "crypto" or (t.endswith("-USD") and t.split("-")[0] in CRYPTO_BASES) or t.split("_")[0] in CRYPTO_BASES:
        return "crypto", "crypto asset"
    return "us_equity", "default"


# ------------------------------------------------------------------------------------------------------------------ what was final when
def decision_instants(decision: str | Session, dates, latency_minutes: int = 0) -> pd.DatetimeIndex:
    """The instants (UTC) at which the decisions of ``dates`` are taken: the close of the decision session, plus ``latency_minutes`` for the time the model takes."""
    return get(decision).close_utc(dates) + pd.Timedelta(minutes=int(latency_minutes))


def asof_panel(frame: pd.DataFrame, sessions: dict | str, decision: str | Session, dates=None, latency_minutes: int = 0, max_age_hours: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(values, age_hours)`` on the decision dates: for each asset, the latest bar that was final at the decision instant and how many hours before it that was.

    ``frame`` is date x asset, labelled by each bar's own date; ``sessions`` maps asset to session name (one name for all; an asset missing from the mapping trades in the decision session). A date on which an asset
    has no bar (a closure) is not a bar, so an older one is the latest; with ``max_age_hours`` a bar older than that is treated as no value at all (NaN), not carried.
    """
    decision_dates = pd.DatetimeIndex(frame.index if dates is None else dates)
    instants = ns(decision_instants(decision, decision_dates, latency_minutes))
    values = pd.DataFrame(np.nan, index=decision_dates, columns=frame.columns, dtype=float)
    ages = pd.DataFrame(np.nan, index=decision_dates, columns=frame.columns, dtype=float)
    dec = get(decision)
    for col in frame.columns:
        sess = get(sessions) if isinstance(sessions, (str, Session)) else get(sessions.get(col, dec))
        column = frame[col]
        have = column.notna().to_numpy()
        if not have.any():
            continue
        bars = frame.index[have]
        closed = ns(sess.close_utc(bars))
        k = np.searchsorted(closed, instants, side="right") - 1
        ok = k >= 0
        v = np.full(len(instants), np.nan)
        a = np.full(len(instants), np.nan)
        v[ok] = column.to_numpy(dtype=float)[have][k[ok]]
        a[ok] = (instants[ok] - closed[k[ok]]) / 3.6e12
        if max_age_hours is not None:
            stale = a > float(max_age_hours)
            v[stale] = np.nan
        values[col], ages[col] = v, a
    return values, ages


def information_lag(session: str | Session, decision: str | Session, dates=None) -> int:
    """How many bars of ``session`` a decision at the close of ``decision`` has not yet seen that carry its own date label: 0 if the bar of a date is final at the decision instant on every date probed, 1 if it never is.

    Raises if it is sometimes one and sometimes the other (two markets whose order of closing changes with daylight saving would need a finer rule than a bar of lag)."""
    probe = pd.bdate_range("2023-01-02", "2024-12-31") if dates is None else pd.DatetimeIndex(dates)
    final = ns(get(session).close_utc(probe)) <= ns(decision_instants(decision, probe))
    if final.all():
        return 0
    if not final.any():
        return 1
    raise SessionError(f"the {get(session).name} bar of a date is final before the {get(decision).name} decision on some dates and after it on others (daylight saving); give an explicit latency or a different decision session")


def information_lags(sessions: dict, decision: str | Session) -> dict:
    """``information_lag`` of every asset's session against the decision session."""
    cache: dict = {}
    out = {}
    for asset, name in sessions.items():
        key = get(name).name
        if key not in cache:
            cache[key] = information_lag(name, decision)
        out[asset] = cache[key]
    return out


def entry_check(sessions: dict, decision: str | Session, lag: int, dates=None) -> None:
    """Refuse a label convention whose entry price is not in the future of the decision. A label measured from the close of day ``t + lag`` is executable only if that bar closes at or after the decision instant of
    ``t``: with ``lag = 0`` an asset whose market closed before the decision cannot be bought at the price the label assumes."""
    probe = pd.bdate_range("2023-01-02", "2024-12-31") if dates is None else pd.DatetimeIndex(dates)
    decided = ns(decision_instants(decision, probe))
    bad = []
    for asset, name in sessions.items():
        entry = ns(get(name).close_utc(probe + pd.offsets.BDay(int(lag))))
        if (entry < decided).any():
            bad.append(f"{asset} ({get(name).name})")
    if bad:
        raise SessionError(f"with lag {lag} the entry price of {', '.join(bad[:6])}{'...' if len(bad) > 6 else ''} is the close of a bar that was final before the {get(decision).name} decision, so the label cannot be earned; use lag 1 or more")


# ------------------------------------------------------------------------------------------------------------------ scheduled releases
def release_instant(dates, local_time: time, tz: str) -> pd.DatetimeIndex:
    """The UTC instant of a release at ``local_time`` in ``tz`` on each date (US CPI: 08:30 America/New_York)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize() + pd.Timedelta(hours=local_time.hour, minutes=local_time.minute)
    return idx.tz_localize(tz).tz_convert("UTC")


def release_known(release: pd.DatetimeIndex, decision: str | Session, decision_dates, latency_minutes: int = 0) -> np.ndarray:
    """For each decision date, whether a release at the (UTC) ``release`` instant, one per decision date, was public at the decision instant."""
    return np.asarray(ns(release) <= ns(decision_instants(decision, decision_dates, latency_minutes)))


def known_on(release_instants: pd.DatetimeIndex, values: pd.Series, decision: str | Session, decision_dates, latency_minutes: int = 0) -> pd.Series:
    """The latest ``values`` whose release (UTC instant, aligned with ``values``) was public at each decision instant: the as-of join of a published series on availability rather than on its reference date.
    A value released after the decision instant is not there yet; a value released late is first seen on the first decision after its release."""
    released = ns(release_instants)
    order = np.argsort(released, kind="stable")
    rel, val = released[order], values.to_numpy(dtype=float)[order]
    dec = ns(decision_instants(decision, pd.DatetimeIndex(decision_dates), latency_minutes))
    k = np.searchsorted(rel, dec, side="right") - 1
    out = np.where(k >= 0, val[np.maximum(k, 0)], np.nan)
    return pd.Series(out, index=pd.DatetimeIndex(decision_dates))


# ------------------------------------------------------------------------------------------------------------------ the dataset setting
SETTINGS = {"decision", "assets", "policy", "latency_minutes", "max_age_hours"}
POLICIES = ("asof", "date_labels")


def validate_settings(settings) -> dict:
    """The ``dataset.sessions`` setting, checked: ``decision`` (the session whose close the strategy trades at), ``assets`` (ticker -> session overrides), ``policy`` (``asof``: a bar enters a decision only if it was final then;
    ``date_labels``: the legacy join on the date label, kept to reproduce earlier results), ``latency_minutes`` and ``max_age_hours`` (a carried bar older than this is missing)."""
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise SessionError("dataset.sessions must be a mapping")
    unknown = set(settings) - SETTINGS
    if unknown:
        raise SessionError(f"unknown dataset.sessions setting(s) {sorted(unknown)}; allowed: {sorted(SETTINGS)}")
    if settings.get("decision") is not None:
        get(settings["decision"])
    for ticker, name in (settings.get("assets") or {}).items():
        get(name)
    if settings.get("policy", "asof") not in POLICIES:
        raise SessionError(f"dataset.sessions.policy must be one of {POLICIES}")
    if int(settings.get("latency_minutes", 0)) < 0 or float(settings.get("max_age_hours", 120)) <= 0:
        raise SessionError("dataset.sessions.latency_minutes must be >= 0 and max_age_hours > 0")
    return dict(settings)
