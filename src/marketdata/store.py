"""The point-in-time store: market data as it was KNOWN, not as it later turned out.

Every query takes a decision time ``ts`` and sees only events with ``available_at <= ts``. Among those it returns the latest OBSERVATION (largest ``timestamp``); if that observation
was revised, the ``revision_policy`` decides which version a trader could have seen: ``latest_known`` (the most recently published revision as of ``ts``, the default) or
``first_release`` (the original number, ignoring later corrections: the honest choice for anything that is heavily revised, such as macro data).

The store keeps an audit of what it returned: ``max_available_returned`` must never exceed the query time, and :meth:`assert_no_lookahead` checks that, so a leak (a bug in an adapter
or in the engine's clock) fails loudly instead of flattering a backtest.

Prices of any kind are available through :meth:`price`, which looks across quote, trade, bar, mark and settlement events and reports what it found and how old it is: age is a first-class
output because a stale price is the usual way a mixed-asset backtest goes wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import REVISION_POLICIES
from .events import COLUMNS

PRICE_ORDER = ("quote", "trade", "bar", "mark", "settlement")


class Obs:
    """One event row, read lazily from the store's column arrays (attribute access: ``obs.bid``, ``obs.timestamp``, ``obs.curve_values``)."""

    __slots__ = ("_store", "row")

    def __init__(self, store: "PointInTimeStore", row: int):
        self._store, self.row = store, row

    def __getattr__(self, name):
        try:
            return self._store._cols[name][self.row]
        except KeyError:
            raise AttributeError(name) from None

    def get(self, name, default=None):
        v = self._store._cols[name][self.row]
        return default if v is None or (isinstance(v, float) and v != v) else v

    def as_dict(self) -> dict:
        return {c: self._store._cols[c][self.row] for c in COLUMNS}

    def __repr__(self):
        return f"Obs({self.instrument_id} {self.event_type} @{pd.Timestamp(self.timestamp)} avail {pd.Timestamp(self.available_at)})"


@dataclass(frozen=True)
class PriceSnapshot:
    instrument_id: str
    source: str                      # the event type the price came from
    timestamp: pd.Timestamp          # observation time
    available_at: pd.Timestamp
    mid: float
    bid: float
    ask: float
    last: float
    volume: float
    age: pd.Timedelta                # decision time minus observation time

    @property
    def spread(self) -> float:
        return self.ask - self.bid if np.isfinite(self.ask) and np.isfinite(self.bid) else float("nan")


class _Group:
    __slots__ = ("avail", "obs", "rows", "best", "has_revisions")

    def __init__(self, avail, obs, rows):
        self.avail, self.obs, self.rows = avail, obs, rows
        order = np.lexsort((np.arange(len(avail)), avail))              # rows arrive sorted by availability; keep that, ties by position
        assert (order == np.arange(len(avail))).all()
        best = np.empty(len(avail), dtype=np.int64)
        cur = 0
        for i in range(len(avail)):
            if (obs[i], avail[i]) >= (obs[cur], avail[cur]):
                cur = i
            best[i] = cur
        self.best = best
        self.has_revisions = len(np.unique(obs)) != len(obs)


class PointInTimeStore:
    def __init__(self, events: pd.DataFrame, revision_policy: str = "latest_known", max_age: str | pd.Timedelta | None = None):
        if revision_policy not in REVISION_POLICIES:
            raise ValueError(f"revision_policy must be one of {REVISION_POLICIES}")
        df = events
        if revision_policy == "first_release":
            df = df.sort_values(["available_at", "timestamp"], kind="stable").drop_duplicates(["instrument_id", "event_type", "timestamp"], keep="first")
        df = df.sort_values(["available_at", "timestamp", "instrument_id", "event_type", "revision"], kind="stable").reset_index(drop=True)
        self.events = df
        self.revision_policy = revision_policy
        self.max_age = None if max_age is None else pd.Timedelta(max_age)
        self._cols = {c: df[c].to_numpy() for c in COLUMNS}
        avail = df["available_at"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        obs = df["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        self._avail_all = avail
        self._groups: dict[tuple, _Group] = {}
        keys = list(zip(df["instrument_id"].to_numpy(), df["event_type"].to_numpy()))
        positions: dict[tuple, list] = {}
        for i, k in enumerate(keys):
            positions.setdefault(k, []).append(i)
        for k, rows in positions.items():
            rows = np.asarray(rows, dtype=np.int64)
            self._groups[k] = _Group(avail[rows], obs[rows], rows)
        self.max_available_returned = pd.Timestamp.min
        self.violations = 0
        self.queries = 0

    # -------------------------------------------------------------------------------------------------------------------------------- basics
    def instruments(self, event_type: str | None = None) -> list[str]:
        return sorted({i for i, e in self._groups if event_type is None or e == event_type})

    def event_types(self, instrument_id: str) -> list[str]:
        return sorted(e for i, e in self._groups if i == instrument_id)

    def has(self, instrument_id: str, event_type: str) -> bool:
        return (instrument_id, event_type) in self._groups

    @property
    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.events["available_at"].iloc[0]), pd.Timestamp(self.events["available_at"].iloc[-1])

    def _note(self, ts: pd.Timestamp, avail_ns: int):
        self.queries += 1
        a = pd.Timestamp(avail_ns)
        if a > self.max_available_returned:
            self.max_available_returned = a
        if a > ts:
            self.violations += 1

    def assert_no_lookahead(self):
        if self.violations:
            raise AssertionError(f"{self.violations} queries returned data that was not yet available")

    # ----------------------------------------------------------------------------------------------------------------------------- lookups
    def latest(self, instrument_id: str, event_type: str, ts, max_age=None) -> Obs | None:
        """The latest observation of ``(instrument, type)`` known at ``ts``, or None if there is none or it is older than ``max_age``."""
        g = self._groups.get((instrument_id, event_type))
        if g is None:
            return None
        ts = pd.Timestamp(ts)
        k = int(np.searchsorted(g.avail, ts.value, side="right"))
        if k == 0:
            return None
        j = int(g.best[k - 1])
        age_limit = self.max_age if max_age is None else pd.Timedelta(max_age)
        if age_limit is not None and ts.value - g.obs[j] > age_limit.value:
            return None
        self._note(ts, g.avail[j])
        return Obs(self, int(g.rows[j]))

    def history(self, instrument_id: str, event_type: str, field: str, ts, n: int | None = None, since=None) -> pd.Series:
        """The series of ``field`` over observation times known at ``ts`` (revisions resolved by the policy), last ``n`` points, from ``since`` if given."""
        g = self._groups.get((instrument_id, event_type))
        empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([], dtype="datetime64[ns]"), name=field)
        if g is None:
            return empty
        ts = pd.Timestamp(ts)
        k = int(np.searchsorted(g.avail, ts.value, side="right"))
        if k == 0:
            return empty
        if not g.has_revisions:
            obs, rows, av = g.obs[:k], g.rows[:k], g.avail[:k]
            if n is not None and not len(obs) <= n:
                order = np.argsort(obs, kind="stable")[-n:]
                obs, rows, av = obs[order], rows[order], av[order]
            elif len(obs) > 1 and np.any(np.diff(obs) < 0):
                order = np.argsort(obs, kind="stable")
                obs, rows, av = obs[order], rows[order], av[order]
        else:
            frame = pd.DataFrame({"obs": g.obs[:k], "avail": g.avail[:k], "row": g.rows[:k]}).sort_values(["obs", "avail"], kind="stable").drop_duplicates("obs", keep="last")
            if n is not None:
                frame = frame.tail(n)
            obs, rows, av = frame["obs"].to_numpy(), frame["row"].to_numpy(), frame["avail"].to_numpy()
        if since is not None:
            keep = obs >= pd.Timestamp(since).value
            obs, rows, av = obs[keep], rows[keep], av[keep]
        if len(av):
            self._note(ts, int(av.max()))
        values = self._cols[field][rows]
        if values.dtype == object:
            return pd.Series(values, index=pd.DatetimeIndex(obs.astype("datetime64[ns]")), name=field)
        return pd.Series(values.astype("float64"), index=pd.DatetimeIndex(obs.astype("datetime64[ns]")), name=field)

    def wide(self, instrument_ids, event_type: str, field: str, ts, n: int | None = None) -> pd.DataFrame:
        """A frame of ``field`` histories, one column per instrument, on the union of observation times (no forward fill: gaps stay visible)."""
        cols = {i: self.history(i, event_type, field, ts, n) for i in instrument_ids}
        return pd.DataFrame(cols).sort_index()

    def price(self, instrument_id: str, ts, order=PRICE_ORDER, max_age=None) -> PriceSnapshot | None:
        """The most recent price information for an instrument across event types: the observation with the latest timestamp, ties broken by ``order``."""
        ts = pd.Timestamp(ts)
        best: tuple | None = None
        for rank, et in enumerate(order):
            o = self.latest(instrument_id, et, ts, max_age)
            if o is None:
                continue
            key = (pd.Timestamp(o.timestamp).value, -rank)
            if best is None or key > best[0]:
                best = (key, et, o)
        if best is None:
            return None
        _, et, o = best
        bid = ask = last = float("nan")
        if et == "quote":
            bid, ask = float(o.bid), float(o.ask)
            last = float(o.trade) if o.trade == o.trade else (bid + ask) / 2.0
            mid = (bid + ask) / 2.0
        elif et == "trade":
            last = mid = float(o.trade)
        elif et == "bar":
            last = mid = float(o.close)
        elif et == "mark":
            last = mid = float(o.mark_price)
        else:
            last = mid = float(o.settlement)
        vol = float(o.volume) if o.volume == o.volume else float("nan")
        return PriceSnapshot(instrument_id, et, pd.Timestamp(o.timestamp), pd.Timestamp(o.available_at), mid, bid, ask, last, vol, ts - pd.Timestamp(o.timestamp))

    def curve(self, curve_id: str, ts, max_age=None) -> tuple[pd.Timestamp, dict] | None:
        o = self.latest(curve_id, "curve", ts, max_age)
        return None if o is None else (pd.Timestamp(o.timestamp), dict(o.curve_values))

    def funding(self, instrument_id: str, ts) -> Obs | None:
        return self.latest(instrument_id, "funding", ts)

    def fixing(self, index_id: str, date, ts) -> float | None:
        """The fixing of ``index_id`` observed on ``date`` as known at ``ts`` (None if not yet published)."""
        h = self.history(index_id, "fixing", "value", ts)
        if h.empty:
            return None
        d = pd.Timestamp(date).normalize()
        match = h[h.index.normalize() == d]
        return None if match.empty else float(match.iloc[-1])

    # ------------------------------------------------------------------------------------------------------------------------------ replay
    def slice(self, start=None, end=None, instrument_ids=None, event_types=None) -> pd.DataFrame:
        """Events with ``start < available_at <= end`` in delivery order."""
        a = self._avail_all
        lo = 0 if start is None else int(np.searchsorted(a, pd.Timestamp(start).value, side="right"))
        hi = len(a) if end is None else int(np.searchsorted(a, pd.Timestamp(end).value, side="right"))
        out = self.events.iloc[lo:hi]
        if instrument_ids is not None:
            out = out[out["instrument_id"].isin(list(instrument_ids))]
        if event_types is not None:
            out = out[out["event_type"].isin(list(event_types))]
        return out
