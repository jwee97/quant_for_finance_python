"""Point-in-time data access for strategies: what a trader at the engine's current time could see.

:class:`PITData` wraps the market-data store with the engine clock fixed in it. Every method answers "as of now": a strategy cannot ask for tomorrow's price or the next
settlement because there is no argument that lets it. Prices of any instrument come back through one interface (``price``, ``history``, ``returns``), whichever event type the data
arrive as (a bar, a settlement, a trade, a mark or a quote mid); curves, funding, fixings, volume, open interest, futures chains and option chains have their own methods.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..instruments.futures import FutureChain
from ..instruments.options import Option
from ..marketdata.store import PointInTimeStore, PriceSnapshot

SERIES_ORDER = ("bar", "settlement", "trade", "mark", "quote")


class PITData:
    def __init__(self, store: PointInTimeStore, registry, now: Callable[[], pd.Timestamp], max_age=None):
        self.store, self.registry, self._now, self.max_age = store, registry, now, max_age

    @property
    def now(self) -> pd.Timestamp:
        return self._now()

    # ----------------------------------------------------------------------------------------------------------------------------- prices
    def price(self, instrument_id: str, max_age=None) -> PriceSnapshot | None:
        return self.store.price(instrument_id, self.now, max_age=max_age or self.max_age)

    def mid(self, instrument_id: str, max_age=None) -> float:
        p = self.price(instrument_id, max_age)
        return float("nan") if p is None else p.mid

    def _series_type(self, instrument_id: str) -> str | None:
        for et in SERIES_ORDER:
            if self.store.has(instrument_id, et):
                return et
        return None

    def history(self, instrument_id: str, n: int | None = 252, event_type: str | None = None, since=None) -> pd.Series:
        """Prices (close, settlement, trade, mark or quote mid, in that order of preference) known now, indexed by observation time."""
        et = event_type or self._series_type(instrument_id)
        if et is None:
            return pd.Series(dtype="float64", index=pd.DatetimeIndex([], dtype="datetime64[ns]"))
        if et == "quote":
            b = self.store.history(instrument_id, "quote", "bid", self.now, n, since)
            a = self.store.history(instrument_id, "quote", "ask", self.now, n, since)
            return ((b + a) / 2.0).rename(instrument_id)
        field = {"bar": "close", "settlement": "settlement", "trade": "trade", "mark": "mark_price"}[et]
        return self.store.history(instrument_id, et, field, self.now, n, since).rename(instrument_id)

    def returns(self, instrument_id: str, n: int | None = 252, log: bool = True) -> pd.Series:
        p = self.history(instrument_id, None if n is None else n + 1)
        r = np.log(p).diff() if log else p.pct_change()
        return r.dropna().tail(n) if n is not None else r.dropna()

    def wide(self, instrument_ids, n: int | None = 252, ffill_limit: int = 3) -> pd.DataFrame:
        frame = pd.DataFrame({i: self.history(i, n) for i in instrument_ids}).sort_index()
        return frame.ffill(limit=ffill_limit)

    def volume(self, instrument_id: str, n: int | None = 20) -> pd.Series:
        for et in ("bar", "trade"):
            if self.store.has(instrument_id, et):
                return self.store.history(instrument_id, et, "volume", self.now, n)
        return pd.Series(dtype="float64")

    def open_interest(self, instrument_id: str, n: int | None = 20) -> pd.Series:
        for et in ("open_interest", "settlement", "bar", "quote"):
            if self.store.has(instrument_id, et):
                s = self.store.history(instrument_id, et, "open_interest", self.now, n).dropna()
                if len(s):
                    return s
        return pd.Series(dtype="float64")

    # -------------------------------------------------------------------------------------------------------------------- curves and series
    def curve(self, curve_id: str, max_age=None):
        """``(observation time, {tenor in years: rate})`` of the latest curve known now, or None."""
        return self.store.curve(curve_id, self.now, max_age or self.max_age)

    def funding(self, instrument_id: str) -> float:
        o = self.store.latest(instrument_id, "funding", self.now)
        return float("nan") if o is None else float(o.funding)

    def predicted_funding(self, instrument_id: str) -> float:
        o = self.store.latest(instrument_id, "funding_predicted", self.now)
        return float("nan") if o is None else float(o.funding)

    def funding_history(self, instrument_id: str, n: int | None = 90) -> pd.Series:
        return self.store.history(instrument_id, "funding", "funding", self.now, n)

    def fixing(self, index_id: str, date) -> float | None:
        return self.store.fixing(index_id, date, self.now)

    def reference(self, series_id: str) -> float:
        o = self.store.latest(series_id, "reference", self.now)
        return float("nan") if o is None else float(o.value)

    # -------------------------------------------------------------------------------------------------------------------------- structure
    def instrument(self, instrument_id: str):
        return self.registry.get(instrument_id)

    def chain(self, chain_id: str) -> FutureChain:
        return self.registry.chain(chain_id)

    def front(self, chain_id: str, rank: int = 0):
        return self.registry.chain(chain_id).front(self.now, rank)

    def universe(self, asset_class: str | None = None, with_data: bool = True) -> list[str]:
        out = []
        for inst in self.registry.active(self.now):
            if asset_class is not None and inst.asset_class != asset_class:
                continue
            if with_data and self._series_type(inst.instrument_id) is None:
                continue
            out.append(inst.instrument_id)
        return sorted(out)

    def option_chain(self, underlying_id: str, min_days: float = 0.0, max_days: float | None = None) -> pd.DataFrame:
        """Option contracts on ``underlying_id`` alive now with a quote known now: ``instrument_id, expiry, strike, right, bid, ask, mid, days``."""
        now = self.now
        rows = []
        for inst in self.registry.by_underlying(underlying_id):
            if not isinstance(inst, Option) or inst.expiry <= now:
                continue
            days = (inst.expiry - now).total_seconds() / 86400.0
            if days < min_days or (max_days is not None and days > max_days):
                continue
            o = self.store.latest(inst.instrument_id, "quote", now, self.max_age)
            if o is None or not (o.bid == o.bid and o.ask == o.ask):
                continue
            rows.append({"instrument_id": inst.instrument_id, "expiry": inst.expiry, "strike": inst.strike, "right": inst.right, "bid": float(o.bid), "ask": float(o.ask),
                         "mid": (float(o.bid) + float(o.ask)) / 2.0, "days": days})
        return pd.DataFrame(rows)
