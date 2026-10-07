"""Marks: the price (or present value) at which each open position is carried, with the age of the information it rests on.

Most instruments are marked from their own price events (a quote mid, a bar close, a settlement, an exchange mark price); the preference order depends on the instrument: futures use the
official settlement when there is one, perpetuals the exchange mark (the price margin and liquidation use), options the quote mid. OTC contracts whose value is a model output (swaps,
FX forwards, basis swaps) are marked by a pricing model registered for their instrument type (``MARK_MODELS``): the swap module registers the curve-based present-value models.

A mark older than ``max_age`` is not used: ``mark`` returns ``None`` and the engine keeps the last mark and counts a stale mark, which is how a position in an illiquid instrument
shows up in the diagnostics instead of being valued at a number nobody has seen for a month.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..marketdata.store import PointInTimeStore

MARK_ORDER = {"future": ("settlement", "quote", "trade", "bar", "mark"), "crypto_perp": ("mark", "quote", "trade", "bar"), "crypto_future": ("mark", "quote", "trade", "bar", "settlement"),
              "option": ("quote", "trade", "bar", "settlement"), "default": ("quote", "trade", "bar", "mark", "settlement")}

MARK_MODELS: dict[str, Callable] = {}                    # instrument_type -> fn(provider, instrument, ts) -> MarkResult | None


def register_mark_model(instrument_type: str):
    def wrap(fn):
        MARK_MODELS[instrument_type] = fn
        return fn
    return wrap


@dataclass(frozen=True)
class MarkResult:
    price: float
    age: pd.Timedelta
    source: str
    bid: float = float("nan")          # model marks (swaps, forwards) can carry the bid and ask present values: the cost of trading them is the distance between the two
    ask: float = float("nan")


class MarkProvider:
    def __init__(self, store: PointInTimeStore, registry, max_age=None, curve_ids: dict | None = None):
        self.store, self.registry = store, registry
        self.max_age = None if max_age is None else pd.Timedelta(max_age)
        self.curve_ids = curve_ids or {}

    def underlying_price(self, inst, ts) -> float:
        if inst.underlying_id is None or inst.underlying_id not in self.registry:
            return float("nan")
        r = self.mark(self.registry.get(inst.underlying_id), ts)
        return float("nan") if r is None else r.price

    def mark(self, inst, ts) -> MarkResult | None:
        ts = pd.Timestamp(ts)
        if inst.expiry is not None and ts > inst.expiry + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1):
            return None
        model = MARK_MODELS.get(inst.instrument_type)
        if model is not None:
            return model(self, inst, ts)
        order = MARK_ORDER.get(inst.instrument_type, MARK_ORDER["default"])
        snap = self.store.price(inst.instrument_id, ts, order=order, max_age=self.max_age)
        if snap is None:
            return None
        if inst.instrument_type in ("future", "crypto_perp", "crypto_future") and snap.source != order[0]:
            # prefer the preferred type when it is known at all and not older than the alternative by more than one day
            pref = self.store.latest(inst.instrument_id, order[0], ts, self.max_age)
            if pref is not None and (snap.timestamp - pd.Timestamp(pref.timestamp)) <= pd.Timedelta(days=1):
                v = {"settlement": pref.settlement, "mark": pref.mark_price}.get(order[0])
                if v is not None and v == v:
                    return MarkResult(float(v), ts - pd.Timestamp(pref.timestamp), order[0])
        return MarkResult(snap.mid, snap.age, snap.source, snap.bid, snap.ask)

    def mark_all(self, ts, instrument_ids) -> tuple[dict, dict]:
        """``(marks, ages)`` for the instruments that can be marked now."""
        marks, ages = {}, {}
        for iid in instrument_ids:
            r = self.mark(self.registry.get(iid), ts)
            if r is not None:
                marks[iid], ages[iid] = r.price, r.age
        return marks, ages
