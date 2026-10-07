"""Relative-value strategies: calendar spreads, futures butterflies and FX triangular dislocations, traded as hedged packages on the common API.

* **calendar**: the log ratio of the second to the first contract of a future chain (the spread's value) is compared with its own history; a z-score above ``entry_z`` sells the
  expensive deferred contract against the front (and vice versa), notional neutral. The legs are held until the z-score returns inside ``exit_z`` and are rolled by the engine with the
  chain, so a position is always in the two designated contracts.
* **butterfly**: ``ln F1 - 2 ln F2 + ln F3`` (curvature) against its history, traded as ``+1, -2, +1`` contracts in notional terms.
* **fx_triangle**: for three pairs that form a cycle (``EURUSD``, ``USDJPY``, ``EURJPY``) the log gap ``ln(EURUSD x USDJPY / EURJPY)`` should be zero; a gap beyond ``entry_z``
  sigmas of its own history is traded back toward zero (a hedged three-leg trade).

All z-scores use only point-in-time history kept in the strategy's own store.
"""

from __future__ import annotations

import numpy as np

from ..strategy import Schedule, Signal, Strategy, Target


class RelativeValueStrategy(Strategy):
    name = "relative_value"
    schedule = Schedule("daily", "16:30", 4, "US")

    def __init__(self, kind: str = "calendar", chain_id: str | None = None, pairs: tuple = (), weight: float = 0.25, lookback: int = 60, entry_z: float = 1.5, exit_z: float = 0.3,
                 name: str | None = None):
        if kind not in ("calendar", "butterfly", "fx_triangle"):
            raise ValueError("kind must be calendar, butterfly or fx_triangle")
        if kind in ("calendar", "butterfly") and not chain_id:
            raise ValueError("calendar and butterfly need a chain_id")
        if kind == "fx_triangle" and len(pairs) != 3:
            raise ValueError("fx_triangle needs three pairs: (EURUSD, USDJPY, EURJPY)")
        self.kind, self.chain_id, self.pairs, self.weight, self.lookback, self.entry_z, self.exit_z = kind, chain_id, tuple(pairs), weight, lookback, entry_z, exit_z
        if name:
            self.name = name

    def _legs(self, ctx):
        if self.kind == "fx_triangle":
            return list(self.pairs)
        chain = ctx.registry.chain(self.chain_id)
        n = 2 if self.kind == "calendar" else 3
        legs = [chain.front(ctx.ts, k) for k in range(n)]
        return None if any(l is None for l in legs) else [l.instrument_id for l in legs]

    def value(self, ctx, legs) -> float:
        p = [ctx.data.mid(i) for i in legs]
        if not all(np.isfinite(x) and x > 0 for x in p):
            return float("nan")
        if self.kind == "calendar":
            return float(np.log(p[1] / p[0]))
        if self.kind == "butterfly":
            return float(np.log(p[0]) - 2.0 * np.log(p[1]) + np.log(p[2]))
        return float(np.log(p[0]) + np.log(p[1]) - np.log(p[2]))

    def generate_signals(self, ctx):
        legs = self._legs(ctx)
        if legs is None:
            return []
        v = self.value(ctx, legs)
        if not np.isfinite(v):
            return []
        hist = ctx.store.setdefault("values", [])
        # a roll changes the contracts behind the spread: a spread history is only comparable within the same designated contracts
        if ctx.store.get("legs") != legs and self.kind != "fx_triangle":
            ctx.store["legs"] = legs
            hist.clear()
        hist.append(v)
        h = np.asarray(hist[-self.lookback:])
        if len(h) < max(10, self.lookback // 3):
            return []
        z = float((v - h.mean()) / (h.std(ddof=1) or np.nan))
        return [Signal("spread", z, meta={"value": v, "legs": legs})]

    def map_to_targets(self, ctx, signals):
        state = ctx.store.get("state", 0)
        legs = ctx.store.get("held_legs")
        if signals and np.isfinite(signals[0].value):
            z, cur_legs = signals[0].value, signals[0].meta["legs"]
            want = state
            if state == 0:
                want = -1 if z > self.entry_z else (1 if z < -self.entry_z else 0)           # +1: long the spread (buy the high leg's relative cheapness)
            elif abs(z) < self.exit_z or (state == 1 and z > self.exit_z * 3) and False:
                want = 0
            if want != state:
                state = want
                ctx.store["state"] = state
                ctx.store["held_legs"] = legs = cur_legs if state != 0 else None
            elif state != 0 and legs is not None and legs != cur_legs and self.kind != "fx_triangle":
                ctx.store["held_legs"] = legs = cur_legs                                       # the chain rolled: re-establish in the new contracts
        if state != 0 and legs is not None and self.kind != "fx_triangle":
            current = self._legs(ctx)                                                           # between signals (a roll empties the history) keep holding the designated contracts
            if current is not None and current != legs:
                ctx.store["held_legs"] = legs = current
        if state == 0 or legs is None:
            stale = ctx.store.pop("last_targets", None) or []                       # flatten once; ids that have expired since then are skipped
            return [Target(i, weight=0.0) for i in stale if ctx.registry.get(i).expiry is None or ctx.registry.get(i).expiry >= ctx.ts]
        w = self.weight * state
        coef = {"calendar": (-1.0, 1.0), "butterfly": (1.0, -2.0, 1.0), "fx_triangle": (1.0, 1.0, -1.0)}[self.kind]
        targets = [Target(i, weight=w * c) for i, c in zip(legs, coef)]
        ctx.store["last_targets"] = list(legs)
        return targets
