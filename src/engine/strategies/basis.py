"""Basis strategies: cash-and-carry, reverse cash-and-carry and basis mean reversion between a spot (or index) and its perpetual or dated future.

**Cash-and-carry** buys the spot and sells the derivative in the same notional when the derivative is rich: the position is market neutral and earns the basis (and, for a perpetual,
the funding the short receives) until convergence. **Reverse cash-and-carry** (sell spot or borrow it, buy the derivative) earns the opposite when the derivative trades cheap.

The annualised basis is ``ln(F / S) / T`` for a dated future and the observed premium ``F / S - 1`` annualised by an assumed horizon for a perpetual; the expected annual yield of
the cash-and-carry adds the funding the short perpetual earns (``funding x periods per year``) and subtracts the round-trip trading costs spread over the expected holding period.
Enter when the expected yield exceeds ``entry`` and exit when it falls below ``exit``; with ``mode='mean_reversion'`` the signal is the z-score of the premium against its own history
and the legs are held until the z-score is back inside ``exit_z`` (a trade on convergence of the premium rather than a carry harvest).
"""

from __future__ import annotations

import numpy as np

from ..strategy import Schedule, Signal, Strategy, Target


class BasisStrategy(Strategy):
    name = "basis"
    schedule = Schedule("daily", "16:30", 4, "WEEKDAY")

    def __init__(self, spot_id: str, deriv_id: str, weight: float = 0.5, entry: float = 0.08, exit: float = 0.02, mode: str = "carry", lookback: int = 60, entry_z: float = 1.5,
                 exit_z: float = 0.3, round_trip_cost: float = 0.001, hold_days: float = 30.0, name: str | None = None):
        if mode not in ("carry", "mean_reversion"):
            raise ValueError("mode must be carry or mean_reversion")
        self.spot_id, self.deriv_id, self.weight, self.entry, self.exit, self.mode = spot_id, deriv_id, weight, entry, exit, mode
        self.lookback, self.entry_z, self.exit_z, self.round_trip_cost, self.hold_days = lookback, entry_z, exit_z, round_trip_cost, hold_days
        if name:
            self.name = name

    def premium(self, ctx) -> float:
        s, f = ctx.data.mid(self.spot_id), ctx.data.mid(self.deriv_id)
        return float(f / s - 1.0) if np.isfinite(s) and np.isfinite(f) and s > 0 else float("nan")

    def expected_yield(self, ctx) -> float:
        prem = self.premium(ctx)
        if not np.isfinite(prem):
            return float("nan")
        d = ctx.registry.get(self.deriv_id)
        if d.expiry is not None:
            T = max((d.expiry - ctx.ts).total_seconds() / (365.0 * 86400.0), 1.0 / 365.0)
            basis = np.log1p(prem) / T
            funding = 0.0
        else:
            basis = 0.0
            fund = ctx.data.funding(self.deriv_id)
            predicted = ctx.data.predicted_funding(self.deriv_id)
            rate = predicted if np.isfinite(predicted) else fund
            funding = rate * 24.0 / d.funding_interval_hours * 365.0 if np.isfinite(rate) else 0.0
        cost = self.round_trip_cost * 365.0 / self.hold_days
        return float(basis + funding - cost if prem >= 0 or d.expiry is None else basis + funding + cost)

    def generate_signals(self, ctx):
        prem = self.premium(ctx)
        if not np.isfinite(prem):
            return []
        hist = ctx.store.setdefault("prem", [])
        hist.append(prem)
        z = float("nan")
        if len(hist) >= max(10, self.lookback // 3):
            h = np.asarray(hist[-self.lookback:])
            z = float((prem - h.mean()) / (h.std(ddof=1) or np.nan))
        return [Signal(self.deriv_id, self.expected_yield(ctx) if self.mode == "carry" else -z, meta={"premium": prem, "z": z})]

    def map_to_targets(self, ctx, signals):
        state = ctx.store.get("state", 0)
        if not signals or not np.isfinite(signals[0].value):
            return self._hold(state, ctx)
        v = signals[0].value
        want = state
        if self.mode == "carry":
            if state == 0:
                want = 1 if v > self.entry else (-1 if v < -self.entry else 0)
            elif (state == 1 and v < self.exit) or (state == -1 and v > -self.exit):
                want = 0
        else:
            z = signals[0].meta["z"]
            if state == 0 and np.isfinite(z):
                want = 1 if z > self.entry_z else (-1 if z < -self.entry_z else 0)        # premium rich: cash-and-carry (long spot, short derivative)
            elif state != 0 and (not np.isfinite(z) or abs(z) < self.exit_z):
                want = 0
        ctx.store["state"] = want
        return self._hold(want, ctx)

    def _hold(self, state: int, ctx) -> list[Target]:
        w = self.weight * state
        return [Target(self.spot_id, weight=w), Target(self.deriv_id, weight=-w)]
