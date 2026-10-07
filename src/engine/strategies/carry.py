"""Cross-asset carry on the common API: futures roll yield, FX interest differentials and perpetual funding, ranked within asset class.

*Carry* is what a position earns if prices do not move: for a future the annualised roll yield ``ln(F1 / F2) / (T2 - T1)`` (backwardation pays the long), for a currency pair the interest
rate of the base currency minus the quote currency's (``RATE-{ccy}`` reference series), for a long perpetual MINUS the annualised funding it pays. Carries are divided by volatility to
make them comparable (a carry Sharpe), standardised within each asset class and traded long-short: long the high-carry, short the low-carry, dollar neutral per class, with weights
sized by inverse volatility to a volatility target (Koijen, Moskowitz, Pedersen and Vrugt 2018). A custom carry function per instrument (``carry_overrides``) can replace any of these.
"""

from __future__ import annotations

import numpy as np

from ..strategy import Schedule, Signal, Strategy, Target
from .common import ann_vol, risk_weights


class CarryStrategy(Strategy):
    name = "carry"
    schedule = Schedule("weekly", "16:30", 4, "US")

    def __init__(self, futures_chains=(), fx_pairs=(), perps=(), carry_overrides: dict | None = None, target_vol: float = 0.10, vol_window: int = 60, max_leverage: float = 3.0,
                 min_per_class: int = 2, name: str | None = None):
        self.futures_chains, self.fx_pairs, self.perps = tuple(futures_chains), tuple(fx_pairs), tuple(perps)
        self.carry_overrides = carry_overrides or {}
        self.target_vol, self.vol_window, self.max_leverage, self.min_per_class = target_vol, vol_window, max_leverage, min_per_class
        if name:
            self.name = name

    # ----------------------------------------------------------------------------------------------------------------------------------- carries
    def futures_carry(self, ctx, chain_id: str) -> float:
        chain = ctx.registry.chain(chain_id)
        c1, c2 = chain.front(ctx.ts, 0), chain.front(ctx.ts, 1)
        if c1 is None or c2 is None:
            return float("nan")
        p1, p2 = ctx.data.mid(c1.instrument_id), ctx.data.mid(c2.instrument_id)
        dt = (c2.expiry - c1.expiry).days / 365.0
        if not (np.isfinite(p1) and np.isfinite(p2) and p1 > 0 and p2 > 0 and dt > 0):
            return float("nan")
        return float(np.log(p1 / p2) / dt)

    def fx_carry(self, ctx, pair_id: str) -> float:
        inst = ctx.registry.get(pair_id)
        rb, rq = ctx.data.reference(f"RATE-{inst.base_currency}"), ctx.data.reference(f"RATE-{inst.quote_currency}")
        return float(rb - rq) if np.isfinite(rb) and np.isfinite(rq) else float("nan")

    def perp_carry(self, ctx, perp_id: str) -> float:
        inst = ctx.registry.get(perp_id)
        rate = ctx.data.funding(perp_id)
        return float(-rate * 24.0 / inst.funding_interval_hours * 365.0) if np.isfinite(rate) else float("nan")

    def carries(self, ctx) -> dict[str, tuple[str, float]]:
        out = {}
        for c in self.futures_chains:
            out[c] = ("futures", self.futures_carry(ctx, c))
        for p in self.fx_pairs:
            out[p] = ("fx", self.fx_carry(ctx, p))
        for p in self.perps:
            out[p] = ("crypto", self.perp_carry(ctx, p))
        for iid, fn in self.carry_overrides.items():
            out[iid] = (out.get(iid, ("custom", 0))[0], float(fn(ctx)))
        return out

    def generate_signals(self, ctx):
        raw = self.carries(ctx)
        by_class: dict[str, list] = {}
        for iid, (cls, c) in raw.items():
            vol = ann_vol(ctx, iid, self.vol_window)
            if np.isfinite(c) and np.isfinite(vol) and vol > 0:
                by_class.setdefault(cls, []).append((iid, c, c / vol))
        out = []
        for cls, items in by_class.items():
            if len(items) < self.min_per_class:
                continue
            s = np.array([x[2] for x in items])
            z = (s - s.mean()) / (s.std(ddof=0) or 1.0)
            for (iid, c, cs), zi in zip(items, z):
                out.append(Signal(iid, float(np.clip(zi, -2.0, 2.0)), 1.0, meta={"carry": c, "carry_sharpe": cs, "class": cls}))
        return out

    def map_to_targets(self, ctx, signals):
        sig = {s.instrument_id: s.value for s in signals}
        vols = {i: ann_vol(ctx, i, self.vol_window) for i in sig}
        w = risk_weights(sig, vols, self.target_vol, self.max_leverage)
        return [Target(i, weight=v) for i, v in w.items()]
