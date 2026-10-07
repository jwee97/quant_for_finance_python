"""Volatility strategies on the common API: the variance premium, delta-hedged options (long or short gamma), skew and term-structure trades.

:class:`VolatilityPremiumStrategy` works on one underlying with a listed option chain. At each decision it reads the chain known now (``ctx.data.option_chain``), picks the expiry
closest to ``target_days`` and the at-the-money pair, inverts each option's mid price to an implied volatility, compares it with the underlying's realised volatility
(``rv_window`` days of returns) and trades by the sign of the difference:

* ``mode='vrp'``: SELL the straddle (or a strangle at ``wing`` standard deviations) when implied volatility exceeds realised by ``edge``, buy it back below ``-edge`` or at ``exit``;
* ``mode='gamma'``: BUY the straddle when realised volatility exceeds implied by ``edge`` (gamma scalping);
* ``mode='skew'``: trade the risk reversal (25-delta-ish put against call) when the skew is far from its history;
* ``mode='term'``: sell the near straddle against the far one when the near implied volatility is rich against the far.

Positions are sized by a VEGA budget (currency per volatility point) and delta-hedged with the underlying at every decision (the hedge uses the options' Greeks at their implied
volatilities); options are closed ``close_days`` before expiry. Every trade pays the quoted bid/ask of the options and the hedge pays the underlying's costs, so what the strategy earns is
net of the costs that dominate option backtests.
"""

from __future__ import annotations

import numpy as np

from ...derivatives.iv import implied_vol
from ..strategy import Schedule, Signal, Strategy, Target


class VolatilityPremiumStrategy(Strategy):
    name = "volatility"
    schedule = Schedule("daily", "16:30", 4, "US")

    def __init__(self, underlying_id: str, mode: str = "vrp", target_days: float = 30.0, close_days: float = 7.0, rv_window: int = 21, edge: float = 0.03, exit: float = 0.0, skew_z: float = 1.5,
                 vega_budget: float = 500.0, wing: float = 0.0, hedge: bool = True, rate: float = 0.0, max_notional: float = 1.0, name: str | None = None):
        if mode not in ("vrp", "gamma", "skew", "term"):
            raise ValueError("mode must be vrp, gamma, skew or term")
        self.underlying_id, self.mode, self.target_days, self.close_days, self.rv_window = underlying_id, mode, target_days, close_days, rv_window
        self.skew_z = skew_z
        self.edge, self.exit, self.vega_budget, self.wing, self.hedge, self.rate, self.max_notional = edge, exit, vega_budget, wing, hedge, rate, max_notional
        if name:
            self.name = name

    # ------------------------------------------------------------------------------------------------------------------------- market reading
    def _chain(self, ctx):
        return ctx.data.option_chain(self.underlying_id, min_days=self.close_days + 1)

    def _pick_expiry(self, chain, target):
        days = chain.groupby("expiry")["days"].first()
        return days.index[np.argmin(np.abs(days.values - target))]

    def _atm(self, chain, expiry, S):
        sub = chain[chain["expiry"] == expiry]
        k = sub["strike"].iloc[(sub["strike"] - S).abs().argsort().iloc[0]]
        call = sub[(sub["strike"] == k) & (sub["right"] == "call")]
        put = sub[(sub["strike"] == k) & (sub["right"] == "put")]
        return (None, None, k) if call.empty or put.empty else (call.iloc[0], put.iloc[0], k)

    def _iv(self, ctx, row, S):
        inst = ctx.registry.get(row["instrument_id"])
        T = inst.time_to_expiry(ctx.ts)
        if T <= 0:
            return float("nan")
        return float(np.asarray(implied_vol(row["mid"], S, inst.strike, T, self.rate, 0.0, inst.is_call)).ravel()[0])

    def realised(self, ctx) -> float:
        r = ctx.data.returns(self.underlying_id, self.rv_window)
        return float(r.std(ddof=1) * np.sqrt(252.0)) if len(r) >= max(8, self.rv_window // 2) else float("nan")

    def generate_signals(self, ctx):
        S = ctx.data.mid(self.underlying_id)
        chain = self._chain(ctx)
        if chain.empty or not np.isfinite(S):
            return []
        expiry = self._pick_expiry(chain, self.target_days)
        call, put, k = self._atm(chain, expiry, S)
        if call is None:
            return []
        iv = np.nanmean([self._iv(ctx, call, S), self._iv(ctx, put, S)])
        rv = self.realised(ctx)
        if not (np.isfinite(iv) and np.isfinite(rv)):
            return []
        meta = {"iv": float(iv), "rv": rv, "expiry": expiry, "strike": float(k), "call": call["instrument_id"], "put": put["instrument_id"], "S": S}
        if self.mode in ("vrp", "gamma"):
            value = iv - rv if self.mode == "vrp" else rv - iv
        elif self.mode == "skew":
            sub = chain[chain["expiry"] == expiry]
            lo = sub[(sub["right"] == "put") & (sub["strike"] <= S * 0.95)]
            hi = sub[(sub["right"] == "call") & (sub["strike"] >= S * 1.05)]
            if lo.empty or hi.empty:
                return []
            lo, hi = lo.iloc[(lo["strike"] - S * 0.95).abs().argsort().iloc[0]], hi.iloc[(hi["strike"] - S * 1.05).abs().argsort().iloc[0]]
            skew = self._iv(ctx, lo, S) - self._iv(ctx, hi, S)
            hist = ctx.store.setdefault("skew", [])
            hist.append(skew)
            h = np.asarray(hist[-60:])
            value = float((skew - h.mean()) / (h.std(ddof=1) or np.nan)) if len(h) >= 20 else float("nan")
            meta.update({"put_wing": lo["instrument_id"], "call_wing": hi["instrument_id"], "skew": skew})
        else:
            far = self._pick_expiry(chain[chain["days"] > self.target_days * 1.8], self.target_days * 2.5) if (chain["days"] > self.target_days * 1.8).any() else None
            if far is None:
                return []
            fc, fp, _ = self._atm(chain, far, S)
            if fc is None:
                return []
            iv_far = np.nanmean([self._iv(ctx, fc, S), self._iv(ctx, fp, S)])
            value = float(iv - iv_far)
            meta.update({"far_call": fc["instrument_id"], "far_put": fp["instrument_id"], "iv_far": float(iv_far)})
        return [Signal(self.underlying_id, float(value), meta=meta)]

    # --------------------------------------------------------------------------------------------------------------------------- positions
    def _legs_for(self, sig: Signal, side: int) -> list[tuple[str, float]]:
        m = sig.meta
        if self.mode in ("vrp", "gamma"):
            return [(m["call"], -side), (m["put"], -side)]            # side +1 = short the straddle
        if self.mode == "skew":
            return [(m["put_wing"], -side), (m["call_wing"], side)]
        return [(m["call"], -side), (m["put"], -side), (m["far_call"], side), (m["far_put"], side)]

    def map_to_targets(self, ctx, signals):
        held = ctx.store.get("legs", [])
        today = ctx.ts
        side = ctx.store.get("side", 0)
        targets: list[Target] = []
        if held:                                                           # close positions near expiry
            nearest = min(ctx.registry.get(i).expiry for i, _ in held)
            if (nearest - today).total_seconds() / 86400.0 <= self.close_days:
                side = 0
        if signals and side == 0 and not held:
            v = signals[0].value
            if self.mode in ("vrp", "term", "skew"):
                edge = self.skew_z if self.mode == "skew" else self.edge
                side = 1 if v > edge else (-1 if v < -edge and self.mode != "vrp" else 0)
            else:
                side = 1 if v > self.edge else 0
            if side != 0:
                targets += self._enter(ctx, signals[0], side)
        elif signals and held:
            v = signals[0].value
            if (side == 1 and v < self.exit) or (side == -1 and v > -self.exit):
                side = 0
        if side == 0 and held:
            targets += [Target(i, quantity=0.0) for i, _ in held]
            ctx.store["legs"] = []
        elif held and side != 0:
            targets += [Target(i, quantity=q) for i, q in held]
        ctx.store["side"] = side
        if self.hedge:
            targets += self._hedge(ctx, signals, side)
        return targets

    def _enter(self, ctx, sig: Signal, side: int) -> list[Target]:
        S = sig.meta["S"]
        legs = self._legs_for(sig, side)
        # size by vega: the budget is currency per vol point for one unit of the structure
        vega = 0.0
        for iid, sign in legs:
            inst = ctx.registry.get(iid)
            iv = sig.meta["iv"] if self.mode != "term" else sig.meta["iv"]
            g = inst.greeks(S, iv, ctx.ts, self.rate)
            vega += abs(g["vega"]) * 0.01 * inst.contract_multiplier
        units = max(self.vega_budget / vega, 0.0) if vega > 0 else 0.0
        per_unit = max(ctx.registry.get(iid).notional(S, 1.0) for iid, _ in legs)
        if per_unit > 0:                                                   # the underlying notional of the structure never exceeds ``max_notional`` times the strategy's capital
            units = min(units, self.max_notional * ctx.portfolio.capital / per_unit)
        out, held = [], []
        for iid, sign in legs:
            inst = ctx.registry.get(iid)
            q = inst.round_quantity(sign * units)
            if q != 0:
                out.append(Target(iid, quantity=q))
                held.append((iid, q))
        ctx.store["legs"] = held
        return out

    def _hedge(self, ctx, signals, side: int) -> list[Target]:
        legs = ctx.store.get("legs", [])
        if not legs:
            return [Target(self.underlying_id, quantity=0.0)]
        S = ctx.data.mid(self.underlying_id)
        under = ctx.registry.get(self.underlying_id)
        delta = 0.0
        for iid, q in legs:
            inst = ctx.registry.get(iid)
            mid = ctx.data.mid(iid)
            T = inst.time_to_expiry(ctx.ts)
            if T <= 0 or not np.isfinite(mid) or not np.isfinite(S):
                continue
            iv = float(np.asarray(implied_vol(mid, S, inst.strike, T, self.rate, 0.0, inst.is_call)).ravel()[0])
            if not np.isfinite(iv):
                continue
            delta += inst.greeks(S, iv, ctx.ts, self.rate)["delta"] * inst.contract_multiplier * q
        return [Target(self.underlying_id, quantity=under.round_quantity(-delta / under.contract_multiplier))]
