"""Option strategies for the options backtester: premium selling, covered calls, spreads, hedged volatility trades and a skew trade.

Every strategy sees the chain, the spot, the position and the underlying history through the close of the decision date only, and its orders fill on a later date at the
quotes then (see ``backtest``). Delta, DTE and width parameters follow the conventions of the practitioner literature on systematic option selling (Israelov & Nielsen 2015
on covered calls; Bondarenko 2014 and Carr & Wu 2009 on the variance risk premium; Coval & Shumway 2001 on index straddles; Cboe PUT / BXM benchmark indices).

``ShortPut``              cash-secured put: sell a ~25-delta put at ~45 DTE, hold to ``close_dte`` days before expiry, roll
``CoveredCall``           long the underlying, short a ~30-delta call (Cboe BXM-style), roll monthly
``ShortStrangle``         short 16-delta put and call, optionally delta-hedged
``IronCondor``            the strangle with long 5-delta wings: defined risk
``DeltaHedgedStraddle``   short (``side=-1``) or long (``+1``) the at-the-money straddle with the delta hedged daily: a pure bet on realised against implied volatility
``RiskReversal``          long a 25-delta call against a short 25-delta put, delta-hedged: the skew premium
``CalendarSpread``        short the front-month ATM option, long the back month
``VRPTimed``              wraps another strategy and trades it only while the TRAILING variance risk premium (implied minus realised volatility) is positive
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Hedge, Order, OptionStrategy, StrategyContext, pick_atm, pick_by_delta, pick_expiry, quoted


class _Roller(OptionStrategy):
    """Shared logic: flat -> open a package; ``close_dte`` days before expiry (or at expiry) -> close it and open the next one."""

    def __init__(self, dte: int = 45, close_dte: int = 7, size: float = 1.0, min_open_dte: int = 20):
        if dte < 7 or close_dte < 0 or size <= 0 or min_open_dte < 1:
            raise ValueError("dte >= 7, close_dte >= 0, size > 0, min_open_dte >= 1")
        self.dte, self.close_dte, self.size, self.min_open_dte = dte, close_dte, size, min_open_dte

    def legs(self, ctx: StrategyContext, expiry) -> list[Order] | None:
        raise NotImplementedError

    def hedge_units(self, ctx: StrategyContext) -> float | None:
        return None

    def on_date(self, ctx: StrategyContext) -> list:
        orders: list = []
        if ctx.positions:
            expiry = min(k[0] for k in ctx.positions)
            if (expiry - ctx.date).days <= self.close_dte:
                orders += [Order(k[0], k[1], k[2], -q) for k, q in ctx.positions.items()]
                new = self._open(ctx)
                if new:
                    orders += new
            elif self.hedge_units(ctx) is not None:
                orders.append(Hedge(self.hedge_units(ctx)))
            return orders
        return self._open(ctx) or []

    def _open(self, ctx: StrategyContext) -> list | None:
        expiry = pick_expiry(ctx.chain, self.dte, self.min_open_dte)
        if expiry is None:
            return None
        legs = self.legs(ctx, expiry)
        return legs or None


class ShortPut(_Roller):
    name = "short_put"

    def __init__(self, delta: float = 0.25, **kw):
        super().__init__(**kw)
        self.delta = delta

    def legs(self, ctx, expiry):
        pick = pick_by_delta(ctx.chain, expiry, "P", self.delta)
        return [Order(expiry, pick[0], "P", -self.size)] if pick else None


class CoveredCall(_Roller):
    name = "covered_call"

    def __init__(self, delta: float = 0.30, shares_per_contract: float = 100.0, **kw):
        super().__init__(**kw)
        self.delta, self.shares = delta, shares_per_contract

    def legs(self, ctx, expiry):
        pick = pick_by_delta(ctx.chain, expiry, "C", self.delta)
        return [Order(expiry, pick[0], "C", -self.size), Hedge(self.size * self.shares)] if pick else None

    def on_date(self, ctx):
        orders = super().on_date(ctx)
        if not ctx.positions and ctx.underlying_units == 0 and not any(isinstance(o, Hedge) for o in orders):
            return orders
        return orders


class ShortStrangle(_Roller):
    name = "short_strangle"

    def __init__(self, delta: float = 0.16, hedge: bool = False, **kw):
        super().__init__(**kw)
        self.delta, self.hedge = delta, hedge

    def legs(self, ctx, expiry):
        p, c = pick_by_delta(ctx.chain, expiry, "P", self.delta), pick_by_delta(ctx.chain, expiry, "C", self.delta)
        if not (p and c):
            return None
        return [Order(expiry, p[0], "P", -self.size), Order(expiry, c[0], "C", -self.size)]

    def hedge_units(self, ctx):
        if not self.hedge:
            return None
        option_delta = ctx.greeks["delta"] - ctx.underlying_units
        return -option_delta


class IronCondor(_Roller):
    name = "iron_condor"

    def __init__(self, short_delta: float = 0.16, wing_delta: float = 0.05, **kw):
        super().__init__(**kw)
        self.short_delta, self.wing_delta = short_delta, wing_delta

    def legs(self, ctx, expiry):
        sp, sc = pick_by_delta(ctx.chain, expiry, "P", self.short_delta), pick_by_delta(ctx.chain, expiry, "C", self.short_delta)
        wp, wc = pick_by_delta(ctx.chain, expiry, "P", self.wing_delta), pick_by_delta(ctx.chain, expiry, "C", self.wing_delta)
        if not (sp and sc and wp and wc) or wp[0] >= sp[0] or wc[0] <= sc[0]:
            return None
        return [Order(expiry, sp[0], "P", -self.size), Order(expiry, sc[0], "C", -self.size), Order(expiry, wp[0], "P", self.size), Order(expiry, wc[0], "C", self.size)]


class DeltaHedgedStraddle(_Roller):
    name = "delta_hedged_straddle"

    def __init__(self, side: int = -1, hedge_threshold: float = 0.0, **kw):
        super().__init__(**kw)
        if side not in (-1, 1):
            raise ValueError("side must be -1 (short volatility) or +1 (long volatility)")
        self.side, self.hedge_threshold = side, hedge_threshold

    def legs(self, ctx, expiry):
        k = pick_atm(ctx.chain, expiry)
        if not (quoted(ctx.chain, expiry, k, "C") and quoted(ctx.chain, expiry, k, "P")):
            return None
        return [Order(expiry, k, "C", self.side * self.size), Order(expiry, k, "P", self.side * self.size)]

    def hedge_units(self, ctx):
        option_delta = ctx.greeks["delta"] - ctx.underlying_units
        target = -option_delta
        if self.hedge_threshold and abs(target - ctx.underlying_units) < self.hedge_threshold * 100.0:
            return ctx.underlying_units
        return target


class RiskReversal(_Roller):
    name = "risk_reversal"

    def __init__(self, delta: float = 0.25, side: int = 1, **kw):
        super().__init__(**kw)
        self.delta, self.side = delta, side

    def legs(self, ctx, expiry):
        c, p = pick_by_delta(ctx.chain, expiry, "C", self.delta), pick_by_delta(ctx.chain, expiry, "P", self.delta)
        if not (c and p):
            return None
        return [Order(expiry, c[0], "C", self.side * self.size), Order(expiry, p[0], "P", -self.side * self.size)]

    def hedge_units(self, ctx):
        return -(ctx.greeks["delta"] - ctx.underlying_units)


class CalendarSpread(_Roller):
    name = "calendar_spread"

    def __init__(self, front_dte: int = 30, back_dte: int = 90, **kw):
        kw.setdefault("dte", front_dte)
        super().__init__(**kw)
        self.front_dte, self.back_dte = front_dte, back_dte

    def legs(self, ctx, expiry):
        back = pick_expiry(ctx.chain, self.back_dte, self.front_dte + 20)
        if back is None or back <= expiry:
            return None
        k = pick_atm(ctx.chain, expiry)
        if not (quoted(ctx.chain, expiry, k, "C") and quoted(ctx.chain, back, k, "C")):
            return None
        return [Order(expiry, k, "C", -self.size), Order(back, k, "C", self.size)]

    def on_date(self, ctx):
        if ctx.positions:
            front = min(k[0] for k in ctx.positions)
            if (front - ctx.date).days <= self.close_dte:
                return [Order(k[0], k[1], k[2], -q) for k, q in ctx.positions.items()]
            return []
        return self._open(ctx) or []


class VRPTimed(OptionStrategy):
    """Trade ``inner`` only while the trailing variance risk premium is positive: the ATM implied volatility of the ~30-day expiry minus the volatility realised over the last
    ``window`` days (both known at the decision date). When the premium turns negative the position is closed and not re-opened."""

    name = "vrp_timed"

    def __init__(self, inner: OptionStrategy, window: int = 21, threshold: float = 0.0, target_dte: int = 30):
        self.inner, self.window, self.threshold, self.target_dte = inner, window, threshold, target_dte
        self.name = f"vrp_timed_{inner.name}"

    def premium(self, ctx: StrategyContext) -> float:
        expiry = pick_expiry(ctx.chain, self.target_dte, 10)
        if expiry is None or len(ctx.history) <= self.window:
            return float("nan")
        g = ctx.chain[(ctx.chain["expiry"] == expiry)]
        atm = g.iloc[(g["strike"] - g["F"]).abs().argsort().iloc[0]]
        realised = float(np.log(ctx.history).diff().iloc[-self.window:].std() * np.sqrt(252))
        return float(atm["iv"]) - realised

    def on_date(self, ctx):
        prem = self.premium(ctx)
        on = np.isfinite(prem) and prem > self.threshold
        if on:
            return self.inner.on_date(ctx)
        if ctx.positions or ctx.underlying_units:
            return [Order(k[0], k[1], k[2], -q) for k, q in ctx.positions.items()] + ([Hedge(0.0)] if ctx.underlying_units else [])
        return []


STRATEGIES = {"short_put": ShortPut, "covered_call": CoveredCall, "short_strangle": ShortStrangle, "iron_condor": IronCondor, "delta_hedged_straddle": DeltaHedgedStraddle,
              "risk_reversal": RiskReversal, "calendar_spread": CalendarSpread}


def make_strategy(name: str, **params) -> OptionStrategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown option strategy '{name}'; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)


def realised_volatility(spot: pd.Series, window: int = 21) -> pd.Series:
    return np.log(spot).diff().rolling(window).std() * np.sqrt(252)
