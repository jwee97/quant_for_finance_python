"""Swap strategies on the common strategy API: curve trades, carry and roll-down, tenor basis.

Swaps are OTC contracts struck at the market when they are entered: each trade is a NEW instrument (the fixed rate is part of the contract), registered with the engine
(``ctx.register_instrument``) and traded like any other. The strategies size their positions in DV01 (currency per basis point) so that a risk budget, not a notional, is what the
portfolio holds, and keep their open legs in ``ctx.store`` so they hold a swap until their exit rule says otherwise (a swap does not need re-trading every rebalance).

* :class:`CurveTradeStrategy`: a DV01-neutral steepener or flattener between two tenors, entered when the slope's z-score is extreme and exited when it normalises (mean reversion) or
  entered with the move (momentum).
* :class:`CarryRolldownStrategy`: receive the tenor with the best carry plus roll-down per unit of DV01 and pay the worst, DV01 neutral.
* :class:`TenorBasisStrategy`: trade the 3M/6M tenor basis swap against its own history (needs a 6M projection curve).

The curves are read from the point-in-time curve events (``{ccy}-DISCOUNT`` and ``{ccy}-PROJ-3M`` by default), valued on the observation date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.strategy import Schedule, Signal, Strategy, Target
from ..instruments.rates import BasisSwap, make_irs
from .curves import CurveSet, DiscountCurve
from .pricing import par_rate, par_spread, price
from .risk import carry_rolldown, pv01


def curves_now(ctx, currency: str = "USD", discount_id: str | None = None, projection_ids: tuple = ()) -> tuple[CurveSet, pd.Timestamp] | None:
    """The curves known now as a ``CurveSet`` and their valuation date, or None if the discount curve is not available."""
    did = discount_id or f"{currency}-DISCOUNT"
    pid = projection_ids or (f"{currency}-PROJ-3M",)
    c = ctx.data.curve(did)
    if c is None:
        return None
    val = pd.Timestamp(c[0]).normalize()
    cs = CurveSet({did: DiscountCurve.from_values(val, c[1], name=did)})
    for p in pid:
        pc = ctx.data.curve(p)
        if pc is not None:
            cs.projection[p] = DiscountCurve.from_values(pd.Timestamp(pc[0]).normalize(), pc[1], name=p)
    return cs, val


def par_swap_now(ctx, curves: CurveSet, val: pd.Timestamp, tenor_years: float, pay_fixed: bool, currency: str = "USD", calendar: str = "US", notional: float = 1_000_000.0,
                 effective_lag_days: int = 2, discount_id: str | None = None, projection_id: str | None = None):
    """A swap of ``tenor_years`` struck at today's par rate, registered with the engine."""
    effective = get_effective(val, calendar, effective_lag_days)
    maturity = effective + pd.DateOffset(years=int(tenor_years), months=int(round((tenor_years % 1) * 12)))
    probe = make_irs(currency, effective, maturity, 0.0, notional, pay_fixed, calendar=calendar, discount_curve_id=discount_id or f"{currency}-DISCOUNT",
                     projection_curve_id=projection_id or f"{currency}-PROJ-3M")
    k = round(par_rate(probe, curves, val), 7)
    spec = probe.replace(fixed_rate=k, instrument_id=f"IRS-{currency}-{effective:%Y%m%d}-{maturity:%Y%m%d}-{'PAY' if pay_fixed else 'REC'}-{k * 1e4:.2f}bp-N{notional:.0f}")
    return ctx.register_instrument(spec), k


def get_effective(val: pd.Timestamp, calendar: str, lag: int) -> pd.Timestamp:
    from ..instruments.calendars import get_calendar

    return get_calendar(calendar).add_business_days(val, lag)


class _SwapBook(Strategy):
    """Shared machinery: hold a set of swap legs until told to change."""

    schedule = Schedule("weekly", "16:30", 4, "US")

    def __init__(self, currency: str = "USD", dv01_budget: float = 2000.0, calendar: str = "US", notional: float = 1_000_000.0, discount_id: str | None = None,
                 projection_id: str | None = None):
        self.currency, self.dv01_budget, self.calendar, self.notional = currency, dv01_budget, calendar, notional
        self.discount_id, self.projection_id = discount_id, projection_id

    def _legs_open(self, ctx) -> list[tuple[str, float]]:
        return list(ctx.store.get("legs", []))

    def _close_legs(self, ctx) -> list[Target]:
        legs = self._legs_open(ctx)
        ctx.store["legs"] = []
        return [Target(iid, quantity=0.0) for iid, _ in legs]

    def _enter_legs(self, ctx, curves: CurveSet, val, plan: list[tuple[float, bool]]) -> list[Target]:
        """``plan`` is a list of ``(tenor_years, pay_fixed)``; every leg gets the same DV01 budget."""
        targets, legs = [], []
        for tenor, pay in plan:
            inst, _ = par_swap_now(ctx, curves, val, tenor, pay, self.currency, self.calendar, self.notional, discount_id=self.discount_id, projection_id=self.projection_id)
            d = abs(pv01(inst, curves, val))
            qty = self.dv01_budget / d if d > 0 else 0.0
            qty = inst.round_quantity(qty)
            targets.append(Target(inst.instrument_id, quantity=qty))
            legs.append((inst.instrument_id, qty))
        ctx.store["legs"] = legs
        return targets

    def _holding(self, ctx) -> list[Target]:
        return [Target(iid, quantity=q) for iid, q in self._legs_open(ctx)]


class CurveTradeStrategy(_SwapBook):
    name = "curve_trade"

    def __init__(self, short_tenor: float = 2.0, long_tenor: float = 10.0, lookback: int = 60, z_in: float = 1.0, z_out: float = 0.25, mode: str = "mean_reversion", **kwargs):
        super().__init__(**kwargs)
        if mode not in ("mean_reversion", "momentum"):
            raise ValueError("mode must be mean_reversion or momentum")
        self.short_tenor, self.long_tenor, self.lookback, self.z_in, self.z_out, self.mode = short_tenor, long_tenor, lookback, z_in, z_out, mode

    def generate_signals(self, ctx):
        got = curves_now(ctx, self.currency, self.discount_id, (self.projection_id or f"{self.currency}-PROJ-3M",))
        if got is None:
            return []
        curves, val = got
        slopes = ctx.store.setdefault("slopes", [])
        probe = lambda T: make_irs(self.currency, get_effective(val, self.calendar, 2), get_effective(val, self.calendar, 2) + pd.DateOffset(years=int(T)), 0.0, calendar=self.calendar,       # noqa: E731
                                   discount_curve_id=self.discount_id or f"{self.currency}-DISCOUNT", projection_curve_id=self.projection_id or f"{self.currency}-PROJ-3M")
        slope = par_rate(probe(self.long_tenor), curves, val) - par_rate(probe(self.short_tenor), curves, val)
        slopes.append(slope)
        ctx.store["curves"] = (curves, val)
        hist = np.asarray(slopes[-self.lookback:])
        if len(hist) < max(10, self.lookback // 3):
            return []
        z = (slope - hist.mean()) / (hist.std(ddof=1) or np.nan)
        return [Signal("slope", float(z), meta={"slope": slope})]

    def map_to_targets(self, ctx, signals):
        if not signals or "curves" not in ctx.store:
            return self._holding(ctx)
        z = signals[0].value
        curves, val = ctx.store["curves"]
        legs = self._legs_open(ctx)
        state = ctx.store.get("state", 0)
        want = state
        if self.mode == "mean_reversion":
            if state == 0 and abs(z) >= self.z_in:
                want = -1 if z > 0 else 1          # slope too steep: expect flattening (-1 = flattener)
            elif state != 0 and abs(z) <= self.z_out:
                want = 0
        else:
            if state == 0 and abs(z) >= self.z_in:
                want = 1 if z > 0 else -1
            elif state != 0 and abs(z) <= self.z_out:
                want = 0
        if want == state:
            return self._holding(ctx)
        targets = self._close_legs(ctx) if legs else []
        if want != 0:
            # steepener (+1): pay the long tenor, receive the short tenor; flattener is the reverse
            targets += self._enter_legs(ctx, curves, val, [(self.long_tenor, want > 0), (self.short_tenor, want < 0)])
        ctx.store["state"] = want
        return targets


class CarryRolldownStrategy(_SwapBook):
    name = "swap_carry"

    def __init__(self, tenors=(2.0, 5.0, 10.0), horizon_days: int = 91, min_edge_bp: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.tenors, self.horizon_days, self.min_edge_bp = tuple(tenors), horizon_days, min_edge_bp

    def generate_signals(self, ctx):
        got = curves_now(ctx, self.currency, self.discount_id, (self.projection_id or f"{self.currency}-PROJ-3M",))
        if got is None:
            return []
        curves, val = got
        ctx.store["curves"] = (curves, val)
        out = []
        for T in self.tenors:
            eff = get_effective(val, self.calendar, 2)
            spec = make_irs(self.currency, eff, eff + pd.DateOffset(years=int(T)), 0.0, calendar=self.calendar, discount_curve_id=self.discount_id or f"{self.currency}-DISCOUNT",
                            projection_curve_id=self.projection_id or f"{self.currency}-PROJ-3M")
            spec = spec.replace(fixed_rate=par_rate(spec, curves, val))
            cr = carry_rolldown(spec, curves, val, self.horizon_days)
            d = pv01(spec, curves, val)
            out.append(Signal(f"T{T:g}", float(cr["total"] / abs(d)) if d else 0.0, meta={"tenor": T, "pv01": d}))      # payer carry+rolldown per bp of DV01
        return out

    def map_to_targets(self, ctx, signals):
        if len(signals) < 2 or "curves" not in ctx.store:
            return self._holding(ctx)
        curves, val = ctx.store["curves"]
        best_pay = max(signals, key=lambda s: s.value)           # the payer earns most here
        best_rec = min(signals, key=lambda s: s.value)           # the receiver earns most here (payer value most negative)
        edge = best_pay.value - best_rec.value
        want = (best_pay.meta["tenor"], best_rec.meta["tenor"]) if edge > self.min_edge_bp else None
        cur = ctx.store.get("pair")
        if want == cur:
            return self._holding(ctx)
        targets = self._close_legs(ctx) if self._legs_open(ctx) else []
        if want is not None and want[0] != want[1]:
            targets += self._enter_legs(ctx, curves, val, [(want[0], True), (want[1], False)])
        ctx.store["pair"] = want
        return targets


class TenorBasisStrategy(_SwapBook):
    """Trade the 3M/6M basis: when the par spread of the basis swap is well above (below) its history, receive (pay) it. Needs ``{ccy}-PROJ-6M`` curve events."""

    name = "tenor_basis"

    def __init__(self, tenor_years: float = 5.0, lookback: int = 60, z_in: float = 1.0, z_out: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.tenor_years, self.lookback, self.z_in, self.z_out = tenor_years, lookback, z_in, z_out

    def _basis_spec(self, val, pay: bool = True):
        eff = get_effective(val, self.calendar, 2)
        ccy = self.currency
        return BasisSwap(instrument_id=f"BASIS-{ccy}-{eff:%Y%m%d}-{self.tenor_years:g}y", asset_class="swap", instrument_type="basis_swap", currency=ccy, tick_size=1e-6, lot_size=1e-3,
                         calendar=self.calendar, expiry=eff + pd.DateOffset(years=int(self.tenor_years)), settlement_type="cash", effective_date=eff, principal=self.notional,
                         payment_calendar=self.calendar, reset_calendar=self.calendar, discount_curve_id=f"{ccy}-DISCOUNT", projection_curve_id=f"{ccy}-PROJ-3M",
                         index_1=f"{ccy}-3M", tenor_1="3M", frequency_1="3M", index_2=f"{ccy}-6M", tenor_2="6M", frequency_2="6M",
                         projection_curve_id_1=f"{ccy}-PROJ-3M", projection_curve_id_2=f"{ccy}-PROJ-6M")

    def generate_signals(self, ctx):
        got = curves_now(ctx, self.currency, self.discount_id, (f"{self.currency}-PROJ-3M", f"{self.currency}-PROJ-6M"))
        if got is None or f"{self.currency}-PROJ-6M" not in got[0].projection:
            return []
        curves, val = got
        spec = self._basis_spec(val)
        spread = par_spread(spec, curves, val)
        hist = ctx.store.setdefault("spreads", [])
        hist.append(spread)
        ctx.store["curves"] = (curves, val)
        arr = np.asarray(hist[-self.lookback:])
        if len(arr) < max(10, self.lookback // 3):
            return []
        return [Signal("basis", float((spread - arr.mean()) / (arr.std(ddof=1) or np.nan)), meta={"spread": spread})]

    def map_to_targets(self, ctx, signals):
        if not signals or "curves" not in ctx.store:
            return self._holding(ctx)
        z = signals[0].value
        curves, val = ctx.store["curves"]
        state = ctx.store.get("state", 0)
        want = state
        if state == 0 and abs(z) >= self.z_in:
            want = 1 if z > 0 else -1
        elif state != 0 and abs(z) <= self.z_out:
            want = 0
        if want == state:
            return self._holding(ctx)
        targets = self._close_legs(ctx) if self._legs_open(ctx) else []
        if want != 0:
            spec = self._basis_spec(val)
            spec = spec.replace(spread_1=round(par_spread(spec, curves, val), 7), instrument_id=spec.instrument_id + f"-{val:%Y%m%d}")
            ctx.register_instrument(spec)
            d = abs(price(spec, curves, val).annuity) * 1e-4          # PV per basis point of spread
            q = spec.round_quantity(self.dv01_budget / d) if d > 0 else 0.0
            # +1: receive the basis (z high: spread rich, expect it to fall) is the position paying leg 1 (3M + spread) and receiving leg 2; -1 is the reverse
            sign = 1.0 if want < 0 else -1.0
            targets.append(Target(spec.instrument_id, quantity=sign * q))
            ctx.store["legs"] = [(spec.instrument_id, sign * q)]
        ctx.store["state"] = want
        return targets
