"""Swap pricing: present value, par rates, annuities, bid/ask values and mark-to-market P&L explanation.

``price(spec, curves, valuation, fixing_fn, spot)`` discounts every future cashflow (``cashflows.cashflow_table``) and returns a :class:`SwapValuation`: the dirty present value of ONE
unit of position in the contract currency (cross-currency swaps convert the second currency at ``spot``, in units of the first currency per unit of the second), the present value
by leg, and the fixed-leg annuity ``sum N tau DF`` over future periods.

* ``par_rate(spec, ...)``: the fixed rate that makes the swap worth zero (floating leg PV over annuity); for basis swaps ``par_spread`` is the leg-1 spread that does the same.
* ``bid_ask(spec, ..., half_spread_bp)``: the present values at which one could trade, a fixed number of basis points of annuity either side of the mid, which is the transaction
  cost of the contract in value terms (and what the engine's mark model hands to the cost layer).
* ``explain_pnl``: what a mark-to-market change was made of: the cashflows received, the passage of time at an unchanged curve (carry and roll-down) and the curve move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ..instruments.rates import BasisSwap, CrossCurrencyBasisSwap, InterestRateSwap
from .cashflows import cashflow_table, projected_fixings
from .curves import CurveSet


@dataclass
class SwapValuation:
    pv: float
    pv_by_leg: dict
    annuity: float
    table: pd.DataFrame = field(repr=False, default=None)


def price(spec, curves: CurveSet, valuation, fixing_fn: Callable | None = None, spot: float | None = None) -> SwapValuation:
    val = pd.Timestamp(valuation)
    table = cashflow_table(spec, curves, val, fixing_fn, spot)
    future = table[table["pay_date"] > val]
    rate_of = {spec.currency: 1.0}
    if isinstance(spec, CrossCurrencyBasisSwap):
        if spot is None:
            raise ValueError("a cross-currency swap needs the spot exchange rate (units of the first currency per unit of the second)")
        rate_of[spec.currency_2] = spot
    pv_by_leg: dict[str, float] = {}
    total = 0.0
    for r in future.itertuples(index=False):
        v = r.pv * rate_of[r.currency]
        pv_by_leg[r.leg] = pv_by_leg.get(r.leg, 0.0) + v
        total += v
    leg = future[future["leg"] == ("fixed" if isinstance(spec, InterestRateSwap) else "leg1")]
    annuity = float((leg["notional"] * leg["accrual"] * leg["df"]).sum() * rate_of[spec.currency])
    return SwapValuation(float(total), pv_by_leg, annuity, table)


def pv(spec, curves: CurveSet, valuation, fixing_fn: Callable | None = None, spot: float | None = None) -> float:
    return price(spec, curves, valuation, fixing_fn, spot).pv


def par_rate(spec: InterestRateSwap, curves: CurveSet, valuation, fixing_fn: Callable | None = None) -> float:
    """The fixed rate that sets the PV of the swap to zero (``PV_float / annuity``)."""
    v = price(spec, curves, valuation, fixing_fn)
    return float(abs(v.pv_by_leg.get("float", 0.0)) / v.annuity) if v.annuity else float("nan")


def par_spread(spec, curves: CurveSet, valuation, fixing_fn: Callable | None = None, spot: float | None = None) -> float:
    """For a basis or cross-currency swap: the spread on the second-leg (cross-currency) or first-leg (basis) that sets PV to zero, in decimal rate units."""
    if isinstance(spec, BasisSwap):
        base, bumped = price(spec, curves, valuation, fixing_fn).pv, price(spec.replace(spread_1=spec.spread_1 + 1e-4), curves, valuation, fixing_fn).pv
    else:
        base, bumped = price(spec, curves, valuation, fixing_fn, spot).pv, price(spec.replace(spread_2=spec.spread_2 + 1e-4), curves, valuation, fixing_fn, spot).pv
    slope = (bumped - base) / 1e-4
    cur = spec.spread_1 if isinstance(spec, BasisSwap) else spec.spread_2
    return float(cur - base / slope) if slope else float("nan")


def bid_ask(spec, curves: CurveSet, valuation, half_spread_bp: float = 0.25, fixing_fn: Callable | None = None, spot: float | None = None) -> tuple[float, float, float]:
    """``(bid, mid, ask)`` present values: the mid plus or minus ``half_spread_bp`` basis points of annuity on the notional (the cost of crossing the market)."""
    v = price(spec, curves, valuation, fixing_fn, spot)
    half = half_spread_bp * 1e-4 * v.annuity
    return v.pv - half, v.pv, v.pv + half


def explain_pnl(spec, curves_start: CurveSet, curves_end: CurveSet, start, end, fixing_fn: Callable | None = None, spot_start: float | None = None, spot_end: float | None = None) -> dict:
    """Decompose the change in value between two dates: cashflows paid in between, carry and roll-down (time passing with the START curve held fixed in tenor space) and the curve move.

    ``total = pv_end + cashflows_paid - pv_start = (time effect) + (curve effect)`` where the time effect values the swap at the end date on the start curves (rolled forward in
    calendar time but with an unchanged curve shape) and the curve effect is the remainder."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    v0 = price(spec, curves_start, start, fixing_fn, spot_start).pv
    t0 = cashflow_table(spec, curves_start, start, fixing_fn, spot_start)
    paid = t0[(t0["pay_date"] > start) & (t0["pay_date"] <= end)]
    rate_of = {spec.currency: 1.0}
    if isinstance(spec, CrossCurrencyBasisSwap):
        rate_of[spec.currency_2] = spot_end or spot_start
    cash_paid = float(sum(r.amount * rate_of[r.currency] for r in paid.itertuples(index=False)))
    shape_fixed = CurveSet({k: type(c)(end, c.tenors, c.zeros, c.interpolation, c.name) for k, c in curves_start.discount.items()},
                           {k: type(c)(end, c.tenors, c.zeros, c.interpolation, c.name) for k, c in curves_start.projection.items()})
    v_time = price(spec, shape_fixed, end, projected_fixings(t0, fixing_fn), spot_start).pv
    v1 = price(spec, curves_end, end, projected_fixings(t0, fixing_fn), spot_end).pv
    total = v1 + cash_paid - v0
    time_effect = v_time + cash_paid - v0
    return {"start_pv": v0, "end_pv": v1, "cashflows_paid": cash_paid, "total": total, "time_effect": time_effect, "curve_effect": total - time_effect}
