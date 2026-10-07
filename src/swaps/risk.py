"""Swap risk: PV01 and key-rate PV01, convexity, carry and roll-down, curve-trade hedge ratios and curve scenarios.

All sensitivities are per unit of position in the contract currency, from FULL REPRICING under bumped curves (not an analytic approximation), so they include the discount-curve and
projection-curve effects together and separately:

* ``pv01``: the change in value for a one basis point parallel rise in all rates (``discount=False`` or ``projection=False`` isolate one set: a swap's PV01 is mostly projection
  risk on the floating leg offset by the fixed leg's discounting, which is why the two matter separately in a multi-curve world);
* ``key_rate_pv01``: the same for a triangular bump centred on each tenor node, so the exposure is shown by maturity bucket and the buckets sum to (about) the parallel PV01;
* ``convexity_pv``: the second difference ``PV(+b) + PV(-b) - 2 PV`` per bp squared (the curvature of value in rates);
* ``carry_rolldown``: over a horizon, the value change if nothing moves in the market, split into CARRY (the curve realises its forward rates; with the cashflows received in between)
  and ROLL-DOWN (the curve keeps its shape as time passes: a steep curve rolls to lower rates for the same remaining maturity);
* ``dv01_neutral_ratio`` and ``butterfly_weights``: hedge ratios for curve trades (steepeners, flatteners, butterflies);
* ``scenario`` and the shock functions ``parallel``, ``steepener``, ``twist``: the P&L of a position under a curve shock.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .curves import CurveSet
from .cashflows import projected_fixings
from .pricing import price


def pv01(spec, curves: CurveSet, valuation, fixing_fn: Callable | None = None, spot: float | None = None, discount: bool = True, projection: bool = True, bp: float = 1.0) -> float:
    """Change in PV per +1bp (central difference)."""
    up = price(spec, curves.bumped(bp, discount, projection), valuation, fixing_fn, spot).pv
    dn = price(spec, curves.bumped(-bp, discount, projection), valuation, fixing_fn, spot).pv
    return float((up - dn) / (2.0 * bp))


def key_rate_pv01(spec, curves: CurveSet, valuation, tenors=None, fixing_fn: Callable | None = None, spot: float | None = None) -> pd.Series:
    """PV01 by tenor bucket (a triangular bump on every curve at each node)."""
    nodes = tenors if tenors is not None else sorted({t for c in list(curves.discount.values()) + list(curves.projection.values()) for t in c.tenors})
    out = {}
    for t in nodes:
        up = price(spec, curves.bumped(1.0, tenor=t), valuation, fixing_fn, spot).pv
        dn = price(spec, curves.bumped(-1.0, tenor=t), valuation, fixing_fn, spot).pv
        out[t] = (up - dn) / 2.0
    return pd.Series(out, name="key_rate_pv01")


def convexity_pv(spec, curves: CurveSet, valuation, fixing_fn: Callable | None = None, spot: float | None = None, bp: float = 25.0) -> float:
    """``(PV(+bp) + PV(-bp) - 2 PV) / bp^2``: curvature per bp squared."""
    p0 = price(spec, curves, valuation, fixing_fn, spot).pv
    up = price(spec, curves.bumped(bp), valuation, fixing_fn, spot).pv
    dn = price(spec, curves.bumped(-bp), valuation, fixing_fn, spot).pv
    return float((up + dn - 2.0 * p0) / bp ** 2)


def carry_rolldown(spec, curves: CurveSet, valuation, horizon_days: int = 30, fixing_fn: Callable | None = None, spot: float | None = None) -> dict:
    """What the position earns over ``horizon_days`` if the market does not move.

    * ``carry``: the net coupon accrued over the horizon at the CURRENT rates (fixed accrual against the floating rate projected for the running period), the number a trader quotes
      as "carry";
    * ``rolldown``: the rest of the change in value when the curve keeps its shape in tenor space as time passes (``total - carry``): a swap on a steep curve re-prices to a lower
      rate for the same remaining maturity;
    * ``total``: ``PV(end, curve shape unchanged) + cashflows paid - PV(start)``;
    * ``forward_pnl``: for reference, the change in value if instead the FORWARD rates are realised (about the financing cost of the position's value, zero for a swap at par).
    """
    val = pd.Timestamp(valuation)
    end = val + pd.Timedelta(days=horizon_days)
    years = horizon_days / 365.0
    v0 = price(spec, curves, val, fixing_fn, spot)
    table = v0.table
    rate_of = {spec.currency: 1.0}
    if hasattr(spec, "currency_2"):
        rate_of[spec.currency_2] = spot if spot is not None else 1.0
    paid = table[(table["pay_date"] > val) & (table["pay_date"] <= end) & (table["leg"] != "exchange")]
    cash = float(sum(r.amount * rate_of[r.currency] for r in table[(table["pay_date"] > val) & (table["pay_date"] <= end)].itertuples(index=False)))
    carry = 0.0
    for r in table[table["leg"] != "exchange"].itertuples(index=False):
        s, e = pd.Timestamp(r.start), pd.Timestamp(r.end)
        if e <= val or s >= end or e <= s:
            continue
        overlap = (min(e, end) - max(s, val)).days
        carry += r.amount * rate_of[r.currency] * overlap / (e - s).days
    fx = projected_fixings(table, fixing_fn)
    forward_curves = CurveSet({k: c.rolled(years) for k, c in curves.discount.items()}, {k: c.rolled(years) for k, c in curves.projection.items()})
    v_fwd = price(spec, forward_curves, end, fx, spot).pv
    static = CurveSet({k: type(c)(end, c.tenors, c.zeros, c.interpolation, c.name) for k, c in curves.discount.items()},
                      {k: type(c)(end, c.tenors, c.zeros, c.interpolation, c.name) for k, c in curves.projection.items()})
    v_static = price(spec, static, end, fx, spot).pv
    total = v_static + cash - v0.pv
    return {"carry": float(carry), "rolldown": float(total - carry), "total": float(total), "forward_pnl": float(v_fwd + cash - v0.pv), "cashflows": cash, "pv": v0.pv, "horizon_days": horizon_days}


def dv01_neutral_ratio(dv01_a: float, dv01_b: float) -> float:
    """Units of B per unit of A that make ``A + ratio B`` insensitive to a parallel shift: ``-dv01_a / dv01_b``."""
    if dv01_b == 0:
        raise ValueError("cannot hedge with an instrument that has no DV01")
    return float(-dv01_a / dv01_b)


def butterfly_weights(dv01_front: float, dv01_belly: float, dv01_back: float, wings: str = "equal_risk") -> tuple[float, float, float]:
    """Weights ``(front, belly, back)`` for a butterfly that is DV01 neutral: the belly is traded against the wings. ``equal_risk`` splits the belly's DV01 half to each wing; the
    result is for one unit of the belly (weight -1 for a long-belly butterfly is ``(+w1, -1, +w3)``)."""
    if wings != "equal_risk":
        raise ValueError("only the equal_risk split is implemented")
    return (float(0.5 * dv01_belly / dv01_front), -1.0, float(0.5 * dv01_belly / dv01_back))


def parallel(bp: float) -> Callable:
    return lambda t: np.full_like(np.asarray(t, float), bp * 1e-4)


def steepener(bp: float, pivot: float = 5.0) -> Callable:
    """Front rates fall and long rates rise around ``pivot`` (a steepening of ``bp`` basis points between 0 and 2x pivot, linear in tenor, capped)."""
    return lambda t: np.clip((np.asarray(t, float) - pivot) / pivot, -1.0, 1.0) * bp * 1e-4


def twist(bp: float, pivot: float = 5.0) -> Callable:
    """Belly moves against the wings (a hump around ``pivot``)."""
    return lambda t: (1.0 - np.clip(np.abs(np.asarray(t, float) - pivot) / pivot, 0.0, 1.0)) * bp * 1e-4


def scenario(spec, curves: CurveSet, valuation, shock: Callable, fixing_fn: Callable | None = None, spot: float | None = None) -> float:
    """P&L per unit of position when the curves move by ``shock(tenor)``."""
    return float(price(spec, curves.shifted(shock), valuation, fixing_fn, spot).pv - price(spec, curves, valuation, fixing_fn, spot).pv)
