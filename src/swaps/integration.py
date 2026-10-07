"""Plugging swaps, basis swaps, cross-currency swaps and FX forwards into the engine: marks from curve events and payments from the cashflow model.

Importing this module registers

* **mark models** (``engine.marks.MARK_MODELS``) that value a contract from the point-in-time curve events: the discount and projection curves named in the contract are read with
  the store's ``curve`` lookup (curve events carry ``{tenor in years: zero rate}``, continuously compounded; an event whose ``reference_values`` says ``{"kind": "par"}`` is bootstrapped
  from par swap rates), published floating fixings come from ``fixing`` events, and the returned mark is the present value per unit in the contract currency with bid and ask values
  (mid +- ``HALF_SPREAD_BP`` basis points of annuity) for the cost layer;
* **cashflow models** (``engine.lifecycle``) that give the net payment per currency on every payment date.

A curve older than the engine's ``max_mark_age`` is not used (no mark: the position keeps its last value and a stale mark is counted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.lifecycle import register_cashflows
from ..engine.marks import MarkResult, register_mark_model
from ..instruments.rates import BasisSwap, CrossCurrencyBasisSwap
from .cashflows import payment_amounts
from .contracts import payment_dates
from .curves import CurveSet, DiscountCurve, bootstrap_par_curve
from .pricing import price

HALF_SPREAD_BP = 0.25


def curve_from_event(provider, curve_id: str, ts) -> tuple[DiscountCurve | None, pd.Timedelta | None]:
    """The curve known at ``ts`` for ``curve_id``, as a :class:`DiscountCurve` valued on the curve event's date, and its age."""
    o = provider.store.latest(curve_id, "curve", ts, provider.max_age)
    if o is None:
        return None, None
    values = dict(o.curve_values)
    kind = (o.reference_values or {}).get("kind", "zero") if isinstance(o.reference_values, dict) else "zero"
    obs = pd.Timestamp(o.timestamp).normalize()
    if kind == "par":
        freq = int((o.reference_values or {}).get("freq", 2))
        tenors = sorted(values)
        curve = bootstrap_par_curve(obs, tenors, [values[t] for t in tenors], freq, curve_id)
    else:
        curve = DiscountCurve.from_values(obs, values, name=curve_id)
    return curve, pd.Timestamp(ts) - pd.Timestamp(o.timestamp)


def _curveset(provider, spec, ts):
    ids = {"discount": [spec.discount_curve_id], "projection": [spec.projection_curve_id]}
    if isinstance(spec, BasisSwap):
        ids["projection"] += [spec.projection_curve_id_1, spec.projection_curve_id_2]
    if isinstance(spec, CrossCurrencyBasisSwap):
        ids["discount"].append(spec.discount_curve_id_2)
        ids["projection"].append(spec.projection_curve_id_2)
    cs = CurveSet()
    worst = pd.Timedelta(0)
    for kind in ("discount", "projection"):
        for cid in filter(None, ids[kind]):
            c, age = curve_from_event(provider, cid, ts)
            if c is None:
                if kind == "discount":
                    return None, None
                continue
            (cs.discount if kind == "discount" else cs.projection)[cid] = c
            worst = max(worst, age)
    return cs, worst


def _fixing_fn(provider, ts):
    return lambda index, date: provider.store.fixing(index, date, ts)


def swap_mark(provider, inst, ts) -> MarkResult | None:
    cs, age = _curveset(provider, inst, ts)
    if cs is None:
        return None
    valuation = min(pd.Timestamp(ts), max(c.valuation for c in cs.discount.values()))
    spot = None
    if isinstance(inst, CrossCurrencyBasisSwap):
        snap = provider.store.price(inst.fx_spot_id, ts, order=("quote", "trade", "bar", "mark", "settlement"), max_age=provider.max_age)
        if snap is None:
            return None
        spot = snap.mid
    v = price(inst, cs, valuation, _fixing_fn(provider, ts), spot)
    half = HALF_SPREAD_BP * 1e-4 * v.annuity
    return MarkResult(v.pv, age, "model", v.pv - half, v.pv + half)


def forward_mark(provider, inst, ts) -> MarkResult | None:
    ids = provider.curve_ids
    base_c, a1 = curve_from_event(provider, ids.get(inst.base_currency, f"{inst.base_currency}-DISCOUNT"), ts)
    quote_c, a2 = curve_from_event(provider, ids.get(inst.quote_currency, f"{inst.quote_currency}-DISCOUNT"), ts)
    spot = provider.store.price(inst.underlying_id, ts, order=("quote", "trade", "bar", "mark", "settlement"), max_age=provider.max_age)
    if base_c is None or quote_c is None or spot is None:
        return None
    val = pd.Timestamp(ts).normalize()
    df_b, df_q = base_c.df_date(inst.expiry) / base_c.df_date(val), quote_c.df_date(inst.expiry) / quote_c.df_date(val)
    bid = (spot.bid if np.isfinite(spot.bid) else spot.mid) * df_b - inst.strike * df_q
    ask = (spot.ask if np.isfinite(spot.ask) else spot.mid) * df_b - inst.strike * df_q
    mid = spot.mid * df_b - inst.strike * df_q
    return MarkResult(mid, max(a1, a2, spot.age), "model", bid, ask)


def _dates(inst):
    return payment_dates(inst)


def _amounts(engine, inst, date):
    fixing = lambda index, d: engine.store.fixing(index, d, engine.clock.now)               # noqa: E731
    cs = None
    try:
        cs = _curveset(engine.marks, inst, engine.clock.now)[0]
    except Exception:                                                                       # curves only matter for a missing fixing
        cs = None
    return payment_amounts(inst, date, fixing, cs)


def register() -> None:
    for t in ("irs", "basis_swap", "xccy_basis_swap"):
        register_mark_model(t)(swap_mark)
        register_cashflows(t, _dates, _amounts)
    register_mark_model("fx_forward")(forward_mark)


register()
