"""Cashflow generation: every payment of a swap, projected or fixed, from the holder's point of view for ONE unit of position.

``cashflow_table(spec, curves, valuation, fixing_fn, spot)`` returns one row per payment with the leg, currency, payment date, fixing date, period, notional, accrual fraction, the rate
used (the contractual fixed rate, a fixing already published, or the forward rate projected from the projection curve), the signed amount, whether the rate is known, the discount
factor and the present value (only for payments after ``valuation``). Signs: a payer swap pays the fixed leg and receives the floating leg, a basis swap or cross-currency swap pays
leg 1 and receives leg 2 (see ``instruments.rates``).

**Fixings.** A floating coupon whose fixing date is on or before the valuation date uses the PUBLISHED fixing (``fixing_fn(index, date)``); if none is available the row is flagged
``missing_fixing`` and the projected forward is used, so a gap in the fixings history is visible rather than silently filled. Coupons with a later fixing are projected from the curve.

**Notional exchanges.** A cross-currency swap exchanges notionals at the start and the end (opposite directions); they appear as ``exchange`` rows.

``payment_amounts`` gives the net amount per currency paid on one date, which is what the engine's lifecycle posts.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..instruments.rates import BasisSwap, CrossCurrencyBasisSwap, InterestRateSwap
from .contracts import leg_tables
from .curves import CurveSet, DiscountCurve

COLUMNS = ["leg", "currency", "pay_date", "fixing_date", "start", "end", "notional", "accrual", "rate", "amount", "known", "missing_fixing", "index", "base_rate", "df", "pv"]


def _proj(curves: CurveSet | None, curve_id: str, fallback: str):
    if curves is None:
        return None
    return curves.get_projection(curve_id, fallback)


def _float_rows(spec, leg: str, table: pd.DataFrame, index: str, spread: float, sign: float, ccy: str, proj: DiscountCurve | None, disc: DiscountCurve | None, valuation,
                fixing_fn: Callable | None) -> list[dict]:
    rows = []
    val = pd.Timestamp(valuation)
    for r in table.itertuples(index=False):
        fixed_by_now = r.fixing_date <= val
        rate, known, missing = np.nan, False, False
        if fixed_by_now and fixing_fn is not None:
            f = fixing_fn(index, r.fixing_date)
            if f is not None:
                rate, known = float(f), True
        if not known:
            missing = bool(fixed_by_now)
            if proj is None:
                if missing:
                    rate = np.nan
                else:
                    raise ValueError(f"a projection curve is needed to project the {leg} coupon fixing on {r.fixing_date.date()}")
            else:
                rate = proj.forward_rate_dates(r.adj_start, r.adj_end, r.accrual)
        total = rate + spread
        amount = sign * r.notional * total * r.accrual
        rows.append({"leg": leg, "currency": ccy, "pay_date": r.pay_date, "fixing_date": r.fixing_date, "start": r.adj_start, "end": r.adj_end, "notional": r.notional, "accrual": r.accrual,
                     "rate": total, "amount": amount, "known": known, "missing_fixing": missing, "index": index, "base_rate": rate})
    return rows


def cashflow_table(spec, curves: CurveSet | None, valuation, fixing_fn: Callable | None = None, spot: float | None = None) -> pd.DataFrame:
    val = pd.Timestamp(valuation)
    legs = leg_tables(spec)
    rows: list[dict] = []
    if isinstance(spec, InterestRateSwap):
        s_fixed = -1.0 if spec.pay_fixed else 1.0
        for r in legs["fixed"].itertuples(index=False):
            rows.append({"leg": "fixed", "currency": spec.currency, "pay_date": r.pay_date, "fixing_date": pd.NaT, "start": r.adj_start, "end": r.adj_end, "notional": r.notional,
                         "accrual": r.accrual, "rate": spec.fixed_rate, "amount": s_fixed * r.notional * spec.fixed_rate * r.accrual, "known": True, "missing_fixing": False,
                         "index": "", "base_rate": spec.fixed_rate})
        proj = _proj(curves, spec.projection_curve_id, spec.discount_curve_id)
        rows += _float_rows(spec, "float", legs["float"], spec.float_index, spec.float_spread, -s_fixed, spec.currency, proj, None, val, fixing_fn)
        disc = {spec.currency: curves.get_discount(spec.discount_curve_id) if curves else None}
    elif isinstance(spec, BasisSwap):
        p1 = _proj(curves, spec.projection_curve_id_1 or spec.projection_curve_id, spec.discount_curve_id)
        p2 = _proj(curves, spec.projection_curve_id_2 or spec.projection_curve_id, spec.discount_curve_id)
        rows += _float_rows(spec, "leg1", legs["leg1"], spec.index_1, spec.spread_1, -1.0, spec.currency, p1, None, val, fixing_fn)
        rows += _float_rows(spec, "leg2", legs["leg2"], spec.index_2, spec.spread_2, 1.0, spec.currency, p2, None, val, fixing_fn)
        disc = {spec.currency: curves.get_discount(spec.discount_curve_id) if curves else None}
    elif isinstance(spec, CrossCurrencyBasisSwap):
        p1 = _proj(curves, spec.projection_curve_id, spec.discount_curve_id)
        p2 = _proj(curves, spec.projection_curve_id_2, spec.discount_curve_id_2)
        rows += _float_rows(spec, "leg1", legs["leg1"], spec.index_1, 0.0, -1.0, spec.currency, p1, None, val, fixing_fn)
        rows += _float_rows(spec, "leg2", legs["leg2"], spec.index_2, spec.spread_2, 1.0, spec.currency_2, p2, None, val, fixing_fn)
        if spec.initial_exchange:
            rows.append({"leg": "exchange", "currency": spec.currency, "pay_date": spec.effective_date, "fixing_date": pd.NaT, "start": spec.effective_date, "end": spec.effective_date,
                         "notional": spec.principal, "accrual": 0.0, "rate": np.nan, "amount": spec.principal, "known": True, "missing_fixing": False, "index": "", "base_rate": np.nan})
            rows.append({"leg": "exchange", "currency": spec.currency_2, "pay_date": spec.effective_date, "fixing_date": pd.NaT, "start": spec.effective_date, "end": spec.effective_date,
                         "notional": spec.notional_2, "accrual": 0.0, "rate": np.nan, "amount": -spec.notional_2, "known": True, "missing_fixing": False, "index": "", "base_rate": np.nan})
        if spec.final_exchange:
            last = max(legs["leg1"]["pay_date"].max(), legs["leg2"]["pay_date"].max())
            rows.append({"leg": "exchange", "currency": spec.currency, "pay_date": last, "fixing_date": pd.NaT, "start": last, "end": last, "notional": spec.principal, "accrual": 0.0,
                         "rate": np.nan, "amount": -spec.principal, "known": True, "missing_fixing": False, "index": "", "base_rate": np.nan})
            rows.append({"leg": "exchange", "currency": spec.currency_2, "pay_date": last, "fixing_date": pd.NaT, "start": last, "end": last, "notional": spec.notional_2, "accrual": 0.0,
                         "rate": np.nan, "amount": spec.notional_2, "known": True, "missing_fixing": False, "index": "", "base_rate": np.nan})
        disc = {spec.currency: curves.get_discount(spec.discount_curve_id) if curves else None, spec.currency_2: curves.get_discount(spec.discount_curve_id_2) if curves else None}
    else:
        raise TypeError(f"{type(spec).__name__} has no cashflow model")
    df = pd.DataFrame(rows, columns=COLUMNS[:-2])
    dfs, pvs = [], []
    for r in df.itertuples(index=False):
        curve = disc.get(r.currency) if curves is not None else None
        d = float(curve.df_date(r.pay_date)) if curve is not None else np.nan
        dfs.append(d)
        pvs.append(r.amount * d if r.pay_date > val and np.isfinite(d) and np.isfinite(r.amount) else 0.0)
    df["df"], df["pv"] = dfs, pvs
    return df.sort_values(["pay_date", "leg"]).reset_index(drop=True)


def payment_amounts(spec, date, fixing_fn: Callable | None, curves: CurveSet | None = None) -> dict[str, float]:
    """The net amount per currency, per unit of position, paid on ``date`` (the coupons of every leg that pays then, netted)."""
    date = pd.Timestamp(date)
    table = cashflow_table(spec, curves, date, fixing_fn)
    due = table[table["pay_date"] == date]
    if due["missing_fixing"].any() and due["amount"].isna().any():
        raise ValueError(f"missing floating-rate fixing for a coupon paid on {date.date()} and no curve to project it")
    return {ccy: float(g["amount"].sum()) for ccy, g in due.groupby("currency")}


def projected_fixings(table: pd.DataFrame, fixing_fn: Callable | None = None) -> Callable:
    """A fixing function that answers with the RATES PROJECTED in ``table`` (for periods whose fixing date is past in a later valuation but which the original curve forecast), falling back
    to ``fixing_fn``. Used to carry a swap forward in time: a coupon that has already reset keeps the rate the original curve implied for it."""
    lookup = {(r.index, pd.Timestamp(r.fixing_date)): r.base_rate for r in table.itertuples(index=False) if r.index}

    def fn(index, date):
        if fixing_fn is not None:
            f = fixing_fn(index, date)
            if f is not None:
                return f
        return lookup.get((index, pd.Timestamp(date)))
    return fn
