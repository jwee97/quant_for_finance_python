"""Interest-rate swaps as first-class instruments: schedules, curves, cashflows, pricing, risk, strategies and the engine integration. See each module's docstring."""

from . import integration  # noqa: F401  registers the mark and cashflow models with the engine
from .cashflows import cashflow_table, payment_amounts, projected_fixings
from .contracts import leg_tables, par_swap, payment_dates
from .curves import CurveSet, DiscountCurve, bootstrap_par_curve, tenor_basis_curve
from .pricing import SwapValuation, bid_ask, explain_pnl, par_rate, par_spread, price, pv
from .risk import butterfly_weights, carry_rolldown, convexity_pv, dv01_neutral_ratio, key_rate_pv01, parallel, pv01, scenario, steepener, twist
from .schedules import generate_schedule, notional_profile, year_fraction

__all__ = [n for n in dir() if not n.startswith("_")]
