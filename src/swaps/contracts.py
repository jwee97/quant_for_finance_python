"""Swap contracts: from a contract specification (``instruments.rates``) to the leg tables a cashflow engine needs.

``leg_tables(spec)`` returns the accrual schedule of every leg of the contract:

``InterestRateSwap``    ``fixed`` and ``float``
``BasisSwap``           ``leg1`` and ``leg2`` (two floating legs on different tenors)
``CrossCurrencyBasisSwap``  ``leg1`` and ``leg2`` (floating legs in two currencies); the notional exchanges are separate cashflows (see ``cashflows``)

Each table has one row per accrual period with its dates, accrual fraction and START-OF-PERIOD notional (the notional profile applied: bullet, amortizing, accreting or custom).
``make_irs`` (re-exported from ``instruments.rates``) and the ``make_*`` helpers here construct contracts with market-standard terms; ``par_swap`` constructs a payer or receiver swap
struck at the CURRENT par rate of a curve (so its value is approximately zero at inception, less the cost of trading it).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..instruments.rates import BasisSwap, CrossCurrencyBasisSwap, InterestRateSwap, make_irs
from .schedules import generate_schedule, notional_profile


def _legs(spec, frequency: str, daycount: str, in_arrears: bool = False, adjust_accrual: bool = True) -> pd.DataFrame:
    sched = generate_schedule(spec.effective_date, spec.expiry, frequency, daycount, spec.payment_calendar, spec.business_day_convention, spec.stub, spec.end_of_month,
                              spec.payment_lag_days, spec.payment_calendar, spec.fixing_lag_days, spec.reset_calendar, in_arrears, adjust_accrual)
    sched["notional"] = notional_profile(spec.principal, len(sched), spec.notional_schedule, spec.amortization)
    return sched


def leg_tables(spec) -> dict[str, pd.DataFrame]:
    """The accrual schedule of every leg (cached per contract: the tables are read-only and a contract's schedule never changes)."""
    return _leg_tables(spec)


@lru_cache(maxsize=1024)
def _leg_tables(spec) -> dict[str, pd.DataFrame]:
    if isinstance(spec, InterestRateSwap):
        return {"fixed": _legs(spec, spec.fixed_frequency, spec.fixed_daycount), "float": _legs(spec, spec.float_frequency, spec.float_daycount, spec.fixing_in_arrears)}
    if isinstance(spec, BasisSwap):
        return {"leg1": _legs(spec, spec.frequency_1, spec.daycount_1), "leg2": _legs(spec, spec.frequency_2, spec.daycount_2)}
    if isinstance(spec, CrossCurrencyBasisSwap):
        leg1 = _legs(spec, spec.frequency, spec.daycount_1)
        leg2 = _legs(spec, spec.frequency, spec.daycount_2)
        leg2["notional"] = notional_profile(spec.notional_2, len(leg2), spec.notional_schedule, tuple(a * spec.notional_2 / spec.principal for a in spec.amortization) if spec.amortization else ())
        return {"leg1": leg1, "leg2": leg2}
    raise TypeError(f"{type(spec).__name__} is not a swap contract")


def payment_dates(spec) -> list[pd.Timestamp]:
    """Every date on which this contract pays something (coupons and notional exchanges)."""
    dates = set()
    for table in leg_tables(spec).values():
        dates |= set(pd.to_datetime(table["pay_date"]))
    if isinstance(spec, CrossCurrencyBasisSwap):
        if spec.initial_exchange:
            dates.add(spec.effective_date)
        if spec.final_exchange:
            dates.add(max(dates))
    return sorted(dates)


def par_swap(curves, valuation, effective, maturity, currency: str = "USD", pay_fixed: bool = True, notional: float = 1_000_000.0, calendar: str = "US", **kwargs) -> InterestRateSwap:
    """A vanilla swap struck at the par rate implied by ``curves`` on ``valuation`` (``curves`` is a ``CurveSet``; ids follow the ``make_irs`` defaults unless overridden)."""
    from .pricing import par_rate

    probe = make_irs(currency, effective, maturity, 0.0, notional, pay_fixed, calendar=calendar, **kwargs)
    k = par_rate(probe, curves, valuation)
    return probe.replace(fixed_rate=round(k, 8), instrument_id=probe.instrument_id.replace("0.0bp", f"{k * 1e4:.1f}bp"))
