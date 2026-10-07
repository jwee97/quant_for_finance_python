"""Interest-rate instruments: fixed/float swaps, tenor basis swaps, cross-currency basis swaps and fixed-coupon bonds, as contract SPECIFICATIONS.

The specifications carry every term a cashflow engine needs: notional and amortisation, fixed and floating legs, floating index and tenor, payment frequency, day-count convention,
fixing and payment lags, business-day convention and calendars, stubs, and the ids of the discounting and projection curves. Schedules, cashflows, pricing and risk are in
``src.swaps``; here a swap is just an immutable contract that the engine can hold a position in.

**Position convention.** One unit of position is one swap exactly as specified (``pay_fixed=True`` is a payer swap: pay fixed, receive floating). A position of ``-1`` is the opposite
swap. The mark is the present value of ONE unit in the contract currency, so a swap struck at the par rate marks near zero and ``cash_style`` is ``otc_mtm`` (value carried, cashflows
paid on their dates) or ``variation_margin`` for a cleared swap (price-alignment and change in value settled in cash daily).

Day counts: ``ACT/360``, ``ACT/365F``, ``30/360`` (US bond basis), ``30E/360``, ``ACT/ACT`` (ISDA). Frequencies: ``1M``, ``3M``, ``6M``, ``12M`` (or ``ZERO`` for a single payment).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Instrument, register_kind

DAY_COUNTS = ("ACT/360", "ACT/365F", "30/360", "30E/360", "ACT/ACT")
FREQUENCIES = ("1M", "3M", "6M", "12M", "ZERO")
NOTIONAL_SCHEDULES = ("bullet", "amortizing", "accreting", "custom")
SETTLEMENT_MODES = ("mtm", "variation_margin")
STUBS = ("short_front", "short_back", "long_front", "long_back")


def months_in(frequency: str) -> int:
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}")
    return 0 if frequency == "ZERO" else int(frequency[:-1])


@dataclass(frozen=True, kw_only=True)
class _SwapTerms(Instrument):
    """Terms common to the swaps: set at construction, validated once."""

    principal: float = 1_000_000.0            # the contract notional in ``currency`` (for the first leg of a cross-currency swap)
    effective_date: pd.Timestamp | None = None
    business_day_convention: str = "modified_following"
    payment_calendar: str = "WEEKDAY"
    reset_calendar: str = "WEEKDAY"
    payment_lag_days: int = 0
    fixing_lag_days: int = 2
    stub: str = "short_front"
    end_of_month: bool = False
    notional_schedule: str = "bullet"
    amortization: tuple = ()                 # notional at the START of each accrual period for `custom`; per-period amount repaid for `amortizing`; per-period increase for `accreting`
    discount_curve_id: str = ""
    projection_curve_id: str = ""
    settlement_mode: str = "mtm"

    TIMESTAMP_FIELDS = ("expiry", "effective_date")

    def __post_init__(self):
        Instrument.__post_init__(self)
        if self.effective_date is not None and not isinstance(self.effective_date, pd.Timestamp):
            object.__setattr__(self, "effective_date", pd.Timestamp(self.effective_date))
        if self.expiry is None or self.effective_date is None or self.expiry <= self.effective_date:
            raise ValueError("a swap needs effective_date < maturity (expiry)")
        if self.principal <= 0:
            raise ValueError("principal (the notional) must be positive")
        if self.notional_schedule not in NOTIONAL_SCHEDULES or self.stub not in STUBS or self.settlement_mode not in SETTLEMENT_MODES:
            raise ValueError("notional_schedule, stub or settlement_mode not recognised")
        if self.notional_schedule in ("amortizing", "accreting", "custom") and not self.amortization:
            raise ValueError(f"notional_schedule '{self.notional_schedule}' needs an amortization schedule")

    @property
    def cash_style(self) -> str:
        return "variation_margin" if self.settlement_mode == "variation_margin" else "otc_mtm"

    def notional(self, price: float = 0.0, quantity: float = 1.0) -> float:
        """The contract notional of ``quantity`` units: independent of the mark (a swap's present value says nothing about the size of the exposure)."""
        return abs(quantity) * self.principal

    @classmethod
    def decode_sequence(cls, key, value):
        return tuple(value)


@register_kind("irs")
@dataclass(frozen=True, kw_only=True)
class InterestRateSwap(_SwapTerms):
    """A fixed-versus-floating swap. ``pay_fixed=True`` pays the fixed rate and receives the floating index."""

    fixed_rate: float = 0.0
    pay_fixed: bool = True
    fixed_frequency: str = "6M"
    float_frequency: str = "3M"
    fixed_daycount: str = "30/360"
    float_daycount: str = "ACT/360"
    float_index: str = "SOFR-3M"
    float_tenor: str = "3M"
    float_spread: float = 0.0
    fixing_in_arrears: bool = False

    def __post_init__(self):
        _SwapTerms.__post_init__(self)
        if self.fixed_daycount not in DAY_COUNTS or self.float_daycount not in DAY_COUNTS:
            raise ValueError(f"day counts must be in {DAY_COUNTS}")
        months_in(self.fixed_frequency), months_in(self.float_frequency), months_in(self.float_tenor)

    @property
    def tenor_years(self) -> float:
        return float((self.expiry - self.effective_date).days / 365.25)

    def reversed(self) -> "InterestRateSwap":
        return self.replace(pay_fixed=not self.pay_fixed)


@register_kind("basis_swap")
@dataclass(frozen=True, kw_only=True)
class BasisSwap(_SwapTerms):
    """Two floating legs in the same currency on different tenors (e.g. 3M against 6M): the holder pays leg 1 (``tenor_1``, plus ``spread_1``) and receives leg 2 (``tenor_2``)."""

    index_1: str = "SOFR-3M"
    tenor_1: str = "3M"
    frequency_1: str = "3M"
    daycount_1: str = "ACT/360"
    spread_1: float = 0.0
    projection_curve_id_1: str = ""
    index_2: str = "SOFR-6M"
    tenor_2: str = "6M"
    frequency_2: str = "6M"
    daycount_2: str = "ACT/360"
    spread_2: float = 0.0
    projection_curve_id_2: str = ""

    def __post_init__(self):
        _SwapTerms.__post_init__(self)
        for f in (self.frequency_1, self.frequency_2, self.tenor_1, self.tenor_2):
            months_in(f)
        if self.daycount_1 not in DAY_COUNTS or self.daycount_2 not in DAY_COUNTS:
            raise ValueError(f"day counts must be in {DAY_COUNTS}")


@register_kind("xccy_basis_swap")
@dataclass(frozen=True, kw_only=True)
class CrossCurrencyBasisSwap(_SwapTerms):
    """Floating against floating in two currencies with notional exchange. The holder pays leg 1 (``currency``, notional ``principal``) and receives leg 2 (``currency_2``, notional
    ``notional_2`` = ``principal`` x the initial spot), with the basis ``spread_2`` added to leg 2. Notionals are exchanged at the start (``initial_exchange``) and returned at maturity."""

    currency_2: str = "EUR"
    notional_2: float = 0.0
    fx_spot_id: str = ""
    initial_exchange: bool = True
    final_exchange: bool = True
    index_1: str = "SOFR-3M"
    index_2: str = "EURIBOR-3M"
    frequency: str = "3M"
    daycount_1: str = "ACT/360"
    daycount_2: str = "ACT/360"
    spread_2: float = 0.0
    discount_curve_id_2: str = ""
    projection_curve_id_2: str = ""

    def __post_init__(self):
        _SwapTerms.__post_init__(self)
        if self.notional_2 <= 0:
            raise ValueError("notional_2 (the second currency's notional) must be positive")
        months_in(self.frequency)


@register_kind("bond")
@dataclass(frozen=True, kw_only=True)
class Bond(Instrument):
    """A fixed-coupon bullet bond quoted as a price per 100 of face (``contract_multiplier = face / 100``, so one unit of position is one bond)."""

    face: float = 100.0
    coupon: float = 0.0
    frequency: str = "6M"
    daycount: str = "30/360"
    issue_date: pd.Timestamp | None = None
    business_day_convention: str = "following"
    discount_curve_id: str = ""

    CASH_STYLE = "full_payment"
    TIMESTAMP_FIELDS = ("expiry", "issue_date")

    def __post_init__(self):
        Instrument.__post_init__(self)
        if self.expiry is None or self.issue_date is None:
            raise ValueError("a bond needs issue_date and maturity (expiry)")
        if self.issue_date is not None and not isinstance(self.issue_date, pd.Timestamp):
            object.__setattr__(self, "issue_date", pd.Timestamp(self.issue_date))
        months_in(self.frequency)
        if self.daycount not in DAY_COUNTS:
            raise ValueError(f"daycount must be in {DAY_COUNTS}")


def make_irs(currency: str, effective, maturity, fixed_rate: float, notional: float = 1_000_000.0, pay_fixed: bool = True, index: str | None = None, calendar: str = "US",
             discount_curve_id: str | None = None, projection_curve_id: str | None = None, **kwargs) -> InterestRateSwap:
    """A vanilla swap with market-standard USD-style terms (semi-annual 30/360 fixed against quarterly ACT/360 floating, modified following, two-day fixing lag)."""
    index = index or f"{currency}-3M"
    side = "PAY" if pay_fixed else "REC"
    iid = f"IRS-{currency}-{pd.Timestamp(effective):%Y%m%d}-{pd.Timestamp(maturity):%Y%m%d}-{side}-{fixed_rate * 1e4:.1f}bp-N{notional:.0f}"
    defaults = dict(instrument_id=iid, asset_class="swap", instrument_type="irs", currency=currency, tick_size=1e-6, lot_size=1e-3, contract_multiplier=1.0, calendar=calendar,
                    expiry=pd.Timestamp(maturity), settlement_type="cash", effective_date=pd.Timestamp(effective), principal=notional, fixed_rate=fixed_rate, pay_fixed=pay_fixed,
                    float_index=index, payment_calendar=calendar, reset_calendar=calendar, price_precision=2,
                    discount_curve_id=discount_curve_id or f"{currency}-DISCOUNT", projection_curve_id=projection_curve_id or f"{currency}-PROJ-3M")
    defaults.update(kwargs)
    return InterestRateSwap(**defaults)
