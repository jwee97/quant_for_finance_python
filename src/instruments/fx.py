"""FX instruments: spot, outright forwards, FX swaps and the quoting conventions that go with them.

**Quote convention.** ``EURUSD = 1.10`` is the number of USD (the *quote* currency) per EUR (the *base* currency); the market convention fixes which currency is the base from a
priority list (``market_pair``). A position of ``+q`` is long ``q`` units of the base currency and short ``q x price`` of the quote currency.

* ``FXSpot``: delivery ``spot_lag_days`` business days after the trade on the JOINT calendar of both currencies. The position is the pair of currency balances
  (``currency_exchange`` cash style), so there is no separate mark-to-market and no cash is "margin".
* ``FXForward``: an OTC contract to exchange ``quantity`` of the base currency for ``quantity x strike`` of the quote currency on ``value_date``. Its mark is the present value in
  the quote currency per unit of base notional, ``(F(t, T) - strike) x DF_quote(t, T)``; at the value date the notionals are exchanged (deliverable) or the difference is paid
  in the quote currency against a fixing (non-deliverable).
* ``FXSwap``: a spot-or-near leg and a far leg in opposite directions on the same notional; ``legs`` returns the two forwards, which are what the engine trades.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Instrument, register_kind
from .calendars import get_calendar

CURRENCY_CALENDARS = {"USD": "US", "EUR": "TARGET", "GBP": "GBLO", "JPY": "WEEKDAY", "CHF": "WEEKDAY", "AUD": "WEEKDAY", "CAD": "WEEKDAY", "NZD": "WEEKDAY", "SEK": "WEEKDAY",
                      "NOK": "WEEKDAY", "BTC": "24x7", "ETH": "24x7", "USDT": "24x7", "USDC": "24x7"}
# market convention: the earlier currency in this list is the base of a pair (EUR/USD, GBP/JPY, AUD/NZD, USD/JPY ...)
BASE_PRIORITY = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "NOK", "SEK", "JPY")


def market_pair(a: str, b: str) -> tuple[str, str]:
    """The (base, quote) order the interbank market quotes the two currencies in."""
    ia = BASE_PRIORITY.index(a) if a in BASE_PRIORITY else len(BASE_PRIORITY)
    ib = BASE_PRIORITY.index(b) if b in BASE_PRIORITY else len(BASE_PRIORITY)
    return (a, b) if ia <= ib else (b, a)


def pair_calendar(base: str, quote: str) -> str:
    names = sorted({CURRENCY_CALENDARS.get(base, "WEEKDAY"), CURRENCY_CALENDARS.get(quote, "WEEKDAY")})
    return "+".join(names) if len(names) > 1 else names[0]


def value_date(trade_date, base: str, quote: str, days: int = 2) -> pd.Timestamp:
    """Spot (``days = 2``) or tom/next value date: business days on the joint calendar of both currencies."""
    return get_calendar(pair_calendar(base, quote)).add_business_days(trade_date, days)


@register_kind("fx_spot")
@dataclass(frozen=True, kw_only=True)
class FXSpot(Instrument):
    base_currency: str
    quote_currency: str
    spot_lag_days: int = 2

    CASH_STYLE = "currency_exchange"

    def __post_init__(self):
        Instrument.__post_init__(self)
        if self.base_currency == self.quote_currency:
            raise ValueError("an FX pair needs two different currencies")

    @property
    def pair(self) -> str:
        return f"{self.base_currency}{self.quote_currency}"

    def settle_date(self, trade_date) -> pd.Timestamp:
        return value_date(trade_date, self.base_currency, self.quote_currency, self.spot_lag_days)


def fx_spot(base: str, quote: str, lot_size: float = 1000.0, **kwargs) -> FXSpot:
    """A spot pair with market conventions: five decimals (three for a JPY quote), a 1,000-unit minimum lot, T+2 delivery, ``base-quote`` ordering as given."""
    jpy = quote == "JPY"
    defaults = dict(instrument_id=f"{base}{quote}", asset_class="fx", instrument_type="fx_spot", currency=quote, tick_size=0.001 if jpy else 0.00001, lot_size=lot_size,
                    calendar=pair_calendar(base, quote), price_precision=3 if jpy else 5, settlement_type="physical", base_currency=base, quote_currency=quote)
    defaults.update(kwargs)
    return FXSpot(**defaults)


@register_kind("fx_forward")
@dataclass(frozen=True, kw_only=True)
class FXForward(Instrument):
    base_currency: str
    quote_currency: str
    strike: float                         # the contracted forward rate (quote per base)
    deliverable: bool = True              # False: non-deliverable, settled in the quote currency against a fixing
    fixing_lag_days: int = 2

    CASH_STYLE = "otc_mtm"
    TIMESTAMP_FIELDS = ("expiry",)

    def __post_init__(self):
        Instrument.__post_init__(self)
        if self.expiry is None:
            raise ValueError("a forward needs a value date (expiry)")
        if self.strike <= 0:
            raise ValueError("strike must be positive")

    @property
    def value_date(self) -> pd.Timestamp:
        return self.expiry

    def notional(self, price: float = 0.0, quantity: float = 1.0) -> float:
        """Notional in the quote currency: base units x the contracted rate (independent of the present value mark)."""
        return abs(quantity) * self.strike

    def pv_per_unit(self, forward_rate: float, discount_factor_quote: float = 1.0) -> float:
        """Present value in the quote currency of ONE unit of base notional: ``(F - K) DF_quote``."""
        return (forward_rate - self.strike) * discount_factor_quote

    def settlement_amounts(self, quantity: float, fixing: float | None = None) -> dict[str, float]:
        """Cash movements at the value date for ``quantity`` units of base notional (signed): deliverable exchanges both notionals, non-deliverable pays ``q (fixing - K)`` in quote."""
        if self.deliverable:
            return {self.base_currency: quantity, self.quote_currency: -quantity * self.strike}
        if fixing is None:
            raise ValueError("a non-deliverable forward needs the fixing")
        return {self.quote_currency: quantity * (fixing - self.strike)}


def fx_forward(base: str, quote: str, value_date_: pd.Timestamp, strike: float, **kwargs) -> FXForward:
    spot = fx_spot(base, quote)
    iid = f"{base}{quote}-FWD-{pd.Timestamp(value_date_):%Y%m%d}@{strike:.{spot.price_precision}f}"
    defaults = dict(instrument_id=iid, asset_class="fx", instrument_type="fx_forward", currency=quote, tick_size=spot.tick_size, lot_size=spot.lot_size, calendar=spot.calendar,
                    price_precision=spot.price_precision + 2, expiry=pd.Timestamp(value_date_), settlement_type="physical", underlying_id=spot.instrument_id,
                    base_currency=base, quote_currency=quote, strike=strike)
    defaults.update(kwargs)
    return FXForward(**defaults)


@register_kind("fx_swap")
@dataclass(frozen=True, kw_only=True)
class FXSwap(Instrument):
    """Buy ``q`` base at ``near_rate`` on ``near_date`` and sell it back at ``far_rate`` on ``far_date`` (or the reverse for a negative quantity)."""

    base_currency: str
    quote_currency: str
    near_date: pd.Timestamp
    near_rate: float
    far_rate: float

    CASH_STYLE = "otc_mtm"
    TIMESTAMP_FIELDS = ("expiry", "near_date")

    def __post_init__(self):
        Instrument.__post_init__(self)
        if self.expiry is None or pd.Timestamp(self.expiry) <= pd.Timestamp(self.near_date):
            raise ValueError("the far date (expiry) must be after the near date")

    @property
    def far_date(self) -> pd.Timestamp:
        return self.expiry

    def swap_points(self) -> float:
        return self.far_rate - self.near_rate

    def legs(self, quantity: float = 1.0) -> list[tuple[FXForward, float]]:
        """The two forwards (and signed quantities) that make the swap: ``+q`` near, ``-q`` far for a positive quantity."""
        near = fx_forward(self.base_currency, self.quote_currency, self.near_date, self.near_rate)
        far = fx_forward(self.base_currency, self.quote_currency, self.expiry, self.far_rate)
        return [(near, quantity), (far, -quantity)]


def fx_swap(base: str, quote: str, near_date, far_date, near_rate: float, far_rate: float, **kwargs) -> FXSwap:
    spot = fx_spot(base, quote)
    iid = f"{base}{quote}-SWAP-{pd.Timestamp(near_date):%Y%m%d}-{pd.Timestamp(far_date):%Y%m%d}"
    defaults = dict(instrument_id=iid, asset_class="fx", instrument_type="fx_swap", currency=quote, tick_size=spot.tick_size, lot_size=spot.lot_size, calendar=spot.calendar,
                    price_precision=spot.price_precision + 2, expiry=pd.Timestamp(far_date), settlement_type="physical", underlying_id=spot.instrument_id, base_currency=base,
                    quote_currency=quote, near_date=pd.Timestamp(near_date), near_rate=near_rate, far_rate=far_rate)
    defaults.update(kwargs)
    return FXSwap(**defaults)
