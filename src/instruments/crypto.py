"""Crypto instruments: spot, perpetual swaps, dated futures and the venue conventions that make each exchange different.

* ``CryptoSpot`` (``currency_exchange`` style): the position is the coin balance; the quote balance pays for it.
* ``CryptoPerp``: a perpetual with a funding payment every ``funding_interval_hours`` (``funding = -position notional at the MARK price x funding rate``: longs pay when the rate is positive). ``contract_type`` is ``linear`` (margined and settled in the quote
  currency, USDT-style, ``pnl = q x size x (exit - entry)``) or ``inverse`` (margined and settled in the coin, ``pnl = q x size x (1/entry - 1/exit)`` in coin). Margin is isolated or cross; the venue
  fixes the maximum leverage and the maintenance margin rate, which together give the liquidation price (``liquidation_price``).
* ``CryptoFuture``: a dated future (same linear or inverse P&L) that converges to the index at expiry and settles in cash.

Venue conventions are a small table (``VENUES``); ``crypto_perp`` and friends read it, so that adding an exchange is a dictionary entry. Identifiers follow ``VENUE:BASE-QUOTE:TYPE`` and
``canonical_symbol`` translates the exchanges' native symbols (``BTCUSDT``, ``BTC-PERPETUAL``, ``BTC-USDT-SWAP``) into them so that cross-venue spreads can be expressed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .base import Instrument, register_kind

VENUES = {
    "BINANCE": {"perp_type": "linear", "quote": "USDT", "funding_hours": 8, "tick": 0.1, "lot": 0.001, "max_leverage": 125.0, "mmr": 0.004, "contract_size": 1.0, "taker_bps": 4.0},
    "BYBIT": {"perp_type": "linear", "quote": "USDT", "funding_hours": 8, "tick": 0.1, "lot": 0.001, "max_leverage": 100.0, "mmr": 0.005, "contract_size": 1.0, "taker_bps": 5.5},
    "OKX": {"perp_type": "linear", "quote": "USDT", "funding_hours": 8, "tick": 0.1, "lot": 0.01, "max_leverage": 100.0, "mmr": 0.004, "contract_size": 0.01, "taker_bps": 5.0},
    "DERIBIT": {"perp_type": "inverse", "quote": "USD", "funding_hours": 8, "tick": 0.5, "lot": 10.0, "max_leverage": 50.0, "mmr": 0.005, "contract_size": 1.0, "taker_bps": 5.0},
}

_SYMBOL_PATTERNS = [
    (re.compile(r"^(?P<base>[A-Z0-9]+?)(?P<quote>USDT|USDC|BUSD|USD)$"), "spot_or_perp"),                 # BTCUSDT
    (re.compile(r"^(?P<base>[A-Z0-9]+)-PERPETUAL$"), "deribit_perp"),                                       # BTC-PERPETUAL
    (re.compile(r"^(?P<base>[A-Z0-9]+)-(?P<quote>[A-Z]+)-SWAP$"), "okx_perp"),                              # BTC-USDT-SWAP
    (re.compile(r"^(?P<base>[A-Z0-9]+)-(?P<quote>[A-Z]+)$"), "okx_spot"),                                   # BTC-USDT
]


def canonical_symbol(venue: str, native: str, kind: str | None = None) -> str:
    """``VENUE:BASE-QUOTE:TYPE`` from a native symbol; ``kind`` ("spot" or "perp") disambiguates symbols that are the same on both (``BTCUSDT``)."""
    venue = venue.upper()
    for pattern, tag in _SYMBOL_PATTERNS:
        m = pattern.match(native.upper())
        if not m:
            continue
        base = m.group("base")
        quote = m.groupdict().get("quote") or VENUES.get(venue, {}).get("quote", "USD")
        typ = {"deribit_perp": "PERP", "okx_perp": "PERP", "okx_spot": "SPOT"}.get(tag, (kind or "spot").upper())
        return f"{venue}:{base}-{quote}:{typ}"
    raise ValueError(f"cannot parse symbol '{native}' for {venue}")


@register_kind("crypto_spot")
@dataclass(frozen=True, kw_only=True)
class CryptoSpot(Instrument):
    base_currency: str
    quote_currency: str
    venue: str = ""

    CASH_STYLE = "currency_exchange"

    @property
    def pair(self) -> str:
        return f"{self.base_currency}{self.quote_currency}"


@register_kind("crypto_perp")
@dataclass(frozen=True, kw_only=True)
class CryptoPerp(Instrument):
    base_currency: str
    quote_currency: str
    venue: str = ""
    contract_type: str = "linear"                 # linear: settled in the quote currency; inverse: settled in the base coin
    funding_interval_hours: int = 8
    max_leverage: float = 20.0
    maintenance_margin_rate: float = 0.005
    margin_mode: str = "isolated"                 # isolated or cross
    funding_rate_cap: float = 0.0075              # per interval (0.75%)

    CASH_STYLE = "variation_margin"

    def __post_init__(self):
        Instrument.__post_init__(self)
        if self.contract_type not in ("linear", "inverse"):
            raise ValueError("contract_type must be linear or inverse")
        if self.margin_mode not in ("isolated", "cross"):
            raise ValueError("margin_mode must be isolated or cross")
        if self.funding_interval_hours <= 0 or 24 % self.funding_interval_hours:
            raise ValueError("funding_interval_hours must divide 24")
        if self.max_leverage < 1:
            raise ValueError("max_leverage must be >= 1")

    @property
    def inverse(self) -> bool:
        return self.contract_type == "inverse"

    def notional(self, price: float, quantity: float = 1.0) -> float:
        """Linear: ``|q| size price`` in quote; inverse: ``|q| size`` in USD (each contract is a fixed dollar amount)."""
        return abs(quantity) * self.contract_multiplier * (1.0 if self.inverse else price)

    def settlement_notional(self, price: float, quantity: float = 1.0) -> float:
        return abs(quantity) * self.contract_multiplier / price if self.inverse else self.notional(price, quantity)

    def pnl(self, entry: float, exit: float, quantity: float) -> float:
        return quantity * self.contract_multiplier * (1.0 / entry - 1.0 / exit) if self.inverse else quantity * self.contract_multiplier * (exit - entry)

    def funding_times(self, start, end) -> pd.DatetimeIndex:
        """Funding timestamps in ``(start, end]`` (UTC, on the interval grid starting at 00:00)."""
        grid = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + pd.Timedelta(days=1), freq=f"{self.funding_interval_hours}h")
        return grid[(grid > pd.Timestamp(start)) & (grid <= pd.Timestamp(end))]

    def funding_payment(self, quantity: float, mark_price: float, funding_rate: float) -> float:
        """Cash paid (negative) or received (positive) by a position of ``quantity`` contracts: ``-q x notional x rate`` (in the settlement currency; coin for inverse)."""
        rate = max(min(funding_rate, self.funding_rate_cap), -self.funding_rate_cap)
        if self.inverse:
            return -quantity * self.contract_multiplier / mark_price * rate
        return -quantity * self.contract_multiplier * mark_price * rate

    def liquidation_price(self, entry: float, quantity: float, leverage: float | None = None) -> float:
        """Isolated-margin liquidation price: the mark at which equity (initial margin plus P&L) falls to the maintenance requirement. Positive for a long, above entry for a short."""
        lev = min(leverage or self.max_leverage, self.max_leverage)
        mmr = self.maintenance_margin_rate
        if quantity > 0:
            return entry * (1.0 - 1.0 / lev + mmr)
        if quantity < 0:
            return entry * (1.0 + 1.0 / lev - mmr)
        return float("nan")


@register_kind("crypto_future")
@dataclass(frozen=True, kw_only=True)
class CryptoFuture(CryptoPerp):
    """A dated crypto future: the perpetual's P&L and margin rules, no funding, converging to the index at ``expiry`` (cash settled)."""

    funding_interval_hours: int = 24
    funding_rate_cap: float = 0.0

    def __post_init__(self):
        CryptoPerp.__post_init__(self)
        if self.expiry is None:
            raise ValueError("a dated future needs an expiry")

    def funding_times(self, start, end) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([])

    def funding_payment(self, quantity: float, mark_price: float, funding_rate: float) -> float:
        return 0.0


def _crypto_defaults(venue: str, base: str, quote: str | None, contract_type: str | None) -> dict:
    v = VENUES.get(venue.upper())
    if v is None:
        raise KeyError(f"unknown venue '{venue}'; known: {sorted(VENUES)} (add an entry to VENUES)")
    quote = quote or v["quote"]
    ctype = contract_type or v["perp_type"]
    ccy = base if ctype == "inverse" else quote
    return dict(v=v, quote=quote, ctype=ctype, ccy=ccy)


def crypto_spot(venue: str, base: str, quote: str | None = None, **kwargs) -> CryptoSpot:
    d = _crypto_defaults(venue, base, quote, None)
    iid = f"{venue.upper()}:{base}-{d['quote']}:SPOT"
    defaults = dict(instrument_id=iid, asset_class="crypto", instrument_type="crypto_spot", currency=d["quote"], exchange=venue.upper(), tick_size=d["v"]["tick"], lot_size=d["v"]["lot"],
                    calendar="24x7", price_precision=2, quantity_precision=6, settlement_type="physical", base_currency=base, quote_currency=d["quote"], venue=venue.upper())
    defaults.update(kwargs)
    return CryptoSpot(**defaults)


def crypto_perp(venue: str, base: str, quote: str | None = None, contract_type: str | None = None, **kwargs) -> CryptoPerp:
    d = _crypto_defaults(venue, base, quote, contract_type)
    v = d["v"]
    iid = f"{venue.upper()}:{base}-{d['quote']}:PERP"
    defaults = dict(instrument_id=iid, asset_class="crypto", instrument_type="crypto_perp", currency=d["ccy"], exchange=venue.upper(), tick_size=v["tick"], lot_size=v["lot"],
                    contract_multiplier=v["contract_size"], calendar="24x7", price_precision=2, quantity_precision=6, settlement_type="cash", margin_type="percent",
                    initial_margin=1.0 / v["max_leverage"], maintenance_margin=v["mmr"], underlying_id=f"{venue.upper()}:{base}-{d['quote']}:SPOT", base_currency=base,
                    quote_currency=d["quote"], venue=venue.upper(), contract_type=d["ctype"], funding_interval_hours=v["funding_hours"], max_leverage=v["max_leverage"],
                    maintenance_margin_rate=v["mmr"])
    defaults.update(kwargs)
    return CryptoPerp(**defaults)


def crypto_future(venue: str, base: str, expiry, quote: str | None = None, contract_type: str | None = None, **kwargs) -> CryptoFuture:
    d = _crypto_defaults(venue, base, quote, contract_type)
    v = d["v"]
    iid = f"{venue.upper()}:{base}-{d['quote']}:FUT:{pd.Timestamp(expiry):%Y%m%d}"
    defaults = dict(instrument_id=iid, asset_class="crypto", instrument_type="crypto_future", currency=d["ccy"], exchange=venue.upper(), tick_size=v["tick"], lot_size=v["lot"],
                    contract_multiplier=v["contract_size"], calendar="24x7", price_precision=2, quantity_precision=6, settlement_type="cash", margin_type="percent",
                    initial_margin=1.0 / v["max_leverage"], maintenance_margin=v["mmr"], underlying_id=f"{venue.upper()}:{base}-{d['quote']}:SPOT", expiry=pd.Timestamp(expiry),
                    base_currency=base, quote_currency=d["quote"], venue=venue.upper(), contract_type=d["ctype"], max_leverage=v["max_leverage"], maintenance_margin_rate=v["mmr"])
    defaults.update(kwargs)
    return CryptoFuture(**defaults)
