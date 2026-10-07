"""The instrument and contract-specification model: *what is being traded*, independent of any data feed, engine or strategy.

Every tradable thing is an immutable :class:`Instrument` with the same core fields, so one engine and one ledger can handle all of them:

====================  =================================================================================================================================
field                 meaning
====================  =================================================================================================================================
``instrument_id``     unique key, used everywhere (market data, orders, positions); convention ``VENUE:BASE-QUOTE:TYPE[:EXPIRY]`` for crypto, root plus month code for futures
``asset_class``       ``fx``, ``equity``, ``future``, ``crypto``, ``option``, ``swap``, ``bond`` ...
``instrument_type``   the specialisation: ``fx_spot``, ``fx_forward``, ``fx_swap``, ``xccy_basis_swap``, ``future``, ``crypto_spot``, ``crypto_perp``, ``crypto_future``, ``option``, ``irs``, ...
``currency``          the currency P&L and prices are settled in (the quote currency; the coin for inverse contracts)
``tick_size``         the minimum price increment; ``price_precision`` the decimals shown
``lot_size``          the minimum quantity increment; ``quantity_precision`` the decimals shown
``contract_multiplier``  currency units per price point per contract (a future's point value, an option's 100 shares, a swap's notional scale)
``calendar``          the trading calendar name (see ``calendars``); ``session`` the daily trading hours
``underlying_id``     what a derivative is written on; ``expiry`` when it ceases to exist; ``settlement_type`` ``cash``, ``physical`` or ``none``
``margin_type``       how collateral is required: ``none``, ``fixed`` (per contract), ``percent`` (of notional), ``premium`` (options), ``portfolio`` (scenario scan)
====================  =================================================================================================================================

The cash treatment a position needs follows from the instrument type and is exposed as :attr:`Instrument.cash_style`:

``full_payment``       pay the price in cash, hold an asset worth quantity x multiplier x mark (shares, bonds)
``premium``            the same arithmetic for options: the premium is paid or received in full
``otc_mtm``            an OTC contract entered at (near) zero value and carried at its present value (swaps, forwards); cash moves only when cashflows are paid
``variation_margin``   no cash at entry; every mark-to-market gain or loss is settled in cash (futures, perpetuals, cleared swaps)
``currency_exchange``  buying one currency or coin with another: the balances *are* the position (spot FX, spot crypto)

Subclasses live in ``fx``, ``futures``, ``crypto``, ``options`` and ``rates``. Every instrument serialises to and from a plain dictionary (``to_dict`` / ``instrument_from_dict``).
"""

from __future__ import annotations

import math
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from typing import ClassVar

import pandas as pd

from .calendars import Session, get_calendar

ASSET_CLASSES = ("fx", "equity", "future", "crypto", "option", "swap", "bond", "rate", "commodity", "cash")
SETTLEMENT_TYPES = ("cash", "physical", "none")
MARGIN_TYPES = ("none", "fixed", "percent", "premium", "portfolio")
CASH_STYLES = ("full_payment", "premium", "otc_mtm", "variation_margin", "currency_exchange")

_KINDS: dict[str, type] = {}


def register_kind(kind: str):
    """Class decorator: make ``instrument_from_dict`` able to rebuild instruments of this ``kind``."""
    def wrap(cls):
        cls.KIND = kind
        _KINDS[kind] = cls
        return cls
    return wrap


def _ts(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else pd.Timestamp(value)


@register_kind("instrument")
@dataclass(frozen=True, kw_only=True)
class Instrument:
    instrument_id: str
    asset_class: str = "equity"
    instrument_type: str = "instrument"
    currency: str = "USD"
    exchange: str = ""
    tick_size: float = 0.01
    lot_size: float = 1.0
    contract_multiplier: float = 1.0
    calendar: str = "WEEKDAY"
    session: Session | None = None
    underlying_id: str | None = None
    expiry: pd.Timestamp | None = None
    settlement_type: str = "none"
    margin_type: str = "none"
    price_precision: int = 2
    quantity_precision: int = 0
    initial_margin: float = 0.0                 # per contract (margin_type fixed) or fraction of notional (percent)
    maintenance_margin: float = 0.0
    tags: tuple = ()

    KIND: ClassVar[str] = "instrument"
    CASH_STYLE: ClassVar[str] = "full_payment"
    TIMESTAMP_FIELDS: ClassVar[tuple] = ("expiry",)

    def __post_init__(self):
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(f"asset_class '{self.asset_class}' not in {ASSET_CLASSES}")
        if self.settlement_type not in SETTLEMENT_TYPES:
            raise ValueError(f"settlement_type must be one of {SETTLEMENT_TYPES}")
        if self.margin_type not in MARGIN_TYPES:
            raise ValueError(f"margin_type must be one of {MARGIN_TYPES}")
        if self.tick_size <= 0 or self.lot_size <= 0 or self.contract_multiplier <= 0:
            raise ValueError("tick_size, lot_size and contract_multiplier must be positive")
        if self.initial_margin < 0 or self.maintenance_margin < 0 or self.maintenance_margin > self.initial_margin and self.initial_margin > 0:
            raise ValueError("margins must be non-negative with maintenance <= initial")
        if self.expiry is not None and not isinstance(self.expiry, pd.Timestamp):
            object.__setattr__(self, "expiry", pd.Timestamp(self.expiry))
        get_calendar(self.calendar)                                                  # fails early on an unknown calendar

    # ----------------------------------------------------------------------------------------------------------------------------- conventions
    @property
    def cash_style(self) -> str:
        return self.CASH_STYLE

    @property
    def is_derivative(self) -> bool:
        return self.underlying_id is not None

    @property
    def tick_value(self) -> float:
        """Currency value of one tick for one contract."""
        return self.tick_size * self.contract_multiplier

    @property
    def cal(self):
        return get_calendar(self.calendar)

    def round_price(self, price: float) -> float:
        """Round to the nearest tick (and the price precision)."""
        return round(round(price / self.tick_size) * self.tick_size, max(self.price_precision, 10 if self.tick_size < 1e-10 else self.price_precision))

    def round_quantity(self, quantity: float) -> float:
        """Round TOWARD ZERO to a whole number of lots: an order may be smaller than requested, never larger."""
        lots = math.floor(abs(quantity) / self.lot_size + 1e-9)
        return math.copysign(round(lots * self.lot_size, max(self.quantity_precision, 0) + 6), quantity) if lots else 0.0

    def notional(self, price: float, quantity: float = 1.0) -> float:
        """Absolute notional value in ``currency`` (the quantity of risk the position carries)."""
        return abs(quantity) * self.contract_multiplier * price

    def settlement_notional(self, price: float, quantity: float = 1.0) -> float:
        """Notional expressed in the SETTLEMENT currency (``currency``): the coin for an inverse contract, otherwise the same as :meth:`notional`."""
        return self.notional(price, quantity)

    def pnl(self, entry: float, exit: float, quantity: float) -> float:
        """Profit in ``currency`` of ``quantity`` contracts (signed) between two prices; inverse contracts override."""
        return quantity * self.contract_multiplier * (exit - entry)

    def is_expired(self, ts) -> bool:
        return self.expiry is not None and pd.Timestamp(ts) > self.expiry

    def has_expired_by(self, ts) -> bool:
        """True from the expiry timestamp itself onward (the contract no longer trades once the engine reaches it)."""
        return self.expiry is not None and pd.Timestamp(ts) >= self.expiry

    # ------------------------------------------------------------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, pd.Timestamp):
                v = v.isoformat()
            elif isinstance(v, Session):
                v = asdict(v)
            elif hasattr(v, "to_dict") and not isinstance(v, (dict, pd.Series, pd.DataFrame)):
                v = v.to_dict()
            elif isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        out["kind"] = self.KIND
        return out

    def replace(self, **changes):
        return replace(self, **changes)

    def describe(self) -> str:
        return f"{self.instrument_id} ({self.instrument_type}, {self.currency}, x{self.contract_multiplier:g}, tick {self.tick_size:g})"


def instrument_from_dict(data: dict) -> Instrument:
    """Rebuild an instrument from :meth:`Instrument.to_dict` output."""
    data = dict(data)
    kind = data.pop("kind", "instrument")
    cls = _KINDS.get(kind)
    if cls is None:
        raise KeyError(f"unknown instrument kind '{kind}'; known: {sorted(_KINDS)}")
    kwargs = {}
    names = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"{kind} has no field '{key}'")
        if key in cls.TIMESTAMP_FIELDS:
            value = _ts(value)
        elif key == "session" and isinstance(value, dict):
            value = Session(**value)
        elif isinstance(value, list):
            value = cls.decode_sequence(key, value) if hasattr(cls, "decode_sequence") else tuple(value)
        elif isinstance(value, dict) and hasattr(cls, "decode_mapping"):
            value = cls.decode_mapping(key, value)
        kwargs[key] = value
    return cls(**kwargs)


def required_fields(cls) -> list[str]:
    return [f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING]


__all__ = ["Instrument", "instrument_from_dict", "register_kind", "ASSET_CLASSES", "SETTLEMENT_TYPES", "MARGIN_TYPES", "CASH_STYLES", "field", "dataclass"]
