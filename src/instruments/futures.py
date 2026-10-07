"""Futures contracts, contract chains and roll specifications.

A :class:`Future` is ONE contract (``ESH22``); a :class:`FutureChain` is the ordered set of contracts on one root (``ES``) plus a :class:`RollSpec` saying when a position in the front
contract moves to the next. The engine holds positions in individual contracts, so a roll is an explicit pair of trades with its own costs, not a relabelling of a price series.

Roll methods (``RollSpec.method``):

``calendar``        roll ``days_before_expiry`` business days before the last trading day (the default; needs only the expiry calendar)
``volume``          roll once the next contract's volume exceeds the front's by ``ratio`` (needs volume data; falls back to the calendar roll ``days_before_expiry`` days out at the latest)
``open_interest``   the same on open interest
``fixed_days``      roll on a fixed calendar day count before expiry (``days_before_expiry`` calendar days)

``FutureChain.roll_schedule`` gives the calendar dates for the calendar and fixed-day methods; the data-driven methods are evaluated by the engine against point-in-time data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .base import Instrument, register_kind
from .calendars import get_calendar

ROLL_METHODS = ("calendar", "volume", "open_interest", "fixed_days")
MONTH_CODES = "FGHJKMNQUVXZ"


@dataclass(frozen=True)
class RollSpec:
    method: str = "calendar"
    days_before_expiry: int = 5
    ratio: float = 1.0                         # next-contract volume (or open interest) must exceed the front's by this factor

    def __post_init__(self):
        if self.method not in ROLL_METHODS:
            raise ValueError(f"roll method must be one of {ROLL_METHODS}")
        if self.days_before_expiry < 0 or self.ratio <= 0:
            raise ValueError("days_before_expiry must be >= 0 and ratio > 0")

    def to_dict(self) -> dict:
        return {"method": self.method, "days_before_expiry": self.days_before_expiry, "ratio": self.ratio}


@register_kind("future")
@dataclass(frozen=True, kw_only=True)
class Future(Instrument):
    chain_id: str = ""
    root: str = ""
    delivery_month: pd.Timestamp | None = None
    first_trade: pd.Timestamp | None = None
    first_notice: pd.Timestamp | None = None
    inverse: bool = False
    roll: RollSpec = field(default_factory=RollSpec)

    CASH_STYLE = "variation_margin"
    TIMESTAMP_FIELDS = ("expiry", "delivery_month", "first_trade", "first_notice")

    def __post_init__(self):
        Instrument.__post_init__(self)
        if self.expiry is None:
            raise ValueError("a future needs an expiry (last trading day)")
        for name in ("delivery_month", "first_trade", "first_notice"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, pd.Timestamp):
                object.__setattr__(self, name, pd.Timestamp(v))
        if isinstance(self.roll, dict):
            object.__setattr__(self, "roll", RollSpec(**self.roll))
        if self.first_trade is not None and self.first_trade > self.expiry:
            raise ValueError("first_trade is after expiry")

    @property
    def last_trade(self) -> pd.Timestamp:
        return self.expiry

    def days_to_expiry(self, ts) -> int:
        return int((self.expiry.normalize() - pd.Timestamp(ts).normalize()).days)

    def is_tradable(self, ts) -> bool:
        ts = pd.Timestamp(ts)
        return (self.first_trade is None or ts >= self.first_trade) and ts <= self.expiry + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    def notional(self, price: float, quantity: float = 1.0) -> float:
        return abs(quantity) * self.contract_multiplier * (1.0 if self.inverse else price)

    def settlement_notional(self, price: float, quantity: float = 1.0) -> float:
        return abs(quantity) * self.contract_multiplier / price if self.inverse else self.notional(price, quantity)

    def pnl(self, entry: float, exit: float, quantity: float) -> float:
        return quantity * self.contract_multiplier * (1.0 / entry - 1.0 / exit) if self.inverse else quantity * self.contract_multiplier * (exit - entry)

    @classmethod
    def decode_mapping(cls, key, value):
        return RollSpec(**value) if key == "roll" else value

    def to_dict(self) -> dict:
        out = Instrument.to_dict(self)
        out["roll"] = self.roll.to_dict()
        return out


@dataclass(frozen=True)
class FutureChain:
    chain_id: str
    root: str
    contracts: tuple
    roll: RollSpec = field(default_factory=RollSpec)

    def __post_init__(self):
        if not self.contracts:
            raise ValueError("a chain needs at least one contract")
        object.__setattr__(self, "contracts", tuple(sorted(self.contracts, key=lambda c: c.expiry)))

    def active(self, ts) -> list[Future]:
        """Contracts that exist and have not expired at ``ts`` (nearest first)."""
        ts = pd.Timestamp(ts)
        return [c for c in self.contracts if c.is_tradable(ts)]

    def front(self, ts, rank: int = 0) -> Future | None:
        a = self.active(ts)
        return a[rank] if len(a) > rank else None

    def after(self, contract: Future) -> Future | None:
        ids = [c.instrument_id for c in self.contracts]
        i = ids.index(contract.instrument_id)
        return self.contracts[i + 1] if i + 1 < len(ids) else None

    def roll_date(self, contract: Future) -> pd.Timestamp:
        """The calendar roll date of ``contract`` under this chain's ``RollSpec`` (the data-driven methods use it as the latest allowed date)."""
        cal = get_calendar(contract.calendar)
        if self.roll.method == "fixed_days":
            return (contract.expiry - pd.Timedelta(days=self.roll.days_before_expiry)).normalize()
        return cal.add_business_days(contract.expiry.normalize(), -self.roll.days_before_expiry)

    def roll_schedule(self) -> pd.DataFrame:
        rows = []
        for c in self.contracts[:-1]:
            nxt = self.after(c)
            rows.append({"from_contract": c.instrument_id, "to_contract": nxt.instrument_id, "roll_date": self.roll_date(c), "expiry": c.expiry})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return {"chain_id": self.chain_id, "root": self.root, "roll": self.roll.to_dict(), "contracts": [c.to_dict() for c in self.contracts]}


def build_chain(root: str, start, end, months=tuple(range(1, 13)), expiry_rule: str = "third_friday", multiplier: float = 1.0, tick_size: float = 0.01, currency: str = "USD",
                calendar: str = "US", exchange: str = "", initial_margin: float = 0.0, maintenance_margin: float = 0.0, roll: RollSpec | None = None, asset_class: str = "future",
                settlement_type: str = "cash", underlying_id: str | None = None, first_trade_months: int = 12, price_precision: int = 2, inverse: bool = False) -> FutureChain:
    """Generate a chain from an expiry rule (``src.assets.futures.contract_calendar`` rules), adjusting each expiry to a business day of ``calendar``.

    Each contract lists ``first_trade_months`` before its delivery month. ``initial_margin`` and ``maintenance_margin`` are per contract (``margin_type='fixed'``)."""
    from ..assets.futures import contract_calendar

    cal = get_calendar(calendar)
    roll = roll or RollSpec()
    table = contract_calendar(start, end, months, expiry_rule)
    contracts = []
    for _, row in table.iterrows():
        expiry = cal.adjust(row["expiry"], "preceding")
        delivery = pd.Timestamp(row["delivery_month"])
        code = row["contract"]
        contracts.append(Future(instrument_id=f"{root}{code}", asset_class=asset_class, instrument_type="future", currency=currency, exchange=exchange, tick_size=tick_size,
                                contract_multiplier=multiplier, calendar=calendar, underlying_id=underlying_id, expiry=expiry, settlement_type=settlement_type,
                                margin_type="fixed" if initial_margin else "none", price_precision=price_precision, initial_margin=initial_margin,
                                maintenance_margin=maintenance_margin, chain_id=root, root=root, delivery_month=delivery,
                                first_trade=expiry - pd.DateOffset(months=first_trade_months), roll=roll, inverse=inverse))
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.DateOffset(months=18)
    contracts = [c for c in contracts if lo - pd.DateOffset(months=first_trade_months) <= c.expiry <= hi]
    return FutureChain(root, root, tuple(contracts), roll)
