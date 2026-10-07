"""Orders: requests to trade, their lifecycle and validation against the instrument's contract specification.

Types: ``market``, ``limit`` (rest until the market reaches the price), ``stop`` (becomes a market order when the price trades through the stop), ``stop_limit``. Time in force: ``GTC``
(stays working), ``DAY`` (expires at the end of the submission day), ``IOC`` (fill what you can now, cancel the rest) and ``FOK`` (fill completely now or cancel). ``reduce_only`` orders can
only shrink the current position. Quantities are signed (positive buys) and are rounded TOWARD ZERO to whole lots: an order is never larger than requested, and one smaller than a lot
is rejected with a reason rather than silently dropped.

Status flow: ``new -> working -> (partial) -> filled`` or ``-> cancelled / rejected / expired``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ORDER_TYPES = ("market", "limit", "stop", "stop_limit")
TIME_IN_FORCE = ("GTC", "DAY", "IOC", "FOK")
OPEN_STATUSES = ("new", "working", "partial")


@dataclass
class Order:
    instrument_id: str
    quantity: float
    type: str = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = "GTC"
    strategy: str = ""
    tags: tuple = ()
    reduce_only: bool = False
    participation: float | None = None          # cap on the share of an event's volume this order may take
    id: int = -1
    submitted_at: pd.Timestamp | None = None
    arrival_at: pd.Timestamp | None = None
    status: str = "new"
    filled: float = 0.0
    avg_price: float = 0.0
    reason: str = ""
    triggered: bool = False
    fees: float = 0.0

    @property
    def remaining(self) -> float:
        return self.quantity - self.filled

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def side(self) -> int:
        return 1 if self.quantity > 0 else -1


def validate_order(order: Order, inst, current_position: float, ts) -> str:
    """The reason an order cannot be accepted, or an empty string. Rounds the quantity to lots in place."""
    if order.type not in ORDER_TYPES or order.tif not in TIME_IN_FORCE:
        return f"unknown order type '{order.type}' or time in force '{order.tif}'"
    if order.quantity == 0 or order.quantity != order.quantity:
        return "quantity is zero or not a number"
    rounded = inst.round_quantity(order.quantity)
    if rounded == 0.0:
        return f"quantity {order.quantity:g} is smaller than one lot ({inst.lot_size:g})"
    order.quantity = rounded
    if order.type in ("limit", "stop_limit") and (order.limit_price is None or not order.limit_price > 0):
        return "a limit order needs a positive limit_price"
    if order.type in ("stop", "stop_limit") and (order.stop_price is None or not order.stop_price > 0):
        return "a stop order needs a positive stop_price"
    ts = pd.Timestamp(ts)
    if inst.expiry is not None and ts > inst.expiry + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1):
        return f"{inst.instrument_id} expired on {inst.expiry}"
    first = getattr(inst, "first_trade", None)
    if first is not None and ts < first:
        return f"{inst.instrument_id} is not listed before {first}"
    if order.reduce_only:
        if current_position == 0 or (current_position > 0) == (order.quantity > 0):
            return "reduce_only order would not reduce the position"
        if abs(order.quantity) > abs(current_position):
            order.quantity = -current_position
    return ""
