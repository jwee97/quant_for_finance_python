"""Positions and fills: the records the ledger keeps and consumes."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Fill:
    """One execution. ``price`` is the all-in fill price; ``spread_price`` and ``impact_price`` are the per-unit components of the distance from the mid (always >= 0, both costs).

    ``quantity`` is signed (positive buys). ``fee`` is in ``fee_currency`` (the instrument's currency if empty). ``settlement`` marks a delivery at a contractual price (exercise or
    assignment), whose distance from the market goes to ``lifecycle_settlement`` instead of ``spread``/``impact``/``slippage``."""

    ts: pd.Timestamp
    instrument_id: str
    quantity: float
    price: float
    order_id: int = -1
    fee: float = 0.0
    fee_currency: str = ""
    spread_price: float = 0.0
    impact_price: float = 0.0
    tags: tuple = ()
    strategy: str = ""
    settlement: bool = False                  # a delivery at a contractual price (exercise, assignment): the gap to the market is lifecycle P&L, not a trading cost


@dataclass
class Position:
    instrument_id: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0                 # in the instrument's currency, from quantity closed (average-cost method)
    last_mark: float = float("nan")           # price at the last revaluation
    last_value: float = 0.0                   # position value in the instrument's currency at the last revaluation (zero for variation-margin and currency-exchange styles)
    opened_at: pd.Timestamp | None = None
    fees_paid: float = 0.0
    funding_paid: float = 0.0                 # signed: negative = paid
    tags: set = field(default_factory=set)

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-12

    def unrealized_pnl(self, instrument) -> float:
        """Open profit in the instrument's currency against the average entry price at the last mark (informational: cash is moved by the cash style)."""
        if self.is_flat or self.last_mark != self.last_mark:
            return 0.0
        return instrument.pnl(self.avg_price, self.last_mark, self.quantity)
