"""The journal: an append-only record of every movement of cash and value, each entry explained.

Every state change in the ledger writes at least one :class:`JournalEntry`, and each entry satisfies a per-entry accounting identity in the ledger's base currency::

    cash_base + value + translation  =  pnl + transfer

``cash`` is the movement of cash balances (local currencies) and ``cash_base`` its value at the rates in force when the entry was written, ``value`` the change in the value of positions, ``translation`` the effect of exchange-rate changes on holdings,
``pnl`` the profit or loss the entry represents and ``transfer`` an external deposit or withdrawal. A trade at the mid price moves cash and position value by equal and opposite
amounts (pnl zero); a fee has pnl equal to the cash paid; funding, coupons and variation margin are cash and pnl together. The ledger CHECKS the identity as it writes, so an entry that
creates or destroys value without saying so is rejected at the moment it is made, not found at the end of the run.

Categories (``CATEGORIES``): ``trade`` (principal at the mid), ``spread``, ``impact``, ``slippage`` (the cost of crossing from the mid to the fill), ``fee``, ``mtm`` (change in the
value of non-margined positions), ``variation_margin`` (cash settlement of futures-style positions), ``funding`` (perpetual funding), ``borrow`` (fees on borrowed securities or coins),
``interest`` (earned on cash), ``financing`` (paid on negative cash), ``coupon`` (swap, bond and forward cashflows), ``dividend``, ``corporate_action``, ``lifecycle_settlement``
(expiry, exercise, assignment, delivery), ``fx_translation``, ``transfer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CATEGORIES = ("trade", "spread", "impact", "slippage", "fee", "mtm", "variation_margin", "funding", "borrow", "interest", "financing", "coupon", "dividend", "corporate_action",
              "lifecycle_settlement", "fx_translation", "transfer")
PNL_CATEGORIES = tuple(c for c in CATEGORIES if c not in ("trade", "transfer"))


@dataclass(frozen=True)
class JournalEntry:
    seq: int
    ts: pd.Timestamp
    category: str
    instrument_id: str
    asset_class: str
    currency: str                      # the currency the entry is mainly denominated in (informational)
    cash: dict = field(default_factory=dict)
    cash_base: float = 0.0             # the cash movements valued at the exchange rates in force when the entry was written
    value: float = 0.0
    translation: float = 0.0
    pnl: float = 0.0
    transfer: float = 0.0
    tags: tuple = ()
    strategy: str = ""
    note: str = ""


def to_frame(entries: list[JournalEntry]) -> pd.DataFrame:
    rows = []
    for e in entries:
        rows.append({"seq": e.seq, "ts": e.ts, "category": e.category, "instrument_id": e.instrument_id, "asset_class": e.asset_class, "currency": e.currency, "cash_base": e.cash_base, "value": e.value,
                     "translation": e.translation, "pnl": e.pnl, "transfer": e.transfer, "tags": ",".join(e.tags), "strategy": e.strategy, "note": e.note,
                     "cash": dict(e.cash)})
    return pd.DataFrame(rows)
