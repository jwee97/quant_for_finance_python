"""The portfolio ledger: one account for every instrument, in every currency, with every movement explained.

**What it holds.** Cash balances per currency (negative = borrowed), a :class:`Position` per instrument, a currency converter built from observed pair prices, and the append-only
:class:`~src.ledger.journal.JournalEntry` list. Equity in the base currency is::

    equity = sum over currencies of  rate[ccy] x ( cash[ccy] + value of the positions denominated in ccy )

where a position's value depends on the instrument's CASH STYLE (see ``instruments.base``): shares, bonds, options and OTC contracts are worth quantity x multiplier x mark; futures and
perpetuals are worth zero because every mark-to-market change is paid in cash (variation margin); a spot FX or crypto position IS the balances of the two currencies.

**What it guarantees.** Every operation writes entries that obey the per-entry identity of the journal, so:

* cash conservation: ``cash[ccy] = starting cash + sum of the entries' cash movements``;
* P&L attribution: ``equity - starting equity = sum of (pnl + transfer)`` over the journal, with nothing created or destroyed unexplained;
* an operation that breaks its own identity raises :class:`LedgerError` immediately.

**Operations.** ``deposit``; ``revalue`` (mark positions, settle variation margin, translate foreign balances); ``apply_fill`` (a trade, split into principal at the mid price and the
costs of crossing to the fill: spread, impact, residual slippage, plus fees); ``post_cashflow`` (funding, borrow, interest, financing, coupons, dividends); ``close_position`` and
``remove_position_value`` (lifecycle settlement, exercise, assignment); ``split`` (corporate action); ``accrue_interest`` and ``accrue_borrow`` (financing on balances and short positions).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..instruments.base import Instrument
from .currency import CurrencyConverter
from .journal import CATEGORIES, JournalEntry, to_frame
from .position import Fill, Position


class LedgerError(Exception):
    pass


class Ledger:
    def __init__(self, registry, base_currency: str = "USD", initial_cash: Mapping[str, float] | None = None, converter: CurrencyConverter | None = None,
                 tolerance: float = 1e-6, start: pd.Timestamp | None = None):
        self.registry = registry
        self.base = base_currency
        self.converter = converter or CurrencyConverter(base_currency, pegs=None)
        self.tolerance = tolerance
        self.cash: dict[str, float] = {}
        self.positions: dict[str, Position] = {}
        self.rates: dict[str, float] = {base_currency: 1.0}
        self.journal: list[JournalEntry] = []
        self.fills: list[Fill] = []
        self.closed: list[Position] = []
        self.starting_cash: dict[str, float] = {}
        self.transfers = 0.0
        self._seq = 0
        self.ts = pd.Timestamp(start) if start is not None else None
        for ccy, amount in (initial_cash or {}).items():
            self.deposit(self.ts or pd.Timestamp("1970-01-01"), ccy, amount)

    # ------------------------------------------------------------------------------------------------------------------------------ helpers
    def _inst(self, iid: str) -> Instrument:
        return self.registry.get(iid)

    def rate(self, ccy: str) -> float:
        r = self.rates.get(ccy)
        if r is None:
            r = self.converter.rate(ccy)
            self.rates[ccy] = r
        return r

    def _post(self, ts, category: str, instrument: Instrument | None = None, currency: str = "", cash: Mapping[str, float] | None = None, value: float = 0.0, translation: float = 0.0,
              pnl: float | None = None, transfer: float = 0.0, tags=(), strategy: str = "", note: str = "") -> JournalEntry:
        if category not in CATEGORIES:
            raise LedgerError(f"unknown journal category '{category}'")
        cash = {c: float(a) for c, a in (cash or {}).items() if a != 0.0}
        cash_base = sum(a * self.rate(c) for c, a in cash.items())
        lhs = cash_base + value + translation
        if pnl is None:
            pnl = lhs - transfer
        elif abs(lhs - (pnl + transfer)) > self.tolerance * max(1.0, abs(lhs), abs(pnl)):
            raise LedgerError(f"entry '{category}' for {instrument.instrument_id if instrument else ''} breaks the accounting identity: movements {lhs:.8g} != pnl + transfer {pnl + transfer:.8g}")
        for c, a in cash.items():
            self.cash[c] = self.cash.get(c, 0.0) + a
        self.ts = pd.Timestamp(ts)
        self._seq += 1
        entry = JournalEntry(self._seq, pd.Timestamp(ts), category, instrument.instrument_id if instrument else "", instrument.asset_class if instrument else "cash",
                             currency or (instrument.currency if instrument else self.base), cash, float(cash_base), float(value), float(translation), float(pnl), float(transfer), tuple(tags), strategy, note)
        self.journal.append(entry)
        return entry

    def _position(self, iid: str) -> Position:
        pos = self.positions.get(iid)
        if pos is None:
            pos = self.positions[iid] = Position(iid)
        return pos

    # --------------------------------------------------------------------------------------------------------------------------- operations
    def deposit(self, ts, ccy: str, amount: float, note: str = "deposit") -> None:
        """An external transfer of cash in (positive) or out (negative): changes equity but is not profit."""
        base_amount = amount * self.rate(ccy) if ccy in self.rates or self.converter.has_rate(ccy) else None
        if base_amount is None:
            raise LedgerError(f"no rate for {ccy}: observe a pair linking it to {self.base} (or add a peg) before depositing it")
        self.starting_cash[ccy] = self.starting_cash.get(ccy, 0.0) + amount
        self.transfers += base_amount
        self._post(ts, "transfer", None, ccy, {ccy: amount}, 0.0, 0.0, 0.0, base_amount, note=note)

    def post_cashflow(self, ts, category: str, ccy: str, amount: float, instrument_id: str = "", tags=(), strategy: str = "", note: str = "") -> JournalEntry | None:
        """Cash that is also profit or loss: ``funding``, ``borrow``, ``interest``, ``financing``, ``coupon``, ``dividend``, ``corporate_action``, ``lifecycle_settlement``."""
        if category in ("trade", "transfer", "mtm", "variation_margin", "fx_translation"):
            raise LedgerError(f"post_cashflow cannot write '{category}' entries")
        if amount == 0.0:
            return None
        inst = self._inst(instrument_id) if instrument_id else None
        return self._post(ts, category, inst, ccy, {ccy: amount}, tags=tags, strategy=strategy, note=note)

    def revalue(self, ts, marks: Mapping[str, float], strategy: str = "") -> None:
        """Mark positions to ``marks`` (instrument id -> mark in price terms): updates exchange rates from currency-pair marks, translates foreign balances, settles variation margin in
        cash and carries other positions at their new value. Positions with no mark keep their last one."""
        ts = pd.Timestamp(ts)
        for iid, m in marks.items():
            if iid not in self.registry:
                continue
            inst = self._inst(iid)
            if inst.cash_style == "currency_exchange" and hasattr(inst, "base_currency") and m == m and m > 0:
                self.converter.set_pair(inst.base_currency, inst.quote_currency, m)
        held = set(self.cash) | {self._inst(i).currency for i, p in self.positions.items() if not p.is_flat}
        for ccy in sorted(held):
            if ccy == self.base:
                continue
            old = self.rates.get(ccy)
            try:
                new = self.converter.rate(ccy)
            except KeyError:
                if old is None and (self.cash.get(ccy, 0.0) != 0.0):
                    raise LedgerError(f"cannot value the {ccy} balance: no rate for {ccy} against {self.base}") from None
                continue
            if old is not None and new != old:
                na = self.cash.get(ccy, 0.0) + sum(p.last_value for i, p in self.positions.items() if self._inst(i).currency == ccy)
                self.rates[ccy] = new
                if na != 0.0:
                    self._post(ts, "fx_translation", None, ccy, None, 0.0, na * (new - old), None, 0.0, strategy=strategy, note=f"{ccy} {old:.6g} -> {new:.6g}")
            else:
                self.rates[ccy] = new
        for iid, m in marks.items():
            pos = self.positions.get(iid)
            if pos is None or pos.is_flat or not (m == m):
                continue
            inst = self._inst(iid)
            style = inst.cash_style
            ccy = inst.currency
            if style == "variation_margin":
                if pos.last_mark == pos.last_mark and m != pos.last_mark:
                    delta = inst.pnl(pos.last_mark, m, pos.quantity)
                    self._post(ts, "variation_margin", inst, ccy, {ccy: delta}, strategy=strategy)
                pos.last_mark = m
            elif style == "currency_exchange":
                pos.last_mark = m
            else:
                new_value = pos.quantity * inst.contract_multiplier * m
                delta = new_value - pos.last_value
                if delta != 0.0:
                    pos.last_value = new_value
                    self._post(ts, "mtm", inst, ccy, None, delta * self.rate(ccy), 0.0, None, 0.0, strategy=strategy)
                pos.last_mark = m
        self.ts = ts

    def apply_fill(self, ts, fill: Fill, mid: float, strategy: str | None = None) -> None:
        """Book a trade. ``mid`` is the mid price when the fill happened: principal is booked at the mid and the distance to the fill price is booked as spread, impact and residual
        slippage costs (so the journal shows what crossing the market cost), then the fee."""
        ts = pd.Timestamp(ts)
        inst = self._inst(fill.instrument_id)
        strategy = fill.strategy if strategy is None else strategy
        q = fill.quantity
        if q == 0.0:
            return
        self.revalue(ts, {inst.instrument_id: mid}, strategy)
        pos = self._position(inst.instrument_id)
        style, ccy, mult = inst.cash_style, inst.currency, inst.contract_multiplier
        old_q = pos.quantity
        # average-cost bookkeeping
        if old_q == 0.0 or np.sign(old_q) == np.sign(q):
            new_q = old_q + q
            pos.avg_price = (abs(old_q) * pos.avg_price + abs(q) * fill.price) / abs(new_q)
            if old_q == 0.0:
                pos.opened_at = ts
        else:
            closed = min(abs(q), abs(old_q)) * np.sign(old_q)
            pos.realized_pnl += inst.pnl(pos.avg_price, fill.price, closed)
            new_q = old_q + q
            if abs(new_q) < 1e-12:
                new_q, pos.avg_price = 0.0, 0.0
            elif np.sign(new_q) != np.sign(old_q):
                pos.avg_price = fill.price
        pos.quantity = new_q
        pos.fees_paid += fill.fee
        sgn = 1.0 if q > 0 else -1.0
        spread_c = -inst.pnl(mid + sgn * fill.spread_price, mid, q)
        impact_c = -inst.pnl(mid + sgn * (fill.spread_price + fill.impact_price), mid, q) - spread_c
        total_c = -inst.pnl(fill.price, mid, q)
        slip_c = total_c - spread_c - impact_c
        # principal at the mid
        if style in ("full_payment", "premium", "otc_mtm"):
            principal = q * mult * mid
            new_value = pos.quantity * mult * mid
            value_change = (new_value - pos.last_value)
            pos.last_value = new_value
            self._post(ts, "trade", inst, ccy, {ccy: -principal}, value_change * self.rate(ccy), 0.0, None, 0.0, fill.tags, strategy, f"qty {q:g} @ mid {mid:g}")
        elif style == "currency_exchange":
            base_ccy, quote_ccy = inst.base_currency, inst.quote_currency
            self._post(ts, "trade", inst, quote_ccy, {base_ccy: q * mult, quote_ccy: -q * mult * mid}, 0.0, 0.0, None, 0.0, fill.tags, strategy, f"qty {q:g} @ mid {mid:g}")
        # variation margin: nothing moves at entry; the position is carried from the mid
        pos.last_mark = mid
        for category, amount in (("spread", spread_c), ("impact", impact_c), ("slippage", slip_c)):
            if abs(amount) > 1e-14:
                self._post(ts, category, inst, ccy, {ccy: -amount}, tags=fill.tags, strategy=strategy)
        if fill.fee:
            fee_ccy = fill.fee_currency or ccy
            self._post(ts, "fee", inst, fee_ccy, {fee_ccy: -fill.fee}, tags=fill.tags, strategy=strategy, note=f"order {fill.order_id}")
        self.fills.append(fill)
        if pos.is_flat:
            pos.last_value = 0.0
            self.closed.append(self.positions.pop(inst.instrument_id))

    def close_position(self, ts, instrument_id: str, price: float, category: str = "lifecycle_settlement", note: str = "", strategy: str = "", tags=()) -> None:
        """Settle a whole position at ``price`` with cash (expiry, cash exercise, final settlement): marks it, pays or receives the value, removes it."""
        pos = self.positions.get(instrument_id)
        if pos is None or pos.is_flat:
            return
        ts = pd.Timestamp(ts)
        inst = self._inst(instrument_id)
        self.revalue(ts, {instrument_id: price}, strategy)
        style, ccy = inst.cash_style, inst.currency
        if style in ("full_payment", "premium", "otc_mtm"):
            amount = pos.quantity * inst.contract_multiplier * price
            self._post(ts, category, inst, ccy, {ccy: amount}, -pos.last_value * self.rate(ccy), 0.0, None, 0.0, tags, strategy, note)
            pos.last_value = 0.0
        elif style == "currency_exchange":
            raise LedgerError("a currency-exchange position cannot be settled this way: trade it out")
        else:
            self._post(ts, category, inst, ccy, None, 0.0, 0.0, 0.0, 0.0, tags, strategy, note or "position removed at settlement")
        pos.realized_pnl += inst.pnl(pos.avg_price, price, pos.quantity)
        pos.quantity = 0.0
        self.closed.append(self.positions.pop(instrument_id))

    def remove_position_value(self, ts, instrument_id: str, mark: float, note: str = "", strategy: str = "", tags=()) -> float:
        """Remove a position whose value converts into something else (physical exercise or assignment): mark it, then drop it with NO cash (the value reappears as the delivered
        position). Returns the quantity removed."""
        pos = self.positions.get(instrument_id)
        if pos is None or pos.is_flat:
            return 0.0
        ts = pd.Timestamp(ts)
        inst = self._inst(instrument_id)
        self.revalue(ts, {instrument_id: mark}, strategy)
        removed = pos.quantity
        if inst.cash_style in ("full_payment", "premium", "otc_mtm"):
            self._post(ts, "lifecycle_settlement", inst, inst.currency, None, -pos.last_value * self.rate(inst.currency), 0.0, None, 0.0, tags, strategy, note or "converted by exercise")
            pos.last_value = 0.0
        pos.realized_pnl += inst.pnl(pos.avg_price, mark, pos.quantity)
        pos.quantity = 0.0
        self.closed.append(self.positions.pop(instrument_id))
        return removed

    def settle_with_cash(self, ts, instrument_id: str, cash: Mapping[str, float], mark: float, category: str = "lifecycle_settlement", note: str = "", strategy: str = "", tags=()) -> float:
        """Remove a position and exchange the stated cash for it: the position is marked at ``mark``, its value leaves the books and ``cash`` (currency -> amount) arrives. Used for
        physically settled forwards (both notionals move) and for the final payment of a swap or bond. Whatever difference there is between the value removed and the cash received is
        booked as profit or loss of the entry (zero when ``mark`` is the value of what is exchanged)."""
        pos = self.positions.get(instrument_id)
        if pos is None or pos.is_flat:
            return 0.0
        ts = pd.Timestamp(ts)
        inst = self._inst(instrument_id)
        self.revalue(ts, {instrument_id: mark}, strategy)
        removed = pos.quantity
        value = -pos.last_value * self.rate(inst.currency)
        self._post(ts, category, inst, inst.currency, cash, value, 0.0, None, 0.0, tags, strategy, note)
        pos.last_value = 0.0
        pos.realized_pnl += inst.pnl(pos.avg_price, mark, pos.quantity)
        pos.quantity = 0.0
        self.closed.append(self.positions.pop(instrument_id))
        return removed

    def split(self, ts, instrument_id: str, ratio: float, note: str = "") -> None:
        """A stock split of ``ratio`` new shares per old one: quantity scales up, prices scale down, value is unchanged."""
        pos = self.positions.get(instrument_id)
        if pos is None or ratio <= 0:
            return
        pos.quantity *= ratio
        pos.avg_price /= ratio
        pos.last_mark = pos.last_mark / ratio if pos.last_mark == pos.last_mark else pos.last_mark
        self._post(ts, "corporate_action", self._inst(instrument_id), "", None, 0.0, 0.0, 0.0, 0.0, note=note or f"split {ratio:g}:1")

    def accrue_interest(self, ts, dt_years: float, deposit_rates: Mapping[str, float], borrow_rates: Mapping[str, float] | None = None, strategy: str = "") -> None:
        """Interest on cash: positive balances earn ``deposit_rates``, negative ones pay ``borrow_rates`` (default: the deposit rate plus 50 bp)."""
        for ccy, bal in list(self.cash.items()):
            if bal == 0.0:
                continue
            if bal > 0:
                amount = bal * deposit_rates.get(ccy, 0.0) * dt_years
                self.post_cashflow(ts, "interest", ccy, amount, strategy=strategy, note=f"{ccy} deposit")
            else:
                rate = (borrow_rates or {}).get(ccy, deposit_rates.get(ccy, 0.0) + 0.005)
                self.post_cashflow(ts, "financing", ccy, bal * rate * dt_years, strategy=strategy, note=f"{ccy} borrow")

    def accrue_borrow(self, ts, dt_years: float, borrow_rates: Mapping[str, float], marks: Mapping[str, float], strategy: str = "") -> None:
        """Borrow fees on short positions in securities or coins (full-payment and currency-exchange styles): ``|short notional| x rate x dt`` per instrument."""
        for iid, pos in list(self.positions.items()):
            if pos.quantity >= 0 or iid not in borrow_rates:
                continue
            inst = self._inst(iid)
            if inst.cash_style not in ("full_payment", "currency_exchange"):
                continue
            mark = marks.get(iid, pos.last_mark)
            if mark != mark:
                continue
            fee = inst.notional(mark, pos.quantity) * borrow_rates[iid] * dt_years
            self.post_cashflow(ts, "borrow", inst.currency, -fee, iid, strategy=strategy, note=f"borrow rate {borrow_rates[iid]:.4%}")
            pos.funding_paid -= fee

    # ------------------------------------------------------------------------------------------------------------------------------ queries
    def position_value(self, iid: str) -> float:
        p = self.positions.get(iid)
        return 0.0 if p is None else p.last_value

    def net_assets(self) -> dict[str, float]:
        """Cash plus position value per currency (local units)."""
        out = dict(self.cash)
        for iid, p in self.positions.items():
            c = self._inst(iid).currency
            out[c] = out.get(c, 0.0) + p.last_value
        return out

    def equity(self) -> float:
        """Total value in the base currency."""
        return float(sum(v * self.rate(c) for c, v in self.net_assets().items() if v != 0.0))

    def cash_base(self) -> float:
        return float(sum(v * self.rate(c) for c, v in self.cash.items()))

    def quantity(self, iid: str) -> float:
        p = self.positions.get(iid)
        return 0.0 if p is None else p.quantity

    def exposures(self, marks: Mapping[str, float]) -> pd.DataFrame:
        """One row per open position: quantity, mark, notional and signed notional in the base currency, P&L to date, cash style."""
        rows = []
        for iid, p in self.positions.items():
            inst = self._inst(iid)
            m = marks.get(iid, p.last_mark)
            notional = inst.settlement_notional(m, p.quantity) if m == m else float("nan")
            sign = 1.0 if p.quantity > 0 else -1.0
            notional_base = notional * self.rate(inst.currency)
            rows.append({"instrument_id": iid, "asset_class": inst.asset_class, "type": inst.instrument_type, "currency": inst.currency, "quantity": p.quantity, "avg_price": p.avg_price,
                         "mark": m, "notional_base": notional_base, "signed_notional_base": sign * notional_base, "unrealized_pnl": p.unrealized_pnl(inst), "realized_pnl": p.realized_pnl,
                         "cash_style": inst.cash_style})
        return pd.DataFrame(rows).set_index("instrument_id") if rows else pd.DataFrame(columns=["asset_class", "type", "currency", "quantity", "avg_price", "mark", "notional_base"])

    def gross_exposure(self, marks: Mapping[str, float]) -> float:
        e = self.exposures(marks)
        return float(e["notional_base"].sum()) if len(e) else 0.0

    def net_exposure(self, marks: Mapping[str, float]) -> float:
        e = self.exposures(marks)
        return float(e["signed_notional_base"].sum()) if len(e) else 0.0

    def snapshot(self, ts, marks: Mapping[str, float] | None = None) -> dict:
        d = {"ts": pd.Timestamp(ts), "equity": self.equity(), "cash_base": self.cash_base(), "n_positions": len(self.positions), "journal_entries": len(self.journal)}
        if marks is not None:
            d["gross_exposure"] = self.gross_exposure(marks)
            d["net_exposure"] = self.net_exposure(marks)
        return d

    def journal_frame(self) -> pd.DataFrame:
        return to_frame(self.journal)

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"ts": f.ts, "instrument_id": f.instrument_id, "quantity": f.quantity, "price": f.price, "fee": f.fee, "spread_price": f.spread_price, "impact_price": f.impact_price,
                              "tags": ",".join(f.tags), "strategy": f.strategy, "order_id": f.order_id} for f in self.fills])

    def pnl_by(self, key: str = "category") -> pd.Series:
        """Total P&L in the base currency grouped by ``category``, ``instrument_id``, ``asset_class``, ``strategy`` or ``tags``."""
        j = self.journal_frame()
        return j.groupby(key)["pnl"].sum() if len(j) else pd.Series(dtype=float)
