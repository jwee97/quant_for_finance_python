"""Reconciliation: independent recomputation of what the journal says, and the invariants that must hold.

``reconcile`` recomputes cash and equity from the journal and compares them with the ledger's own state:

* **cash conservation**: for every currency, ``balance = starting cash + sum of journal cash movements``;
* **equity conservation**: ``equity - starting equity = sum of (pnl + transfer)`` over the journal (nothing is created or destroyed without an entry);
* **per-entry identity**: re-checked for every entry at the final exchange rates;
* **P&L attribution**: the P&L categories sum to total P&L (trivially, but it is the table an analyst reads);
* **position conservation**: each open position's quantity equals the sum of its fills plus the lifecycle and corporate-action adjustments recorded by the engine (``expected_quantities``);
* **no expired contracts held**: no open position in an instrument that has expired by ``ts``.

It returns a :class:`ReconciliationReport` with the numeric gaps; ``assert_reconciled`` raises with the failing check named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd


@dataclass
class ReconciliationReport:
    cash_gap: dict = field(default_factory=dict)
    equity_gap: float = 0.0
    identity_violations: int = 0
    position_gaps: dict = field(default_factory=dict)
    expired_held: list = field(default_factory=list)
    total_pnl: float = 0.0
    pnl_by_category: dict = field(default_factory=dict)
    tolerance: float = 1e-6

    @property
    def ok(self) -> bool:
        scale = max(1.0, abs(self.total_pnl))
        return (all(abs(g) <= self.tolerance * max(1.0, scale) for g in self.cash_gap.values()) and abs(self.equity_gap) <= self.tolerance * scale and self.identity_violations == 0
                and all(abs(g) <= 1e-9 for g in self.position_gaps.values()) and not self.expired_held)

    def failures(self) -> list[str]:
        out = []
        scale = max(1.0, abs(self.total_pnl))
        for c, g in self.cash_gap.items():
            if abs(g) > self.tolerance * max(1.0, scale):
                out.append(f"cash[{c}] differs from the journal by {g:.6g}")
        if abs(self.equity_gap) > self.tolerance * scale:
            out.append(f"equity differs from starting equity + journal P&L by {self.equity_gap:.6g}")
        if self.identity_violations:
            out.append(f"{self.identity_violations} journal entries break the per-entry identity")
        out += [f"position {i} differs from its fills by {g:.6g}" for i, g in self.position_gaps.items() if abs(g) > 1e-9]
        out += [f"expired contract still held: {i}" for i in self.expired_held]
        return out


def reconcile(ledger, start_equity: float | None = None, expected_quantities: Mapping[str, float] | None = None, ts=None, tolerance: float = 1e-6) -> ReconciliationReport:
    rep = ReconciliationReport(tolerance=tolerance)
    journal = ledger.journal
    cash_from_journal: dict[str, float] = {}
    for e in journal:
        for c, a in e.cash.items():
            cash_from_journal[c] = cash_from_journal.get(c, 0.0) + a
        lhs = e.cash_base + e.value + e.translation
        if abs(lhs - (e.pnl + e.transfer)) > tolerance * max(1.0, abs(lhs), abs(e.pnl)):
            rep.identity_violations += 1
    for c in set(ledger.cash) | set(cash_from_journal):
        rep.cash_gap[c] = ledger.cash.get(c, 0.0) - cash_from_journal.get(c, 0.0)
    start = 0.0 if start_equity is None else start_equity
    total_pnl = sum(e.pnl for e in journal)
    transfers = sum(e.transfer for e in journal)
    rep.total_pnl = float(total_pnl)
    rep.equity_gap = float(ledger.equity() - (start + total_pnl + transfers))
    cats: dict[str, float] = {}
    for e in journal:
        cats[e.category] = cats.get(e.category, 0.0) + e.pnl
    rep.pnl_by_category = cats
    if expected_quantities is not None:
        ids = set(expected_quantities) | set(ledger.positions)
        for i in ids:
            rep.position_gaps[i] = ledger.quantity(i) - expected_quantities.get(i, 0.0)
    if ts is not None:
        for iid, p in ledger.positions.items():
            inst = ledger._inst(iid)
            if inst.expiry is not None and pd.Timestamp(ts) > inst.expiry + pd.Timedelta(days=1) and not p.is_flat:
                rep.expired_held.append(iid)
    return rep


def assert_reconciled(ledger, **kwargs) -> ReconciliationReport:
    rep = reconcile(ledger, **kwargs)
    if not rep.ok:
        raise AssertionError("ledger does not reconcile: " + "; ".join(rep.failures()))
    return rep
