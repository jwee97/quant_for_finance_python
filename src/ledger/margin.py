"""Margin, collateral and liquidation tests.

A :class:`MarginModel` turns the open positions into an initial and a maintenance requirement, instrument by instrument according to ``margin_type``:

``fixed``      ``initial_margin`` (maintenance) currency units per contract (futures: the exchange's performance bond)
``percent``    a fraction of the position's notional (crypto perpetuals: ``1 / max leverage``; futures quoted by percentage)
``premium``    options: long positions need nothing (the premium is paid), short ones follow ``Option.margin_requirement``
``portfolio``  a scenario scan: reprice each underlying's positions under spot and volatility shocks with a delta-gamma-vega approximation and require the worst loss, so offsetting
               positions (a hedged option book, a calendar spread) pay less than their sum
``none``       no requirement

Collateral is cash and cash-like holdings with a HAIRCUT per currency (a coin counts for less than a dollar). ``margin_state`` returns equity, the requirements, excess
collateral, utilisation and whether the account is below maintenance (a margin call); the engine decides what to liquidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd


@dataclass
class MarginState:
    equity: float
    initial_margin: float
    maintenance_margin: float
    collateral_value: float
    excess_initial: float
    excess_maintenance: float
    utilization: float
    margin_call: bool
    by_instrument: dict = field(default_factory=dict)
    collateral_by_currency: dict = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        return pd.Series({k: v for k, v in self.__dict__.items() if not isinstance(v, dict)})


@dataclass
class MarginModel:
    haircuts: dict = field(default_factory=dict)                 # currency -> haircut applied to collateral value
    spot_shocks: tuple = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
    vol_shocks: tuple = (-0.25, 0.0, 0.25)                       # relative changes in implied volatility
    maintenance_ratio: float = 0.75                              # portfolio margin: maintenance = this share of initial

    def position_requirement(self, inst, quantity: float, mark: float, underlying_price: float | None = None) -> tuple[float, float]:
        """(initial, maintenance) requirement in the instrument's currency for ONE position under the instrument-level rules."""
        mt = inst.margin_type
        if mt in ("none", "portfolio") or quantity == 0.0:
            return 0.0, 0.0
        if mt == "fixed":
            return abs(quantity) * inst.initial_margin, abs(quantity) * (inst.maintenance_margin or inst.initial_margin)
        if mt == "percent":
            n = inst.notional(mark, quantity)
            return n * inst.initial_margin, n * (inst.maintenance_margin or inst.initial_margin)
        if mt == "premium":
            if hasattr(inst, "margin_requirement"):
                u = underlying_price if underlying_price is not None else mark
                r = inst.margin_requirement(quantity, u, mark)
                return r, r * (inst.maintenance_margin / inst.initial_margin if inst.initial_margin else 1.0) if r else 0.0
        return 0.0, 0.0

    def portfolio_scan(self, groups: Mapping[str, list[dict]]) -> dict[str, float]:
        """Scenario-scan requirement per underlying. Each group is a list of position risk records with ``delta``, ``gamma``, ``vega`` (position-level, in currency per unit move
        of the underlying: delta in currency per 1.0 of spot, gamma per 1.0^2, vega per 1.00 of volatility), and ``spot`` and ``vol`` levels; the worst loss over the shock grid is
        the requirement (never negative)."""
        out = {}
        for under, recs in groups.items():
            worst = 0.0
            for s in self.spot_shocks:
                for v in self.vol_shocks:
                    pnl = 0.0
                    for r in recs:
                        dS = r["spot"] * s
                        dV = r.get("vol", 0.0) * v
                        pnl += r["delta"] * dS + 0.5 * r["gamma"] * dS * dS + r.get("vega", 0.0) * dV
                    worst = min(worst, pnl)
            out[under] = -worst
        return out

    def state(self, ledger, marks: Mapping[str, float], risk_records: Mapping[str, dict] | None = None, underlying_prices: Mapping[str, float] | None = None) -> MarginState:
        """The margin position of the ledger at ``marks``. ``risk_records`` (instrument id -> ``delta``, ``gamma``, ``vega``, ``spot``, ``vol``, ``underlying``) feeds the portfolio-scan
        instruments."""
        init = maint = 0.0
        by_inst = {}
        groups: dict[str, list[dict]] = {}
        for iid, pos in ledger.positions.items():
            inst = ledger._inst(iid)
            m = marks.get(iid, pos.last_mark)
            if inst.margin_type == "portfolio":
                rec = (risk_records or {}).get(iid)
                if rec is not None:
                    groups.setdefault(rec.get("underlying", inst.underlying_id or iid), []).append(
                        {"delta": rec["delta"] * pos.quantity, "gamma": rec["gamma"] * pos.quantity, "vega": rec.get("vega", 0.0) * pos.quantity, "spot": rec["spot"], "vol": rec.get("vol", 0.0)})
                continue
            u = (underlying_prices or {}).get(inst.underlying_id) if inst.underlying_id else None
            i_req, m_req = self.position_requirement(inst, pos.quantity, m, u)
            rate = ledger.rate(inst.currency)
            by_inst[iid] = (i_req * rate, m_req * rate)
            init += i_req * rate
            maint += m_req * rate
        for under, req in self.portfolio_scan(groups).items():
            init += req
            maint += req * self.maintenance_ratio
            by_inst[f"scan:{under}"] = (req, req * self.maintenance_ratio)
        equity = ledger.equity()
        collateral = 0.0
        by_ccy = {}
        for ccy, bal in ledger.net_assets().items():
            value = bal * ledger.rate(ccy) * (1.0 - self.haircuts.get(ccy, 0.0) if bal > 0 else 1.0)
            by_ccy[ccy] = value
            collateral += value
        util = init / equity if equity > 0 else (float("inf") if init > 0 else 0.0)
        return MarginState(equity, init, maint, collateral, equity - init, equity - maint, util, equity < maint and maint > 0, by_inst, by_ccy)
