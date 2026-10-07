"""Analysis of an engine run: equity and returns, the journal, attribution by any key, a tear sheet and comparison across runs.

:class:`BacktestResult` holds everything a run produced: the equity curve (one row per daily mark, with exposures and margin), the ledger journal (every cash and P&L movement,
explained), the fills, the orders, the diagnostics (stale marks, rejected orders, margin calls, missing funding...), the replay digest and the reconciliation report. Attribution
works on the journal, so the P&L of a run can be cut by category (price, costs, funding, coupons, settlement...), by instrument, asset class, strategy or tag (for example the cost of
rolling futures is the fees and spreads on fills tagged ``roll``) and always adds up to the change in equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..backtest.metrics import performance_summary

COST_CATEGORIES = ("spread", "impact", "slippage", "fee")
FINANCING_CATEGORIES = ("funding", "borrow", "interest", "financing")
PRICE_CATEGORIES = ("mtm", "variation_margin")


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    journal: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    diagnostics: dict
    digest: str
    reconciliation: object
    start_equity: float
    events: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @classmethod
    def from_engine(cls, engine) -> "BacktestResult":
        eq = pd.DataFrame(engine.equity_rows)
        if len(eq):
            eq = eq.drop_duplicates("ts", keep="last").set_index("ts")
        orders = pd.DataFrame([{"id": o.id, "ts": o.submitted_at, "instrument_id": o.instrument_id, "quantity": o.quantity, "type": o.type, "status": o.status, "filled": o.filled,
                                "avg_price": o.avg_price, "fees": o.fees, "strategy": o.strategy, "tags": ",".join(o.tags), "reason": o.reason} for o in engine.orders.values()])
        return cls(eq, engine.ledger.journal_frame(), engine.ledger.fills_frame(), orders, dict(engine.diagnostics), engine.log.digest, engine.reconciliation(), engine.start_equity,
                   list(engine.instrument_events), {"start": str(engine.start), "end": str(engine.end), "base": engine.config.base_currency})

    # ---------------------------------------------------------------------------------------------------------------------------- performance
    @property
    def equity_curve(self) -> pd.Series:
        return self.equity["equity"]

    def returns(self, freq: str = "D") -> pd.Series:
        """Simple returns of the equity curve (one per day by default), with external transfers removed."""
        eq = self.equity_curve
        if freq != "D":
            eq = eq.resample(freq).last().dropna()
        else:
            eq = eq.groupby(eq.index.normalize()).last()
        return eq.pct_change().dropna()

    def summary(self, periods_per_year: int = 252) -> dict:
        r = self.returns()
        out = performance_summary(r, periods_per_year=periods_per_year) if len(r) > 2 else {}
        out["final_equity"] = float(self.equity_curve.iloc[-1])
        out["total_pnl"] = float(self.equity_curve.iloc[-1] - self.start_equity)
        out["n_fills"] = int(len(self.fills))
        out["reconciled"] = bool(self.reconciliation.ok)
        return out

    # ------------------------------------------------------------------------------------------------------------------------- attribution
    def attribution(self, by: str = "category") -> pd.Series:
        """Total P&L (base currency) grouped by a journal column: ``category``, ``instrument_id``, ``asset_class``, ``strategy``, ``currency`` or ``tags``."""
        j = self.journal
        if j.empty:
            return pd.Series(dtype=float)
        j = j[j["category"] != "transfer"]
        return j.groupby(by)["pnl"].sum().sort_values(ascending=False)

    def pnl_matrix(self, rows: str = "asset_class", cols: str = "category") -> pd.DataFrame:
        j = self.journal
        j = j[j["category"] != "transfer"]
        return j.pivot_table(index=rows, columns=cols, values="pnl", aggfunc="sum", fill_value=0.0)

    def cost_summary(self) -> pd.Series:
        j = self.journal
        out = {c: -float(j.loc[j["category"] == c, "pnl"].sum()) for c in COST_CATEGORIES}
        out["roll_costs"] = -float(j.loc[j["category"].isin(COST_CATEGORIES) & j["tags"].str.contains("roll"), "pnl"].sum())
        out["financing_and_funding"] = -float(j.loc[j["category"].isin(FINANCING_CATEGORIES), "pnl"].sum())
        out["total_costs"] = sum(out[c] for c in COST_CATEGORIES)
        return pd.Series(out)

    def pnl_by_period(self, freq: str = "ME") -> pd.DataFrame:
        j = self.journal[self.journal["category"] != "transfer"].copy()
        j["period"] = pd.to_datetime(j["ts"]).dt.to_period(freq[0] if freq in ("ME", "MS") else freq)
        return j.pivot_table(index="period", columns="category", values="pnl", aggfunc="sum", fill_value=0.0)

    def exposure(self) -> pd.DataFrame:
        cols = [c for c in ("gross_exposure", "net_exposure", "margin_utilization", "drawdown") if c in self.equity.columns]
        return self.equity[cols]

    # ------------------------------------------------------------------------------------------------------------------------ reporting
    def tearsheet(self) -> str:
        s = self.summary()
        lines = ["# Backtest tear sheet", "", f"Period {self.config.get('start')} to {self.config.get('end')}; base currency {self.config.get('base')}; replay digest `{self.digest[:16]}`.", "",
                 "## Performance", ""]
        for k in ("final_equity", "total_pnl", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate"):
            if k in s:
                v = s[k]
                lines.append(f"- {k}: {v:,.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines += ["", "## P&L by category (base currency)", "", self.attribution("category").round(2).to_frame("pnl").to_markdown(), "",
                  "## P&L by asset class", "", self.attribution("asset_class").round(2).to_frame("pnl").to_markdown(), "",
                  "## Costs", "", self.cost_summary().round(2).to_frame("amount").to_markdown(), "",
                  "## Reconciliation", "", f"- ok: {self.reconciliation.ok}", f"- journal entries: {len(self.journal)}", f"- fills: {len(self.fills)}"]
        if self.diagnostics:
            lines += ["", "## Diagnostics", ""] + [f"- {k}: {v}" for k, v in sorted(self.diagnostics.items())]
        return "\n".join(lines)


def compare(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """A table of headline statistics, one column per run."""
    cols = {}
    for name, r in results.items():
        s = r.summary()
        c = r.cost_summary()
        cols[name] = {"final_equity": s.get("final_equity"), "cagr": s.get("cagr"), "ann_vol": s.get("ann_vol"), "sharpe": s.get("sharpe"), "max_drawdown": s.get("max_drawdown"),
                      "total_costs": c["total_costs"], "financing_and_funding": c["financing_and_funding"], "fills": s.get("n_fills"), "digest": r.digest[:12]}
    return pd.DataFrame(cols)
