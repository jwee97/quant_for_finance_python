"""How a derived figure was calculated: the formula, the input observations (with their source), the fiscal periods used, the currency, what was missing and whether the calculation is applicable.

``explain`` re-evaluates one metric on one row with a recording context, so the inputs listed are exactly the ones the formula read (never a hand-written list that could drift from the code). Nothing is
estimated or filled: an input that is absent is listed as missing and the result is missing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import canonical, metrics
from .schema import now_iso


class _Recording(metrics.Ctx):
    def __init__(self, tables, basis):
        super().__init__(tables, basis)
        self.read: set[str] = set()
        self.read_prior: set[str] = set()

    def __getitem__(self, name):
        self.read.add(name)
        return super().__getitem__(name)

    def prior(self, name, years=1):
        self.read_prior.add(name)
        return super().prior(name, years)


def explain(store, tables, security_id: int, metric_name: str, period_end: str, basis: str = "fy") -> dict:
    """The calculation of ``metric_name`` for the company's row ending ``period_end``: formula, inputs, provenance, periods, warnings. Raises ``KeyError`` for an unknown metric."""
    from . import analytics, scores  # noqa: F401  (register the scores)
    d = next((x for x in metrics.DEFINITIONS if x.name == metric_name), None)
    if d is None:
        raise KeyError(metric_name)
    table = tables.table(basis)
    sel = table[(table["security_id"] == security_id) & (pd.to_datetime(table["period_end"]) == pd.Timestamp(period_end))]
    if sel.empty:
        return {"metric": metric_name, "applicable": False, "reason": f"no {basis.upper()} row for this company ending {period_end}", "calculated_at": now_iso()}
    metrics.compute(tables, basis)                                                       # stage-2 metrics read stage-1 columns that compute() attaches
    ctx = _Recording(tables, basis)
    value = pd.Series(np.asarray(d.formula(ctx), dtype=float), index=table.index).loc[sel.index[0]]
    row = sel.iloc[0]
    prior_row = tables.prior(basis).loc[sel.index[0]] if hasattr(tables, "prior") else None
    inputs = []
    for name in sorted(ctx.read | ctx.read_prior):
        cur = row.get(name, np.nan)
        pri = prior_row.get(name, np.nan) if prior_row is not None and name in prior_row.index else np.nan
        meaning = canonical.METRICS.get(name)
        inputs.append({"input": name, "label": meaning.label if meaning else name, "kind": "canonical observation" if meaning else "derived metric", "current": None if pd.isna(cur) else float(cur),
                       "prior": None if pd.isna(pri) else float(pri), "used_current": name in ctx.read, "used_prior": name in ctx.read_prior, "missing_current": name in ctx.read and bool(pd.isna(cur)),
                       "missing_prior": name in ctx.read_prior and bool(pd.isna(pri))})
    with store.db.transaction() as c:
        prov = c.execute("SELECT v.metric_code, v.source_kind, v.source_ref, v.accession, v.page, v.extraction_method, v.verified, v.currency, v.reported_unit, v.edited FROM observation_provenance v "
                         "JOIN financial_periods p ON p.id = v.financial_period_id WHERE p.security_id = ? AND p.period_end = ?", (int(security_id), period_end)).fetchall()
        ccy = {r[0] for r in c.execute("SELECT DISTINCT reporting_currency FROM financial_periods WHERE security_id = ? AND period_end = ? AND reporting_currency IS NOT NULL", (int(security_id), period_end)).fetchall()}
    by = {r[0]: dict(zip(("metric_code", "source_kind", "source_ref", "accession", "page", "extraction_method", "verified", "currency", "reported_unit", "edited"), r)) for r in prov}
    for i in inputs:
        i["source"] = by.get(i["input"]) or {"source_kind": "provider or derived", "verified": None}
    warnings = []
    if len(ccy) > 1:
        warnings.append(f"the period is held in more than one currency ({', '.join(sorted(ccy))}): inputs in another currency are not mixed in, check the result")
    missing = [i["input"] for i in inputs if i["missing_current"] or i["missing_prior"]]
    if missing:
        warnings.append("missing inputs: " + ", ".join(missing))
    finite = value is not None and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
    return {"metric": metric_name, "label": d.label, "family": d.family, "unit": d.unit, "formula": d.text, "basis": basis, "period_end": period_end, "value": float(value) if finite else None,
            "applicable": bool(finite), "reason": None if finite else ("an input is missing, or the denominator is not positive: the figure is not calculated, never filled in" if missing or True else None),
            "inputs": inputs, "currency": sorted(ccy), "fiscal_periods": {"current": str(row.get("fiscal_period")), "prior": "the same period one year earlier (where an input is prior)"},
            "available_date": str(row.get("available_date"))[:10] if "available_date" in row.index else None, "calculation_version": metrics.VERSION, "calculated_at": now_iso(), "warnings": warnings}


def alpha_inputs(store, fields: list[str], security_ids: list[int] | None = None) -> list[dict]:
    """What a factor stands on, field by field, before it is run: how many companies and periods hold the field, where the figures came from (a provider, or a person's confirmation), how many were flagged
    out of research (``quality_flag``), and how many scanner readings of that line are waiting in the import archive **unconfirmed** (they are not in any factor, which is the point)."""
    obs = store.observations(security_ids)
    periods = store.periods(security_ids)
    frame = obs.merge(periods[["period_id", "provider", "period_end", "filing_date", "pit_quality"]].rename(columns={"provider": "period_provider"}), on="period_id", how="left")
    with store.db.transaction() as c:
        waiting = dict(c.execute("SELECT COALESCE(user_metric, metric_guess), COUNT(*) FROM raw_observations WHERE status IN ('UNCONFIRMED', 'UNRESOLVED') AND COALESCE(user_metric, metric_guess) IS NOT NULL "
                                 "GROUP BY 1").fetchall())
    out = []
    for f in fields:
        g = frame[frame["metric_code"] == f]
        excluded = g[g["quality_flag"].isin(("unverified", "identity_error", "negative_value", "suspicious_jump", "provider_disagreement"))]
        usable = g.drop(excluded.index)
        out.append({"field": f, "known": f in canonical.METRICS, "securities": int(usable["security_id"].nunique()), "periods": int(len(usable)),
                    "from_confirmed_imports": int((usable["period_provider"] == "manual").sum()), "from_providers": int((usable["period_provider"] != "manual").sum()),
                    "flagged_out": int(len(excluded)), "estimated_availability": int((usable["pit_quality"] == "estimated").sum()), "unconfirmed_readings_waiting": int(waiting.get(f, 0)),
                    "status": "ok" if len(usable) else "no research-ready data"})
    return out


def valuation_inputs(ctx, security_id: int, date: str) -> dict:
    """Exactly what a market capitalisation and an enterprise value on ``date`` were built from: the unadjusted close and the day it is from, the share count and the day its filing became public, the splits
    carried across, the reporting and trading currency with the exchange rate used, each balance-sheet component in both currencies, and what makes a ratio unavailable (negative earnings or equity, a
    non-positive enterprise value). Nothing is estimated: a missing input is listed as missing."""
    sid = int(security_id)
    if sid not in ctx.ids:
        return {"security_id": sid, "available": False, "reason": "not in this research universe"}
    d = ctx.dates[ctx.dates.get_indexer([pd.Timestamp(date)], method="pad")[0]] if len(ctx.dates) else pd.Timestamp(date)
    at = lambda frame: (None if pd.isna(frame.loc[d, sid]) else float(frame.loc[d, sid]))  # noqa: E731
    raw_shares = ctx.panel("shares_outstanding").fillna(ctx.panel("weighted_avg_shares_diluted"))
    rep, trade = ctx.reporting_currency().get(sid), ctx.info.loc[sid, "currency"]
    one = pd.DataFrame(1.0, index=ctx.dates, columns=ctx.ids)
    fx = at(ctx.to_trading(one))
    comps = {k: at(ctx.panel(k)) for k in ("total_debt", "cash_and_equivalents", "cash", "minority_interest", "preferred_equity", "net_income", "shareholders_equity", "operating_income")}
    if ctx.has("temporary_equity"):
        comps["temporary_equity"] = at(ctx.panel("temporary_equity"))
    mc, ev = at(ctx.market_cap()), at(ctx.enterprise_value())
    unavailable = []
    if mc is None:
        unavailable.append("no market capitalisation: a price or a share count is missing")
    if comps["net_income"] is not None and comps["net_income"] <= 0:
        unavailable.append("net income is not positive: no meaningful P/E (the earnings yield is negative)")
    if comps["shareholders_equity"] is not None and comps["shareholders_equity"] <= 0:
        unavailable.append("shareholders' equity is not positive: no meaningful price-to-book")
    if ev is None or ev <= 0:
        unavailable.append("enterprise value is missing or not positive: no EV multiples")
    if rep and trade and rep != trade and fx is None:
        unavailable.append(f"no {rep}/{trade} exchange rate on this date: balance-sheet components cannot be put in the trading currency")
    return {"security_id": sid, "date": str(d)[:10], "available": mc is not None, "price": at(ctx.price()), "price_basis": "unadjusted close, last close within five trading days", "trading_currency": trade,
            "reporting_currency": rep, "exchange_rate_reporting_to_trading": fx, "shares_filed": at(raw_shares), "split_factor_since_filing": at(ctx.split_factor(raw_shares)), "shares_used": at(ctx.shares()),
            "market_cap": mc, "enterprise_value": ev, "components_reporting_currency": comps, "splits_found": [s for s in getattr(ctx, "splits", []) if s["security_id"] == sid],
            "unavailable_because": unavailable, "point_in_time": "every panel value is the latest figure public on or before the date (filing date plus the information lag)"}
