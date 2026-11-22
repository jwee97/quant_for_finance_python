"""Screening: filter and rank companies by factor rules, on any date, from what was public on that date.

A rule is ``{"factor": name, "op": ">" | ">=" | "<" | "<=" | "between" | "top_pct" | "bottom_pct", "value": x | [lo, hi]}``. Every excluded company says why: it **failed** a rule, the factor is **not applicable**
to its industry (a bank has no gross margin), its figure is **missing**, or it is not **investable** (size, liquidity, history). Missing and not-applicable are never treated as passing or as failing. The
same rules run over past dates give a historical screen that uses exactly what was knowable then (the panels are point in time), so a screen can be backtested without look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import factors, sectors
from .errors import FundamentalsError

OPS = (">", ">=", "<", "<=", "between", "top_pct", "bottom_pct")


def _check(rules: list[dict]) -> None:
    if not rules:
        raise FundamentalsError("a screen needs at least one rule")
    for r in rules:
        if r.get("factor") not in factors.FACTORS and not str(r.get("factor", "")).startswith("custom:"):
            raise FundamentalsError(f"no factor called {r.get('factor')!r}")
        if r.get("op") not in OPS:
            raise FundamentalsError(f"op is one of {', '.join(OPS)}")
        v = r.get("value")
        if r["op"] == "between" and not (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] <= v[1]):
            raise FundamentalsError("between takes [low, high]")
        if r["op"] in ("top_pct", "bottom_pct") and not (isinstance(v, (int, float)) and 0 < v <= 100):
            raise FundamentalsError("top_pct and bottom_pct take a percentage above 0 and up to 100")
        if r["op"] not in ("between", "top_pct", "bottom_pct") and not isinstance(v, (int, float)):
            raise FundamentalsError(f"{r['op']} takes a number")


def _pass(values: pd.Series, rule: dict, eligible: pd.Series) -> pd.Series:
    op, v = rule["op"], rule["value"]
    if op == ">":
        return values > v
    if op == ">=":
        return values >= v
    if op == "<":
        return values < v
    if op == "<=":
        return values <= v
    if op == "between":
        return (values >= v[0]) & (values <= v[1])
    pool = values[eligible & values.notna()]                                                    # percentiles are taken among the companies that have the figure and are eligible
    if pool.empty:
        return pd.Series(False, index=values.index)
    cut = pool.quantile(1 - v / 100.0) if op == "top_pct" else pool.quantile(v / 100.0)
    return values >= cut if op == "top_pct" else values <= cut


def run(ctx, rules: list[dict], as_of: str | None = None, rank_by: str | None = None, ascending: bool = False, top: int | None = None, within_sector: bool = False) -> dict:
    _check(rules)
    date = ctx.dates[ctx.dates.get_indexer([pd.Timestamp(as_of)], method="pad")[0]] if as_of else ctx.dates[-1]
    inv = ctx.investable().loc[date]
    ids = list(ctx.ids)
    status = {i: "pass" for i in ids}
    why: dict[int, list[str]] = {i: [] for i in ids}
    panels = {}
    for rule in rules:
        name = rule["factor"]
        vals = factors.compute(ctx, name).loc[date]
        panels[name] = vals
        na = sectors.applicable_mask(ctx, name) if name in factors.FACTORS else pd.Series(True, index=ids)
        sector_key = ctx.info["sector"].fillna("unknown") if within_sector else None
        ok = pd.Series(False, index=ids)
        if within_sector and rule["op"] in ("top_pct", "bottom_pct"):
            for _, grp in sector_key.groupby(sector_key):
                idx = grp.index
                ok.loc[idx] = _pass(vals.loc[idx], rule, inv.loc[idx])
        else:
            ok = _pass(vals, rule, inv)
        for i in ids:
            if status[i] != "pass":
                continue
            if not bool(inv.get(i, False)):
                status[i], why[i] = "not investable", ["outside the universe filters on this date (size, liquidity or history)"]
            elif not bool(na.get(i, True)):
                status[i], why[i] = "not applicable", [f"{name} does not apply to this company's industry"]
            elif pd.isna(vals.get(i)):
                status[i], why[i] = "missing", [f"{name} is not available on this date"]
            elif not bool(ok.get(i)):
                status[i], why[i] = "failed", [f"{name} {rule['op']} {rule['value']} not met ({vals[i]:.4g})"]
    passed = [i for i in ids if status[i] == "pass"]
    key = rank_by or rules[0]["factor"]
    rank_vals = factors.compute(ctx, key).loc[date]
    ranked = sorted(passed, key=lambda i: (np.inf if pd.isna(rank_vals.get(i)) else (rank_vals[i] if ascending else -rank_vals[i])))
    if top:
        ranked = ranked[: int(top)]
    rows = [{"security_id": int(i), "ticker": ctx.info.loc[i, "ticker"], "name": ctx.info.loc[i, "company_name"], "sector": ctx.info.loc[i, "sector"], "rank": n + 1,
             "values": {k: (None if pd.isna(p.get(i)) else float(p[i])) for k, p in panels.items()} | ({key: None if pd.isna(rank_vals.get(i)) else float(rank_vals[i])} if key not in panels else {})} for n, i in enumerate(ranked)]
    counts = {s: sum(1 for i in ids if status[i] == s) for s in ("pass", "failed", "missing", "not applicable", "not investable")}
    return {"as_of": str(date)[:10], "rules": rules, "ranked_by": key, "counts": counts, "universe": len(ids), "rows": rows,
            "excluded": [{"security_id": int(i), "status": status[i], "why": why[i][0]} for i in ids if status[i] != "pass"][:500],
            "point_in_time": "values are those public on the date plus the information lag; the same rules on a past date give that date's screen"}


def history(ctx, rules: list[dict], start: str | None = None, end: str | None = None, step: int = 1) -> dict:
    """The names passing the rules on each date (every ``step``-th grid date): who was in the screen then, with turnover between dates."""
    dates = [d for d in ctx.dates if (not start or d >= pd.Timestamp(start)) and (not end or d <= pd.Timestamp(end))][:: max(1, int(step))]
    series, prev = [], None
    for d in dates:
        names = {r["security_id"] for r in run(ctx, rules, str(d)[:10])["rows"]}
        turn = None if prev is None or not (names | prev) else len(names ^ prev) / max(len(names | prev), 1)
        series.append({"date": str(d)[:10], "count": len(names), "turnover_vs_previous": turn, "members": sorted(names)})
        prev = names
    return {"rules": rules, "dates": series}
