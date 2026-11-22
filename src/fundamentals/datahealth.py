"""One screen of facts per company about whether its fundamentals can be trusted for research today.

Every field is counted from the database, none is estimated: the latest filing and latest quarter held, the periods missing between the first and the last held, source conflicts, observations the scanner
found that nobody confirmed, restated periods, how many of the inputs a standard factor set needs are available, when a provider was last asked and whether it answered.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from . import canonical, sources

# the lines the common factors stand on; readiness is the share of them held (research-ready) for the latest annual period
FACTOR_INPUTS = ("revenue", "net_income", "operating_income", "total_assets", "total_liabilities", "shareholders_equity", "operating_cash_flow", "shares_outstanding", "cash_and_equivalents", "total_debt")


def _missing_years(ends: list[date]) -> list[str]:
    if len(ends) < 2:
        return []
    ys = sorted({e.year for e in ends})
    have = set(ys)
    return [str(y) for y in range(ys[0], ys[-1] + 1) if y not in have]


def company(store, security_id: int, _bulk: dict | None = None) -> dict:
    sid = int(security_id)
    if _bulk is None:
        periods, obs = store.periods([sid]), store.observations([sid])
    else:
        periods = _bulk["periods"].get(sid, _bulk["empty_periods"])
        obs = _bulk["obs"].get(sid, _bulk["empty_obs"])
    out: dict = {"security_id": sid}
    with store.db.transaction() as c:
        sec = c.execute("SELECT ticker, company_name, currency FROM securities WHERE security_id = ?", (sid,)).fetchone()
        waiting = c.execute("SELECT COUNT(*) FROM raw_observations r JOIN import_items i ON i.item_id = r.item_id JOIN import_batches b ON b.batch_id = i.batch_id "
                            "WHERE b.security_id = ? AND r.status IN ('UNCONFIRMED', 'UNRESOLVED') AND i.status NOT IN ('rejected', 'duplicate')", (sid,)).fetchone()[0]
        unresolved = c.execute("SELECT COUNT(*) FROM raw_observations r JOIN import_items i ON i.item_id = r.item_id JOIN import_batches b ON b.batch_id = i.batch_id WHERE b.security_id = ? AND r.status = 'UNRESOLVED'",
                               (sid,)).fetchone()[0]
        sync = c.execute("SELECT provider, last_checked, last_success, status, error FROM sync_state WHERE security_id = ? ORDER BY last_checked DESC", (sid,)).fetchall()
        custom = c.execute("SELECT COUNT(*) FROM custom_observations WHERE security_id = ?", (sid,)).fetchone()[0]
    out.update(ticker=sec[0] if sec else None, name=sec[1] if sec else None, trading_currency=sec[2] if sec else None)
    if periods.empty:
        out.update(latest_annual=None, latest_quarter=None, missing_periods=[], restatements=0, source_conflicts=0, unconfirmed_observations=waiting, unresolved_observations=unresolved,
                   factor_readiness=0.0, factor_inputs_missing=list(FACTOR_INPUTS), custom_confirmed=custom, status="no data")
    else:
        first = periods[periods["vintage"] == 0]
        fy = first[first["fiscal_period"] == "FY"].sort_values("period_end")
        qs = first[first["fiscal_period"].isin(("Q1", "Q2", "Q3", "Q4"))].sort_values("period_end")
        latest_fy = fy.iloc[-1] if len(fy) else None
        out["latest_annual"] = None if latest_fy is None else {"period_end": str(latest_fy["period_end"])[:10], "filing_date": str(latest_fy["filing_date"])[:10] if pd.notna(latest_fy["filing_date"]) else None, "provider": latest_fy["provider"]}
        out["latest_quarter"] = None if qs.empty else {"period": qs.iloc[-1]["fiscal_period"], "period_end": str(qs.iloc[-1]["period_end"])[:10], "provider": qs.iloc[-1]["provider"]}
        out["missing_periods"] = _missing_years([pd.Timestamp(e).date() for e in fy["period_end"]])
        out["restatements"] = int((periods["vintage"] > 0).sum())
        out["source_conflicts"] = len(sources.compare(store, sid, only_conflicts=True, frames=None if _bulk is None else {"periods": periods, "obs": obs, "prefs": _bulk["prefs"], "confirmed": _bulk["confirmed"]}))
        out["unconfirmed_observations"], out["unresolved_observations"], out["custom_confirmed"] = waiting, unresolved, custom
        latest_ids = list(fy[fy["period_end"] == fy["period_end"].max()]["period_id"]) if len(fy) else []      # every source's row for the latest annual period
        usable = obs[obs["period_id"].isin(latest_ids) & ~obs["quality_flag"].isin(("unverified", "identity_error", "negative_value", "suspicious_jump", "provider_disagreement"))]
        held = set(usable["metric_code"])
        out["factor_inputs_missing"] = [m for m in FACTOR_INPUTS if m not in held and m in canonical.METRICS]
        out["factor_readiness"] = round(1 - len(out["factor_inputs_missing"]) / len(FACTOR_INPUTS), 3)
        flags = []
        if out["missing_periods"]:
            flags.append("missing years")
        if out["source_conflicts"]:
            flags.append("source conflicts")
        if waiting:
            flags.append("unconfirmed observations")
        if out["factor_readiness"] < 0.7:
            flags.append("low factor readiness")
        out["status"] = "ok" if not flags else "attention: " + ", ".join(flags)
    out["providers"] = [{"provider": r[0], "last_checked": r[1], "last_success": r[2], "status": r[3], "error": r[4]} for r in sync]
    out["last_provider_refresh"] = max((r[2] for r in sync if r[2]), default=None)
    return out


def overview(store, security_ids: list[int] | None = None) -> dict:
    ids = security_ids if security_ids is not None else [int(i) for i in store.securities()["security_id"]]
    periods, obs = store.periods(ids), store.observations(ids)                                      # loaded once for every company: one query, not one per company
    with store.db.transaction() as c:
        confirmed = {(r[0], r[1]) for r in c.execute("SELECT financial_period_id, metric_code FROM observation_provenance WHERE verified = 1").fetchall()}
    bulk = {"periods": {k: v for k, v in periods.groupby("security_id")}, "obs": {k: v for k, v in obs.groupby("security_id")}, "empty_periods": periods.iloc[0:0], "empty_obs": obs.iloc[0:0],
            "prefs": store.source_preferences(ids), "confirmed": confirmed}
    rows = [company(store, i, bulk) for i in ids]
    return {"companies": rows, "n": len(rows), "needing_attention": sum(1 for r in rows if r["status"] != "ok"), "unconfirmed_total": sum(r["unconfirmed_observations"] for r in rows)}
