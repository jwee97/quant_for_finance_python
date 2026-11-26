"""Two checks a fundamentals figure should pass before a study relies on it.

``pit_audit``  when could each stored period actually have been known? It reports how many availability dates come from a real filing, acceptance or publication date and how many are the conservative lag
               assumed when none is known (the point-in-time engine never uses the period end), the distribution of the lag, periods that would be known on or before the day they end (impossible, and the sign
               of a wrong date), restated figures, and a verdict on whether a factor built from these figures may enter a study: ``ready`` (every date is a real one), ``ready_with_assumptions`` (some are
               assumed, and the assumption is named) or ``blocked`` (look-ahead is possible, or nothing is available). It looks at dates only: it does not look at returns, so it cannot be tuned to a result.

``annual_vs_quarterly``  for one company, does each year's annual figure equal the sum of its quarters? Flow items (revenue, net income, operating cash flow) must; a year that does not reconcile is
               reported, with how far apart the two are, and a year with missing quarters is reported as incomplete, never as matching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pit import PitSettings, availability
from .store import FundamentalsStore

FLOWS = ("revenue", "net_income", "operating_cash_flow")
TOLERANCE = 0.01                                    # the quarters may differ from the year by 1% (rounding, restated quarters) before the year is called not reconciled
MAX_ESTIMATED_SHARE = 0.5                           # beyond this share of assumed dates the audit says so in its verdict


def pit_audit(store: FundamentalsStore, settings: PitSettings | None = None, security_ids: list[int] | None = None) -> dict:
    settings = settings or PitSettings()
    periods = store.periods(security_ids)
    periods = periods[periods["vintage"] == 0].copy() if len(periods) else periods
    out = {"settings": settings.describe(), "periods": int(len(periods))}
    reasons: list[str] = []
    assumptions: list[str] = []
    if periods.empty:
        out.update({"companies": 0, "verdict": {"status": "blocked", "reasons": ["no financial periods are stored for these companies"]}, "assumptions": [], "by_period": [], "lag_days": None,
                    "actual_share": None, "estimated": 0, "impossible": 0, "restated": 0, "worst_companies": []})
        return out
    periods["available"], periods["estimated"] = availability(periods, settings)
    periods["lag"] = (periods["available"] - periods["period_end"]).dt.days
    impossible = periods["lag"] <= 0
    est = periods["estimated"].astype(bool)
    restated = periods["pit_quality"].eq("restated")
    lag = periods.loc[~est, "lag"]
    by = []
    for fp, g in periods.groupby("fiscal_period"):
        e = g["estimated"].astype(bool)
        known = g.loc[~e, "lag"]
        by.append({"fiscal_period": str(fp), "periods": int(len(g)), "actual_dates": int((~e).sum()), "assumed_dates": int(e.sum()), "assumed_lag_days": settings.lag_for(str(fp)),
                   "median_lag_days": float(known.median()) if len(known) else None, "longest_lag_days": int(known.max()) if len(known) else None})
    per_company = []
    for sid, g in periods.groupby("security_id"):
        e = g["estimated"].astype(bool)
        per_company.append({"security_id": int(sid), "periods": int(len(g)), "assumed_share": float(e.mean())})
    per_company.sort(key=lambda r: -r["assumed_share"])
    n_est, n_imp = int(est.sum()), int(impossible.sum())
    out.update({"companies": int(periods["security_id"].nunique()), "actual_share": float(1 - est.mean()), "estimated": n_est, "impossible": n_imp, "restated": int(restated.sum()),
                "lag_days": ({"median": float(lag.median()), "p10": float(lag.quantile(0.1)), "p90": float(lag.quantile(0.9)), "min": int(lag.min()), "max": int(lag.max())} if len(lag) else None),
                "by_period": by, "worst_companies": per_company[:10]})
    if settings.naive_availability:
        reasons.append("the naive availability switch is on: every figure is treated as known on its period end, which is look-ahead by construction (it exists to measure how much look-ahead would flatter a result)")
    if n_imp:
        reasons.append(f"{n_imp} period{'s are' if n_imp != 1 else ' is'} available on or before the day {'they end' if n_imp != 1 else 'it ends'}: a filing date earlier than the period it reports is a wrong date")
    if reasons:
        status = "blocked"
    elif n_est:
        status = "ready_with_assumptions"
        if settings.use_estimated:
            assumptions.append(f"{n_est} of {len(periods)} periods ({n_est / len(periods):.0%}) have no filing, acceptance or publication date; they are assumed public after the lag for their kind of period "
                               f"({', '.join(f'{k} {v} days' for k, v in settings.estimated_lag_days)}) plus {settings.info_lag_days} day. A company that filed later than the assumed lag is seen early by this much.")
        else:
            assumptions.append(f"{n_est} periods without a real date are left out; the study sees fewer companies and years, not earlier figures.")
        if n_est / len(periods) > MAX_ESTIMATED_SHARE:
            assumptions.append("more than half of the dates are assumed: the timing of any result rests mostly on the assumed lag.")
    else:
        status = "ready"
    if int(restated.sum()):
        assumptions.append(f"{int(restated.sum())} periods are restatements; each is available from the day it was first seen, not from the original filing, so a restated number never enters a date before it was public.")
    if status == "ready":
        reasons.append("every availability date comes from a real filing, acceptance or publication date")
    out["assumptions"] = assumptions
    out["verdict"] = {"status": status, "reasons": reasons}
    return out


def annual_vs_quarterly(store: FundamentalsStore, security_id: int, metrics: tuple[str, ...] = FLOWS, tolerance: float = TOLERANCE) -> dict:
    periods = store.periods([security_id])
    if periods.empty:
        return {"security_id": int(security_id), "rows": [], "summary": {"reconciles": 0, "differs": 0, "incomplete": 0, "no_quarters": 0}, "note": "no periods are stored for this company"}
    periods = periods[periods["vintage"] == 0]
    obs = store.observations([security_id], list(metrics))
    obs = obs[~obs["quality_flag"].isin(("unverified", "excluded", "bad"))] if len(obs) else obs
    val = {(int(r.period_id), r.metric_code): float(r.value) for r in obs.itertuples()}
    rows = []
    for fy, year in periods[periods["fiscal_period"] == "FY"].groupby("fiscal_year"):
        annual_period = year.sort_values("period_end").iloc[-1]
        parts = periods[(periods["fiscal_year"] == fy) & periods["fiscal_period"].isin(("Q1", "Q2", "Q3", "Q4", "H1", "H2"))]
        quarters = parts[parts["fiscal_period"].isin(("Q1", "Q2", "Q3", "Q4"))]
        halves = parts[parts["fiscal_period"].isin(("H1", "H2"))]
        use, need = (quarters, 4) if len(quarters) >= len(halves) else (halves, 2)
        for m in metrics:
            a = val.get((int(annual_period["period_id"]), m))
            have = [val.get((int(r.period_id), m)) for r in use.drop_duplicates("fiscal_period").itertuples()]
            have = [v for v in have if v is not None and np.isfinite(v)]
            row = {"fiscal_year": int(fy), "metric": m, "annual": a, "parts": len(have), "parts_needed": need, "sum_parts": float(sum(have)) if have else None, "difference": None, "relative_difference": None}
            if a is None:
                row["status"] = "no_annual"
            elif not have:
                row["status"] = "no_quarters"
            elif len(have) < need:
                row["status"] = "incomplete"
            else:
                row["difference"] = float(row["sum_parts"] - a)
                row["relative_difference"] = float(row["difference"] / abs(a)) if a else None
                row["status"] = "reconciles" if (row["relative_difference"] is not None and abs(row["relative_difference"]) <= tolerance) or (a == 0 and row["difference"] == 0) else "differs"
            rows.append(row)
    summary = {k: sum(1 for r in rows if r["status"] == k) for k in ("reconciles", "differs", "incomplete", "no_quarters", "no_annual")}
    return {"security_id": int(security_id), "rows": rows, "summary": summary, "tolerance": tolerance, "metrics": list(metrics),
            "note": "Quarters are added up and compared with the annual figure. A year with fewer than four quarters (or two half-years) is incomplete, not matching; a year that differs by more than the tolerance is "
                    "either a restatement, a year-to-date number stored as a quarter, or an extraction error, and is worth looking at before the figures are used."}
