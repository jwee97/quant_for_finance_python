"""What the database holds, company by company, year by year and quarter by quarter, and what is wrong with it: the data behind the Data coverage page.

A cell of the grid is one fiscal year (``FY``) or one fiscal quarter of a company. Its state:

* ``held``      a statement for it is in the database (the source is kept in ``sources``);
* ``derived``   the fourth quarter, worked out as the year less the nine months (it was never filed, see :mod:`providers.sec_quarters`);
* ``missing``   between the company's first and its latest held period (or, for a year, up to the latest one that should have been filed by now) and not held: a gap a backfill or an import can fill;
* ``notyet``    later than anything held and not yet due (a filing is expected about 90 days after a quarter and 150 after a year).

Nothing here talks to a provider: it reads the database, and says what an import or a backfill could add.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta

import pandas as pd

ANNUAL_LAG = 150
QUARTER_LAG = 90


def _periods(store) -> pd.DataFrame:
    sql = ("SELECT p.security_id, p.fiscal_year, p.fiscal_period, p.period_end, p.provider, p.vintage, p.pit_quality, p.provider_record_id FROM financial_periods p")
    with store.db.transaction() as conn:
        frame = pd.read_sql_query(sql, conn)
    if frame.empty:
        return frame
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["fiscal_year"] = frame["fiscal_year"].fillna(frame["period_end"].dt.year).astype(int)
    return frame


def grid(rows: pd.DataFrame, today: date, years: int) -> dict:
    """The annual and quarterly cells of one company from its stored periods (a frame of one company's ``_periods`` rows)."""
    out = {"annual": {}, "quarters": {}, "sources": {}}
    if rows.empty:
        return out
    last_year = int(rows["fiscal_year"].max())
    first_year = int(rows["fiscal_year"].min())
    window = range(max(first_year, last_year - years + 1), last_year + 2)
    held_annual = {int(y): g for y, g in rows[rows["fiscal_period"] == "FY"].groupby("fiscal_year")}
    held_q = {(int(y), p): g for (y, p), g in rows[rows["fiscal_period"].isin(["Q1", "Q2", "Q3", "Q4"])].groupby(["fiscal_year", "fiscal_period"])}
    ends = rows["period_end"]
    for y in window:
        if y in held_annual:
            g = held_annual[y]
            out["annual"][str(y)] = {"state": "held", "sources": sorted(set(g["provider"])), "vintages": int(g["vintage"].max()) + 1, "estimated": bool((g["pit_quality"] == "estimated").any())}
        else:
            end = ends[rows["fiscal_year"] == y].max() if (rows["fiscal_year"] == y).any() else pd.Timestamp(date(y, 12, 31))
            due = (end + timedelta(days=ANNUAL_LAG)).date() <= today
            out["annual"][str(y)] = {"state": "missing" if first_year <= y <= last_year or due and y < last_year + 1 else "notyet", "sources": []}
        for q in ("Q1", "Q2", "Q3", "Q4"):
            key = (y, q)
            if key in held_q:
                g = held_q[key]
                derived = bool(g["provider_record_id"].astype(str).str.endswith("#Q4").any()) if q == "Q4" else False
                out["quarters"][f"{y}{q}"] = {"state": "derived" if derived else "held", "sources": sorted(set(g["provider"])), "estimated": bool((g["pit_quality"] == "estimated").any())}
            else:
                approx = pd.Timestamp(date(y, 3 * int(q[1]), 28))
                due = (approx + timedelta(days=QUARTER_LAG)).date() <= today
                out["quarters"][f"{y}{q}"] = {"state": "missing" if (first_year <= y <= last_year and due) else "notyet", "sources": []}
    out["sources"] = dict(Counter(rows["provider"]))
    return out


def overview(store, importer, years: int = 12, today: date | None = None) -> dict:
    today = today or date.today()
    frame = _periods(store)
    secs = store.securities()
    companies: list[dict] = []
    totals = Counter()
    by_provider: dict[str, set] = defaultdict(set)
    counts_by_provider: Counter = Counter()
    if not frame.empty:
        for sid, rows in frame.groupby("security_id"):
            g = grid(rows, today, years)
            sec = secs[secs["security_id"] == sid]
            ann = [c for c in g["annual"].values() if c["state"] in ("held", "missing")]
            qs = [c for c in g["quarters"].values() if c["state"] in ("held", "derived", "missing")]
            a_held, a_missing = sum(c["state"] == "held" for c in ann), sum(c["state"] == "missing" for c in ann)
            q_held, q_missing = sum(c["state"] in ("held", "derived") for c in qs), sum(c["state"] == "missing" for c in qs)
            companies.append({"security_id": int(sid), "ticker": None if sec.empty else sec.iloc[0]["ticker"], "name": None if sec.empty else sec.iloc[0]["company_name"],
                              "country": None if sec.empty else sec.iloc[0]["country"], "annual": g["annual"], "quarters": g["quarters"], "annual_held": a_held, "annual_missing": a_missing, "quarter_held": q_held,
                              "quarter_missing": q_missing, "completeness": (a_held + q_held) / max(a_held + a_missing + q_held + q_missing, 1), "sources": g["sources"],
                              "estimated": sum(1 for c in list(g["annual"].values()) + list(g["quarters"].values()) if c.get("estimated"))})
            totals["annual_held"] += a_held
            totals["annual_missing"] += a_missing
            totals["quarter_held"] += q_held
            totals["quarter_missing"] += q_missing
            for prov, n in g["sources"].items():
                by_provider[prov].add(int(sid))
                counts_by_provider[prov] += n
    companies.sort(key=lambda c: (c["completeness"], c["ticker"] or ""))
    hist = importer.history(40)
    pending = failed = conflicted = 0
    for b in hist:
        pending += b["counts"].get("needs_review", 0) + b["counts"].get("extracted", 0)
        failed += b["counts"].get("failed", 0)
    with store.db.transaction() as c:
        rows = c.execute("SELECT draft FROM import_items WHERE status = 'needs_review' AND draft LIKE '%\"conflicts\": [{%'").fetchall()
        conflicted = len(rows)
        flagged = Counter(dict(c.execute("SELECT quality_flag, COUNT(*) FROM fundamental_observations WHERE quality_flag != 'ok' GROUP BY quality_flag").fetchall()))
        estimated = c.execute("SELECT COUNT(*) FROM financial_periods WHERE pit_quality = 'estimated'").fetchone()[0]
    return {"as_of": today.isoformat(), "years": years, "companies": companies, "totals": dict(totals),
            "sources": [{"provider": p, "companies": len(by_provider[p]), "periods": counts_by_provider[p]} for p in sorted(by_provider)],
            "problems": {"awaiting_review": pending, "failed_imports": failed, "import_conflicts": conflicted, "estimated_availability": int(estimated), "quality_flags": dict(flagged)},
            "recent_imports": hist[:10]}


def companies_with_gaps(store, periods=("annual", "quarter"), today: date | None = None) -> list[int]:
    """The companies that have at least one missing cell, for a backfill to fill (a backfill asks the SEC, so only companies with an identifier there are worth asking: the caller's provider decides)."""
    today = today or date.today()
    frame = _periods(store)
    if frame.empty:
        return []
    out = []
    for sid, rows in frame.groupby("security_id"):
        g = grid(rows, today, 30)
        if any(c["state"] == "missing" for c in g["annual"].values()) or ("quarter" in periods and any(c["state"] == "missing" for c in g["quarters"].values())):
            out.append(int(sid))
    return out
