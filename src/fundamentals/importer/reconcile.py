"""Cross-checks on an imported draft that look beyond a single period: against the company's other periods in the draft, and against what the database already holds.

:func:`manual.check_entry` already tests each period by itself (assets against liabilities plus equity, gross profit, free cash flow, earnings per share times shares, revenue against assets). What a
single period cannot show, and what goes wrong when reports are imported in bulk, is checked here, each as an issue with a level (``error`` blocks the save, ``warning`` asks for a decision, ``info``
informs) and, where it concerns a stored figure, a *conflict* record so the review page can show the two numbers side by side:

* **Cash flow ties to cash.** Operating + investing + financing cash flow should equal the change in cash between two consecutive balance sheets of the draft (currency effects explain a small gap).
* **Quarters add up to the year.** When a draft holds the four quarters and the year, the flows of the quarters should sum to the year's: a year-to-date column mistaken for a quarter shows here.
* **The year to date is not a quarter.** Within one fiscal year, a "quarter" whose flows are larger than the following quarter's by about the sum of those before is year-to-date.
* **Duplicates.** The same period already in the database with the same figures is a duplicate (nothing to store); with different figures it is a *conflict* and, when the import is of a later filing than the stored
  one, a candidate *restatement*, stored as a new vintage and never over the old one.
* **Units against history.** A figure a hundred times the company's stored figure for a neighbouring period is almost always a unit mistake (thousands read as units).
* **Currency.** A draft in another currency than the company's stored one is flagged.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .. import canonical

def pd_notna(x) -> bool:
    return x is not None and not pd.isna(x)


SAME = 0.005                                                                                     # figures within half a per cent are the same figure (rounding in a report's own units)
FLOWS = [c for c, m in canonical.METRICS.items() if m.kind == "flow"]


def _scale(draft: dict) -> float:
    return {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}.get((draft.get("units") or {}).get("money", "units"), 1.0)


def _close(a: float, b: float, tolerance: float = SAME) -> bool:
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1.0)


def _issue(level: str, message: str, period: str | None = None, field: str | None = None, kind: str = "check") -> dict:
    return {"level": level, "message": message, "period": period, "field": field, "kind": kind}


def within_draft(draft: dict) -> list[dict]:
    """The checks that need two or more periods of the draft itself."""
    out: list[dict] = []
    periods = sorted(draft.get("periods", []), key=lambda p: p["period_end"] or "")
    scale = _scale(draft)
    # cash flow against the change in cash, for consecutive balance sheets
    for a, b in zip(periods, periods[1:]):
        va, vb = a["values"], b["values"]
        if not all(k in vb for k in ("operating_cash_flow", "investing_cash_flow", "financing_cash_flow")) or "cash_and_equivalents" not in va or "cash_and_equivalents" not in vb:
            continue
        gap_days = (date.fromisoformat(b["period_end"]) - date.fromisoformat(a["period_end"])).days
        if not (60 <= gap_days <= 400):
            continue
        net = vb["operating_cash_flow"] + vb["investing_cash_flow"] + vb["financing_cash_flow"]
        change = vb["cash_and_equivalents"] - va["cash_and_equivalents"]
        if not _close(net, change, 0.05) and abs(net - change) > 0.02 * max(abs(vb["cash_and_equivalents"]), 1.0):
            out.append(_issue("warning", f"operating + investing + financing cash flow is {net * scale:,.0f} but cash changed by {change * scale:,.0f} between {a['period_end']} and {b['period_end']}: a line is missing, "
                                         "or one of the cash flow figures is year-to-date rather than for the period (currency effects explain a small gap, not this one).", f"{b['fiscal_period']} to {b['period_end']}", "operating_cash_flow", "cash_tie"))
    # quarters against the year
    by_year: dict[int, dict] = {}
    for p in periods:
        by_year.setdefault(p.get("fiscal_year") or int(p["period_end"][:4]), {})[p["fiscal_period"]] = p
    for year, held in by_year.items():
        if "FY" in held and all(q in held for q in ("Q1", "Q2", "Q3", "Q4")):
            for code in FLOWS:
                if code in held["FY"]["values"] and all(code in held[q]["values"] for q in ("Q1", "Q2", "Q3", "Q4")):
                    total = sum(held[q]["values"][code] for q in ("Q1", "Q2", "Q3", "Q4"))
                    if not _close(total, held["FY"]["values"][code], 0.02):
                        out.append(_issue("warning", f"the four quarters of {year} add up to {total * scale:,.0f} for {canonical.METRICS[code].label.lower()} but the year is {held['FY']['values'][code] * scale:,.0f}: one of the "
                                                     "quarters may be a year-to-date figure, or a different fiscal year.", f"FY {year}", code, "quarters_vs_year"))
                        break
        qs = [held[q] for q in ("Q1", "Q2", "Q3") if q in held]
        if len(qs) >= 2:                                                                       # a growing run of cash flows that matches a running total is year-to-date
            for code in ("operating_cash_flow", "capital_expenditure"):
                run = [q["values"].get(code) for q in qs]
                if all(v is not None and v > 0 for v in run) and all(run[i + 1] > run[i] * 1.6 for i in range(len(run) - 1)) and len(run) >= 2:
                    out.append(_issue("warning", f"{canonical.METRICS[code].label.lower()} grows through the quarters of {year} the way a running total does ({', '.join(f'{v * scale:,.0f}' for v in run)}): "
                                                 "a 10-Q reports it year to date. If these are year-to-date figures, subtract the earlier quarters before approving.", f"FY {year}", code, "ytd_as_quarter"))
                    break
    return out


def against_database(store, security_id: int | None, draft: dict, filing_dates: dict | None = None) -> tuple[list[dict], list[dict]]:
    """``(issues, conflicts)`` of the draft's periods against the company's stored ones. A company that is not in the database yet has nothing to conflict with."""
    issues: list[dict] = []
    conflicts: list[dict] = []
    if security_id is None:
        return issues, conflicts
    scale = _scale(draft)
    periods = store.periods([security_id])
    obs = store.observations([security_id])
    if periods.empty or obs.empty:
        return issues, conflicts
    stored_currency = periods["reporting_currency"].dropna().mode()
    cur = draft.get("currency")
    if cur and len(stored_currency) and str(stored_currency.iloc[0]).upper() != str(cur).upper():
        issues.append(_issue("warning", f"the draft is in {cur} but the company's stored statements are in {stored_currency.iloc[0]}: figures in two currencies must not be mixed in one series.", None, None, "currency"))
    for p in draft.get("periods", []):
        label = f"{p['fiscal_period']} to {p['period_end']}"
        same = periods[(periods["fiscal_period"] == p["fiscal_period"]) & (periods["period_end"].dt.strftime("%Y-%m-%d") == str(p["period_end"])[:10])]
        for _, row in same.sort_values("vintage").groupby("provider").tail(1).iterrows():
            held = obs[obs["period_id"] == row["period_id"]]
            stored = {r["metric_code"]: float(r["value"]) for _, r in held.iterrows()}
            both = [c for c in p["values"] if c in stored]
            if not both:
                continue
            diffs = [(c, p["values"][c] * scale, stored[c]) for c in both if not _close(p["values"][c] * scale, stored[c])]
            if not diffs:
                issues.append(_issue("info", f"{label} is already in the database from {row['provider']} with the same figures ({len(both)} compared): importing it adds nothing for those lines.", label, None, "duplicate"))
                continue
            later = bool(filing_dates and filing_dates.get(label) and pd_notna(row.get("filing_date")) and str(filing_dates[label]) > str(row["filing_date"])[:10])
            kind = "restatement" if later else "source_conflict"
            for c, new, old in diffs:
                conflicts.append({"period": label, "metric": c, "imported": new, "stored": old, "stored_provider": row["provider"], "stored_filing_date": str(row["filing_date"])[:10] if pd_notna(row.get("filing_date")) else None, "kind": kind,
                                  "difference": (new - old) / abs(old) if old else None})
            issues.append(_issue("warning", f"{label} differs from what {row['provider']} holds in {len(diffs)} of {len(both)} lines (for example {canonical.METRICS[diffs[0][0]].label.lower()}: {diffs[0][1]:,.0f} against {diffs[0][2]:,.0f}). "
                                            + ("The imported filing is later than the stored one: saved as a restatement, the old figure stays as the earlier vintage." if later else "Decide which source to believe: the old figures are never overwritten."), label, diffs[0][0], kind))
        # units against the neighbouring history: a hundredfold gap is a unit mistake
        for code in ("total_assets", "revenue"):
            if code not in p["values"]:
                continue
            hist = obs[(obs["metric_code"] == code)]
            if hist.empty:
                continue
            ratio = (p["values"][code] * scale) / max(abs(float(hist["value"].median())), 1.0)
            if ratio >= 100 or (0 < ratio <= 0.01):
                issues.append(_issue("error", f"{canonical.METRICS[code].label.lower()} is {ratio:,.3g} times the company's stored median: the unit is probably wrong (thousands read as units, or the reverse). Choose the unit of the report and check again.",
                                     label, code, "units_vs_history"))
                break
    return issues, conflicts


def check(store, security_id: int | None, draft: dict, filing_dates: dict | None = None) -> dict:
    issues = within_draft(draft)
    more, conflicts = against_database(store, security_id, draft, filing_dates)
    return {"issues": issues + more, "conflicts": conflicts}
