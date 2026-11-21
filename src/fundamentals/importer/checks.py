"""Accounting reconciliations, cross-statement checks, duplicate and anomaly detection, and the confidence they feed.

Every check names the lines it used. A check that passes raises the confidence of those lines (two statements agreeing is evidence that both were read correctly); a check that fails lowers it and is
reported with the figures. Nothing here changes a value: it only says how far each one can be trusted.
"""

from __future__ import annotations

CRITICAL = ("revenue", "net_income", "total_assets", "shareholders_equity", "operating_cash_flow")
LOW_CONFIDENCE = 0.75                                  # below this a figure is stored as "unverified" and kept out of research until a person confirms it
BOOST, PENALTY = 0.08, 0.6


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def _check(cid: str, label: str, status: str, detail: str, lines=()) -> dict:
    return {"id": cid, "label": label, "status": status, "detail": detail, "lines": list(lines)}


def run(draft: dict, cross: dict | None = None, conflicts: list | None = None) -> list[dict]:
    """Run every check on every period of the draft; adjust the confidence of the lines in ``draft['periods'][i]['sources']``; return the checks."""
    out: list[dict] = []
    cross = cross or {}
    for p in draft.get("periods", []):
        v, src = p["values"], p.setdefault("sources", {})
        label = f"{p['fiscal_period']} to {p['period_end']}"

        def settle(check: dict, used):
            for code in used:
                if code in src and "confidence" in src[code]:
                    c = src[code]["confidence"]
                    src[code]["confidence"] = round(min(1.0, c + BOOST) if check["status"] == "pass" else c * PENALTY if check["status"] == "fail" else c, 3)
                    src[code].setdefault("checks", []).append({"id": check["id"], "status": check["status"]})
            check["period"] = label
            out.append(check)

        if {"total_assets", "total_liabilities", "shareholders_equity"} <= set(v):
            a, l, e = v["total_assets"], v["total_liabilities"], v["shareholders_equity"]
            ok = _rel(a, l + e) <= 0.015 or _rel(a, l + e + v.get("minority_interest", 0.0)) <= 0.015
            settle(_check("bs_identity", "Assets = liabilities + equity", "pass" if ok else "fail",
                          f"assets {a:,.0f} vs liabilities {l:,.0f} + equity {e:,.0f} = {l + e:,.0f} ({_rel(a, l + e):.1%} apart)", ("total_assets", "total_liabilities", "shareholders_equity")),
                   ("total_assets", "total_liabilities", "shareholders_equity"))
        if {"revenue", "cost_of_revenue", "gross_profit"} <= set(v):
            ok = _rel(v["gross_profit"], v["revenue"] - abs(v["cost_of_revenue"])) <= 0.01
            settle(_check("gross_profit", "Gross profit = revenue − cost of revenue", "pass" if ok else "fail",
                          f"{v['gross_profit']:,.0f} vs {v['revenue']:,.0f} − {abs(v['cost_of_revenue']):,.0f}", ("revenue", "cost_of_revenue", "gross_profit")), ("revenue", "cost_of_revenue", "gross_profit"))
        x = cross.get(p["period_end"], {}) if cross else {}
        if "net_income" in v and x.get("cf_net_income") is not None:
            ok = _rel(v["net_income"], x["cf_net_income"]) <= 0.05
            settle(_check("ni_cross", "Net income: income statement = start of the cash flow statement", "pass" if ok else "warn" if _rel(v["net_income"], x["cf_net_income"]) <= 0.35 else "fail",
                          f"{v['net_income']:,.0f} vs {x['cf_net_income']:,.0f} (a cash flow statement may start from profit before minority interests)", ("net_income",)), ("net_income",))
        cash = v.get("cash_and_equivalents", v.get("cash"))
        if cash is not None and x.get("cf_end_cash") is not None:
            ok = _rel(cash, x["cf_end_cash"]) <= 0.01
            settle(_check("cash_cross", "Cash: balance sheet = end of the cash flow statement", "pass" if ok else "warn", f"{cash:,.0f} vs {x['cf_end_cash']:,.0f}", ("cash_and_equivalents" if "cash_and_equivalents" in v else "cash",)),
                   ("cash_and_equivalents" if "cash_and_equivalents" in v else "cash",))
        bad = [c for c in ("revenue", "total_assets", "shareholders_equity") if c in v and v[c] < 0 and c != "shareholders_equity"]
        if bad:
            settle(_check("sign", "Sign of a line that cannot be negative", "fail", ", ".join(f"{c} = {v[c]:,.0f}" for c in bad), bad), bad)
        if "revenue" in v and "total_assets" in v and v["revenue"] > 0 and v["total_assets"] > 0:
            ratio = v["revenue"] / v["total_assets"]
            ok = 0.0005 < ratio < 60
            settle(_check("scale", "Revenue and assets are of a comparable scale (one unit for both statements)", "pass" if ok else "fail",
                          f"revenue / assets = {ratio:.4g}" + ("" if ok else ": a thousandfold gap usually means two statements in different units"), ("revenue", "total_assets")), ("revenue", "total_assets") if not ok else ())
        if "net_income" in v and "revenue" in v and v["revenue"] > 0 and not (-5 <= v["net_income"] / v["revenue"] <= 1.0):
            settle(_check("margin", "Net margin is plausible", "warn", f"net income / revenue = {v['net_income'] / v['revenue']:.2f}", ("net_income", "revenue")), ())
        if {"net_income", "eps_basic", "shares_outstanding"} <= set(v) and v["eps_basic"]:
            implied = v["net_income"] / v["eps_basic"]
            ok = 0.4 <= implied / v["shares_outstanding"] <= 2.5 if v["shares_outstanding"] else True
            settle(_check("eps", "Net income / EPS is close to the share count", "pass" if ok else "warn", f"implied {implied:,.0f} shares vs {v['shares_outstanding']:,.0f}", ("eps_basic", "net_income")), ("eps_basic",) if ok else ())
    for c in conflicts or []:
        out.append({"id": "duplicate_conflict", "label": "The same line is given twice with different values", "status": "fail", "detail": c["detail"], "lines": [c["code"]], "period": c["period"]})
    return out


def confidence_summary(draft: dict) -> dict:
    allv, crit, low = [], [], []
    for p in draft.get("periods", []):
        for code, s in (p.get("sources") or {}).items():
            c = s.get("confidence")
            if c is None:
                continue
            allv.append(c)
            if code in CRITICAL:
                crit.append(c)
            if c < LOW_CONFIDENCE:
                low.append({"period": f"{p['fiscal_period']} to {p['period_end']}", "code": code, "confidence": c})
    return {"mean": round(sum(allv) / len(allv), 3) if allv else None, "min_critical": min(crit) if crit else None, "low": low, "threshold": LOW_CONFIDENCE, "n": len(allv)}


__all__ = ["run", "confidence_summary", "CRITICAL", "LOW_CONFIDENCE"]
