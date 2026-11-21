"""Accounting reconciliations, cross-statement checks, duplicate and anomaly detection, the confidence they feed, and the validation verdict.

Every check has one of four outcomes (:class:`~.model.Check`): **PASS**, **FAIL**, **INSUFFICIENT_DATA** (an input line is missing: this is *not* a pass) and **NOT_APPLICABLE** (the identity does not exist
for what was extracted). A check that passes raises the confidence of the lines it used (two statements agreeing is evidence that both were read right); one that fails lowers it. Nothing here changes a value.

A reconciliation is necessary where it applies and never sufficient: several wrong figures can reconcile, so the verdict (:func:`validation`) also needs the statements found, the periods and columns
resolved with confidence, one currency and unit per statement, no unresolved conflict between candidates, and no critical line missing. Tolerances scale with the unit of the report (rounding of each
printed figure) and with the size of the amounts.
"""

from __future__ import annotations

from .model import Check, ValidationResult

CRITICAL = ("revenue", "net_income", "total_assets", "shareholders_equity", "operating_cash_flow")
CRITICAL_ANY = (("revenue",), ("net_income", "net_income_attributable"), ("total_assets",), ("shareholders_equity", "total_liabilities"), ("operating_cash_flow",))
LOW_CONFIDENCE = 0.75                                  # below this a figure is stored as "unverified" and kept out of research until a person confirms it
BOOST, PENALTY = 0.08, 0.6
REL_TOL = 0.0015


def _tol(*values, terms: int = 3) -> float:
    """Rounding tolerance: each printed figure is rounded to a unit of the report, so a sum of ``terms`` figures can be off by up to ``terms`` units; plus a small relative allowance."""
    big = max((abs(v) for v in values if v is not None), default=0.0)
    return max(terms * 1.0, REL_TOL * big)


def _res(cid: str, label: str, status: Check, detail: str = "", lines=(), period: str | None = None) -> dict:
    return ValidationResult(cid, label, status, detail, list(lines), period).to_dict()


def run(draft: dict, cross: dict | None = None, conflicts: list | None = None, document: dict | None = None) -> list[dict]:
    """Run every check on every period of the draft; adjust the confidence of the lines in ``draft['periods'][i]['sources']``; return the results (dictionaries with ``status`` in PASS / FAIL /
    INSUFFICIENT_DATA / NOT_APPLICABLE)."""
    out: list[dict] = []
    cross = cross or {}
    for p in draft.get("periods", []):                                                                 # idempotent: running again (after a correction) starts from the confidence the reading had
        for s_ in (p.get("sources") or {}).values():
            if s_.get("confidence") is not None:
                s_.setdefault("base_confidence", s_["confidence"])
                s_["confidence"] = 1.0 if (s_.get("edited") or s_.get("verified")) else s_["base_confidence"]
                s_.pop("checks", None)
    by_end: dict[str, dict] = {}
    for p in draft.get("periods", []):
        by_end.setdefault(p["period_end"], {}).update(p["values"])
    for p in draft.get("periods", []):
        v, src = p["values"], p.setdefault("sources", {})
        label = f"{p['fiscal_period']} to {p['period_end']}"

        def settle(res: dict, used=()):
            for code in used:
                if code in src and "confidence" in src[code]:
                    c = src[code]["confidence"]
                    st = res["status"]
                    src[code]["confidence"] = round(min(1.0, c + BOOST) if st == "PASS" else c * PENALTY if st == "FAIL" else c, 3)
                    src[code].setdefault("checks", []).append({"id": res["id"], "status": st})
            res["period"] = label
            out.append(res)

        bs_lines = {"total_assets", "total_liabilities", "shareholders_equity", "temporary_equity", "minority_interest"}
        if v.keys() & bs_lines:
            need = ("total_assets", "total_liabilities", "shareholders_equity")
            if all(k in v for k in need):
                a, l, e = v["total_assets"], v["total_liabilities"], v["shareholders_equity"]
                mezz, mi = v.get("temporary_equity", 0.0), v.get("minority_interest", 0.0)
                variants = [("liabilities + equity", l + e), ("liabilities + mezzanine equity + equity", l + mezz + e), ("liabilities + equity + minority interests", l + e + mi), ("liabilities + mezzanine + equity + minority", l + mezz + e + mi)]
                best = min(variants, key=lambda x: abs(a - x[1]))
                ok = abs(a - best[1]) <= _tol(a, best[1])
                settle(_res("bs_identity", "Assets = liabilities (+ mezzanine equity) + equity", Check.PASS if ok else Check.FAIL,
                            f"assets {a:,.0f} vs {best[0]} = {best[1]:,.0f} (difference {a - best[1]:,.0f})", need + (("temporary_equity",) if "temporary_equity" in v else ())), tuple(k for k in need) + (("temporary_equity",) if "temporary_equity" in v else ()))
            else:
                missing = [k for k in need if k not in v]
                settle(_res("bs_identity", "Assets = liabilities (+ mezzanine equity) + equity", Check.INSUFFICIENT_DATA, "missing: " + ", ".join(missing), need))
        else:
            out.append(_res("bs_identity", "Assets = liabilities (+ mezzanine equity) + equity", Check.NOT_APPLICABLE, "no balance sheet lines for this period", (), label))
        gp = ("revenue", "cost_of_revenue", "gross_profit")
        if sum(k in v for k in gp) == 3:
            exp = v["revenue"] - abs(v["cost_of_revenue"])
            ok = abs(v["gross_profit"] - exp) <= _tol(v["revenue"], v["gross_profit"])
            settle(_res("gross_profit", "Gross profit = revenue − cost of revenue", Check.PASS if ok else Check.FAIL, f"{v['gross_profit']:,.2f} vs {v['revenue']:,.2f} − {abs(v['cost_of_revenue']):,.2f}", gp), gp)
        elif any(k in v for k in gp):
            settle(_res("gross_profit", "Gross profit = revenue − cost of revenue", Check.INSUFFICIENT_DATA, "missing: " + ", ".join(k for k in gp if k not in v), gp))
        if {"gross_profit", "operating_income"} <= set(v) and ("research_and_development" in v or "selling_general_admin" in v):
            exp = v["gross_profit"] - abs(v.get("research_and_development", 0.0)) - abs(v.get("selling_general_admin", 0.0))
            ok = abs(v["operating_income"] - exp) <= _tol(v["gross_profit"], v["operating_income"], terms=4)
            used = tuple(k for k in ("gross_profit", "research_and_development", "selling_general_admin", "operating_income") if k in v)
            settle(_res("operating_bridge", "Operating income = gross profit − R&D − SG&A (other operating items may exist)", Check.PASS if ok else Check.INSUFFICIENT_DATA,
                        f"{v['operating_income']:,.2f} vs {exp:,.2f}" + ("" if ok else ": other operating items are not extracted, so the bridge cannot be confirmed"), used), used if ok else ())
        x = cross.get(p["period_end"], {}) if cross else {}
        ni = v.get("net_income")
        if ni is not None and x.get("cf_net_income") is not None:
            diff = abs(ni - x["cf_net_income"])
            res = Check.PASS if diff <= _tol(ni, x["cf_net_income"]) else (Check.INSUFFICIENT_DATA if diff <= 0.35 * max(abs(ni), 1) else Check.FAIL)
            settle(_res("ni_cross", "Net income: income statement = start of the cash flow statement", res, f"{ni:,.2f} vs {x['cf_net_income']:,.2f}" + ("" if res == Check.PASS else " (a cash flow statement may start from profit before minority interests)"), ("net_income",)), ("net_income",))
        cash = v.get("cash_and_equivalents", v.get("cash"))
        if cash is not None and x.get("cf_end_cash") is not None:
            ok = abs(cash - x["cf_end_cash"]) <= _tol(cash, x["cf_end_cash"])
            code = "cash_and_equivalents" if "cash_and_equivalents" in v else "cash"
            settle(_res("cash_cross", "Cash: balance sheet = end of the cash flow statement", Check.PASS if ok else Check.INSUFFICIENT_DATA,
                        f"{cash:,.2f} vs {x['cf_end_cash']:,.2f}" + ("" if ok else " (restricted cash and exchange effects may explain it)"), (code,)), (code,))
        bad = [c for c in ("revenue", "total_assets") if c in v and v[c] < 0]
        if bad:
            settle(_res("sign", "Sign of a line that cannot be negative", Check.FAIL, ", ".join(f"{c} = {v[c]:,.0f}" for c in bad), bad), bad)
        if "revenue" in v and "total_assets" in v and v["revenue"] > 0 and v["total_assets"] > 0:
            ratio = v["revenue"] / v["total_assets"]
            ok = 0.0005 < ratio < 60
            settle(_res("scale", "Revenue and assets are of a comparable scale (one unit for both statements)", Check.PASS if ok else Check.FAIL,
                        f"revenue / assets = {ratio:.4g}" + ("" if ok else ": a thousandfold gap usually means two statements in different units"), ("revenue", "total_assets")), ("revenue", "total_assets") if not ok else ())
        if "net_income" in v and "revenue" in v and v["revenue"] > 0 and not (-5 <= v["net_income"] / v["revenue"] <= 1.0):
            settle(_res("margin", "Net margin is plausible", Check.INSUFFICIENT_DATA, f"net income / revenue = {v['net_income'] / v['revenue']:.2f}", ("net_income", "revenue")))
        if {"net_income", "net_income_attributable"} <= set(v) and v["net_income_attributable"] > v["net_income"] * 1.0001 + 3 and v["net_income"] > 0:
            settle(_res("attributable_le_total", "Profit attributable to the parent is not more than profit for the period (when minorities lose nothing)", Check.INSUFFICIENT_DATA,
                        f"{v['net_income_attributable']:,.2f} > {v['net_income']:,.2f}: a loss-making minority can explain it", ("net_income", "net_income_attributable")))
        if {"net_income", "eps_basic", "shares_outstanding"} <= set(v) and v["eps_basic"]:
            implied = v["net_income"] / v["eps_basic"]
            ok = 0.4 <= implied / v["shares_outstanding"] <= 2.5 if v["shares_outstanding"] else True
            settle(_res("eps", "Net income / EPS is close to the share count", Check.PASS if ok else Check.INSUFFICIENT_DATA, f"implied {implied:,.0f} shares vs {v['shares_outstanding']:,.0f}", ("eps_basic", "net_income")), ("eps_basic",) if ok else ())
    # the cash roll-forward across two consecutive balance sheets: beginning cash + operating + investing + financing = ending cash, when nothing else (exchange effects, restricted cash) is extracted
    ends = sorted(by_end)
    for a, b in zip(ends, ends[1:]):
        va, vb = by_end[a], by_end[b]
        ca, cb = va.get("cash_and_equivalents", va.get("cash")), vb.get("cash_and_equivalents", vb.get("cash"))
        flows = [vb.get(k) for k in ("operating_cash_flow", "investing_cash_flow", "financing_cash_flow")]
        if ca is not None and cb is not None and all(f is not None for f in flows):
            move = sum(flows)
            ok = abs((cb - ca) - move) <= _tol(cb, ca, move)
            out.append(_res("cash_roll", "Beginning cash + operating + investing + financing = ending cash", Check.PASS if ok else Check.INSUFFICIENT_DATA,
                            f"change in cash {cb - ca:,.0f} vs flows {move:,.0f}" + ("" if ok else " (exchange effects and restricted cash are not extracted)"), ("cash_and_equivalents",), b))
        elif ca is not None and cb is not None and any(f is not None for f in flows):
            out.append(_res("cash_roll", "Beginning cash + operating + investing + financing = ending cash", Check.INSUFFICIENT_DATA, "missing cash flow lines", (), b))
    for c in conflicts or []:
        out.append({"id": "duplicate_conflict", "label": "The same line is given twice with different values", "status": "FAIL", "detail": c["detail"], "lines": [c["code"]], "period": c["period"]})
    return out


def validation(draft: dict, expected_statements=("IS", "BS", "CF")) -> dict:
    """The verdict for a draft. ``VALIDATED`` only if nothing blocks it; otherwise ``REVIEW_REQUIRED`` with every reason. A missing input is never counted as success."""
    reasons: list[dict] = []

    def block(kind: str, message: str):
        reasons.append({"kind": kind, "message": message})

    res = draft.get("checks") or []
    fails = [c for c in res if c["status"] == "FAIL"]
    for c in fails:
        block("check_failed", f"{c['label']}: {c['detail']}" + (f" ({c['period']})" if c.get("period") else ""))
    found = {s["kind"] for s in draft.get("statements", []) if s.get("confidence", 0) > 0}
    for k in expected_statements:
        if k not in found and draft.get("statements") is not None and k in ("IS", "BS"):
            block("statement_missing", {"IS": "no income statement was recognised", "BS": "no balance sheet was recognised", "CF": "no cash flow statement was recognised"}[k])
    guessed = [p for p in draft.get("periods", []) if p.get("guessed")]
    if guessed:
        block("period_unresolved", "a period was inferred, not read from a heading: " + ", ".join(sorted({f"{p['fiscal_period']} {p['period_end']}" for p in guessed})[:4]))
    if not draft.get("units_stated", True):
        block("unit_unstated", "the report does not state its unit")
    if draft.get("currency") is None:
        block("currency_unknown", "no reporting currency was found")
    mixed = draft.get("mixed_units") or []
    if mixed:
        block("unit_mismatch", "statements of the same report use different units or currencies: " + "; ".join(mixed[:3]))
    present = {c for p in draft.get("periods", []) for c in p["values"]}
    missing = [" / ".join(g) for g in CRITICAL_ANY if not any(c in present for c in g)]
    if missing and draft.get("statements") is not None:
        block("critical_missing", "critical lines not found: " + ", ".join(missing))
    # a critical line is validated only if some check involving it passed
    crit: dict[str, dict] = {}
    for g in CRITICAL_ANY:
        for code in g:
            if code in present:
                involved = [c for c in res if code in c.get("lines", [])]
                crit[code] = {"checks": {s: sum(1 for c in involved if c["status"] == s) for s in ("PASS", "FAIL", "INSUFFICIENT_DATA")}, "validated": any(c["status"] == "PASS" for c in involved) and not any(c["status"] == "FAIL" for c in involved)}
    unvalidated = [c for c, v in crit.items() if not v["validated"]]
    if unvalidated:
        block("critical_unvalidated", "no reconciliation confirmed: " + ", ".join(unvalidated))
    low = confidence_summary(draft)["low"]
    if low:
        block("low_confidence", f"{len(low)} figure(s) below the confidence threshold")
    status = "VALIDATED" if not reasons else "REVIEW_REQUIRED"
    return {"status": status, "reasons": reasons, "blocking": [r["kind"] for r in reasons], "critical": crit, "counts": {s: sum(1 for c in res if c["status"] == s) for s in ("PASS", "FAIL", "INSUFFICIENT_DATA", "NOT_APPLICABLE")},
            "missing_critical": missing}


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


__all__ = ["run", "validation", "confidence_summary", "CRITICAL", "CRITICAL_ANY", "LOW_CONFIDENCE"]
