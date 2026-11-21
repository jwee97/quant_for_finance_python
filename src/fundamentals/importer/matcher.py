"""Context-aware matching: IR + taxonomy mapping -> the importer's draft.

Order of work: classify every table as a statement (:mod:`statements`); keep the primary statements of the consolidated group; read each one's period columns; match each row's label through the
taxonomy mapping; accept the line only if the statement it appears in is one the line belongs to; choose between candidates; record every line that was *not* used and why; run the accounting checks
(:mod:`checks`), which move the confidence of the lines that took part. A label that matches is never enough on its own.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

from .. import canonical
from . import checks, statements, taxonomy
from . import pdf as pdf_module
from .ir import Document
from .statements import STATEMENT_NAME

READER_CONFIDENCE = {"pdf-layout": 0.95, "pdf-ocr": 0.7, "html": 0.95, "xlsx": 1.0, "csv": 1.0, "docling": 0.85}
UNIT_FACTOR = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
KIND_STATEMENT = {"income": "IS", "balance": "BS", "cashflow": "CF"}


def _allowed(code: str, kind: str) -> bool:
    m = canonical.METRICS[code]
    if m.statement == "shares":
        return kind in ("IS", "BS")
    return KIND_STATEMENT.get(m.statement) == kind


def build(doc: Document, hint: dict | None = None) -> dict:
    hint = hint or {}
    standard, how = taxonomy.detect_standard(hint, doc.text_head, doc.language)
    mapping = taxonomy.get(standard)
    lines = doc.text_head.splitlines() or [doc.text_head]
    ctx = pdf_module.context_from_lines(lines + [t.title for t in doc.tables] + [t.context for t in doc.tables])
    year_end = statements.doc_year_end(doc.text_head)
    sts = [statements.classify(t, mapping) for t in doc.tables]
    primary = [s for s in sts if s.kind in ("BS", "IS", "CF")]
    if not primary:
        for s in sts:
            if s.kind == "SUMMARY":
                s.confidence = min(s.confidence, 0.45)
                primary.append(s)
    rejections: list[dict] = []
    for s in sts:
        if s.kind == "OTHER" and s.lines:
            rejections.append({"reason": s.why, "table": s.table.table_id, "page": s.table.page, "title": s.table.title[:80]})
    for s in primary:
        statements.resolve_columns(s, ctx, year_end)
    if year_end is None:                                                                              # a balance sheet names its date in full ("31 December 2023"): that is the fiscal year end the bare-year headings of the other statements share
        named = Counter((date.fromisoformat(c.period_end).month, date.fromisoformat(c.period_end).day) for s in primary if s.kind == "BS" for c in s.columns if c.period_end and not c.guessed and not c.note)
        if named:
            year_end = named.most_common(1)[0][0]
            for s in primary:
                if any(c.guessed for c in s.columns):
                    statements.resolve_columns(s, ctx, year_end)
    statements.annotate_sections(doc, primary)
    reader_conf = READER_CONFIDENCE.get(primary[0].table.kind if primary else "", 0.9) if primary else 0.9
    unit_counts = Counter(s.unit for s in primary if s.unit)
    text_unit = pdf_module.detect_units(doc.text_head[:6000])
    base_unit = (unit_counts.most_common(1)[0][0] if unit_counts else text_unit) or "units"
    units_stated = bool(unit_counts or text_unit)
    cur_counts = Counter(s.currency for s in primary if s.currency)
    from .tabular import detect_currency
    currency = (cur_counts.most_common(1)[0][0] if cur_counts else detect_currency(doc.text_head))

    cands: dict[tuple, list[dict]] = {}
    cross: dict[str, dict] = {}
    for s in primary:
        if s.scope == "parent":
            rejections.append({"reason": "parent-company (separate) statement: only the consolidated group is read", "table": s.table.table_id, "page": s.table.page, "title": s.table.title[:80]})
            continue
        if s.confidence <= 0:
            rejections.append({"reason": s.why, "table": s.table.table_id, "page": s.table.page, "title": s.table.title[:80]})
            continue
        scope_conf = 1.0 if s.scope == "consolidated" else 0.92
        factor = UNIT_FACTOR.get(s.unit or base_unit, 1.0) / UNIT_FACTOR[base_unit]
        cols = [c for c in s.columns if not c.note and c.period_end]
        for r in s.table.rows():
            row = r[0].row
            if row in s.table.header_rows or row not in s.lines:
                continue
            code = s.lines[row]
            label = next((c.text for c in r if c.col == 0), "")
            m = mapping.match(label)
            kind = s.kind
            for col in cols:
                cell = next((c for c in r if c.col == col.col), None)
                if cell is None or cell.number is None:
                    continue
                if kind == "CF" and code in ("net_income", "cash", "cash_and_equivalents"):
                    key = "cf_net_income" if code == "net_income" else "cf_end_cash"
                    cross.setdefault(col.period_end, {})
                    if key == "cf_end_cash" or key not in cross[col.period_end]:
                        cross[col.period_end][key] = cell.number * factor
                    continue
                if kind != "SUMMARY" and not _allowed(code, kind):
                    rejections.append({"reason": f"“{label}” reads as {canonical.METRICS[code].label}, which belongs to the {canonical.METRICS[canonical.METRICS[code].code].statement} statement, not the {STATEMENT_NAME[kind]}",
                                       "table": s.table.table_id, "page": s.table.page, "label": label})
                    break
                value = cell.number * (1.0 if canonical.METRICS[code].kind in ("per_share", "shares") else factor)
                conf = s.confidence * m.confidence * (0.7 if col.guessed else 1.0) * scope_conf * reader_conf
                if not units_stated and canonical.METRICS[code].kind in ("flow", "stock"):
                    conf *= 0.7                                                                       # the scale of an amount is unknown: it is stored unverified until a person chooses the unit
                cands.setdefault((col.period_end, col.kind, code, kind), []).append({
                    "value": value, "reported": cell.number, "confidence": conf, "page": s.table.page, "row": cell.row, "col": cell.col, "bbox": cell.bbox, "text": " | ".join(c.text for c in r)[:160],
                    "table": s.table.table_id, "statement": kind, "label": label, "rule": m.rule, "unit": s.unit or base_unit, "scope": s.scope, "guessed": col.guessed, "col_kind": col.kind, "stated_unit": bool(s.unit)})
    # periods: flows keep their duration kind; a balance sheet attaches to the flow period that ends the same day, else is a period of its own
    periods: dict[tuple, dict] = {}
    conflicts: list[dict] = []
    ends_with_flow = {}
    for (end, ckind, code, skind) in cands:
        if skind in ("IS", "CF"):
            ends_with_flow.setdefault(end, ckind)
    for (end, ckind, code, skind), items in sorted(cands.items()):
        if skind == "BS" or (skind == "SUMMARY" and canonical.METRICS[code].kind in ("stock", "shares")):
            key_kind = ends_with_flow.get(end) or _guess_kind(end, year_end, ckind)
        else:
            key_kind = ckind
        if key_kind == "YTD":
            rejections.append({"reason": "a nine-month / year-to-date column is not stored as a quarter", "label": items[0]["label"], "page": items[0]["page"]})
            continue
        best = max(items, key=lambda i: (i["confidence"], -(i["page"] or 0), -i["row"]))
        distinct = {round(i["value"], 6) for i in items}
        if len(distinct) > 1:
            conflicts.append({"period": f"{end}", "code": code, "detail": f"{canonical.METRICS[code].label}: " + " vs ".join(f"{i['value']:,.0f} (page {i['page']})" for i in items[:4])})
            best = dict(best, confidence=best["confidence"] * 0.55)
        elif len(items) > 1:
            best = dict(best, confidence=min(1.0, best["confidence"] + 0.03))                      # the same figure twice (a summary repeating a statement) agrees with itself
        p = periods.setdefault((end, key_kind), {"fiscal_period": "FY" if key_kind == "FY" else "H1" if key_kind == "H1" else "Q" + str((date.fromisoformat(end).month - 1) // 3 + 1), "period_end": end,
                                                   "fiscal_year": date.fromisoformat(end).year, "filing_date": None, "currency": currency, "values": {}, "sources": {}, "guessed": False})
        if code in p["values"]:
            if skind == "BS" and canonical.METRICS[code].statement != "balance":
                continue
            continue
        p["values"][code] = best["value"]
        p["guessed"] = p["guessed"] or best["guessed"] or key_kind in ("Q",)
        p["sources"][code] = {"tag": None, "reported": best["reported"], "unit": best["unit"], "page": best["page"], "text": best["text"], "method": doc.reader.get("name", "layout"),
                              "confidence": round(best["confidence"], 3), "statement": STATEMENT_NAME.get(best["statement"], best["statement"]), "scope": best["scope"], "label": best["label"], "mapping_rule": best["rule"],
                              "bbox": list(best["bbox"]) if best["bbox"] else None, "row": best["row"], "col": best["col"], "table": best["table"],
                              "alternatives": [{"value": i["value"], "page": i["page"]} for i in items if i is not best][:3]}
    plist = sorted(periods.values(), key=lambda p: (p["period_end"], p["fiscal_period"]))
    draft = {"periods": plist, "units": {"money": base_unit, "shares": "units"}, "currency": currency, "company": {}, "notes": [], "ytd_only": [], "unmapped": [], "skipped": rejections[:80],
             "method": primary[0].table.kind if primary else doc.format, "review_required": doc.format in ("pdf", "pdf-scan", "html"), "units_stated": units_stated, "accession": None, "pages": doc.pages}
    draft["standard"] = {"standard": standard, "how": how, "mapping_version": mapping.version, "taxonomy_version": taxonomy.TAXONOMY_VERSION}
    draft["statements"] = [{"table": s.table.table_id, "page": s.table.page, "kind": s.kind, "confidence": round(s.confidence, 3), "scope": s.scope, "why": s.why, "unit": s.unit,
                            "columns": [{"col": c.col, "period_end": c.period_end, "kind": c.kind, "guessed": c.guessed, "note": c.note} for c in s.columns], "lines": len(s.lines)} for s in sts if s.kind != "OTHER"]
    draft["rejections"] = rejections[:200]
    draft["checks"] = checks.run(draft, cross, conflicts)
    draft["confidence"] = checks.confidence_summary(draft)
    if doc.warnings:
        draft["notes"] += doc.warnings
    if not units_stated:
        draft["notes"].append("The report does not state its unit in the text read: amounts are read as units. Choose the unit if they are in thousands or millions.")
    if any(c["guessed"] for p in plist for c in [p]) and any(s["guessed"] for st in draft["statements"] for s in st["columns"]):
        draft["notes"].append("A period column was not named by its heading and was inferred from the report's dates: check the dates.")
    if how in ("default", "script", "language"):
        draft["notes"].append(f"The accounting standard was assumed to be {mapping.label} ({how}): pass the company's country to be exact.")
    return draft


def _guess_kind(end: str, year_end, ckind: str) -> str:
    d = date.fromisoformat(end)
    if year_end:
        return "FY" if (d.month, d.day) == year_end else ("H1" if d.month == (year_end[0] + 5) % 12 + 1 else "Q")
    return "FY" if d.month == 12 else "Q"


__all__ = ["build", "READER_CONFIDENCE"]
