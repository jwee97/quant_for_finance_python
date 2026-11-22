"""Context-aware matching: IR + taxonomy mapping -> the importer's draft, in separate, inspectable stages.

1. **Statement identification** (:mod:`statements`): which table is which statement, consolidated or not, which accounting basis, with continuation pages joined.
2. **Column resolution** (:mod:`columns`): the role, period, currency, unit and scope of every column. Only reported-period columns can hold facts.
3. **Candidate extraction**: every figure in every mapped row of every table becomes a :class:`~.model.FinancialCandidate`, kept with the evidence for and against it. Nothing is discarded early.
4. **Normalisation**: raw values stay raw; normalised values use exact decimal arithmetic and the column's own unit.
5. **Ranking**: authority (primary consolidated statement 100, unstated scope 90, summary 40, note or segment 10), then total rows over their components, column confidence, then position.
6. **Validation** (:mod:`checks`): reconciliations with four-valued outcomes; unresolved conflicts between candidates of equal authority are not decided by order, they are reported.
7. **Selection**: the winner per (period, line); the others, and why they lost, stay in ``draft["candidates"]``.

A label that matches is never enough on its own: the statement, scope, basis, column role and period must agree.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from decimal import Decimal

from .. import canonical
from . import checks, observations as observations_module, periods as period_module, statements, taxonomy
from . import pdf as pdf_module
from .ir import Document
from .model import ColumnRole, FinancialCandidate, normalise
from .statements import STATEMENT_NAME
from .trace import Trace

READER_CONFIDENCE = {"pdf-layout": 0.95, "pdf-ocr": 0.7, "html": 0.95, "xlsx": 1.0, "csv": 1.0, "docling": 0.85, "pdfplumber": 0.9}
UNIT_FACTOR = {"units": 1, "thousands": 1_000, "millions": 1_000_000, "billions": 1_000_000_000}
KIND_STATEMENT = {"income": "IS", "balance": "BS", "cashflow": "CF"}
AUTHORITY = {"consolidated": 100, "unspecified": 90, "summary": 40, "other": 10}
BASIS_WORDS = (("us-gaap", r"u\.?s\.?\s+gaap|united\s+states\s+generally\s+accepted|generally\s+accepted\s+accounting\s+principles\s+in\s+the\s+united\s+states"), ("ifrs", r"\bifrs\b|international\s+financial\s+reporting\s+standards"),
               ("sfrs-i", r"sfrs\(i\)|singapore\s+financial\s+reporting"), ("k-ifrs", r"k-ifrs|korean\s+international"), ("j-gaap", r"j-gaap|japanese\s+gaap|企業会計基準|日本基準"), ("cas", r"企业会计准则|chinese\s+accounting\s+standards|prc\s+gaap"))
BASIS_PREPARED = re.compile(r"(?:in\s+accordance\s+with|under|according\s+to|based\s+on|conform(?:s|ing)?\s+to|compl(?:y|ies|ying)\s+with)\s+(?:the\s+)?(?:applicable\s+)?(?:with\s+)?(?:requirements\s+of\s+)?$", re.I)


def detect_basis(text: str) -> tuple[str | None, str]:
    """The accounting basis the financial statements say they are prepared under: the standard named in a "prepared in accordance with…" sentence, else the most mentioned one. Not inferred from the country."""
    low = (text or "")[:120000].lower()
    scores: Counter = Counter()
    for std, pattern in BASIS_WORDS:
        for m in re.finditer(pattern, low):
            scores[std] += 5 if BASIS_PREPARED.search(low[max(0, m.start() - 45): m.start()]) else 1                 # "…prepared in accordance with IFRS": the phrase ends right before the name
    if not scores:
        return None, "not stated"
    std, n = scores.most_common(1)[0]
    return std, "stated in the report" if n >= 5 else "mentioned in the report"


def table_basis(title: str) -> str | None:
    low = (title or "").lower()
    for std, pattern in BASIS_WORDS:
        if re.search(pattern, low):
            return std
    return None


# Lines that legitimately appear in more than one primary statement (depreciation is a cost line of the income statement and the first adjustment of the cash flow statement; interest and stock compensation likewise).
MULTI = {"depreciation_amortization": ("IS", "CF"), "stock_based_compensation": ("IS", "CF"), "interest_expense": ("IS", "CF"), "ebitda": ("IS",), "ebit": ("IS",), "free_cash_flow": ("CF", "IS"),
         "share_repurchases": ("CF", "BS"), "weighted_avg_shares_diluted": ("IS", "BS"), "shares_outstanding": ("IS", "BS"), "retained_earnings": ("BS",), "preferred_equity": ("BS",)}


def _allowed(code: str, kind: str) -> bool:
    if code in MULTI:
        return kind in MULTI[code]
    m = canonical.METRICS[code]
    if m.statement == "shares":
        return kind in ("IS", "BS")
    return KIND_STATEMENT.get(m.statement) == kind


def _is_total(label: str) -> bool:
    return bool(re.match(r"\s*(?:total|sum|gesamt|summe|合計|合计|총|总)\b", label, re.I)) or bool(re.search(r"合計$|合计$|계$|総計|総额|总额", label))


def _same(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= max(Decimal("2"), Decimal("0.0015") * max(abs(a), abs(b)))


def build(doc: Document, hint: dict | None = None) -> dict:
    hint = hint or {}
    trace = Trace(hint.get("trace"))
    standard, how = taxonomy.detect_standard(hint, doc.text_head, doc.language)
    basis, basis_how = detect_basis(doc.text_head)
    if hint.get("standard"):
        basis, basis_how = hint["standard"], "hint"
    if basis and basis_how == "stated in the report" and how != "hint" and basis in taxonomy.standards_by_name():
        standard, how = basis, "report basis"                                                           # the report says which standard it follows: that decides the label mapping, not the country
    mapping = taxonomy.get(standard)
    basis = basis or standard
    lines = doc.text_head.splitlines() or [doc.text_head]
    tables = statements.merge_continuations([t for t in doc.tables])
    ctx = pdf_module.context_from_lines(lines + [t.title for t in tables] + [t.context for t in tables])
    year_end = statements.doc_year_end(doc.text_head)
    sts = [statements.classify(t, mapping) for t in tables]
    for s in sts:
        trace.emit("statement_classification", table=s.table.table_id, pages=s.table.pages or [s.table.page], title=s.table.title[:80], kind=s.kind, confidence=round(s.confidence, 3), scope=s.scope, why=s.why)
    primary = [s for s in sts if s.kind in ("BS", "IS", "CF")]
    summary = [s for s in sts if s.kind == "SUMMARY"]
    other = [s for s in sts if s.kind == "OTHER" and s.lines]
    rejections: list[dict] = []
    for s in sts:
        if s.kind == "OTHER" and s.lines:
            rejections.append({"reason": s.why, "table": s.table.table_id, "page": s.table.page, "title": s.table.title[:80]})
    for s in primary + summary + other:
        statements.resolve_columns(s, ctx, year_end)
    for s in sts:                                                                                      # tables that map to no known line still have columns: their figures are observations too
        if s not in primary and s not in summary and s not in other:
            try:
                statements.resolve_columns(s, ctx, year_end)
            except Exception:                                                                          # a table nobody asked for must never stop the statements being read
                s.columns = []
    if year_end is None:                                                                              # a balance sheet names its date in full ("31 December 2023"): that is the fiscal year end the bare-year headings share
        named = Counter((date.fromisoformat(c.period_end).month, date.fromisoformat(c.period_end).day) for s in primary if s.kind == "BS" for c in s.columns if c.usable and not c.guessed)
        for s in primary:                                                                              # "As at 31 March 2020" in a balance sheet's title or heading
            if s.kind == "BS":
                full = pdf_module._headings(s.table.title, None, {})
                if len(full) == 1 and not full[0].get("guessed"):
                    d0 = date.fromisoformat(full[0]["period_end"])
                    named[(d0.month, d0.day)] += 2
        if named:
            year_end = named.most_common(1)[0][0]
            for s in primary + summary + other:
                if any(c.guessed for c in s.columns):
                    statements.resolve_columns(s, ctx, year_end)
    for s in primary + summary + other:
        for c in s.columns:
            trace.emit("column_resolution", table=s.table.table_id, page=s.table.page, statement=s.kind, column=c.col, bbox=c.bbox, original_header=c.original_header, resolved_header=c.resolved_header,
                       column_type=c.role, period_end=c.period_end, currency=c.currency, unit=c.unit, scope=c.scope, confidence=round(c.confidence, 3), reasons=c.reasons)
    ye_month = year_end[0] if year_end else None
    reader_conf = READER_CONFIDENCE.get((primary or summary or [None])[0].table.kind if (primary or summary) else "", 0.9)
    usable_cols = [(s, c) for s in primary for c in s.columns if c.usable]
    ccy_counts = Counter(c.currency for _, c in usable_cols if c.currency)
    unit_counts = Counter(c.unit for _, c in usable_cols if c.unit)
    from .tabular import detect_currency
    currency = hint.get("currency") or (ccy_counts.most_common(1)[0][0] if ccy_counts else detect_currency(doc.text_head))
    text_unit = pdf_module.detect_units(doc.text_head[:6000])
    base_unit = (unit_counts.most_common(1)[0][0] if unit_counts else text_unit) or "units"
    units_stated = bool(unit_counts or text_unit)
    primary_basis = basis

    known_ends = sorted({c.period_end for s in primary for c in s.columns if c.usable and c.period_end})

    def snap(end: str) -> str:
        """A period end that another table names a few days differently (a 52/53-week year: September 26 against September 28) is the same period: it takes the end date of the primary statement."""
        if not end or end in known_ends:
            return end
        d0 = date.fromisoformat(end)
        for k in known_ends:
            dk = date.fromisoformat(k)
            if abs((dk - d0).days) <= 7 and period_module.nominal_month(dk) == period_module.nominal_month(d0):
                return k
        return end

    cands: list[FinancialCandidate] = []
    meta: dict[int, dict] = {}                                                                         # id(candidate) -> extra evidence
    cross: dict[str, dict] = {}
    presentations: list[dict] = []
    scoped: list[dict] = []
    sigs_units: dict[str, set] = {}
    for s in primary + summary + other:
        tbasis = table_basis(s.table.title)
        base_auth = AUTHORITY["summary"] if s.kind == "SUMMARY" else AUTHORITY["other"] if s.kind == "OTHER" else None
        for r in s.table.rows():
            row = r[0].row
            if row in s.table.header_rows or row not in s.lines:
                continue
            code = s.lines[row]
            label = next((c.text for c in r if c.col == 0), "")
            m = mapping.match(label)
            if m is None:
                continue
            for col in s.columns:
                cell = next((c for c in r if c.col == col.col), None)
                if cell is None or cell.number is None or cell.percent:
                    continue
                reasons: list[str] = []
                status = "candidate"
                scope = col.scope if col.scope != "unspecified" else s.scope
                if s.kind in ("BS", "IS", "CF") and s.confidence <= 0:
                    status, reasons = "rejected", [s.why]
                if col.role not in ("CURRENT_PERIOD", "PRIOR_PERIOD", "HISTORICAL_PERIOD", "RESTATED", "ALTERNATIVE_CURRENCY") and status == "candidate":
                    status, reasons = "rejected", [f"column is {col.role} (“{col.original_header}”), not a reported period"]
                    trace.emit("candidate_rejected", metric=code, candidate_value=cell.number, column_header=col.original_header, column_type=col.role, accepted=False, reason="non_reporting_period_column", table=s.table.table_id)
                if tbasis and primary_basis and tbasis != primary_basis and status == "candidate":
                    status, reasons = "rejected", [f"accounting basis {tbasis} differs from the report's basis {primary_basis}"]
                if scope == "parent" and status == "candidate":
                    status, reasons = "rejected", ["parent-company / non-consolidated scope"]
                if s.kind == "CF" and code in ("net_income", "cash", "cash_and_equivalents") and status == "candidate":
                    key = "cf_net_income" if code == "net_income" else "cf_end_cash"
                    if col.period_end and col.role in ("CURRENT_PERIOD", "PRIOR_PERIOD", "HISTORICAL_PERIOD"):
                        mult = UNIT_FACTOR.get(col.unit or base_unit, 1) / UNIT_FACTOR[base_unit]
                        cross.setdefault(col.period_end, {})
                        if key == "cf_end_cash" or key not in cross[col.period_end]:
                            cross[col.period_end][key] = cell.number * mult
                    continue
                if s.kind in ("BS", "IS", "CF") and not _allowed(code, s.kind) and status == "candidate":
                    status, reasons = "rejected", [f"“{label}” reads as {canonical.METRICS[code].label}, which belongs to the {canonical.METRICS[code].statement} statement, not the {STATEMENT_NAME[s.kind]}"]
                if col.kind == "YTD" and status == "candidate":
                    status, reasons = "rejected", ["a nine-month / year-to-date column is not stored as a quarter"]
                kind_of = canonical.METRICS[code].kind
                mult = 1 if kind_of in ("per_share", "shares") else UNIT_FACTOR.get(col.unit or s.unit or base_unit, 1)
                raw = Decimal(int(cell.number)) if float(cell.number).is_integer() else Decimal(str(cell.number))                 # the figure as printed: 1,000 is 1000, not 1000.0
                authority = base_auth or (AUTHORITY["consolidated"] if scope == "consolidated" else AUTHORITY["unspecified"])
                cand = FinancialCandidate(metric=code, label=label, raw_value=raw, page=cell.page or s.table.page, table_id=s.table.table_id, row=cell.row, col=cell.col, bbox=cell.bbox, statement=s.kind, scope=scope,
                                          column_role=col.role, period_end=snap(col.period_end or ""), currency=col.currency or s.currency or currency, unit=col.unit or s.unit or base_unit, authority=authority, status=status, reasons=reasons)
                meta[id(cand)] = {"mult": mult, "col": col, "stmt": s, "map": m, "total": _is_total(label), "kind": kind_of}
                cands.append(cand)
    # ---- alternative-currency columns and parent-scope data are kept apart, never merged into the reported figures
    selectable = []
    for c in cands:
        mt = meta[id(c)]
        if c.column_role == "ALTERNATIVE_CURRENCY" and c.status == "candidate":
            presentations.append({"metric": c.metric, "period_end": c.period_end, "currency": c.currency, "unit": c.unit, "raw_value": str(c.raw_value), "normalized_value": str(normalise(c.raw_value, mt["mult"])), "page": c.page,
                                  "table": c.table_id, "statement": c.statement, "label": c.label})
            c.status, c.reasons = "rejected", ["another currency than the report's primary currency: kept as a presentation"]
        elif c.scope == "parent" and c.statement in ("BS", "IS", "CF") and c.column_role in ("CURRENT_PERIOD", "PRIOR_PERIOD", "HISTORICAL_PERIOD") and c.period_end:
            scoped.append({"scope": "parent", "metric": c.metric, "period_end": c.period_end, "currency": c.currency, "raw_value": str(c.raw_value), "normalized_value": str(normalise(c.raw_value, mt["mult"])), "page": c.page, "table": c.table_id})
        elif c.status == "candidate":
            selectable.append(c)
    primary_ends = {c.period_end for c in selectable if c.statement in ("BS", "IS", "CF")}
    primary_flow_kinds = {(c.period_end, meta[id(c)]["col"].kind) for c in selectable if c.statement in ("IS", "CF")}
    for c in list(selectable):
        mt = meta[id(c)]
        if c.statement == "SUMMARY":
            # a summary or highlights table corroborates; it is selected only for a current period that no primary statement covers (a quarter inside an annual report), never for history
            if (c.period_end, mt["col"].kind) in primary_flow_kinds or (c.period_end in primary_ends and mt["kind"] in ("stock", "shares")):
                c.status, c.reasons = "rejected", ["a primary statement covers this period: the summary only corroborates it"]
            elif c.column_role != "CURRENT_PERIOD":
                c.status, c.reasons = "rejected", ["a summary table's history is not stored: only a primary statement creates a period"]
            elif not primary:
                pass
            selectable.remove(c) if c.status == "rejected" else None
        elif c.statement == "OTHER":
            c.status, c.reasons = "rejected", ["not a primary statement (segment, note or reconciliation table)"]
            selectable.remove(c)
    # ---- ranking: authority, then totals over components, then column confidence, then position; conflicts of equal authority are reported, not decided by order
    groups: dict[tuple, list[FinancialCandidate]] = {}
    ends_with_flow: dict[str, str] = {}
    for c in selectable:
        if c.statement in ("IS", "CF", "SUMMARY") and meta[id(c)]["kind"] in ("flow", "per_share"):
            ends_with_flow.setdefault(c.period_end, meta[id(c)]["col"].kind)

    def attach_kind(c: FinancialCandidate) -> str:
        mt = meta[id(c)]
        if mt["kind"] in ("stock", "shares") or c.statement == "BS":
            end = date.fromisoformat(c.period_end)
            for e, k in ends_with_flow.items():                                                       # a 52/53-week year: the balance sheet date and the income statement date may differ by days
                d = date.fromisoformat(e)
                if period_module.nominal_month(d) == period_module.nominal_month(end) and abs((d - end).days) <= 7:
                    return k, e
            fp = period_module.fiscal_period_label(period_module.build(end, "", True, ye_month), ye_month)
            return ("FY" if fp == "FY" else "H1" if fp == "H1" else "Q"), c.period_end
        return mt["col"].kind, c.period_end

    for c in selectable:
        k, end = attach_kind(c)
        mt = meta[id(c)]
        c.score = c.authority * 1000 + (60 if mt["total"] else 0) + mt["col"].confidence * 10 + mt["map"].confidence * 5 - (c.page or 0) * 0.01 - c.row * 0.0001
        groups.setdefault((end, k, c.metric), []).append(c)
    periods: dict[tuple, dict] = {}
    conflicts: list[dict] = []
    for (end, ckind, code), items in sorted(groups.items()):
        if ckind == "YTD":
            continue
        items.sort(key=lambda c: -c.score)
        best = items[0]
        top = [c for c in items if c.authority == best.authority and (meta[id(c)]["total"] == meta[id(best)]["total"])]
        disagree = [c for c in top[1:] if not _same(c.raw_value * Decimal(str(meta[id(c)]["mult"])), best.raw_value * Decimal(str(meta[id(best)]["mult"])))]
        for c in items[1:]:
            c.status, c.reasons = "rejected", c.reasons or [("lower authority" if c.authority < best.authority else "an equal candidate was ranked higher") + f" (score {c.score:.1f} vs {best.score:.1f})"]
        best.status = "selected"
        conf_mult = 1.0
        if disagree:
            conflicts.append({"period": end, "code": code, "detail": f"{canonical.METRICS[code].label}: " + " vs ".join(f"{c.raw_value:,} (page {c.page}, table {c.table_id})" for c in [best] + disagree[:3])})
            conf_mult = 0.55
            trace.emit("candidate_conflict", metric=code, period_end=end, values=[str(c.raw_value) for c in [best] + disagree])
        mt = meta[id(best)]
        col = mt["col"]
        fp_obj = col.period
        label = period_module.fiscal_period_label(fp_obj, ye_month) if fp_obj and fp_obj.period_type != "INSTANT" else {"FY": "FY", "Q": "Q" + str(period_module.fiscal_quarter_of(date.fromisoformat(end), ye_month)), "H1": "H1"}[ckind]
        if ckind == "Q" and not label.startswith("Q"):
            label = "Q" + str(period_module.fiscal_quarter_of(date.fromisoformat(end), ye_month))
        p = periods.setdefault((end, ckind), {"fiscal_period": label, "period_end": end, "fiscal_year": period_module.fiscal_year_of(date.fromisoformat(end), ye_month), "filing_date": None, "currency": best.currency or currency, "values": {},
                                              "sources": {}, "guessed": False, "period_type": fp_obj.period_type if fp_obj and fp_obj.period_type != "INSTANT" else {"FY": "ANNUAL", "Q": "QUARTER", "H1": "HALF_YEAR"}[ckind],
                                              "start_date": fp_obj.start_date if fp_obj and fp_obj.period_type != "INSTANT" else None, "duration_months": fp_obj.duration_months if fp_obj and fp_obj.period_type != "INSTANT" else None,
                                              "is_forecast": False, "is_restated": bool(fp_obj and fp_obj.is_restated)})
        conf = best.authority / 100.0 * mt["stmt"].confidence * mt["map"].confidence * (0.7 if col.guessed else 1.0) * reader_conf * conf_mult
        conf = min(conf, 1.0) if best.statement != "SUMMARY" else min(conf, 0.6)
        if not units_stated and mt["kind"] in ("flow", "stock"):
            conf *= 0.7
        agree = [c for c in items[1:] if c.statement == "SUMMARY" and _same(c.raw_value, best.raw_value)]
        value = float(best.raw_value * Decimal(str(mt["mult"])) / Decimal(UNIT_FACTOR[base_unit])) if mt["kind"] not in ("per_share", "shares") else float(best.raw_value)
        if mt["kind"] not in ("per_share", "shares") and (col.unit or base_unit) == base_unit:
            value = float(best.raw_value)
        p["values"][code] = value
        p["guessed"] = p["guessed"] or col.guessed or ckind == "Q" and best.statement != "SUMMARY"
        p["sources"][code] = {"tag": None, "reported": float(best.raw_value), "raw_value": str(best.raw_value), "normalized_value": str(normalise(best.raw_value, mt["mult"])), "unit": col.unit or base_unit,
                              "unit_multiplier": mt["mult"], "currency": best.currency, "page": best.page, "text": " | ".join(c.text for c in next(rr for rr in mt["stmt"].table.rows() if rr[0].row == best.row))[:160],
                              "method": doc.reader.get("name", "layout"), "confidence": round(conf, 3), "statement": STATEMENT_NAME.get(best.statement, best.statement.lower()), "scope": best.scope, "label": best.label,
                              "mapping_rule": mt["map"].rule, "bbox": list(best.bbox) if best.bbox else None, "row": best.row, "col": best.col, "table": best.table_id, "column_role": best.column_role,
                              "column_header": col.original_header, "authority": best.authority, "candidates_considered": len(items), "corroborated_by_summary": len(agree),
                              "alternatives": [{"value": float(c.raw_value), "page": c.page, "table": c.table_id, "reason": c.reasons[0] if c.reasons else ""} for c in items[1:4]]}
        trace.emit("candidate_selected", metric=code, period_end=end, value=str(best.raw_value), table=best.table_id, page=best.page, authority=best.authority, considered=len(items))
    # the values converted to the base unit
    for (end, ckind), p in periods.items():
        for code in list(p["values"]):
            src = p["sources"][code]
            kind_of = canonical.METRICS[code].kind
            if kind_of in ("per_share", "shares"):
                continue
            p["values"][code] = float(Decimal(src["raw_value"]) * Decimal(str(src["unit_multiplier"])) / Decimal(UNIT_FACTOR[base_unit]))
    plist = sorted(periods.values(), key=lambda p: (p["period_end"], p["fiscal_period"]))
    # one currency and unit per statement: a mix inside a statement is a finding
    mixed = []
    for s in primary:
        seen = {(c.currency, c.unit) for c in s.columns if c.usable}
        if len(seen) > 1:
            mixed.append(f"{STATEMENT_NAME.get(s.kind, s.kind)} (page {s.table.page}): " + ", ".join(f"{a or '?'} {b or '?'}" for a, b in sorted(seen, key=str)))
    draft = {"periods": plist, "units": {"money": base_unit, "shares": "units"}, "currency": currency, "company": {}, "notes": [], "ytd_only": [], "unmapped": [], "skipped": rejections[:80],
             "method": (primary or summary or [None])[0].table.kind if (primary or summary) else doc.format, "review_required": doc.format in ("pdf", "pdf-scan", "html"), "units_stated": units_stated, "accession": None, "pages": doc.pages,
             "mixed_units": mixed, "fiscal_year_end": list(year_end) if year_end else None}
    draft["standard"] = {"standard": standard, "how": how, "mapping_version": mapping.version, "taxonomy_version": taxonomy.TAXONOMY_VERSION, "basis": basis, "basis_how": basis_how}
    draft["statements"] = [{"table": s.table.table_id, "page": s.table.page, "pages": s.table.pages or [s.table.page], "kind": s.kind, "confidence": round(s.confidence, 3), "scope": s.scope, "why": s.why, "unit": s.unit,
                            "currency": s.currency, "columns": [{"col": c.col, "period_end": c.period_end, "kind": c.kind, "guessed": c.guessed, "note": c.note, "role": c.role, "currency": c.currency, "unit": c.unit, "scope": c.scope,
                                                                 "original_header": c.original_header, "resolved_header": c.resolved_header, "confidence": round(c.confidence, 3), "reasons": c.reasons, "bbox": c.bbox} for c in s.columns],
                            "lines": len(s.lines)} for s in sts if s.kind != "OTHER" or s.lines]
    draft["rejections"] = (rejections + [{"reason": c.reasons[0], "label": c.label, "page": c.page, "table": c.table_id, "value": str(c.raw_value), "metric": c.metric} for c in cands if c.status == "rejected" and c.reasons][:150])[:250]
    draft["candidates"] = [c.to_dict() for c in cands][:1500]
    draft["presentations"] = presentations[:400]
    draft["scoped"] = scoped[:400]
    draft["cross_values"] = cross
    try:
        draft["raw_observations"] = observations_module.collect(sts, cands, base_unit, currency, snap, doc.reader.get("name", "layout"), draft["standard"] | {"extractor": taxonomy.EXTRACTOR_VERSION,
                                                                 "mapping": mapping.version, "taxonomy": taxonomy.TAXONOMY_VERSION})
    except Exception as error:                                                                         # the archive of observations never blocks the statements
        draft["raw_observations"] = []
        draft["notes"].append(f"The archive of all figures could not be built ({error}).")
    draft["checks"] = checks.run(draft, cross, conflicts)
    draft["conflicts_between_candidates"] = conflicts
    draft["confidence"] = checks.confidence_summary(draft)
    draft["validation"] = checks.validation(draft)
    for c in draft["checks"]:
        trace.emit("reconciliation", check=c["id"], status=c["status"], detail=c["detail"], period=c.get("period"))
    if trace.enabled:
        draft["trace"] = trace.events
    if doc.warnings:
        draft["notes"] += doc.warnings
    if not units_stated:
        draft["notes"].append("The report does not state its unit in the text read: amounts are read as units. Choose the unit if they are in thousands or millions.")
    if how in ("default", "script", "language"):
        draft["notes"].append(f"The accounting standard for the label mapping was assumed to be {mapping.label} ({how}); the report's own basis is {basis or 'not stated'} ({basis_how}).")
    if presentations:
        draft["notes"].append(f"{len(presentations)} figure(s) in another currency were kept as presentations and not stored as reported figures.")
    return draft


__all__ = ["build", "detect_basis", "READER_CONFIDENCE"]
