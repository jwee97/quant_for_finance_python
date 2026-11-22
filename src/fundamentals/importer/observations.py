"""Layer 1 of the import: **every quantitative observation the scanner found**, mapped or not.

The importer used to keep only the cells whose row label matched a line of the canonical taxonomy; everything else (segment revenue, deferred revenue, vehicle deliveries, regulatory capital ratios, a
management-adjusted measure, a debt maturity) was thrown away before a person could look at it. This module keeps them.

An *observation* is one figure of one table cell, with its evidence (page, table, row, column, box), the original label and column heading, the column's role, period, currency, unit and scope, and the
canonical line the scanner *guessed* for it, if any (``metric_guess``; the guess is a suggestion, never a fact). Nothing here is a decision: every observation starts ``UNCONFIRMED``.

Not every number is a financial observation. Page numbers, note references, footnote markers, years, telephone numbers and the like are left out (and counted, so the omission is visible).
"""

from __future__ import annotations

import re

from .. import canonical

STATUSES = ("UNCONFIRMED", "CONFIRMED", "CORRECTED_AND_CONFIRMED", "EXCLUDED", "UNRESOLVED")
PUBLISHABLE = ("CONFIRMED", "CORRECTED_AND_CONFIRMED")
MAX_OBSERVATIONS = 20000

GROUPS = ("income_statement", "balance_sheet", "cash_flow", "equity", "segment", "notes", "kpi", "summary", "other")
GROUP_OF_KIND = {"IS": "income_statement", "BS": "balance_sheet", "CF": "cash_flow", "SUMMARY": "summary"}
GROUP_WORDS = (("equity", r"changes\s+in\s+(?:shareholders'?\s+|stockholders'?\s+|owners'?\s+)?equity|statement\s+of\s+(?:shareholders|stockholders)'?\s+equity|持分変動|株主資本等変動|자본변동|权益变动|權益變動"),
               ("segment", r"segment|reportable|geograph|business\s+unit|division|セグメント|부문|分部|分部資料|分部报告"),
               ("kpi", r"operating\s+(?:data|statistics|highlights|metrics)|key\s+(?:performance|operating|metrics)|deliver|subscribers|monthly\s+active|users|headcount|employees|backlog|bookings|orders|"
                       r"capital\s+ratio|cet1|tier\s*1|non[- ]performing|npl\b|net\s+interest\s+margin|solvency|combined\s+ratio"),
               ("notes", r"\bnotes?\b|schedule|maturit|reconciliation|lease|borrowings|provisions|commitments|contingen|fair\s+value|financial\s+instruments|注記|주석|附注|附註"))
NOISE_LABEL = re.compile(r"^\s*(?:page|pg\.?|p\.|note|notes|ref\.?|tel\.?|telephone|fax|phone|www\.|http|\W*$)", re.I)
NOISE_ANY = re.compile(r"@|https?://|www\.|\bpage\s+\d+\b|\btelephone\b", re.I)
ONLY_NUMBER_LABEL = re.compile(r"^[\s\d\W]+$")
SKIP_ROLES = ("NOTE_REFERENCE",)


def group_for(kind: str, title: str, context: str) -> str:
    if kind in GROUP_OF_KIND:
        return GROUP_OF_KIND[kind]
    low = f"{title} {context}".lower()
    for name, pattern in GROUP_WORDS:
        if re.search(pattern, low, re.I):
            return name
    return "other"


def _key(table_id: str, row: int, col: int) -> str:
    return f"{table_id}:{row}:{col}"


def is_financial(label: str, cell) -> tuple[bool, str]:
    """A figure beside a label is a financial observation unless the label shows it is not (a page number, a reference, an address) or the label is only digits."""
    if not label or not re.search(r"[^\W\d_]{2,}", label):
        return False, "no descriptive label"
    if NOISE_LABEL.match(label) or NOISE_ANY.search(label):
        return False, "page number, reference or contact detail"
    if ONLY_NUMBER_LABEL.match(label):
        return False, "label is only numbers"
    return True, ""


def collect(sts: list, cands: list, base_unit: str, currency: str | None, snap, method: str, versions: dict, pages_with_text: int | None = None) -> list[dict]:
    """Every figure of every table that the scanner read, as observation dicts. ``cands`` are the matcher's candidates (their metric is the guess; ``selected`` says it became a value of the draft)."""
    by_cell: dict[tuple, object] = {}
    for c in cands:
        by_cell[(c.table_id, c.row, c.col)] = c
    out: list[dict] = []
    seen_noise = 0
    for s in sts:
        t = s.table
        if not s.columns:
            continue
        group = group_for(s.kind, t.title, t.context)
        for r in t.rows():
            row = r[0].row
            if row in t.header_rows:
                continue
            label = next((c.text.strip() for c in r if c.col == 0), "")
            ok, why = is_financial(label, None)
            if not ok:
                seen_noise += sum(1 for c in r if c.number is not None and c.col != 0)
                continue
            for col in s.columns:
                if col.role in SKIP_ROLES:
                    continue
                cell = next((c for c in r if c.col == col.col), None)
                if cell is None or cell.number is None:
                    continue
                cand = by_cell.get((t.table_id, cell.row, cell.col))
                period = col.period
                ptype = getattr(period, "period_type", None) if period else None
                end = snap(col.period_end) if col.period_end else None
                unit = "percent" if cell.percent else (col.unit or s.unit or base_unit)
                guess = cand.metric if cand is not None else None
                out.append({
                    "obs_key": _key(t.table_id, cell.row, cell.col), "group": group, "label": label, "value_text": cell.text.strip(), "value": float(cell.number), "is_percent": bool(cell.percent),
                    "page": cell.page or t.page, "table": t.table_id, "row": cell.row, "col": cell.col, "bbox": list(cell.bbox) if cell.bbox else None,
                    "column_header": col.original_header, "column_role": col.role, "period_end": end, "period_start": getattr(period, "start_date", None) if period else None,
                    "period_type": ptype, "period_guessed": bool(col.guessed or not end), "fiscal_period": None,
                    "currency": col.currency or s.currency or currency, "unit": unit, "scope": col.scope if col.scope != "unspecified" else s.scope, "statement": s.kind, "table_title": t.title[:120],
                    "metric_guess": guess, "selected": bool(cand is not None and cand.status == "selected"), "guess_reason": (cand.reasons[0] if cand is not None and cand.reasons else None),
                    "method": method, "extractor_version": versions.get("extractor"), "mapping_version": versions.get("mapping"), "taxonomy_version": versions.get("taxonomy"),
                })
                if len(out) >= MAX_OBSERVATIONS:
                    return out
    return out


def from_draft(draft: dict) -> list[dict]:
    """Observations for a draft that did not come from tables with coordinates (XBRL facts, a workbook read line by line, a legacy reader): one per period and line, plus the unmapped facts it kept."""
    v = draft.get("versions") or {}
    out = []
    for p in draft.get("periods") or []:
        for code, value in (p.get("values") or {}).items():
            if value is None:
                continue
            src = (p.get("sources") or {}).get(code, {})
            m = canonical.METRICS.get(code)
            out.append({"obs_key": f"d:{p.get('fiscal_period')}:{p.get('period_end')}:{code}", "group": {"IS": "income_statement", "BS": "balance_sheet", "CF": "cash_flow"}.get(
                            {"income": "IS", "balance": "BS", "cashflow": "CF"}.get(m.statement if m else "", ""), "other"), "label": src.get("label") or (m.label if m else code),
                        "value_text": str(src.get("raw_value") or src.get("reported") or value), "value": float(value), "is_percent": False, "page": src.get("page"), "table": src.get("table"),
                        "row": src.get("row"), "col": src.get("col"), "bbox": src.get("bbox"), "column_header": src.get("column_header"), "column_role": src.get("column_role") or "CURRENT_PERIOD",
                        "period_end": p.get("period_end"), "period_start": p.get("start_date"), "period_type": p.get("period_type"), "period_guessed": bool(p.get("guessed")), "fiscal_period": p.get("fiscal_period"),
                        "currency": p.get("currency") or draft.get("currency"), "unit": "units" if m and m.kind in ("per_share", "shares") else (draft.get("units") or {}).get("money"), "scope": src.get("scope"),
                        "statement": src.get("statement"), "table_title": None, "metric_guess": code, "selected": True, "guess_reason": None, "method": src.get("method") or draft.get("method"),
                        "extractor_version": v.get("extractor"), "mapping_version": v.get("mapping"), "taxonomy_version": v.get("taxonomy")})
    for u in draft.get("unmapped") or []:
        if not isinstance(u, dict) or u.get("value") is None:
            continue
        label = str(u.get("label") or u.get("tag") or u.get("name") or "")
        if not label:
            continue
        out.append({"obs_key": f"u:{label}:{u.get('period_end')}:{u.get('value')}", "group": "other", "label": label, "value_text": str(u.get("value")), "value": float(u["value"]), "is_percent": False,
                    "page": u.get("page"), "table": None, "row": None, "col": None, "bbox": None, "column_header": None, "column_role": "CURRENT_PERIOD", "period_end": u.get("period_end"),
                    "period_start": u.get("period_start"), "period_type": u.get("period_type"), "period_guessed": not u.get("period_end"), "fiscal_period": None,
                    "currency": u.get("currency") or draft.get("currency"), "unit": u.get("unit") or (draft.get("units") or {}).get("money"), "scope": u.get("scope"), "statement": None,
                    "table_title": None, "metric_guess": None, "selected": False, "guess_reason": None, "method": draft.get("method"), "extractor_version": v.get("extractor"),
                    "mapping_version": v.get("mapping"), "taxonomy_version": v.get("taxonomy")})
    return out


def signature(o: dict) -> tuple:
    return (re.sub(r"\W+", "", (o.get("label") or "").lower()), o.get("period_end"), round(float(o.get("value") or 0), 6), o.get("currency"))


def merge(primary: list[dict], secondary: list[dict], tag: str = " (second reading)") -> list[dict]:
    """Candidates of a second reading are added to those of the first; the first reading's observations are never dropped or replaced (a fallback must not destroy what was read correctly)."""
    have = {signature(o) for o in primary}
    extra = [{**o, "method": (o.get("method") or "") + tag} for o in secondary if signature(o) not in have]
    return (primary + extra)[:MAX_OBSERVATIONS]
