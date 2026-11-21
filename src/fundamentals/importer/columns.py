"""Semantic column resolution: what each column of a statement table *is*, from its heading hierarchy and, when the heading cannot say, from the numbers.

Every figure column gets a role (:class:`~.model.ColumnRole`). Only the periods of reported figures (current, prior, historical, restated) produce financial facts; a change column, a percentage column,
a note reference, a forecast and a column in another currency never do. The role comes from, in this order:

1. the heading words (Increase/(Decrease), %, Note, Forecast, Restated, Group, Company, a currency, a unit; in English, Japanese, Korean, Chinese, French, German, Spanish);
2. the figures: a column whose cells are, row by row, the difference of two other columns is a change column whatever its heading says, and a column of small whole numbers on few rows is a note reference;
3. the dates: among the period columns of one (scope, currency) the latest is current, the next prior, the rest historical; two columns that name the same date in the same group are not guessed between.

Each decision keeps the reasons that produced it, so an unexpected choice can be explained from the log.
"""

from __future__ import annotations

import re
from datetime import date

from . import pdf as _pdf
from . import periods as period_module

pdf_module_TERM = _pdf.TERM
from .ir import Table
from .model import ColumnRole, FinancialPeriod, TableColumn
from .tabular import CURRENCY_WORDS, UNIT_WORDS, detect_currency, detect_units

R = ColumnRole
ROLE_WORDS = [
    (R.PERCENT_CHANGE, re.compile(r"%|\bpercent|\bpct\b|百分|比率|増減率|증감률|%变化", re.I)),
    (R.ABSOLUTE_CHANGE, re.compile(r"increase|decrease|\bchange\b|variance|difference|\bvar\.?\b|\byoy\b|増減|差異|증감|增减|变动|變動|增減|variation|évolution|veränderung|cambio|variación|\bδ|\bΔ", re.I)),
    (R.NOTE_REFERENCE, re.compile(r"^\s*(?:notes?|注記|注|주석|附注|附註|note\(s\)|anmerkung(?:en)?|nota(?:s)?)\s*$", re.I)),
]
SCOPE_CONSOLIDATED = re.compile(r"\bgroup\b|consolidated|連結|연결|合并|合併|consolidé|konzern|consolidado", re.I)
SCOPE_PARENT = re.compile(r"\bcompany\b(?!\s+(?:limited|ltd))|parent|separate|母公司|個別|単体|별도|单独|individuel|einzel|individual", re.I)
SMALL_INTS = 100


def _figure_cells(t: Table, col: int):
    return [c for c in t.cells if c.col == col and c.row not in t.header_rows and (c.number is not None or c.dash or c.percent)]


def _header_levels(t: Table, col: int) -> list[str]:
    out = []
    for r in t.header_rows:
        texts = [c.text for c in t.row(r) if c.col > 0 and c.col <= col < c.col + max(c.span, 1)]
        if texts:
            out.append(" ".join(texts))
    return out


def _units_by_currency(text: str) -> dict[str, str]:
    """Pairs of a currency and the unit stated for it: "(KRW in millions, USD in thousands)", "in thousands of US dollars", "RMB'000", "HK$ million"."""
    pairs: dict[str, str] = {}
    unit_spans = []
    for name, pattern in UNIT_WORDS:
        for m in re.finditer(pattern, text, re.I):
            unit_spans.append((m.start(), m.end(), name))
    cur_spans = []
    for code, pattern in CURRENCY_WORDS:
        for m in re.finditer(pattern, text, re.I):
            cur_spans.append((m.start(), m.end(), code))
    for u0, u1, name in unit_spans:
        best = None
        for c0, c1, code in cur_spans:
            dist = min(abs(c0 - u1), abs(u0 - c1)) if (c1 <= u0 or c0 >= u1) else 0
            if dist <= 45 and (best is None or dist < best[0]):
                best = (dist, code)
        if best:
            pairs.setdefault(best[1], name)
    return pairs


def _is_difference(t: Table, a: int, b: int, c: int) -> float:
    """The share of rows (at least 3 needed) in which column ``c`` equals the difference of columns ``a`` and ``b`` (either way round), within rounding."""
    def val(col):
        return {x.row: x.number for x in t.cells if x.col == col and x.number is not None and x.row not in t.header_rows}
    va, vb, vc = val(a), val(b), val(c)
    rows = sorted(set(va) & set(vb) & set(vc))
    if len(rows) < 3:
        return 0.0
    hit = 0
    for r in rows:
        tol = max(2.0, 0.002 * max(abs(va[r]), abs(vb[r])))
        if abs(vc[r] - (vb[r] - va[r])) <= tol or abs(vc[r] - (va[r] - vb[r])) <= tol:
            hit += 1
    return hit / len(rows)


def resolve(t: Table, kind: str, ctx: dict, year_end: tuple | None, table_currency: str | None = None, table_unit: str | None = None) -> list[TableColumn]:
    """The columns of ``t`` (a statement of ``kind`` BS | IS | CF | SUMMARY) with role, period, currency, unit and scope."""
    ye_month = year_end[0] if year_end else None
    figure_cols = sorted({c.col for c in t.cells if c.col > 0 and c.row not in t.header_rows and (c.number is not None or c.dash or c.percent)})
    unit_text = " ".join([t.unit_text, t.title, t.context])
    by_ccy = _units_by_currency(unit_text)
    all_header = " ".join(c.text for r in t.header_rows for c in t.row(r))
    table_scope = "parent" if SCOPE_PARENT.search(t.title) and not SCOPE_CONSOLIDATED.search(t.title) else "consolidated" if SCOPE_CONSOLIDATED.search(t.title + " " + all_header) and not SCOPE_PARENT.search(all_header) else "unspecified"
    cols: list[TableColumn] = []
    for col in figure_cols:
        levels = _header_levels(t, col)
        text = " ".join(levels)
        ext = t.column_extents.get(col)
        tc = TableColumn(col=col, header_levels=levels, bbox=(ext[0], 0, ext[1], 0) if ext else None)
        cells = _figure_cells(t, col)
        n_cells = len(cells)
        pct_share = sum(1 for c in cells if c.percent) / max(n_cells, 1)
        numbers = [c.number for c in cells if c.number is not None]
        reasons = tc.reasons
        # ---- scope, currency, unit from this column's own heading
        tc.scope = "parent" if SCOPE_PARENT.search(text) and not SCOPE_CONSOLIDATED.search(text) else "consolidated" if SCOPE_CONSOLIDATED.search(text) else table_scope
        tc.currency = detect_currency(text) or None
        tc.unit = detect_units(text)
        # ---- role from the heading words, then from the figures
        role = None
        for r_, rx in ROLE_WORDS:
            if rx.search(text):
                role = r_
                reasons.append(f"heading matches {r_.value}")
                break
        if role is None and pct_share >= 0.6:
            role, _ = R.PERCENT_CHANGE, reasons.append("most cells are percentages")
        if role is None and numbers and n_cells and all(float(x).is_integer() and 0 < abs(x) < SMALL_INTS for x in numbers) and len(figure_cols) >= 3 and (col == figure_cols[0] or not text.strip()) and len(numbers) <= max(2, len(t.rows()) // 2):
            role, _ = R.NOTE_REFERENCE, reasons.append("a column of small whole numbers on few rows, before the figures")
        if role is None and period_module.FORECAST.search(text):
            role, _ = R.FORECAST, reasons.append("heading says forecast")
        tc.role = role or R.OTHER
        cols.append(tc)

    # ---- dates for the period candidates
    cands = [c for c in cols if c.role in (R.OTHER,)]
    range_free = re.sub(r"(?:19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}\s*(?:부터|から|至|to)\s*(?:19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}\s*(?:까지|まで|止)?", " ", all_header)   # "2023.01.01 부터 2023.12.31 까지" names one span
    table_dates = period_module.heading_dates(re.sub(r"\s+", " ", range_free), year_end, ctx) if t.header_rows else []
    own: dict[int, FinancialPeriod] = {}
    for tc in cands:
        got = period_module.column_date([re.sub(r"(?:19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}\s*(?:부터|から|至|to)\s*(?:19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}\s*(?:까지|まで|止)?", " ", lv) for lv in tc.header_levels], t.title + " " + range_free, year_end, ctx)
        if got:
            end = date.fromisoformat(got[0])
            tc.period = period_module.build(end, " ".join(tc.header_levels) + " " + t.title, kind == "BS", ye_month, got[1])
            tc.confidence = 0.95 if not got[1] else 0.7
            tc.reasons.append("date read from this column's heading" + ("" if not got[1] else " (day and month assumed)"))
            own[tc.col] = tc.period
    if kind in ("IS", "CF"):
        # numbered terms ("제 55(당) 기 / 제 54(전) 기"): the balance sheet's earlier term ends at the last year end, but an income or cash flow statement's earlier term is the same span a year before
        termed = sorted([c for c in cands if c.col in own and any(pdf_module_TERM.search(lv) for lv in c.header_levels)], key=lambda c: c.col)
        if len(termed) >= 2:
            first = date.fromisoformat(own[termed[0].col].end_date)
            for i, c in enumerate(termed[1:], start=1):
                end = date(first.year - i, first.month, min(first.day, period_module.month_end(first.year - i, first.month)))
                c.period = period_module.build(end, " ".join(c.header_levels) + " " + t.title, False, ye_month, False)
                own[c.col] = c.period
                c.reasons.append("an earlier numbered term of a flow statement: the same span one year before the current term")
    unresolved = [c for c in cands if c.col not in own]
    if unresolved:
        # the heading rows name the dates but not above each figure (a merged or centred heading): the dates in order, if they are as many as the columns
        pool = [d for d in table_dates]
        remaining_dates = [d for d in pool if d["period_end"] not in {p.end_date for p in own.values()}]
        if remaining_dates and len(remaining_dates) == len(unresolved):
            for tc, d in zip(unresolved, remaining_dates):
                end = date.fromisoformat(d["period_end"])
                tc.period = period_module.build(end, all_header + " " + t.title, kind == "BS", ye_month, d.get("guessed", False))
                tc.confidence = 0.75
                tc.reasons.append("date taken from the heading rows in order (as many dates as columns)")
        elif not t.header_rows and ctx.get("latest") and len(unresolved) <= 3:
            d0 = ctx["latest"]
            for i, tc in enumerate(unresolved):
                end = date(d0.year - i, d0.month, min(d0.day, period_module.month_end(d0.year - i, d0.month)))
                tc.period = period_module.build(end, t.title, kind == "BS", ye_month, True)
                tc.confidence = 0.4
                tc.reasons.append("no heading: the report's latest date and the years before it (a guess)")
    # ---- a change column the heading did not announce: the figures say so
    loose = [c for c in cols if c.role == R.OTHER]
    if len(loose) >= 3:
        for c in sorted(loose, key=lambda c: (c.period is not None, -c.col)):                          # the column with no heading of its own, then the rightmost: change columns follow the periods
            if c.role != R.OTHER:
                continue
            others = [o for o in loose if o.col != c.col and o.role == R.OTHER]
            for i, a in enumerate(others):
                for b in others[i + 1:]:
                    share = _is_difference(t, a.col, b.col, c.col)
                    if share >= 0.7:
                        c.role = R.ABSOLUTE_CHANGE
                        c.reasons.append(f"its figures are the difference of columns {a.col} and {b.col} in {share:.0%} of rows")
                        c.period = None
                        break
                if c.role == R.ABSOLUTE_CHANGE:
                    break
    # ---- currency and unit per column: own heading, then the unit stated for its currency, then the table's
    body_symbols = " ".join(c.text for c in t.cells if c.col > 0 and c.row not in t.header_rows and re.match(r"^(?:US\$|HK\$|S\$|R\$|A\$|C\$|[$€£¥￥₩])", c.text.strip()))[:200]
    base_ccy = table_currency or detect_currency(unit_text) or (detect_currency(body_symbols) if body_symbols else None)    # "$ 274,515": a symbol printed with the figures states the currency too
    for tc in cols:
        if tc.currency is None:
            tc.currency = base_ccy
        tc.unit = tc.unit or by_ccy.get(tc.currency or "") or table_unit or detect_units(unit_text)
    # ---- forecasts and alternative currencies; roles of the dated columns
    primary_ccy = None
    dated = [c for c in cols if c.period is not None and c.role == R.OTHER]
    if dated:
        primary_ccy = min(dated, key=lambda c: c.col).currency
    groups: dict[tuple, list[TableColumn]] = {}
    for tc in dated:
        if tc.period.is_forecast:
            tc.role = R.FORECAST
            tc.reasons.append("period is a forecast")
        elif tc.currency and primary_ccy and tc.currency != primary_ccy:
            tc.role = R.ALTERNATIVE_CURRENCY
            tc.reasons.append(f"another currency ({tc.currency}) than the first period column ({primary_ccy})")
        else:
            groups.setdefault((tc.scope, tc.currency), []).append(tc)
    # a date later than the periods the other columns name, in a flow statement, whose heading names a plan year, is not a result
    for key, g in groups.items():
        g.sort(key=lambda c: c.period.end_date, reverse=True)
        ends = [c.period.end_date for c in g]
        for i, c in enumerate(g):
            if ends.count(c.period.end_date) > 1:
                c.role = R.OTHER
                c.confidence = min(c.confidence, 0.3)
                c.reasons.append("two columns of the same scope and currency name this date: not guessed between")
                continue
            c.role = R.CURRENT_PERIOD if i == 0 else R.PRIOR_PERIOD if i == 1 else R.HISTORICAL_PERIOD
            if c.period.is_restated:
                c.reasons.append("restated figures")
    for tc in cols:
        if tc.role == R.OTHER and tc.period is None:
            tc.reasons.append("no period could be read for this column")
        tc.resolved_header = f"{tc.role.value}" + (f" {tc.period.end_date}" if tc.period else "") + (f" {tc.currency}" if tc.currency else "") + (f" ({tc.scope})" if tc.scope != "unspecified" else "")
    return cols


__all__ = ["resolve"]
