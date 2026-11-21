"""Statement-first structure: which table is which statement, for which scope, with which period columns.

A figure is only ever taken from a table that has been recognised as a primary statement (income, balance sheet, cash flow), for the consolidated group, with a period column that could be read.
Recognition uses two independent kinds of evidence, the title and the lines in the table: a table titled "Consolidated statement of financial position" whose lines are assets, liabilities and equity
is a balance sheet with full confidence; a table whose title and lines disagree is kept at low confidence; a table of "Segment information", a note, or a five-year summary is not a primary statement
and is not used for the figures (a summary is a labelled, low-confidence fallback when nothing else exists).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .. import canonical
from . import pdf as pdf_module
from . import taxonomy
from .ir import Cell, Document, Table
from .tabular import detect_currency, detect_units

TITLES = {
    "BS": r"balance\s+sheet|statements?\s+of\s+financial\s+position|financial\s+position|assets\s+and\s+liabilities|貸借対照表|連結貸借|財政状態計算書|대차대조표|재무상태표|资产负债表|資產負債表|財務狀況表|bilan|situación financiera|balanço patrimonial|bilanz|ativo",
    "IS": r"income\s+statements?|statements?\s+of\s+(?:comprehensive\s+)?(?:income|profit|operations|earnings)|profit\s+or\s+loss|profit\s+and\s+loss|results\s+of\s+operations|損益計算書|包括利益計算書|손익계산서|포괄손익계산서|利润表|損益表|綜合收益表|综合收益表|compte\s+de\s+résultat|cuenta\s+de\s+resultados|demonstração\s+do\s+resultado|gewinn-\s+und\s+verlustrechnung",
    "CF": r"cash\s+flows?|キャッシュ・?フロー計算書|現金流量表|현금흐름표|现金流量表|tableau\s+(?:\w+\s+){0,2}des\s+flux|flujos\s+de\s+efectivo|fluxo\s+de\s+caixa|kapitalflussrechnung",
}
NOT_PRIMARY = r"segment|geographical|geographic|notes?\s+to|note\s+\d|reconciliation\s+of|changes\s+in\s+equity|stockholders.\s+equity\s+statement|変動計算書|자본변동표|所有者权益变动表|権益変動|pro\s+forma|supplementary|ratios?|non-?ifrs\s+reconciliation"
SUMMARY_TITLES = r"five[-\s]year|financial\s+highlights|highlights|selected\s+financial|summary|quarterly\s+results|(?:first|second|third|fourth)\s+quarter|quarter\s+\d{4}\s+results|results\s+for\s+the\s+quarter|forecast|outlook|五年|財務ハイライト|要約|하이라이트"
SCOPE_PARENT = r"non-?consolidated|unconsolidated|stand-?alone|\bparent\s+(?:company|only)\b|company\s+only|separate\s+(?:financial|statement)|company\s+statement|\bthe\s+company\b\s*(?:statement|balance)|個別|単体|별도|母公司|单独|nur\s+ag|individuel|individual\b|solo\b"
SCOPE_CONSOL = r"(?<!non-)(?<!non\s)(?<!un)consolidated|group\b|連結|연결|合并|合併|consolidé|consolidado|konzern"
PRIMARY_LINES = {"BS": {"total_assets", "total_liabilities", "shareholders_equity", "current_assets", "current_liabilities", "cash", "cash_and_equivalents", "inventory", "accounts_receivable", "ppe", "retained_earnings"},
                 "IS": {"revenue", "net_income", "operating_income", "gross_profit", "cost_of_revenue", "income_before_tax", "eps_basic", "eps_diluted", "income_tax"},
                 "CF": {"operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "capital_expenditure", "dividends_paid", "depreciation_amortization"}}
STATEMENT_NAME = {"BS": "balance sheet", "IS": "income statement", "CF": "cash flow statement"}


@dataclass
class Column:
    col: int
    period_end: str | None
    kind: str = "FY"                  # FY | Q | H1 | YTD
    guessed: bool = False
    note: bool = False                # the "Notes" reference column: not a period
    role: str = "OTHER"               # a model.ColumnRole value
    period: object = None             # model.FinancialPeriod
    currency: str | None = None
    unit: str | None = None
    scope: str = "unspecified"
    confidence: float = 0.0
    reasons: list = field(default_factory=list)
    original_header: str = ""
    resolved_header: str = ""
    bbox: tuple | None = None

    @property
    def usable(self) -> bool:
        return self.role in ("CURRENT_PERIOD", "PRIOR_PERIOD", "HISTORICAL_PERIOD", "RESTATED") and self.period_end is not None


@dataclass
class StatementTable:
    table: Table
    kind: str                         # BS | IS | CF | SUMMARY | OTHER
    confidence: float
    scope: str                        # consolidated | parent | unspecified
    why: str
    columns: list[Column] = field(default_factory=list)
    unit: str | None = None
    currency: str | None = None
    lines: dict[int, str] = field(default_factory=dict)


def _kind_words(text: str) -> str:
    low = text.lower()
    if pdf_module.KIND_Q.search(low):
        return "Q"
    if pdf_module.KIND_H1.search(low):
        return "H1"
    if pdf_module.KIND_YTD.search(low):
        return "YTD"
    return "FY"


def classify(table: Table, mapping: taxonomy.Mapping, page_context: str = "") -> StatementTable:
    title = f"{table.title} {table.context}"[:600]
    low = title.lower()
    scores = {k: bool(re.search(rx, title, re.I)) for k, rx in TITLES.items()}
    # the title that names the table is the nearest text: prefer the table's own title; the page context only breaks ties
    own = {k: bool(re.search(rx, table.title, re.I)) for k, rx in TITLES.items()}
    titled = [k for k, v in own.items() if v] or [k for k, v in scores.items() if v]
    lines: dict[int, str] = {}
    votes = {"BS": 0, "IS": 0, "CF": 0}
    for r in table.rows():
        label = next((c.text for c in r if c.col == 0), "")
        m = mapping.match(label)
        if m:
            lines[r[0].row] = m.code
            for k, codes in PRIMARY_LINES.items():
                if m.code in codes:
                    votes[k] += 1
    best = max(votes, key=votes.get) if any(votes.values()) else None
    scope = "unspecified"
    scope_text = f"{table.title} {table.context}"
    if re.search(SCOPE_PARENT, scope_text, re.I) and not re.search(SCOPE_CONSOL, table.title, re.I):
        scope = "parent"
    elif re.search(SCOPE_CONSOL, scope_text, re.I):
        scope = "consolidated"
    if re.search(SUMMARY_TITLES, table.title, re.I) and not any(own.values()):
        return StatementTable(table, "SUMMARY", 0.45, scope, "a summary or highlights table (" + re.search(SUMMARY_TITLES, table.title, re.I).group(0) + "): corroborates, never replaces, a primary statement", lines=lines)
    not_primary = bool(re.search(NOT_PRIMARY, table.title, re.I))
    if not_primary and not titled:
        return StatementTable(table, "OTHER", 0.0, scope, "not a primary statement (" + (re.search(NOT_PRIMARY, table.title, re.I).group(0)) + ")", lines=lines)
    if not_primary and titled and re.search(NOT_PRIMARY, table.title, re.I).group(0).lower() in ("notes to", "segment", "geographical", "geographic", "reconciliation of"):
        return StatementTable(table, "OTHER", 0.0, scope, "not a primary statement (" + re.search(NOT_PRIMARY, table.title, re.I).group(0) + ")", lines=lines)
    kind, conf, why = "OTHER", 0.0, "no statement title and too few known lines"
    if titled and best and titled[0] == best and votes[best] >= 2:
        kind, conf, why = best, 1.0, f"title and {votes[best]} lines agree"
    elif titled and (not best or votes[best] < 2):
        kind, conf, why = titled[0], 0.7, "title only"
    elif titled and best and titled[0] != best and votes[best] >= 2:
        kind, conf, why = best, 0.55, f"title says {STATEMENT_NAME[titled[0]]}, lines say {STATEMENT_NAME[best]}"
    elif best and votes[best] >= 3:
        kind, conf, why = best, 0.7, f"{votes[best]} lines, no title"
    elif best and votes[best] >= 1 and len(lines) >= 2:
        kind, conf, why = "SUMMARY", 0.45, "a few known lines without a statement title: a summary, not a statement"
    return StatementTable(table, kind, conf, scope, why, lines=lines)


def _is_note_column(table: Table, col: int) -> bool:
    nums = [c.number for c in table.cells if c.col == col and c.number is not None and c.row not in table.header_rows]
    return bool(nums) and all(float(n).is_integer() and 0 < n < 100 for n in nums) and len(nums) >= 1


def resolve_columns(st: StatementTable, doc_ctx: dict, year_end: tuple | None) -> None:
    """The role, period, currency, unit and scope of every figure column of the statement table (see :mod:`columns`)."""
    from . import columns as columns_module
    from .model import ColumnRole
    t = st.table
    unit_text = " ".join([t.unit_text, t.title, t.context])
    st.unit = detect_units(unit_text)
    st.currency = detect_currency(unit_text)
    cols = columns_module.resolve(t, st.kind, doc_ctx, year_end, st.currency, st.unit)
    out = []
    for c in cols:
        p = c.period
        kind = {"ANNUAL": "FY", "QUARTER": "Q", "HALF_YEAR": "H1", "YTD": "YTD", "INSTANT": "FY", "UNKNOWN_FLOW": "FY"}.get(p.period_type, "FY") if p else "FY"
        usable = c.role in (ColumnRole.CURRENT_PERIOD, ColumnRole.PRIOR_PERIOD, ColumnRole.HISTORICAL_PERIOD)
        out.append(Column(col=c.col, period_end=p.end_date if p else None, kind=kind, guessed=bool(p) and c.confidence < 0.9, note=c.role == ColumnRole.NOTE_REFERENCE, role=c.role.value, period=p, currency=c.currency,
                          unit=c.unit, scope=c.scope, confidence=c.confidence if usable else 0.0, reasons=c.reasons, original_header=c.original_header, resolved_header=c.resolved_header, bbox=c.bbox))
    st.columns = out
    if not [c for c in out if c.usable]:
        st.confidence *= 0.0
        st.why += "; no period columns could be read"
    elif any(c.guessed for c in out if c.usable):
        st.why += "; period columns partly guessed"


def annotate_sections(doc: Document, tables: list[StatementTable]) -> None:
    from .ir import Section
    seen: dict[tuple, Section] = {}
    for st in tables:
        key = (st.table.page, st.kind)
        sec = seen.get(key)
        if sec is None:
            sec = Section(title=st.table.title or STATEMENT_NAME.get(st.kind, st.kind), page_start=st.table.page, page_end=st.table.page, scope=st.scope)
            seen[key] = sec
        sec.table_ids.append(st.table.table_id)
    doc.sections = list(seen.values())


def doc_year_end(text: str) -> tuple | None:
    m = re.search(r"(?:fiscal\s+)?year\s+ended\s+([A-Za-z]+)\s+(\d{1,2})", text, re.I)
    if m and pdf_module.month_of(m.group(1)):
        return pdf_module.month_of(m.group(1)), int(m.group(2))
    m = re.search(r"year\s+ended\s+(\d{1,2})\s+([A-Za-z]+)", text, re.I)
    if m and pdf_module.month_of(m.group(2)):
        return pdf_module.month_of(m.group(2)), int(m.group(1))
    return None


__all__ = ["Column", "StatementTable", "classify", "resolve_columns", "annotate_sections", "doc_year_end", "TITLES"]


def merge_continuations(tables: list[Table]) -> list[Table]:
    """A statement that runs over a page break is one statement. A table that starts the next page without a title of its own, or whose title says "continued", and whose columns sit where the previous
    table's columns sit, is appended to it (its rows renumbered, its cells keeping their own page and box); a repeated heading is dropped."""
    out: list[Table] = []
    for t in tables:
        prev = out[-1] if out else None
        if prev is not None and t.page is not None and prev.pages and t.page == max(prev.pages) + 1 and _continues(prev, t):
            offset = prev.n_rows()
            keep = {r for r in range(t.n_rows()) if r not in t.header_rows} if t.header_rows and _same_header(prev, t) else set(range(t.n_rows()))
            remap = {}
            for r in sorted(keep):
                remap[r] = offset + len(remap)
            cmap = _col_map(prev, t) or {}
            for c in t.cells:
                if c.row in remap:
                    prev.cells.append(Cell(c.text, remap[c.row], cmap.get(c.col, c.col) if c.col else 0, c.page, c.bbox, c.number, c.dash, c.percent, c.span))
            prev.pages.append(t.page)
            prev.raw_words += t.raw_words
            for k, v in t.column_extents.items():
                kk = cmap.get(k, k)
                a = prev.column_extents.get(kk)
                prev.column_extents[kk] = (min(a[0], v[0]), max(a[1], v[1])) if a else v
            prev.title = prev.title or t.title
            continue
        if t.page is not None and not t.pages:
            t.pages = [t.page]
        out.append(t)
    return out


def _same_header(a: Table, b: Table) -> bool:
    ha = " ".join(c.text for r in a.header_rows for c in a.row(r)).strip().lower()
    hb = " ".join(c.text for r in b.header_rows for c in b.row(r)).strip().lower()
    return bool(hb) and (ha == hb or hb in ha or ha in hb)


def _col_map(prev: Table, t: Table) -> dict[int, int] | None:
    """Each column of ``t`` mapped to the column of ``prev`` that sits where it sits (within 14 points), or None if one has no counterpart."""
    out = {}
    for k, b in t.column_extents.items():
        def gap(kv):
            a = kv[1]
            return min(abs(a[1] - b[1]), abs(a[0] - b[0]), abs((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2))          # right-aligned figures share a right edge, left-aligned ones a left edge
        best = min(prev.column_extents.items(), key=gap, default=None)
        if best is None or gap(best) > 10:
            return None
        out[k] = best[0]
    return out if len(set(out.values())) == len(out) else None


def _continues(prev: Table, t: Table) -> bool:
    """Whether ``t`` (the first table of its page) is the rest of ``prev``: it says "continued", or it has no title; and its columns sit where columns of ``prev`` sit (a note column may be absent), and a
    heading it carries repeats prev's."""
    cont_words = re.search(r"continued|cont['’]?d|\(cont|続き|계속|续|續|suite|fortsetzung|continuación", t.title or "", re.I)
    title = (t.title or "").strip()
    section_line = bool(title) and not t.header_rows and len(title) <= 45 and not any(re.search(rx, title, re.I) for rx in TITLES.values())     # "Current liabilities": a section line of the statement, not a new title
    titled = bool(title) and not cont_words and not section_line
    if titled or not t.column_extents or not prev.column_extents:
        return False
    if _col_map(prev, t) is None:
        return False
    if t.header_rows and not _same_header(prev, t):
        return False
    return True
