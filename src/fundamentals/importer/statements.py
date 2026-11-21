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
from .ir import Document, Table
from .tabular import detect_currency, detect_units

TITLES = {
    "BS": r"balance\s+sheet|statements?\s+of\s+financial\s+position|financial\s+position|assets\s+and\s+liabilities|貸借対照表|連結貸借|財政状態計算書|대차대조표|재무상태표|资产负债表|資產負債表|財務狀況表|bilan|situación financiera|balanço patrimonial|bilanz|ativo",
    "IS": r"income\s+statements?|statements?\s+of\s+(?:comprehensive\s+)?(?:income|profit|operations|earnings)|profit\s+or\s+loss|profit\s+and\s+loss|results\s+of\s+operations|損益計算書|包括利益計算書|손익계산서|포괄손익계산서|利润表|損益表|綜合收益表|综合收益表|compte\s+de\s+résultat|cuenta\s+de\s+resultados|demonstração\s+do\s+resultado|gewinn-\s+und\s+verlustrechnung",
    "CF": r"cash\s+flows?|キャッシュ・?フロー計算書|現金流量表|현금흐름표|现金流量表|tableau\s+(?:\w+\s+){0,2}des\s+flux|flujos\s+de\s+efectivo|fluxo\s+de\s+caixa|kapitalflussrechnung",
}
NOT_PRIMARY = r"segment|geographical|geographic|notes?\s+to|note\s+\d|five[-\s]year|financial\s+highlights|selected\s+financial|summary|quarterly\s+results|reconciliation\s+of|changes\s+in\s+equity|stockholders.\s+equity\s+statement|変動計算書|자본변동표|所有者权益变动表|権益変動|pro\s+forma|supplementary|ratios?"
SCOPE_PARENT = r"\bparent\s+(?:company|only)\b|company\s+only|separate\s+(?:financial|statement)|company\s+statement|\bthe\s+company\b\s*(?:statement|balance)|個別|単体|별도|母公司|单独|nur\s+ag|individuel|individual\b|solo\b"
SCOPE_CONSOL = r"consolidated|group\b|連結|연결|合并|合併|consolidé|consolidado|konzern"
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
    """The period of each figure column, from the table's own heading rows (all of them read together, in order), the terms and dates the report names elsewhere, and the words of the table."""
    t = st.table
    numeric_cols = sorted({c.col for c in t.cells if c.col > 0 and c.number is not None and c.row not in t.header_rows} | {c.col for c in t.cells if c.col > 0 and c.dash and c.row not in t.header_rows})
    note_cols = [c for c in numeric_cols if _is_note_column(t, c)]
    # a column of small integers is a notes column only when a heading says so, or it is the first of more figure columns than headings
    heads = " ".join(c.text for r in t.header_rows for c in t.row(r) if c.col > 0 or len(t.row(r)) == 1)
    header_all = " ".join(c.text for r in t.header_rows for c in t.row(r))
    if not header_all.strip():
        header_all = t.title
    def range_free(x: str) -> str:                                                                    # "2023.01.01 부터 2023.12.31 까지" names one period's span, not two columns
        return re.sub(r"(?:19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}\s*(?:부터|から|至|to)\s*(?:19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}\s*(?:까지|まで|止)?", " ", x)
    header_all = range_free(header_all)
    dates = pdf_module._headings(header_all, year_end, doc_ctx) if t.header_rows else []
    if len(dates) < 2:                                                                               # no heading row named the periods: the title may ("Year ended 31 December 2023 and 2022")
        dates = pdf_module._headings(header_all + " " + range_free(t.title), year_end, doc_ctx) or dates
    if st.kind in ("IS", "CF") and len(dates) >= 2 and len(set(pdf_module.TERM.findall(header_all))) >= 2 and not pdf_module.DATE_NUM.search(header_all):
        # numbered terms ("제 55(당) 기 / 제 54(전) 기"): the balance sheet's earlier term ends at the last year end, but an income or cash flow statement's earlier term is the same span a year before
        d0 = date.fromisoformat(dates[0]["period_end"])
        dates = [dates[0]] + [{"period_end": date(d0.year - i, d0.month, min(d0.day, 28 if d0.month == 2 else d0.day)).isoformat(), "guessed": False} for i in range(1, len(dates))]
    kind = _kind_words(header_all + " " + t.title)
    figure_cols = [c for c in numeric_cols if c not in note_cols] if note_cols else numeric_cols
    if note_cols and len(dates) == len(numeric_cols):                                                 # as many headings as columns: the "notes" column holds figures after all
        figure_cols = numeric_cols
        note_cols = []
    cols: list[Column] = []
    if dates and len(dates) == len(figure_cols):
        for c, d in zip(figure_cols, dates):
            cols.append(Column(c, d["period_end"], kind, d.get("guessed", False)))
    elif dates and len(dates) < len(figure_cols) and len(figure_cols) - len(dates) == 1 and figure_cols[0] in note_cols + figure_cols[:1]:
        for c in figure_cols[:1]:
            cols.append(Column(c, None, kind, True, note=True))
        for c, d in zip(figure_cols[1:], dates):
            cols.append(Column(c, d["period_end"], kind, d.get("guessed", False)))
    elif dates and len(dates) > len(figure_cols) >= 1:                                                # more dates than columns: the first ones name the periods of those columns
        for c, d in zip(figure_cols, dates):
            cols.append(Column(c, d["period_end"], kind, True))
    elif not dates and doc_ctx.get("latest") and len(figure_cols) in (1, 2, 3):                        # no heading read: the report's latest date, a year before it, and so on
        d0 = doc_ctx["latest"]
        for i, c in enumerate(figure_cols):
            cols.append(Column(c, date(d0.year - i, d0.month, min(d0.day, 28 if d0.month == 2 else d0.day)).isoformat(), kind, True))
    st.columns = cols
    if not [c for c in cols if not c.note]:
        st.confidence *= 0.0
        st.why += "; no period columns could be read"
    elif any(c.guessed for c in cols):
        st.why += "; period columns partly guessed"
    unit_text = " ".join([t.unit_text, t.title, t.context])
    st.unit = detect_units(unit_text)
    st.currency = detect_currency(unit_text)


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
