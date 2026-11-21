"""Workbooks and delimited text to tables: a sheet is split into blocks at its blank rows, a block's title is the text row above its figures, cells keep their sheet reference."""

from __future__ import annotations

import io

from ..ir import Cell, Document, Table
from ._cells import to_number


def _blocks(rows: list[list], sheet: str, counter: list[int], kind: str, page: int | None) -> list[Table]:
    out: list[Table] = []
    block: list[tuple[int, list]] = []

    def flush():
        if not block:
            return
        figures = sum(1 for _, r in block for v in r[1:] if to_number(str(v) if v is not None else "")[0] is not None)
        if figures >= 2:
            counter[0] += 1
            t = Table(table_id=f"t{counter[0]}", page=page, kind=kind, context=sheet)
            first_fig = None
            for k, (rowno, r) in enumerate(block):
                if first_fig is None and any(isinstance(v, (int, float)) or to_number(str(v))[0] is not None for v in r[1:] if v is not None) and not _headerish(r):
                    first_fig = k
                for j, v in enumerate(r):
                    if v is None or str(v).strip() == "":
                        continue
                    text = f"{v:,.10g}" if isinstance(v, float) else str(v)
                    num, dash = (float(v), False) if isinstance(v, (int, float)) and not isinstance(v, bool) and j > 0 else (to_number(str(v)) if j > 0 else (None, False))
                    t.cells.append(Cell(text.strip(), k, j, page, None, num, dash))
            first_fig = first_fig or 0
            t.header_rows = list(range(first_fig))
            texts = [" ".join(str(v) for v in r if v is not None) for _, r in block[:first_fig]]
            t.title = (texts[0] if texts else sheet)[:160]
            t.unit_text = " ".join(texts) + " " + sheet
            out.append(t)
        block.clear()

    for rowno, r in enumerate(rows):
        if all(v is None or str(v).strip() == "" for v in r):
            flush()
        else:
            block.append((rowno, r))
    flush()
    return out


def _headerish(r: list) -> bool:
    """A row of years or dates is a heading even though its cells parse as numbers."""
    vals = [v for v in r[1:] if v is not None and str(v).strip()]
    def yearlike(v):
        try:
            return 1990 <= float(str(v).replace(",", "")) <= 2100 and float(str(v).replace(",", "")).is_integer()
        except ValueError:
            return False
    return bool(vals) and all(yearlike(v) or (isinstance(v, str) and any(ch.isalpha() for ch in v)) or hasattr(v, "year") for v in vals)


def from_xlsx(data: bytes) -> Document:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    counter = [0]
    tables: list[Table] = []
    head: list[str] = []
    for n, ws in enumerate(wb.worksheets, start=1):
        rows = [list(r) for r in ws.iter_rows(values_only=True)][:5000]
        head.append(ws.title + " " + " ".join(str(v) for r in rows[:4] for v in r if v is not None))
        for r in rows:
            for j, v in enumerate(r):
                if hasattr(v, "year") and not isinstance(v, (int, float)):
                    r[j] = v.strftime("%d %B %Y") if hasattr(v, "strftime") else str(v)
        tables += _blocks(rows, ws.title, counter, "xlsx", n)
    return Document(format="xlsx", pages=len(wb.worksheets), tables=tables, text_head=" ".join(head)[:20000], reader={"name": "xlsx_layout"})


def from_text(text: str, delimiter: str | None = None) -> Document:
    import csv
    sample = text[:4000]
    delim = delimiter or ("\t" if sample.count("\t") > sample.count(",") else ";" if sample.count(";") > sample.count(",") else ",")
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim)]
    counter = [0]
    tables = _blocks([[(c.strip() or None) for c in r] for r in rows], "csv", counter, "csv", 1)
    return Document(format="csv", pages=1, tables=tables, text_head=text[:20000], reader={"name": "xlsx_layout", "delimiter": delim})


__all__ = ["from_xlsx", "from_text"]
