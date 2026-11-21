"""HTML (and inline-XBRL pages read as plain tables) to tables. Each ``<table>`` becomes a grid: spacer columns and the stray ``$``, ``)`` and ``%`` cells that web statements put in columns of their own
are folded into the figure beside them; the title is the caption or the nearest heading text above the table."""

from __future__ import annotations

import re

from ..ir import Cell, Document, Table
from ._cells import to_number

MAX_TABLES = 600
_STRAY = re.compile(r"^[\s$€£¥￥₩)%]*$")


def _text(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True).replace(" ", " ")).strip()


def _grid(table) -> list[list[str]]:
    rows = []
    for tr in table.find_all("tr"):
        row: list[str] = []
        for td in tr.find_all(["td", "th"], recursive=False) or tr.find_all(["td", "th"]):
            text = _text(td)
            span = max(1, min(int(td.get("colspan", 1) or 1), 20)) if str(td.get("colspan", "1")).isdigit() else 1
            row.append(text)
            row.extend([""] * (span - 1))
        if any(row):
            rows.append(row)
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def _fold(rows: list[list[str]]) -> list[list[str]]:
    """Merge ``(1,234`` + ``)`` and ``12`` + ``%`` fragments, drop columns that hold nothing but spacers and symbols."""
    out = []
    for r in rows:
        r = list(r)
        for j in range(1, len(r)):
            if r[j].strip() == ")" and j > 0:
                k = j - 1
                while k > 0 and not r[k].strip():
                    k -= 1
                if r[k].strip().startswith("("):
                    r[k] = r[k].strip() + ")"
                    r[j] = ""
        for j in range(len(r) - 1):
            if r[j].strip() in ("$", "€", "£", "¥", "₩") and r[j + 1].strip():
                r[j] = ""
        out.append(r)
    keep = [j for j in range(len(out[0])) if j == 0 or any(r[j].strip() and not _STRAY.match(r[j]) for r in out)] if out else []
    return [[r[j] for j in keep] for r in out]


def _context_above(table, limit: int = 3) -> str:
    out = []
    for prev in table.find_all_previous(string=True, limit=40):
        s = re.sub(r"\s+", " ", str(prev).replace(" ", " ")).strip()
        if len(s) > 2 and prev.parent.name not in ("script", "style", "td", "th"):
            out.append(s)
            if len(out) >= limit:
                break
    return " ".join(reversed(out))


def extract(data: bytes, language: str = "en") -> Document:
    from bs4 import BeautifulSoup

    from ..html import decode
    soup = BeautifulSoup(decode(data), "lxml")
    for s in soup(["script", "style"]):
        s.decompose()
    tables: list[Table] = []
    for n, tbl in enumerate(soup.find_all("table")[:MAX_TABLES], start=1):
        rows = _fold(_grid(tbl))
        if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
            continue
        cap = tbl.find("caption")
        above = _context_above(tbl)
        t = Table(table_id=f"t{n}", page=None, kind="html", title=(_text(cap) if cap else "") or above[-160:], context=above, unit_text=above + " " + (_text(cap) if cap else ""))
        for i, r in enumerate(rows):
            for j, text in enumerate(r):
                if not text.strip():
                    continue
                num, dash = to_number(text) if j > 0 else (None, False)
                t.cells.append(Cell(text.strip(), i, j, None, None, num, dash))
        first_fig = next((i for i, r in enumerate(rows) if any(c.is_number or c.dash for c in t.row(i))), len(rows))
        t.header_rows = list(range(first_fig))
        tables.append(t)
    text = _text(soup)[:20000]
    return Document(format="html", pages=0, tables=tables, text_head=text, language=language, reader={"name": "html_layout"})


__all__ = ["extract"]
