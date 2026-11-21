"""The universal intermediate representation (IR) of a financial document.

Physical extraction (reading a PDF, a scan, a web page, a workbook) produces an IR and knows nothing about accounting. Everything after it (statement classification, taxonomy mapping, matching,
checks) reads only the IR, so a new file format needs one new reader, and a new accounting standard needs one new mapping, and neither touches the other.

An IR is plain data and round-trips through JSON (:meth:`Document.to_dict` / :meth:`Document.from_dict`), so it can be stored next to the original bytes and a later version of the mapping can be
run over it without reading the file again.

* :class:`Cell` – one cell of a table: its text, its row and column, the page and the box (in page units; spreadsheets use the cell reference) it was read from, and the number it holds, if any.
* :class:`Table` – cells in a grid with a title, the page it is on, and what the reader could tell about its header (column headings, period words), unit and currency.
* :class:`Section` – a titled run of pages ("Consolidated statement of financial position") with the tables in it and the consolidation scope it states.
* :class:`Fact` – a value that is already structured (XBRL): concept, value, period, unit, scope.
* :class:`Document` – all of the above, plus the format, language, page count, and what the reader could not do (warnings).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

BBox = tuple[float, float, float, float]


@dataclass
class Cell:
    text: str
    row: int
    col: int
    page: int | None = None
    bbox: BBox | None = None
    number: float | None = None                 # the number the text holds ((1,234) is -1234); None for a label, a heading, a dash
    dash: bool = False                          # "-", "—": a figure that is not there (kept apart from a missing cell)
    percent: bool = False                       # "(1.2)%": a percentage, never a reported amount
    span: int = 1                               # a heading over several columns covers this many, starting at ``col``

    @property
    def is_number(self) -> bool:
        return self.number is not None


@dataclass
class Table:
    table_id: str
    page: int | None
    cells: list[Cell] = field(default_factory=list)
    title: str = ""
    kind: str = ""                              # pdf-layout | pdf-ocr | html | xlsx | csv | docling
    bbox: BBox | None = None
    header_rows: list[int] = field(default_factory=list)
    unit_text: str = ""                         # words near the table that state its unit or currency
    context: str = ""                           # the heading text of the page or sheet above it (scope, statement name)
    raw_words: list = field(default_factory=list)        # (text, box) of every word of the table region, as read: the geometry is never discarded
    column_extents: dict = field(default_factory=dict)   # column -> (x0, x1) span of its figures
    pages: list = field(default_factory=list)            # the pages a statement that continues over several pages covers

    def n_rows(self) -> int:
        return 1 + max((c.row for c in self.cells), default=-1)

    def n_cols(self) -> int:
        return 1 + max((c.col for c in self.cells), default=-1)

    def row(self, r: int) -> list[Cell]:
        return sorted((c for c in self.cells if c.row == r), key=lambda c: c.col)

    def rows(self) -> list[list[Cell]]:
        by: dict[int, list[Cell]] = {}
        for c in self.cells:
            by.setdefault(c.row, []).append(c)
        return [sorted(by[r], key=lambda c: c.col) for r in sorted(by)]

    def label(self, r: int) -> str:
        cells = [c for c in self.row(r) if c.col == 0]
        return cells[0].text.strip() if cells else ""

    def text(self) -> str:
        return " ".join(c.text for c in sorted(self.cells, key=lambda c: (c.row, c.col)))


@dataclass
class Section:
    title: str
    page_start: int | None = None
    page_end: int | None = None
    table_ids: list[str] = field(default_factory=list)
    scope: str = "unspecified"                  # consolidated | parent | unspecified


@dataclass
class Fact:
    concept: str
    value: float
    period_end: str
    period_start: str | None = None
    unit: str | None = None                     # an ISO currency, "shares", "pure"
    scope: str = "consolidated"                 # consolidated | parent | segment
    dimensions: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)


@dataclass
class Document:
    format: str                                 # pdf | pdf-scan | html | xlsx | csv | xbrl
    source_name: str = ""
    language: str = "en"
    pages: int = 0
    tables: list[Table] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    text_head: str = ""                         # the first pages' text: where a report states its unit, currency, fiscal year end, scope
    warnings: list[str] = field(default_factory=list)
    reader: dict = field(default_factory=dict)  # which reader, which version, which OCR engine

    def table(self, table_id: str) -> Table | None:
        return next((t for t in self.tables if t.table_id == table_id), None)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        tables = [Table(**{**t, "column_extents": {int(k): tuple(v) for k, v in (t.get("column_extents") or {}).items()}, "raw_words": [(w[0], tuple(w[1])) for w in (t.get("raw_words") or [])], "cells": [Cell(**{**c, "bbox": tuple(c["bbox"]) if c.get("bbox") else None}) for c in t["cells"]], "bbox": tuple(t["bbox"]) if t.get("bbox") else None}) for t in d.get("tables", [])]
        return cls(**{**d, "tables": tables, "sections": [Section(**s) for s in d.get("sections", [])], "facts": [Fact(**f) for f in d.get("facts", [])]})


__all__ = ["Cell", "Table", "Section", "Fact", "Document"]
