"""Figures from an HTML report that carries no XBRL: the statements as tables in a web page.

Annual reports on a company's own site, and exchange filings before XBRL, are often HTML. Each ``<table>`` is read with BeautifulSoup (the page is decoded by its declared character set, then by the usual
Japanese, Korean and Chinese encodings, since a Shift-JIS or GBK page read as UTF-8 is garbage), turned into tab-separated text and given to the same table reader as a spreadsheet; the tables of one page (an
income statement, a balance sheet, a cash flow statement) are put together by period. Tables that hold no known lines are ignored. The unit is read from the page's own words ("in millions of yen", "百万円",
"단위: 백만원", "单位：万元"), and a page that states none is said to.
"""

from __future__ import annotations

import re

from .. import manual
from ..errors import FundamentalsError
from . import tabular

ENCODINGS = ("utf-8", "shift_jis", "cp932", "euc-jp", "euc-kr", "cp949", "gb18030", "big5", "cp1252", "latin-1")

def _noise(text: str) -> int:
    """How wrong a decoding looks: replacement characters and half-width katakana, which is what a Korean or Chinese page reads as when decoded as Shift-JIS."""
    return text.count("\ufffd") * 5 + sum(1 for c in text if "\uff61" <= c <= "\uff9f")


def decode(data: bytes) -> str:
    """The page as text: by the character set it declares, else the candidate that decodes without noise (several encodings accept the same bytes; the wrong one leaves the marks :func:`_noise` counts)."""
    head = data[:4000].decode("ascii", "ignore").lower()
    m = re.search(r"charset=['\"]?([\w\-]+)", head)
    if m:
        try:
            return data.decode(m.group(1))
        except (LookupError, UnicodeDecodeError):
            pass
    try:                                                                                             # charset-normalizer (a dependency of requests) tells Korean from Japanese from Chinese far better than a try-in-order
        from charset_normalizer import from_bytes
        found = from_bytes(data).best()
        if found is not None:
            return str(found)
    except ImportError:
        pass
    best, best_score = None, None
    for enc in ENCODINGS:
        try:
            text = data.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
        score = _noise(text)
        if best_score is None or score < best_score:
            best, best_score = text, score
        if score == 0 and enc == "utf-8":
            break
    return best if best is not None else data.decode("utf-8", "replace")


def units_of(text: str) -> str | None:
    return tabular.detect_units(text)


def extract(data: bytes) -> dict:
    from bs4 import BeautifulSoup
    html = decode(data)
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    units = units_of(text[:8000])
    merged: dict[tuple, dict] = {}
    notes, n_tables = [], 0
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)).replace("\xa0", " ").strip() for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append("\t".join(cells))
        if len(rows) < 3:
            continue
        try:
            parsed = manual.parse_table("\n".join(rows))
        except FundamentalsError:
            continue
        n_tables += 1
        for j, col in enumerate(parsed["columns"]):
            key = (col.get("period_end"), col.get("fiscal_period"))
            slot = merged.setdefault(key, {"col": col, "values": {}})
            for code, row in parsed["values"].items():
                v = row[j] if j < len(row) else None
                if v is not None:
                    slot["values"].setdefault(code, v)
    periods = []
    for (end, kind), slot in sorted(merged.items(), key=lambda kv: kv[0][0] or ""):
        if not slot["values"]:
            continue
        c = slot["col"]
        periods.append({"fiscal_period": kind or "FY", "period_end": end, "fiscal_year": c.get("fiscal_year"), "filing_date": c.get("filing_date"), "currency": c.get("currency"), "values": slot["values"],
                        "sources": {code: {"tag": None, "reported": v, "unit": units or "units", "method": "html-table"} for code, v in slot["values"].items()}, "guessed": bool(c.get("guessed"))})
    if not periods:
        raise FundamentalsError("no table of the page holds known statement lines under a heading of periods")
    notes.append(f"Read from {n_tables} table(s) of an HTML page: check each figure against the page.")
    if not units:
        notes.append("The page does not state its unit: amounts are read as units. Choose the unit below if they are in thousands or millions.")
    return {"periods": periods, "units": {"money": units or "units", "shares": "units"}, "currency": None, "company": {}, "notes": notes, "ytd_only": [], "unmapped": [], "method": "html-table",
            "review_required": True, "units_stated": units is not None, "accession": None}
