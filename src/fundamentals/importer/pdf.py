"""Figures from the text of a PDF financial report: the least reliable source, and always sent for review.

A PDF has no fields, only words placed on a page. This reader rebuilds the rows of each page from the words' positions, recognises the lines of the statements by name (the same aliases the
Enter-reports grid uses), finds the column headings that give each figure its period, and reads the unit the statement states ("in thousands, except per share"). It is deterministic, and it is
wrong in the ways a table in a PDF lets it be wrong: a heading split over lines, a line name that wraps, a two-column layout, a scanned page with no text at all. So:

* every figure carries its page and the text of the line it came from, so a person can look at the page;
* every period is marked as needing review; nothing from a PDF is stored without an approval;
* a figure whose line has fewer numbers than the heading has columns is not stored at all (which column it belongs to cannot be known);
* a scanned PDF (no text layer) is refused with that reason: there is no OCR here, and a guess at the digits would be a fabricated number.

Where the company also files XBRL (every US-listed company does), prefer it: the tagged figures are exact, and nothing here is needed.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date

from .. import manual
from ..errors import FundamentalsError
from .tabular import detect_units  # multilingual: "in thousands", 千円, 백만원, 单位：万元

MAX_PAGES = 400
NUMBER = re.compile(r"^\(?-?\$?\s?\d[\d,]*(\.\d+)?\)?%?$")
YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])}
MONTH_ABBR = {k[:3]: v for k, v in MONTHS.items()}
DATE_TEXT = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b", re.I)


def _words(data: bytes) -> list[list[tuple]]:
    """Per page, the words as ``(x0, y_mid, x1, text)``."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            fitz = None
    pages: list[list[tuple]] = []
    if fitz is not None:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as error:
            raise FundamentalsError(f"this PDF could not be opened ({type(error).__name__}): is it damaged or password-protected?") from None
        if doc.needs_pass:
            raise FundamentalsError("this PDF is password-protected")
        for page in list(doc)[:MAX_PAGES]:
            pages.append([(w[0], (w[1] + w[3]) / 2, w[2], w[4]) for w in page.get_text("words")])
        return pages
    try:
        import io

        import pdfplumber
    except ImportError:
        raise FundamentalsError("reading a PDF needs PyMuPDF or pdfplumber (pip install pymupdf); XBRL, CSV and Excel files can be imported without them") from None
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages[:MAX_PAGES]:
            pages.append([(w["x0"], (w["top"] + w["bottom"]) / 2, w["x1"], w["text"]) for w in page.extract_words()])
    return pages


def _rows(words: list[tuple], tolerance: float = 3.0) -> list[list[tuple]]:
    rows: list[list[tuple]] = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        if rows and abs(rows[-1][0][1] - w[1]) <= tolerance:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w[0]) for r in rows]


def _number(token: str) -> float | None:
    t = token.strip()
    if not NUMBER.match(t) or t.endswith("%"):
        return None
    neg = t.startswith("(") or t.startswith("-")
    t = t.strip("()-$ ").replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def _split(row: list[tuple]) -> tuple[str, list[float | None]]:
    """The label (words up to the first figure) and the figures of a row. A lone ``$`` is dropped; a dash is a figure that is not there."""
    tokens = [w[3] for w in row if w[3] not in ("$", "€", "£")]
    first = next((i for i, t in enumerate(tokens) if _number(t) is not None and i > 0 or t in ("—", "–")), None)
    if first is None:
        return " ".join(tokens), []
    figures = []
    for t in tokens[first:]:
        if t in ("—", "–", "-"):
            figures.append(None)
            continue
        v = _number(t)
        if v is not None:
            figures.append(v)
    return " ".join(tokens[:first]), figures


def _headings(text: str, year_end: tuple[int, int] | None) -> list[dict]:
    """The period columns named in a heading row: full dates ("December 31, 2023"), or bare years (which borrow the month and day of the fiscal year end the report names)."""
    out = []
    for m in DATE_TEXT.finditer(text):
        month = MONTH_ABBR[m.group(1)[:3].lower()]
        out.append({"period_end": date(int(m.group(3)), month, min(int(m.group(2)), 28 if month == 2 else 31)).isoformat(), "guessed": False})
    if len(out) >= 2:
        return out
    years = [int(y) for y in YEAR.findall(text)]
    if len(years) >= 2 and len(set(years)) == len(years):
        month, day = year_end or (12, 31)
        return [{"period_end": date(y, month, day).isoformat(), "guessed": year_end is None} for y in years]
    return out if out else []


def extract(data: bytes, ocr=None, language: str = "en") -> dict:
    """Read a PDF. A page with no text layer is sent to an OCR engine when one is installed (or the ``ocr`` engine given); without one the PDF is refused, because guessed digits would be invented numbers."""
    pages = _words(data)
    method = "pdf-text"
    if not any(pages):
        from . import ocr as ocr_module
        if ocr is None and not ocr_module.available():
            raise FundamentalsError(ocr_module.missing_message())
        pages, method = ocr_module.pages(data, language, ocr), "pdf-ocr"
        if not any(pages):
            raise FundamentalsError("OCR found no text on any page of this PDF")
    full_text = " ".join(" ".join(w[3] for w in p) for p in pages[:6])
    stated = detect_units(full_text)
    year_end = None
    m = re.search(r"(?:fiscal\s+)?year\s+ended\s+([A-Za-z]+)\s+(\d{1,2})", full_text, re.I)
    if m and m.group(1).lower()[:3] in MONTH_ABBR:
        year_end = (MONTH_ABBR[m.group(1).lower()[:3]], int(m.group(2)))
    found: dict[tuple, dict] = {}                                                                   # (column key) -> {code: (value, page, line)}
    columns: dict[tuple, dict] = {}
    skipped: list[dict] = []
    page_units: Counter = Counter()
    for pno, words in enumerate(pages, start=1):
        rows = _rows(words)
        header: list[dict] = []
        header_kind = "FY"
        page_text = " ".join(" ".join(w[3] for w in r) for r in rows[:12])
        u = detect_units(page_text)
        if u:
            page_units[u] += 1
        for row in rows:
            line = " ".join(w[3] for w in row)
            heads = _headings(line, year_end)
            if len(heads) >= 2:
                header = heads
                low = line.lower()
                header_kind = "Q" if re.search(r"three\s+months|quarter", low) else ("YTD" if re.search(r"six\s+months|nine\s+months|year[\s-]to[\s-]date", low) else "FY")
                continue
            if not header:
                continue
            label, figures = _split(row)
            code = manual.ALIASES.get(manual._norm(label)) if label else None
            if code is None or not figures:
                continue
            if len(figures) != len(header):
                skipped.append({"label": label, "page": pno, "message": f"{len(figures)} figure(s) under a heading of {len(header)} columns: not stored (which column each belongs to is not known)"})
                continue
            if header_kind == "YTD":
                skipped.append({"label": label, "page": pno, "message": "a year-to-date column: not stored as a quarter"})
                continue
            for h, v in zip(header, figures):
                if v is None:
                    continue
                key = (h["period_end"], header_kind)
                columns.setdefault(key, {"period_end": h["period_end"], "guessed": h["guessed"], "kind": header_kind})
                found.setdefault(key, {}).setdefault(code, (v, pno, line))
    if not found:
        raise FundamentalsError("no statement lines this application knows were found in the text of this PDF (Revenue, Net income, Total assets…) under a heading of periods: the report may be laid out in a way this reader cannot follow")
    units = stated or (page_units.most_common(1)[0][0] if page_units else None)
    notes = ["Read from the text of a PDF: every figure carries its page and its line, and every period needs a person's review before it is stored."]
    if method == "pdf-ocr":
        notes.insert(0, "This PDF was scanned: its text was recognised by an OCR engine, and a misread digit looks like a right one. Check every figure against the page image.")
    if not units:
        notes.append("The report does not state its unit in the pages read: amounts are read as units. Choose the unit below if they are in thousands or millions.")
    if len(page_units) > 1:
        notes.append("Different pages state different units (" + ", ".join(f"{k} on {v}" for k, v in page_units.items()) + "): check every statement.")
    if any(c["guessed"] for c in columns.values()):
        notes.append("A heading gave only years, so the period end was taken as December 31 (the report names no fiscal year end): check the dates.")
    periods = []
    for key in sorted(found):
        c = columns[key]
        end = date.fromisoformat(c["period_end"])
        kind = c["kind"]
        values = {code: v for code, (v, _, _) in found[key].items()}
        sources = {code: {"tag": None, "reported": v, "unit": units or "units", "page": pno, "text": line[:160], "method": method} for code, (v, pno, line) in found[key].items()}
        periods.append({"fiscal_period": "FY" if kind == "FY" else "Q" + str((end.month - 1) // 3 + 1), "period_end": c["period_end"], "fiscal_year": end.year, "filing_date": None, "currency": None, "values": values,
                        "sources": sources, "guessed": c["guessed"] or kind == "Q"})
    if any(p["fiscal_period"] != "FY" for p in periods):
        notes.append("A quarterly column's fiscal quarter number was taken from its calendar month; a company with an off-calendar fiscal year numbers its quarters differently: check it.")
    return {"periods": periods, "units": {"money": units or "units", "shares": "units"}, "currency": None, "company": {}, "notes": notes, "ytd_only": [], "unmapped": [], "skipped": skipped[:60],
            "method": method, "review_required": True, "units_stated": units is not None, "accession": None, "pages": len(pages)}
