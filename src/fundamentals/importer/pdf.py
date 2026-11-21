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
from .tabular import detect_currency, detect_units  # multilingual: "in thousands", 千円, 백만원, 单位：万元

MAX_PAGES = 400
NUMBER = re.compile(r"^\(?-?\$?\s?\d[\d,]*(\.\d+)?\)?%?$")
YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])}
MONTH_ABBR = {k[:3]: v for k, v in MONTHS.items()}
DATE_TEXT = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b", re.I)
DATE_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+((?:19|20)\d{2})\b", re.I)  # "31 December 2023" (Hong Kong, Singapore)
DATE_CJK = re.compile(r"((?:19|20)\d{2})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]?")                                   # "2023年12月31日"
DATE_NUM = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})(?!\d)")                                   # "2023.12.31" (Korea), "2023-12-31", "2023/12/31"
TERM = re.compile(r"제\s*(\d{1,3})\s*(?:\([^)]*\))?\s*기")                                                                     # Korean "제 55(당) 기": the 55th term of the company
RELATIVE = re.compile(r"(당기|本期|当期|本年度?|当年).{0,12}?(전기|前期|上期|上年同期|上年度?|前年)")                                      # "current / prior" column words
KIND_Q = re.compile(r"three\s+months|quarter|三個月|三个月|第[一二三四1-4]季|3개월", re.I)
KIND_H1 = re.compile(r"six\s+months|half[\s-]year|interim period|六個月|六个月|上半年|半期|6개월", re.I)
KIND_YTD = re.compile(r"nine\s+months|year[\s-]to[\s-]date|九個月|九个月|前三季|9개월", re.I)


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


def _day_month(text: str) -> tuple[int, int] | None:
    """The day and month a heading names without a year: "ended 31 March", "ended March 31", "止6月30日", "6월 30일"."""
    m = re.search(r"(?:ended|ending|as\s+at|as\s+of|au|zum|al|a)\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-zéûôç]{3,})\b(?!\s+\d{4})", text, re.I)
    if m and m.group(2).lower()[:3] in MONTH_ABBR:
        return MONTH_ABBR[m.group(2).lower()[:3]], int(m.group(1))
    m = re.search(r"(?:ended|ending|as\s+at|as\s+of)\s+([A-Za-z]{3,})\.?\s+(\d{1,2})\b(?!\s*,?\s*\d{4})", text, re.I)
    if m and m.group(1).lower()[:3] in MONTH_ABBR:
        return MONTH_ABBR[m.group(1).lower()[:3]], int(m.group(2))
    m = re.search(r"(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]", text)
    if m and not re.search(r"\d{4}\s*[年년]\s*\d{1,2}\s*[月월]", text):
        return int(m.group(1)), int(m.group(2))
    return None


def _headings(text: str, year_end: tuple[int, int] | None, ctx: dict | None = None) -> list[dict]:
    """The period columns named in a heading row: full dates ("December 31, 2023"), or bare years (which borrow the month and day of the fiscal year end the report names)."""
    found = []
    for m in DATE_TEXT.finditer(text):
        found.append((m.start(), int(m.group(3)), MONTH_ABBR[m.group(1)[:3].lower()], int(m.group(2))))
    for m in DATE_DMY.finditer(text):
        found.append((m.start(), int(m.group(3)), MONTH_ABBR[m.group(2)[:3].lower()], int(m.group(1))))
    for m in DATE_CJK.finditer(text):
        found.append((m.start(), int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in DATE_NUM.finditer(text):
        found.append((m.start(), int(m.group(1)), int(m.group(2)), int(m.group(3))))
    ctx = ctx or {}
    terms = list(dict.fromkeys(int(t) for t in TERM.findall(text)))                                  # "제 55 기 … 제 55(당) 기  제 54(전) 기": each term once, in order
    if len(terms) >= 2 and len(found) < 2 and ctx.get("terms"):                                       # "제 55(당) 기  제 54(전) 기": the terms name the columns, the dates come from the lines that introduce them
        known = ctx["terms"]
        top = max(known)
        cols = []
        for t in terms:
            d = known.get(t) or (date(known[top].year - (top - t), known[top].month, min(known[top].day, 28 if known[top].month == 2 else known[top].day)) if top - t < 40 else None)
            if d is None:
                return []
            cols.append({"period_end": d.isoformat(), "guessed": t not in known})
        return cols
    if len(found) < 2 and RELATIVE.search(text) and ctx.get("latest"):                               # "当期 / 前期": the latest date the report names, and a year before it
        d = ctx["latest"]
        return [{"period_end": d.isoformat(), "guessed": True}, {"period_end": date(d.year - 1, d.month, min(d.day, 28 if d.month == 2 else d.day)).isoformat(), "guessed": True}]
    out = []
    seen = set()
    for _, y, mo, d in sorted(found):                                                                 # a date named twice ("…ended 30 June 2024" then the column "30 June 2024") is one column
        if 1 <= mo <= 12 and (y, mo, d) not in seen:
            seen.add((y, mo, d))
            out.append({"period_end": date(y, mo, max(1, min(d, 28 if mo == 2 else 30 if mo in (4, 6, 9, 11) else 31))).isoformat(), "guessed": False})
    if len(out) >= 2:
        return out
    if len(out) == 1:                                                                                 # "Six months ended 30 June 2024 2023": the later years share the first date's day and month
        first = date.fromisoformat(out[0]["period_end"])
        rest = [int(y) for y in YEAR.findall(text[text.rfind(str(first.year)) + 4:])]
        if rest and first.year not in rest and len(set(rest)) == len(rest):
            return out + [{"period_end": date(y, first.month, first.day).isoformat(), "guessed": False} for y in rest]
    years = [int(y) for y in YEAR.findall(text)]
    if len(years) >= 2 and len(set(years)) == len(years):
        named = _day_month(text)                                                                     # "Three months ended 31 March" above "2024  2023": the day and month are named once
        month, day = named or year_end or (12, 31)
        return [{"period_end": date(y, month, min(day, 28 if month == 2 else 30 if month in (4, 6, 9, 11) else 31)).isoformat(), "guessed": named is None and year_end is None} for y in years]
    return out if out else []


def _context(pages) -> dict:
    """What the report says outside its column headings: which date each numbered term ends on ("제 55 기: 2023.12.31 현재"), and the latest date it names."""
    lines = []
    for words in pages[:40]:
        for row in _rows(words):
            lines.append(" ".join(w[3] for w in row))
    return context_from_lines(lines)


def context_from_lines(lines) -> dict:
    terms: dict[int, date] = {}
    latest = None
    for line in lines:
        dates = []
        for rx in (DATE_NUM, DATE_CJK):
            for m in rx.finditer(line):
                try:
                    dates.append((m.start(), date(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
                except ValueError:
                    pass
        for m in DATE_DMY.finditer(line):
            try:
                dates.append((m.start(), date(int(m.group(3)), MONTH_ABBR[m.group(2)[:3].lower()], int(m.group(1)))))
            except ValueError:
                pass
        if not dates:
            continue
        dates.sort()
        top = max(d for _, d in dates)
        latest = top if latest is None else max(latest, top)
        term = TERM.search(line)
        if term and len(TERM.findall(line)) == 1:
            terms.setdefault(int(term.group(1)), dates[-1][1])
    return {"terms": terms, "latest": latest}


def _code(label: str):
    """The statement line a label names. A trailing note in brackets ("당기순이익(손실)", "Net income (loss)") and a leading outline number ("Ⅰ. 매출액", "1. Revenue") do not change what the line is."""
    candidates = [label]
    stripped = re.sub(r"\s*[(（][^)）]*[)）]\s*$", "", label).strip()
    candidates.append(stripped)
    candidates.append(re.sub(r"^\s*(?:[IVXⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|\d{1,2}|[가-힣]|[a-z])\s*[.)、]\s*", "", stripped, flags=re.I))
    for c in candidates:
        code = manual.ALIASES.get(manual._norm(c)) if c else None
        if code:
            return code
    return None


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
    if year_end is None:
        m = re.search(r"year\s+ended\s+(\d{1,2})\s+([A-Za-z]+)", full_text, re.I)
        if m and m.group(2).lower()[:3] in MONTH_ABBR:
            year_end = (MONTH_ABBR[m.group(2).lower()[:3]], int(m.group(1)))
    currency = detect_currency(full_text)
    ctx = _context(pages)
    found: dict[tuple, dict] = {}                                                                   # (column key) -> {code: (value, page, line)}
    columns: dict[tuple, dict] = {}
    skipped: list[dict] = []
    page_units: Counter = Counter()
    headerless = False
    for attempt in (0, 1):
      if attempt == 1:
        if found or not ctx["latest"]:
            break
        headerless = True                                                                           # no column headings were understood: read two-figure lines as [latest date, a year earlier]
        skipped = []
      for pno, words in enumerate(pages, start=1):
          rows = _rows(words)
          d0 = ctx["latest"]
          header: list[dict] = ([{"period_end": d0.isoformat(), "guessed": True}, {"period_end": date(d0.year - 1, d0.month, min(d0.day, 28 if d0.month == 2 else d0.day)).isoformat(), "guessed": True}] if headerless else [])
          header_kind = "FY"
          page_text = " ".join(" ".join(w[3] for w in r) for r in rows[:12])
          u = detect_units(page_text)
          if u:
              page_units[u] += 1
          for row in rows:
              line = " ".join(w[3] for w in row)
              heads = _headings(line, year_end, ctx)
              if len(heads) >= 2:
                  header = heads
                  low = line.lower()
                  header_kind = "Q" if KIND_Q.search(low) else ("H1" if KIND_H1.search(low) else ("YTD" if KIND_YTD.search(low) else "FY"))
                  continue
              if not header:
                  continue
              label, figures = _split(row)
              code = _code(label) if label else None
              if code is None or not figures:
                  continue
              if len(figures) == len(header) + 1 and figures[0] is not None and figures[0] == int(figures[0]) and 0 < figures[0] < 100:
                  figures = figures[1:]                                                                # a "Notes" reference column ("Revenue  4  1,234  1,100")
              if len(figures) != len(header):
                  skipped.append({"label": label, "page": pno, "message": f"{len(figures)} figure(s) under a heading of {len(header)} columns: not stored (which column each belongs to is not known)"})
                  continue
              if header_kind == "YTD":
                  skipped.append({"label": label, "page": pno, "message": "a nine-month / year-to-date column: not stored as a quarter"})
                  continue
              for h, v in zip(header, figures):
                  if v is None:
                      continue
                  key = (h["period_end"], header_kind)
                  columns.setdefault(key, {"period_end": h["period_end"], "guessed": h["guessed"], "kind": header_kind})
                  found.setdefault(key, {}).setdefault(code, (v, pno, line))
    for key in [k for k in found if k[1] == "FY"]:                                                    # a balance sheet at 30 June carries no "six months" in its heading: join it to the interim column of that date
        for other in ("H1", "Q"):
            if (key[0], other) in found:
                for code, item in found.pop(key).items():
                    found[(key[0], other)].setdefault(code, item)
                columns.pop(key, None)
                break
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
    if headerless:
        notes.append("No column headings of periods could be read, so each statement line's two figures were taken as the latest date the report names and the year before it: check which column is which.")
    if any(c["guessed"] for c in columns.values()):
        notes.append("A heading gave only years, so the period end was taken as December 31 (the report names no fiscal year end): check the dates.")
    periods = []
    for key in sorted(found):
        c = columns[key]
        end = date.fromisoformat(c["period_end"])
        kind = c["kind"]
        values = {code: v for code, (v, _, _) in found[key].items()}
        sources = {code: {"tag": None, "reported": v, "unit": units or "units", "page": pno, "text": line[:160], "method": method} for code, (v, pno, line) in found[key].items()}
        periods.append({"fiscal_period": "FY" if kind == "FY" else "H1" if kind == "H1" else "Q" + str((end.month - 1) // 3 + 1), "period_end": c["period_end"], "fiscal_year": end.year, "filing_date": None, "currency": currency, "values": values,
                        "sources": sources, "guessed": c["guessed"] or kind == "Q"})
    if any(p["fiscal_period"] != "FY" for p in periods):
        notes.append("A quarterly column's fiscal quarter number was taken from its calendar month; a company with an off-calendar fiscal year numbers its quarters differently: check it.")
    return {"periods": periods, "units": {"money": units or "units", "shares": "units"}, "currency": currency, "company": {}, "notes": notes, "ytd_only": [], "unmapped": [], "skipped": skipped[:60],
            "method": method, "review_required": True, "units_stated": units is not None, "accession": None, "pages": len(pages)}
