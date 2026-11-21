"""PDF (text layer or OCR) to tables with coordinates.

The words of a page, with their boxes, are grouped into rows by height and into segments by the gaps between them (a label of several words is one segment; so is "31 December 2023"). A run of rows
that carry figures is a table; its columns are found from where the figures *end* (statements right-align numbers), so a figure that is missing from one row stays in its own column and a row with
fewer figures than the heading has columns is read correctly, which is what the line-by-line reader could not do. Rows above the figures that name periods are the table's header; the text above
that is its title.
"""

from __future__ import annotations

import re

from ..ir import Cell, Document, Table
from ._cells import to_number

GAP = 7.0                                   # horizontal gap (page units) that separates two segments of a row
COLUMN_TOL = 14.0                           # figures whose right edges lie within this of each other are in one column
MAX_PAGES = 400
HEADER_WORDS = re.compile(r"\b(19|20)\d{2}\b|[年년]|月|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b(note|notes|restated|unaudited|audited|three|six|nine|twelve|months?|year)\b|주석|附注|附註|注記|当期|前期|당기|전기|本期|上期|제\s*\d+", re.I)


GROUP_WORDS = re.compile(r"^(?:group|company|consolidated|parent|separate|bank|in\s+(?:millions|thousands|billions)|\(?(?:krw|usd|rmb|cny|jpy|eur|hkd|sgd|us\$|s\$|hk\$)\b)", re.I)
PCT = re.compile(r"^\(?[-+−]?\d[\d,]*(?:\.\d+)?\)?\s*%$")
PERIOD_STRONG = re.compile(r"\b(ended|ending|months?|as\s+at|as\s+of)\b|止|終了|까지|부터|截至", re.I)
PERIOD_PHRASE = re.compile(r"\b(ended|ending|months?|year|quarter|half)\b|止|終了|까지|부터|截至|décembre|clos", re.I)


def _segments(row: list[tuple], gap_scale: float = 1.0) -> list[dict]:
    segs: list[dict] = []
    widths = sorted((x1 - x0) / max(len(text), 1) for x0, _, x1, text in row)
    gap = gap_scale * (max(GAP, 1.3 * widths[len(widths) // 2]) if widths else GAP)            # a space is a fraction of a character; a column gap is several characters (wide CJK characters widen both)
    for x0, y, x1, text in row:
        if segs and x0 - segs[-1]["x1"] <= gap:
            s = segs[-1]
            s["text"] += " " + text
            s["x1"] = x1
        else:
            segs.append({"x0": x0, "x1": x1, "y": y, "text": text})
    merged: list[dict] = []
    for sg in segs:                                                                                    # a currency symbol standing alone before its figure ("$   274,515") belongs to the figure
        if merged and merged[-1]["text"] in ("$", "€", "£", "¥", "￥", "₩", "US$", "HK$", "S$", "R$", "A$", "C$") and sg["x0"] - merged[-1]["x1"] <= 60 and to_number(sg["text"])[0] is not None:
            merged[-1] = {**sg, "text": merged[-1]["text"] + " " + sg["text"], "x0": merged[-1]["x0"]}
            continue
        merged.append(sg)
    segs = merged
    for s in segs:
        s["number"], s["dash"] = to_number(s["text"])
        s["pct"] = bool(PCT.match(s["text"].strip())) and s["number"] is None
        s["fig"] = s["number"] is not None or s["dash"] or s["pct"]                          # a percentage holds a column position but no reported amount
    return segs


def _is_header_row(segs: list[dict]) -> bool:
    """Period headings: the text of every segment after the first names years, dates, notes or period words (a year alone is a number to ``to_number``, so it is judged by its words)."""
    rest = [s for s in segs if s["x0"] > segs[0]["x0"] + 1] if len(segs) > 1 and segs[0]["number"] is None and not segs[0]["dash"] else segs
    if not rest:
        return False
    return all(HEADER_WORDS.search(s["text"]) for s in rest)


def _line_height(rows: list[list[tuple]]) -> float:
    ys = sorted({round(r[0][1], 1) for r in rows if r})
    gaps = [b - a for a, b in zip(ys, ys[1:]) if 4 < b - a < 40]
    return sorted(gaps)[len(gaps) // 2] if gaps else 12.0


CONNECTORS = re.compile(r"(?:\b(?:and|of|from|for|in|on|to|the|or|with|less|net|by|at|other|et|de|des|du|la|le|und|der|die|y|del|los|las)|[,\-–/&(])$", re.I)


def _join_wrapped(seg_rows, rows, ys, lh, has_figures):
    """A long label that wraps: its first line has no figures and the next line, which has them, starts in lower case or the first line ends mid-phrase ("Profit from" / "operations  724  314"). The two
    lines are one label; read separately, the second half ("assets") would match a line of its own."""
    keep_seg, keep_rows, keep_ys = [], [], []
    i = 0
    while i < len(seg_rows):
        cur = seg_rows[i]
        if (i + 1 < len(seg_rows) and len(cur) == 1 and cur[0]["number"] is None and not cur[0]["dash"] and not has_figures(cur) and has_figures(seg_rows[i + 1])
                and 0 < ys[i + 1] - ys[i] <= 1.7 * lh and abs(seg_rows[i + 1][0]["x0"] - cur[0]["x0"]) < 12):
            nxt = seg_rows[i + 1]
            first = nxt[0]["text"]
            if first[:1].islower() or CONNECTORS.search(cur[0]["text"].strip()):
                merged = [dict(nxt[0], text=cur[0]["text"].strip() + " " + first, x0=min(cur[0]["x0"], nxt[0]["x0"]))] + nxt[1:]
                keep_seg.append(merged)
                keep_rows.append(rows[i + 1])
                keep_ys.append(ys[i + 1])
                i += 2
                continue
        keep_seg.append(cur)
        keep_rows.append(rows[i])
        keep_ys.append(ys[i])
        i += 1
    return keep_seg, keep_rows, keep_ys


_DATE_TOKEN = re.compile(r"(?:\d{1,2}\s+[^\W\d_]{3,12}\.?,?\s+(?:19|20)\d{2})|(?:[^\W\d_]{3,12}\.?\s+\d{1,2},?\s+(?:19|20)\d{2})|(?:(?:19|20)\d{2}\s*[年년]\s*\d{1,2}\s*[月월]\s*\d{1,2}\s*[日일]?)|(?:(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2})|(?:(?:19|20)\d{2}\s*[年년]?)")


def _split_heading(text: str, n: int) -> list[str] | None:
    """A heading segment that merged the headings of ``n`` columns ("March 31, 2019 March 31, 2020", "2020 2019", "FY2019 FY2020"): its pieces, if it holds exactly ``n`` dates or years and nothing else of weight."""
    toks = [m.group(0) for m in _DATE_TOKEN.finditer(text)]
    if len(toks) == n:
        rest = text
        for t in toks:
            rest = rest.replace(t, " ", 1)
        if len(re.sub(r"[\s,/\-–()]|FY|fy|年度|[Yy]ears?|ended|Ended|YEAR|Year", "", rest)) <= 6:
            return toks
    terms = [m.group(0) for m in re.finditer(r"제\s*\d{1,3}\s*(?:\([^)]*\))?\s*기", text)]                       # Korean numbered terms: "제 55(당) 기제 54(전) 기"
    if len(terms) == n:
        return terms
    words = text.split()
    if len(words) == n and all(HEADER_WORDS.search(w) or len(w) <= 8 for w in words):
        return words
    return None


def _build_page(page_no: int, words: list[tuple], kind: str, counter: list[int], gap_scale: float = 1.0, row_tolerance: float = 3.0) -> tuple[list[Table], str]:
    from ..pdf import _rows
    rows = _rows(words, row_tolerance)
    seg_rows = [_segments(r, gap_scale) for r in rows]
    lh = _line_height(rows)
    ys = [r[0][1] for r in rows]
    head_text = " ".join(" ".join(s["text"] for s in sr) for sr in seg_rows[:4])

    def has_figures(sr):
        return any(s["fig"] and not HEADER_WORDS.search(s["text"]) for s in sr)

    seg_rows, rows, ys = _join_wrapped(seg_rows, rows, ys, lh, has_figures)
    data_idx = [i for i, sr in enumerate(seg_rows) if has_figures(sr) and not _is_header_row(sr)]
    blocks: list[list[int]] = []
    for i in data_idx:
        if blocks and ys[i] - ys[blocks[-1][-1]] <= 3.2 * lh:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    tables: list[Table] = []
    for block in blocks:
        first, last = block[0], block[-1]
        body = [i for i in range(first, last + 1)]
        numeric_segs = [s for i in body for s in seg_rows[i] if s["fig"] and s is not seg_rows[i][0]] or [s for i in body for s in seg_rows[i] if s["fig"]]

        def cluster(values):
            out: list[list[float]] = []
            for e in sorted(values):
                if out and e - out[-1][-1] <= COLUMN_TOL:
                    out[-1].append(e)
                else:
                    out.append([e])
            return out

        right, left = cluster(s["x1"] for s in numeric_segs), cluster(s["x0"] for s in numeric_segs)
        edge = "x0" if len(left) < len(right) else "x1"                                              # statements right-align figures; a left-aligned column is found by its left edge
        clusters = left if edge == "x0" else right
        centers = [sum(c) / len(c) for c in clusters]

        def col_of(s):
            x = s[edge]
            return 1 + min(range(len(centers)), key=lambda k: abs(centers[k] - x)) if centers else 1

        extents: dict[int, list[float]] = {}                                                          # the horizontal span of the figures of each column
        for sg in numeric_segs:
            k = col_of(sg)
            e = extents.setdefault(k, [sg["x0"], sg["x1"]])
            e[0], e[1] = min(e[0], sg["x0"]), max(e[1], sg["x1"])

        def over(sg, row=None):
            """The columns a heading segment sits above: those whose figure span it overlaps (a heading is centred, right-aligned or left-aligned over its figures, or spans several columns). A heading that
            overlaps none (a centred group heading between two columns) takes the columns that are nearer to it than to any other heading of its row."""
            hit = [k for k, (a0, a1) in sorted(extents.items()) if min(sg["x1"], a1 + 4) - max(sg["x0"], a0 - 4) > 0]
            group = bool(GROUP_WORDS.match(sg["text"].strip()))
            if (hit and not group) or not extents:
                return hit
            cx = (sg["x0"] + sg["x1"]) / 2
            others = [(o["x0"] + o["x1"]) / 2 for o in (row or [sg]) if GROUP_WORDS.match(o["text"].strip()) or not [k for k, (a0, a1) in extents.items() if min(o["x1"], a1 + 4) - max(o["x0"], a0 - 4) > 0]]
            mine = [k for k, (a0, a1) in sorted(extents.items()) if min(others, key=lambda c: abs(c - (a0 + a1) / 2)) == cx]
            return mine or [min(extents, key=lambda k: abs((extents[k][0] + extents[k][1]) / 2 - cx))]

        def short(sr):
            return all(len(sg["text"]) <= 40 for sg in sr)

        header_idx = []
        j = first - 1
        while j >= 0 and ys[first] - ys[j] <= 7.5 * lh and len(header_idx) < 6:
            sr = seg_rows[j]
            geometric = (not has_figures(sr) or _is_header_row(sr)) and short(sr) and any(set(over(sg, sr)) for sg in sr if sg["x0"] > 120)
            if geometric or (_is_header_row(sr) and (any(sg["x0"] > 120 for sg in sr) or PERIOD_STRONG.search(" ".join(sg["text"] for sg in sr)))):
                header_idx.append(j)
            elif header_idx and not (len(sr) == 1 and PERIOD_STRONG.search(sr[0]["text"])):
                break
            j -= 1
        header_idx.reverse()
        title_rows = [k for k in range(max(0, (header_idx[0] if header_idx else first) - 3), header_idx[0] if header_idx else first) if k not in header_idx and not has_figures(seg_rows[k])]
        counter[0] += 1
        t = Table(table_id=f"t{counter[0]}", page=page_no, kind=kind, title=" ".join(" ".join(s["text"] for s in seg_rows[k]) for k in title_rows).strip(),
                  context=head_text, header_rows=list(range(len(header_idx))))
        t.raw_words = [(w[3], (w[0], w[1] - 5, w[2], w[1] + 5)) for i in ([*title_rows, *header_idx] + body) for w in rows[i]]            # the words and their boxes are kept: nothing about the geometry is thrown away
        t.column_extents = {k: tuple(v) for k, v in extents.items()}
        r = 0
        for hi in header_idx:
            for s in seg_rows[hi]:
                box = (s["x0"], s["y"] - 5, s["x1"], s["y"] + 5)
                if not extents or s["x1"] < min(v[0] for v in extents.values()) - 10 or (s is seg_rows[hi][0] and len(seg_rows[hi]) > 1 and s["x1"] < centers[0] - 40):
                    t.cells.append(Cell(s["text"], r, 0, page_no, box))
                    continue
                cols = over(s, seg_rows[hi])
                if len(cols) > 1:
                    parts = _split_heading(s["text"], len(cols))
                    if parts:
                        for k, part in zip(cols, parts):
                            t.cells.append(Cell(part, r, k, page_no, box))
                        continue
                    t.cells.append(Cell(s["text"], r, cols[0], page_no, box, span=len(cols)))                  # a heading over several columns ("Group", "In millions of Korean won")
                else:
                    t.cells.append(Cell(s["text"], r, cols[0], page_no, box))
            r += 1
        for i in body:
            sr = seg_rows[i]
            for k, s in enumerate(sr):
                is_fig = s["fig"] and not (k == 0 and len(sr) > 1 and s["number"] is None and not s["pct"])
                if is_fig:
                    t.cells.append(Cell(s["text"], r, col_of(s), page_no, (s["x0"], s["y"] - 5, s["x1"], s["y"] + 5), s["number"], s["dash"], percent=s["pct"]))
                else:
                    t.cells.append(Cell(s["text"], r, 0, page_no, (s["x0"], s["y"] - 5, s["x1"], s["y"] + 5)))
            r += 1
        t.unit_text = " ".join([t.title, head_text] + [c.text for c in t.cells if c.row in t.header_rows])
        tables.append(t)
    return tables, head_text


def extract(data: bytes, ocr=None, language: str = "en") -> Document:
    """Read a PDF into an IR. A page with no text layer goes to an OCR engine if one is installed (``ocr`` or whatever :mod:`..ocr` finds); without one the document says it could not."""
    from .. import ocr as ocr_module
    from ..pdf import _words
    pages = _words(data)
    kind, fmt = "pdf-layout", "pdf"
    warnings: list[str] = []
    if not any(pages):
        if ocr is None and not ocr_module.available():
            doc = Document(format="pdf-scan", pages=len(pages), warnings=["scanned PDF: no text layer, and no OCR engine is installed (" + ocr_module.missing_message() + ")"], reader={"name": "pdf_layout", "ocr": None})
            return doc
        scale = 72.0 / ocr_module.RENDER_DPI                                              # an OCR engine reports pixels of the rendered page; the layout works in points
        pages = [[(w[0] * scale, w[1] * scale, w[2] * scale, w[3]) for w in page] for page in ocr_module.pages(data, language, ocr)]
        kind, fmt = "pdf-ocr", "pdf-scan"
        warnings.append("recognised by OCR: a misread digit looks like a right one")
    counter = [0]
    tables: list[Table] = []
    heads: list[str] = []
    for n, words in enumerate(pages[:MAX_PAGES], start=1):
        if not words:
            continue
        ts, head = _build_page(n, words, kind, counter)
        tables += ts
        heads.append(head)
    from ..pdf import _rows
    text = "\n".join(" ".join(w[3] for w in r) for p in pages[:40] for r in _rows(p))                  # one line per row of the page: terms and dates are read line by line
    return Document(format=fmt, pages=len(pages), tables=tables, text_head=text[:80000], language=language, warnings=warnings, reader={"name": "pdf_layout", "ocr": kind == "pdf-ocr"})


__all__ = ["extract"]
