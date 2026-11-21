"""Deterministic generator of the benchmark documents and their ground truth.

Every case is a synthetic financial report: a company's figures are generated with the accounting identities holding exactly, then *rendered* in the layout, language, number format and file format of
a market (PDF with a text layer, scanned PDF, HTML, workbook). The truth file lists every figure printed, with its period, unit and currency.

These are not real filings. They measure whether the extractor survives the layouts and conventions of each market; they cannot measure how it copes with a filing's own quirks. The report says so.

Splits: ``dev`` (the rules were written against these), ``validation`` (checked while developing, different wording and layouts), ``heldout`` (frozen: written separately, run for the report, never used to
tune a rule; ``heldout.lock`` holds the hash of the files so a change is detectable).
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from labels import LABELS, STATEMENT_LINES, STATEMENT_TITLES, UNIT_WORDS  # noqa: E402

UNIT_FACTOR = {"units": 1, "thousands": 1_000, "millions": 1_000_000}
CJK_FONT = {"ja": "japan", "ko": "korea", "zh": "china-s", "zh-hk": "china-t"}
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


# ------------------------------------------------------------------------------------------------------------------------ figures
def figures(seed: int, mag: int) -> dict:
    """One column of statements in printed units: integers, identities exact."""
    r = random.Random(seed)
    rev = r.randint(8, 40) * mag
    cost = int(rev * r.uniform(0.45, 0.7))
    gross = rev - cost
    opx = int(gross * r.uniform(0.35, 0.6))
    op = gross - opx
    pre = op - int(op * r.uniform(0.02, 0.1))
    tax = int(pre * r.uniform(0.18, 0.3))
    ni = pre - tax
    cash, ar, inv = r.randint(3, 9) * mag, r.randint(2, 6) * mag, r.randint(2, 7) * mag
    ca = cash + ar + inv
    ppe = r.randint(6, 20) * mag
    ta = ca + ppe + r.randint(1, 4) * mag
    ap, other_cl = r.randint(2, 6) * mag, r.randint(1, 4) * mag
    cl = ap + other_cl
    tl = cl + r.randint(3, 9) * mag
    eq = ta - tl
    cfo = ni + r.randint(1, 4) * mag
    cfi = -r.randint(2, 8) * mag
    cff = -r.randint(1, 5) * mag
    return {"revenue": rev, "cost_of_revenue": cost, "gross_profit": gross, "operating_income": op, "income_before_tax": pre, "income_tax": tax, "net_income": ni, "eps_basic": round(ni / (r.randint(40, 90) * mag / 10) , 2),
            "cash_and_equivalents": cash, "accounts_receivable": ar, "inventory": inv, "current_assets": ca, "ppe": ppe, "total_assets": ta, "accounts_payable": ap, "current_liabilities": cl, "total_liabilities": tl,
            "shareholders_equity": eq, "operating_cash_flow": cfo, "investing_cash_flow": cfi, "financing_cash_flow": cff}


# ------------------------------------------------------------------------------------------------------------------------ number formats
def fmt(v, market: str, lang: str, decimals: int = 0) -> str:
    neg = v < 0
    a = abs(v)
    if lang in ("fr", "es", "de"):
        s = f"{a:,.{decimals}f}"
        s = s.replace(",", " ") if lang == "fr" else s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"({s})" if neg else s
    s = f"{a:,.{decimals}f}"
    if neg:
        if lang == "ja":
            return "△" + s
        if lang == "zh":
            return "-" + s
        return f"({s})"
    return s


# ------------------------------------------------------------------------------------------------------------------------ PDF renderer
class Pdf:
    def __init__(self, lang: str, left_aligned: bool = False):
        import pymupdf
        self.doc = pymupdf.open()
        self.fitz = pymupdf
        self.font = CJK_FONT.get(lang, "helv")
        self.left_aligned = left_aligned
        self.page = None
        self.y = 0

    def new_page(self):
        self.page = self.doc.new_page()
        self.y = 60

    def text(self, x, s, size=10):
        self.page.insert_text((x, self.y), s, fontsize=size, fontname=self.font)

    def rtext(self, xr, s, size=10):
        w = self.fitz.get_text_length(s, fontname=self.font, fontsize=size)
        self.page.insert_text((xr - (0 if self.left_aligned else w) - (w if self.left_aligned and False else 0), self.y), s, fontsize=size, fontname=self.font) if not self.left_aligned else self.page.insert_text((xr - 60, self.y), s, fontsize=size, fontname=self.font)

    def line(self, gap=15):
        self.y += gap

    def bytes(self) -> bytes:
        try:
            self.doc.subset_fonts()                                                      # embedded CJK fonts are large; keep only the glyphs used
        except Exception:
            pass
        return self.doc.tobytes(garbage=3, deflate=True)


def unit_text(spec: dict, lang: str) -> str:
    u = UNIT_WORDS.get(lang if lang in UNIT_WORDS else "en", {}).get(spec["unit"], "")
    cl = spec.get("currency_label")
    if cl:
        return u.replace("(", "(" + cl + " ").strip() if u else f"({cl})"
    return u


def render_statements(spec: dict, rows_by_stmt: dict, headers: dict, pdf_trap: dict | None = None) -> bytes:
    lang, layout = spec["language"], spec["layout"]
    pdf = Pdf(lang, left_aligned=(layout == "left_aligned"))
    edges = [430, 520]
    note_edge = 340
    trap = spec.get("traps") or []
    if "summary" in trap:
        pdf.new_page()
        pdf.text(40, {"en": "Financial Highlights (Five-Year Summary)"}.get(lang, "Financial Highlights"))
        pdf.line(20)
        for k, label in enumerate(["Revenue", "Net income"]):
            pdf.text(40, label)
            for e, v in zip((300, 380, 460, 540), (pdf_trap["summary"][label][0], "111", "222", "333")):
                pdf.rtext(e, v)
            pdf.line()
    for stmt in ("IS", "BS", "CF"):
        pdf.new_page()
        titles = STATEMENT_TITLES[lang if lang in STATEMENT_TITLES else "en"][stmt]
        pdf.text(40, titles[{"dev": 0, "alt": 1, "held": 2}[spec["wording"]]], 12)
        pdf.line(18)
        unit_line = UNIT_WORDS.get(lang if lang in UNIT_WORDS else "en", {}).get(spec["unit"], "")
        unit_line = unit_text(spec, lang)
        if unit_line:
            pdf.text(40, unit_line)
            pdf.line(18)
        for h in headers[stmt]["lines"]:
            if isinstance(h, str):
                pdf.text(40, h)
                pdf.line(13)
            else:
                label_left, cols = h
                if label_left:
                    pdf.text(40, label_left)
                if layout in ("notes",) and stmt != "CF":
                    pdf.rtext(note_edge, "Note" if lang == "en" else "注記" if lang == "ja" else "주석" if lang == "ko" else "附注" if lang == "zh" else "Note")
                for e, c in zip(edges, cols):
                    pdf.rtext(e, c)
                pdf.line(15)
        for n, (code, label, vals) in enumerate(rows_by_stmt[stmt]):
            if layout == "wrapped" and len(label) > 18 and lang in ("en", "fr", "de", "es"):
                cut = label.rfind(" ", 0, len(label) // 2 + 4)
                pdf.text(40, label[:cut])
                pdf.line(12)
                label = label[cut + 1:]
            pdf.text(40, label)
            if layout == "notes" and stmt != "CF" and n % 3 == 0:
                pdf.rtext(note_edge, str(n + 4))
            for e, v in zip(edges, vals):
                if v is not None:
                    pdf.rtext(e, v)
            pdf.line(14)
        if layout == "parent_after" and stmt == "BS":
            pdf.line(10)
    if "parent" in trap:
        pdf.new_page()
        pdf.text(40, "Balance Sheet of the Company (Parent Company Only)" if lang == "en" else {"ja": "貸借対照表(個別)", "ko": "별도재무상태표", "zh": "母公司资产负债表", "zh-hk": "本公司財務狀況表"}.get(lang, "Company balance sheet"), 12)
        pdf.line(18)
        pdf.text(40, "")
        for e, c in zip(edges, headers["BS"]["cols"]):
            pdf.rtext(e, c)
        pdf.line(15)
        for code, label, vals in rows_by_stmt["BS"]:
            if code in ("total_assets", "total_liabilities", "shareholders_equity", "current_assets"):
                pdf.text(40, label)
                for e, v in zip(edges, pdf_trap["parent"][code]):
                    pdf.rtext(e, v)
                pdf.line(14)
    if "segment" in trap:
        pdf.new_page()
        pdf.text(40, "Note 5 - Segment information", 12)
        pdf.line(18)
        for e, c in zip(edges, headers["IS"]["cols"]):
            pdf.rtext(e, c)
        pdf.line(15)
        pdf.text(40, "Revenue")
        for e, v in zip(edges, pdf_trap["segment"]):
            pdf.rtext(e, v)
        pdf.line(14)
        pdf.text(40, "Net income")
        for e, v in zip(edges, pdf_trap["segment_ni"]):
            pdf.rtext(e, v)
    return pdf.bytes()


# ------------------------------------------------------------------------------------------------------------------------ case construction
def build_case(spec: dict) -> dict:
    """Figures, rendered rows, headers and truth for a case spec."""
    lang, market, unit = spec["language"], spec["market"], spec["unit"]
    wording = {"dev": 0, "alt": 1, "held": 2}[spec["wording"]]
    labels = LABELS[lang]
    seed = spec["seed"]
    mag = spec.get("mag", 100)
    cols = [figures(seed, mag), figures(seed + 1000, mag)]
    if spec.get("restated"):
        cols[1] = dict(cols[1])
    ends = spec["ends"]                                   # [current_end, prior_end] ISO dates (flows); the balance sheet uses spec["bs_ends"]
    bs_ends = spec.get("bs_ends", ends)
    period_type = spec["period_type"]
    rows_by_stmt, truth_vals = {}, {}
    for stmt in ("IS", "BS", "CF"):
        rows = []
        for code in STATEMENT_LINES[stmt]:
            if code not in labels:
                continue
            dec = 2 if code == "eps_basic" else 0
            vals = [fmt(c[code], market, lang, dec) for c in cols]
            rows.append((code, labels[code][wording], vals))
            for end, c in zip(bs_ends if stmt == "BS" else ends, cols):
                truth_vals.setdefault(end, {})[code] = c[code] * (1 if code == "eps_basic" else UNIT_FACTOR[unit])
        rows_by_stmt[stmt] = rows
    headers = {}
    for stmt in ("IS", "BS", "CF"):
        e = bs_ends if stmt == "BS" else ends
        headers[stmt] = {"lines": header_lines(spec, stmt, e), "cols": [d[:4] for d in e]}
    periods = []
    ftype = {"annual": "FY", "half": "H1", "quarter": "Q"}[period_type]
    for stmt_ends, flows in ((ends, True),):
        pass
    for end in sorted(truth_vals):
        flow = end in ends
        fp = ftype if flow else "FY"
        if flow and period_type == "quarter":
            fp = "Q" + str((int(end[5:7]) - 1) // 3 + 1)
        periods.append({"fiscal_period": fp, "period_end": end, "values": truth_vals[end], "currency": spec["currency"]})
    truth = {"periods": periods, "unit": unit, "currency": spec["currency"], "statements": ["IS", "BS", "CF"], "standard": spec["standard"], "lines_printed": sum(len(v) for v in truth_vals.values())}
    return {"rows": rows_by_stmt, "headers": headers, "truth": truth, "cols": cols}


def header_lines(spec: dict, stmt: str, ends: list[str]) -> list:
    lang, ptype = spec["language"], spec["period_type"]
    d = [(int(e[:4]), int(e[5:7]), int(e[8:10])) for e in ends]
    if lang == "ko":
        t = [f"제 {spec['term']}(당) 기", f"제 {spec['term'] - 1}(전) 기"]
        first = f"제 {spec['term']} 기 : {ends[0].replace('-', '.')} 현재" if stmt == "BS" else f"제 {spec['term']} 기 {d[0][0]}.01.01 부터 {ends[0].replace('-', '.')} 까지"
        return [first, ("과 목", t)] if stmt != "BS" else [first, f"제 {spec['term'] - 1} 기 : {ends[1].replace('-', '.')} 현재", ("과 목", t)]
    if lang == "ja":
        dates = [f"{y}年{m}月{dd}日" for y, m, dd in d]
        return [("", dates)]
    if lang == "zh":
        dates = [f"{y}年{m}月{dd}日" for y, m, dd in d]
        pre = {"half": f"截至{d[0][0]}年{d[0][1]}月{d[0][2]}日止六个月", "quarter": f"截至{d[0][0]}年{d[0][1]}月{d[0][2]}日止三个月"}.get(ptype)
        return ([pre] if pre and stmt != "BS" else []) + [("", dates)]
    if lang == "zh-hk":
        dates = [f"{y}年{m}月{dd}日" for y, m, dd in d]
        pre = {"half": f"截至{d[0][0]}年{d[0][1]}月{d[0][2]}日止六個月", "quarter": f"截至{d[0][0]}年{d[0][1]}月{d[0][2]}日止三個月"}.get(ptype)
        return ([pre] if pre and stmt != "BS" else []) + [("", dates)]
    month = lambda m: {"fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"], "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"],
                       "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]}.get(lang, MONTHS_EN)[m - 1]
    if spec["layout"] in ("dated_header", "wrapped") or stmt == "BS" or lang != "en":
        dates = [f"{dd} {month(m)} {y}" for y, m, dd in d]
        pre = {"half": "Six months ended" if lang == "en" else "", "quarter": "Three months ended" if lang == "en" else ""}.get(ptype) if stmt != "BS" else None
        return ([pre] if pre else []) + [("", dates)]
    pre = {"annual": "Year ended " + f"{d[0][2]} {MONTHS_EN[d[0][1] - 1]}", "half": f"Six months ended {d[0][2]} {MONTHS_EN[d[0][1] - 1]}", "quarter": f"Three months ended {d[0][2]} {MONTHS_EN[d[0][1] - 1]}"}[ptype]
    return [pre, ("", [str(y) for y, _, _ in d])]


# ------------------------------------------------------------------------------------------------------------------------ other formats
def render_html(spec: dict, built: dict) -> tuple[bytes, str]:
    lang = spec["language"]
    enc = "cp932" if lang == "ja" else "utf-8"
    parts = [f"<html><head><meta charset='{'shift_jis' if lang == 'ja' else 'utf-8'}'></head><body>"]
    for stmt in ("IS", "BS", "CF"):
        titles = STATEMENT_TITLES[lang if lang in STATEMENT_TITLES else "en"][stmt]
        parts.append(f"<h2>{titles[{'dev': 0, 'alt': 1, 'held': 2}[spec['wording']]]}</h2>")
        u = unit_text(spec, lang)
        if u:
            parts.append(f"<p>{u}</p>")
        parts.append("<table>")
        hdr = built["headers"][stmt]["lines"][-1]
        cols = hdr[1] if isinstance(hdr, tuple) else built["headers"][stmt]["cols"]
        parts.append("<tr><th></th>" + "".join((f"<th></th><th>{c}</th>" if spec["market"] == "US" else f"<th>{c}</th>") for c in cols) + "</tr>")
        for code, label, vals in built["rows"][stmt]:
            if spec["market"] == "US":                                                   # SEC-style: a "$" cell beside the figure on the first and total rows, an empty cell in its place on the others
                cells = "".join((f"<td>$</td><td>{v}</td>" if code in ("revenue", "net_income", "total_assets", "operating_cash_flow", "cash_and_equivalents") else f"<td></td><td>{v}</td>") for v in vals)
            else:
                cells = "".join(f"<td>{v}</td>" for v in vals)
            parts.append(f"<tr><td>{label}</td>{cells}</tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    return "".join(parts).encode(enc, "ignore"), "html"


def render_xlsx(spec: dict, built: dict) -> bytes:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Financials"
    lang = spec["language"]
    for stmt in ("IS", "BS", "CF"):
        titles = STATEMENT_TITLES[lang if lang in STATEMENT_TITLES else "en"][stmt]
        ws.append([titles[{"dev": 0, "alt": 1, "held": 2}[spec["wording"]]]])
        u = unit_text(spec, lang)
        if u:
            ws.append([u])
        hdr = built["headers"][stmt]["lines"][-1]
        ws.append([""] + (hdr[1] if isinstance(hdr, tuple) else built["headers"][stmt]["cols"]))
        for code, label, vals in built["rows"][stmt]:
            nums = []
            for v in vals:
                n = v.replace("\u00a0", "").replace(" ", "")
                if lang in ("de", "es"):                                                  # 1.234,5 -> 1234.5
                    n = n.replace(".", "").replace(",", ".")
                elif lang == "fr":
                    n = n.replace(",", ".")
                else:
                    n = n.replace(",", "")
                if n.startswith("("):
                    n = "-" + n.strip("()")
                if n.startswith("△"):
                    n = "-" + n[1:]
                try:
                    nums.append(float(n) if "." in n else int(n))
                except ValueError:
                    nums.append(v)
            ws.append([label] + nums)
        ws.append([])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def scan_pdf(text_pdf: bytes) -> bytes:
    """An image-only copy: every page rasterised, no text layer."""
    import pymupdf
    src = pymupdf.open(stream=text_pdf, filetype="pdf")
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=100, colorspace=pymupdf.csGRAY)
        p = out.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=pix.tobytes("jpeg", jpg_quality=55))
    return out.tobytes(garbage=3, deflate=True)


# ------------------------------------------------------------------------------------------------------------------------ the case list
def specs() -> list[dict]:
    S = []

    def add(split, market, fmt_, lang, standard, unit, currency, ptype, layout="classic", wording="dev", ends=None, bs_ends=None, **kw):
        n = sum(1 for s in S if s["split"] == split and s["market"] == market and s["format"] == fmt_) + 1
        if market == "US":
            kw.setdefault("currency_label", "$")
        ends = ends or ({"annual": ["2023-12-31", "2022-12-31"], "half": ["2024-06-30", "2023-06-30"], "quarter": ["2024-03-31", "2023-03-31"]}[ptype])
        if bs_ends is None and ptype != "annual":
            bs_ends = [ends[0], f"{int(ends[0][:4]) - 1}-12-31"]
        S.append({"id": f"{split}-{market.lower()}-{fmt_}-{n}", "split": split, "market": market, "format": fmt_, "language": lang, "standard": standard, "unit": unit, "currency": currency, "period_type": ptype,
                  "layout": layout, "wording": wording, "ends": ends, "bs_ends": bs_ends or ends, "seed": len(S) * 37 + 11, "term": 55, "mag": kw.pop("mag", 100), **kw})

    # ---- dev: the wording and layouts the rules were written against
    add("dev", "US", "pdf", "en", "us-gaap", "millions", "USD", "annual", currency_label="$")
    add("dev", "US", "pdf", "en", "us-gaap", "millions", "USD", "quarter", currency_label="$")
    add("dev", "US", "html", "en", "us-gaap", "millions", "USD", "annual", currency_label="$")
    add("dev", "US", "xlsx", "en", "us-gaap", "millions", "USD", "annual", currency_label="$")
    add("dev", "SG", "pdf", "en", "sfrs-i", "thousands", "SGD", "annual", layout="notes", currency_label="S$'000")
    add("dev", "SG", "pdf", "en", "sfrs-i", "thousands", "SGD", "half", currency_label="S$'000")
    add("dev", "JP", "pdf", "ja", "j-gaap", "millions", "JPY", "annual", mag=1000)
    add("dev", "JP", "html", "ja", "j-gaap", "millions", "JPY", "annual", mag=1000)
    add("dev", "KR", "pdf", "ko", "k-ifrs", "millions", "KRW", "annual", mag=1000)
    add("dev", "CN", "pdf", "zh", "cas", "thousands", "CNY", "annual", mag=1000)
    add("dev", "HK", "pdf", "en", "ifrs", "thousands", "HKD", "half", currency_label="HK$'000")
    add("dev", "EU", "pdf", "fr", "ifrs", "thousands", "EUR", "annual")
    add("dev", "EU", "pdf", "de", "ifrs", "millions", "EUR", "annual")
    add("dev", "EU", "pdf", "es", "ifrs", "thousands", "EUR", "annual")
    # ---- validation: other wording, other layouts, checked while developing
    add("validation", "US", "pdf", "en", "us-gaap", "millions", "USD", "annual", layout="dated_header", wording="alt")
    add("validation", "SG", "pdf", "en", "sfrs-i", "thousands", "SGD", "annual", layout="wrapped", wording="alt", currency_label="S$'000")
    add("validation", "JP", "pdf", "ja", "j-gaap", "millions", "JPY", "annual", wording="alt", mag=1000)
    add("validation", "KR", "pdf", "ko", "k-ifrs", "millions", "KRW", "annual", wording="alt", mag=1000)
    add("validation", "CN", "pdf", "zh", "cas", "thousands", "CNY", "half", wording="alt", mag=1000)
    add("validation", "HK", "pdf", "zh-hk", "ifrs", "thousands", "HKD", "annual", wording="alt", mag=1000)
    add("validation", "EU", "pdf", "fr", "ifrs", "millions", "EUR", "annual", wording="alt")
    add("validation", "EU", "xlsx", "de", "ifrs", "millions", "EUR", "annual", wording="alt")
    # ---- held-out: frozen. Wording and layouts written separately; traps; scans; restatements
    H = "heldout"
    add(H, "US", "pdf", "en", "us-gaap", "millions", "USD", "annual", layout="left_aligned", wording="held")
    add(H, "US", "pdf", "en", "us-gaap", "millions", "USD", "quarter", layout="dated_header", wording="held", traps=["segment"])
    add(H, "US", "html", "en", "us-gaap", "millions", "USD", "annual", wording="held")
    add(H, "US", "pdf", "en", "us-gaap", "millions", "USD", "annual", layout="classic", wording="held", scanned=True)
    add(H, "US", "pdf", "en", "us-gaap", "millions", "USD", "annual", layout="notes", wording="held", restated=True, traps=["summary"])
    add(H, "SG", "pdf", "en", "sfrs-i", "thousands", "SGD", "annual", layout="parent_after", wording="held", currency_label="S$'000", traps=["parent"])
    add(H, "SG", "pdf", "en", "sfrs-i", "thousands", "SGD", "half", layout="notes", wording="held", currency_label="S$'000")
    add(H, "SG", "pdf", "en", "sfrs-i", "thousands", "SGD", "annual", layout="wrapped", wording="held", currency_label="S$'000", scanned=True)
    add(H, "JP", "pdf", "ja", "j-gaap", "millions", "JPY", "annual", layout="classic", wording="held", mag=1000, traps=["parent"])
    add(H, "JP", "html", "ja", "j-gaap", "millions", "JPY", "annual", wording="held", mag=1000)
    add(H, "JP", "pdf", "ja", "j-gaap", "millions", "JPY", "half", layout="notes", wording="held", mag=1000)
    add(H, "KR", "pdf", "ko", "k-ifrs", "millions", "KRW", "annual", layout="notes", wording="held", mag=1000, traps=["parent"])
    add(H, "KR", "pdf", "ko", "k-ifrs", "millions", "KRW", "half", wording="held", mag=1000)
    add(H, "KR", "pdf", "ko", "k-ifrs", "millions", "KRW", "annual", wording="held", mag=1000, scanned=True)
    add(H, "CN", "pdf", "zh", "cas", "thousands", "CNY", "annual", layout="classic", wording="held", mag=1000, traps=["parent"])
    add(H, "CN", "pdf", "zh", "cas", "thousands", "CNY", "half", layout="left_aligned", wording="held", mag=1000)
    add(H, "CN", "xlsx", "zh", "cas", "thousands", "CNY", "annual", wording="held", mag=1000)
    add(H, "HK", "pdf", "zh-hk", "ifrs", "thousands", "HKD", "half", layout="notes", wording="held", mag=1000)
    add(H, "HK", "pdf", "en", "ifrs", "thousands", "HKD", "annual", layout="parent_after", wording="held", currency_label="HK$'000", traps=["parent"], mag=1000)
    add(H, "HK", "pdf", "en", "ifrs", "thousands", "HKD", "annual", layout="classic", wording="held", currency_label="HK$'000", scanned=True, mag=1000)
    add(H, "EU", "pdf", "fr", "ifrs", "millions", "EUR", "annual", layout="notes", wording="held", restated=True)
    add(H, "EU", "pdf", "de", "ifrs", "millions", "EUR", "half", wording="held")
    add(H, "EU", "pdf", "es", "ifrs", "thousands", "EUR", "annual", layout="dated_header", wording="held", traps=["segment"])
    add(H, "EU", "html", "fr", "ifrs", "thousands", "EUR", "annual", wording="held")
    return S


def trap_figures(spec: dict, built: dict) -> dict:
    cols = built["cols"]
    f = lambda v: fmt(v, spec["market"], spec["language"])  # noqa: E731
    return {"parent": {c: [f(int(cols[0][c] * 0.6)), f(int(cols[1][c] * 0.6))] for c in ("total_assets", "total_liabilities", "shareholders_equity", "current_assets")},
            "segment": [f(int(cols[0]["revenue"] * 0.4)), f(int(cols[1]["revenue"] * 0.4))], "segment_ni": [f(int(cols[0]["net_income"] * 0.3)), f(int(cols[1]["net_income"] * 0.3))],
            "summary": {"Revenue": [f"{int(cols[0]['revenue'] * 1.07):,}"], "Net income": [f"{int(cols[0]['net_income'] * 1.1):,}"]}}


def write_all(root: Path = HERE / "data") -> list[dict]:
    import shutil
    if root.exists():
        shutil.rmtree(root)
    manifest = []
    for spec in specs():
        built = build_case(spec)
        d = root / spec["split"]
        d.mkdir(parents=True, exist_ok=True)
        name = spec["id"]
        if spec["format"] == "pdf":
            data = render_statements(spec, built["rows"], built["headers"], trap_figures(spec, built))
            if spec.get("scanned"):
                (d / f"{name}.src.pdf").write_bytes(data)
                data = scan_pdf(data)
            (d / f"{name}.pdf").write_bytes(data)
            ext = "pdf"
        elif spec["format"] == "html":
            data, _ = render_html(spec, built)
            (d / f"{name}.html").write_bytes(data)
            ext = "html"
        else:
            (d / f"{name}.xlsx").write_bytes(render_xlsx(spec, built))
            ext = "xlsx"
        (d / f"{name}.truth.json").write_text(json.dumps({**built["truth"], "spec": {k: v for k, v in spec.items() if k != "seed"}}, ensure_ascii=False, indent=1))
        manifest.append({"id": name, "file": f"{spec['split']}/{name}.{ext}", "truth": f"{spec['split']}/{name}.truth.json", **{k: spec.get(k) for k in ("split", "market", "format", "language", "standard", "period_type", "layout", "wording", "unit", "currency")},
                         "scanned": bool(spec.get("scanned")), "restated": bool(spec.get("restated")), "traps": spec.get("traps", [])})
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    lock = hashlib.sha256()
    for f in sorted((root / "heldout").glob("*")):
        lock.update(f.name.encode() + f.read_bytes())
    (HERE / "heldout.lock").write_text(lock.hexdigest() + "\n")
    return manifest


if __name__ == "__main__":
    m = write_all()
    from collections import Counter
    print(len(m), "cases", dict(Counter(c["split"] for c in m)))
