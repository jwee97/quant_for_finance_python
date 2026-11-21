"""The readers beyond the first ones: HTML tables in the page's own encoding, scanned PDFs through an OCR engine, Docling for tables the word positions cannot recover, and AI-assisted extraction that is off
by default and accepts only figures it can find in the document. The OCR, Docling and AI services are stand-ins: none is installed or called here."""

from __future__ import annotations

import json

import pytest

from src.fundamentals.errors import FundamentalsError
from src.fundamentals.importer import ai_extract, docling_adapter, ocr, pdf
from src.fundamentals.importer import html as html_reader
from src.fundamentals.importer.pipeline import ImportService, sniff
from src.fundamentals.store import FundamentalsStore


def page(title="", unit="", head=("2024年度", "2023年度"), charset="shift_jis") -> bytes:
    rows = lambda *r: "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in r)  # noqa: E731
    body = (f'<html><head><meta http-equiv="Content-Type" content="text/html; charset={charset}"></head><body><h1>{title}</h1><p>{unit}</p>'
            f'<table><tr><th>科目</th><th>{head[0]}</th><th>{head[1]}</th></tr>{rows(("売上高", "1,200,000", "1,000,000"), ("営業利益", "200,000", "150,000"), ("当期純利益", "150,000", "△120,000"))}</table>'
            f'<table><tr><th>科目</th><th>{head[0]}</th><th>{head[1]}</th></tr>{rows(("資産合計", "5,000,000", "4,500,000"), ("負債合計", "3,000,000", "2,700,000"), ("純資産合計", "2,000,000", "1,800,000"))}</table>'
            f'<table><tr><td>noise</td><td>x</td></tr></table></body></html>')
    return body.encode(charset)


def test_an_html_report_is_decoded_by_its_own_charset_and_its_tables_are_put_together_by_period():
    data = page("連結財務諸表", "（単位：百万円）")
    assert sniff("report.htm", data) == "html", "an HTML page with no XBRL is read as a page of tables"
    d = html_reader.extract(data)
    assert d["method"] == "html-table" and d["units"]["money"] == "millions" and d["units_stated"] and d["review_required"]
    by = {p["period_end"][:4]: p for p in d["periods"]}
    assert by["2024"]["values"] == {"revenue": 1_200_000, "operating_income": 200_000, "net_income": 150_000, "total_assets": 5_000_000, "total_liabilities": 3_000_000, "shareholders_equity": 2_000_000}, "income statement and balance sheet joined"
    assert by["2023"]["values"]["net_income"] == -120_000, "the triangle is a negative figure"
    assert by["2024"]["sources"]["revenue"]["method"] == "html-table" and "check each figure" in d["notes"][0]


def test_html_in_other_encodings_and_without_a_unit_or_tables_says_so():
    assert html_reader.extract(page("x", head=("2024年度", "2023年度"), charset="utf-8"))["periods"]
    assert html_reader.decode("日本語の売上高".encode("shift_jis")) == "日本語の売上高" and html_reader.decode("매출액 영업이익 당기순이익 자산총계 부채총계 자본총계".encode("euc-kr")) == "매출액 영업이익 당기순이익 자산총계 부채총계 자본총계" and html_reader.decode("营业收入 营业利润 净利润 资产总计 负债合计 所有者权益合计".encode("gb18030")) == "营业收入 营业利润 净利润 资产总计 负债合计 所有者权益合计"
    assert any("does not state its unit" in n for n in html_reader.extract(page("x", ""))["notes"])
    with pytest.raises(FundamentalsError, match="no table"):
        html_reader.extract(b"<html><body><table><tr><td>a</td><td>b</td></tr></table><p>no tables of figures</p></body></html>")


# ------------------------------------------------------------------------------------------------------------------------ OCR
class FakeOcr(ocr.Engine):
    name = "fake"

    def words(self, image_png, language):
        self.language = language
        rows = [("連結損益計算書 (単位:千円)", None, None), ("事業年度", "2024", "2023"), ("売上高", "1,200,000", "1,000,000"), ("営業利益", "200,000", "150,000"), ("当期純利益", "150,000", "120,000"), ("資産合計", "5,000,000", "4,500,000"),
                ("負債合計", "3,000,000", "2,700,000")]
        out, y = [], 100.0
        for label, a, b in rows:
            out.append((40.0, y, 200.0, label, 0.93))
            if a:
                out += [(330.0, y, 380.0, a, 0.9), (430.0, y, 480.0, b, 0.9)]
            y += 20
        return out


def scanned() -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()
    return doc.tobytes()


def test_a_scanned_pdf_is_refused_without_an_engine_and_read_through_one_when_given():
    pytest.importorskip("pymupdf")
    if not ocr.available():
        with pytest.raises(FundamentalsError, match="no OCR engine is installed.*paddleocr|PaddleOCR"):
            pdf.extract(scanned())
    eng = FakeOcr()
    d = pdf.extract(scanned(), eng, "ja")
    assert eng.language == "ja" and d["method"] == "pdf-ocr" and d["review_required"] and d["units"]["money"] == "thousands"
    by = {p["period_end"][:4]: p for p in d["periods"]}
    assert by["2024"]["values"]["revenue"] == 1_200_000 and by["2024"]["values"]["total_assets"] == 5_000_000 and by["2023"]["values"]["net_income"] == 120_000
    assert by["2024"]["sources"]["revenue"]["method"] == "pdf-ocr" and "scanned" in d["notes"][0] and "misread digit" in d["notes"][0]


def test_the_ocr_engines_are_chosen_by_name_and_a_missing_one_is_named(monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: [])
    with pytest.raises(FundamentalsError, match="paddleocr"):
        ocr.engine()
    monkeypatch.setattr(ocr, "available", lambda: ["surya", "paddleocr"])
    with pytest.raises(FundamentalsError, match="must be one of"):
        ocr.engine("tesseract")
    assert ocr.engine().name == "surya" and ocr.engine("paddleocr").name == "paddleocr"
    assert set(ocr.LANG) >= {"en", "ja", "ko", "zh", "zh-tw"}


# ------------------------------------------------------------------------------------------------------------------------ the chain for a PDF
def thin_pdf() -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    p = doc.new_page()
    for i, t in enumerate(["Year ended December 31,   2023   2022", "Total revenues    1,200    1,000", "Net income    150    120"]):
        p.insert_text((40, 60 + 14 * i), t.split("   ")[0], fontsize=9)
        for j, cell in enumerate(t.split("   ")[1:]):
            p.insert_text((330 + 100 * j, 60 + 14 * i), cell.strip(), fontsize=9)
    return doc.tobytes()


@pytest.fixture()
def svc(tmp_path):
    return ImportService(FundamentalsStore(tmp_path / "f.db"))


def test_a_pdf_with_too_few_lines_goes_to_docling_when_it_is_installed(svc, monkeypatch):
    monkeypatch.setattr(docling_adapter, "available", lambda: True)
    monkeypatch.setattr(docling_adapter, "tables", lambda data: ["\tFY2023\tFY2022\nRevenue\t1200\t1000\nNet income\t150\t120\nTotal assets\t5000\t4500\nOperating income\t200\t150\nTotal liabilities\t3000\t2700\n(in millions)"])
    d = svc._pdf(thin_pdf(), "r.pdf", {})
    assert d["method"] == "pdf-docling" and d["review_required"] and "Docling" in d["notes"][0] and d["periods"][0]["values"]["revenue"] in (1200, 1000)
    monkeypatch.setattr(docling_adapter, "available", lambda: False)
    with pytest.raises(FundamentalsError, match="only 2 statement line"):
        svc._pdf(thin_pdf(), "r.pdf", {})


# ------------------------------------------------------------------------------------------------------------------------ AI-assisted
TEXT = "CONSOLIDATED STATEMENTS (in millions of yen)\nYear ended March 31, 2024\nNet sales 48,036,704 37,154,298\nOperating income 5,352,934 2,725,025\nProfit for the year 4,944,933 2,451,318"


def answer(**over):
    base = {"unit": "millions", "currency": "JPY", "periods": [{"fiscal_period": "FY", "period_end": "2024-03-31", "values": {
        "revenue": {"value": 48036704, "evidence": "Net sales 48,036,704 37,154,298"}, "operating_income": {"value": 5352934, "evidence": "Operating income 5,352,934 2,725,025"},
        "net_income": {"value": 9999999, "evidence": "Profit for the year 4,944,933 2,451,318"},                          # not the number in its evidence
        "total_assets": {"value": 90000000, "evidence": "Total assets 90,000,000"},                                        # evidence that is not in the document
        "made_up_line": {"value": 1, "evidence": "Net sales 48,036,704"}}}]}
    base.update(over)
    return base


def test_ai_extraction_is_off_by_default_and_keeps_only_figures_found_in_the_document():
    with pytest.raises(FundamentalsError, match="is off"):
        ai_extract.extract(TEXT, client=lambda p: json.dumps(answer()), env={})
    env = {"FUNDAMENTALS_AI_EXTRACTION": "1"}
    seen = []
    d = ai_extract.extract(TEXT, client=lambda p: (seen.append(p), "```json\n" + json.dumps(answer()) + "\n```")[1], language="ja", env=env)
    assert d["method"] == "ai-assisted" and d["review_required"] and d["units"]["money"] == "millions" and d["currency"] == "JPY"
    assert d["periods"][0]["values"] == {"revenue": 48036704.0, "operating_income": 5352934.0}, "an invented total, a wrong number and an unknown line are dropped"
    assert {r["line"]: r["reason"] for r in d["rejected"]} == {"net_income": "the number is not in the quoted evidence", "total_assets": "the quoted evidence is not in the document", "made_up_line": "not a line this application knows"}
    assert d["periods"][0]["sources"]["revenue"]["text"].startswith("Net sales") and "DOCUMENT:" in seen[0] and "Do not infer" in seen[0]
    with pytest.raises(FundamentalsError, match="needs a client"):
        ai_extract.extract(TEXT, env={**env, "ANTHROPIC_API_KEY": ""})
    with pytest.raises(FundamentalsError, match="did not answer in the structure"):
        ai_extract.extract(TEXT, client=lambda p: "I cannot help", env=env)
    with pytest.raises(FundamentalsError, match="no figure the model gave was found"):
        ai_extract.extract(TEXT, client=lambda p: json.dumps({"unit": "units", "periods": [{"period_end": "2024-03-31", "values": {"revenue": {"value": 5, "evidence": "nowhere"}}}]}), env=env)


def test_the_pdf_chain_reaches_the_ai_model_only_when_it_is_switched_on(svc, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((40, 60), "Net sales 48,036,704 37,154,298", fontsize=9)
    p.insert_text((40, 80), "Operating income 5,352,934 2,725,025", fontsize=9)
    data = doc.tobytes()
    with pytest.raises(FundamentalsError):
        svc._pdf(data, "r.pdf", {})
    monkeypatch.setenv("FUNDAMENTALS_AI_EXTRACTION", "1")
    good = {"unit": "millions", "periods": [{"fiscal_period": "FY", "period_end": "2024-03-31", "values": {"revenue": {"value": 48036704, "evidence": "Net sales 48,036,704 37,154,298"}}}]}
    d = svc._pdf(data, "r.pdf", {"ai_client": lambda prompt: json.dumps(good)})
    assert d["method"] == "ai-assisted" and d["periods"][0]["values"] == {"revenue": 48036704.0}
