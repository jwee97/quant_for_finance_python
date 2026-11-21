"""The readers beyond the first ones: HTML tables in the page's own encoding, scanned PDFs through an OCR engine, Docling for tables the word positions cannot recover, and AI-assisted extraction that is off
by default and accepts only figures it can find in the document. The OCR, Docling and AI services are stand-ins: none is installed or called here."""

from __future__ import annotations

import json

import pytest

from src.fundamentals.errors import FundamentalsError
from src.fundamentals.importer import ai_extract, docling_adapter, ocr, pdf
from src.fundamentals.importer import engine as engine_module
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


# ------------------------------------------------------------------------------------------------------------------------ Arelle
def _arelle_facts(read, drop=None, change=None, errors=0):
    facts = []
    for f in read["facts"]:
        c = read["contexts"][f["context"]]
        v = f["value"] * (2 if change and f["field"] == change else 1)
        if drop and f["field"] == drop:
            continue
        facts.append({"name": f["field"], "value": v, "text": None, "unit": "USD", "start": c["start"].isoformat() if c["start"] else None, "end": c["end"].isoformat(), "dims": "x" if c["dimensional"] else ""})
    return {"facts": facts, "messages": [{"level": "error", "message": "x"}] * errors + [{"level": "warning", "message": "w"}]}


def test_the_arelle_cross_check_counts_agreement_names_differences_and_says_when_arelle_read_no_values():
    from tests.test_fund_import import ixbrl
    from src.fundamentals.importer import arelle_check, xbrl
    read = xbrl.read_facts(ixbrl())
    ok = arelle_check.compare(read, _arelle_facts(read))
    assert ok["agrees"] and ok["agree"] == ok["compared"] >= 7 and ok["different"] == [] and ok["validation_warnings"] == 1 and ok["facts_arelle"] == len(read["facts"])
    bad = arelle_check.compare(read, _arelle_facts(read, change="us-gaap:Revenues", drop="us-gaap:Assets"))
    assert not bad["agrees"] and [d["fact"] for d in bad["different"]] == ["us-gaap:Revenues"] and bad["different"][0]["arelle"] == 2 * bad["different"][0]["this_reader"]
    assert [o["fact"] for o in bad["only_this_reader"]] == ["us-gaap:Assets"]
    flawed = arelle_check.compare(read, _arelle_facts(read, errors=2))
    assert flawed["agrees"] and not flawed["valid"] and flawed["validation_errors"] == 2, "a validation error is about the filing, not a disagreement between the readers"
    names_only = {"facts": [{"name": "us-gaap:Assets", "value": None, "text": "", "unit": None, "start": None, "end": None, "dims": ""}], "messages": []}
    assert arelle_check.compare(read, names_only)["no_values"]
    assert arelle_check.wanted({"arelle": True}, {}) and arelle_check.wanted({}, {"FUNDAMENTALS_ARELLE": "1"}) and not arelle_check.wanted({}, {})


def test_a_failed_arelle_run_is_a_note_on_the_draft_not_a_failed_import(tmp_path, monkeypatch):
    from tests.test_fund_import import ixbrl
    from src.fundamentals.importer import arelle_check
    svc = ImportService(FundamentalsStore(tmp_path / "f.db"))
    monkeypatch.setattr(arelle_check, "available", lambda: False)
    d = svc._extract_bytes("xbrl", "x.htm", ixbrl(), {"arelle": True})
    assert d["periods"] and any("Arelle cross-check was not possible: Arelle is not installed" in n for n in d["notes"])
    monkeypatch.setattr(arelle_check, "crosscheck", lambda data, name, cache=None, online=True, package=None: {"agree": 8, "compared": 8, "facts_arelle": 19, "different": [], "validation_errors": 0, "agrees": True})
    assert any("8 of 8 company-wide numeric facts agree" in n for n in svc._extract_bytes("xbrl", "x.htm", ixbrl(), {"arelle": True})["notes"])


# ---- PDF reports of Hong Kong / Singapore / mainland China style: day-month-year and CJK dates, interim columns, Notes column, currency ----

def _pdf(pages, font=None):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for lines in pages:
        page, y = doc.new_page(), 60
        for line in lines:
            x = 40
            for token, dx in line:
                page.insert_text((x, y), token, fontsize=10, **({"fontname": font} if font else {}))
                x += dx
            y += 16
    return doc.tobytes()


def test_hong_kong_interim_pdf_reads_h1_notes_column_and_currency():
    data = _pdf([[
        [("INTERIM REPORT (Expressed in HK$'000)", 0)],
        [("Six months ended 30 June 2024 2023", 0)],
        [("Revenue", 230), ("4", 40), ("1,234,000", 80), ("1,100,000", 80)],
        [("Net income", 230), ("4", 40), ("(50,000)", 80), ("20,000", 80)],
        [("30 June 2024  31 December 2023", 0)],
        [("Total assets", 270), ("9,000,000", 90), ("8,500,000", 90)],
    ]])
    out = pdf.extract(data)
    by = {(p["fiscal_period"], p["period_end"]): p for p in out["periods"]}
    assert out["currency"] == "HKD" and out["units"]["money"] == "thousands" and out["review_required"]
    h1 = by[("H1", "2024-06-30")]
    assert h1["values"] == {"revenue": 1234000.0, "net_income": -50000.0, "total_assets": 9000000.0}      # the balance sheet at 30 June joins the half-year column
    assert by[("H1", "2023-06-30")]["values"]["revenue"] == 1100000.0
    assert by[("FY", "2023-12-31")]["values"] == {"total_assets": 8500000.0}                                # the December comparative stays a year-end balance
    assert h1["currency"] == "HKD"


def test_chinese_pdf_reads_cjk_dates_labels_and_unit():
    data = _pdf([[
        [("（人民币千元）", 0)],
        [("截至2024年6月30日止六个月  2024年6月30日 2023年6月30日", 0)],
        [("营业收入", 230), ("5,000", 80), ("4,000", 80)],
        [("净利润", 230), ("800", 80), ("(300)", 80)],
        [("资产总计", 230), ("90,000", 80), ("80,000", 80)],
    ]], font="china-s")
    out = pdf.extract(data)
    assert out["currency"] == "CNY" and out["units"]["money"] == "thousands"
    by = {p["period_end"]: p for p in out["periods"]}
    assert by["2024-06-30"]["fiscal_period"] == "H1"
    assert by["2024-06-30"]["values"] == {"revenue": 5000.0, "net_income": 800.0, "total_assets": 90000.0}
    assert by["2023-06-30"]["values"]["net_income"] == -300.0


def test_currency_detection_does_not_confuse_us_dollar_with_singapore_dollar():
    from src.fundamentals.importer.tabular import detect_currency
    assert detect_currency("US$'000") == "USD" and detect_currency("S$ million") == "SGD" and detect_currency("RMB'000") == "CNY"


def test_korean_pdf_reads_terms_dotted_dates_outline_numbers_and_loss_notes():
    data = _pdf([[
        [("(단위 : 백만원)", 0)],
        [("제 55 기 : 2023.12.31 현재", 0)],
        [("제 54 기 : 2022.12.31 현재", 0)],
        [("과 목", 150), ("주석", 60), ("제 55(당) 기", 130), ("제 54(전) 기", 130)],
        [("자산총계", 150), ("", 60), ("448,424,507", 130), ("448,424,507", 130)],
    ], [
        [("제 55 기 2023.01.01 부터 2023.12.31 까지", 0)],
        [("과 목", 150), ("주석", 60), ("제 55(당) 기", 130), ("제 54(전) 기", 130)],
        [("Ⅰ. 매출액", 150), ("", 60), ("258,935,494", 130), ("302,231,360", 130)],
        [("당기순이익(손실)", 150), ("", 60), ("14,473,401", 130), ("55,654,077", 130)],
    ]], font="korea")
    out = pdf.extract(data)
    by = {p["period_end"]: p for p in out["periods"]}
    assert out["currency"] == "KRW" and out["units"]["money"] == "millions"
    assert by["2023-12-31"]["values"] == {"total_assets": 448424507.0, "revenue": 258935494.0, "net_income": 14473401.0}
    assert by["2022-12-31"]["values"]["revenue"] == 302231360.0


def test_pdf_without_readable_column_headings_falls_back_to_latest_date_and_says_so():
    data = _pdf([[
        [("Statement as of 31 December 2023", 0)],
        [("Revenue", 200), ("500", 80), ("400", 80)],
        [("Net income", 200), ("50", 80), ("40", 80)],
    ]])
    out = pdf.extract(data)
    by = {p["period_end"]: p for p in out["periods"]}
    assert by["2023-12-31"]["values"] == {"revenue": 500.0, "net_income": 50.0} and by["2022-12-31"]["values"]["revenue"] == 400.0
    assert all(p["guessed"] for p in out["periods"]) and any("No column headings" in n for n in out["notes"])


def test_rapidocr_adapter_turns_recognised_lines_into_words_and_is_listed_after_it_is_installed(monkeypatch):
    import sys
    import types

    assert ocr.split_line(100, 50, 300, "Total assets 1,200 1,000")[0][3] == "Total" and len(ocr.split_line(100, 50, 300, "Total assets 1,200 1,000")) == 4
    words = ocr.split_line(100, 50, 300, "Revenue 1,200")
    assert words[0][0] == 100 and words[-1][2] == pytest.approx(300) and words[0][2] < words[1][0]
    assert "rapidocr" in ocr.ENGINES and ocr.ENGINES[0] == "paddleocr"
    monkeypatch.delitem(sys.modules, "rapidocr_onnxruntime", raising=False)
    if "rapidocr" in ocr.available():
        pytest.skip("RapidOCR is really installed")
    with pytest.raises(FundamentalsError, match="rapidocr-onnxruntime"):
        ocr.RapidEngine().words(b"x", "en")
    import importlib.machinery
    fake = types.ModuleType("rapidocr_onnxruntime")
    fake.__spec__ = importlib.machinery.ModuleSpec("rapidocr_onnxruntime", None)

    class FakeRapid:
        def __call__(self, image):
            return [([[10, 10], [200, 10], [200, 30], [10, 30]], "Total assets 5,000", 0.97)], None
    fake.RapidOCR = FakeRapid
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake)
    got = ocr.RapidEngine().words(b"png", "en")
    assert [w[3] for w in got] == ["Total", "assets", "5,000"] and all(w[4] == 0.97 for w in got)
    assert "rapidocr" in ocr.available() and isinstance(ocr.engine(), ocr.RapidEngine)


def test_ocr_is_tried_only_for_pages_without_a_text_layer_and_unavailable_ocr_is_said_only_then():
    from tests.fixtures import real_like as rl
    data, _ = rl.samsung()
    draft, _ = engine_module.extract(data, "s.pdf", "pdf", {"country": "KR"})
    assert not any(f["step"] == "ocr-pages" for f in draft["fallbacks"]), "a PDF with a text layer was offered OCR"


def _png():
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, "PNG")
    return buf.getvalue()


def _fake_paddle(monkeypatch, version):
    import importlib.machinery
    import sys
    import types
    seen = {}
    poly = lambda x0, x1: [[x0, 10], [x1, 10], [x1, 20], [x0, 20]]  # noqa: E731
    fake = types.ModuleType("paddleocr")
    fake.__spec__ = importlib.machinery.ModuleSpec("paddleocr", None)

    class PaddleOCR:
        def __init__(self, **kw):
            if version == 2 and "use_textline_orientation" in kw:
                raise TypeError("unexpected keyword")
            seen["kw"] = kw
        if version == 3:
            def predict(self, img):
                seen["shape"] = img.shape
                return [{"res": {"rec_polys": [poly(100, 300)], "rec_texts": ["Total assets 1,200"], "rec_scores": [0.97]}}]
        else:
            def ocr(self, img, cls=True):
                seen["shape"] = img.shape
                return [[[poly(100, 300), ("Total assets 1,200", 0.97)]]]
    fake.PaddleOCR = PaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    return seen


@pytest.mark.parametrize("version", [2, 3])
def test_paddleocr_adapter_reads_both_interfaces_and_picks_the_language_model(monkeypatch, version):
    seen = _fake_paddle(monkeypatch, version)
    eng = ocr.PaddleEngine()
    out = eng.words(_png(), "ko")
    assert [w[3] for w in out] == ["Total", "assets", "1,200"] and out[0][0] == 100 and out[-1][2] == pytest.approx(300) and out[0][4] == 0.97
    assert seen["shape"][:2] == (20, 40) and seen["kw"]["lang"] == "korean"
    assert "paddleocr" in ocr.available() and isinstance(ocr.engine(), ocr.PaddleEngine)


def test_ocr_language_follows_the_hint_then_the_country():
    assert ocr.language_for({"language": "ja"}) == "ja" and ocr.language_for({"country": "KR"}) == "ko"
    assert ocr.language_for({"country": "hk"}) == "zh-hk" and ocr.language_for({}) == "en" and ocr.language_for(None) == "en"
    assert all("paddleocr" in v for v in ocr.LANG.values())
