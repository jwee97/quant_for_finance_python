"""The layout-aware extraction engine: IR, statements, taxonomy, checks, gating of low-confidence figures, database migration, versioned reprocessing and the benchmark regression."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from src.fundamentals import quality, schema
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.importer import checks, diagnostics, engine, ir, matcher, statements, taxonomy
from src.fundamentals.importer.layout import html_layout, pdf_layout, xlsx_layout
from src.fundamentals.importer.layout._cells import to_number
from src.fundamentals.importer.pipeline import ImportService
from src.fundamentals.pit import PeriodTables, PitSettings
from src.fundamentals.providers.base import CompanyProfile
from src.fundamentals.store import FundamentalsStore

BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "fundamentals_importer"
DATA = BENCH / "data"
pymupdf = pytest.importorskip("pymupdf")


def _pdf(pages, font=None):
    doc = pymupdf.open()
    for lines in pages:
        page, y = doc.new_page(), 60
        for line in lines:
            for x, text, right in line:
                if right:
                    x = x - pymupdf.get_text_length(text, fontname=font or "helv", fontsize=10)
                page.insert_text((x, y), text, fontsize=10, fontname=font or "helv")
            y += 15
    return doc.tobytes()


def _statement(title, header, rows, unit="(in millions)"):
    lines = [[(40, title, False)], [(40, unit, False)], [(430, header[0], True), (520, header[1], True)]]
    for label, a, b in rows:
        lines.append([(40, label, False), (430, a, True), (520, b, True)])
    return lines


# ------------------------------------------------------------------------------------------------------------------------ cells and the IR
def test_cell_numbers_read_every_market_convention():
    assert to_number("(1,234)") == (-1234.0, False) and to_number("△1,234") == (-1234.0, False) and to_number("1.234,5") == (1234.5, False) and to_number("1.300") == (1300.0, False)
    assert to_number("15 170") == (15170.0, False) and to_number("-") == (None, True) and to_number("12%")[0] is None and to_number("Revenue")[0] is None and to_number("$ 1,200")[0] == 1200.0


def test_the_ir_survives_json_and_carries_coordinates_pages_and_titles():
    data = _pdf([_statement("Consolidated Income Statement", ("2023", "2022"), [("Revenue", "1,200", "1,000"), ("Net income", "150", "120")])])
    doc = pdf_layout.extract(data)
    again = ir.Document.from_dict(json.loads(json.dumps(doc.to_dict())))
    t = again.tables[0]
    assert t.page == 1 and "Income Statement" in t.title and t.header_rows == [0]
    revenue = next(c for c in t.cells if c.text == "1,200")
    assert revenue.number == 1200.0 and revenue.col == 1 and revenue.bbox and revenue.bbox[2] > revenue.bbox[0]
    assert again.format == "pdf" and again.to_dict() == doc.to_dict()


def test_a_missing_figure_stays_in_its_own_column():
    lines = _statement("Consolidated Income Statement", ("2023", "2022"), [("Revenue", "1,200", "1,000")])
    lines.append([(40, "Net income", False), (520, "(40)", True)])                                    # only the second column has a figure
    doc = pdf_layout.extract(_pdf([lines]))
    row = [c for c in doc.tables[0].cells if c.row == 2]
    assert [(c.col, c.text) for c in row if c.col] == [(2, "(40)")]


def test_html_and_workbook_readers_make_tables_too():
    html = b"<html><body><h2>Consolidated Balance Sheet</h2><table><tr><th></th><th>2023</th><th>2022</th></tr><tr><td>Total assets</td><td>$</td><td>5,000</td><td>$</td><td>4,500</td></tr></table></body></html>"
    d = html_layout.extract(html)
    assert d.tables and "Balance Sheet" in d.tables[0].title
    openpyxl = pytest.importorskip("openpyxl")
    import io
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in (["Consolidated Income Statement"], ["", "2023", "2022"], ["Revenue", 1200, 1000], ["Net income", 150, 120]):
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    x = xlsx_layout.from_xlsx(buf.getvalue())
    assert x.tables[0].header_rows == [0, 1] and x.tables[0].title.startswith("Consolidated Income")


# ------------------------------------------------------------------------------------------------------------------------ statement-first extraction
def _report(extra=()):
    pages = [_statement("Consolidated Income Statement", ("2023", "2022"), [("Revenue", "1,200", "1,000"), ("Cost of revenue", "700", "600"), ("Gross profit", "500", "400"), ("Net income", "150", "120")]),
             _statement("Consolidated Balance Sheet", ("31 December 2023", "31 December 2022"), [("Total assets", "5,000", "4,500"), ("Total liabilities", "3,000", "2,800"), ("Total shareholders' equity", "2,000", "1,700")]),
             _statement("Consolidated Statement of Cash Flows", ("2023", "2022"), [("Net cash provided by operating activities", "220", "190"), ("Net cash used in investing activities", "(90)", "(80)")])]
    return _pdf(pages + list(extra))


def test_values_come_from_the_statement_and_scope_they_belong_to_not_from_a_label_that_matches():
    traps = [_statement("Note 5 - Segment information", ("2023", "2022"), [("Revenue", "480", "400"), ("Net income", "45", "36")]),
             _statement("Balance Sheet of the Company (Parent Company Only)", ("31 December 2023", "31 December 2022"), [("Total assets", "3,000", "2,700"), ("Total liabilities", "1,800", "1,700")]),
             _statement("Financial Highlights (Five-Year Summary)", ("2023", "2022"), [("Revenue", "1,284", "1,000"), ("Net income", "165", "120")])]
    draft, doc = engine.extract(_report(traps), "r.pdf", "pdf", {"country": "US"})
    by = {p["period_end"]: p["values"] for p in draft["periods"]}
    assert by["2023-12-31"]["revenue"] == 1200 and by["2023-12-31"]["net_income"] == 150 and by["2023-12-31"]["total_assets"] == 5000 and by["2023-12-31"]["total_liabilities"] == 3000
    reasons = " ".join(r["reason"] for r in draft["rejections"])
    assert "parent-company" in reasons and "not a primary statement" in reasons
    assert {s["kind"] for s in draft["statements"] if s["confidence"] > 0} >= {"IS", "BS", "CF"}
    src = next(p for p in draft["periods"] if p["period_end"] == "2023-12-31")["sources"]["revenue"]
    assert src["statement"] == "income statement" and src["scope"] in ("consolidated", "unspecified") and src["page"] == 1 and src["bbox"] and src["confidence"] >= 0.9


def test_a_line_found_in_the_wrong_statement_is_not_taken():
    page = _statement("Consolidated Balance Sheet", ("31 December 2023", "31 December 2022"), [("Total assets", "5,000", "4,500"), ("Total liabilities", "3,000", "2,800"), ("Revenue", "1,200", "1,000")])
    draft, _ = engine.extract(_pdf([page]), "r.pdf", "pdf", {"country": "US"})
    assert all("revenue" not in p["values"] for p in draft["periods"]) and any("belongs to" in r["reason"] for r in draft["rejections"])


def test_wrapped_labels_are_joined_not_matched_in_halves():
    page = _statement("Consolidated Balance Sheet", ("31 December 2023", "31 December 2022"), [("Total assets", "5,000", "4,500")])
    page.append([(40, "Total current", False)])
    page.append([(40, "assets", False), (430, "2,100", True), (520, "2,000", True)])
    draft, _ = engine.extract(_pdf([page]), "r.pdf", "pdf", {"country": "US"})
    v = next(p for p in draft["periods"] if p["period_end"] == "2023-12-31")["values"]
    assert v["total_assets"] == 5000 and v["current_assets"] == 2100


def test_period_columns_unit_and_currency_come_from_the_document_in_each_script():
    kr = _pdf([[[(40, "(단위 : 백만원)", False)], [(40, "제 55 기 : 2023.12.31 현재", False)], [(40, "제 54 기 : 2022.12.31 현재", False)], [(40, "연결재무상태표", False)],
                [(300, "제 55(당) 기", True), (420, "제 54(전) 기", True)], [(40, "자산총계", False), (300, "2,000", True), (420, "1,800", True)], [(40, "부채총계", False), (300, "1,200", True), (420, "1,100", True)]]], font="korea")
    d, _ = engine.extract(kr, "kr.pdf", "pdf", {"country": "KR"})
    assert d["standard"]["standard"] == "k-ifrs" and d["units"]["money"] == "millions" and d["currency"] == "KRW"
    assert {p["period_end"] for p in d["periods"]} == {"2023-12-31", "2022-12-31"}


# ------------------------------------------------------------------------------------------------------------------------ taxonomy mappings
def test_mappings_are_per_standard_and_ordinary_income_is_not_net_income():
    assert taxonomy.get("j-gaap").match("経常利益") is None and taxonomy.get("j-gaap").match("売上高").code == "revenue"
    assert taxonomy.get("k-ifrs").match("Ⅰ. 매출액").code == "revenue" and taxonomy.get("ifrs").match("Revenue from related parties") is None
    assert taxonomy.detect_standard({"country": "SG"})[0] == "sfrs-i" and taxonomy.detect_standard({}, "", "ko")[0] == "k-ifrs" and taxonomy.detect_standard({}, "prepared under US GAAP")[0] == "us-gaap"
    assert taxonomy.detect_standard({"standard": "cas", "country": "US"}) == ("cas", "hint")


def test_a_custom_taxonomy_can_be_registered_and_a_typo_is_refused():
    m = taxonomy.custom("acme-chart", {"Net turnover": "revenue", "Result for the year": "net_income"}, version="3")
    assert taxonomy.get("acme-chart").match("Net turnover").code == "revenue" and m.version == "3" and any(s["standard"] == "acme-chart" for s in taxonomy.standards())
    with pytest.raises(ValueError, match="not canonical"):
        taxonomy.custom("bad", {"Sales": "revnue"})


# ------------------------------------------------------------------------------------------------------------------------ reconciliations and confidence
def _draft(values, conf=0.9):
    return {"periods": [{"fiscal_period": "FY", "period_end": "2023-12-31", "values": dict(values), "sources": {k: {"confidence": conf} for k in values}}]}


def test_reconciliations_raise_confidence_when_they_agree_and_lower_it_when_they_do_not():
    good = _draft({"total_assets": 5000, "total_liabilities": 3000, "shareholders_equity": 2000})
    out = checks.run(good)
    assert out[0]["id"] == "bs_identity" and out[0]["status"] == "pass" and good["periods"][0]["sources"]["total_assets"]["confidence"] > 0.9
    bad = _draft({"total_assets": 5000, "total_liabilities": 3000, "shareholders_equity": 1000})
    out = checks.run(bad)
    assert out[0]["status"] == "fail" and bad["periods"][0]["sources"]["total_assets"]["confidence"] < checks.LOW_CONFIDENCE
    s = checks.run(_draft({"revenue": 3_000_000, "total_assets": 5}))
    assert any(c["id"] == "scale" and c["status"] == "fail" for c in s)
    dup = checks.run(_draft({"revenue": 1}), conflicts=[{"period": "2023-12-31", "code": "revenue", "detail": "1 vs 2"}])
    assert any(c["id"] == "duplicate_conflict" for c in dup)
    xs = checks.run(_draft({"net_income": 100, "cash_and_equivalents": 50}), cross={"2023-12-31": {"cf_net_income": 101, "cf_end_cash": 50}})
    assert {c["id"]: c["status"] for c in xs} == {"ni_cross": "pass", "cash_cross": "pass"}


# ------------------------------------------------------------------------------------------------------------------------ a figure nobody has confirmed never reaches the Alpha Lab
@pytest.fixture()
def store(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    s.upsert_security(CompanyProfile("manual", "us:RNI", "RNI", "Round Numbers Inc", exchange="US", market="US", country="US", currency="USD"))
    return s


def _structured(svc, store, revenue_conf):
    sid = int(store.securities().iloc[0]["security_id"])
    batch = svc.create_batch("t", security_id=sid)
    draft = {"method": "pdf-layout", "units": {"money": "units", "shares": "units"}, "currency": "USD", "units_stated": True, "notes": [], "company": {},
             "periods": [{"fiscal_period": "FY", "period_end": "2023-12-31", "fiscal_year": 2023, "filing_date": "2024-02-20", "currency": "USD", "values": {"revenue": 1000.0, "net_income": 100.0, "total_assets": 5000.0},
                          "sources": {"revenue": {"confidence": revenue_conf, "statement": "income statement"}, "net_income": {"confidence": 0.95}, "total_assets": {"confidence": 0.95}}}],
             "versions": {"extractor": "9.9.9", "mapping": "1.0.0", "taxonomy": "1.0.0"}}
    return svc.add_structured(batch, "r.pdf", None, b"unique-" + str(revenue_conf).encode(), draft, kind="file")["item_id"], sid


def test_a_low_confidence_figure_is_stored_unverified_kept_out_of_research_until_confirmed(store):
    svc = ImportService(store)
    item, sid = _structured(svc, store, 0.5)
    issues = svc.item(item, full=True)["issues"]
    assert any(i["kind"] == "low_confidence" for i in issues)
    svc.approve(item)
    obs = store.observations([sid])
    flags = dict(zip(obs["metric_code"], obs["quality_flag"]))
    assert flags["revenue"] == "unverified" and flags["net_income"] == "ok" and "unverified" in quality.DEFAULT_EXCLUDE and "unverified" in PitSettings().exclude_flags
    tables = PeriodTables(store, [sid])
    assert tables.fy["revenue"].isna().all() and tables.fy["net_income"].notna().all(), "research does not see the unverified figure"
    quality.check_security(store, sid)                                                         # the data-quality pass must not clear it
    assert dict(zip(store.observations([sid])["metric_code"], store.observations([sid])["quality_flag"]))["revenue"] == "unverified"
    with store.db.transaction() as c:
        row = c.execute("SELECT confidence, verified, extractor_version, statement FROM observation_provenance WHERE metric_code = 'revenue'").fetchone()
    assert row[0] == 0.5 and row[1] == 0 and row[2] == "9.9.9" and row[3] == "income statement"
    assert svc.verify_saved(item, ["revenue"])["confirmed"] == 1
    assert PeriodTables(store, [sid]).fy["revenue"].notna().all()


def test_a_person_who_types_or_confirms_a_low_confidence_figure_makes_it_verified(store):
    svc = ImportService(store)
    item, sid = _structured(svc, store, 0.4)
    svc.review(item, {"verify": ["revenue"]})
    svc.approve(item)
    flags = dict(zip(store.observations([sid])["metric_code"], store.observations([sid])["quality_flag"]))
    assert flags["revenue"] == "ok"


# ------------------------------------------------------------------------------------------------------------------------ versioned extractions and reprocessing
def test_reprocessing_keeps_every_reading_and_never_touches_stored_figures(store, monkeypatch):
    svc = ImportService(store)
    sid = int(store.securities().iloc[0]["security_id"])
    batch = svc.create_batch("t", security_id=sid)
    pdf = _report()
    item = svc.add_file(batch, "r.pdf", pdf)[0]["item_id"]
    svc.process(batch, hint={"country": "US", "filing_date": "2024-02-20"})
    first = svc.extractions(item)
    assert len(first) == 1 and first[0]["extractor_version"] == taxonomy.EXTRACTOR_VERSION and first[0]["standard"] == "us-gaap"
    svc.approve(item, security={"security_id": sid})
    before = store.observations([sid]).sort_values(["metric_code", "period_id"]).reset_index(drop=True)
    monkeypatch.setattr(taxonomy, "EXTRACTOR_VERSION", "2.1.0")
    out = svc.reprocess(item)
    assert out["saved_figures_untouched"] and len(out["versions"]) == 2 and out["versions"][1]["extractor_version"] == "2.1.0" and out["changed"] is False
    after = store.observations([sid]).sort_values(["metric_code", "period_id"]).reset_index(drop=True)
    assert before[["metric_code", "value", "quality_flag"]].equals(after[["metric_code", "value", "quality_flag"]])
    assert svc.item(item)["status"] == "saved"


def test_an_old_database_is_migrated_without_losing_rows_and_an_old_build_can_still_open_the_new_one(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(schema.DDL)
    conn.execute("DROP TABLE observation_provenance")
    conn.execute("DROP TABLE import_extractions")
    conn.execute("CREATE TABLE observation_provenance (id INTEGER PRIMARY KEY, financial_period_id INTEGER NOT NULL, metric_code TEXT NOT NULL, source_kind TEXT NOT NULL, source_ref TEXT, content_hash TEXT NOT NULL DEFAULT '', "
                 "accession TEXT, page INTEGER, taxonomy_tag TEXT, reported_value REAL, reported_unit TEXT, extraction_method TEXT, extraction_status TEXT NOT NULL DEFAULT 'approved', edited INTEGER NOT NULL DEFAULT 0, "
                 "import_item_id INTEGER, created_at TEXT NOT NULL)")
    conn.execute("INSERT INTO observation_provenance (financial_period_id, metric_code, source_kind, content_hash, reported_value, created_at) VALUES (1, 'revenue', 'file', 'abc', 1234.0, '2020-01-01')")
    conn.commit()
    conn.close()
    db = schema.Database(path)
    with db.transaction() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(observation_provenance)")}
        row = c.execute("SELECT reported_value, verified, confidence FROM observation_provenance").fetchone()
        version = c.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"confidence", "statement", "scope", "extractor_version", "mapping_version", "original_currency", "bbox", "verified"} <= cols
    assert row[0] == 1234.0 and row[1] == 0 and row[2] is None and "import_extractions" in tables and int(version) == schema.SCHEMA_VERSION, "additive only: the schema version is unchanged"
    schema.Database(path).transaction().__enter__().close()                                    # opening twice is harmless


# ------------------------------------------------------------------------------------------------------------------------ diagnostics
def test_the_diagnostics_count_what_was_read_what_reconciled_and_what_was_not_used(store):
    svc = ImportService(store)
    sid = int(store.securities().iloc[0]["security_id"])
    batch = svc.create_batch("t", security_id=sid)
    svc.add_file(batch, "r.pdf", _report([_statement("Note 5 - Segment information", ("2023", "2022"), [("Revenue", "480", "400"), ("Net income", "45", "36")])]))
    svc.add_file(batch, "junk.pdf", _pdf([[[(40, "nothing to see here", False)]]]))
    svc.process(batch, hint={"country": "US"})
    d = diagnostics.live(store)
    assert d["items"] == 2 and d["by_status"].get("failed") == 1 and d["extraction_success"] == 0.5 and d["statement_coverage"] == {"IS": 1.0, "BS": 1.0, "CF": 1.0}
    assert d["reconciliation"].get("pass", 0) >= 1 and any("not a primary statement" in k for k in d["rejection_reasons"]) and d["failure_reasons"]
    assert d["extractions_by_version"][0]["extractor_version"] == taxonomy.EXTRACTOR_VERSION
    assert diagnostics.benchmark()["available"] and "synthetic" in diagnostics.benchmark()["note"]


# ------------------------------------------------------------------------------------------------------------------------ the cross-market regression suite
@pytest.fixture(scope="module")
def results():
    sys.path.insert(0, str(BENCH))
    import harness
    manifest = json.loads((DATA / "manifest.json").read_text())
    return [harness.run_case(c, DATA) for c in manifest]


def test_the_held_out_files_have_not_been_changed_since_they_were_frozen():
    h = hashlib.sha256()
    for f in sorted((DATA / "heldout").glob("*")):
        h.update(f.name.encode() + f.read_bytes())
    assert h.hexdigest() == (BENCH / "heldout.lock").read_text().strip(), "the held-out split changed: regenerate a NEW held-out batch instead of editing this one, and say so in the report"


def test_every_market_and_format_is_covered_in_every_split(results):
    markets = {"US", "SG", "JP", "KR", "CN", "HK", "EU"}
    for split in ("dev", "heldout"):
        assert {r["market"] for r in results if r["split"] == split} == markets
    held = [r for r in results if r["split"] == "heldout"]
    assert any(r["scanned"] for r in held) and any(r["restated"] for r in held) and {r["period_type"] for r in held} == {"annual", "half", "quarter"}
    assert {r["language"] for r in held} >= {"en", "ja", "ko", "zh", "zh-hk", "fr", "de", "es"} and {r["format"] for r in held} == {"pdf", "html", "xlsx"}
    assert {t for r in held for t in r["traps"]} == {"parent", "segment", "summary"}


def test_acceptance_thresholds_hold_on_dev_and_validation_and_no_split_has_a_confident_error(results):
    from harness import aggregate
    for split in ("dev", "validation"):
        a = aggregate([r for r in results if r["split"] == split])["overall"]
        assert a["extraction_success"] == 1.0 and a["recall"] >= 0.98 and a["critical_accuracy"] >= 0.98 and a["unit_accuracy"] == 1.0 and a["currency_accuracy"] == 1.0, (split, a)
    assert sum(r["confident_errors"] for r in results) == 0, [r["id"] for r in results if r["confident_errors"]]


def test_held_out_regression_floor_and_traps_and_scans(results):
    from harness import aggregate
    held = [r for r in results if r["split"] == "heldout"]
    a = aggregate(held)["overall"]
    assert a["recall"] >= 0.65 and a["critical_accuracy"] >= 0.65 and a["period_accuracy"] >= 0.95 and a["unit_accuracy"] == 1.0 and a["currency_accuracy"] == 1.0, a
    assert all(r["wrong"] == 0 for r in held if r["traps"]), "a parent-company, segment or summary table supplied a wrong value"
    scans = [r for r in held if r["scanned"]]
    assert sum(r["wrong"] for r in scans) == sum(r["wrong_caught"] for r in scans), "an OCR-damaged value was stored as confident"
