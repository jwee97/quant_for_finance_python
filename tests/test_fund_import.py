"""Report import: files, links and archives into reviewed, provenance-tagged figures.

What is pinned: each file type is read into the right lines and periods (an inline-XBRL filing exactly as tagged, a CSV and an Excel workbook through the Enter-reports reader, a PDF from the words on its
page with the page and line kept); a year-to-date cash flow is never stored as a quarter; a batch resumes; a file is read once (a duplicate is recorded as one); an archive is opened with limits; a
person's correction is kept apart from the first reading; nothing is stored until approval, a conflict with held figures asks for a decision and a later filing is stored as a new vintage, never over the
old one; every stored figure says where it came from; a link to a private address is refused, a server error is retried and the SEC needs a declared contact."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from src.fundamentals import canonical
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.importer import fetch, pdf, pipeline, reconcile, tabular, xbrl
from src.fundamentals.importer.fetch import Fetcher
from src.fundamentals.importer.pipeline import ImportService, sniff
from src.fundamentals.store import CompanyProfile, FundamentalsStore

CTX = '<xbrli:context id="{id}"><xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">0000000001</xbrli:identifier></xbrli:entity><xbrli:period>{period}</xbrli:period></xbrli:context>'


def ixbrl(period_end="2023-12-31", doc="10-K", focus="FY", year="2023", flows=None, ytd=None, scale=6, extra="", start="2023-01-01") -> bytes:
    """An inline-XBRL filing the way a filer writes one: figures in millions (scale 6), a company-wide context for the year and the balance date, and a segment context that must be ignored."""
    flows = {"Revenues": 1200, "NetIncomeLoss": 150, "OperatingIncomeLoss": 220} if flows is None else flows
    facts = "".join(f'<ix:nonFraction name="us-gaap:{k}" contextRef="dur" unitRef="usd" scale="{scale}" decimals="-6">{v:,}</ix:nonFraction>' for k, v in flows.items())
    for k, v in (ytd or {}).items():
        facts += f'<ix:nonFraction name="us-gaap:{k}" contextRef="ytd" unitRef="usd" scale="{scale}">{v:,}</ix:nonFraction>'
    stocks = {"Assets": 5000, "Liabilities": 3000, "StockholdersEquity": 2000, "CashAndCashEquivalentsAtCarryingValue": 400}
    facts += "".join(f'<ix:nonFraction name="us-gaap:{k}" contextRef="inst" unitRef="usd" scale="{scale}">{v:,}</ix:nonFraction>' for k, v in stocks.items())
    facts += '<ix:nonFraction name="us-gaap:Revenues" contextRef="seg" unitRef="usd" scale="6">999,999</ix:nonFraction>'                 # a business segment: not the company's figure
    return f'''<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" xmlns:xbrli="http://www.xbrl.org/2003/instance"
xmlns:us-gaap="http://fasb.org/us-gaap/2023" xmlns:dei="http://xbrl.sec.gov/dei/2023" xmlns:iso4217="http://www.xbrl.org/2003/iso4217" xmlns:xbrldi="http://xbrl.org/2006/xbrldi"><body>
<ix:header><ix:resources>{CTX.format(id="dur", period=f"<xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{period_end}</xbrli:endDate>")}
{CTX.format(id="inst", period=f"<xbrli:instant>{period_end}</xbrli:instant>")}
{CTX.format(id="ytd", period=f"<xbrli:startDate>2023-01-01</xbrli:startDate><xbrli:endDate>{period_end}</xbrli:endDate>")}
<xbrli:context id="seg"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier><xbrli:segment><xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">x:A</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
<xbrli:period><xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{period_end}</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit></ix:resources></ix:header>
<ix:nonNumeric name="dei:DocumentType" contextRef="dur">{doc}</ix:nonNumeric><ix:nonNumeric name="dei:DocumentPeriodEndDate" contextRef="dur">{period_end}</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus" contextRef="dur">{focus}</ix:nonNumeric><ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="dur">{year}</ix:nonNumeric>
<ix:nonNumeric name="dei:EntityRegistrantName" contextRef="dur">Round Numbers Inc</ix:nonNumeric><ix:nonNumeric name="dei:TradingSymbol" contextRef="dur">RNI</ix:nonNumeric>{facts}{extra}</body></html>'''.encode()


def csv_report(unit_line="(in millions)", revenue=1200) -> bytes:
    return f"Line item,FY2023,FY2022\nRevenue,{revenue},1000\nNet income,150,120\nTotal assets,5000,4500\nTotal liabilities,3000,2700\nTotal shareholders equity,2000,1800\n{unit_line}\n".encode()


def pdf_report() -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    y = [60]

    def row(label, a=None, b=None):
        page.insert_text((40, y[0]), label, fontsize=9)
        if a is not None:
            page.insert_text((330, y[0]), a, fontsize=9)
        if b is not None:
            page.insert_text((430, y[0]), b, fontsize=9)
        y[0] += 14

    row("CONSOLIDATED STATEMENTS OF INCOME (in thousands, except per share data)")
    row("Year ended December 31,", "2023", "2022")
    row("Total revenues", "1,200,000", "1,000,000")
    row("Operating income", "200,000", "150,000")
    row("Net income", "150,000", "(40,000)")
    row("Total assets", "5,000,000", "4,500,000")
    row("Total liabilities", "3,000,000")                                           # one figure under two headings: its column is unknown
    return doc.tobytes()


@pytest.fixture()
def store(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    s.upsert_security(CompanyProfile("manual", "us:RNI", "RNI", "Round Numbers Inc", exchange="US", market="US", country="US", currency="USD"))
    return s


@pytest.fixture()
def svc(store):
    return ImportService(store)


def sid(store) -> int:
    return int(store.securities().iloc[0]["security_id"])


def one(svc, batch, name, data, **kw):
    items = svc.add_file(batch, name, data, **kw)
    svc.process(batch)
    return svc.item(items[0]["item_id"], full=True)


# ------------------------------------------------------------------------------------------------------------------------ reading each kind of file
def test_an_inline_xbrl_filing_is_read_exactly_as_tagged_and_a_segment_is_not_the_company():
    d = xbrl.extract(ixbrl(), "0000000001-24-000005.htm")
    p = d["periods"][0]
    assert (d["method"], d["accession"], d["document_type"]) == ("ixbrl", "0000000001-24-000005", "10-K") and d["company"]["ticker"] == "RNI" and d["currency"] == "USD"
    assert (p["fiscal_period"], p["period_end"], p["fiscal_year"]) == ("FY", "2023-12-31", 2023)
    assert p["values"]["revenue"] == 1200e6 and p["values"]["net_income"] == 150e6 and p["values"]["total_assets"] == 5000e6 and p["values"]["shareholders_equity"] == 2000e6, "scale 6 is millions, exactly"
    assert p["sources"]["revenue"]["tag"] == "us-gaap:Revenues" and p["sources"]["revenue"]["method"] == "xbrl" and "gross_profit" not in p["values"], "derived lines are the application's to make, not imported as reported"


def test_a_ten_q_keeps_the_quarter_and_leaves_year_to_date_lines_out_instead_of_storing_them_as_a_quarter():
    data = ixbrl("2023-09-30", "10-Q", "Q3", flows={"Revenues": 300, "NetIncomeLoss": 40}, ytd={"NetCashProvidedByUsedInOperatingActivities": 280, "Revenues": 880}, start="2023-07-01")
    d = xbrl.extract(data, "x.htm")
    p = d["periods"][0]
    assert p["fiscal_period"] == "Q3" and p["values"]["revenue"] == 300e6, "the three-month fact, not the nine-month one"
    assert "operating_cash_flow" not in p["values"] and "NetCashProvidedByUsedInOperatingActivities" in " ".join(d["ytd_only"]) and any("year to date" in n for n in d["notes"])


def test_a_file_that_is_not_a_filing_or_has_nothing_to_read_is_refused_with_a_reason():
    with pytest.raises(FundamentalsError, match="not an XBRL|not readable"):
        xbrl.extract(b"<html><body>hello</body></html>")
    with pytest.raises(FundamentalsError, match="none of the lines"):
        xbrl.extract(ixbrl(flows={"SomethingUnknown": 5}).replace(b"us-gaap:Assets", b"us-gaap:Zzz").replace(b"us-gaap:Liabilities", b"us-gaap:Zzz2").replace(b"us-gaap:StockholdersEquity", b"us-gaap:Zzz3").replace(b"us-gaap:CashAndCashEquivalentsAtCarryingValue", b"us-gaap:Zzz4"))


def test_a_csv_and_an_excel_workbook_read_through_the_enter_reports_table_and_the_stated_unit():
    d = tabular.from_text(csv_report().decode(), "csv")
    assert d["units"]["money"] == "millions" and [p["period_end"][:4] for p in d["periods"]] == ["2023", "2022"] and d["periods"][0]["values"]["revenue"] == 1200
    assert any("does not say" in n for n in tabular.from_text(csv_report(unit_line="").decode())["notes"]), "an unstated unit is said, not assumed"
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "Cover"
    wb.active.append(["Annual report"])
    ws = wb.create_sheet("Statements")
    for r in [["(in thousands)"], ["Line item", "FY2023", "FY2022"], ["Revenue", 1200, 1000], ["Net income", 150, 120], ["Total assets", 5000, 4500]]:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    x = tabular.from_xlsx(buf.getvalue())
    assert x["method"] == "xlsx" and x["units"]["money"] == "thousands" and x["periods"][0]["values"]["total_assets"] == 5000 and "Statements" in x["notes"][0]


def test_a_pdf_is_read_from_its_words_with_the_page_and_line_and_always_asks_for_review():
    d = pdf.extract(pdf_report())
    assert d["units"]["money"] == "thousands" and d["review_required"] and d["method"] == "pdf-text"
    by_end = {p["period_end"]: p for p in d["periods"]}
    assert by_end["2023-12-31"]["values"]["revenue"] == 1_200_000 and by_end["2022-12-31"]["values"]["net_income"] == -40_000, "a bracketed figure is negative"
    src = by_end["2023-12-31"]["sources"]["revenue"]
    assert src["page"] == 1 and "Total revenues" in src["text"] and src["method"] == "pdf-text"
    assert "total_liabilities" not in by_end["2023-12-31"]["values"] and any("1 figure(s) under a heading of 2" in s["message"] for s in d["skipped"]), "a figure whose column is unknown is not stored"


def test_a_pdf_with_no_text_layer_is_refused_rather_than_guessed():
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(FundamentalsError, match="no text layer"):
        pdf.extract(doc.tobytes())


def test_what_a_file_is_comes_from_its_bytes_before_its_name():
    assert sniff("report.txt", pdf_report()) == "pdf" and sniff("x.csv", ixbrl()) == "xbrl" and sniff("a.csv", csv_report()) == "csv" and sniff("b.bin", b"\x00\x01\x02") == "unknown"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.csv", "x")
    assert sniff("a.zip", buf.getvalue()) == "zip"


# ------------------------------------------------------------------------------------------------------------------------ batches
def test_a_batch_reads_each_file_once_resumes_and_records_a_repeat_as_a_duplicate(svc, store):
    batch = svc.create_batch("history", sid(store))
    a = svc.add_file(batch, "FY2023.csv", csv_report())[0]
    b = svc.add_file(batch, "copy.csv", csv_report())[0]
    assert a["status"] == "queued" and b["status"] == "duplicate" and "already in this batch" in b["error"]
    svc.add_file(batch, "filing.htm", ixbrl())
    out = svc.process(batch, stop=lambda: svc.batch(batch)["counts"].get("needs_review", 0) + svc.batch(batch)["counts"].get("extracted", 0) >= 1)
    assert out["processed"] == 1 and out["remaining"] == 1, "stopped half way"
    again = svc.process(batch)
    assert again["processed"] == 1 and again["remaining"] == 0, "it carries on where it stopped; what was read is not read again"
    counts = svc.batch(batch)["counts"]
    assert counts.get("duplicate") == 1 and sum(counts.values()) == 3
    assert [h["batch_id"] for h in svc.history()] == [batch] and svc.history()[0]["items"] == 3


def test_an_archive_is_opened_member_by_member_with_limits(svc, store, monkeypatch):
    batch = svc.create_batch("zip", sid(store))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("2023/FY2023.csv", csv_report())
        z.writestr("2023/filing.htm", ixbrl())
        z.writestr("__MACOSX/._junk", b"x")
        z.writestr("notes.bin", b"\x00\x01\x02\x03")
    items = svc.add_file(batch, "reports.zip", buf.getvalue())
    assert sorted(i["source_name"] for i in items) == ["FY2023.csv", "filing.htm", "notes.bin"] and all(i["parent_item"] for i in items)
    svc.process(batch)
    got = {i["source_name"]: i["status"] for i in svc.batch(batch)["items"] if i["source_kind"] != "zip"}
    assert got["notes.bin"] == "failed" and got["filing.htm"] in ("extracted", "needs_review") and got["FY2023.csv"] in ("extracted", "needs_review")
    monkeypatch.setattr(pipeline, "MAX_ZIP_MEMBERS", 2)
    with pytest.raises(FundamentalsError, match="at most 2 are read"):
        svc.add_file(batch, "big.zip", buf.getvalue() + b"")
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as z:
        z.writestr("inner.zip", buf.getvalue())
    monkeypatch.setattr(pipeline, "MAX_ZIP_MEMBERS", 200)
    with pytest.raises(FundamentalsError, match="inside an archive"):
        svc.add_file(batch, "outer.zip", nested.getvalue())


# ------------------------------------------------------------------------------------------------------------------------ review and approval
def test_a_correction_is_kept_apart_from_the_first_reading_and_checked_again(svc, store):
    batch = svc.create_batch("fix", sid(store))
    it = one(svc, batch, "FY2023.csv", csv_report(revenue=12000))                                # a typo: ten times too big against revenue's own ratio? the identity still holds, units do not
    first = it["draft"]["periods"][0]["values"]["revenue"]
    fixed = svc.review(it["item_id"], {"periods": [{"values": {"revenue": 1200}, "filing_date": "2024-02-08"}, {}]})
    assert fixed["draft"]["periods"][0]["values"]["revenue"] == first == 12000, "the first reading is untouched"
    assert fixed["reviewed"]["periods"][0]["values"]["revenue"] == 1200 and fixed["reviewed"]["periods"][0]["sources"]["revenue"]["edited"] is True and fixed["reviewed"]["periods"][0]["filing_date"] == "2024-02-08"
    with pytest.raises(FundamentalsError, match="not a line"):
        svc.review(it["item_id"], {"periods": [{"values": {"made_up": 1}}]})


def test_nothing_is_stored_until_approval_and_every_stored_figure_says_where_it_came_from(svc, store):
    batch = svc.create_batch("filing", sid(store))
    items = svc.add_file(batch, "0000000001-24-000005.htm", ixbrl())
    svc.process(batch, hint={"filing_date": "2024-02-08"})
    item = svc.item(items[0]["item_id"], full=True)
    assert item["status"] == "extracted" and store.periods([sid(store)]).empty, "extracted is not stored"
    loose = svc.create_batch("no company chosen")
    stray = svc.add_file(loose, "stray.htm", ixbrl("2022-12-31", year="2022", start="2022-01-01"))[0]
    svc.process(loose)
    with pytest.raises(FundamentalsError, match="choose the company"):
        svc.approve(stray["item_id"], security={})
    out = svc.approve(item["item_id"], security={"security_id": sid(store)})
    assert out["item"]["status"] == "saved" and out["saved"][0]["lines"] >= 5
    periods = store.periods([sid(store)])
    row = periods.iloc[0]
    assert row["provider"] == "manual" and str(row["filing_date"])[:10] == "2024-02-08" and row["pit_quality"] == "filed"
    prov = {p["metric_code"]: p for p in svc.provenance(sid(store))}
    assert prov["revenue"]["accession"] == "0000000001-24-000005" and prov["revenue"]["taxonomy_tag"] == "us-gaap:Revenues" and prov["revenue"]["reported_value"] == 1200e6
    assert prov["revenue"]["extraction_method"] == "xbrl" and prov["revenue"]["edited"] == 0 and prov["revenue"]["source_kind"] == "xbrl"
    with pytest.raises(FundamentalsError, match="already saved"):
        svc.approve(item["item_id"], security={"security_id": sid(store)})
    again = svc.add_file(batch, "same-again.htm", ixbrl())[0]
    assert again["status"] == "duplicate" and "already saved" in again["error"], "a file already saved is not read again"


def test_a_figure_that_differs_from_the_held_one_needs_a_decision_and_a_later_filing_is_a_new_vintage(svc, store):
    batch = svc.create_batch("restate", sid(store))
    first = svc.add_file(batch, "orig.htm", ixbrl())[0]
    svc.process(batch, hint={"filing_date": "2024-02-08"})
    svc.approve(first["item_id"], security={"security_id": sid(store)})
    changed = ixbrl(flows={"Revenues": 1180, "NetIncomeLoss": 150, "OperatingIncomeLoss": 220})
    second = svc.add_file(batch, "amended.htm", changed)[0]
    svc.process(batch, hint={"filing_date": "2024-09-30"})
    item = svc.item(second["item_id"], full=True)
    assert item["status"] == "needs_review" and any(c["kind"] == "restatement" and c["metric"] == "revenue" and c["stored"] == 1200e6 and c["imported"] == 1180e6 for c in item["draft"]["conflicts"])
    with pytest.raises(FundamentalsError, match="restatement"):
        svc.approve(second["item_id"], security={"security_id": sid(store)})
    out = svc.approve(second["item_id"], security={"security_id": sid(store)}, decision="restate")
    assert out["saved"][0]["vintage"] == 1
    stored = store.periods([sid(store)]).sort_values("vintage")
    assert list(stored["vintage"]) == [0, 1] and str(stored.iloc[1]["filing_date"])[:10] in ("2024-09-30", "2024-02-08"), "both vintages are there: the old figure was not overwritten"
    values = store.observations([sid(store)], ["revenue"]).sort_values("period_id")
    assert list(values["value"]) == [1200e6, 1180e6]


def test_a_duplicate_of_held_figures_is_said_to_add_nothing_and_a_hundredfold_unit_slip_is_an_error(svc, store):
    batch = svc.create_batch("dup", sid(store))
    a = svc.add_file(batch, "a.htm", ixbrl())[0]
    svc.process(batch, hint={"filing_date": "2024-02-08"})
    svc.approve(a["item_id"], security={"security_id": sid(store)})
    b = svc.add_file(batch, "b.csv", b"Line item,FY2023\nRevenue,1200\nNet income,150\nTotal assets,5000\n(in millions)\n")[0]
    svc.process(batch)
    issues = svc.item(b["item_id"], full=True)["issues"]
    assert any(i["kind"] == "duplicate" and i["level"] == "info" for i in issues)
    c = svc.add_file(batch, "thousands.csv", b"Line item,FY2022\nRevenue,1100\nTotal assets,4800\n(in thousands)\n")[0]
    svc.process(batch)
    bad = svc.item(c["item_id"], full=True)
    assert any(i["kind"] == "units_vs_history" and i["level"] == "error" for i in bad["issues"])
    with pytest.raises(FundamentalsError, match="not approved"):
        svc.approve(c["item_id"], security={"security_id": sid(store)})


def test_the_cash_flow_must_tie_to_the_change_in_cash_and_quarters_must_add_up_to_the_year():
    def p(fp, end, **v):
        return {"fiscal_period": fp, "period_end": end, "fiscal_year": int(end[:4]), "values": v}
    tied = {"units": {"money": "units"}, "periods": [p("FY", "2022-12-31", cash_and_equivalents=100), p("FY", "2023-12-31", cash_and_equivalents=160, operating_cash_flow=90, investing_cash_flow=-40, financing_cash_flow=10)]}
    assert not [i for i in reconcile.within_draft(tied) if i["kind"] == "cash_tie"]
    tied["periods"][1]["values"]["operating_cash_flow"] = 20
    assert [i for i in reconcile.within_draft(tied) if i["kind"] == "cash_tie"]
    year = {"units": {"money": "units"}, "periods": [p("FY", "2023-12-31", revenue=460), p("Q1", "2023-03-31", revenue=100), p("Q2", "2023-06-30", revenue=110), p("Q3", "2023-09-30", revenue=120), p("Q4", "2023-12-31", revenue=130)]}
    assert not reconcile.within_draft(year)
    year["periods"][4]["values"]["revenue"] = 330                                             # the nine-month figure typed as the fourth quarter
    assert [i for i in reconcile.within_draft(year) if i["kind"] == "quarters_vs_year"]
    ytd = {"units": {"money": "units"}, "periods": [p("Q1", "2023-03-31", operating_cash_flow=20), p("Q2", "2023-06-30", operating_cash_flow=45), p("Q3", "2023-09-30", operating_cash_flow=75)]}
    assert [i for i in reconcile.within_draft(ytd) if i["kind"] == "ytd_as_quarter"]


# ------------------------------------------------------------------------------------------------------------------------ links
PUBLIC = lambda host, port, type=None: [(2, 1, 6, "", ("93.184.216.34", port))]            # noqa: E731


def test_a_link_to_a_private_or_odd_address_is_refused():
    private = lambda host, port, type=None: [(2, 1, 6, "", ("10.0.0.5", port))]            # noqa: E731
    for url, why in (("ftp://example.org/a.pdf", "only http"), ("https://user:pw@example.org/a.pdf", "user name"), ("https://", "no host"), ("", "give the link")):
        with pytest.raises(FundamentalsError, match=why):
            fetch.check_url(url, PUBLIC)
    for ip in ("10.0.0.5", "127.0.0.1", "169.254.169.254", "192.168.1.1", "::1"):
        with pytest.raises(FundamentalsError, match="not a public address"):
            fetch.check_url("https://example.org/a.pdf", lambda h, p, type=None, ip=ip: [(2, 1, 6, "", (ip, p))])
    assert fetch.check_url("https://example.org/report.pdf", PUBLIC) == "https://example.org/report.pdf"
    with pytest.raises(FundamentalsError, match="not a public address"):
        fetch.check_url("http://internal.example/a", private)


def test_a_fetch_retries_a_busy_server_waits_between_requests_and_checks_every_redirect():
    calls, waits = [], []
    script = [(503, {}, b""), (429, {"Retry-After": "2"}, b""), (200, {"Content-Type": "application/pdf; charset=x"}, b"%PDF-1.4 data")]

    def transport(url, headers, timeout):
        calls.append(url)
        return script.pop(0)
    f = Fetcher(transport=transport, resolver=PUBLIC, sleep=waits.append, clock=lambda: 0.0)
    got = f.fetch("https://example.org/report.pdf")
    assert got.data.startswith(b"%PDF") and got.content_type == "application/pdf" and len(calls) == 3 and 2.0 in waits and 1.0 in waits, "503 then 429 (honouring Retry-After), then the report"
    hops = [(302, {"Location": "http://10.0.0.1/x"}, b"")]
    with pytest.raises(FundamentalsError, match="not a public address"):
        Fetcher(transport=lambda *a: hops.pop(0), resolver=lambda h, p, type=None: [(2, 1, 6, "", ("10.0.0.1" if h == "10.0.0.1" else "93.184.216.34", p))], sleep=lambda s: None).fetch("https://example.org/x")
    with pytest.raises(FundamentalsError, match="did not answer after several tries"):
        Fetcher(transport=lambda *a: (0, {}, b""), resolver=PUBLIC, sleep=lambda s: None, retries=2).fetch("https://example.org/x")
    with pytest.raises(FundamentalsError, match="HTTP 404"):
        Fetcher(transport=lambda *a: (404, {}, b""), resolver=PUBLIC, sleep=lambda s: None).fetch("https://example.org/x")


def test_the_sec_needs_a_declared_contact_and_a_link_becomes_a_batch_item(svc, store, monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    with pytest.raises(FundamentalsError, match="SEC_EDGAR_USER_AGENT"):
        Fetcher(transport=lambda *a: (200, {}, b"x"), resolver=PUBLIC, sleep=lambda s: None).fetch("https://www.sec.gov/Archives/edgar/data/1/x.htm")
    sent = {}

    def transport(url, headers, timeout):
        sent.update(headers)
        return 200, {"Content-Type": "application/xhtml+xml"}, ixbrl()
    svc.fetcher = Fetcher(transport=transport, resolver=PUBLIC, sleep=lambda s: None, user_agent="Test Person test@example.org")
    batch = svc.create_batch("link", sid(store))
    items = svc.add_url(batch, "https://www.sec.gov/Archives/edgar/data/1/000000000124000005/rni-20231231.htm")
    assert items[0]["source_kind"] == "xbrl" and items[0]["source_url"].endswith("rni-20231231.htm") and sent["User-Agent"] == "Test Person test@example.org"
    svc.process(batch)
    assert svc.batch(batch)["items"][0]["status"] in ("extracted", "needs_review")
