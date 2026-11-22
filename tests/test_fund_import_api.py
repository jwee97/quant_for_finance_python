"""Report import and data coverage through the Fundamentals tab's endpoints (``/api/fund/import/*``, ``/api/fund/coverage/overview``): a file sent in pieces is put together and added once, read as a
job, reviewed, approved into the database with its provenance, and shown in the history; the SEC's filing list is read from the submissions record; the coverage grid counts held, derived and missing
years and quarters; every refusal is a message and every answer plain JSON. Real data only."""

from __future__ import annotations

import base64
import json

import pytest

from src.webapp.api import ApiError
from tests.test_fund_api import get, make_app, post, run
from tests.test_fund_import import ixbrl


@pytest.fixture()
def app(tmp_path):
    return make_app(tmp_path)


def plain(value):
    json.dumps(value, allow_nan=False)
    return value


def upload(app, batch, name, data, size=120):
    """Send a file the way the page does: base64 pieces of at most ``size`` bytes."""
    pieces = [data[i:i + size] for i in range(0, len(data), size)] or [b""]
    out = None
    for i, piece in enumerate(pieces):
        out = post(app, "/api/fund/import/chunk", batch_id=batch, upload_id="u1", name=name, index=i, total=len(pieces), data=base64.b64encode(piece).decode())
    return out


def company(app):
    from tests.test_fund_manual import COMPANY
    return post(app, "/api/fund/manual/save", company=dict(COMPANY), units={"money": "units", "shares": "units"}, periods=[{"fiscal_period": "FY", "period_end": "2021-12-31", "filing_date": "2022-02-10", "values": {"revenue": 900e6, "total_assets": 4000e6}}])


def test_the_routes_exist_and_the_demo_database_is_not_a_place_to_import(app):
    for path in ("/api/fund/import/history", "/api/fund/import/batch", "/api/fund/import/item", "/api/fund/import/provenance", "/api/fund/coverage/overview"):
        assert path in app.extra_get
    for path in ("create", "chunk", "urls", "process", "review", "approve", "reject", "filings", "filings/add"):
        assert f"/api/fund/import/{path}" in app.extra_post
    assert "/api/fund/coverage/backfill" in app.extra_post


def test_a_file_sent_in_pieces_is_put_together_read_reviewed_and_approved_with_its_provenance(app):
    sid = company(app)["security_id"]
    batch = post(app, "/api/fund/import/create", label="FY2023", security_id=sid)["batch_id"]
    data = ixbrl()
    first = post(app, "/api/fund/import/chunk", batch_id=batch, upload_id="x", name="0000000001-24-000005.htm", index=0, total=2, data=base64.b64encode(data[:100]).decode())
    assert first == {"received": 1, "total": 2, "done": False}
    done = post(app, "/api/fund/import/chunk", batch_id=batch, upload_id="x", name="0000000001-24-000005.htm", index=1, total=2, data=base64.b64encode(data[100:]).decode())
    assert done["done"] and done["items"][0]["source_kind"] == "xbrl" and done["items"][0]["status"] == "queued"
    again = upload(app, batch, "copy.htm", data)
    assert again["items"][0]["status"] == "duplicate", "the same bytes are recognised however they were named or sent"
    out = run(app, "/api/fund/import/process", batch_id=batch)
    assert out["processed"] == 1 and out["remaining"] == 0
    item_id = done["items"][0]["item_id"]
    item = plain(get(app, "/api/fund/import/item", id=item_id))
    assert item["status"] in ("extracted", "needs_review") and item["draft"]["periods"][0]["values"]["revenue"] == 1200e6 and item["draft"]["company"]["ticker"] == "RNI"
    reviewed = post(app, "/api/fund/import/review", item_id=item_id, edits={"periods": [{"filing_date": "2024-02-08"}]})
    assert reviewed["reviewed"]["periods"][0]["filing_date"] == "2024-02-08" and reviewed["draft"]["periods"][0]["filing_date"] is None
    with pytest.raises(Exception, match="no observation has been confirmed"):
        post(app, "/api/fund/import/approve", item_id=item_id, security_id=sid)
    obs = get(app, "/api/fund/import/observations", item_id=item_id)
    assert obs["total"] >= 5 and obs["by_status"]["UNCONFIRMED"] == obs["total"] and all(o["status"] == "UNCONFIRMED" for o in obs["rows"])
    post(app, "/api/fund/import/decide", item_id=item_id, actions=[{"action": "confirm_suggested"}])
    plan = get(app, "/api/fund/import/plan", item_id=item_id)
    assert plan["approvable"] and plan["counts"]["confirmed"] >= 5 and {l["metric"] for l in plan["will_commit"]} >= {"revenue", "net_income", "total_assets"}
    approved = post(app, "/api/fund/import/approve", item_id=item_id, security_id=sid)
    assert approved["item"]["status"] == "saved" and approved["saved"][0]["lines"] >= 5
    rows = get(app, "/api/fund/import/provenance", security_id=sid, fiscal_period="FY", period_end="2023-12-31")["rows"]
    assert {r["metric_code"] for r in rows} >= {"revenue", "net_income", "total_assets"} and all(r["accession"] == "0000000001-24-000005" for r in rows)
    hist = get(app, "/api/fund/import/history")["batches"]
    assert hist[0]["batch_id"] == batch and hist[0]["counts"].get("saved") == 1 and hist[0]["counts"].get("duplicate") == 1
    detail = get(app, "/api/fund/import/batch", id=batch)
    assert detail["counts"] == {"saved": 1, "duplicate": 1} and len(detail["items"]) == 2


def test_every_refusal_is_a_message(app):
    for path, body, why in (("/api/fund/import/chunk", {"batch_id": "B", "upload_id": "", "name": "a", "index": 0, "total": 1, "data": ""}, "upload_id"),
                            ("/api/fund/import/chunk", {"batch_id": "B", "upload_id": "u", "name": "a", "index": 3, "total": 2, "data": ""}, "describe a file"),
                            ("/api/fund/import/chunk", {"batch_id": "B", "upload_id": "u", "name": "a", "index": 0, "total": 1, "data": "!!notbase64!!"}, "base64"),
                            ("/api/fund/import/chunk", {"batch_id": "nope", "upload_id": "u", "name": "a", "index": 0, "total": 1, "data": base64.b64encode(b"x").decode()}, "no such import batch"),
                            ("/api/fund/import/urls", {"batch_id": "B", "urls": []}, "1 to 50 links"), ("/api/fund/import/review", {"item_id": 1, "edits": []}, "edits must be"),
                            ("/api/fund/import/approve", {"item_id": "x"}, "item_id must be a whole number"), ("/api/fund/import/filings", {}, "say which company"),
                            ("/api/fund/import/filings/add", {"batch_id": "B", "filings": []}, "1 to 60")):
        with pytest.raises(ApiError, match=why):
            post(app, path, **body)
    with pytest.raises(ApiError, match="no such import"):
        get(app, "/api/fund/import/batch", id="B0000")
    with pytest.raises(ApiError, match="no such import item"):
        get(app, "/api/fund/import/item", id=99)


def test_a_fetch_that_fails_is_reported_per_link_not_as_a_crash(app):
    batch = post(app, "/api/fund/import/create", label="links")["batch_id"]
    out = run(app, "/api/fund/import/urls", batch_id=batch, urls=["ftp://example.org/a.pdf", "http://localhost/report.pdf"])
    assert out["added"] == 0 and [f["error"][:20] for f in out["failed"]] == ["only http and https ", "localhost is not a pu"] or len(out["failed"]) == 2
    assert all("error" in f for f in out["failed"])


def test_the_coverage_grid_counts_held_missing_and_derived_periods(app):
    post(app, "/api/fund/manual/save", company={"ticker": "GAP", "name": "Gap Co", "exchange": "US", "country": "US", "currency": "USD"}, units={"money": "units", "shares": "units"},
         periods=[{"fiscal_period": "FY", "period_end": f"{y}-12-31", "filing_date": f"{y + 1}-02-15", "values": {"revenue": 100.0 * (y - 2018), "total_assets": 500.0}} for y in (2019, 2021, 2022)]
         + [{"fiscal_period": "Q1", "period_end": "2022-03-31", "filing_date": "2022-05-05", "values": {"revenue": 25.0, "total_assets": 500.0}}])
    cov = plain(get(app, "/api/fund/coverage/overview", years=6))
    gap = next(c for c in cov["companies"] if c["ticker"] == "GAP")
    assert gap["annual"]["2019"]["state"] == "held" and gap["annual"]["2020"]["state"] == "missing" and gap["annual"]["2021"]["state"] == "held" and gap["annual_missing"] == 1
    assert gap["quarters"]["2022Q1"]["state"] == "held" and gap["quarters"]["2022Q2"]["state"] == "missing" and gap["quarter_held"] == 1
    assert cov["totals"]["annual_missing"] >= 1 and cov["sources"][0]["provider"] == "manual" and 0 <= gap["completeness"] < 1
    assert set(cov["problems"]) >= {"awaiting_review", "failed_imports", "import_conflicts", "estimated_availability", "quality_flags"}


def test_the_sec_filing_list_reads_the_submissions_record_including_older_pages(app, tmp_path):
    from tests.test_fund_sec_edgar import provider
    prov, fake = provider(tmp_path)
    original = fake.__call__

    def with_filings(url, headers, timeout):
        if "/submissions/CIK0000320193.json" in url:
            body = {"name": "Apple Inc.", "filings": {"recent": {"form": ["10-K", "8-K", "10-Q"], "accessionNumber": ["0000320193-23-000106", "0000320193-23-000070", "0000320193-23-000077"],
                                                                 "filingDate": ["2023-11-03", "2023-08-01", "2023-08-04"], "reportDate": ["2023-09-30", "", "2023-07-01"], "primaryDocument": ["aapl-20230930.htm", "x.htm", "aapl-20230701.htm"],
                                                                 "isXBRL": [1, 0, 1], "isInlineXBRL": [1, 0, 1]}, "files": [{"name": "CIK0000320193-submissions-001.json"}]}}
            return 200, {}, json.dumps(body).encode()
        if "submissions-001" in url:
            return 200, {}, json.dumps({"form": ["10-K"], "accessionNumber": ["0000320193-10-000001"], "filingDate": ["2010-10-27"], "reportDate": ["2010-09-25"], "primaryDocument": ["a.htm"], "isXBRL": [0], "isInlineXBRL": [0]}).encode()
        return original(url, headers, timeout)
    prov.transport = with_filings
    out = prov.filings("cik:320193", ("10-K", "10-Q"), 10)
    assert [f["form"] for f in out["filings"]] == ["10-K", "10-Q", "10-K"] and out["filings"][0]["accession"] == "0000320193-23-000106"
    assert out["filings"][0]["url"] == "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm" and out["filings"][0]["xbrl"] and not out["filings"][2]["xbrl"]
    assert out["older_pages"] == 1 and out["filings"][2]["filing_date"] == "2010-10-27"


def test_coverage_expects_each_countrys_periods_not_every_period(app):
    post(app, "/api/fund/manual/save", company={"ticker": "HKX", "name": "HK Co", "exchange": "HK", "country": "HK", "currency": "HKD"}, units={"money": "units", "shares": "units"},
         periods=[{"fiscal_period": "FY", "period_end": "2020-12-31", "filing_date": "2021-03-20", "values": {"revenue": 100.0, "total_assets": 500.0}},
                  {"fiscal_period": "H1", "period_end": "2021-06-30", "filing_date": "2021-08-25", "values": {"revenue": 60.0, "total_assets": 500.0}},
                  {"fiscal_period": "FY", "period_end": "2021-12-31", "filing_date": "2022-03-20", "values": {"revenue": 120.0, "total_assets": 520.0}},
                  {"fiscal_period": "FY", "period_end": "2022-12-31", "filing_date": "2023-03-20", "values": {"revenue": 140.0, "total_assets": 540.0}}])
    post(app, "/api/fund/manual/save", company={"ticker": "USX", "name": "US Co", "exchange": "US", "country": "US", "currency": "USD"}, units={"money": "units", "shares": "units"},
         periods=[{"fiscal_period": p, "period_end": e, "filing_date": f, "values": {"revenue": 10.0, "total_assets": 50.0}} for p, e, f in (("Q1", "2021-03-31", "2021-05-05"), ("Q3", "2021-09-30", "2021-11-05"), ("FY", "2021-12-31", "2022-02-20"))])
    cov = get(app, "/api/fund/coverage/overview", years=6)
    hk, us = (next(c for c in cov["companies"] if c["ticker"] == t) for t in ("HKX", "USX"))
    assert hk["country"] == "HK" and hk["expected"] == ["H1", "FY"] and hk["standard"].startswith("HKFRS")
    assert hk["quarters"]["2021Q1"]["state"] == "na" and hk["quarters"]["2021Q3"]["state"] == "na", "a Hong Kong company files no Q1 or Q3: they are not gaps"
    assert hk["quarters"]["2021H1"]["state"] == "held" and hk["quarters"]["2022H1"]["state"] == "missing" and hk["quarters"]["2020H1"]["state"] == "missing", "the half-year it should have filed is a gap"
    assert us["quarters"]["2021Q2"]["state"] == "missing" and us["quarters"]["2021Q1"]["state"] == "held" and us["quarters"]["2021H1"]["state"] == "na"


def test_the_connector_routes_describe_search_and_fetch_within_each_sources_rules(app, monkeypatch):
    for name in ("EDINET_API_KEY", "OPENDART_API_KEY", "SEC_EDGAR_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)
    out = get(app, "/api/fund/connectors")
    by = {c["name"]: c for c in out["connectors"]}
    assert not by["edinet"]["available"] and by["cvm"]["available"] and by["hkex"]["discovery"] == "links_only" and out["calendar"]["JP"]["currency"] == "JPY" and "H1" in out["calendar"]["HK"]["periods"]
    assert run(app, "/api/fund/connectors/search", connector="hkex", company={"ticker": "0700"})["filings"] == []
    from src.webapp.api import ApiError as E
    with pytest.raises(E, match="no connector called"):
        post(app, "/api/fund/connectors/search", connector="nope", company={})
    batch = post(app, "/api/fund/import/create", label="c")["batch_id"]
    with pytest.raises(E, match="own site"):
        post(app, "/api/fund/connectors/fetch", batch_id=batch, connector="hkex", refs=[{"source": "hkex", "country": "HK", "company_id": "", "doc_id": "a", "url": "https://evil.example/a.pdf"}])
    with pytest.raises(E, match="doc_id may hold only"):
        post(app, "/api/fund/connectors/fetch", batch_id=batch, connector="cvm", refs=[{"source": "cvm", "country": "BR", "company_id": "1", "doc_id": "../../etc"}])
    bad = run(app, "/api/fund/connectors/search", connector="cvm", company={"cd_cvm": "9512"}, since="2023-01-01", until="2023-12-31")
    assert {f["doc_type"] for f in bad["filings"]} == {"DFP", "ITR"}


def test_the_point_in_time_routes_export_and_answer_as_of_a_day(app):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    company(app)
    with pytest.raises(ApiError, match="no point-in-time export"):
        get(app, "/api/fund/pit/asof", day="2023-01-01")
    m = run(app, "/api/fund/pit/export")
    assert m["rows"] >= 2 and "US/2021" in m["partitions"]
    out = plain(get(app, "/api/fund/pit/asof", day="2023-01-01", metrics="revenue"))
    assert out["rows"] == 1 and out["figures"][0]["value"] == 900e6 and out["figures"][0]["metric_code"] == "revenue"
    assert get(app, "/api/fund/pit/asof", day="2022-02-01", metrics="revenue")["rows"] == 0
    with pytest.raises(ApiError, match="day must be a date"):
        get(app, "/api/fund/pit/asof", day="yesterday")
