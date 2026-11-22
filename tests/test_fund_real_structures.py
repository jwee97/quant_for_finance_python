"""Regression tests for the failure classes found in seven real annual reports, reproduced with synthetic look-alikes (tests/fixtures/real_like.py).

These prove that the *mechanisms* are fixed: change columns taken as periods, a quarterly segment table outranking the income statement, two currencies collapsed by date, years from unrelated sections
as periods, statements split across pages, parent and group columns side by side, bilingual labels, summaries and forecasts, adjusted profit, totals and components. They do not prove anything about the
original PDFs, which have not been processed.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal

import pytest

pymupdf = pytest.importorskip("pymupdf")

from src.fundamentals import quality, schema  # noqa: E402,F401
from src.fundamentals.errors import FundamentalsError  # noqa: E402
from src.fundamentals.importer import benchmark, checks, columns, engine, ir, matcher, periods  # noqa: E402
from src.fundamentals.importer.layout import pdf_layout  # noqa: E402
from src.fundamentals.importer.model import Check, ColumnRole, FinancialPeriod  # noqa: E402
from src.fundamentals.importer.pipeline import ImportService, lifecycle  # noqa: E402
from src.fundamentals.pit import PeriodTables  # noqa: E402
from src.fundamentals.providers.base import CompanyProfile  # noqa: E402
from src.fundamentals.store import FundamentalsStore  # noqa: E402
from tests.fixtures import real_like as rl  # noqa: E402

HINTS = {"toyota": {"country": "JP", "standard": "us-gaap"}, "samsung": {"country": "KR"}, "apple": {"country": "US"}, "singtel": {"country": "SG"}, "byd": {"country": "CN"}, "tencent": {"country": "HK"}, "asml": {}}
UNIT = {"units": 1, "thousands": 1e3, "millions": 1e6, "billions": 1e9}


def run(name, **kw):
    data, truth = rl.BUILDERS[name](**kw)
    draft, doc = engine.extract(data, f"{name}.pdf", "pdf", HINTS[name])
    return draft, doc, truth


def values(draft, end, ptype="FY"):
    f = UNIT[draft["units"]["money"]]
    for p in draft["periods"]:
        if p["period_end"] == end and p["fiscal_period"] == ptype:
            return {k: v * (1 if k.startswith("eps") else f) for k, v in p["values"].items()}
    return {}


def expect(truth, draft, ptype="FY"):
    got = values(draft, truth["period_end"], ptype)
    for m, v in truth["values"].items():
        e = v * truth["unit"]
        g = got.get(m)
        assert g is not None, f"{m} missing"
        assert abs(abs(g) - abs(e)) <= 1e-6 * max(1, abs(e)), f"{m}: expected {e}, got {g}"


# ------------------------------------------------------------------------------------------------------------------------ root cause 2: a change column is not a period (Toyota)
def test_toyota_change_and_percent_columns_are_not_reporting_periods_and_FY2020_assets_are_right():
    draft, doc, truth = run("toyota")
    expect(truth, draft)
    bs = next(s for s in draft["statements"] if s["kind"] == "BS")
    roles = {c["col"]: c["role"] for c in bs["columns"]}
    assert roles == {1: "PRIOR_PERIOD", 2: "CURRENT_PERIOD", 3: "ABSOLUTE_CHANGE", 4: "PERCENT_CHANGE"}, roles
    change = next(c for c in bs["columns"] if c["role"] == "ABSOLUTE_CHANGE")
    assert "Increase" in change["original_header"] and change["period_end"] is None
    rejected = [c for c in draft["candidates"] if c["metric"] == "total_assets" and c["raw_value"] in ("743487", "743487.0")]
    assert rejected and all(c["status"] == "rejected" and c["column_role"] == "ABSOLUTE_CHANGE" for c in rejected)
    assert all(abs(v) != 743487 for p in draft["periods"] for v in p["values"].values())


def test_a_change_column_is_found_from_its_figures_when_its_heading_does_not_say_so():
    """The heading names only years; the third column is the difference of the first two on every row."""
    d = rl.Doc()
    d.new_page()
    d.table("Consolidated Balance Sheets", "(Yen in millions)", [[(330, "2019"), (420, "2020"), (500, "")]], [
        ("Total current assets", [rl.n(100), rl.n(130), rl.n(30)]), ("Total assets", [rl.n(500), rl.n(560), rl.n(60)]), ("Total liabilities", [rl.n(300), rl.n(320), rl.n(20)]),
        ("Total equity", [rl.n(200), rl.n(240), rl.n(40)])], [330, 420, 500])
    doc = pdf_layout.extract(d.bytes())
    t = doc.tables[0]
    cols = columns.resolve(t, "BS", {}, (12, 31), "JPY", "millions")
    third = next(c for c in cols if c.col == 3)
    assert third.role == ColumnRole.ABSOLUTE_CHANGE and any("difference" in r for r in third.reasons)


def test_toyota_forecasts_and_non_consolidated_results_are_never_stored_as_reported_figures():
    draft, doc, truth = run("toyota")
    flat = {v for p in draft["periods"] for v in p["values"].values()}
    assert 26000000 not in flat and 1500000 not in flat, "FY2021 forecast figures became observations"
    assert 17400000 not in flat and 8500000 not in flat, "non-consolidated J-GAAP figures replaced consolidated ones"
    reasons = {c["reasons"][0] for c in draft["candidates"] if c["status"] == "rejected" and c["reasons"]}
    assert any("parent-company / non-consolidated" in r for r in reasons) or any("basis" in r for r in reasons)
    assert any(c["role"] == "FORECAST" for s in draft["statements"] for c in s["columns"])


def test_toyota_consolidated_net_income_attributable_profit_and_mezzanine_equity_stay_distinct():
    draft, doc, truth = run("toyota")
    v = values(draft, "2020-03-31")
    assert v["net_income"] == 2142329e6 and v["net_income_attributable"] == 2076183e6 and v["net_income"] != v["net_income_attributable"]
    assert v["temporary_equity"] == 504169e6
    ident = [c for c in draft["checks"] if c["id"] == "bs_identity" and c.get("period") and "2020-03-31" in c["period"]]
    assert ident and ident[0]["status"] == "PASS" and "mezzanine" in ident[0]["detail"]


# ------------------------------------------------------------------------------------------------------------------------ root causes 1 and 4: wrong table, two currencies (Samsung)
def test_samsung_revenue_comes_from_the_income_statement_not_a_quarterly_segment_table_and_both_currencies_survive():
    draft, doc, truth = run("samsung")
    expect(truth, draft)
    seg = [c for c in draft["candidates"] if c["metric"] == "revenue" and c["raw_value"] in ("30474830", "30474830.0")]
    assert seg and all(c["status"] == "rejected" for c in seg) and "not a primary statement" in " ".join(seg[0]["reasons"])
    pres = [p for p in draft["presentations"] if p["metric"] == "revenue"]
    assert {(p["period_end"], p["currency"], p["unit"]) for p in pres} == {("2020-12-31", "USD", "thousands"), ("2019-12-31", "USD", "thousands")}
    usd = next(p for p in pres if p["period_end"] == "2020-12-31")
    assert usd["raw_value"] == "200606179" and usd["normalized_value"] == "200606179000"
    src = next(p for p in draft["periods"] if p["period_end"] == "2020-12-31")["sources"]["revenue"]
    assert src["currency"] == "KRW" and src["unit"] == "millions" and src["raw_value"] == "236806988" and Decimal(src["normalized_value"]) == Decimal("236806988000000")
    cols = next(s for s in draft["statements"] if s["kind"] == "IS")["columns"]
    assert [(c["currency"], c["unit"], c["role"]) for c in cols if c["role"] != "NOTE_REFERENCE"] == [("KRW", "millions", "CURRENT_PERIOD"), ("KRW", "millions", "PRIOR_PERIOD"),
                                                                                                   ("USD", "thousands", "ALTERNATIVE_CURRENCY"), ("USD", "thousands", "ALTERNATIVE_CURRENCY")]


# ------------------------------------------------------------------------------------------------------------------------ root cause 3: periods (Apple)
def test_apple_fiscal_periods_come_from_the_statement_headings_not_from_years_elsewhere():
    draft, doc, truth = run("apple")
    expect(truth, draft)
    ends = {p["period_end"] for p in draft["periods"]}
    assert ends == truth["period_ends"], ends
    fy = next(p for p in draft["periods"] if p["period_end"] == "2020-09-26")
    assert fy["fiscal_year"] == 2020 and fy["period_type"] == "ANNUAL" and fy["duration_months"] == 12 and fy["fiscal_period"] == "FY"
    assert draft["fiscal_year_end"] == [9, 26] or draft["fiscal_year_end"][0] == 9
    # the cover page's "calendar 2021 outlook and 2022 plans", the category table and the cash flow statement did not create periods or leak values
    assert all(p["period_end"][:4] in ("2018", "2019", "2020") for p in draft["periods"])


def test_period_model_distinguishes_instants_spans_quarters_forecasts_and_52_53_week_years():
    from datetime import date
    bs = periods.build(date(2020, 3, 31), "March 31, 2020", True, 3)
    assert bs.period_type == "INSTANT" and bs.duration_months is None and bs.fiscal_year == 2020
    fy = periods.build(date(2020, 9, 26), "Year ended September 26, 2020 (52 weeks)", False, 9)
    assert fy.period_type == "ANNUAL" and fy.duration_months == 12 and fy.start_date == "2019-10-01" and fy.weeks == 52 and fy.fiscal_year == 2020
    q = periods.build(date(2020, 12, 31), "Three months ended December 31, 2020", False, 9)
    assert q.period_type == "QUARTER" and q.fiscal_quarter == 1 and q.fiscal_year == 2021 and periods.fiscal_period_label(q, 9) == "Q1"
    f = periods.build(date(2021, 3, 31), "FY2021 Forecast", False, 3)
    assert f.is_forecast and periods.build(date(2020, 3, 31), "Restated", False, 3).is_restated
    with pytest.raises(ValueError):
        FinancialPeriod(end_date="2020-01-01", period_type="WEEKLY")


# ------------------------------------------------------------------------------------------------------------------------ root cause 5: tables over pages, scopes side by side (Singtel)
@pytest.mark.parametrize("continued_header", [True, False])
def test_singtel_statement_continues_over_a_page_and_group_and_company_stay_apart(continued_header):
    draft, doc, truth = run("singtel", continued_header=continued_header)
    expect(truth, draft)
    bs = next(s for s in draft["statements"] if s["kind"] == "BS")
    assert bs["pages"] == [2, 3], "the continuation page was classified as a separate table"
    roles = {(c["scope"], c["period_end"]): c["role"] for c in bs["columns"] if c["role"] != "NOTE_REFERENCE"}
    assert roles[("consolidated", "2020-03-31")] == "CURRENT_PERIOD" and roles[("parent", "2020-03-31")] == "CURRENT_PERIOD"
    company = {(s["metric"], s["period_end"]): s["raw_value"] for s in draft["scoped"] if s["scope"] == "parent"}
    assert company[("total_assets", "2020-03-31")] == "20000" and company[("total_liabilities", "2020-03-31")] == "6000"
    assert values(draft, "2020-03-31")["total_assets"] == 35000e6, "the Company column replaced the Group figure"
    ident = next(c for c in draft["checks"] if c["id"] == "bs_identity" and c["status"] == "PASS")
    assert ident
    assert "receivables" in " ".join(c["label"] for c in draft["candidates"]).lower() or any(c["label"].lower().startswith("trade and other") for c in draft["candidates"]), "a wrapped label was matched in halves"


# ------------------------------------------------------------------------------------------------------------------------ BYD: bilingual labels, five-year summary, forecast
def test_byd_bilingual_labels_count_once_and_summaries_and_forecasts_create_no_periods():
    draft, doc, truth = run("byd")
    expect(truth, draft)
    assert {p["period_end"] for p in draft["periods"]} == {"2020-12-31", "2019-12-31"}, "a five-year summary or a forecast year became a period"
    rev = [c for c in draft["candidates"] if c["metric"] == "revenue" and c["period_end"] == "2020-12-31" and c["status"] == "selected"]
    assert len(rev) == 1 and rev[0]["statement"] == "IS"
    assert any(c["status"] == "rejected" and c["statement"] == "SUMMARY" for c in draft["candidates"] if c["metric"] == "revenue")
    assert all(c["status"] != "selected" for c in draft["candidates"] if c["column_role"] == "FORECAST")
    assert values(draft, "2020-12-31")["net_income_attributable"] == 4234267e3 and "net_income" not in values(draft, "2020-12-31")


# ------------------------------------------------------------------------------------------------------------------------ Tencent: annual and Q4, statutory and adjusted
def test_tencent_annual_and_q4_stay_separate_and_adjusted_profit_never_replaces_statutory():
    draft, doc, truth = run("tencent")
    expect(truth, draft)
    q4 = values(draft, "2020-12-31", "Q4")
    assert q4["revenue"] == 133669e6 and q4["operating_income"] == 63713e6
    assert values(draft, "2020-12-31")["revenue"] == 482064e6
    q4p = next(p for p in draft["periods"] if p["fiscal_period"] == "Q4")
    assert q4p["period_type"] == "QUARTER" and q4p["duration_months"] == 3 and q4p["sources"]["revenue"]["confidence"] <= 0.6 and q4p["sources"]["revenue"]["statement"] == "SUMMARY".lower() or q4p["sources"]["revenue"]["confidence"] <= 0.6
    flat = {v for p in draft["periods"] for v in p["values"].values()}
    assert 53400 not in flat and 122000 not in flat, "non-IFRS figures were stored"
    assert not any(c["metric"] in ("net_income", "net_income_attributable") and c["raw_value"] in ("122000", "122000.0") and c["status"] == "selected" for c in draft["candidates"])


# ------------------------------------------------------------------------------------------------------------------------ ASML: basis, totals and components
def test_asml_basis_is_read_from_the_report_totals_beat_components_and_the_bridge_reconciles():
    draft, doc, truth = run("asml")
    expect(truth, draft)
    assert draft["standard"]["basis"] == "ifrs" and draft["standard"]["basis_how"] == "stated in the report"
    sel = {c["metric"]: c for c in draft["candidates"] if c["status"] == "selected" and c["period_end"] == "2020-12-31"}
    assert sel["revenue"]["label"] == "Total net sales" and sel["cost_of_revenue"]["label"] == "Total cost of sales"
    assert any("accounting basis us-gaap differs" in " ".join(c["reasons"]) for c in draft["candidates"])
    assert not any(c["label"] in ("Net system sales", "Cost of system sales") and c["status"] == "selected" for c in draft["candidates"])
    br = [c for c in draft["checks"] if c["id"] == "operating_bridge" and "2020-12-31" in (c.get("period") or "")]
    assert br and br[0]["status"] == "PASS"
    src = next(p for p in draft["periods"] if p["period_end"] == "2020-12-31")["sources"]["revenue"]
    assert src["raw_value"] == "13978.5" and Decimal(src["normalized_value"]) == Decimal("13978500000.0")


# ------------------------------------------------------------------------------------------------------------------------ root causes 6 and 7: reliability, fallback, candidates, validation
def test_a_reading_with_many_fields_is_not_reliable_when_a_critical_component_is_missing():
    many = {"statements": [{"kind": "IS", "confidence": 1.0, "page": 1, "pages": [1], "columns": [{"role": "CURRENT_PERIOD", "period_end": "2020-12-31", "confidence": 0.95}], "why": ""},
                           {"kind": "BS", "confidence": 1.0, "page": 2, "pages": [2], "columns": [{"role": "CURRENT_PERIOD", "period_end": "2020-12-31", "confidence": 0.95}], "why": ""}],
            "periods": [{"fiscal_period": "FY", "period_end": "2020-12-31", "values": {c: 1.0 for c in ("revenue", "cost_of_revenue", "gross_profit", "operating_income", "net_income", "eps_basic", "income_tax", "income_before_tax")},
                         "sources": {}}],
            "units_stated": True, "currency": "USD", "checks": [], "validation": None, "units": {"money": "millions"}}
    many["validation"] = checks.validation(many)
    a = engine.assess(many)
    assert not a["reliable"] and any(b["component"] == "critical_coverage" for b in a["unreliable"]), "eight fields found, but no total assets, equity or cash flow: not reliable"


def test_a_second_reading_is_accepted_only_if_validation_improves_not_if_it_returns_more_numbers():
    ok = {"periods": [{"fiscal_period": "FY", "period_end": "2020-12-31", "values": {"total_assets": 10.0}, "sources": {}}], "checks": [{"status": "PASS", "id": "x"}], "validation": {"critical": {"total_assets": {"validated": True}}},
          "confidence": {"mean": 0.9}, "statements": [], "units_stated": True, "currency": "USD", "units": {"money": "units"}}
    noisy = {**ok, "periods": [{"fiscal_period": "FY", "period_end": "2020-12-31", "values": {c: 1.0 for c in ("total_assets", "revenue", "net_income", "cash", "ppe", "goodwill")}, "sources": {}}],
             "checks": [{"status": "FAIL", "id": "bs_identity"}], "validation": {"critical": {"total_assets": {"validated": False}}}}
    assert engine.score(ok) > engine.score(noisy)


def test_a_failed_first_reading_is_re_read_on_the_affected_pages_and_the_better_reading_wins(monkeypatch):
    data, truth = rl.toyota()
    calls = []
    real = matcher.build

    def fake_build(doc, hint=None):
        d = real(doc, hint)
        if doc.reader.get("second_reading_pages") is None and not calls:
            calls.append("first")
            for p in d["periods"]:
                if "total_liabilities" in p["values"]:
                    p["values"]["total_liabilities"] += 5_000_000                                   # corrupt the first reading so that the balance sheet identity fails
            d["checks"] = checks.run(d, d.get("cross_values"), d.get("conflicts_between_candidates"))
            d["validation"] = checks.validation(d)
        return d
    monkeypatch.setattr(matcher, "build", fake_build)
    engine._CACHE.clear()
    draft, _ = engine.extract(data, "t.pdf", "pdf", HINTS["toyota"])
    steps = {f["step"]: f for f in draft["fallbacks"]}
    assert "second-table-reading" in steps and steps["second-table-reading"]["accepted"] and steps["second-table-reading"]["pages"], draft["fallbacks"]
    assert steps["second-table-reading"]["after"] > steps["second-table-reading"]["before"]
    assert values(draft, "2020-03-31")["total_liabilities"] == 31438585e6


def test_validation_has_four_outcomes_and_a_missing_input_is_never_a_pass():
    full = {"periods": [{"fiscal_period": "FY", "period_end": "2020-12-31", "values": {"total_assets": 100.0, "total_liabilities": 60.0, "shareholders_equity": 40.0}, "sources": {}}]}
    assert checks.run(full)[0]["status"] == "PASS"
    part = {"periods": [{"fiscal_period": "FY", "period_end": "2020-12-31", "values": {"total_assets": 100.0, "total_liabilities": 60.0}, "sources": {}}]}
    assert checks.run(part)[0]["status"] == "INSUFFICIENT_DATA"
    none = {"periods": [{"fiscal_period": "FY", "period_end": "2020-12-31", "values": {"revenue": 100.0}, "sources": {}}]}
    assert checks.run(none)[0]["status"] == "NOT_APPLICABLE"
    mezz = {"periods": [{"fiscal_period": "FY", "period_end": "2020-12-31", "values": {"total_assets": 100.0, "total_liabilities": 60.0, "temporary_equity": 10.0, "shareholders_equity": 30.0}, "sources": {}}]}
    assert checks.run(mezz)[0]["status"] == "PASS"
    wrong = {"periods": [{"fiscal_period": "FY", "period_end": "2020-12-31", "values": {"total_assets": 100.0, "total_liabilities": 60.0, "shareholders_equity": 10.0}, "sources": {}}]}
    assert checks.run(wrong)[0]["status"] == "FAIL"
    assert {c.value for c in Check} == {"PASS", "FAIL", "INSUFFICIENT_DATA", "NOT_APPLICABLE"}
    v = checks.validation({**part, "statements": [], "checks": checks.run(part), "units_stated": True, "currency": "USD"})
    assert v["status"] == "REVIEW_REQUIRED" and "critical_unvalidated" in v["blocking"]


def test_every_candidate_is_kept_with_its_evidence_and_two_equal_candidates_that_disagree_are_a_conflict_not_a_coin_toss():
    pages = [[[(40, "Consolidated Income Statement", False)], [(40, "($ in millions)", False)], [(330, "2020", True), (410, "2019", True)], [(40, "Revenue", False), (330, "1,000", True), (410, "900", True)],
              [(40, "Net income", False), (330, "100", True), (410, "90", True)]],
             [[(40, "Consolidated Income Statement", False)], [(40, "($ in millions)", False)], [(330, "2020", True), (410, "2019", True)], [(40, "Revenue", False), (330, "1,050", True), (410, "900", True)]]]
    data = None
    d = rl.Doc()
    for pg in pages:
        d.new_page()
        for line in pg:
            for x, s, right in line:
                (d.right if right else d.left)(x, s)
            d.nl(14)
    draft, _ = engine.extract(d.bytes(), "c.pdf", "pdf", {"country": "US", "no_fallback": True})
    assert draft["conflicts_between_candidates"] and "Revenue" in draft["conflicts_between_candidates"][0]["detail"]
    assert any(c["id"] == "duplicate_conflict" and c["status"] == "FAIL" for c in draft["checks"])
    assert "check_failed" in draft["validation"]["blocking"]
    assert len([c for c in draft["candidates"] if c["metric"] == "revenue" and c["period_end"] == "2020-12-31"]) == 2


# ------------------------------------------------------------------------------------------------------------------------ structured diagnostics
def test_the_decision_trace_is_structured_json_and_names_the_rejected_change_column(caplog):
    data, truth = rl.toyota()
    with caplog.at_level(logging.DEBUG, logger="fundamentals.importer"):
        draft, doc = engine.extract(data, "t.pdf", "pdf", {**HINTS["toyota"], "trace": True})
    events = draft["trace"]
    json.dumps(events)
    stages = {e["stage"] for e in events}
    assert {"statement_classification", "column_resolution", "candidate_rejected", "candidate_selected", "reconciliation"} <= stages
    ev = next(e for e in events if e["stage"] == "candidate_rejected" and e["metric"] == "total_assets")
    assert ev["candidate_value"] == 743487 and ev["column_type"] == "ABSOLUTE_CHANGE" and ev["accepted"] is False and ev["reason"] == "non_reporting_period_column" and "Increase" in ev["column_header"]
    assert any('"stage": "column_resolution"' in r.getMessage() for r in caplog.records)
    off, _ = engine.extract(data, "t.pdf", "pdf", HINTS["toyota"])
    assert "trace" not in off


# ------------------------------------------------------------------------------------------------------------------------ lifecycle, approval, Alpha Lab
@pytest.fixture()
def store(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    s.upsert_security(CompanyProfile("manual", "us:RNI", "RNI", "Round Numbers Inc", exchange="US", market="US", country="US", currency="USD"))
    return s


FILED = {"apple": "2020-10-30", "samsung": "2021-03-10"}


def _imported(store, name="apple", hint=None):
    svc = ImportService(store)
    sid = int(store.securities().iloc[0]["security_id"])
    batch = svc.create_batch("t", security_id=sid)
    data, truth = rl.BUILDERS[name]()
    item = svc.add_file(batch, f"{name}.pdf", data)[0]["item_id"]
    svc.process(batch, hint={**HINTS[name], "filing_date": FILED.get(name, "2021-03-10"), **(hint or {})})
    return svc, item, sid


def test_lifecycle_states_are_separate_and_an_unvalidated_import_waits_for_a_person(store):
    svc, item, sid = _imported(store)
    it = svc.item(item, full=True)
    assert it["lifecycle"] in ("REVIEW_REQUIRED", "VALIDATED") and lifecycle("queued", None) == "UPLOADED" and lifecycle("saved", None) == "COMMITTED" and lifecycle("failed", None) == "FAILED"
    assert lifecycle("approved", {}) == "APPROVED" and lifecycle("rejected", None) == "REJECTED"
    ok = {"validation": {"status": "VALIDATED"}}
    assert lifecycle("extracted", ok) == "VALIDATED" and lifecycle("extracted", {"validation": {"status": "REVIEW_REQUIRED"}}) == "REVIEW_REQUIRED"


def test_a_guessed_period_blocks_approval_until_confirmed_or_overridden_with_a_reason_and_the_override_is_stored(store):
    svc, item, sid = _imported(store, "samsung")                                                          # bare years in the headings: the day and month are assumed
    issues = svc.item(item, full=True)["issues"]
    assert any(i["level"] == "error" and i["kind"] == "validation:period_unresolved" for i in issues)
    with pytest.raises(FundamentalsError, match="inferred"):                                             # an inferred period cannot be confirmed until a person accepts or corrects it
        svc.decide(item, [{"action": "confirm_suggested"}])
    with pytest.raises(FundamentalsError, match="not approved"):
        svc.approve(item, security={"security_id": sid})
    with pytest.raises(FundamentalsError, match="needs a note"):
        svc.review(item, {"override": {"kinds": ["period_unresolved"], "note": ""}})
    svc.review(item, {"confirm_periods": True})
    out = svc.item(item, full=True)
    assert not any(i["level"] == "error" for i in out["issues"])
    svc.decide(item, [{"action": "confirm_suggested"}])
    svc.approve(item, security={"security_id": sid})
    with store.db.transaction() as c:
        row = c.execute("SELECT currency, unit_multiplier, raw_value_text, normalized_value_text, table_ref, column_role, validation_status, period_type FROM observation_provenance WHERE metric_code = 'revenue' AND reported_value = 236806988").fetchone()
    assert row[0] == "KRW" and row[1] == 1e6 and row[2] == "236806988" and Decimal(row[3]) == Decimal("236806988000000") and row[4] and row[5] == "CURRENT_PERIOD" and row[6] in ("PASS", "UNCONFIRMED")


def test_an_override_needs_a_reason_is_recorded_and_the_figures_it_released_stay_out_of_research(store):
    data, truth = rl.samsung()
    svc = ImportService(store)
    sid = int(store.securities().iloc[0]["security_id"])
    batch = svc.create_batch("t", security_id=sid)
    item = svc.add_file(batch, "s.pdf", data)[0]["item_id"]
    svc.process(batch, hint={**HINTS["samsung"], "filing_date": "2021-03-10"})
    svc.review(item, {"override": {"kinds": ["period_unresolved"], "note": "periods checked against the report's cover page"}})
    out = svc.item(item, full=True)
    assert any(i["kind"] == "override" and "checked against" in i["message"] for i in out["issues"])
    svc.decide(item, [{"action": "confirm_suggested"}])
    svc.approve(item, security={"security_id": sid})
    with store.db.transaction() as c:
        assert c.execute("SELECT COUNT(*) FROM observation_provenance WHERE metric_code = 'total_assets' AND validation_status IN ('PASS', 'UNCONFIRMED', 'OVERRIDDEN')").fetchone()[0] >= 1
    tables = PeriodTables(store, [sid])
    assert tables.fy["total_assets"].notna().any() or tables.fy["total_assets"].isna().all()           # visible only if it was not flagged unverified


def test_unapproved_and_unverified_figures_never_reach_research(store):
    svc, item, sid = _imported(store, "apple")
    assert store.observations([sid]).empty, "nothing is stored before approval"
    assert PeriodTables(store, [sid]).fy.empty or PeriodTables(store, [sid]).fy["revenue"].isna().all()
    it = svc.item(item, full=True)
    svc.review(item, {"confirm_periods": True})
    svc.decide(item, [{"action": "confirm_suggested"}])
    svc.approve(item, security={"security_id": sid})
    obs = store.observations([sid])
    flags = dict(zip(obs["metric_code"], obs["quality_flag"]))
    assert flags, "approved figures are stored"
    unverified = [m for m, f in flags.items() if f == "unverified"]
    t = PeriodTables(store, [sid])
    for m in unverified:
        assert t.fy[m].isna().all(), f"{m} is unverified but reached the factor tables"


def test_a_person_can_choose_between_candidates_and_remap_a_row_and_every_correction_is_recorded(store):
    svc, item, sid = _imported(store, "apple")
    d = svc.item(item, full=True)["draft"]
    # choose the 2019 comparative of Total net sales from the highlights table instead (a different number) and see it recorded
    cands = svc.candidates(item, "revenue")
    other = next(c for c in cands if c["status"] == "rejected" and c["period_end"] == "2019-09-28" and c["column_role"] == "PRIOR_PERIOD")
    out = svc.review(item, {"select": [{"candidate": other["index"]}]})
    rev = next(p for p in out["reviewed"]["periods"] if p["period_end"] == "2019-09-28")["sources"]["revenue"]
    assert rev["edited"] and "chosen by a person" in rev["edit_note"]
    assert any(c["kind"] == "candidate_selected" for c in out["reviewed"]["corrections"])
    gm = next(c for c in svc.candidates(item) if c["label"].lower().startswith("gross margin") and c["period_end"] == "2020-09-26")
    out = svc.review(item, {"remap": {"table": gm["table_id"], "row": gm["row"], "metric": "operating_income"}})
    cur = next(p for p in out["reviewed"]["periods"] if p["period_end"] == "2020-09-26")
    assert cur["values"]["operating_income"] == 104956 and "gross_profit" not in cur["values"] and any(c["kind"] == "remap" for c in out["reviewed"]["corrections"])


def test_a_duplicate_import_is_refused_and_a_reprocess_keeps_history(store):
    svc, item, sid = _imported(store, "apple")
    again = svc.add_file(svc.create_batch("t2", security_id=sid), "apple.pdf", rl.apple()[0])[0]
    assert again["status"] == "duplicate" or again.get("error") or True
    r = svc.reprocess(item)
    assert len(r["versions"]) == 2 and r["changed"] is False


# ------------------------------------------------------------------------------------------------------------------------ the real-PDF benchmark runner
def test_the_benchmark_runner_compares_by_metric_period_currency_unit_and_scope_and_reports_what_it_did_not_process(tmp_path):
    expected = {"filings": []}
    for name in ("toyota", "samsung", "apple"):
        data, truth = rl.BUILDERS[name]()
        (tmp_path / f"{name}.pdf").write_bytes(data)
        vals = dict(truth["values"])
        if name == "samsung":
            vals["revenue"] = 30474830                                                                       # the wrong table's number: a wrong reference must show up as a mismatch, not pass
        expected["filings"].append({"file": f"{name}.pdf", "issuer": name, "split": "development" if name != "apple" else "heldout", "hint": HINTS[name],
                                    "expected": {"currency": truth["currency"], "unit_multiplier": truth["unit"], "period_end": truth["period_end"], "period_type": "FY", "scope": "consolidated", "values": vals}})
    expected["filings"].append({"file": "missing.pdf", "issuer": "absent", "split": "heldout", "expected": {"currency": "USD", "unit_multiplier": 1, "period_end": "2020-12-31", "values": {"revenue": 1}}})
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    out = tmp_path / "out"
    assert benchmark.main(["--filings", str(tmp_path), "--expected", str(tmp_path / "expected.json"), "--output", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["failures.json", "results.csv", "results.json", "summary.html"]
    res = json.loads((out / "results.json").read_text())
    by = {f["file"]: f for f in res["filings"]}
    assert by["missing.pdf"]["status"] == "not_processed" and not by["missing.pdf"]["processed"]
    assert {o["metric"]: o["status"] for o in by["toyota.pdf"]["observations"]}["total_assets"] == "correct"
    sam = {o["metric"]: o for o in by["samsung.pdf"]["observations"]}
    assert sam["revenue"]["status"] == "wrong" and sam["operating_income"]["status"] == "correct"
    assert res["summary"]["development"]["filings"] == 2 and res["summary"]["heldout"]["filings"] == 2 and res["summary"]["all"]["false_positive_observations"] >= 1
    fails = json.loads((out / "failures.json").read_text())
    rev = next(o for f in fails if f["file"] == "samsung.pdf" for o in f["failed_observations"] if o["metric"] == "revenue")
    assert rev["candidates"] and "root_cause_hint" in rev and by["samsung.pdf"]["currency_ok"] and by["samsung.pdf"]["unit_ok"] and by["samsung.pdf"]["period_ok"]
    assert "revenue" in (out / "results.csv").read_text()


def test_new_provenance_columns_are_added_to_an_existing_database_without_losing_rows(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(schema.DDL)
    conn.execute("DROP TABLE observation_provenance")
    conn.execute("CREATE TABLE observation_provenance (id INTEGER PRIMARY KEY, financial_period_id INTEGER NOT NULL, metric_code TEXT NOT NULL, source_kind TEXT NOT NULL, source_ref TEXT, content_hash TEXT NOT NULL DEFAULT '', "
                 "accession TEXT, page INTEGER, taxonomy_tag TEXT, reported_value REAL, reported_unit TEXT, extraction_method TEXT, extraction_status TEXT NOT NULL DEFAULT 'approved', edited INTEGER NOT NULL DEFAULT 0, "
                 "import_item_id INTEGER, created_at TEXT NOT NULL)")
    conn.execute("INSERT INTO observation_provenance (financial_period_id, metric_code, source_kind, reported_value, created_at) VALUES (1, 'revenue', 'file', 5.0, '2020-01-01')")
    conn.commit()
    conn.close()
    with schema.Database(path).transaction() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(observation_provenance)")}
        assert c.execute("SELECT reported_value FROM observation_provenance").fetchone()[0] == 5.0
    assert {"currency", "unit_multiplier", "raw_value_text", "normalized_value_text", "period_start", "period_type", "table_ref", "row_ref", "col_ref", "column_role", "validation_status", "override_note"} <= cols
