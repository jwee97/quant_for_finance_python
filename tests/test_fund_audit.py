"""The two checks that stand between stored fundamentals and a study: when could each figure have been known, and does a year equal its quarters. Controlled stores with answers worked out by hand."""

from __future__ import annotations

import pytest

from src.fundamentals import audit
from src.fundamentals.pit import PitSettings
from src.fundamentals.providers.base import CompanyProfile
from src.fundamentals.store import FundamentalsStore, PeriodStatement


def _store(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    sid = s.upsert_security(CompanyProfile("manual", "us:Q", "QQQ", "Quarterly Inc", exchange="US", market="US", country="US", currency="USD"))
    return s, sid


def _fy(year, revenue, **dates):
    return PeriodStatement(fiscal_period="FY", period_end=f"{year}-12-31", fiscal_year=year, values={"revenue": revenue, "net_income": revenue * 0.1}, reporting_currency="USD", **dates)


def _q(year, q, revenue, **dates):
    end = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}[q]
    return PeriodStatement(fiscal_period=q, period_end=f"{year}-{end}", fiscal_year=year, values={"revenue": revenue, "net_income": revenue * 0.1}, reporting_currency="USD", **dates)


# ------------------------------------------------------------------------------------------------------------------------ point-in-time audit
def test_real_filing_dates_make_a_study_ready_and_the_lag_is_the_filing_date_minus_the_period_end(tmp_path):
    s, sid = _store(tmp_path)
    s.store_financials(sid, "manual", [_fy(2020, 100, filing_date="2021-02-20"), _fy(2021, 120, filing_date="2022-03-01"), _fy(2022, 130, filing_date="2023-02-15")])
    a = audit.pit_audit(s, PitSettings(info_lag_days=1))
    assert a["periods"] == 3 and a["estimated"] == 0 and a["actual_share"] == 1.0 and a["impossible"] == 0
    assert a["verdict"]["status"] == "ready" and not a["assumptions"]
    lags = sorted([51 + 1, 60 + 1, 46 + 1])                                      # period end to filing, plus the one day's lag; 2020-12-31 -> 2021-02-20 is 51 days, and so on
    assert a["lag_days"]["min"] == lags[0] and a["lag_days"]["max"] == lags[-1] and a["lag_days"]["median"] == 52
    assert a["by_period"][0]["fiscal_period"] == "FY" and a["by_period"][0]["actual_dates"] == 3 and a["by_period"][0]["assumed_dates"] == 0


def test_periods_without_a_date_are_assumed_with_the_lag_named_and_never_the_period_end(tmp_path):
    s, sid = _store(tmp_path)
    s.store_financials(sid, "manual", [_fy(2020, 100, filing_date="2021-02-20"), _fy(2021, 120), _q(2022, "Q1", 30)])
    a = audit.pit_audit(s)
    assert a["estimated"] == 2 and a["verdict"]["status"] == "ready_with_assumptions" and a["impossible"] == 0
    text = " ".join(a["assumptions"])
    assert "2 of 3 periods" in text and "FY 120 days" in text and "Q1 60 days" in text
    fy = next(r for r in a["by_period"] if r["fiscal_period"] == "FY")
    assert fy["assumed_dates"] == 1 and fy["assumed_lag_days"] == 120
    assert a["worst_companies"][0]["assumed_share"] == pytest.approx(2 / 3)
    left_out = audit.pit_audit(s, PitSettings(use_estimated=False))
    assert "left out" in " ".join(left_out["assumptions"])


def test_a_filing_dated_before_the_period_ends_blocks_the_study(tmp_path):
    s, sid = _store(tmp_path)
    s.store_financials(sid, "manual", [_fy(2020, 100, filing_date="2020-11-01")])
    a = audit.pit_audit(s, PitSettings(info_lag_days=0))
    assert a["impossible"] == 1 and a["verdict"]["status"] == "blocked" and "wrong date" in a["verdict"]["reasons"][0]


def test_the_naive_availability_switch_blocks_the_study_and_an_empty_store_has_nothing_to_use(tmp_path):
    s, sid = _store(tmp_path)
    assert audit.pit_audit(s)["verdict"]["status"] == "blocked"
    s.store_financials(sid, "manual", [_fy(2020, 100, filing_date="2021-02-20")])
    a = audit.pit_audit(s, PitSettings(naive_availability=True, info_lag_days=1))
    assert a["verdict"]["status"] == "blocked" and "look-ahead by construction" in a["verdict"]["reasons"][0]


# ------------------------------------------------------------------------------------------------------------------------ annual against quarterly
def test_a_year_that_equals_its_four_quarters_reconciles_and_one_that_does_not_is_reported_with_the_gap(tmp_path):
    s, sid = _store(tmp_path)
    s.store_financials(sid, "manual", [_fy(2021, 400)] + [_q(2021, q, v) for q, v in (("Q1", 90), ("Q2", 100), ("Q3", 105), ("Q4", 105))]
                       + [_fy(2022, 500)] + [_q(2022, q, v) for q, v in (("Q1", 110), ("Q2", 120), ("Q3", 130), ("Q4", 150))])
    out = audit.annual_vs_quarterly(s, sid, metrics=("revenue", "net_income"))
    by = {(r["fiscal_year"], r["metric"]): r for r in out["rows"]}
    assert by[(2021, "revenue")]["status"] == "reconciles" and by[(2021, "revenue")]["sum_parts"] == 400 and by[(2021, "revenue")]["difference"] == 0
    assert by[(2022, "revenue")]["status"] == "differs" and by[(2022, "revenue")]["sum_parts"] == 510 and by[(2022, "revenue")]["difference"] == 10
    assert by[(2022, "revenue")]["relative_difference"] == pytest.approx(0.02)
    assert by[(2021, "net_income")]["status"] == "reconciles" and by[(2022, "net_income")]["status"] == "differs"
    assert out["summary"]["reconciles"] == 2 and out["summary"]["differs"] == 2


def test_missing_quarters_are_incomplete_and_never_a_match_and_a_year_with_no_quarters_says_so(tmp_path):
    s, sid = _store(tmp_path)
    s.store_financials(sid, "manual", [_fy(2021, 400), _q(2021, "Q1", 100), _q(2021, "Q2", 100), _q(2021, "Q3", 100)] + [_fy(2022, 500)])
    out = audit.annual_vs_quarterly(s, sid, metrics=("revenue",))
    by = {r["fiscal_year"]: r for r in out["rows"]}
    assert by[2021]["status"] == "incomplete" and by[2021]["parts"] == 3 and by[2021]["parts_needed"] == 4 and by[2021]["difference"] is None, "three quarters that add to 300 are not compared with 400"
    assert by[2022]["status"] == "no_quarters"
    assert out["summary"]["reconciles"] == 0 and "incomplete" in out["note"]


# ------------------------------------------------------------------------------------------------------------------------ through the endpoints
def test_the_endpoints_return_plain_json_and_refuse_a_missing_company(tmp_path):
    import json

    from src.webapp.api import ApiError
    from tests.test_fund_api import get, make_app, post
    from tests.test_fund_import_api import company

    app = make_app(tmp_path)
    sid = company(app)["security_id"]
    audit_out = get(app, "/api/fund/pit-audit", db="real", ids=str(sid))
    json.dumps(audit_out, allow_nan=False)
    assert audit_out["periods"] >= 1 and audit_out["verdict"]["status"] in ("ready", "ready_with_assumptions") and audit_out["settings"]["info_lag_days"] == 1
    cmp = get(app, "/api/fund/periods/compare", db="real", id=str(sid))
    json.dumps(cmp, allow_nan=False)
    assert cmp["security_id"] == sid and {r["metric"] for r in cmp["rows"]} <= {"revenue", "net_income", "operating_cash_flow"} and "incomplete" in cmp["note"]
    support = get(app, "/api/fund/import/support")
    json.dumps(support, allow_nan=False)
    assert any(r["id"] == "pdf_text" for r in support["formats"])
    with pytest.raises(ApiError):
        get(app, "/api/fund/periods/compare", db="real")
    with pytest.raises(ApiError):
        get(app, "/api/fund/pit-audit", db="real", ids="1,x")
