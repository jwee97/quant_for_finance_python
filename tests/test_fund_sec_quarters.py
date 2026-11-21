"""Quarterly statements from the SEC's 10-Q filings: Q1 to Q3 as filed (the year to date de-cumulated where a quarter is not tagged), Q4 as the year less the nine months, each public on the day of its own
filing. The payload is built by hand with round numbers whose quarters add up to the year, so every figure can be checked on paper; real-company annual figures stay tested in test_fund_sec_edgar.py."""

from __future__ import annotations

from datetime import date

import pytest

from src.fundamentals import canonical
from src.fundamentals.providers.sec_edgar import ANNUAL_FORMS, parse_companyfacts, trim_companyfacts
from src.fundamentals.providers.sec_quarters import QUARTER_FORMS, locate, parse_quarters


def fact(val, end, start=None, accn="A", filed="2021-05-05", form="10-Q", fy=2021):
    f = {"val": val, "end": end, "accn": accn, "filed": filed, "form": form, "fy": fy, "fp": "Q"}
    if start:
        f["start"] = start
    return f


def payload():
    """Revenue 100/110/120/130 (year 460), net income 10/12/14/16 (52), operating cash flow 20/25/30/35 (110) reported only as the year to date, assets and equity at every quarter end."""
    q = {"Q1": ("2021-01-01", "2021-03-31", "0001-21-000001", "2021-05-05"), "Q2": ("2021-01-01", "2021-06-30", "0001-21-000002", "2021-08-05"), "Q3": ("2021-01-01", "2021-09-30", "0001-21-000003", "2021-11-04")}
    rev, ni, cfo, assets, equity, eps = [], [], [], [], [], []
    # fiscal 2020 (only the annual report: it fixes the year end)
    k0 = dict(accn="0001-21-000000", filed="2021-02-20", form="10-K", fy=2020)
    rev.append(fact(400, "2020-12-31", "2020-01-01", **k0)); ni.append(fact(40, "2020-12-31", "2020-01-01", **k0)); cfo.append(fact(90, "2020-12-31", "2020-01-01", **k0))
    assets.append(fact(1000, "2020-12-31", **k0)); equity.append(fact(500, "2020-12-31", **k0)); eps.append(fact(0.4, "2020-12-31", "2020-01-01", **k0))
    three = {"Q1": (100, 10), "Q2": (110, 12), "Q3": (120, 14)}
    ytd_cfo = {"Q1": 20, "Q2": 45, "Q3": 75}
    ytd_rev = {"Q1": 100, "Q2": 210, "Q3": 330}
    ytd_ni = {"Q1": 10, "Q2": 22, "Q3": 36}
    for i, (name, (start, end, accn, filed)) in enumerate(q.items()):
        kw = dict(accn=accn, filed=filed, form="10-Q", fy=2021)
        s3 = {"Q1": "2021-01-01", "Q2": "2021-04-01", "Q3": "2021-07-01"}[name]
        rev.append(fact(three[name][0], end, s3, **kw)); ni.append(fact(three[name][1], end, s3, **kw))
        if name != "Q1":
            rev.append(fact(ytd_rev[name], end, start, **kw)); ni.append(fact(ytd_ni[name], end, start, **kw))
        cfo.append(fact(ytd_cfo[name], end, start, **kw))
        assets.append(fact(1100 + 10 * i, end, **kw)); equity.append(fact(520 + 10 * i, end, **kw))
        eps.append(fact(0.10 + 0.01 * i, end, s3, **kw))
    k1 = dict(accn="0001-22-000009", filed="2022-02-20", form="10-K", fy=2021)
    rev.append(fact(460, "2021-12-31", "2021-01-01", **k1)); ni.append(fact(52, "2021-12-31", "2021-01-01", **k1)); cfo.append(fact(110, "2021-12-31", "2021-01-01", **k1))
    assets.append(fact(1200, "2021-12-31", **k1)); equity.append(fact(560, "2021-12-31", **k1)); eps.append(fact(0.52, "2021-12-31", "2021-01-01", **k1))
    # the first quarter of the next fiscal year, filed after the last year end we hold
    kn = dict(accn="0001-22-000020", filed="2022-05-05", form="10-Q", fy=2022)
    rev.append(fact(140, "2022-03-31", "2022-01-01", **kn)); ni.append(fact(15, "2022-03-31", "2022-01-01", **kn)); cfo.append(fact(30, "2022-03-31", "2022-01-01", **kn))
    assets.append(fact(1250, "2022-03-31", **kn)); equity.append(fact(570, "2022-03-31", **kn))

    def unit(items, u="USD"):
        return {"units": {u: items}}
    return {"cik": 1, "entityName": "Round Numbers Inc", "facts": {"us-gaap": {
        "Revenues": unit(rev), "NetIncomeLoss": unit(ni), "NetCashProvidedByUsedInOperatingActivities": unit(cfo), "Assets": unit(assets), "StockholdersEquity": unit(equity),
        "EarningsPerShareDiluted": unit(eps, "USD/shares")}}}


def by_label(periods):
    return {(p.fiscal_year, int(p.fiscal_period[1:])): p for p in periods}


def v(period, code):
    return canonical.normalise_values("sec_edgar", period.values)[0][code]["value"]


def test_the_quarters_are_placed_in_the_companys_own_fiscal_year():
    ends = [(date(2020, 12, 31), 2020), (date(2021, 12, 31), 2021)]
    assert locate(date(2021, 3, 31), ends) == (2021, 1) and locate(date(2021, 6, 30), ends) == (2021, 2) and locate(date(2021, 9, 30), ends) == (2021, 3)
    assert locate(date(2021, 12, 31), ends) is None, "a year end is not a quarter"
    assert locate(date(2022, 3, 31), ends) == (2022, 1) and locate(date(2022, 9, 30), ends) == (2022, 3), "after the last year end held: the next fiscal year"
    assert locate(date(2020, 6, 30), ends) == (2020, 2), "before the first year end held: the years before it"
    retailer = [(date(2021, 1, 30), 2020), (date(2022, 1, 29), 2021)]
    assert locate(date(2021, 4, 30), retailer) == (2021, 1) and locate(date(2021, 10, 30), retailer) == (2021, 3)


def test_q1_to_q3_are_as_filed_and_a_flow_reported_only_year_to_date_is_the_difference_of_two_filings():
    periods, info = parse_quarters(payload())
    q = by_label(periods)
    assert {k for k in q if k[0] == 2021} == {(2021, 1), (2021, 2), (2021, 3), (2021, 4)} and (2022, 1) in q and not any(k[1] == 4 and k[0] == 2022 for k in q)
    assert [v(q[(2021, n)], "revenue") for n in (1, 2, 3, 4)] == [100, 110, 120, 130], "three-month facts where the filing tags one, the year less nine months for Q4"
    assert [v(q[(2021, n)], "net_income") for n in (1, 2, 3, 4)] == [10, 12, 14, 16]
    assert [v(q[(2021, n)], "operating_cash_flow") for n in (1, 2, 3, 4)] == [20, 25, 30, 35], "cash flow comes only as the year to date: de-cumulated, then Q4 from the year"
    assert [v(q[(2021, n)], "total_assets") for n in (1, 2, 3, 4)] == [1100, 1110, 1120, 1200], "a balance is the one at the quarter end; Q4's is the year-end balance"
    assert sum(v(q[(2021, n)], "revenue") for n in (1, 2, 3, 4)) == 460 and sum(v(q[(2021, n)], "operating_cash_flow") for n in (1, 2, 3, 4)) == 110, "the quarters add up to the year"
    assert v(q[(2022, 1)], "revenue") == 140 and v(q[(2022, 1)], "operating_cash_flow") == 30


def test_each_quarter_is_public_from_its_own_filing_and_q4_is_marked_as_derived():
    q = by_label(parse_quarters(payload())[0])
    assert [q[(2021, n)].filing_date for n in (1, 2, 3, 4)] == ["2021-05-05", "2021-08-05", "2021-11-04", "2022-02-20"] and q[(2022, 1)].filing_date == "2022-05-05"
    assert q[(2021, 2)].provider_record_id == "0001-21-000002" and q[(2021, 4)].provider_record_id == "0001-22-000009#Q4"
    assert [q[(2021, n)].period_end for n in (1, 2, 3, 4)] == ["2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31"] and all(q[k].fiscal_period == f"Q{k[1]}" for k in q)


def test_per_share_figures_are_kept_only_when_the_quarter_itself_is_tagged_and_never_derived_for_q4():
    q = by_label(parse_quarters(payload())[0])
    assert v(q[(2021, 1)], "eps_diluted") == pytest.approx(0.10) and v(q[(2021, 3)], "eps_diluted") == pytest.approx(0.12)
    assert "eps_diluted" not in canonical.normalise_values("sec_edgar", q[(2021, 4)].values)[0], "a year's EPS less nine months' EPS is not the quarter's EPS"


def test_a_quarter_with_no_way_to_get_its_flow_is_left_without_it_not_guessed():
    p = payload()
    cfo = p["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"]
    p["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"] = [f for f in cfo if f["accn"] != "0001-21-000002"]      # the Q2 filing's year to date is not there
    q = by_label(parse_quarters(p)[0])
    assert "operating_cash_flow" not in canonical.normalise_values("sec_edgar", q[(2021, 2)].values)[0] and "operating_cash_flow" not in canonical.normalise_values("sec_edgar", q[(2021, 3)].values)[0]
    assert v(q[(2021, 1)], "operating_cash_flow") == 20 and v(q[(2021, 2)], "revenue") == 110


def test_a_trimmed_copy_with_the_quarterly_forms_reads_back_the_same_quarters_and_the_annual_reading_is_unchanged():
    full = payload()
    trimmed = trim_companyfacts(full, ANNUAL_FORMS | QUARTER_FORMS)
    assert [(p.fiscal_year, p.fiscal_period, p.period_end) for p in parse_quarters(trimmed)[0]] == [(p.fiscal_year, p.fiscal_period, p.period_end) for p in parse_quarters(full)[0]]
    annual = parse_companyfacts(trim_companyfacts(full))[0]
    assert [(p.fiscal_year, p.fiscal_period) for p in annual] == [(2020, "FY"), (2021, "FY")], "the annual reading still sees only the annual forms"
    assert len(trim_companyfacts(full)["facts"]["us-gaap"]["Revenues"]["units"]["USD"]) < len(trimmed["facts"]["us-gaap"]["Revenues"]["units"]["USD"])


def test_a_company_with_no_ten_q_has_no_quarters():
    p = payload()
    for el in p["facts"]["us-gaap"].values():
        for u in el["units"]:
            el["units"][u] = [f for f in el["units"][u] if f["form"] == "10-K"]
    assert parse_quarters(p)[0] == []
