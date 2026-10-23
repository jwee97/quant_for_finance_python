"""The SEC EDGAR provider: free, deep annual history for US-listed companies, read from the XBRL facts the SEC publishes.

Four real companies' facts are the fixtures (``tests/fixtures/sec``: Apple, Walmart, JPMorgan and SAP, from 2016, as the SEC served them, trimmed to what the mapping reads): their figures are checked
against the numbers the companies reported, so that the mapping is tested against the world and not against itself. Everything else is a stand-in for the SEC's servers of the documented shape.
Nothing here talks to the network and no contact address of a real person is used.
"""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src import cli
from src.fundamentals import canonical, coverage, quality
from src.fundamentals import cli as fcli
from src.fundamentals.errors import FundamentalsError, MissingKey, NotFound, PlanError, ProviderError, RateLimited
from src.fundamentals.pit import PeriodTables, PitSettings
from src.fundamentals.providers import PROVIDER_NAMES, make_provider
from src.fundamentals.providers import sec_edgar as sec
from src.fundamentals.providers.sec_edgar import SecEdgarProvider, TickerList, parse_companyfacts, sector_of, trim_companyfacts
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement
from src.fundamentals.workbench import SECRET_NAMES
from src.utils.config import load_config
from src.webapp.api import ApiError
from tests.test_fund_api import failed, get, make_app, post, run

FIXTURES = Path(__file__).parent / "fixtures" / "sec"
CONTACT = "Test Person test.person@example.org"
CIKS = {"AAPL": 320193, "WMT": 104169, "JPM": 19617, "SAP": 1000184}


def load(name: str) -> dict:
    with gzip.open(FIXTURES / f"{name}.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)


def by_year(payload: dict) -> dict[int, PeriodStatement]:
    return {p.fiscal_year: p for p in parse_companyfacts(payload)[0]}


def observed(period: PeriodStatement) -> dict:
    return canonical.normalise_values("sec_edgar", period.values)[0]


def value(period: PeriodStatement, code: str) -> float:
    return observed(period)[code]["value"]


SUBMISSIONS = {   # what /submissions/CIK##########.json says, cut to the fields the application reads
    320193: {"cik": "0000320193", "name": "Apple Inc.", "sic": "3571", "sicDescription": "Electronic Computers", "tickers": ["AAPL"], "exchanges": ["Nasdaq"], "fiscalYearEnd": "0926",
             "category": "Large accelerated filer", "lei": None},
    104169: {"cik": "0000104169", "name": "Walmart Inc.", "sic": "5331", "sicDescription": "RETAIL-VARIETY STORES", "tickers": ["WMT"], "exchanges": ["Nasdaq"], "fiscalYearEnd": "0131",
             "category": "Large accelerated filer", "lei": "5493000K4X9P0F4JXV31"},
    19617: {"cik": "0000019617", "name": "JPMORGAN CHASE & CO", "sic": "6021", "sicDescription": "National Commercial Banks", "tickers": ["JPM", "JPM-PC"], "exchanges": ["NYSE", "NYSE"],
            "fiscalYearEnd": "1231", "category": "Large accelerated filer", "lei": None},
    1000184: {"cik": "0001000184", "name": "SAP SE", "sic": "7372", "sicDescription": "Services-Prepackaged Software", "tickers": ["SAP"], "exchanges": ["NYSE"], "fiscalYearEnd": "1231",
              "category": "Large accelerated filer", "lei": None},
}
TICKER_ROWS = [(1045810, "NVDA", "NVIDIA CORP", "Nasdaq"), (320193, "AAPL", "Apple Inc.", "Nasdaq"), (789019, "MSFT", "MICROSOFT CORP", "Nasdaq"), (1652044, "GOOGL", "Alphabet Inc.", "Nasdaq"),
               (1652044, "GOOG", "Alphabet Inc.", "Nasdaq"), (1067983, "BRK-B", "BERKSHIRE HATHAWAY INC", "NYSE"), (1067983, "BRK-A", "BERKSHIRE HATHAWAY INC", "NYSE"),
               (104169, "WMT", "Walmart Inc.", "Nasdaq"), (19617, "JPM", "JPMORGAN CHASE & CO", "NYSE"), (1000184, "SAP", "SAP SE", "NYSE"), (1318605, "TSLA", "Tesla, Inc.", "Nasdaq"),
               (320187, "NKE", "NIKE, Inc.", "NYSE"), (874766, "HIG", "Hartford Insurance Group, Inc.", "NYSE"), (1137774, "PRU", "PRUDENTIAL FINANCIAL INC", "NYSE")]


def exchange_file(rows=TICKER_ROWS) -> dict:
    return {"fields": ["cik", "name", "ticker", "exchange"], "data": [[c, n, t, e] for c, t, n, e in rows]}


def plain_file(rows=TICKER_ROWS) -> dict:
    return {str(i): {"cik_str": c, "ticker": t, "title": n} for i, (c, t, n, e) in enumerate(rows)}


class FakeSec:
    """A stand-in for data.sec.gov and www.sec.gov: company facts and submissions by CIK, the ticker files, a refusal for a client that does not identify itself, and the rate-threshold page."""

    def __init__(self, ciks=("AAPL", "WMT", "JPM", "SAP")):
        self.facts = {CIKS[t]: load(t) for t in ciks}
        self.requests: list[dict] = []
        self.block = 0                          # answer this many requests with the rate-threshold page
        self.errors: list[int] = []             # answer the next requests with these statuses
        self.tickers_status = 200
        self.exchange_file_status = 200

    def __call__(self, url, headers, timeout):
        self.requests.append({"url": url, "headers": dict(headers)})
        if not headers.get("User-Agent"):
            return 403, {}, b"<html>Undeclared Automated Tool</html>"
        if self.block:
            self.block -= 1
            return 403, {}, b"<html><title>SEC.gov | Request Rate Threshold Exceeded</title></html>"
        if self.errors:
            return self.errors.pop(0), {}, b""
        if url.endswith("company_tickers_exchange.json"):
            return (200, {}, json.dumps(exchange_file()).encode()) if self.exchange_file_status == 200 else (self.exchange_file_status, {}, b"")
        if url.endswith("company_tickers.json"):
            return (200, {}, json.dumps(plain_file()).encode()) if self.tickers_status == 200 else (self.tickers_status, {}, b"")
        cik = int(url.rsplit("CIK", 1)[1][:10])
        if "/api/xbrl/companyfacts/" in url:
            return (200, {}, json.dumps(self.facts[cik]).encode()) if cik in self.facts else (404, {}, b"<Error><Code>NoSuchKey</Code></Error>")
        if "/submissions/" in url:
            return (200, {}, json.dumps(SUBMISSIONS[cik]).encode()) if cik in SUBMISSIONS else (404, {}, b"")
        return 404, {}, b""

    def kinds(self) -> list[str]:
        return ["tickers" if "company_tickers" in r["url"] else "facts" if "companyfacts" in r["url"] else "submissions" for r in self.requests]


def provider(tmp_path, fake=None, contact=CONTACT, cache=True, **kwargs) -> tuple[SecEdgarProvider, FakeSec]:
    fake = fake or FakeSec()
    return SecEdgarProvider(transport=fake, user_agent=contact, cache_path=(tmp_path / "tickers.json") if cache else None, sleep=lambda s: None, **kwargs), fake


# ------------------------------------------------------------------------------------------------------------------------ the figures, against what the companies reported
def test_apples_statements_are_what_apple_reported_with_the_date_each_was_filed():
    years = by_year(load("AAPL"))
    assert sorted(years) == list(range(2016, 2026))
    p = years[2019]
    assert (p.period_start, p.period_end, p.filing_date, p.reporting_currency, p.accounting_standard) == ("2018-09-30", "2019-09-28", "2019-10-31", "USD", "US GAAP")
    assert p.provider_record_id == "0000320193-19-000119" and p.fiscal_period == "FY"
    # revenue 260,174; net income 55,256; total assets 338,516; operating cash flow 69,391; capital expenditure 10,495 (millions of dollars, Apple's 2019 annual report)
    assert [value(p, c) for c in ("revenue", "net_income", "total_assets", "operating_cash_flow", "capital_expenditure")] == [260174e6, 55256e6, 338516e6, 69391e6, 10495e6]
    assert value(years[2023], "revenue") == 383285e6 and value(years[2023], "net_income") == 96995e6 and years[2023].filing_date == "2023-11-03"
    assert min(p.filing_date for p in years.values()) == "2016-10-26"


def test_the_element_a_figure_came_from_is_kept_and_follows_the_company_when_it_renames_a_line():
    years = by_year(load("AAPL"))
    source = {fy: observed(p)["revenue"]["source_metric"] for fy, p in years.items()}
    assert source[2016] == source[2017] == "us-gaap:SalesRevenueNet"
    assert source[2018] == "us-gaap:Revenues"
    assert source[2019] == source[2025] == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert all(value(p, "revenue") > 2e11 for p in years.values())                      # one continuous series across the renaming


def test_a_retailer_whose_year_ends_in_january_is_numbered_the_way_it_numbers_its_years():
    years = by_year(load("WMT"))
    assert sorted(years) == list(range(2016, 2027))
    assert (years[2020].period_end, years[2020].filing_date) == ("2020-01-31", "2020-03-20") and years[2026].period_end == "2026-01-31"
    assert value(years[2020], "revenue") == 523964e6 and value(years[2020], "net_income") == 14881e6
    assert all(b.period_end > a.period_end for a, b in zip(years.values(), list(years.values())[1:]))


def test_total_liabilities_a_company_does_not_tag_are_assets_less_equity_and_say_so():
    p = by_year(load("WMT"))[2020]
    o = observed(p)
    assert o["total_liabilities"]["flag"] == "derived" and o["total_liabilities"]["source_metric"] == canonical.SEC_LIABILITIES
    assert o["total_liabilities"]["value"] == 154943e6 == o["total_assets"]["value"] - 81552e6                      # total equity including the minority interest, Walmart's balance sheet
    reported = observed(by_year(load("AAPL"))[2019])["total_liabilities"]
    assert reported["flag"] == "ok" and reported["source_metric"] == "us-gaap:Liabilities" and reported["value"] == 248028e6


def test_a_bank_has_the_lines_a_bank_reports_and_nothing_is_invented_for_the_rest():
    p = by_year(load("JPM"))[2019]
    o = observed(p)
    assert o["revenue"]["value"] == 115627e6 and o["net_income"]["value"] == 36431e6 and o["total_assets"]["value"] == 2687379e6
    assert not {"cost_of_revenue", "gross_profit", "operating_income", "inventory", "current_assets", "capital_expenditure"} & set(o)


def test_a_foreign_company_filing_in_ifrs_is_read_in_its_own_currency():
    years = by_year(load("SAP"))
    p = years[2024]
    assert (p.reporting_currency, p.accounting_standard, p.filing_date, p.period_end) == ("EUR", "IFRS", "2025-02-27", "2024-12-31")
    o = observed(p)
    assert o["revenue"]["value"] == 34176e6 and o["revenue"]["source_metric"] == "ifrs-full:Revenue"
    assert years[2016].filing_date == years[2017].filing_date == "2018-02-28"                # the first two years appear as comparatives of the 2017 report: dated by it, late and never early


def test_the_parts_of_a_sum_are_not_reported_as_unmapped_fields():
    for name in CIKS:
        for p in parse_companyfacts(load(name))[0]:
            assert canonical.normalise_values("sec_edgar", p.values)[1] == []


def test_every_statement_comes_from_one_filing_and_the_date_is_that_filings():
    for name in CIKS:
        facts = load(name)
        dates = {}
        for taxonomy in facts["facts"].values():
            for element in taxonomy.values():
                for rows in element["units"].values():
                    for row in rows:
                        dates[row["accn"]] = row["filed"]
        for p in parse_companyfacts(facts)[0]:
            assert dates[p.provider_record_id] == p.filing_date


# ------------------------------------------------------------------------------------------------------------------------ point in time: first filed, and nothing from a later filing
def _with_later_filing(payload: dict, source_accn: str, new_accn: str, filed: str, changes: dict) -> dict:
    """The facts with a copy of one filing added under another accession number and a later date, with some values changed: a later report that restates a year."""
    out = copy.deepcopy(payload)
    for taxonomy, elements in out["facts"].items():
        for element, entry in elements.items():
            for unit, rows in entry["units"].items():
                extra = []
                for row in rows:
                    if row["accn"] == source_accn:
                        clone = dict(row, accn=new_accn, filed=filed)
                        if (taxonomy + ":" + element, row["end"]) in changes:
                            clone["val"] = changes[(taxonomy + ":" + element, row["end"])]
                        extra.append(clone)
                rows.extend(extra)
    return out


def test_a_later_report_that_restates_a_year_changes_neither_its_figures_nor_its_date():
    base = load("AAPL")
    key = ("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "2019-09-28")
    later = _with_later_filing(base, "0000320193-20-000096", "0000999999-20-000001", "2020-12-31", {key: 1.0})
    before, after = by_year(base), by_year(later)
    assert value(after[2019], "revenue") == value(before[2019], "revenue") == 260174e6
    assert after[2019].filing_date == before[2019].filing_date == "2019-10-31" and after[2020].provider_record_id == "0000320193-20-000096"
    assert [(p.period_end, p.values) for p in parse_companyfacts(later)[0]] == [(p.period_end, p.values) for p in parse_companyfacts(base)[0]]


def test_a_figure_a_year_did_not_have_when_first_filed_is_not_borrowed_from_a_later_report():
    base = load("AAPL")
    firsts = by_year(base)
    assert "us-gaap:Goodwill" not in firsts[2020].values                                    # Apple stopped showing goodwill; a later report must not put it back
    later = copy.deepcopy(base)
    later["facts"]["us-gaap"]["Goodwill"] = {"units": {"USD": [{"start": None, "end": "2020-09-26", "val": 5e9, "accn": "0000320193-21-000105", "fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29"}]}}
    later["facts"]["us-gaap"]["Goodwill"]["units"]["USD"][0].pop("start")
    assert "us-gaap:Goodwill" not in by_year(later)[2020].values


def test_reports_other_than_annual_ones_and_periods_other_than_a_year_are_never_read():
    base = load("AAPL")
    noisy = copy.deepcopy(base)
    revenue = noisy["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"]
    revenue += [{"start": "2019-06-30", "end": "2019-09-28", "val": 1.0, "accn": "0000320193-19-000100", "fy": 2019, "fp": "Q3", "form": "10-Q", "filed": "2019-08-01"},
                {"start": "2018-09-30", "end": "2019-03-30", "val": 2.0, "accn": "0000320193-19-000050", "fy": 2019, "fp": "FY", "form": "10-K", "filed": "2019-10-31"},        # six months in an annual report
                {"start": "2018-09-30", "end": "2019-09-28", "val": 3.0, "accn": "0000320193-19-000119", "fy": 2019, "fp": "FY", "form": "8-K", "filed": "2019-10-31"}]
    assert [(p.period_end, p.values) for p in parse_companyfacts(noisy)[0]] == [(p.period_end, p.values) for p in parse_companyfacts(base)[0]]


def test_rows_that_are_not_numbers_are_left_out_and_never_stop_the_reading():
    base = load("AAPL")
    broken = copy.deepcopy(base)
    rows = broken["facts"]["us-gaap"]["Assets"]["units"]["USD"]
    rows += [{"end": "2019-09-28", "val": float("nan"), "accn": "0000320193-19-000119", "form": "10-K", "filed": "2019-10-31"}, {"end": "2019-09-28", "val": True, "accn": "0000320193-19-000119", "form": "10-K", "filed": "2019-10-31"},
             {"end": "not a date", "val": 1, "accn": "x", "form": "10-K", "filed": "2019-10-31"}, "junk", {"end": "2019-09-28", "val": "12", "accn": "0000320193-19-000119", "form": "10-K", "filed": "2019-10-31"},
             {"end": "2019-09-28", "val": 5, "accn": "", "form": "10-K", "filed": "2019-10-31"}]
    assert value(by_year(broken)[2019], "total_assets") == 338516e6
    assert parse_companyfacts({})[0] == [] and parse_companyfacts({"facts": []})[0] == [] and parse_companyfacts({"facts": {"us-gaap": {"Assets": "x"}}})[0] == []


def test_a_balance_sheet_dated_a_few_days_from_the_flows_is_still_the_same_year():
    base = load("AAPL")
    shifted = copy.deepcopy(base)
    for rows in shifted["facts"]["us-gaap"]["Assets"]["units"].values():
        for row in rows:
            if row["end"] == "2019-09-28":
                row["end"] = "2019-09-30"
    assert value(by_year(shifted)[2019], "total_assets") == 338516e6
    assert len(parse_companyfacts(shifted)[0]) == len(parse_companyfacts(base)[0])


def test_a_company_that_reports_in_another_currency_is_not_mixed_with_a_convenience_translation():
    base = load("SAP")
    mixed = copy.deepcopy(base)
    mixed["facts"]["ifrs-full"]["Revenue"]["units"]["USD"] = [dict(r, val=r["val"] * 1.1) for r in base["facts"]["ifrs-full"]["Revenue"]["units"]["EUR"]]
    assert value(by_year(mixed)[2024], "revenue") == 34176e6 and by_year(mixed)[2024].reporting_currency == "EUR"


def test_what_is_stored_as_the_raw_answer_reads_back_to_the_same_statements():
    base = load("AAPL")
    padded = copy.deepcopy(base)
    padded["facts"]["us-gaap"]["SomeElementNobodyReads"] = {"units": {"USD": [{"start": "2018-09-30", "end": "2019-09-28", "val": 1, "accn": "0000320193-19-000119", "form": "10-K", "filed": "2019-10-31"}]}}
    padded["facts"]["us-gaap"]["Assets"]["units"]["USD"].append({"end": "2019-06-29", "val": 9, "accn": "0000320193-19-000100", "fy": 2019, "fp": "Q3", "form": "10-Q", "filed": "2019-08-01"})
    trimmed = trim_companyfacts(padded)
    assert "SomeElementNobodyReads" not in trimmed["facts"]["us-gaap"] and all(r["form"] == "10-K" for r in trimmed["facts"]["us-gaap"]["Assets"]["units"]["USD"])
    assert [(p.period_end, p.values) for p in parse_companyfacts(trimmed)[0]] == [(p.period_end, p.values) for p in parse_companyfacts(padded)[0]]
    assert len(json.dumps(trimmed)) < len(json.dumps(padded))


def test_the_names_of_the_elements_a_mapping_reads_all_have_a_kind_and_nothing_is_read_twice_as_two_things():
    assert set(canonical.SEC_FIELDS.values()) == {"duration", "instant", "cover"}
    assert canonical.SEC_FIELDS["us-gaap:Revenues"] == "duration" and canonical.SEC_FIELDS["us-gaap:Assets"] == "instant" and canonical.SEC_FIELDS["dei:EntityCommonStockSharesOutstanding"] == "cover"
    assert canonical.SEC_FIELDS["us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"] == "duration" and canonical.SEC_FIELDS["us-gaap:CommonStockSharesOutstanding"] == "instant"
    for code, options in canonical.SEC_MAP.items():
        assert code in canonical.METRICS and len({s for s, _ in options}) == len(options)
    outflows = {"capital_expenditure", "acquisitions", "share_repurchases", "dividends_paid", "interest_expense"}
    for code in outflows:
        assert all(t == "abs" for _, t in canonical.SEC_MAP[code])                              # a payment is stored as the positive amount spent


def test_a_derived_figure_is_flagged_derived_and_an_approximation_is_flagged_estimated():
    o = canonical.normalise_values("sec_edgar", {"us-gaap:Assets": 100.0, "us-gaap:StockholdersEquity": 40.0, "us-gaap:MinorityInterest": 5.0, "dei:EntityCommonStockSharesOutstanding": 7.0,
                                                 canonical.SEC_LIABILITIES: 55.0})[0]
    assert o["total_liabilities"]["flag"] == "derived" and o["shares_outstanding"]["flag"] == "estimated"
    exact = canonical.normalise_values("sec_edgar", {"us-gaap:CommonStockSharesOutstanding": 7.0})[0]["shares_outstanding"]
    assert exact["flag"] == "ok" and exact["source_metric"] == "us-gaap:CommonStockSharesOutstanding"


def test_the_sums_are_made_only_when_enough_of_their_lines_are_there():
    values = {"us-gaap:Assets": 100.0, "us-gaap:StockholdersEquity": 40.0, "us-gaap:ShortTermBorrowings": 5.0, "us-gaap:LongTermDebtCurrent": 3.0, "us-gaap:CashAndCashEquivalentsAtCarryingValue": 10.0}
    sec._build_synthetic(values)
    assert values[canonical.SEC_DEBT_CURRENT] == 8.0 and values[canonical.SEC_LIABILITIES] == 60.0 and canonical.SEC_CASH_AND_INVESTMENTS not in values          # cash alone is not cash with investments
    values["us-gaap:ShortTermInvestments"] = 20.0
    sec._build_synthetic(values)
    assert values[canonical.SEC_CASH_AND_INVESTMENTS] == 30.0
    alone = {"us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic": 5.0}
    sec._build_synthetic(alone)
    assert canonical.SEC_PRETAX not in alone


# ------------------------------------------------------------------------------------------------------------------------ the sector of a company
def test_sectors_follow_the_sic_code_and_an_unknown_code_has_none():
    expected = {3571: "Technology", 7372: "Technology", 3674: "Technology", 6021: "Financials", 2911: "Energy", 2080: "Consumer Staples", 3711: "Consumer Discretionary", 5331: "Consumer Staples",
                5211: "Consumer Discretionary", 2834: "Health Care", 5961: "Consumer Discretionary", 4911: "Utilities", 4813: "Communication Services", 6798: "Real Estate", 3531: "Industrials",
                1311: "Energy", 1040: "Materials", 6770: "Financials", "7372": "Technology", " 2834 ": "Health Care"}
    for code, sector in expected.items():
        assert sector_of(code) == sector, code
    for code in (None, "", "abc", 9995, 99999, 0, -5, 4650):
        assert sector_of(code) is None, code


# ------------------------------------------------------------------------------------------------------------------------ the ticker list
def test_both_files_the_sec_publishes_give_the_same_list_and_junk_rows_are_dropped():
    a, b = TickerList.parse(exchange_file()), TickerList.parse(plain_file())
    assert [r[:3] for r in a.rows] == [r[:3] for r in b.rows] and a.find("aapl")[0] == 320193 and a.find("BRK.B")[0] == 1067983 and a.find("brk-a")[1] == "BRK-A"
    junk = {"fields": ["cik", "name", "ticker", "exchange"], "data": [[1, "Ok Inc", "OK", "NYSE"], [True, "Bool", "BL", ""], ["x", "Bad", "BD", ""], [2, "No ticker", None, ""], [3, "Spaces", "A B", ""], "row", [4]]}
    assert [r[1] for r in TickerList.parse(junk).rows] == ["OK"]
    for nonsense in (None, [], "x", {"fields": ["a"], "data": []}, {"fields": ["cik", "name", "ticker"], "data": "x"}, {"0": "x"}):
        assert TickerList.parse(nonsense).rows == []


def test_a_search_finds_a_ticker_exactly_a_name_by_its_words_and_a_company_once_however_many_tickers_it_has():
    tickers = TickerList.parse(exchange_file())
    assert tickers.search("aapl")[0][1][:2] == (320193, "AAPL") and tickers.search("AAPL")[0][0] == 100.0
    assert [r[1] for _, r in tickers.search("alphabet")] == [(1652044, "GOOGL", "Alphabet Inc.", "Nasdaq")[1]]
    assert [r[1] for _, r in tickers.search("berkshire hathaway")] == ["BRK-B"]
    assert tickers.search("1045810")[0][1][1] == "NVDA" and tickers.search("1045810")[0][0] == 120.0
    assert tickers.search("nvidia")[0][1][1] == "NVDA" and tickers.search("nothing like it") == [] and tickers.search("") == []
    assert [r[1] for _, r in tickers.search("inc")][:3] == ["AAPL", "GOOGL", "BRK-B"]                  # words inside names: larger companies (earlier in the list) first
    assert len(tickers.search("inc", limit=3)) == 3


def test_the_list_is_fetched_once_kept_beside_the_database_and_renewed_after_a_week(tmp_path):
    days = [date(2026, 10, 1)]
    prov, fake = provider(tmp_path, today=lambda: days[0])
    assert prov.tickers().find("AAPL")[0] == 320193 and fake.kinds() == ["tickers"]
    again, fake2 = provider(tmp_path, today=lambda: days[0])
    assert again.tickers().find("MSFT")[0] == 789019 and fake2.requests == []                    # read from the saved copy
    days[0] = date(2026, 10, 9)
    renewed, fake3 = provider(tmp_path, today=lambda: days[0])
    renewed.tickers()
    assert fake3.kinds() == ["tickers"] and json.loads((tmp_path / "tickers.json").read_text())["fetched"] == "2026-10-09"


def test_a_list_that_cannot_be_renewed_is_used_stale_and_one_that_was_never_there_says_what_to_do(tmp_path):
    prov, fake = provider(tmp_path, today=lambda: date(2026, 10, 1))
    prov.tickers()
    fake2 = FakeSec()
    fake2.exchange_file_status, fake2.tickers_status = 500, 500
    stale, _ = provider(tmp_path, fake=fake2, today=lambda: date(2026, 11, 20))
    assert stale.tickers().find("AAPL") is not None and "could not be renewed" in stale.notes[0] and "2026-10-01" in stale.notes[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    cold, _ = provider(empty, fake=fake2)
    with pytest.raises(ProviderError) as err:
        cold.tickers()
    assert "cik:320193" in str(err.value) and CONTACT not in str(err.value)


def test_the_second_file_is_tried_when_the_first_is_missing(tmp_path):
    fake = FakeSec()
    fake.exchange_file_status = 404
    prov, _ = provider(tmp_path, fake=fake)
    assert prov.tickers().find("AAPL") is not None and fake.kinds() == ["tickers", "tickers"]


def test_a_corrupt_saved_list_is_ignored_and_fetched_again(tmp_path):
    (tmp_path / "tickers.json").write_text("{not json")
    prov, fake = provider(tmp_path)
    assert prov.tickers().find("AAPL") is not None and fake.kinds() == ["tickers"]
    (tmp_path / "tickers.json").write_text(json.dumps({"fetched": "2026-10-23", "rows": [["x"], [1, "ok", "N", "E", 5], 7]}))
    other, fake2 = provider(tmp_path)
    assert other.tickers().find("AAPL") is not None and fake2.kinds() == ["tickers"]


# ------------------------------------------------------------------------------------------------------------------------ a company by name, ticker or CIK
def test_a_symbol_is_one_spelling_whatever_way_it_was_typed(tmp_path):
    prov, _ = provider(tmp_path)
    for text in ("AAPL", "aapl", " us:AAPL ", "US:aapl"):
        assert prov.normalise_symbol(text) == "us:AAPL"
    assert prov.normalise_symbol("brk.b") == "us:BRK.B" and prov.normalise_symbol("BRK-B") == "us:BRK-B"
    for text in ("cik:320193", "CIK:0000320193", "320193", "us:CIK0000320193", "CIK0000320193"):
        assert prov.normalise_symbol(text) == "us:CIK0000320193"
    assert prov.offline_profile("us:AAPL").ticker == "AAPL" and prov.offline_profile("us:AAPL").exchange == "US" and prov.offline_profile("us:AAPL").provider == "sec_edgar"
    for bad in ("jp:7203", "uk:BP", "", "a b", "cik:", "cik:0", "cik:abc", "::"):
        with pytest.raises(ProviderError):
            prov.normalise_symbol(bad)
    with pytest.raises(ProviderError):
        prov.normalise_symbol("AAPL", "jp")


def test_a_cik_is_named_by_its_ticker_once_the_list_is_at_hand_and_asking_for_it_is_never_needed(tmp_path):
    prov, fake = provider(tmp_path)
    assert prov.normalise_symbol("cik:320193") == "us:CIK0000320193" and fake.requests == []
    prov.tickers()
    assert prov.normalise_symbol("cik:320193") == "us:AAPL" and prov.normalise_symbol("cik:5") == "us:CIK0000000005"
    later, fake2 = provider(tmp_path)                                                      # another run: the saved list is read, the SEC is not asked
    assert later.normalise_symbol("cik:320193") == "us:AAPL" and later.offline_profile("cik:789019").ticker == "MSFT" and fake2.requests == []


def test_a_search_costs_nothing_and_says_what_each_hit_is(tmp_path):
    prov, fake = provider(tmp_path)
    detail = prov.search_detail("Alphabet")
    assert detail["credits"] == 0 and detail["kind"] == "name" and [p.provider_symbol for p in detail["profiles"]] == ["us:GOOGL"]
    hit = detail["profiles"][0]
    assert hit.extra["also_listed"] == ["us:GOOG"] and hit.extra["cik"] == 1652044 and hit.extra["has_financials"] and hit.country == "US" and hit.exchange == "US"
    assert prov.search_detail("MSFT")["kind"] == "symbol" and prov.search_detail("MSFT")["profiles"][0].name == "MICROSOFT CORP"
    nothing = prov.search_detail("zzzzzz")
    assert nothing["profiles"] == [] and "no company" in nothing["message"]
    assert fake.kinds() == ["tickers"]                                                           # the list once, and no request for any search
    assert prov.cost("search") == 0 and prov.cost("financials") == 1 and prov.cost("profile") == 1
    with pytest.raises(ProviderError):
        prov.search_plan("MSFT", "isin")


# ------------------------------------------------------------------------------------------------------------------------ talking to the SEC
def test_without_a_contact_nothing_is_asked_and_the_reason_says_what_to_set(tmp_path):
    for contact in (None, "", "   ", "short", "x" * 300, "bad\x07contact@example.org"):
        prov, fake = provider(tmp_path, contact=contact)
        ok, why = prov.available()
        assert not ok and "SEC_EDGAR_USER_AGENT" in why
        with pytest.raises(ProviderError):
            prov.get_financials("cik:320193")
        assert fake.requests == []
    assert provider(tmp_path)[0].available() == (True, "")


def test_every_request_identifies_the_program_asks_for_gzip_and_is_paced(tmp_path):
    ticks = [0.0]
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        ticks[0] += seconds
    fake = FakeSec()
    prov = SecEdgarProvider(transport=fake, user_agent=CONTACT, sleep=sleep, clock=lambda: ticks[0])
    for _ in range(3):
        prov.get_financials("cik:320193")
    assert len(fake.requests) == 3 and all(r["headers"]["User-Agent"] == CONTACT and r["headers"]["Accept-Encoding"] == "gzip" for r in fake.requests)
    assert len(sleeps) == 2 and all(abs(s - sec.MIN_INTERVAL) < 1e-9 for s in sleeps)                # the second and third wait: never faster than the interval


def test_the_contact_never_appears_in_a_message_a_log_or_the_description(tmp_path):
    fake = FakeSec()
    fake.errors = [404, 429, 500, 500, 500]
    prov, _ = provider(tmp_path, fake=fake)
    messages = []
    for _ in range(3):
        with pytest.raises(ProviderError) as err:
            prov.get_financials("cik:320193")
        messages.append(str(err.value))
    fake.block = 1
    with pytest.raises(RateLimited) as err:
        prov.get_financials("cik:320193")
    messages.append(str(err.value))
    messages.append(json.dumps(prov.describe()))
    assert not any("example.org" in m or "Test Person" in m for m in messages)
    assert prov.describe()["contact"] is True and prov.describe()["contact_has_address"] is True and prov.describe()["free"] is True


def test_a_company_the_sec_has_no_facts_for_is_not_found(tmp_path):
    prov, _ = provider(tmp_path)
    with pytest.raises(NotFound) as err:
        prov.get_financials("cik:42")
    assert "no XBRL financial data" in str(err.value)
    with pytest.raises(NotFound):
        prov.get_financials("us:NOSUCH")                                                         # a ticker the list does not have
    assert "ticker list has no company" in str(pytest.raises(NotFound, prov.get_financials, "us:NOSUCH").value)


def test_a_refusal_for_asking_too_often_is_told_apart_from_a_refusal_to_serve_an_unknown_client(tmp_path):
    fake = FakeSec()
    prov, _ = provider(tmp_path, fake=fake)
    fake.block = 1
    with pytest.raises(RateLimited) as err:
        prov.get_financials("cik:320193")
    assert err.value.dimension == "burst" and "ten a second" in str(err.value) and len(fake.requests) == 1                # not asked again
    fake.errors = [403]
    with pytest.raises(ProviderError) as err2:
        prov.get_financials("cik:320193")
    assert not isinstance(err2.value, RateLimited) and "SEC_EDGAR_USER_AGENT" in str(err2.value)
    fake.errors = [429]
    with pytest.raises(RateLimited):
        prov.get_financials("cik:320193")


def test_a_server_error_is_retried_twice_and_then_reported(tmp_path):
    fake = FakeSec()
    prov, _ = provider(tmp_path, fake=fake)
    fake.errors = [503, 502]
    assert len(prov.get_financials("cik:320193").periods) == 10 and len(fake.requests) == 3
    fake.requests.clear()
    fake.errors = [500, 500, 500, 500]
    with pytest.raises(ProviderError) as err:
        prov.get_financials("cik:320193")
    assert "HTTP 500" in str(err.value) and len(fake.requests) == 3


def test_a_ticker_is_resolved_through_the_list_and_the_whole_history_comes_in_one_request(tmp_path):
    prov, fake = provider(tmp_path)
    result = prov.get_financials("us:AAPL")
    assert fake.kinds() == ["tickers", "facts"] and result.requests == 1 and result.cost == 1 and "CIK0000320193" in fake.requests[1]["url"]
    assert [p.fiscal_year for p in result.periods] == list(range(2016, 2026)) and result.exhausted and result.profile.name == "Apple Inc." and result.profile.currency == "USD"
    fake.requests.clear()
    prov.get_financials("AAPL")
    assert fake.kinds() == ["facts"]                                                              # the list is kept


def test_the_years_asked_for_are_the_years_returned_and_the_answer_says_whether_there_is_more(tmp_path):
    prov, _ = provider(tmp_path)
    some = prov.get_financials("cik:320193", start_year=2020, end_year=2022)
    assert [p.fiscal_year for p in some.periods] == [2020, 2021, 2022] and not some.exhausted
    assert (some.request["fiscal_year_from"], some.request["fiscal_year_to"]) == (2020, 2022) and coverage.asked_years(some.request) == (2020, 2022)
    early = prov.get_financials("cik:320193", start_year=2005)
    assert len(early.periods) == 10 and early.exhausted                                          # nothing older exists: the series starts after the year asked for
    everything = prov.get_financials("cik:320193")
    assert coverage.asked_years(everything.request) == (None, None) and everything.exhausted


def test_the_stored_answer_says_what_it_held_and_is_small(tmp_path):
    prov, _ = provider(tmp_path)
    result = prov.get_financials("cik:320193")
    summary = coverage.summarise_answer(result.raw)
    assert summary["entries"] == 10 and summary["annual"] == 10 and (summary["first_year"], summary["last_year"]) == (2016, 2025) and summary["labels"] == {"FY": 10}
    assert len(json.dumps(result.raw)) < 400_000 and result.raw["entityName"] == "Apple Inc."
    assert [p.period_end for p in parse_companyfacts(result.raw)[0]] == [p.period_end for p in result.periods]


def test_quarters_are_not_read_yet_and_say_so(tmp_path):
    prov, fake = provider(tmp_path)
    with pytest.raises(PlanError):
        prov.get_financials("cik:320193", period="quarter")
    assert fake.requests == [] and prov.periods_available == ("annual",)


def test_a_company_lookup_gives_the_name_sector_and_industry_from_the_filers_own_record(tmp_path):
    prov, fake = provider(tmp_path)
    apple = prov.get_company("cik:320193")
    assert (apple.name, apple.sector, apple.industry, apple.ticker) == ("Apple Inc.", "Technology", "Electronic Computers", "CIK0000320193") and fake.kinds() == ["submissions"]
    assert apple.extra["exchanges"] == ["Nasdaq"] and apple.extra["fiscal_year_end"] == "0926" and apple.country == "US"
    walmart = prov.get_company("us:WMT")
    assert (walmart.sector, walmart.industry, walmart.lei, walmart.ticker) == ("Consumer Staples", "Retail-Variety Stores", "5493000K4X9P0F4JXV31", "WMT")             # SIC words in capitals are made readable
    assert prov.get_company("us:JPM").sector == "Financials"


def test_the_test_button_costs_one_small_request(tmp_path):
    prov, fake = provider(tmp_path)
    assert prov.test()["ok"] is True and fake.kinds() == ["submissions"]


def test_a_redirect_is_not_followed_and_an_oversized_or_broken_answer_is_refused():
    class Body:
        def __init__(self, data, encoding=""):
            self._data, self.headers = data, {"Content-Encoding": encoding}

        def read(self, n=-1):
            return self._data[:n] if n >= 0 else self._data
    import zlib
    packed = zlib.compress(b"{}" * 10, wbits=31)
    assert sec._read_body(Body(packed, "gzip")) == b"{}" * 10 and sec._read_body(Body(b"plain")) == b"plain"
    with pytest.raises(ProviderError):
        sec._read_body(Body(b"not gzip at all", "gzip"))
    bomb = zlib.compress(b"0" * (sec.MAX_BYTES + 10), 9, wbits=31) if sec.MAX_BYTES < 5e7 else b""
    if bomb:
        with pytest.raises(ProviderError):
            sec._read_body(Body(bomb, "gzip"))
    assert sec._NoRedirect().redirect_request(None, None, 301, "", {}, "https://elsewhere.example") is None


def test_the_provider_is_one_of_the_names_and_is_made_by_name(tmp_path):
    assert "sec_edgar" in PROVIDER_NAMES
    prov = make_provider("sec_edgar", user_agent=CONTACT)
    assert isinstance(prov, SecEdgarProvider) and prov.name == "sec_edgar" and not prov.keyed and prov.meters_itself and prov.as_reported and prov.has_filing_dates


# ------------------------------------------------------------------------------------------------------------------------ in the application
def service(tmp_path, fake=None, contact=CONTACT, **kwargs) -> tuple[FundamentalsService, SecEdgarProvider, FakeSec]:
    svc = FundamentalsService(tmp_path / "svc", db_path=tmp_path / "svc" / "f.db")
    fake = fake or FakeSec()
    prov = SecEdgarProvider(pool=svc.pool("sec_edgar"), transport=fake, user_agent=contact, cache_path=tmp_path / "svc" / "tickers.json", sleep=lambda s: None, **kwargs)
    svc.register("sec_edgar", prov)
    return svc, prov, fake


def test_a_sync_stores_every_year_and_costs_two_requests_for_a_new_company_and_nothing_to_repeat(tmp_path):
    svc, prov, fake = service(tmp_path)
    plan = svc.plan("sec_edgar", svc.add_companies("sec_edgar", ["AAPL"]))
    assert plan.cost == 2 and plan.unit == "requests" and [t.reason for t in plan.tasks] == ["missing"] and plan.tasks[0].profile
    report = svc.sync("sec_edgar", symbols=["AAPL"])
    assert report["fetched"] == 1 and report["periods"] == 10 and report["cost"] == 2 and report["failed"] == [] and report["unmapped"] == [] and report["stopped"] == ""
    assert fake.kinds() == ["tickers", "submissions", "facts"] and svc.usage.requests_today("sec_edgar") == 3
    sid = svc.add_companies("sec_edgar", ["AAPL"])[0]
    info = svc.store.get_security(sid)
    assert (info["company_name"], info["sector"], info["industry"], info["exchange"], info["country"], info["currency"], info["source"]) == ("Apple Inc.", "Technology", "Electronic Computers", "US", "US", "USD", "sec_edgar")
    held = svc.store.held_periods(sid, "sec_edgar")
    assert [r["fiscal_year"] for r in held] == list(range(2016, 2026)) and held[3]["filing_date"] == "2019-10-31" and all(r["vintage"] == 0 for r in held)
    again = svc.sync("sec_edgar", symbols=["AAPL"])
    assert again["fetched"] == 0 and again["fresh"] == 1 and len(fake.requests) == 3                  # fresh: nothing asked
    assert svc.store.period_summary(sid)["n"] == 10


def test_asking_again_changes_nothing_that_is_stored(tmp_path):
    svc, prov, fake = service(tmp_path)
    svc.sync("sec_edgar", symbols=["WMT"])
    sid = svc.add_companies("sec_edgar", ["WMT"])[0]
    with svc.store.db.transaction() as conn:
        before = conn.execute("SELECT COUNT(*), SUM(value) FROM fundamental_observations").fetchone()
    task = svc.plan("sec_edgar", [sid], start_year=None).tasks
    assert task == []
    result = prov.get_financials("us:WMT")
    counts = svc.store.store_financials(sid, "sec_edgar", result.periods)
    assert counts["periods"] == 0 and counts["vintages"] == 0 and counts["unchanged"] == 11
    with svc.store.db.transaction() as conn:
        assert tuple(conn.execute("SELECT COUNT(*), SUM(value) FROM fundamental_observations").fetchone()) == tuple(before)


def test_the_years_view_shows_what_is_held_and_the_sec_wording_replaces_the_credit_wording(tmp_path):
    svc, _, _ = service(tmp_path)
    svc.sync("sec_edgar", symbols=["AAPL"])
    sid = svc.add_companies("sec_edgar", ["AAPL"])[0]
    cov = svc.coverage(sid, "sec_edgar", date(2026, 10, 23))
    states = {c["year"]: c["state"] for c in cov["annual"]["cells"]}
    assert all(states[y] == "held" for y in range(2016, 2026)) and states[2015] == "empty" and states[2026] == "notyet"
    assert cov["unit"] == "requests" and cov["request_cost"] == 1 and cov["can_fetch"] and cov["provider_word"] == "SEC EDGAR"
    assert "whole history in one free request" in cov["reading"] and "Credits pay" not in cov["reading"] and "first XBRL filing" in cov["reading"]
    assert not cov["quarterly"]["supported"] and "10-Q" in cov["quarterly"]["note"]


def test_the_years_not_yet_held_are_picked_and_fetched_without_touching_the_ones_that_are(tmp_path):
    svc, _, fake = service(tmp_path)
    svc.sync("sec_edgar", symbols=["AAPL"], start_year=2021)
    sid = svc.add_companies("sec_edgar", ["AAPL"])[0]
    assert [r["fiscal_year"] for r in svc.store.held_periods(sid, "sec_edgar")] == [2021, 2022, 2023, 2024, 2025]
    plan = svc.plan_years(sid, [2016, 2017, 2018, 2022], "sec_edgar", today=date(2026, 10, 23))
    assert plan["to_fetch"] == [2016, 2017, 2018] and plan["already_held"] == [2022] and plan["cost"] == 1
    report = svc.fetch_years(sid, [2016, 2017, 2018, 2022], budget=1, provider="sec_edgar", today=date(2026, 10, 23))
    assert [a["year"] for a in report["added"]] == [2016, 2017, 2018] and report["cost"] == 1 and report["unit"] == "requests"
    assert [r["fiscal_year"] for r in svc.store.held_periods(sid, "sec_edgar")] == [2016, 2017, 2018, 2021, 2022, 2023, 2024, 2025]            # the years picked, and none that were not
    assert svc.store.period_summary(sid, "sec_edgar")["n"] == 8


def test_a_run_stops_at_the_first_refusal_and_leaves_the_rest_for_later(tmp_path):
    svc, _, fake = service(tmp_path)
    fake.block = 1
    report = svc.sync("sec_edgar", symbols=["AAPL", "WMT", "JPM"])
    assert report["fetched"] == 0 and report["failed"] == [] and "too many requests" in report["stopped"] and "remain" in report["stopped"]
    assert len(fake.requests) == 1                                                                # one refusal, no more asking


def test_a_company_without_facts_is_remembered_as_not_found_and_not_asked_again_for_two_months(tmp_path):
    svc, _, fake = service(tmp_path)
    report = svc.sync("sec_edgar", symbols=["cik:42"])
    assert report["not_found"] == ["us:CIK0000000042"] and report["fetched"] == 0
    sent = len(fake.requests)
    assert svc.sync("sec_edgar", symbols=["cik:42"])["fresh"] == 1 and len(fake.requests) == sent


def test_a_search_in_the_application_is_free_and_says_whether_the_company_is_already_held(tmp_path):
    svc, _, fake = service(tmp_path)
    svc.sync("sec_edgar", symbols=["AAPL"])
    found = svc.search_companies("sec_edgar", "apple")
    assert found["credits"] == 0 and found["count"] == 1 and found["results"][0]["symbol"] == "us:AAPL" and found["results"][0]["in_database"] and found["results"][0]["periods"] == 10
    assert found["cost_to_fetch"] == 2 and found["unit"] == "requests"
    assert svc.search_companies("sec_edgar", "microsoft")["results"][0]["in_database"] is False
    with pytest.raises(MissingKey):
        service(tmp_path / "other", contact=None)[0].search_companies("sec_edgar", "apple")


def test_the_data_status_lists_the_provider_with_no_key_and_never_the_contact(tmp_path):
    svc, _, _ = service(tmp_path)
    svc.sync("sec_edgar", symbols=["AAPL"])
    status = svc.data_status()
    card = next(p for p in status["providers"] if p["name"] == "sec_edgar")
    assert card["available"] and card["free"] and card["unit"] == "requests" and card["used_today"] == 3 and card["daily_limit"] is None and card["periods"] == ["annual"]
    assert "example.org" not in json.dumps(status) and "Test Person" not in json.dumps(status)
    assert [p["name"] for p in status["providers"]] == ["openfilings", "sec_edgar", "simulated"]


def test_a_company_added_by_two_providers_is_one_company_and_keeps_the_provider_that_added_it_first(tmp_path):
    svc, _, _ = service(tmp_path)
    ids = svc.add_companies("openfilings", ["us:AAPL"])
    again = svc.add_companies("sec_edgar", ["AAPL"])
    assert ids == again and svc.store.get_security(ids[0])["source"] == "openfilings"
    assert svc._lookup("sec_edgar", "us:AAPL") == ids[0] and svc._lookup("openfilings", "us:AAPL") == ids[0]


# ------------------------------------------------------------------------------------------------------------------------ point in time and quality
def _fill(svc, name="AAPL"):
    svc.sync("sec_edgar", symbols=[name])
    return svc.add_companies("sec_edgar", [name])[0]


def test_a_year_is_usable_the_day_after_the_filing_that_first_showed_it(tmp_path):
    svc, _, _ = service(tmp_path)
    sid = _fill(svc)
    tables = PeriodTables(svc.store, [sid])
    fy = tables.fy.set_index("fiscal_year")
    assert fy.loc[2019, "available_date"] == pd.Timestamp("2019-11-01") and not fy.loc[2019, "pit_estimated"] and tables.estimated_share() == 0.0
    dates = pd.bdate_range("2019-10-25", "2019-11-08")
    panel = tables.panel("revenue", dates, "fy", [sid])
    assert panel.loc["2019-10-31", sid] == 265595e6 and panel.loc["2019-11-01", sid] == 260174e6          # the 2018 figure until the day after the 2019 report was filed


def test_the_sec_leads_a_period_both_providers_have_and_the_other_fills_only_what_is_missing(tmp_path):
    svc, _, _ = service(tmp_path)
    sid = _fill(svc)
    other = [PeriodStatement("FY", "2019-09-28", {"revenue": 1.0, "rnd": 2.0, "cost_of_goods_sold": 3.0, "capex": 4.0}, fiscal_year=2019, filing_date="2019-10-31", reporting_currency="USD")]
    svc.store.store_financials(sid, "openfilings", other)
    assert PitSettings().provider_priority == ("sec_edgar", "openfilings", "simulated") and PitSettings().rank("sec_edgar") == 0
    row = PeriodTables(svc.store, [sid]).fy.set_index("fiscal_year").loc[2019]
    assert row["revenue"] == 260174e6 and row["research_and_development"] == value(by_year(load("AAPL"))[2019], "research_and_development")


def test_a_share_count_in_thousands_against_one_in_units_is_flagged_and_left_out_of_research(tmp_path):
    svc, _, _ = service(tmp_path)
    sid = _fill(svc)
    with svc.store.db.transaction() as conn:
        pid = conn.execute("SELECT id FROM financial_periods WHERE security_id = ? AND fiscal_year = 2019", (sid,)).fetchone()[0]
        conn.execute("UPDATE fundamental_observations SET value = value / 1000 WHERE financial_period_id = ? AND metric_code = 'weighted_avg_shares_diluted'", (pid,))
    counts = quality.check_security(svc.store, sid)
    assert counts.get("suspicious_jump") == 1
    obs = svc.store.observations([sid], ["weighted_avg_shares_diluted"])
    assert obs[obs["period_id"] == pid]["quality_flag"].iloc[0] == "suspicious_jump" and (obs["quality_flag"] == "suspicious_jump").sum() == 1


def test_two_providers_are_compared_on_the_figures_each_filed_first_and_not_on_approximations(tmp_path):
    svc, _, _ = service(tmp_path)
    sid = _fill(svc)
    exact = [PeriodStatement("FY", "2019-09-28", {"revenue": 260174e6 * 1.2}, fiscal_year=2019, filing_date="2019-10-31", reporting_currency="USD")]
    svc.store.store_financials(sid, "openfilings", exact)
    quality.check_security(svc.store, sid)
    obs = svc.store.observations([sid], ["revenue"])
    periods = svc.store.periods([sid])
    pid = {p: periods[(periods["provider"] == p) & (periods["fiscal_year"] == 2019)]["period_id"].iloc[0] for p in ("sec_edgar", "openfilings")}
    flags = {(r.period_id, r.metric_code): r.quality_flag for r in obs.itertuples()}
    assert flags[(pid["sec_edgar"], "revenue")] == flags[(pid["openfilings"], "revenue")] == "provider_disagreement"
    shares = [PeriodStatement("FY", "2019-09-28", {"shares_basic": 4.0e9}, fiscal_year=2019, filing_date="2019-10-31", reporting_currency="USD")]
    svc.store.store_financials(sid, "openfilings", shares)
    quality.check_security(svc.store, sid)
    obs = svc.store.observations([sid], ["shares_outstanding"])
    approx = obs[obs["period_id"] == pid["openfilings"]]
    assert approx["quality_flag"].iloc[0] == "estimated"                                              # an approximation is never held against the exact figure


def test_the_cover_page_share_count_is_flagged_estimated_after_the_quality_pass(tmp_path):
    svc, _, _ = service(tmp_path)
    sid = _fill(svc, "WMT")
    quality.check_security(svc.store, sid)
    obs = svc.store.observations([sid], ["shares_outstanding"])
    kinds = {r.source_metric: r.quality_flag for r in obs.itertuples()}
    assert kinds.get("dei:EntityCommonStockSharesOutstanding", "estimated") == "estimated" and all(k in ("ok", "estimated") for k in kinds.values())
    assert kinds.get("us-gaap:CommonStockSharesOutstanding", "ok") == "ok"


def test_a_value_the_database_holds_is_the_value_the_filing_reported(tmp_path):
    svc, _, _ = service(tmp_path)
    sid = _fill(svc, "JPM")
    obs = svc.store.observations([sid], ["revenue", "net_income"])
    periods = svc.store.periods([sid])
    merged = obs.merge(periods[["period_id", "fiscal_year"]], on="period_id")
    got = {(r.fiscal_year, r.metric_code): r.value for r in merged.itertuples()}
    assert got[(2019, "revenue")] == 115627e6 and got[(2019, "net_income")] == 36431e6 and got[(2025, "revenue")] == 182447e6


def test_the_profile_the_application_keeps_is_a_company_profile(tmp_path):
    prov, _ = provider(tmp_path)
    profile = prov.offline_profile("us:AAPL")
    assert isinstance(profile, CompanyProfile) and profile.provider_symbol == "us:AAPL" and profile.market == "US"
    with pytest.raises(FundamentalsError):
        FundamentalsService(tmp_path / "x").provider("nope")
    assert FundamentalsStore(tmp_path / "y.db").get_meta("anything") is None


def test_a_split_or_a_year_near_break_even_is_not_taken_for_a_unit_error_but_a_thousandfold_error_is(tmp_path):
    svc, _, _ = service(tmp_path)
    sid = _fill(svc, "AAPL")
    with svc.store.db.transaction() as conn:
        rows = conn.execute("SELECT o.id, p.fiscal_year FROM fundamental_observations o JOIN financial_periods p ON p.id = o.financial_period_id WHERE o.metric_code = 'shares_outstanding' ORDER BY p.period_end").fetchall()
        for oid, year in rows:
            if year >= 2022:
                conn.execute("UPDATE fundamental_observations SET value = value * 10 WHERE id = ?", (oid,))                  # a ten for one split
        net = conn.execute("SELECT o.id FROM fundamental_observations o JOIN financial_periods p ON p.id = o.financial_period_id WHERE o.metric_code = 'net_income' AND p.fiscal_year = 2020").fetchone()[0]
        conn.execute("UPDATE fundamental_observations SET value = value / 20 WHERE id = ?", (net,))                          # a year that fell to a twentieth of its neighbours
    assert "suspicious_jump" not in quality.check_security(svc.store, sid)
    with svc.store.db.transaction() as conn:
        conn.execute("UPDATE fundamental_observations SET value = value / 1000 WHERE id = ?", (net,))                        # in thousands against units
    assert quality.check_security(svc.store, sid) == {"suspicious_jump": 1}


# ------------------------------------------------------------------------------------------------------------------------ the routes
@pytest.fixture()
def sec_app(tmp_path, monkeypatch):
    """An app whose real database can be filled from the SEC (a stand-in for its servers), with no OpenFilings key and no contact set in the environment."""
    for name in [n for n in list(os.environ) if n.startswith("OPENFILINGS_") or n == "SEC_EDGAR_USER_AGENT"]:
        monkeypatch.delenv(name, raising=False)
    app = make_app(tmp_path)
    svc = app.fundamentals.bench("real").service
    fake = FakeSec()
    svc.register("sec_edgar", SecEdgarProvider(pool=svc.pool("sec_edgar"), transport=fake, user_agent=CONTACT, cache_path=tmp_path / "tickers.json", sleep=lambda s: None))
    return app, fake


def test_the_page_is_told_what_each_provider_can_do_and_the_lookup_it_always_had_follows_the_first_that_works(sec_app, tmp_path, monkeypatch):
    app, _ = sec_app
    out = get(app, "/api/fund/companies", q="")
    sec = out["lookups"]["sec_edgar"]
    assert sec["available"] and sec["free"] and sec["unit"] == "requests" and sec["fetch_cost"] == 2 and sec["cost"] == 0 and sec["word"] == "SEC EDGAR" and sec["daily_limit"] is None
    assert out["lookups"]["openfilings"]["available"] is False and "OPENFILINGS_API_KEY" in out["lookups"]["openfilings"]["reason"]
    assert out["lookup"]["available"] is True and out["lookup"]["unit"] == "requests" and out["lookup"]["fetch_cost"] == 2 and set(out["lookup"]) == {
        "available", "reason", "cost", "fetch_cost", "unit", "remaining_today", "daily_limit", "monthly_limit", "remaining_month"}
    overview = get(app, "/api/fund/overview")
    assert overview["setup"]["available"] is True and overview["lookups"]["sec_edgar"]["available"] is True
    bare = make_app(tmp_path / "bare")                                                         # nothing set up: the lookup is as it always was
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    nothing = get(bare, "/api/fund/companies", q="")
    assert nothing["lookup"]["available"] is False and "OPENFILINGS_API_KEY" in nothing["lookup"]["reason"] and nothing["lookups"]["sec_edgar"]["available"] is False
    assert "SEC_EDGAR_USER_AGENT" in nothing["lookups"]["sec_edgar"]["reason"]


def test_a_company_is_searched_added_and_fetched_from_the_sec_through_the_routes_and_costs_nothing_but_requests(sec_app):
    app, fake = sec_app
    found = post(app, "/api/fund/companies/search", q="apple", provider="sec_edgar")
    assert (found["provider"], found["provider_word"], found["credits"], found["unit"], found["cost_to_fetch"], found["count"]) == ("sec_edgar", "SEC EDGAR", 0, "requests", 2, 1)
    assert found["results"][0]["symbol"] == "us:AAPL" and found["results"][0]["in_database"] is False and found["results"][0]["has_financials"] is True
    again = post(app, "/api/fund/companies/search", q="apple", provider="sec_edgar")
    assert again["cached"] is True
    added = post(app, "/api/fund/companies/add", symbols=["AAPL"], provider="sec_edgar")
    assert added["added"] == 1 and added["companies"][0]["ticker"] == "AAPL"
    with pytest.raises(ApiError, match="requests"):
        post(app, "/api/fund/sync", provider="sec_edgar")                                       # a run without a budget is refused, in the provider's own unit
    report = run(app, "/api/fund/sync", provider="sec_edgar", budget=2)
    assert (report["fetched"], report["periods"], report["cost"], report["unit"]) == (1, 10, 2, "requests") and report["provider"] == "sec_edgar"
    assert post(app, "/api/fund/companies/search", q="apple", provider="sec_edgar", refresh=True)["results"][0]["periods"] == 10
    sid = added["security_ids"][0]
    cov = get(app, "/api/fund/coverage", id=str(sid), provider="sec_edgar")
    assert cov["annual"]["counts"]["held"] == 10 and cov["provider"] == "sec_edgar" and cov["providers"] == [{"name": "sec_edgar", "word": "SEC EDGAR", "periods": 10}]
    left = post(app, "/api/fund/coverage/plan", security_id=sid, years=[2010, 2011], provider="sec_edgar")
    assert left["to_fetch"] == [] and left["left_out_empty"] == [2010, 2011] and left["cost"] == 0                   # the whole history was asked for and the SEC sent nothing that early
    plan = post(app, "/api/fund/coverage/plan", security_id=sid, years=[2010, 2011], provider="sec_edgar", ask_again=True)
    assert plan["to_fetch"] == [2010, 2011] and plan["cost"] == 1 and plan["unit"] == "requests" and plan["provider"] == "sec_edgar"
    assert "contact" not in json.dumps(found).lower() and "example.org" not in json.dumps([found, cov, plan, report])
    with pytest.raises(ApiError, match="budget"):
        post(app, "/api/fund/coverage/fetch", security_id=sid, years=[2010], provider="sec_edgar")
    done = run(app, "/api/fund/coverage/fetch", security_id=sid, years=[2010, 2011], budget=1, provider="sec_edgar", ask_again=True)
    assert done["added"] == [] and done["nothing_for"] == [2010, 2011] and done["cost"] == 1 and "held no periods" in done["text"]                  # the SEC has nothing from before 2016 for this fixture
    assert fake.kinds().count("facts") == 2


def test_a_provider_that_is_not_one_of_the_two_and_a_search_the_sec_cannot_do_are_refused(sec_app):
    app, _ = sec_app
    for route, body in (("/api/fund/companies/search", {"q": "apple"}), ("/api/fund/companies/add", {"symbols": ["AAPL"]}), ("/api/fund/sync/plan", {}), ("/api/fund/provider/test", {})):
        with pytest.raises(ApiError, match="from OpenFilings or SEC EDGAR"):
            post(app, route, provider="fmp", **body)
    with pytest.raises(ApiError, match="from OpenFilings or SEC EDGAR"):
        post(app, "/api/fund/sync", provider="simulated", budget=5)
    with pytest.raises(ApiError, match="kind must be one of auto, name, symbol"):
        post(app, "/api/fund/companies/search", q="US0378331005", kind="isin", provider="sec_edgar")
    with pytest.raises(ApiError, match="provider with keys"):
        post(app, "/api/fund/keys/use", provider="sec_edgar", slot="key 2")
    assert "US SEC" in failed(app, "/api/fund/sync", provider="sec_edgar", budget=2, symbols=["jp:7203"])                # a market the SEC does not cover is refused, not guessed


def test_the_connection_test_of_the_sec_is_one_free_request_and_the_status_lists_it_without_the_contact(sec_app):
    app, fake = sec_app
    out = post(app, "/api/fund/provider/test", provider="sec_edgar")
    assert out["ok"] is True and out["available"] is True and fake.kinds() == ["submissions"]
    status = get(app, "/api/fund/status")
    card = next(p for p in status["providers"] if p["name"] == "sec_edgar")
    assert card["used_today"] == 1 and card["requests_today"] == 1 and card["recent"][0]["endpoint"] == "test" and card["free"] is True
    assert "example.org" not in json.dumps([status, out]) and CONTACT not in json.dumps(status)
    plan = post(app, "/api/fund/sync/plan", provider="sec_edgar")
    assert plan["plan"]["unit"] == "requests" and plan["affordability"]["remaining_today"] is None


def test_without_a_contact_the_sec_says_what_to_set_and_nothing_is_asked(tmp_path, monkeypatch):
    for name in [n for n in list(os.environ) if n.startswith("OPENFILINGS_") or n == "SEC_EDGAR_USER_AGENT"]:
        monkeypatch.delenv(name, raising=False)
    app = make_app(tmp_path)
    with pytest.raises(ApiError, match="SEC_EDGAR_USER_AGENT"):
        post(app, "/api/fund/companies/search", q="apple", provider="sec_edgar")
    out = post(app, "/api/fund/provider/test", provider="sec_edgar")
    assert out["available"] is False and "SEC_EDGAR_USER_AGENT" in out["reason"]
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", CONTACT)                                          # the contact comes from the environment (or the git-ignored .env), and only from there
    later = make_app(tmp_path / "later")
    assert get(later, "/api/fund/companies", q="")["lookups"]["sec_edgar"]["available"] is True
    assert CONTACT not in json.dumps(get(later, "/api/fund/status")) and CONTACT not in json.dumps(get(later, "/api/fund/companies", q=""))


# ------------------------------------------------------------------------------------------------------------------------ the command line
@pytest.fixture()
def sec_cli(tmp_path, monkeypatch, capsys):
    """``quant fundamentals`` in a project folder of its own, with the SEC replaced by a stand-in and no OpenFilings key."""
    for name in SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    config = dataclasses.replace(load_config(), root=tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: config)
    fake = FakeSec()
    real = fcli._bench

    def bench(config, demo, action):
        wb = real(config, demo, action)
        if not demo:
            svc = wb.service
            svc.register("sec_edgar", SecEdgarProvider(pool=svc.pool("sec_edgar"), transport=fake, user_agent=CONTACT, cache_path=tmp_path / "tickers.json", sleep=lambda s: None))
        return wb
    monkeypatch.setattr(fcli, "_bench", bench)

    def run_cli(*argv):
        code = cli.main(["fundamentals", *argv])
        out = capsys.readouterr()
        return code, out.out, out.err
    run_cli.fake = fake
    return run_cli


def test_the_command_searches_adds_plans_and_fetches_from_the_sec_in_requests(sec_cli):
    code, out, _ = sec_cli("search", "apple", "--provider", "sec_edgar")
    assert code == 0 and "SEC EDGAR, \u201capple\u201d read as a company name" in out and "1 match (nothing spent)" in out and "Apple Inc." in out
    assert "quant fundamentals add --provider sec_edgar us:AAPL" in out and "sync --provider sec_edgar us:AAPL --budget 2" in out and "about 2 requests" in out
    code, out, _ = sec_cli("add", "AAPL", "us:WMT", "--provider", "sec_edgar")
    assert code == 0 and "2 companies registered" in out
    code, out, _ = sec_cli("plan", "--provider", "sec_edgar")
    assert code == 0 and "costing 4 requests" in out and "AAPL" in out
    code, _, err = sec_cli("sync", "--provider", "sec_edgar")
    assert code == 2 and "how many requests" in err and "--budget" in err                          # no budget, no spending, even when nothing is charged
    code, out, _ = sec_cli("sync", "--provider", "sec_edgar", "--budget", "4")
    assert code == 0 and "Fetched 2 companies" in out and "spent 4 requests" in out
    code, out, _ = sec_cli("plan", "--provider", "sec_edgar")
    assert code == 0 and "To fetch: 0 (2 already fresh)" in out
    code, out, _ = sec_cli("status")
    assert "SEC EDGAR" in out.replace("sec_edgar", "SEC EDGAR") or "sec_edgar" in out
    assert CONTACT not in out and "example.org" not in out


def test_the_years_of_a_company_from_the_sec_are_listed_and_asked_for_only_within_a_stated_budget(sec_cli):
    assert sec_cli("add", "AAPL", "--provider", "sec_edgar")[0] == 0
    assert sec_cli("sync", "--provider", "sec_edgar", "--budget", "2", "AAPL")[0] == 0
    code, out, _ = sec_cli("coverage", "AAPL", "--provider", "sec_edgar")
    assert code == 0 and "SEC EDGAR" in out and "FY2016" in out
    sent = len(sec_cli.fake.requests)
    code, out, _ = sec_cli("years", "AAPL", "--pick", "2010-2012", "--provider", "sec_edgar", "--ask-again")
    assert code == 0 and "Nothing was asked" in out and "1 request" in out and len(sec_cli.fake.requests) == sent               # the plan says what it would cost and spends nothing
    code, out, _ = sec_cli("years", "AAPL", "--pick", "2010-2012", "--provider", "sec_edgar", "--ask-again", "--budget", "1")
    assert code == 0 and "1 request spent" in out and len(sec_cli.fake.requests) == sent + 1


def test_the_sec_is_refused_in_the_demo_and_is_never_the_default_and_the_command_takes_no_contact(sec_cli):
    code, _, err = sec_cli("search", "apple", "--provider", "sec_edgar", "--demo")
    assert code == 2 and "real data" in err
    code, _, err = sec_cli("search", "apple")                                                     # OpenFilings is still the default, and it has no key here
    assert code == 2 and "OPENFILINGS_API_KEY" in err
    option_strings = {o for a in cli.build_parser()._subparsers._group_actions[0].choices["fundamentals"]._actions for o in a.option_strings}
    assert "--provider" in option_strings and not [o for o in option_strings if re.search(r"user|agent|contact|mail", o, re.I)]                        # the contact is read from the environment, never an argument


# ------------------------------------------------------------------------------------------------------------------------ the real transport, against a server of our own
@pytest.fixture()
def local_sec():
    """A server on this computer that answers the way the SEC does: gzipped JSON, a gzipped refusal page with the SEC's words, a redirect."""
    import http.server
    import threading
    import zlib

    def gz(data: bytes) -> bytes:
        return zlib.compress(data, wbits=31)

    class Handler(http.server.BaseHTTPRequestHandler):
        seen: list = []

        def do_GET(self):  # noqa: N802
            Handler.seen.append(dict(self.headers))
            if self.path == "/facts":
                body, status, extra = gz(b'{"cik": 1, "facts": {}}'), 200, {"Content-Encoding": "gzip", "Content-Type": "application/json"}
            elif self.path == "/slow-down":
                body, status, extra = gz(b"<html><title>SEC.gov | Request Rate Threshold Exceeded</title></html>"), 403, {"Content-Encoding": "gzip", "Content-Type": "text/html"}
            elif self.path == "/moved":
                body, status, extra = b"", 301, {"Location": "http://127.0.0.1:9/elsewhere"}
            elif self.path == "/broken-gzip":
                body, status, extra = b"not gzip at all", 403, {"Content-Encoding": "gzip"}
            else:
                body, status, extra = b"plain", 404, {}
            self.send_response(status)
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", Handler
    server.shutdown()


def test_the_real_transport_unzips_answers_and_refusal_pages_and_never_follows_a_redirect(local_sec):
    base, handler = local_sec
    status, headers, body = sec._urllib_transport(base + "/facts", {"User-Agent": CONTACT, "Accept-Encoding": "gzip"}, 10)
    assert status == 200 and json.loads(body) == {"cik": 1, "facts": {}} and handler.seen[-1]["User-Agent"] == CONTACT
    status, _, body = sec._urllib_transport(base + "/slow-down", {"User-Agent": CONTACT}, 10)
    assert status == 403 and b"Request Rate Threshold Exceeded" in body                         # a refusal page arrives gzipped, and is read
    with pytest.raises(RateLimited):
        sec.SecEdgarProvider._interpret(status, body)
    status, headers, _ = sec._urllib_transport(base + "/moved", {"User-Agent": CONTACT}, 10)
    assert status == 301 and "127.0.0.1:9" in {k.lower(): v for k, v in headers.items()}["location"]                 # nothing is followed: a redirect would take the contact to another host
    assert sec._urllib_transport(base + "/broken-gzip", {"User-Agent": CONTACT}, 10)[2] == b""        # a broken body is an empty one, not a crash
    assert sec._urllib_transport(base + "/nothing", {"User-Agent": CONTACT}, 10)[:1] == (404,)
    with pytest.raises(ProviderError, match="could not reach the SEC"):
        sec._urllib_transport("http://127.0.0.1:1/", {"User-Agent": CONTACT}, 2)
    assert CONTACT not in str(pytest.raises(ProviderError, sec._urllib_transport, "http://127.0.0.1:1/", {"User-Agent": CONTACT}, 2).value)
    prov = SecEdgarProvider(user_agent=CONTACT, sleep=lambda s: None, base_url=base)               # through the provider, over a real socket
    with pytest.raises(NotFound):
        prov.get_financials("cik:5")
