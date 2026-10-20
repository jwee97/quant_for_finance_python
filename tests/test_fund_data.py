"""The fundamentals data foundation: the vocabulary, the database, the key pool and the credit quota, the OpenFilings REST adapter, and the database-first sync, against stand-in answers of the
documented shape and a simulated world whose truth is known. Nothing here talks to the network and no real key is used."""

from __future__ import annotations

import http.server
import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from urllib import parse, request

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import canonical, quality
from src.fundamentals.errors import CreditsExhausted, FundamentalsError, InvalidKey, MissingKey, NotFound, PlanError, ProviderError, QuotaExhausted, RateLimited, SyncPending
from src.fundamentals.keys import KeyPool, discover
from src.fundamentals.providers import openfilings as of
from src.fundamentals.providers.openfilings import OpenFilingsProvider, parse_financials, split_symbol
from src.fundamentals.providers.simulated import SimulatedProvider, SimulatedWorld
from src.fundamentals.schema import Database, initialise
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement
from src.fundamentals.usage import Allowance, UsageLog, allowance

# the line items OpenFilings documents (https://openfilings.org/standardized-financials), income statement, balance sheet, cash flow and shares
DOCUMENTED_KEYS = set("""
revenue revenue_quarter gross_profit operating_income net_income net_income_attributable net_income_minority_interest ebitda rnd income_tax pretax_income eps_diluted ebit interest_expense
non_operating_income non_operating_expenses roe_official equity_ratio_official effective_tax_rate total_shareholder_return total_return_share_price_index adjusted_eps gross_margin_ttm
operating_margin_ttm profit_margin_ttm return_on_equity_ttm return_on_assets_ttm ebitda_ttm cost_of_revenue cost_of_goods_sold sga_expense da_supplemental amort_goodwill_intangibles
other_operating_expense interest_income net_interest_expense equity_in_earnings currency_exchange_gain_loss other_non_operating_income_exp ebt_excl_unusual restructuring_charges
merger_restruct_charges impairment_goodwill gain_loss_sale_investments gain_loss_sale_assets asset_writedown in_process_rd insurance_settlements legal_settlements other_unusual_items
total_unusual_items earnings_continuing_ops earnings_discontinued_ops extraordinary_items preferred_dividends ni_to_common_incl_extra ni_to_common_excl_extra eps_basic
total_assets current_assets current_liabilities total_liabilities equity noncontrolling_interest retained_earnings cash total_debt long_term_debt short_term_debt deferred_revenue_current
accumulated_deficit trade_receivables trade_payables ibd_current ibd_noncurrent noncurrent_assets investment_securities lease_liabilities_cl lease_liabilities_ncl deferred_tax_liabilities
provision_for_bonuses net_defined_benefit_liability net_defined_benefit_asset investments_and_other_assets buildings_and_structures construction_in_progress contract_liabilities
machinery_and_equipment short_term_securities accounts_payable_other advances_received allowance_for_doubtful_accounts adjusted_bps debt_to_equity current_ratio short_term_investments
total_cash_st_investments accounts_receivable total_receivables inventory prepaid_expenses other_current_assets gross_ppe accumulated_depreciation net_ppe long_term_investments goodwill
other_intangibles total_intangibles other_long_term_assets accounts_payable accrued_expenses current_portion_lt_debt current_portion_leases other_current_liabilities capital_leases
pension_liabilities other_long_term_liabilities preferred_stock common_stock additional_paid_in_capital treasury_stock comprehensive_income_equity
operating_cash_flow capex investing_cash_flow depreciation_amortization free_cash_flow stock_based_comp financing_cash_flow cash_dividends_paid dividends_total_announced
adjusted_dividend_per_share free_cash_flow_ttm cf_net_income asset_writedown_restructuring_cf cash_from_discontinued_ops change_in_accounts_receivable change_in_inventories
change_in_accounts_payable change_in_unearned_revenue change_in_income_taxes change_in_deferred_tax change_in_other_working_capital sale_of_ppe cash_acquisitions divestitures
sale_purchase_of_investments other_investing_activities short_term_debt_issued long_term_debt_issued total_debt_issued short_term_debt_repaid long_term_debt_repaid total_debt_repaid
issuance_of_common_stock repurchase_of_common_stock issuance_of_preferred_stock repurchase_of_preferred_stock common_dividends_paid preferred_dividends_paid other_financing_activities
foreign_exchange_effect misc_cash_flow_adjustments net_change_in_cash cash_interest_paid cash_taxes_paid levered_free_cash_flow unlevered_free_cash_flow
shares_diluted shares_basic employee_count directors_shares_held directors_ownership_ratio cross_shareholding_total_book_value cross_shareholding_total_shares_held split_adjustment_factor
""".split())

KEY_1, KEY_2, KEY_3 = "of_test_aaaaaaaaaaaa", "of_test_bbbbbbbbbbbb", "of_test_cccccccccccc"


@pytest.fixture(scope="module")
def world():
    return SimulatedWorld(n_companies=40, start_year=2010, end_year=2017, seed=11)


@pytest.fixture()
def store(tmp_path):
    return FundamentalsStore(tmp_path / "f.db")


class Clock:
    """A settable UTC clock, so that a day or a month can roll over inside a test."""

    def __init__(self, start="2026-10-20 12:00"):
        self.now = datetime.strptime(start, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


# ------------------------------------------------------------------------------------------------------------------------ the vocabulary
def test_the_schema_is_created_once_and_seeds_the_markets(tmp_path):
    db = Database(tmp_path / "x.db")
    with db.transaction() as conn:
        initialise(conn)
        initialise(conn)                                                                  # idempotent
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"securities", "financial_periods", "fundamental_observations", "raw_fundamental_payloads", "fundamental_metrics", "api_usage", "alpha_library", "alpha_graveyard"} <= tables
        assert conn.execute("SELECT COUNT(*) FROM markets WHERE code = 'SGX'").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_every_line_item_the_application_asks_openfilings_for_is_in_its_documented_taxonomy():
    assert len(canonical.OPENFILINGS_KEYS) > 40 and set(canonical.OPENFILINGS_KEYS) <= DOCUMENTED_KEYS
    assert set(canonical.OPENFILINGS_MAP) <= set(canonical.METRICS)                          # and every canonical code it fills is one the application defines


def test_openfilings_outflows_are_read_as_magnitudes_and_a_missing_line_stays_missing():
    values = {"capex": -120.0, "repurchase_of_common_stock": -30.0, "cash_acquisitions": -15.0, "cash_dividends_paid": -10.0, "revenue": 1000.0, "cost_of_revenue": 600.0,
              "interest_expense": -5.0, "operating_cash_flow": 200.0, "short_term_debt": 20.0, "long_term_debt": 80.0, "cash": 50.0, "a_key_nobody_mapped": 1.0, "rnd": None,
              "net_income": float("nan")}
    obs, unmapped = canonical.normalise_values("openfilings", values)
    assert obs["capital_expenditure"]["value"] == 120.0 and obs["share_repurchases"]["value"] == 30.0 and obs["dividends_paid"]["value"] == 10.0 and obs["acquisitions"]["value"] == 15.0
    assert obs["interest_expense"]["value"] == 5.0
    assert obs["gross_profit"] == {"value": 400.0, "source_metric": "derived: revenue - cost_of_revenue", "flag": "derived"}
    assert obs["free_cash_flow"]["value"] == 80.0 and obs["total_debt"]["value"] == 100.0 and obs["cash_and_equivalents"]["value"] == 50.0 and obs["cash"]["flag"] == "derived"
    assert "research_and_development" not in obs and "net_income" not in obs                    # null and not-a-number are missing, never zero
    assert unmapped == ["a_key_nobody_mapped"]
    with pytest.raises(ValueError, match="no field mapping"):
        canonical.normalise_values("nobody", {})
    assert canonical.normalise_values("openfilings", {"shares_basic": 100.0})[0]["shares_outstanding"]["flag"] == "estimated"        # a weighted count standing in for the count in issue


# ------------------------------------------------------------------------------------------------------------------------ the security master and the store
def test_the_same_ticker_on_two_exchanges_is_two_securities_and_a_provider_cannot_blank_a_field(store):
    a = store.upsert_security(CompanyProfile("openfilings", "sg:D05", "D05", "DBS Group", exchange="SGX", market="SGX", country="SG", currency="SGD", sector="Financials"))
    b = store.upsert_security(CompanyProfile("openfilings", "us:D05", "D05", "Another D05", exchange="NYSE", market="NYSE", country="US", currency="USD"))
    assert a != b
    again = store.upsert_security(CompanyProfile("simulated", "D05", "D05", "DBS Group", exchange="SGX", market="SGX", sector=None, industry="Banks"))
    assert again == a
    row = store.get_security(a)
    assert row["sector"] == "Financials" and row["industry"] == "Banks" and row["country"] == "SG"       # kept what another provider supplied, added what it lacked
    hits = store.search("D05")
    assert set(hits["exchange"]) == {"SGX", "NYSE"} and {"country", "company_name"} <= set(hits.columns)


def test_enriching_a_security_fills_blanks_and_never_creates_a_second_company(store):
    sid = store.upsert_security(CompanyProfile("openfilings", "hk:0700", "0700", "", exchange="HK", market="HK"))
    store.enrich_security(sid, CompanyProfile("openfilings", "hk:0700", "0700.HK", "Tencent Holdings", exchange="HK", country="HK", currency="HKD", isin="KYG875721634", sector="Technology"))
    store.enrich_security(sid, CompanyProfile("openfilings", "hk:0700", "0700", "A different name", exchange="HK", sector="Something else"))
    row = store.get_security(sid)
    assert len(store.securities()) == 1 and (row["ticker"], row["company_name"], row["sector"], row["country"], row["isin"]) == ("0700", "Tencent Holdings", "Technology", "HK", "KYG875721634")


def test_storing_the_same_statement_twice_changes_nothing_and_a_restatement_is_a_new_vintage_public_from_the_day_it_was_seen(store):
    sid = store.upsert_security(CompanyProfile("simulated", "AAA", "AAA", "A", exchange="NYSE", market="NYSE"))
    p = PeriodStatement("FY", "2020-12-31", {"revenue": 100.0, "cost_of_revenue": 60.0, "net_income": 10.0}, fiscal_year=2020, filing_date="2021-02-15", reporting_currency="USD")
    first = store.store_financials(sid, "simulated", [p], retrieved="2021-03-01")
    assert first["periods"] == 1 and first["observations"] == 4                                 # revenue, cost, net income and the derived gross profit
    assert store.store_financials(sid, "simulated", [p], retrieved="2021-03-02")["unchanged"] == 1 and store.counts()["observations"] == 4
    restated = PeriodStatement("FY", "2020-12-31", {"revenue": 104.0, "cost_of_revenue": 60.0, "net_income": 10.0}, fiscal_year=2020, filing_date="2021-02-15", reporting_currency="USD")
    assert store.store_financials(sid, "simulated", [restated], retrieved="2022-06-30")["vintages"] == 1
    periods = store.periods([sid]).sort_values("vintage")
    assert list(periods["vintage"]) == [0, 1] and list(periods["pit_quality"]) == ["filed", "restated"]
    assert periods.iloc[1]["publication_date"] == pd.Timestamp("2022-06-30") and pd.isna(periods.iloc[0]["publication_date"])
    obs = store.observations([sid], ["revenue"]).merge(periods[["period_id", "vintage"]], on="period_id").sort_values("vintage")
    assert list(obs["value"]) == [100.0, 104.0] and list(obs["is_restated"]) == [0, 1]


def test_a_period_without_a_filing_date_is_marked_estimated_not_given_the_period_end(store):
    sid = store.upsert_security(CompanyProfile("simulated", "AAA", "AAA", "A", exchange="NYSE"))
    store.store_financials(sid, "simulated", [PeriodStatement("FY", "2019-12-31", {"revenue": 5.0})])
    row = store.periods([sid]).iloc[0]
    assert row["pit_quality"] == "estimated" and pd.isna(row["filing_date"]) and pd.isna(row["accepted_date"]) and pd.isna(row["publication_date"])


def test_a_raw_answer_is_kept_once_and_can_be_read_back(store):
    payload = {"a": [1, 2, 3], "b": "x"}
    first, new1 = store.save_raw("openfilings", None, "financials:annual", {"symbol": "AAA"}, payload)
    second, new2 = store.save_raw("openfilings", None, "financials:annual", {"symbol": "AAA"}, dict(reversed(list(payload.items()))))
    assert new1 and not new2 and first == second and store.load_raw(first) == payload
    assert store.counts()["raw_payloads"] == 1


# ------------------------------------------------------------------------------------------------------------------------ keys
def test_keys_are_found_in_every_form_once_each_and_without_the_wrapping_a_paste_adds(monkeypatch, tmp_path):
    for name in [n for n in list(os.environ) if n.startswith("OPENFILINGS_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENFILINGS_API_KEYS", "of_alpha1111111, of_beta2222222222;\n of_gamma333333333")
    monkeypatch.setenv("OPENFILINGS_API_KEY", '"of_delta4444444444"')
    monkeypatch.setenv("OPENFILINGS_API_KEY_2", "OPENFILINGS_API_KEY_2=of_epsilon55555555")
    monkeypatch.setenv("OPENFILINGS_API_KEY_3", "of_alpha1111111")                         # listed twice
    monkeypatch.setenv("OPENFILINGS_API_KEY_4", "<your key>")                              # a placeholder
    keys, notes = discover("openfilings", tmp_path)
    assert [v for v, _ in keys] == ["OPENFILINGS_API_KEYS"] * 3 + ["OPENFILINGS_API_KEY", "OPENFILINGS_API_KEY_2"]
    assert [k for _, k in keys] == ["of_alpha1111111", "of_beta2222222222", "of_gamma333333333", "of_delta4444444444", "of_epsilon55555555"]
    assert any("counted once" in n for n in notes) and not any("alpha1111111" in n or "delta4444444444" in n for n in notes)
    (tmp_path / ".env").write_text("OPENFILINGS_API_KEY_9=of_zeta66666666\n")
    assert discover("openfilings", tmp_path)[0][-1] == ("OPENFILINGS_API_KEY_9", "of_zeta66666666")


def _pool(tmp_path, keys=(KEY_1,), clock=None, daily=100, monthly=1000, name="u.db"):
    usage = UsageLog(Database(tmp_path / name), clock)
    return KeyPool("openfilings", [(f"OPENFILINGS_API_KEY_{i + 1}", k) for i, k in enumerate(keys)], usage, "salt", Allowance("credits", daily, monthly, "test")), usage


def test_the_first_valid_key_does_the_work_and_the_others_wait_as_standbys_even_when_it_runs_dry(tmp_path):
    pool, usage = _pool(tmp_path, (KEY_1, KEY_2, KEY_3), daily=30)
    used: list[str] = []
    for _ in range(3):
        pool.call(lambda key: used.append(key) or 1, "financials", "us:AAA", credits=10)
    assert used == [KEY_1] * 3 and usage.used_today("openfilings") == 30
    assert [(s["slot"], s["role"], s["state"]) for s in pool.slots()] == [("key 1", "active", "spent_day"), ("key 2", "standby", "ready"), ("key 3", "standby", "ready")]
    with pytest.raises(QuotaExhausted) as info:
        pool.call(lambda key: used.append(key) or 1, "financials", "us:AAA", credits=10)
    assert used == [KEY_1] * 3                                                              # no request went out on a standby key
    message = str(info.value)
    assert "daily allowance is spent" in message and "00:00 UTC" in message and "does not rotate keys" in message and KEY_1 not in message and KEY_2 not in message
    assert pool.remaining_today() == 0


def test_a_revoked_key_hands_over_to_the_next_and_is_not_tried_again(tmp_path):
    pool, _ = _pool(tmp_path, (KEY_1, KEY_2, KEY_3))
    seen: list[str] = []

    def fn(key):
        seen.append(key)
        if key == KEY_1:
            raise InvalidKey("401")
        return "ok"

    assert pool.call(fn, "search", "X") == "ok" and seen == [KEY_1, KEY_2]
    assert [s["state"] for s in pool.slots()] == ["invalid", "ready", "ready"] and [s["role"] for s in pool.slots()] == ["unusable", "active", "standby"]
    seen.clear()
    pool.call(fn, "search", "X")
    assert seen == [KEY_2]
    with pytest.raises(InvalidKey, match="every OPENFILINGS key was rejected"):
        _pool(tmp_path, (KEY_1,), name="v.db")[0].call(fn, "search", "X")


def test_a_provider_answer_that_the_allowance_is_gone_is_remembered_until_the_day_or_month_rolls_over(tmp_path):
    clock = Clock("2026-10-30 23:00")
    pool, usage = _pool(tmp_path, clock=clock)
    calls: list[int] = []

    def spent_day(key):
        calls.append(1)
        raise RateLimited("daily", dimension="daily")

    with pytest.raises(QuotaExhausted):
        pool.call(spent_day, "financials", "us:AAA", credits=10)
    with pytest.raises(QuotaExhausted):
        pool.call(lambda key: calls.append(1), "financials", "us:AAA", credits=10)
    assert len(calls) == 1 and pool.slots()[0]["state"] == "spent_day"                       # the second attempt never left the machine
    clock.advance(hours=2)                                                                    # 01:00 UTC on 31 October: a new day in the same month
    assert pool.call(lambda key: "fresh", "financials", "us:AAA", credits=10) == "fresh"
    with pytest.raises(QuotaExhausted):
        pool.call(lambda key: (_ for _ in ()).throw(RateLimited("monthly", dimension="monthly")), "financials", "us:AAA", credits=10)
    clock.advance(days=1)                                                                     # 1 November 01:00: a new month, a new allowance
    assert pool.call(lambda key: "november", "financials", "us:AAA", credits=10) == "november"
    assert usage.used_month("openfilings") == 10 and usage.used_today("openfilings") == 10


def test_when_the_credits_are_gone_the_pool_stops_until_the_user_tests_after_topping_up(tmp_path):
    pool, usage = _pool(tmp_path)

    def broke(key):
        raise CreditsExhausted("402", balance=0.0, required=10.0)

    with pytest.raises(QuotaExhausted, match="top up"):
        pool.call(broke, "financials", "us:AAA", credits=10)
    assert pool.slots()[0]["state"] == "no_credits"
    with pytest.raises(QuotaExhausted):
        pool.call(lambda key: "x", "financials", "us:AAA", credits=10)
    assert pool.call(lambda key: "after the top-up", "markets", force=True) == "after the top-up"       # the Test button asks the provider whatever was remembered
    assert pool.slots()[0]["state"] == "ready" and pool.call(lambda key: "x", "financials", credits=10) == "x"


def test_credits_are_counted_a_failed_call_costs_nothing_and_the_counts_survive_a_restart(tmp_path):
    pool, usage = _pool(tmp_path)
    pool.call(lambda key: 1, "financials", "us:AAA", credits=10)
    pool.call(lambda key: 1, "lookup", "us:AAA", credits=1)
    with pytest.raises(NotFound):
        pool.call(lambda key: (_ for _ in ()).throw(NotFound("nothing")), "financials", "us:ZZZ", credits=10)
    assert usage.used_today("openfilings") == 11 and usage.requests_today("openfilings") == 3 and pool.remaining_today() == 89
    again = KeyPool("openfilings", [("OPENFILINGS_API_KEY_1", KEY_1)], usage, "salt", Allowance("credits", 100, 1000, "test"))
    assert again.remaining_today() == 89 and again.remaining_month() == 989
    summary = usage.summary("openfilings", again.slots())
    assert (summary.unit, summary.used_today, summary.successful, summary.failed) == ("credits", 11, 2, 1)


def test_a_call_that_would_overspend_the_day_is_not_sent(tmp_path):
    pool, _ = _pool(tmp_path, daily=25)
    pool.call(lambda key: 1, "financials", credits=10)
    pool.call(lambda key: 1, "financials", credits=10)
    sent: list[int] = []
    with pytest.raises(QuotaExhausted, match="needs 10, 5 left"):
        pool.call(lambda key: sent.append(1), "financials", credits=10)
    assert sent == [] and pool.call(lambda key: "a search still fits", "search", credits=1) == "a search still fits"


def test_no_key_material_reaches_the_database_or_the_status_lines(tmp_path):
    secret = "of_SECRETKEYVALUE123456"
    pool, usage = _pool(tmp_path, (secret,))
    pool.call(lambda key: 1, "lookup", "us:AAA")
    with pytest.raises(ProviderError):
        pool.call(lambda key: (_ for _ in ()).throw(ProviderError("boom")), "lookup", "us:AAA")
    dump = json.dumps(pool.slots()) + json.dumps(usage.history("openfilings")) + json.dumps(usage.recent("openfilings"))
    with sqlite3.connect(tmp_path / "u.db") as conn:
        dump += json.dumps([list(r) for r in conn.execute("SELECT * FROM api_usage")])
    assert secret not in dump and secret[:8] not in dump and "key_id" in json.dumps(pool.slots())
    with pytest.raises(MissingKey, match="OPENFILINGS_API_KEY"):
        KeyPool("openfilings", [], usage).call(lambda key: 1, "lookup")


def test_the_allowance_follows_the_plan_and_an_environment_override_wins(monkeypatch):
    for name in ("OPENFILINGS_PLAN", "OPENFILINGS_DAILY_LIMIT", "OPENFILINGS_MONTHLY_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    free = allowance("openfilings")
    assert (free.unit, free.daily, free.monthly) == ("credits", 100, 1000) and "assumed" in free.source
    pro = allowance("openfilings", "pro")
    assert (pro.daily, pro.monthly) == (2500, 25000) and "reported by the account" in pro.source
    monkeypatch.setenv("OPENFILINGS_PLAN", "business")
    assert allowance("openfilings", "pro").daily == 10000
    monkeypatch.setenv("OPENFILINGS_DAILY_LIMIT", "40")
    assert allowance("openfilings").daily == 40 and allowance("openfilings").monthly == 100000
    assert allowance("simulated").unit == "requests"


# ------------------------------------------------------------------------------------------------------------------------ the OpenFilings adapter
def _values(year: int, scale: float = 1.0) -> dict:
    return {"revenue": 1000.0 * scale + year, "cost_of_revenue": 600.0 * scale, "gross_profit": 400.0 * scale + year, "operating_income": 150.0 * scale, "net_income": 90.0 * scale,
            "total_assets": 5000.0 * scale, "total_liabilities": 3000.0 * scale, "equity": 2000.0 * scale, "current_assets": 1500.0 * scale, "current_liabilities": 900.0 * scale,
            "inventory": 40.0 * scale, "operating_cash_flow": 150.0 * scale, "capex": -50.0 * scale, "cash_dividends_paid": -20.0 * scale, "shares_diluted": 100.0, "eps_diluted": 0.9,
            "rnd": None, "ebit": None}


DIRECTORY = [                                                                                # what the stand-in's search knows (the shapes follow the OpenAPI description of /entities/search)
    {"display_names": ["Apple Inc."], "isin": "US0378331005", "lei": "H" * 20,
     "listings": [{"market_id": "us", "ticker": "AAPL", "registry": "sec", "primary": True, "company_name": "Apple Inc."}]},
    {"display_names": ["NVIDIA Corp"], "isin": "US67066G1040", "lei": "N" * 20,
     "listings": [{"market_id": "mx", "ticker": "NVDA", "registry": "bmv", "primary": False}, {"market_id": "us", "ticker": "NVDA", "registry": "sec", "primary": True, "company_name": "NVIDIA Corp"}]},
    {"display_names": ["Toyota Motor Corp"], "isin": "JP3633400001",
     "listings": [{"market_id": "jp", "ticker": "7203", "registry": "edinet", "primary": True, "company_name": "Toyota Motor Corporation"}, {"market_id": "us", "ticker": "TM", "registry": "sec", "primary": False}]},
    {"display_names": ["Toyota Tsusho Corp"], "isin": "JP3635000007", "listings": [{"market_id": "jp", "ticker": "8015", "registry": "edinet", "primary": True}]},
    {"display_names": ["Deutsche Bank AG"], "isin": "DE0005140008", "listings": [{"market_id": "de", "ticker": "DBK", "registry": "bafin", "primary": True}]},      # a market without financials
]


class FakeOpenFilings:
    """A stand-in for the REST API with the behaviour its documentation describes: keys belong to accounts, credits are charged per call against a daily and a monthly allowance, a call that
    fails is not charged, and the error answers have the documented shape."""

    def __init__(self, accounts=None, daily=100, monthly=1000, balance=0, plan="free"):
        self.accounts = {KEY_1: "a"} if accounts is None else accounts
        self.daily, self.monthly, self.balance, self.plan = daily, monthly, balance, plan
        self.day = 0
        self.used_day: dict[tuple[str, int], int] = {}
        self.used_month: dict[str, int] = {}
        self.requests: list[dict] = []
        self.unknown: set[str] = set()
        self.pending: set[str] = set()
        self.bursts = 0
        self.broken = 0
        self.reject_line_items_once = False
        self.entity_ticker = None
        self.directory = [dict(e) for e in DIRECTORY]
        self.empty_search_as_200 = False
        self.skipped: list = []

    def __call__(self, url, headers, timeout):
        parts = parse.urlparse(url)
        q = dict(parse.parse_qsl(parts.query))
        key = headers.get("X-API-Key", "")
        self.requests.append({"url": url, "headers": dict(headers), "path": parts.path, "query": q})
        account = self.accounts.get(key)
        if account is None:
            return 401, {}, json.dumps({"detail": "Invalid API key"}).encode()
        cost = 10 if parts.path.endswith("/companies/financials") else 1
        if self.bursts:
            self.bursts -= 1
            return 429, {"Retry-After": "1"}, json.dumps({"detail": {"error": "slow_down"}}).encode()
        if self.broken:
            self.broken -= 1
            return 502, {}, b'{"detail": "registry unavailable"}'
        used_today = self.used_day.get((account, self.day), 0)
        if used_today + cost > self.daily:
            body = {"detail": {"error": "quota_exceeded", "dimension": "daily", "daily_limit": self.daily, "daily_used": used_today, "daily_remaining": self.daily - used_today}}
            return 429, {"Retry-After": "40000"}, json.dumps(body).encode()
        if self.used_month.get(account, 0) + cost > self.monthly:
            if self.balance < cost:
                return 402, {"X-Credits-Required": str(cost), "X-Credits-Balance": str(self.balance)}, json.dumps({"detail": {"error": "credits_exhausted", "dimension": "balance",
                                                                                                                              "credit_balance": self.balance}}).encode()
            self.balance -= cost
        else:
            self.used_month[account] = self.used_month.get(account, 0) + cost
        ticker, market = q.get("ticker", ""), q.get("market_id", "us")
        if ticker in self.unknown:
            return 404, {}, json.dumps({"detail": "unknown ticker"}).encode()
        if ticker in self.pending:
            return 202, {"Retry-After": "120"}, json.dumps({"status": "syncing", "retry_after": 120, "poll_url": "/x"}).encode()
        self.used_day[(account, self.day)] = used_today + cost
        if parts.path.endswith("/markets"):
            return 200, {}, json.dumps({"plan": self.plan, "markets": [{"id": "us", "label": "United States"}, {"id": "jp", "label": "Japan"}], "allowed_market_ids": ["us", "jp"],
                                        "all_markets": False}).encode()
        if parts.path.endswith("/entities/lookup"):
            return 200, {}, json.dumps({"display_name": f"{ticker} Corp", "identifiers": {"isin": f"US{abs(hash(ticker)) % 10**9:09d}0", "lei": "L" * 20, "yahoo_symbol": ticker},
                                        "industry": {"group": "Technology", "sic": "3571", "damodaran": "Computers"}, "listing": f"{market}:{ticker}"}).encode()
        if parts.path.endswith("/entities/search"):
            status, headers, body = self._search(q)
            if status != 200:                                                                     # a call that fails is refunded
                self.used_day[(account, self.day)] = used_today
                self.used_month[account] = max(0, self.used_month.get(account, 0) - cost)
            return status, headers, body
        if parts.path.endswith("/companies/financials"):
            if q.get("line_items") not in (None, "ALL") and self.reject_line_items_once:
                self.reject_line_items_once = False
                self.used_day[(account, self.day)] = used_today                                 # a rejected call is refunded
                return 422, {}, json.dumps({"detail": [{"loc": ["query", "line_items"], "msg": "unknown line item"}]}).encode()
            first, last = int(q.get("fiscal_year_from", 2015)), int(q.get("fiscal_year_to", 2023))
            first = max(first, 2012)
            periods = [{"fiscal_year": y, "fiscal_period": "FY", "period_end": f"{y}-12-31", "filing_id": f"{market}:{ticker}:annual:{y}", "filing_date": f"{y + 1}-02-{10 + y % 10:02d}",
                        "form_type": "10-K", "currency": "USD", "values": _values(y, 1.0 + 0.05 * (y - 2012))} for y in range(first, last + 1)]
            return 200, {}, json.dumps({"entity": {"name": f"{ticker} Corp", "market_id": market, "ticker": self.entity_ticker or ticker, "lei": "L" * 20, "isin": "US0000000000"},
                                        "query": {"fiscal_year_from": first, "fiscal_year_to": last, "fiscal_period": "fy"}, "currency_latest": "USD", "periods": periods}).encode()
        return 404, {}, b'{"detail": "no such endpoint"}'


    def _search(self, q):
        """``GET /api/v1/entities/search``: ``query`` matches names, ``ticker`` (with an optional ``market_id``) a listing, ``isin`` and ``lei`` the identifiers. The best match is ``entity``, the rest ``entities``."""
        def matches(entity):
            if q.get("query"):
                return q["query"].lower() in " ".join(entity["display_names"]).lower()
            if q.get("ticker"):
                return any(x["ticker"].upper() == q["ticker"].upper() and (not q.get("market_id") or x["market_id"] == q["market_id"]) for x in entity["listings"])
            if q.get("isin"):
                return entity.get("isin") == q["isin"]
            return bool(q.get("lei")) and entity.get("lei") == q["lei"]
        found = [e for e in self.directory if matches(e)]
        if not found:
            if self.empty_search_as_200:
                return 200, {}, json.dumps({"entity": {"display_names": [], "listings": []}, "entities": [], "message": "No issuer matched"}).encode()
            return 404, {}, json.dumps({"detail": "no issuer matches"}).encode()
        return 200, {}, json.dumps({"entity": found[0], "entities": found[1:], "ambiguous": len(found) > 1, "markets_skipped": self.skipped, "message": None}).encode()


def _provider(tmp_path, fake=None, keys=(KEY_1,), clock=None, daily=100, monthly=1000, sleep=lambda s: None):
    pool, usage = _pool(tmp_path, keys, clock, daily, monthly, name="p.db")
    fake = fake or FakeOpenFilings({k: "a" for k in keys})
    return OpenFilingsProvider(pool, transport=fake, sleep=sleep), fake, usage, pool


def test_a_financials_series_is_one_call_of_ten_credits_with_a_filing_date_for_every_year(tmp_path):
    prov, fake, usage, _ = _provider(tmp_path)
    result = prov.get_financials("us:AAPL", start_year=2018, end_year=2022)
    assert result.cost == 10 and result.requests == 1 and usage.used_today("openfilings") == 10 and len(fake.requests) == 1
    assert [p.fiscal_year for p in result.periods] == [2018, 2019, 2020, 2021, 2022]
    newest = result.periods[-1]
    assert newest.fiscal_period == "FY" and newest.period_end == "2022-12-31" and newest.filing_date == "2023-02-12" and newest.reporting_currency == "USD"
    assert newest.provider_record_id == "us:AAPL:annual:2022" and result.profile.provider_symbol == "us:AAPL" and result.profile.isin == "US0000000000"
    sent = fake.requests[0]
    assert sent["path"] == "/api/v1/companies/financials" and sent["query"]["ticker"] == "AAPL" and sent["query"]["market_id"] == "us" and sent["query"]["fiscal_period"] == "fy"
    assert set(sent["query"]["line_items"].split(",")) == set(canonical.OPENFILINGS_KEYS)
    obs, unmapped = canonical.normalise_values("openfilings", newest.values)
    assert obs["capital_expenditure"]["value"] > 0 and obs["shareholders_equity"]["value"] > 0 and "research_and_development" not in obs and unmapped == []
    assert result.exhausted is False                                                          # the first year asked for was returned, so older years may exist


def test_the_history_is_reported_exhausted_only_when_the_provider_starts_later_than_asked(tmp_path):
    prov, _, _, _ = _provider(tmp_path)
    assert prov.get_financials("us:AAPL", start_year=2000, end_year=2015).exhausted is True            # asked for 2000, the first year is 2012
    assert prov.get_financials("us:AAPL", start_year=2014, end_year=2015).exhausted is False


def test_the_key_goes_in_a_header_never_in_the_address_and_never_in_a_message(tmp_path):
    prov, fake, _, _ = _provider(tmp_path, keys=(KEY_1,))
    prov.get_company("us:AAPL")
    prov.search("Apple")
    assert all(KEY_1 not in r["url"] for r in fake.requests) and all(r["headers"]["X-API-Key"] == KEY_1 for r in fake.requests)
    bad, _, _, _ = _provider(tmp_path, FakeOpenFilings({}), keys=("of_wrong_wrong_wrong",))
    with pytest.raises(InvalidKey) as info:
        bad.get_company("us:AAPL")
    assert "of_wrong" not in str(info.value)


def test_a_lookup_gives_the_sector_the_industry_and_the_identifiers_for_one_credit(tmp_path):
    prov, _, usage, _ = _provider(tmp_path)
    company = prov.get_company("jp:7203")
    assert (company.provider_symbol, company.ticker, company.exchange, company.country, company.sector, company.industry) == ("jp:7203", "7203", "JP", "JP", "Technology", "Computers")
    assert company.lei == "L" * 20 and company.isin.startswith("US") and company.extra["yahoo_symbol"] == "7203" and usage.used_today("openfilings") == 1
    assert prov.get_company("AAPL").provider_symbol == "us:AAPL"                               # no market given: the default one
    found = prov.search("Apple")
    assert [(c.provider_symbol, c.name, c.isin) for c in found] == [("us:AAPL", "Apple Inc.", "US0378331005")]
    assert split_symbol("7203.T", "jp") == ("jp", "7203.T") and split_symbol("hk:0700") == ("hk", "0700")


@pytest.mark.parametrize("text, kind, params, other", [
    ("Nvidia", "name", {"query": "Nvidia"}, ["symbol"]),                                       # a name: the other reading is offered, never tried unasked
    ("  Toyota   Motor ", "name", {"query": "Toyota Motor"}, []),
    ("NVDA", "symbol", {"ticker": "NVDA"}, ["name"]),
    ("nvda", "name", {"query": "nvda"}, ["symbol"]),                                           # lower case is a name unless told otherwise: ``--kind symbol`` or us:nvda
    ("7203", "symbol", {"ticker": "7203"}, []),
    ("BRK.B", "symbol", {"ticker": "BRK.B"}, ["name"]),
    ("us:NVDA", "symbol", {"ticker": "NVDA", "market_id": "us"}, ["name"]),
    ("jp: 7203", "symbol", {"ticker": "7203", "market_id": "jp"}, []),
    ("US67066G1040", "isin", {"isin": "US67066G1040"}, []),
    ("5493001KJTIIGC8Y1R12", "lei", {"lei": "5493001KJTIIGC8Y1R12"}, []),
    ("NVIDIA", "name", {"query": "NVIDIA"}, ["symbol"]),                                       # six capitals: a name typed in capitals, not a ticker
])
def test_the_text_decides_how_a_search_is_asked_only_when_its_shape_is_plain(text, kind, params, other):
    plan = of.search_plan(text)
    assert plan["kind"] == kind and {k: v for k, v in plan["params"].items() if v is not None} == params and plan["alternatives"] == other
    forced = of.search_plan("nvda", "symbol", "US")
    assert forced["kind"] == "symbol" and forced["params"] == {"ticker": "NVDA", "market_id": "us"}
    with pytest.raises(ProviderError, match="kind must be one of"):
        of.search_plan("nvda", "isbn")
    with pytest.raises(ProviderError, match="short code"):
        of.search_plan("nvda", "symbol", "united states")


def test_a_search_asks_the_directory_for_one_credit_and_reads_the_best_match_and_the_others(tmp_path):
    prov, fake, usage, _ = _provider(tmp_path)
    found = prov.search("Nvidia")
    assert [(c.provider_symbol, c.name, c.isin, c.country) for c in found] == [("us:NVDA", "NVIDIA Corp", "US67066G1040", "US")]
    assert found[0].extra["primary"] is True and found[0].extra["also_listed"] == ["mx:NVDA"] and found[0].extra["has_financials"] is True      # one company, shown under its primary listing
    sent = fake.requests[-1]
    assert sent["path"] == "/api/v1/entities/search" and sent["query"] == {"query": "Nvidia"} and usage.used_today("openfilings") == 1
    assert prov.search("NVDA")[0].provider_symbol == "us:NVDA" and fake.requests[-1]["query"] == {"ticker": "NVDA"}
    assert prov.search("jp:7203")[0].provider_symbol == "jp:7203" and fake.requests[-1]["query"] == {"ticker": "7203", "market_id": "jp"}
    assert prov.search("US67066G1040")[0].name == "NVIDIA Corp" and fake.requests[-1]["query"] == {"isin": "US67066G1040"}
    assert usage.used_today("openfilings") == 4 and all(KEY_1 not in r["url"] for r in fake.requests)
    toyota = prov.search_detail("Toyota")                                                       # a name can match several issuers: the best one first, the rest after it
    assert toyota["ambiguous"] is True and [c.provider_symbol for c in toyota["profiles"]] == ["jp:7203", "jp:8015"]
    assert toyota["profiles"][0].name == "Toyota Motor Corporation" and toyota["profiles"][0].extra["also_listed"] == ["us:TM"]
    assert [c.provider_symbol for c in prov.search("Toyota", market="us")][:1] == ["us:TM"]       # the listing on the market asked for stands for the company
    assert prov.search("Deutsche Bank")[0].extra["has_financials"] is False                    # the directory lists markets that have no statements: the page says so before a credit is spent on them
    fake.skipped = ["jp", {"market_id": "HK"}, 7, ""]                                           # markets the directory did not search are passed on, whatever shape their entries have
    assert prov.search_detail("Nvidia")["markets_skipped"] == ["jp", "hk"]


def test_a_company_the_directory_does_not_know_is_an_empty_answer_and_costs_nothing(tmp_path):
    prov, fake, usage, _ = _provider(tmp_path)
    nothing = prov.search_detail("Zzyzx Holdings")
    assert nothing["profiles"] == [] and nothing["credits"] == 0 and usage.used_today("openfilings") == 0 and "no issuer" in nothing["message"]
    fake.empty_search_as_200 = True                                                             # an answer with no company in it is not charged differently from any other that was sent
    quiet = prov.search_detail("Zzyzx Holdings")
    assert quiet["profiles"] == [] and quiet["message"] == "No issuer matched" and quiet["credits"] == 1 and usage.used_today("openfilings") == 1
    with pytest.raises(ProviderError, match="say which market"):
        split_symbol("AAPL", None, None)


def test_the_account_test_reports_the_plan_and_the_markets_for_one_credit(tmp_path):
    prov, _, usage, _ = _provider(tmp_path, FakeOpenFilings(plan="pro"))
    result = prov.test()
    assert result["plan"] == "pro" and result["allowed_markets"] == ["us", "jp"] and result["with_filings"] == ["us", "jp"] and usage.used_today("openfilings") == 1


def test_each_kind_of_error_answer_is_told_apart(tmp_path):
    prov, fake, usage, pool = _provider(tmp_path)
    fake.unknown.add("ZZZZ")
    with pytest.raises(NotFound):
        prov.get_financials("us:ZZZZ")
    fake.pending.add("PEND")
    with pytest.raises(SyncPending) as pending:
        prov.get_financials("jp:PEND")
    assert pending.value.retry_after == 120
    assert usage.used_today("openfilings") == 0                                                # not found and still-collecting are not charged
    fake.broken = 1
    with pytest.raises(ProviderError, match="temporarily unavailable"):
        prov.get_company("us:AAPL")
    err = OpenFilingsProvider._interpret
    with pytest.raises(PlanError) as plan:
        err(403, {}, json.dumps({"detail": {"error": "upgrade_required", "feature": "financials for kr", "min_plan": "pro"}}).encode(), KEY_1)
    assert plan.value.min_plan == "pro" and "pro plan" in str(plan.value)
    with pytest.raises(PlanError):
        err(403, {}, b'{"detail": "forbidden"}', KEY_1)
    with pytest.raises(CreditsExhausted) as credits:
        err(402, {"X-Credits-Balance": "0", "X-Credits-Required": "10"}, b'{"detail": {"error": "credits_exhausted", "credit_balance": 0}}', KEY_1)
    assert credits.value.required == 10.0 and credits.value.balance == 0.0
    with pytest.raises(RateLimited) as monthly:
        err(429, {}, json.dumps({"detail": {"error": "quota_exceeded", "dimension": "monthly", "monthly_limit": 1000, "monthly_used": 1000}}).encode(), KEY_1)
    assert monthly.value.dimension == "monthly" and monthly.value.detail["monthly_used"] == 1000 and "1000" in str(monthly.value)
    with pytest.raises(ProviderError, match="not JSON"):
        err(200, {}, b"<html>", KEY_1)
    assert pool.slots()[0]["state"] == "ready"


def test_a_burst_limit_is_waited_out_briefly_and_then_given_up_on_without_blocking_the_key(tmp_path):
    waits: list[float] = []
    prov, fake, _, pool = _provider(tmp_path, sleep=waits.append)
    fake.bursts = 2
    assert prov.get_company("us:AAPL").ticker == "AAPL" and waits == [1.0, 1.0]
    fake.bursts = 5
    with pytest.raises(RateLimited) as info:
        prov.get_company("us:MSFT")
    assert info.value.dimension == "burst" and pool.slots()[0]["state"] == "ready"           # slow down is not "the day is over"


def test_a_daily_limit_answer_stops_the_pool_with_the_providers_own_counts_and_nothing_more_is_sent(tmp_path):
    prov, fake, _, pool = _provider(tmp_path, daily=1000)                                       # the local manager believes it has plenty; the provider says otherwise
    fake.daily = 5
    with pytest.raises(QuotaExhausted, match="daily allowance"):
        prov.get_financials("us:AAPL")
    sent = len(fake.requests)
    with pytest.raises(QuotaExhausted):
        prov.get_financials("us:MSFT")
    assert len(fake.requests) == sent and pool.slots()[0]["state"] == "spent_day"


def test_a_rejected_line_item_list_is_retried_once_for_everything(tmp_path):
    prov, fake, usage, _ = _provider(tmp_path)
    fake.reject_line_items_once = True
    result = prov.get_financials("us:AAPL", start_year=2020, end_year=2021)
    assert [r["query"]["line_items"] != "ALL" for r in fake.requests] == [True, False] and len(result.periods) == 2 and usage.used_today("openfilings") == 10


def test_quarterly_statements_are_refused_before_a_credit_is_spent(tmp_path):
    prov, fake, _, _ = _provider(tmp_path)
    with pytest.raises(PlanError, match="quarterly"):
        prov.get_financials("us:AAPL", period="quarter")
    assert fake.requests == [] and prov.periods_available == ("annual",)


def test_parsing_keeps_the_earliest_filing_per_period_drops_nulls_and_skips_what_is_not_annual():
    payload = {"entity": {"name": "Toyota", "market_id": "jp", "ticker": "7203", "isin": "JP3633400001"}, "currency_latest": "JPY", "periods": [
        {"fiscal_year": 2022, "fiscal_period": "FY", "period_end": "2022-03-31", "filing_id": "late", "filing_date": "2023-06-20", "currency": "JPY", "values": {"revenue": 99.0, "net_income": 5.0}},
        {"fiscal_year": 2022, "fiscal_period": "FY", "period_end": "2022-03-31", "filing_id": "first", "filing_date": "2022-06-24", "currency": "JPY", "values": {"revenue": 100.0, "net_income": None, "equity": 7}},
        {"fiscal_year": 2023, "fiscal_period": "Q2", "period_end": "2022-09-30", "filing_date": "2022-11-10", "values": {"revenue": 40.0}},
        {"fiscal_year": 2023, "fiscal_period": "FY", "period_end": None, "values": {"revenue": 1.0}},
        {"fiscal_year": 2023, "fiscal_period": "FY", "period_end": "2023-03-31", "filing_date": "2023-06-23T08:00:00Z", "values": {}},
        {"fiscal_year": 2024, "period_end": "2024-03-31", "filing_date": "2024-06-21", "values": {"revenue": 120.0, "flag": True}}]}
    periods, profile, skipped = parse_financials(payload)
    assert [(p.period_end, p.filing_date, p.provider_record_id) for p in periods] == [("2022-03-31", "2022-06-24", "first"), ("2024-03-31", "2024-06-21", None)]
    assert periods[0].values == {"revenue": 100.0, "equity": 7.0} and periods[0].reporting_currency == "JPY" and periods[1].values == {"revenue": 120.0}        # a bool is not a figure
    assert skipped == 3 and (profile.provider_symbol, profile.country, profile.isin) == ("jp:7203", "JP", "JP3633400001")
    assert parse_financials({"periods": []})[1] is None and of._label("fy") == "FY" and of._label("YTD") is None and of._label("q3") == "Q3"


class _Redirecting(http.server.BaseHTTPRequestHandler):
    seen: list[tuple[str, str | None]] = []

    def do_GET(self):                                                                          # noqa: N802
        type(self).seen.append((self.path, self.headers.get("X-API-Key")))
        if self.path.startswith("/api/v1/markets"):
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"plan": "free", "markets": []}')

    def log_message(self, *args):
        pass


def test_the_real_transport_does_not_follow_a_redirect_with_the_key_in_the_headers(tmp_path, monkeypatch):
    server = http.server.HTTPServer(("127.0.0.1", 0), _Redirecting)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(of, "_OPENER", request.build_opener(of._NoRedirect, request.ProxyHandler({})))
        pool, _ = _pool(tmp_path, name="r.db")
        prov = OpenFilingsProvider(pool, base_url=f"http://127.0.0.1:{server.server_address[1]}", sleep=lambda s: None)
        with pytest.raises(ProviderError, match="HTTP 302"):
            prov.test()
        assert _Redirecting.seen == [("/api/v1/markets", KEY_1)]                                # one request: the redirect target was never asked
    finally:
        server.shutdown()


# ------------------------------------------------------------------------------------------------------------------------ the sync with OpenFilings
def _service(tmp_path, fake=None, keys=(KEY_1,), clock=None, **allow):
    svc = FundamentalsService(tmp_path / "svc", clock=clock)
    pool = KeyPool("openfilings", [(f"OPENFILINGS_API_KEY_{i + 1}", k) for i, k in enumerate(keys)], svc.usage, svc.store.salt(),
                   Allowance("credits", allow.get("daily", 100), allow.get("monthly", 1000), "test"))
    fake = fake or FakeOpenFilings({k: "a" for k in keys})
    svc.register("openfilings", OpenFilingsProvider(pool, transport=fake, sleep=lambda s: None))
    return svc, fake


def test_registering_costs_nothing_and_a_first_sync_costs_eleven_credits_a_company(tmp_path):
    svc, fake = _service(tmp_path)
    ids = svc.add_companies("openfilings", ["AAPL", "jp:7203", "us:AAPL"])
    assert len(set(ids)) == 2 and fake.requests == [] and svc.usage.used_today("openfilings") == 0
    row = svc.store.get_security(ids[1])
    assert (row["ticker"], row["exchange"], row["country"], row["sector"]) == ("7203", "JP", "JP", None)
    plan = svc.plan("openfilings", ids, ("annual",))
    assert plan.summary()["cost"] == 22 and plan.summary()["unit"] == "credits" and [t.profile for t in plan.tasks] == [True, True]
    fit = svc.affordability("openfilings", plan)
    assert (fit["remaining_today"], fit["fit_today"], fit["days_needed"], fit["monthly_limit"]) == (100, 2, 1, 1000)
    report = svc.sync("openfilings", security_ids=ids, periods=("annual",), start_year=2015)
    assert report["fetched"] == 2 and report["cost"] == 22 and report["unit"] == "credits" and not report["failed"] and report["unmapped"] == []
    assert svc.usage.used_today("openfilings") == 22 and svc.usage.requests_today("openfilings") == 4
    row = svc.store.get_security(ids[0])
    assert row["sector"] == "Technology" and row["industry"] == "Computers" and row["company_name"] == "AAPL Corp" and row["isin"]
    counts = svc.store.counts()
    assert counts["with_financials"] == 2 and counts["annual_periods"] == 2 * 9 and counts["raw_payloads"] == 2
    periods = svc.store.periods([ids[0]])
    assert (periods["pit_quality"] == "filed").all() and periods["filing_date"].notna().all() and (periods["filing_date"] > periods["period_end"]).all()
    second = svc.sync("openfilings", security_ids=ids, periods=("annual",), start_year=2015)
    assert second["fetched"] == 0 and second["fresh"] == 2 and second["cost"] == 0 and svc.usage.used_today("openfilings") == 22


def test_a_search_costs_a_credit_is_kept_and_a_company_added_from_it_keeps_the_name_it_paid_for(tmp_path):
    clock = Clock()
    svc, fake = _service(tmp_path, clock=clock)
    fake.skipped = ["jp"]
    first = svc.search_companies("openfilings", "Nvidia")
    hit = first["results"][0]
    assert first["markets_skipped"] == ["jp"]
    assert (first["kind"], first["kind_word"], first["count"], first["credits"], first["cached"]) == ("name", "a company name", 1, 1, False)
    assert (hit["symbol"], hit["name"], hit["market"], hit["country"], hit["in_database"], hit["has_financials"], hit["plan_allows"]) == ("us:NVDA", "NVIDIA Corp", "us", "US", False, True, None)
    assert (first["remaining_today"], first["cost_to_fetch"], first["unit"], svc.usage.used_today("openfilings"), len(fake.requests)) == (99, 11, "credits", 1, 1)
    again = svc.search_companies("openfilings", "  nvidia ")                                    # the same words: answered from what was saved, for nothing
    assert again["cached"] and again["credits"] == 0 and len(fake.requests) == 1 and svc.usage.used_today("openfilings") == 1 and again["markets_skipped"] == ["jp"]
    assert svc.search_companies("openfilings", "Nvidia", refresh=True)["credits"] == 1 and len(fake.requests) == 2
    ids = svc.add_companies("openfilings", ["us:NVDA"])                                         # adding is free and keeps the name, the ISIN and the LEI the search paid for
    row = svc.store.get_security(ids[0])
    with svc.store.db.transaction() as conn:
        lei = conn.execute("SELECT lei FROM securities WHERE security_id = ?", (ids[0],)).fetchone()[0]
    assert (row["company_name"], row["isin"], lei, row["country"], row["sector"]) == ("NVIDIA Corp", "US67066G1040", "N" * 20, "US", None) and len(fake.requests) == 2
    after = svc.search_companies("openfilings", "Nvidia")
    assert after["results"][0]["in_database"] and after["results"][0]["security_id"] == ids[0] and after["results"][0]["periods"] == 0
    svc.sync("openfilings", security_ids=ids, periods=("annual",))
    assert svc.search_companies("openfilings", "Nvidia")["results"][0]["periods"] >= 8
    clock.advance(days=31)                                                                      # a saved answer is trusted for a month, then asked again
    spent = svc.usage.used_today("openfilings")
    assert svc.search_companies("openfilings", "Nvidia")["cached"] is False and svc.usage.used_today("openfilings") == spent + 1


def test_a_search_that_finds_nothing_is_free_and_is_not_asked_again_the_same_day(tmp_path):
    clock = Clock()
    svc, fake = _service(tmp_path, clock=clock)
    none = svc.search_companies("openfilings", "Zzyzx")
    assert none["count"] == 0 and none["credits"] == 0 and svc.usage.used_today("openfilings") == 0 and none["alternatives"] == ["symbol"] and len(fake.requests) == 1 and "no issuer" in none["message"]
    assert svc.search_companies("openfilings", "Zzyzx")["cached"] is True and len(fake.requests) == 1
    clock.advance(days=2)
    assert svc.search_companies("openfilings", "Zzyzx")["cached"] is False and len(fake.requests) == 2


def test_a_search_is_refused_without_a_key_or_with_nothing_to_look_for_and_never_leaves_the_allowance(tmp_path, monkeypatch):
    for name in [n for n in list(os.environ) if n.startswith("OPENFILINGS_")]:
        monkeypatch.delenv(name, raising=False)
    bare = FundamentalsService(tmp_path / "nokey2")
    with pytest.raises(MissingKey, match="no OPENFILINGS_API_KEY"):
        bare.search_companies("openfilings", "Nvidia")
    svc, fake = _service(tmp_path, daily=1)
    for bad, why in (("", "at least two characters"), ("a", "at least two characters"), ("x" * 121, "at most 120")):
        with pytest.raises(FundamentalsError, match=why):
            svc.search_companies("openfilings", bad)
    with pytest.raises(FundamentalsError, match="kind must be one of"):
        svc.search_companies("openfilings", "Nvidia", kind="isbn")
    assert fake.requests == []
    svc.search_companies("openfilings", "Nvidia")                                               # the one credit of the day
    with pytest.raises(QuotaExhausted, match="daily allowance is spent"):
        svc.search_companies("openfilings", "Apple")
    assert len(fake.requests) == 1


def test_the_plan_a_connection_test_reports_marks_the_markets_a_search_may_not_use(tmp_path):
    svc, _ = _service(tmp_path)
    assert svc.search_companies("openfilings", "Toyota")["results"][0]["plan_allows"] is None   # nothing is known about the plan before it is tested
    assert svc.test_provider("openfilings")["ok"] is True and svc.allowed_markets("openfilings") == ({"us", "jp"}, False)
    by_symbol = {r["symbol"]: r for r in svc.search_companies("openfilings", "Toyota")["results"]}
    assert by_symbol["jp:7203"]["plan_allows"] is True
    assert svc.search_companies("openfilings", "Deutsche Bank")["results"][0]["plan_allows"] is False


def test_a_company_whose_sector_is_known_skips_the_lookup(tmp_path):
    svc, _ = _service(tmp_path)
    sid = svc.add_companies("openfilings", ["AAPL"])[0]
    svc.store.enrich_security(sid, CompanyProfile("openfilings", "us:AAPL", "AAPL", sector="Technology"))
    plan = svc.plan("openfilings", [sid], ("annual",))
    assert plan.cost == 10 and not plan.tasks[0].profile


def test_the_run_stops_when_the_days_credits_are_spent_and_carries_on_the_next_day(tmp_path):
    clock = Clock()
    svc, fake = _service(tmp_path, clock=clock)
    ids = svc.add_companies("openfilings", [f"T{i:02d}" for i in range(12)])
    first = svc.sync("openfilings", security_ids=ids, start_year=2015)
    assert first["fetched"] == 9 and first["cost"] == 99 and "daily allowance is spent" in first["stopped"] and "3 companies remain" in first["stopped"]
    assert svc.pool("openfilings").remaining_today() == 1
    again = svc.sync("openfilings", security_ids=ids, start_year=2015)
    assert again["fetched"] == 0 and again["cost"] == 0 and "remain" in again["stopped"]
    fake.day += 1
    clock.advance(days=1)
    rest = svc.sync("openfilings", security_ids=ids, start_year=2015)
    assert rest["fetched"] == 3 and not rest["stopped"] and svc.store.counts()["with_financials"] == 12
    assert [h["credits"] for h in svc.usage.history("openfilings")] == [99, 33]


def test_a_budget_caps_one_run_and_the_standby_keys_are_not_used_to_get_past_the_allowance(tmp_path):
    svc, fake = _service(tmp_path, keys=(KEY_1, KEY_2), daily=30)
    ids = svc.add_companies("openfilings", ["A1", "A2", "A3", "A4"])
    capped = svc.sync("openfilings", security_ids=ids, budget=22, start_year=2015)
    assert capped["fetched"] == 2 and capped["cost"] == 22 and "budget of 22 credits" in capped["stopped"]
    more = svc.sync("openfilings", security_ids=ids, start_year=2015)
    assert more["fetched"] == 0 and "daily allowance is spent" in more["stopped"] and "does not rotate keys" in more["stopped"]
    assert {r["headers"]["X-API-Key"] for r in fake.requests} == {KEY_1}


def test_a_revoked_first_key_hands_over_to_the_second_during_a_sync(tmp_path):
    svc, fake = _service(tmp_path, FakeOpenFilings({KEY_2: "b"}), keys=(KEY_1, KEY_2))
    ids = svc.add_companies("openfilings", ["AAPL"])
    report = svc.sync("openfilings", security_ids=ids, start_year=2018)
    assert report["fetched"] == 1 and [s["state"] for s in svc.pool("openfilings").slots()] == ["invalid", "ready"]
    assert fake.requests[0]["headers"]["X-API-Key"] == KEY_1 and all(r["headers"]["X-API-Key"] == KEY_2 for r in fake.requests[1:])


def test_when_every_key_is_rejected_the_run_says_so_and_stops(tmp_path):
    svc, _ = _service(tmp_path, FakeOpenFilings({}), keys=(KEY_1,))
    ids = svc.add_companies("openfilings", ["AAPL", "MSFT"])
    report = svc.sync("openfilings", security_ids=ids)
    assert report["fetched"] == 0 and "rejected as invalid or revoked" in report["stopped"]


def test_a_company_the_provider_does_not_know_is_recorded_and_not_asked_about_again_for_two_months(tmp_path):
    svc, fake = _service(tmp_path)
    fake.unknown.add("NOPE")
    ids = svc.add_companies("openfilings", ["NOPE", "AAPL"])
    report = svc.sync("openfilings", security_ids=ids, start_year=2018)
    assert report["not_found"] == ["us:NOPE"] and report["fetched"] == 1 and report["cost"] == 11           # the lookup that found nothing was not charged; the other company cost a lookup and a series
    assert svc.store.get_sync(ids[0], "openfilings", "financials:annual")["status"] == "not_found"
    assert svc.plan("openfilings", ids, ("annual",)).summary()["to_fetch"] == 0


def test_a_company_still_being_collected_is_asked_again_the_next_day_not_the_same_day(tmp_path):
    clock = Clock()
    svc, fake = _service(tmp_path, clock=clock)
    fake.pending.add("TOYO")
    ids = svc.add_companies("openfilings", ["jp:TOYO"])
    report = svc.sync("openfilings", security_ids=ids)
    assert report["pending"] == ["jp:TOYO"] and report["failed"] == [] and report["cost"] == 0
    assert svc.plan("openfilings", ids, ("annual",)).summary()["to_fetch"] == 0
    clock.advance(days=1)
    fake.pending.clear()
    assert svc.plan("openfilings", ids, ("annual",)).summary()["to_fetch"] == 1


def test_an_answer_that_spells_the_ticker_differently_does_not_create_a_second_company(tmp_path):
    svc, fake = _service(tmp_path)
    fake.entity_ticker = "0700.HK"
    ids = svc.add_companies("openfilings", ["hk:0700"])
    svc.sync("openfilings", security_ids=ids, start_year=2020)
    assert len(svc.store.securities()) == 1 and svc.store.get_security(ids[0])["ticker"] == "0700"


def test_testing_the_account_sets_the_allowance_from_its_plan_and_a_new_session_remembers_it(tmp_path, monkeypatch):
    for name in ("OPENFILINGS_PLAN", "OPENFILINGS_DAILY_LIMIT", "OPENFILINGS_MONTHLY_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    svc, _ = _service(tmp_path, FakeOpenFilings(plan="pro"))
    assert svc.pool("openfilings").allow.daily == 100                                          # until the account says otherwise, the free plan is assumed
    report = svc.test_provider("openfilings")
    assert report["ok"] and report["plan"] == "pro" and svc.pool("openfilings").plan == "pro" and svc.plan_of("openfilings") == "pro"
    fresh = FundamentalsService(tmp_path / "svc")
    assert fresh.pool("openfilings").allow.daily == 2500 and "reported by the account" in fresh.pool("openfilings").allow.source


def test_a_test_with_no_key_explains_what_to_do_and_a_test_that_fails_says_why(tmp_path, monkeypatch):
    for name in [n for n in list(os.environ) if n.startswith("OPENFILINGS_")]:
        monkeypatch.delenv(name, raising=False)
    svc = FundamentalsService(tmp_path / "nokey")
    result = svc.test_provider("openfilings")
    assert result["available"] is False and "OPENFILINGS_API_KEY" in result["reason"] and "Tools > Secrets" in result["reason"]
    with pytest.raises(MissingKey):
        svc.sync("openfilings", symbols=["AAPL"])
    broke, _ = _service(tmp_path, FakeOpenFilings(daily=0))
    out = broke.test_provider("openfilings")
    assert out["ok"] is False and "daily allowance" in out["error"]


def test_unsupported_periods_are_left_out_of_the_plan_with_a_note(tmp_path):
    svc, _ = _service(tmp_path)
    ids = svc.add_companies("openfilings", ["AAPL"])
    plan = svc.plan("openfilings", ids, ("annual", "quarter"))
    assert len(plan.tasks) == 1 and plan.tasks[0].period == "annual" and any("no quarter statements" in n for n in plan.notes)


def test_the_data_status_shows_credits_per_key_and_never_a_key(tmp_path):
    svc, _ = _service(tmp_path, keys=(KEY_1, KEY_2))
    ids = svc.add_companies("openfilings", ["AAPL"])
    svc.sync("openfilings", security_ids=ids, start_year=2020)
    status = svc.data_status()
    row = next(p for p in status["providers"] if p["name"] == "openfilings")
    assert (row["unit"], row["used_today"], row["daily_limit"], row["monthly_limit"], row["remaining_today"], row["cost_per_company"]) == ("credits", 11, 100, 1000, 89, 11)
    assert [(k["slot"], k["role"], k["used_today"]) for k in row["keys"]] == [("key 1", "active", 11), ("key 2", "standby", 0)]
    assert row["resets"]["daily"].endswith("00:00 UTC") and row["resets"]["monthly"].endswith("00:00 UTC") and len(row["recent"]) == 2 and not status["providers"][1]["keyed"]
    dump = json.dumps(status, default=str)
    assert KEY_1 not in dump and KEY_2 not in dump and "key_id" not in dump


# ------------------------------------------------------------------------------------------------------------------------ the sync with the simulated world
def _sim(tmp_path, world, **kw):
    return FundamentalsService(tmp_path, providers={"simulated": SimulatedProvider(world=world)}, **kw)


def test_the_simulated_world_is_the_same_every_time_and_keeps_its_accounting(world):
    again = SimulatedWorld(n_companies=40, start_year=2010, end_year=2017, seed=11)
    assert np.array_equal(world.annual["revenue"], again.annual["revenue"]) and np.array_equal(world.logret, again.logret, equal_nan=True)
    s = world.stocks
    assert np.allclose(s["total_assets"], s["total_liabilities"] + s["shareholders_equity"])
    assert np.allclose(world.annual["free_cash_flow"], world.annual["operating_cash_flow"] - world.annual["capital_expenditure"])
    assert world.truth["planted"]["delta_roic"] > 0 and world.truth["delisted"] >= 1
    assert (world.period_end[:, :, 3] < world.filing_annual).all() and (world.period_end[:, :, 0] < world.period_end[:, :, 3]).all()


def test_the_database_is_asked_first_and_a_second_sync_costs_nothing(tmp_path, world):
    svc = _sim(tmp_path, world)
    prov = svc.provider("simulated")
    ids = svc.add_companies("simulated", prov.symbols()[:6])
    assert svc.usage.used_today("simulated") == 0                                              # registering a company asks nobody
    plan = svc.plan("simulated", ids, ("annual", "quarter"))
    assert plan.summary()["to_fetch"] == 12 and plan.summary()["reasons"] == {"missing": 12} and plan.cost == 12
    first = svc.sync("simulated", security_ids=ids, periods=("annual", "quarter"))
    assert first["fetched"] == 12 and first["cost"] == 12 and not first["failed"]
    used = svc.usage.used_today("simulated")
    second = svc.sync("simulated", security_ids=ids, periods=("annual", "quarter"))
    assert second["fetched"] == 0 and second["fresh"] == 12 and second["cost"] == 0 and svc.usage.used_today("simulated") == used
    assert svc.store.counts()["raw_payloads"] == 12 and svc.store.counts()["with_financials"] == 6


def test_a_company_is_stale_only_when_its_next_filing_should_exist_and_a_week_has_passed_since_the_last_look(tmp_path, world):
    svc = _sim(tmp_path, world)
    ids = svc.add_companies("simulated", svc.provider("simulated").symbols()[:2])
    svc.sync("simulated", security_ids=ids, periods=("annual",))
    sid = ids[0]
    last_end = pd.Timestamp(svc.store.get_sync(sid, "simulated", "financials:annual")["last_period_end"])
    svc.store.set_sync(sid, "simulated", "financials:annual", last_checked="2000-01-01")
    assert svc.reason_to_sync(sid, "simulated", "annual", None, last_end + pd.Timedelta(days=365 + 100)) is None            # the next annual report is not due yet
    due = last_end + pd.Timedelta(days=365 + 151)
    assert svc.reason_to_sync(sid, "simulated", "annual", None, due) == "stale"
    svc.store.set_sync(sid, "simulated", "financials:annual", last_checked=str((due - pd.Timedelta(days=2)).date()))
    assert svc.reason_to_sync(sid, "simulated", "annual", None, due) is None                                                  # looked two days ago: not again so soon
    svc.store.set_sync(sid, "simulated", "financials:annual", earliest_period_end="2012-12-31", history_exhausted=0)
    assert svc.reason_to_sync(sid, "simulated", "annual", 2005, due) == "history"                                            # older years were asked for and nobody said there are none
    svc.store.set_sync(sid, "simulated", "financials:annual", history_exhausted=1)
    assert svc.reason_to_sync(sid, "simulated", "annual", 2005, due - pd.Timedelta(days=300)) is None


def test_when_the_allowance_runs_out_the_sync_stops_cleanly_and_the_rest_waits(tmp_path, world, monkeypatch):
    monkeypatch.setenv("SIMULATED_DAILY_LIMIT", "5")
    svc = _sim(tmp_path, world)
    ids = svc.add_companies("simulated", svc.provider("simulated").symbols()[:8])
    report = svc.sync("simulated", security_ids=ids, periods=("annual",))
    assert report["fetched"] == 5 and "daily allowance is spent" in report["stopped"] and "3 companies remain" in report["stopped"]
    assert svc.store.counts()["with_financials"] == 5


def test_a_run_can_be_capped_and_unknown_and_restricted_companies_are_recorded_not_retried(tmp_path, world):
    svc = _sim(tmp_path, world)
    ids = svc.add_companies("simulated", svc.provider("simulated").symbols()[:4])
    capped = svc.sync("simulated", security_ids=ids, periods=("annual",), budget=2)
    assert capped["fetched"] == 2 and "budget of 2 requests" in capped["stopped"]

    class Picky(SimulatedProvider):
        def get_financials(self, symbol, *a, **k):
            if symbol == self.symbols()[2]:
                raise NotFound("nothing")
            if symbol == self.symbols()[3]:
                raise PlanError("not in the plan")
            return super().get_financials(symbol, *a, **k)

    svc2 = FundamentalsService(tmp_path / "second", providers={"simulated": Picky(world=world)})
    ids2 = svc2.add_companies("simulated", svc2.provider("simulated").symbols()[:4])
    rep = svc2.sync("simulated", security_ids=ids2, periods=("annual",))
    assert rep["fetched"] == 2 and rep["not_found"] == [world.ticker[2]] and rep["failed"][0]["symbol"] == world.ticker[3]
    assert svc2.plan("simulated", ids2, ("annual",)).summary()["to_fetch"] == 1             # the missing company is not asked about again for two months; the failed one is retried
    assert svc2.store.get_sync(ids2[2], "simulated", "financials:annual")["status"] == "not_found"


def test_a_restatement_found_on_a_later_sync_is_a_new_vintage(tmp_path, world):
    svc = _sim(tmp_path, world)
    prov = svc.provider("simulated")
    alive = [str(t) for t, last in zip(world.ticker, world.last_month) if last >= len(world.months)][0]
    ids = svc.add_companies("simulated", [alive])
    svc.sync("simulated", security_ids=ids, periods=("annual",))
    before = svc.store.counts()["observations"]
    prov.restate(alive, int(world.years[-1]), 1.1)
    svc.store.set_sync(ids[0], "simulated", "financials:annual", last_checked="2000-01-01", last_period_end="2000-01-01")        # make the company due for a look
    rep = svc.sync("simulated", security_ids=ids, periods=("annual",))
    assert rep["vintages"] == 1 and svc.store.counts()["restated_vintages"] == 1 and svc.store.counts()["observations"] > before
    assert svc.store.quality_counts()["observations"].get("restated", 0) > 0


# ------------------------------------------------------------------------------------------------------------------------ quality
def test_the_quality_engine_finds_the_planted_unit_error_and_nothing_else_of_note(tmp_path, world):
    svc = _sim(tmp_path, world)
    prov = svc.provider("simulated")
    ids = svc.add_companies("simulated", prov.symbols())
    svc.sync("simulated", security_ids=ids, periods=("annual", "quarter"))
    bad_company, bad_year = world.unit_error
    flagged = svc.store.inspect(ids[bad_company])
    hit = flagged[flagged["quality_flag"].isin(quality.SEVERE)]
    assert set(hit["fiscal_year"]) == {bad_year} and {"identity_error", "suspicious_jump"} & set(hit["quality_flag"])
    everyone = svc.store.quality_counts()["observations"]
    severe = sum(v for k, v in everyone.items() if k in quality.SEVERE)
    assert 0 < severe <= 40 and everyone["ok"] > 20_000


def test_rerunning_the_quality_check_clears_a_flag_that_no_longer_applies(store):
    sid = store.upsert_security(CompanyProfile("simulated", "AAA", "AAA", "A", exchange="NYSE"))
    periods = [PeriodStatement("FY", f"{y}-12-31", {"total_assets": 100.0, "total_liabilities": 60.0, "shareholders_equity": 40.0, "revenue": 50.0}, filing_date=f"{y + 1}-03-01") for y in (2018, 2019, 2020)]
    periods[1].values["total_assets"] = 5000.0                                              # a mistake in the middle year
    store.store_financials(sid, "simulated", periods)
    assert quality.check_security(store, sid)["identity_error"] >= 1
    with store.db.transaction() as conn:                                                      # the provider corrects it (updated in place for this test)
        conn.execute("UPDATE fundamental_observations SET value = 100 WHERE metric_code = 'total_assets'")
    quality.check_security(store, sid)
    assert "identity_error" not in store.quality_counts()["observations"]
