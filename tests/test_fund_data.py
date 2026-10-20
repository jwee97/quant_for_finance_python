"""The fundamentals data foundation: the vocabulary, the database, the key pool, the providers and the database-first sync, and the quality engine, against a simulated world whose truth is known."""

from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import canonical, quality
from src.fundamentals.errors import InvalidKey, MissingKey, NotFound, PlanError, ProviderError, QuotaExhausted, RateLimited
from src.fundamentals.keys import KeyPool, discover
from src.fundamentals.providers.fmp import FMPProvider
from src.fundamentals.providers.openfilings import OpenFilingsProvider, clean_environment, parse_financials
from src.fundamentals.providers.simulated import SimulatedProvider, SimulatedWorld
from src.fundamentals.schema import Database, initialise
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement
from src.fundamentals.usage import UsageLog


@pytest.fixture(scope="module")
def world():
    return SimulatedWorld(n_companies=40, start_year=2010, end_year=2017, seed=11)


@pytest.fixture()
def store(tmp_path):
    return FundamentalsStore(tmp_path / "f.db")


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


def test_fmp_signs_are_made_positive_and_a_proceed_is_not_mistaken_for_a_spend():
    values = {"cashflow:capitalExpenditure": -120.0, "cashflow:commonStockRepurchased": -30.0, "cashflow:acquisitionsNet": 15.0, "cashflow:commonDividendsPaid": -10.0, "income:revenue": 1000.0,
              "income:costOfRevenue": 600.0, "income:interestExpense": 5.0, "cashflow:netCashProvidedByOperatingActivities": 200.0, "balance:shortTermDebt": 20.0, "balance:longTermDebt": 80.0,
              "balance:cashAndCashEquivalents": 50.0, "income:somethingNew": 1.0}
    obs, unmapped = canonical.normalise_values("fmp", values)
    assert obs["capital_expenditure"]["value"] == 120.0 and obs["share_repurchases"]["value"] == 30.0 and obs["dividends_paid"]["value"] == 10.0
    assert "acquisitions" not in obs                                                      # a positive "acquisitions" figure is a disposal's proceeds, not money spent
    assert obs["gross_profit"] == {"value": 400.0, "source_metric": "derived: revenue - cost_of_revenue", "flag": "derived"}
    assert obs["free_cash_flow"]["value"] == 80.0 and obs["total_debt"]["value"] == 100.0 and obs["cash"]["flag"] == "derived"
    assert unmapped == ["income:somethingNew"]


def test_openfilings_finance_costs_are_marked_as_an_approximation_of_interest_expense():
    obs, _ = canonical.normalise_values("openfilings", {"revenue": 10.0, "finance_costs": -2.0, "net_income_loss": 1.0})
    assert obs["interest_expense"]["value"] == 2.0 and obs["interest_expense"]["flag"] == "estimated"
    with pytest.raises(ValueError, match="no field mapping"):
        canonical.normalise_values("nobody", {})


# ------------------------------------------------------------------------------------------------------------------------ the security master and the store
def test_the_same_ticker_on_two_exchanges_is_two_securities_and_a_provider_cannot_blank_a_field(store):
    a = store.upsert_security(CompanyProfile("fmp", "D05.SI", "D05", "DBS Group", exchange="SGX", market="SGX", country="SG", currency="SGD", sector="Financials"))
    b = store.upsert_security(CompanyProfile("fmp", "D05.XX", "D05", "Another D05", exchange="NYSE", market="NYSE", country="US", currency="USD"))
    assert a != b
    again = store.upsert_security(CompanyProfile("simulated", "D05", "D05", "DBS Group", exchange="SGX", market="SGX", sector=None, industry="Banks"))
    assert again == a
    row = store.get_security(a)
    assert row["sector"] == "Financials" and row["industry"] == "Banks" and row["country"] == "SG"       # kept what another provider supplied, added what it lacked
    hits = store.search("D05")
    assert set(hits["exchange"]) == {"SGX", "NYSE"} and {"country", "company_name"} <= set(hits.columns)


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
    first, new1 = store.save_raw("fmp", None, "financials:annual", {"symbol": "AAA"}, payload)
    second, new2 = store.save_raw("fmp", None, "financials:annual", {"symbol": "AAA"}, dict(reversed(list(payload.items()))))
    assert new1 and not new2 and first == second and store.load_raw(first) == payload
    assert store.counts()["raw_payloads"] == 1


# ------------------------------------------------------------------------------------------------------------------------ keys and the quota
def test_keys_are_found_in_every_form_once_each_and_without_the_wrapping_a_paste_adds(monkeypatch, tmp_path):
    for name in [n for n in list(os.environ) if n.startswith("FMP_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FMP_API_KEYS", "alpha1111111, beta2222222222;\n gamma333333333")
    monkeypatch.setenv("FMP_API_KEY", '"delta4444444444"')
    monkeypatch.setenv("FMP_API_KEY_2", "FMP_API_KEY_2=epsilon55555555")
    monkeypatch.setenv("FMP_API_KEY_3", "alpha1111111")                                  # listed twice
    monkeypatch.setenv("FMP_API_KEY_4", "<your key>")                                     # a placeholder
    keys, notes = discover("fmp", tmp_path)
    assert [v for v, _ in keys] == ["FMP_API_KEYS", "FMP_API_KEYS", "FMP_API_KEYS", "FMP_API_KEY", "FMP_API_KEY_2"]
    assert [k for _, k in keys] == ["alpha1111111", "beta2222222222", "gamma333333333", "delta4444444444", "epsilon55555555"]
    assert any("counted once" in n for n in notes) and not any("alpha1111111" in n or "delta4444444444" in n for n in notes)
    (tmp_path / ".env").write_text("FMP_API_KEY_9=zeta66666666\n")
    assert discover("fmp", tmp_path)[0][-1] == ("FMP_API_KEY_9", "zeta66666666")


def test_five_keys_give_five_allowances_and_the_one_with_most_left_is_used_first(tmp_path):
    usage = UsageLog(Database(tmp_path / "u.db"))
    pool = KeyPool("fmp", [(f"FMP_API_KEY_{i}", f"key-number-{i}-0000") for i in range(1, 6)], usage, "salt", limit=3)
    used: list[str] = []
    for _ in range(15):
        pool.call(lambda key: used.append(key) or 1, "income-statement", "AAA")
    assert set(used) == {f"key-number-{i}-0000" for i in range(1, 6)} and all(used.count(k) == 3 for k in set(used))
    assert pool.remaining_today() == 0 and {s["state"] for s in pool.slots()} == {"exhausted"}
    with pytest.raises(QuotaExhausted, match="out for today"):
        pool.call(lambda key: 1, "income-statement", "AAA")
    again = KeyPool("fmp", [(f"FMP_API_KEY_{i}", f"key-number-{i}-0000") for i in range(1, 6)], usage, "salt", limit=3)       # a restart: the counts are in the database
    assert again.remaining_today() == 0 and usage.used_today("fmp") == 15


def test_a_rate_limit_moves_on_to_the_next_key_an_invalid_key_is_set_aside_and_a_plan_error_stops_at_once(tmp_path):
    usage = UsageLog(Database(tmp_path / "u.db"))
    pool = KeyPool("fmp", [("A", "aaaaaaaaaaaaa"), ("B", "bbbbbbbbbbbbb"), ("C", "ccccccccccccc")], usage, "s", limit=100)
    seen: list[str] = []

    def fn(key):
        seen.append(key[0])
        if key[0] == "a":
            raise RateLimited("429")
        if key[0] == "b":
            raise InvalidKey("401")
        return "ok"

    assert pool.call(fn, "profile", "X") == "ok" and seen == ["a", "b", "c"]
    states = {s["slot"]: s["state"] for s in pool.slots()}
    assert states == {"key 1": "exhausted", "key 2": "invalid", "key 3": "ready"}
    seen.clear()
    pool.call(fn, "profile", "X")
    assert seen == ["c"]                                                                  # the spent and the invalid keys are not tried again today
    plan_pool = KeyPool("fmp", [("A", "aaaaaaaaaaaaa"), ("B", "bbbbbbbbbbbbb")], usage, "s2", limit=100)
    calls = []

    def restricted(key):
        calls.append(key)
        raise PlanError("premium endpoint")

    with pytest.raises(PlanError):
        plan_pool.call(restricted, "income-statement", "X")
    assert len(calls) == 1                                                                # every key on the plan would say the same: do not spend them all


def test_no_key_material_reaches_the_database_or_the_status_lines(tmp_path):
    usage = UsageLog(Database(tmp_path / "u.db"))
    secret = "SECRETKEYVALUE123456"
    pool = KeyPool("fmp", [("FMP_API_KEY", secret)], usage, "salt", limit=10)
    pool.call(lambda key: 1, "profile", "AAA")
    with pytest.raises(ProviderError):
        pool.call(lambda key: (_ for _ in ()).throw(ProviderError("boom")), "profile", "AAA")
    dump = json.dumps(pool.slots()) + json.dumps(usage.history("fmp"))
    with sqlite3.connect(tmp_path / "u.db") as conn:
        dump += json.dumps([list(r) for r in conn.execute("SELECT * FROM api_usage")])
    assert secret not in dump and secret[:6] not in dump and "key_id" in json.dumps(pool.slots())
    with pytest.raises(MissingKey, match="FMP_API_KEY"):
        KeyPool("fmp", [], usage).call(lambda key: 1, "profile")


def test_a_provider_that_needs_no_key_still_has_a_daily_budget(tmp_path):
    usage = UsageLog(Database(tmp_path / "u.db"))
    pool = KeyPool.without_key("openfilings", usage, limit=2)
    pool.call(lambda key: 1, "search", "Nokia")
    pool.call(lambda key: 1, "financials", "Nokia")
    with pytest.raises(QuotaExhausted):
        pool.call(lambda key: 1, "financials", "Nokia")
    assert usage.summary("openfilings", pool.slots()).requests == 2 and pool.slots()[0]["slot"] == "(no key needed)"


# ------------------------------------------------------------------------------------------------------------------------ the providers
def _fmp_rows(symbol: str, n: int, period: str = "FY"):
    income = [{"date": f"{2023 - i}-12-31", "symbol": symbol, "reportedCurrency": "SGD", "filingDate": f"{2024 - i}-02-20", "acceptedDate": f"{2024 - i}-02-20 06:00:00", "fiscalYear": str(2023 - i),
               "period": period, "revenue": 1000.0 + i, "costOfRevenue": 600.0, "netIncome": 90.0, "eps": 0.9, "epsDiluted": 0.88, "weightedAverageShsOutDil": 100.0} for i in range(n)]
    balance = [{"date": f"{2023 - i}-12-31", "symbol": symbol, "period": period, "fiscalYear": str(2023 - i), "totalAssets": 5000.0, "totalLiabilities": 3000.0, "totalStockholdersEquity": 2000.0,
                "inventory": 40.0, "totalDebt": 700.0} for i in range(n)]
    cash = [{"date": f"{2023 - i}-12-31", "symbol": symbol, "period": period, "fiscalYear": str(2023 - i), "netCashProvidedByOperatingActivities": 150.0, "capitalExpenditure": -50.0,
             "inventory": -3.0, "commonStockRepurchased": -10.0} for i in range(n)]
    return income, balance, cash


class FakeFMP:
    """A stand-in transport that behaves like FMP: statements by endpoint, a key check, a limit and an error shaped like the real one."""

    def __init__(self, good_keys=None, limit_per_key=1000):
        self.calls: list[str] = []
        self.good = set(good_keys or [])
        self.limit = limit_per_key
        self.per_key: dict[str, int] = {}

    def __call__(self, url: str, timeout: float):
        from urllib import parse
        parts = parse.urlparse(url)
        q = dict(parse.parse_qsl(parts.query))
        key = q.get("apikey", "")
        self.calls.append(parts.path.rsplit("/", 1)[-1])
        if self.good and key not in self.good:
            return 401, json.dumps({"Error Message": f"Invalid API KEY {key}."}).encode()
        self.per_key[key] = self.per_key.get(key, 0) + 1
        if self.per_key[key] > self.limit:
            return 429, json.dumps({"Error Message": "Limit Reach. Please upgrade your plan"}).encode()
        path = parts.path.rsplit("/", 1)[-1]
        sym = q.get("symbol", "")
        if sym == "NOPE":
            return 200, b"[]"
        n = int(q.get("limit", 3))
        income, balance, cash = _fmp_rows(sym, n, "FY" if q.get("period") == "annual" else "Q4")
        if path == "profile":
            return 200, json.dumps([{"symbol": sym, "companyName": "DBS Group", "currency": "SGD", "exchange": "SES", "sector": "Financials", "industry": "Banks", "country": "sg", "isActivelyTrading": True}]).encode()
        if path == "premium-only":
            return 402, b'{"Error Message": "Special Endpoint : This endpoint is only for premium members."}'
        return 200, json.dumps({"income-statement": income, "balance-sheet-statement": balance, "cash-flow-statement": cash}[path]).encode()


def test_fmp_merges_the_three_statements_and_reads_the_filing_dates(tmp_path):
    usage = UsageLog(Database(tmp_path / "u.db"))
    prov = FMPProvider(KeyPool("fmp", [("FMP_API_KEY", "goodkey1234567")], usage, "s", limit=100), transport=FakeFMP({"goodkey1234567"}))
    company = prov.get_company("D05.SI")
    assert (company.ticker, company.exchange, company.country, company.sector) == ("D05", "SGX", "SG", "Financials")
    result = prov.get_financials("D05.SI", period="annual", limit=3)
    assert result.requests == 3 and len(result.periods) == 3 and result.exhausted is False
    newest = result.periods[-1]
    assert newest.period_end == "2023-12-31" and newest.filing_date == "2024-02-20" and newest.accepted_date == "2024-02-20" and newest.reporting_currency == "SGD"
    obs, unmapped = canonical.normalise_values("fmp", newest.values)
    assert obs["capital_expenditure"]["value"] == 50.0 and obs["inventory"]["value"] == 40.0          # the balance-sheet level, not the cash-flow change of -3
    assert obs["shareholders_equity"]["value"] == 2000.0 and obs["free_cash_flow"]["value"] == 100.0
    assert usage.used_today("fmp") == 4                                                      # one profile and three statements, each counted


def test_fmp_error_answers_are_told_apart_and_the_key_never_appears_in_a_message(tmp_path):
    usage = UsageLog(Database(tmp_path / "u.db"))
    fake = FakeFMP({"goodkey1234567"}, limit_per_key=3)
    prov = FMPProvider(KeyPool("fmp", [("A", "wrongkey7654321"), ("B", "goodkey1234567")], usage, "s", limit=100), transport=fake)
    assert prov.get_company("D05.SI").ticker == "D05"                                       # the wrong key is set aside, the right one answers
    assert {s["slot"]: s["state"] for s in prov.pool.slots()} == {"key 1": "invalid", "key 2": "ready"}
    with pytest.raises(QuotaExhausted) as info:
        for _ in range(5):
            prov.get_company("D05.SI")
    assert "goodkey1234567" not in str(info.value) and "wrongkey7654321" not in str(info.value)
    none = FMPProvider(None)
    assert none.available()[0] is False and "FMP_API_KEY" in none.available()[1]
    with pytest.raises(MissingKey):
        none.get_company("X")
    ok = FMPProvider(KeyPool("fmp", [("A", "goodkey1234567")], UsageLog(Database(tmp_path / "v.db")), "s2", limit=100), transport=FakeFMP({"goodkey1234567"}))
    with pytest.raises(PlanError, match="plan"):
        ok._get("premium-only", {})


BRIDGE_ANSWER = {"company_id": "esef:fi:nokia", "filings": [
    {"id": "f2", "filing_date": "2024-03-01", "period_end": "2023-12-31", "financials": {"statements": [
        {"statement_type": "income_statement", "currency": "EUR", "line_items": [
            {"code": "revenue", "values": [{"period": {"end_date": "2023-12-31", "start_date": "2023-01-01", "kind": "duration"}, "value": "1200.5"},
                                           {"period": {"end_date": "2022-12-31", "start_date": "2022-01-01", "kind": "duration"}, "value": "1100"},
                                           {"period": {"end_date": "2023-09-30", "start_date": "2023-07-01", "kind": "duration"}, "value": "300"}]},
            {"code": "net_income_loss", "values": [{"period": {"end_date": "2023-12-31", "start_date": "2023-01-01", "kind": "duration"}, "value": "80"}]}]},
        {"statement_type": "balance_sheet", "currency": "EUR", "line_items": [
            {"code": "total_assets", "values": [{"period": {"end_date": "2023-12-31", "kind": "instant"}, "value": "5000"}, {"period": {"end_date": "2022-12-31", "kind": "instant"}, "value": "4800"}]}]}]}},
    {"id": "f1", "filing_date": "2023-03-02", "period_end": "2022-12-31", "financials": {"statements": [
        {"statement_type": "income_statement", "currency": "EUR", "line_items": [
            {"code": "revenue", "values": [{"period": {"end_date": "2022-12-31", "start_date": "2022-01-01", "kind": "duration"}, "value": "1090"}]}]}]}},
    {"id": "f0", "filing_date": "2022-03-02", "period_end": "2021-12-31", "financials": None, "extraction_error": "ValueError: no tagged data"}]}


def test_openfilings_answers_become_annual_periods_with_the_date_of_the_filing_that_reported_them():
    periods = {p.period_end: p for p in parse_financials(BRIDGE_ANSWER)}
    assert sorted(periods) == ["2022-12-31", "2023-12-31"]                                  # the quarter inside the annual accounts is not a financial year
    y23, y22 = periods["2023-12-31"], periods["2022-12-31"]
    assert y23.values["revenue"] == 1200.5 and y23.filing_date == "2024-03-01" and y23.reporting_currency == "EUR" and y23.fiscal_period == "FY"
    # the filing about 2022 beats the later filing's comparative figure (1090, not 1100) and gives the earlier date; nothing is borrowed from the later filing, which would not have been public then
    assert y22.values == {"revenue": 1090.0} and y22.filing_date == "2023-03-02"
    obs, _ = canonical.normalise_values("openfilings", y23.values)
    assert obs["revenue"]["value"] == 1200.5 and obs["total_assets"]["value"] == 5000.0 and obs["net_income"]["value"] == 80.0


def test_the_bridge_runs_in_a_clean_environment_and_a_missing_openfilings_is_explained(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "secret-fmp-key")
    monkeypatch.setenv("ITICK_API_KEY", "secret-itick-key")
    monkeypatch.setenv("EDINET_API_KEY", "edinet-key-needed")
    monkeypatch.setenv("OPENFILINGS_DATA_DIR", "/tmp/of")
    env = clean_environment()
    assert "FMP_API_KEY" not in env and "ITICK_API_KEY" not in env and env["EDINET_API_KEY"] == "edinet-key-needed" and env["OPENFILINGS_DATA_DIR"] == "/tmp/of"
    calls = []

    def runner(cmd, timeout):
        calls.append(cmd[2])
        if cmd[2] == "probe":
            return 2, json.dumps({"error": "openfilings is not installed in this interpreter", "detail": "No module named 'openfilings'"}), ""
        raise AssertionError("not reached")

    prov = OpenFilingsProvider(python="/usr/bin/python3", runner=runner)
    ok, why = prov.available()
    assert not ok and "OPENFILINGS_PYTHON" in why and "Python 3.11" in why
    with pytest.raises(ProviderError, match="not available"):
        prov.search("Nokia")
    assert calls == ["probe"]                                                              # asked once, remembered


def test_openfilings_search_and_financials_go_through_the_bridge():
    def runner(cmd, timeout):
        if cmd[2] == "probe":
            return 0, json.dumps({"service": ["search_companies"]}), ""
        if cmd[2] == "search":
            return 0, json.dumps({"companies": [{"id": "esef:fi:nokia", "name": "Nokia Oyj", "market": "FI", "country_code": "FI", "ticker": "NOKIA", "lei": "L1"}]}), ""
        return 0, json.dumps(BRIDGE_ANSWER), ""

    prov = OpenFilingsProvider(python="py", runner=runner)
    assert prov.available() == (True, "")
    company = prov.get_company("Nokia")
    assert (company.provider_symbol, company.ticker, company.country, company.name) == ("esef:fi:nokia", "NOKIA", "FI", "Nokia Oyj")
    result = prov.get_financials("esef:fi:nokia", limit=8)
    assert len(result.periods) == 2 and result.exhausted is True and result.raw["company_id"] == "esef:fi:nokia"
    with pytest.raises(ProviderError, match="not JSON"):
        OpenFilingsProvider(python="py", runner=lambda cmd, t: (0, "garbage", "") if cmd[2] != "probe" else (0, json.dumps({"service": []}), "")).search("x")


# ------------------------------------------------------------------------------------------------------------------------ the sync
def _service(tmp_path, world, **kw):
    svc = FundamentalsService(tmp_path, providers={"simulated": SimulatedProvider(world=world)}, **kw)
    return svc


def test_the_simulated_world_is_the_same_every_time_and_keeps_its_accounting(world):
    again = SimulatedWorld(n_companies=40, start_year=2010, end_year=2017, seed=11)
    assert np.array_equal(world.annual["revenue"], again.annual["revenue"]) and np.array_equal(world.logret, again.logret, equal_nan=True)
    s = world.stocks
    assert np.allclose(s["total_assets"], s["total_liabilities"] + s["shareholders_equity"])
    assert np.allclose(world.annual["free_cash_flow"], world.annual["operating_cash_flow"] - world.annual["capital_expenditure"])
    assert world.truth["planted"]["delta_roic"] > 0 and world.truth["delisted"] >= 1
    assert (world.period_end[:, :, 3] < world.filing_annual).all() and (world.period_end[:, :, 0] < world.period_end[:, :, 3]).all()


def test_the_database_is_asked_first_and_a_second_sync_costs_nothing(tmp_path, world):
    svc = _service(tmp_path, world)
    prov = svc.provider("simulated")
    ids = svc.add_companies("simulated", prov.symbols()[:6])
    plan = svc.plan("simulated", ids, ("annual", "quarter"))
    assert plan.summary()["to_fetch"] == 12 and plan.summary()["reasons"] == {"missing": 12}
    first = svc.sync("simulated", security_ids=ids, periods=("annual", "quarter"))
    assert first["fetched"] == 12 and first["requests"] == 12 and not first["failed"]
    used = svc.usage.used_today("simulated")
    second = svc.sync("simulated", security_ids=ids, periods=("annual", "quarter"))
    assert second["fetched"] == 0 and second["fresh"] == 12 and second["requests"] == 0 and svc.usage.used_today("simulated") == used
    assert svc.store.counts()["raw_payloads"] == 12 and svc.store.counts()["with_financials"] == 6


def test_a_company_is_stale_only_when_its_next_filing_should_exist_and_a_week_has_passed_since_the_last_look(tmp_path, world):
    svc = _service(tmp_path, world)
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


def test_when_the_allowance_runs_out_the_sync_stops_cleanly_and_the_rest_waits_for_tomorrow(tmp_path, world, monkeypatch):
    monkeypatch.setenv("SIMULATED_DAILY_LIMIT", "5")
    svc = _service(tmp_path, world)
    ids = svc.add_companies("simulated", svc.provider("simulated").symbols()[:3])           # three lookups of the five allowed
    report = svc.sync("simulated", security_ids=ids, periods=("annual",))
    assert report["fetched"] == 2 and "out for today" in report["stopped"] and "remain for tomorrow" in report["stopped"]
    assert svc.store.counts()["with_financials"] == 2


def test_requests_for_one_run_can_be_capped_and_unknown_and_restricted_companies_are_recorded_not_retried(tmp_path, world):
    svc = _service(tmp_path, world)
    ids = svc.add_companies("simulated", svc.provider("simulated").symbols()[:4])
    capped = svc.sync("simulated", security_ids=ids, periods=("annual",), max_requests=2)
    assert capped["fetched"] == 2 and "limit of 2 requests" in capped["stopped"]

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
    svc = _service(tmp_path, world)
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
    svc = _service(tmp_path, world)
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
