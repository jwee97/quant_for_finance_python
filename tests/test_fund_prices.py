"""Prices and exchange rates for the companies in the database: the Yahoo symbol a market and a ticker make, prices fetched in batches and stored both unadjusted and adjusted, the units London and Tel Aviv
quote in, a symbol remembered once it works, a company that returns nothing reported with its reason, exchange rates turned into units per dollar, and prices read from a file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import prices as P
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.store import CompanyProfile, FundamentalsStore


def company(store: FundamentalsStore, market: str, ticker: str, country: str, currency: str | None = None) -> int:
    return store.upsert_security(CompanyProfile("openfilings", f"{market}:{ticker}", ticker, f"{ticker} Inc", exchange=market.upper(), market=market.upper(), country=country, currency=currency))


@pytest.fixture()
def store(tmp_path):
    return FundamentalsStore(tmp_path / "f.db")


def bars(n: int = 30, start: str = "2020-01-01", close: float = 100.0, adj: float = 90.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"close": close + np.arange(n, dtype=float), "adj_close": adj + np.arange(n, dtype=float), "volume": 1000.0}, index=idx)


def test_a_symbol_is_the_ticker_with_the_suffix_of_its_market():
    assert P.yahoo_symbol("aapl", "us") == ("AAPL", 1.0) and P.yahoo_symbol("7203", "jp") == ("7203.T", 1.0)
    assert P.yahoo_symbol("700", "hk") == ("0700.HK", 1.0) and P.yahoo_symbol("5930", "kr") == ("005930.KS", 1.0)
    assert P.yahoo_symbol("600519", "cn")[0] == "600519.SS" and P.yahoo_symbol("000001", "cn")[0] == "000001.SZ" and P.yahoo_symbol("300750", "cn")[0] == "300750.SZ"
    assert P.yahoo_symbol("HSBA", "uk") == ("HSBA.L", 100.0) and P.yahoo_symbol("TEVA", "il") == ("TEVA.TA", 100.0)
    assert P.yahoo_symbol("SAP", "eu", "DE") == ("SAP.DE", 1.0) and P.yahoo_symbol("ASML", "eu", "NL")[0] == "ASML.AS" and P.yahoo_symbol("VOW3", None, None)[0] == "VOW3"
    with pytest.raises(FundamentalsError, match="country"):
        P.yahoo_symbol("SAP", "eu", None)
    with pytest.raises(FundamentalsError, match="no Yahoo suffix"):
        P.yahoo_symbol("X", "zz")


def test_prices_are_stored_unadjusted_and_adjusted_in_pounds_not_pence_and_the_symbol_is_remembered(store):
    us, uk, jp = company(store, "us", "AAA", "US", "USD"), company(store, "uk", "BBB", "GB", "GBP"), company(store, "jp", "7203", "JP", "JPY")
    seen = []

    def fetch(symbols, start):
        seen.append((list(symbols), start))
        return {s: bars() for s in symbols}

    report = P.load_prices(store, fetch=fetch, start="2020-01-01")
    assert report["loaded"] == 3 and report["rows"] == 90 and not report["failed"] and seen == [(["AAA", "BBB.L", "7203.T"], "2020-01-01")]
    close, adj, volume = store.closes([us, uk, jp]), store.prices([us, uk, jp]), store.volumes([us, uk, jp])
    assert close[us].iloc[0] == pytest.approx(100.0) and close[uk].iloc[0] == pytest.approx(1.0) and adj[uk].iloc[0] == pytest.approx(0.9) and adj[jp].iloc[3] == pytest.approx(93.0)
    assert volume[us].iloc[0] == 1000.0 and any("pence" in n for n in report["notes"]) and any("delisted" in n for n in report["notes"])
    ids = store.identifiers(uk)
    assert ((ids["provider"] == "yahoo") & (ids["id_type"] == "price_symbol") & (ids["provider_id"] == "BBB.L")).any()
    assert P.price_symbol(store, store.get_security(uk)) == ("BBB.L", 100.0, "saved") and P.price_symbol(store, store.get_security(us))[2] == "saved"


def test_a_company_that_returns_nothing_is_reported_with_the_reason_and_the_rest_are_kept(store):
    ok, gone = company(store, "us", "AAA", "US"), company(store, "us", "GONE", "US")

    def fetch(symbols, start):
        return {"AAA": bars(), "GONE": "no data found (the symbol may be wrong, or the company delisted)"}

    report = P.load_prices(store, fetch=fetch)
    assert report["loaded"] == 1 and report["failed"] == [{"security_id": gone, "ticker": "GONE", "symbol": "GONE", "reason": "no data found (the symbol may be wrong, or the company delisted)"}]
    assert store.prices([ok]).shape[0] == 30 and store.prices([gone]).empty
    empty = P.load_prices(store, [gone], fetch=lambda s, t: {"GONE": pd.DataFrame(columns=["close"])})
    assert empty["loaded"] == 0 and empty["failed"][0]["reason"] == "no answer"


def test_a_symbol_the_rule_gets_wrong_can_be_set_by_hand_and_is_then_used(store):
    uk = company(store, "uk", "RR", "GB")
    P.set_price_symbol(store, uk, "rr.l")
    asked = []
    P.load_prices(store, fetch=lambda symbols, start: (asked.append(list(symbols)), {s: bars() for s in symbols})[1])
    assert asked == [["RR.L"]]
    P.set_price_symbol(store, uk, "RR-B.L")
    assert P.price_symbol(store, store.get_security(uk))[0] == "RR-B.L"
    for bad in ("", "has space", "x" * 30, "a;b"):
        with pytest.raises(FundamentalsError, match="price symbol"):
            P.set_price_symbol(store, uk, bad)


def test_an_update_starts_just_before_the_last_stored_price_and_a_refresh_starts_again(store):
    us = company(store, "us", "AAA", "US")
    P.load_prices(store, fetch=lambda symbols, start: {s: bars(30, "2020-01-01") for s in symbols}, start="2019-01-01")
    last = store.prices([us]).index[-1]
    asked = []

    def fetch(symbols, start):
        asked.append(start)
        return {s: bars(40, "2020-01-01", close=200.0, adj=190.0) for s in symbols}

    P.load_prices(store, fetch=fetch, start="2019-01-01")
    assert asked == [str((last - pd.Timedelta(days=P.OVERLAP_DAYS)).date())]
    assert store.closes([us]).iloc[-1, 0] == pytest.approx(239.0) and store.closes([us]).shape[0] == 40
    P.load_prices(store, fetch=fetch, start="2019-01-01", refresh=True)
    assert asked[-1] == "2019-01-01"


def test_the_batches_are_no_larger_than_the_limit_and_a_stop_ends_the_run_between_them(store, monkeypatch):
    for i in range(5):
        company(store, "us", f"T{i}", "US")
    monkeypatch.setattr(P, "MAX_BATCH", 2)
    batches = []
    stop = {"now": False}

    def fetch(symbols, start):
        batches.append(list(symbols))
        stop["now"] = len(batches) >= 2
        return {s: bars() for s in symbols}

    report = P.load_prices(store, fetch=fetch, stop=lambda: stop["now"])
    assert batches == [["T0", "T1"], ["T2", "T3"]] and report["loaded"] == 4 and report["stopped"] == "stopped on request"
    with pytest.raises(FundamentalsError, match="no companies"):
        P.load_prices(FundamentalsStore(store.db.path.parent / "empty.db"), fetch=fetch)


def test_exchange_rates_are_stored_as_units_of_the_currency_per_dollar(store):
    company(store, "uk", "BBB", "GB", "GBP")
    company(store, "jp", "7203", "JP", "JPY")
    company(store, "us", "AAA", "US", "USD")
    asked = []

    def fetch(symbols, start):
        asked.extend(symbols)
        idx = pd.bdate_range("2020-01-01", periods=5)
        return {"GBPUSD=X": pd.DataFrame({"close": 1.25}, index=idx), "JPYUSD=X": "no data found"}

    report = P.load_fx(store, fetch=fetch)
    assert asked == ["GBPUSD=X", "JPYUSD=X"] and report["loaded"] == ["GBP"] and report["failed"] == {"JPY": "no data found"} and report["rows"] == 5
    fx = store.fx()
    assert set(fx["base"]) == {"USD"} and set(fx["quote"]) == {"GBP"} and fx["rate"].iloc[0] == pytest.approx(0.8)
    assert P.load_fx(FundamentalsStore(store.db.path.parent / "none.db"), fetch=fetch) == {"currencies": [], "loaded": [], "failed": {}, "rows": 0}


def test_prices_can_be_read_from_a_file(store, tmp_path):
    us, uk = company(store, "us", "AAA", "US"), company(store, "uk", "BBB", "GB")
    path = tmp_path / "prices.csv"
    path.write_text("ticker,date,close,volume\nAAA,2020-01-02,10,100\nAAA,2020-01-03,11,100\nuk:BBB,2020-01-02,5,50\nZZZ,2020-01-02,1,1\n")
    report = P.import_csv(store, path)
    assert report == {"rows": 3, "companies": 2, "unknown_tickers": ["ZZZ"], "unknown_count": 1}
    assert store.closes([us]).iloc[1, 0] == 11.0 and store.prices([uk]).iloc[0, 0] == 5.0 and store.volumes([uk]).iloc[0, 0] == 50.0
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker,date\nAAA,2020-01-02\n")
    with pytest.raises(FundamentalsError, match="needs the columns"):
        P.import_csv(store, bad)


def test_the_yahoo_answer_is_split_by_symbol_and_the_columns_renamed(monkeypatch):
    yf = pytest.importorskip("yfinance")
    idx = pd.bdate_range("2020-01-01", periods=4)
    cols = pd.MultiIndex.from_product([["Close", "Adj Close", "Volume"], ["AAA", "BBB"]], names=["Price", "Ticker"])
    data = pd.DataFrame(np.arange(24, dtype=float).reshape(4, 6), index=idx, columns=cols)
    data[("Close", "BBB")] = np.nan                                                                      # BBB has no prices at all
    monkeypatch.setattr(yf, "download", lambda *a, **k: data)
    got = P.yahoo_download(["AAA", "BBB"], "2020-01-01")
    assert list(got["AAA"].columns) == ["close", "adj_close", "volume"] and got["AAA"]["close"].iloc[0] == 0.0 and got["AAA"]["adj_close"].iloc[0] == 2.0
    assert isinstance(got["BBB"], str) and "no data found" in got["BBB"]

    def boom(*a, **k):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(yf, "download", boom)
    assert P.yahoo_download(["AAA"], "2020-01-01")["AAA"].startswith("download failed: RuntimeError")
