"""The Massive client and adapter against a stand-in that follows the documented shapes."""
import pandas as pd
import pytest

from src.marketdata.client import MassiveClient
from src.marketdata.massive import MassiveProvider
from src.marketdata.models import BarsRequest, MarketDataError
from tests.fake_massive import FakeMassive

KEY = "TESTKEY123456"


def make(tmp_path, **kw):
    fake = FakeMassive(**kw)
    client = MassiveClient(key=KEY, root=tmp_path, session=fake, sleep=lambda s: None, retries=3)
    return MassiveProvider(client, root=tmp_path), fake


def test_the_key_goes_in_the_header_only_and_pagination_follows_next_url(tmp_path):
    p, fake = make(tmp_path)
    bars = p.get_historical_bars(BarsRequest("AAPL", "stocks", "2024-05-01", "2024-05-31"))
    assert len(bars.frame) == 23 and bars.provenance.pages > 1 and bars.provenance.provider == "massive"
    assert all(c["headers"]["Authorization"] == f"Bearer {KEY}" for c in fake.calls)
    assert not any(KEY in c["url"] or "apiKey" in str(c["params"]) for c in fake.calls)


def test_a_next_url_with_an_api_key_is_cleaned_and_another_host_is_refused(tmp_path):
    p, fake = make(tmp_path)
    c = p.client
    assert "apiKey" not in c._strip_key("https://api.massive.com/v3/x?apiKey=ABC&cursor=1") and "cursor=1" in c._strip_key("https://api.massive.com/v3/x?apiKey=ABC&cursor=1")
    with pytest.raises(MarketDataError, match="another host"):
        c.get("https://evil.example.com/v3/x")


def test_missing_wrong_and_unentitled_keys_give_distinct_errors(tmp_path):
    fake = FakeMassive(deny=["/v3/quotes"])
    none = MassiveProvider(MassiveClient(key="", root=tmp_path, session=fake))
    with pytest.raises(MarketDataError) as e:
        none.get_historical_bars(BarsRequest("AAPL"))
    assert e.value.kind == "not_configured" and "MASSIVE_API_KEY" in str(e.value)
    wrong = MassiveProvider(MassiveClient(key="WRONGKEY123456", root=tmp_path, session=fake))
    with pytest.raises(MarketDataError) as e:
        wrong.get_historical_bars(BarsRequest("AAPL"))
    assert e.value.kind == "auth"
    p = MassiveProvider(MassiveClient(key=KEY, root=tmp_path, session=fake))
    with pytest.raises(MarketDataError) as e:
        p.get_quotes("AAPL")
    assert e.value.kind == "entitlement" and "does not include" in str(e.value)
    n = len(fake.calls)
    with pytest.raises(MarketDataError, match="refused this dataset"):
        p.get_quotes("AAPL")                                         # the refusal is remembered: no second request is spent
    assert len(fake.calls) == n


def test_429_and_5xx_are_retried_with_backoff_and_counted(tmp_path):
    waits = []
    fake = FakeMassive(flaky=[429, 503])
    c = MassiveClient(key=KEY, root=tmp_path, session=fake, sleep=waits.append, retries=3)
    out = MassiveProvider(c).get_instruments("stocks", limit=2, max_pages=1)
    assert len(out.frame) == 2 and len(waits) == 2 and c.usage()["retries"] == 2 and c.usage()["rate_limited"] >= 1
    c2 = MassiveClient(key=KEY, root=tmp_path, session=FakeMassive(flaky=[500] * 9), sleep=lambda s: None, retries=2)
    with pytest.raises(MarketDataError) as e:
        MassiveProvider(c2).get_instruments("stocks", limit=2, max_pages=1)
    assert e.value.kind == "server"


def test_cancel_stops_pagination(tmp_path):
    p, fake = make(tmp_path)
    seen = []
    with pytest.raises(MarketDataError) as e:
        p.get_historical_bars(BarsRequest("AAPL", "stocks", "2024-05-01", "2024-05-31"), cancel=lambda: len(fake.calls) >= 2)
    assert e.value.kind == "cancelled" and len(fake.calls) == 2


def test_stock_and_crypto_daily_bars_are_dated_in_their_own_time_zones(tmp_path):
    p, _ = make(tmp_path)
    s = p.get_historical_bars(BarsRequest("AAPL", "stocks", "2024-05-01", "2024-05-10")).frame
    assert s.index[0] == pd.Timestamp("2024-05-01") and list(s.columns[:5]) == ["open", "high", "low", "close", "volume"]
    c = p.get_historical_bars(BarsRequest("X:BTCUSD", "crypto", "2024-05-01", "2024-05-07")).frame
    assert len(c) == 7 and c.index[0] == pd.Timestamp("2024-05-01")


def test_futures_bars_use_their_own_endpoint_session_dates_and_settlement(tmp_path):
    p, fake = make(tmp_path)
    b = p.get_historical_bars(BarsRequest("CLN4", "futures", "2024-05-01", "2024-05-10"))
    call = [c for c in fake.calls if c["path"].startswith("/futures/v1/aggs/CLN4")][0]
    assert call["params"]["resolution"] == "1session" and "window_start.gte" in call["params"]
    assert b.frame.index[0] == pd.Timestamp("2024-05-01") and b.frame["settlement_price"].notna().all() and b.frame["vwap"].isna().all()


def test_options_contracts_futures_contracts_and_the_chain_snapshot(tmp_path):
    p, fake = make(tmp_path)
    oc = p.get_contracts("options", "SPY", as_of="2024-06-14", expired=True)
    assert len(oc.frame) == 10 and fake.calls[-1]["params"]["expired"] == "true" and fake.calls[-1]["params"]["as_of"] == "2024-06-14"
    fc = p.get_contracts("futures", "CL", as_of="2024-06-14", active=True)
    assert list(fc.frame["ticker"]) == ["CLN4", "CLQ4", "CLU4", "CLV4"] and "open interest" in fc.provenance.notes[0]
    chain = p.get_option_chain("SPY")
    call = chain.frame[chain.frame["right"] == "C"].iloc[0]
    assert call["iv_source"] == "provider" and call["greeks_source"] == "provider" and call["delta"] == 0.5
    put = chain.frame[chain.frame["right"] == "P"].iloc[0]
    assert put["iv_source"] == "missing" and put["greeks_source"] == "missing" and any("no provider implied volatility" in w for w in chain.warnings)


def test_corporate_actions_use_the_current_endpoints(tmp_path):
    p, fake = make(tmp_path, splits=[{"ticker": "AAPL", "execution_date": "2020-08-31", "split_from": 1, "split_to": 4, "historical_adjustment_factor": 0.25}])
    t = p.get_corporate_actions("AAPL")
    assert {c["path"] for c in fake.calls} == {"/stocks/v1/splits", "/stocks/v1/dividends"} and set(t.frame["kind"]) == {"split", "dividend"}


def test_futures_quotes_are_not_enabled_and_the_capability_table_never_overclaims(tmp_path):
    p, _ = make(tmp_path)
    with pytest.raises(MarketDataError) as e:
        p.get_quotes("CLN4", "futures")
    assert e.value.kind == "unsupported"
    assert all(c.status == "documented" for c in p.get_provider_capabilities())                      # nothing is "available" before it was asked


def test_the_probe_records_what_the_key_can_and_cannot_reach(tmp_path):
    p, _ = make(tmp_path, deny=["/v3/quotes", "/v3/trades", "/futures/v1/trades"], today=str(pd.Timestamp.now().date()))
    caps = {f"{c.asset_class}.{c.dataset}": c for c in p.probe()}
    assert caps["stocks.bars"].status == "available" and caps["options.contracts"].status == "available" and caps["futures.bars"].status == "available"
    assert caps["stocks.quotes"].status == "not_entitled" and caps["futures.trades"].status == "not_entitled" and caps["options.trades"].status == "not_entitled"
    again = MassiveProvider(MassiveClient(key=KEY, root=tmp_path, session=FakeMassive()), root=tmp_path)
    assert {c.dataset: c.status for c in again.get_provider_capabilities() if c.asset_class == "stocks"}["quotes"] == "not_entitled"      # remembered across restarts
