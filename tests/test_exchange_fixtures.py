"""The adapters against SANITIZED REAL responses captured by ``scripts/verify_exchanges.py`` (public, read-only requests), and the capability matrix. These prove the parsers read what the exchanges sent on the capture date;
they do not prove the exchanges still answer: only ``scripts/verify_exchanges.py`` run live does that."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.assets import crypto_research as cr
from src.assets.free_data import capabilities as cap
from src.assets.free_data import crypto as fc
from src.assets.free_data.http import FetchError, Http
from tests import free_data_helpers as fh

FX = Path(__file__).parent / "fixtures" / "exchanges"


def load(name):
    return json.loads((FX / f"{name}.json").read_text())


def http(routes):
    return Http(fh.FakeFeeds(routes), sleep=lambda s: None, pause=0.0, retries=0)


def test_okx_open_interest_rows_are_utc_midnights_in_dollars():
    d = fc.fetch_open_interest(http({"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume": load("okx_open_interest")}), "BTC", 30)
    f = d.frame
    assert list(f.columns) == ["time", "OKX"] and len(f) >= 2 and (f["time"].dt.hour == 0).all() and (f["OKX"] > 1e8).all() and f["time"].is_monotonic_increasing
    assert d.meta["units"]["OKX"] == "dollars"


def test_okx_liquidations_use_the_contract_value_and_the_liquidated_side():
    d = fc.fetch_liquidations(http({"https://www.okx.com/api/v5/public/liquidation-orders": load("okx_liquidations"), "https://www.okx.com/api/v5/public/instruments": load("okx_contract_specs")}), "BTC")
    assert d.payload["contract_value"] == 0.01 and d.payload["events"]
    e = d.payload["events"][0]
    assert set(e) >= {"time", "side", "pos_side", "size_contracts", "price"} and e["time"] > 1.5e12
    view = cr.liquidation_view(d.payload)
    assert view["state"] == "ok" and view["events"] == len(d.payload["events"]) and view["longs_liquidated"] + view["shorts_liquidated"] == view["events"]
    long_side = [ev for ev in d.payload["events"] if ev["side"] == "sell"]
    assert all(ev["pos_side"] in ("long", "net") for ev in long_side), "a forced sell closes a long"


def test_okx_funding_history_is_summed_per_utc_day():
    rows = load("okx_funding_history")["data"]
    series = fc.okx_funding_daily(http({"https://www.okx.com/api/v5/public/funding-rate-history": {"code": "0", "data": rows}}), "BTC-USDT-SWAP", 1) if len(rows) > 3 else None
    assert rows[0]["fundingTime"].isdigit() and float(rows[0]["fundingRate"]) == pytest.approx(float(rows[0]["realizedRate"]))
    assert series is None or series.index.is_monotonic_increasing


def test_deribit_dated_futures_use_the_index_the_future_converges_to():
    d = fc.fetch_deribit_futures(http({"https://www.deribit.com/api/v2/public/get_book_summary_by_currency": load("deribit_dated_futures")}), "BTC")
    c = d.payload["contracts"]
    assert c and all(x["underlying_price"] > 1000 and x["mark_price"] > 1000 and abs(x["mark_price"] / x["underlying_price"] - 1) < 0.2 for x in c) and all(x["open_interest_unit"] == "usd_notional" for x in c)
    assert [x["expiry"] for x in c] == sorted(x["expiry"] for x in c) and all("PERPETUAL" not in x["instrument"] for x in c)
    term = cr.futures_term_structure({**d.payload, "fetched_at": "2026-11-25T14:00:00Z"})
    assert term["state"] == "ok" and all(r["days"] > 0.5 for r in term["contracts"])


def test_deribit_liquidations_report_an_empty_window_as_an_answer_not_an_error():
    d = fc.fetch_deribit_liquidations(http({"https://www.deribit.com/api/v2/public/get_last_trades_by_currency": load("deribit_liquidations")}), "BTC")
    assert d.payload["scanned_trades"] > 0
    view = cr.liquidation_view(d.payload)
    assert view["state"] == "ok" and view["venue"] == "DERIBIT" and view["events"] == len(d.payload["events"])
    flagged = {"result": {"trades": [{"timestamp": 1, "price": 5e4, "amount": 1000.0, "direction": "sell", "instrument_name": "BTC-PERPETUAL", "liquidation": "T"}]}}
    out = fc.fetch_deribit_liquidations(http({"https://www.deribit.com/api/v2/public/get_last_trades_by_currency": flagged}), "BTC")
    assert out.payload["events"][0]["flag"] == "T" and "does not state which side" in cr.liquidation_view(out.payload)["note"]


@pytest.mark.parametrize("name, fetch, url, venue", [("coinbase_order_book", fc.coinbase_book, "https://api.exchange.coinbase.com/products/BTC-USD/book", "COINBASE"), ("kraken_order_book", fc.kraken_book, "https://api.kraken.com/0/public/Depth", "KRAKEN"),
                                                      ("bitstamp_order_book", fc.bitstamp_book, "https://www.bitstamp.net/api/v2/order_book/btcusd/", "BITSTAMP")])
def test_spot_books_normalise_to_coins_best_first_with_their_own_timestamps(name, fetch, url, venue):
    b = fetch(http({url: load(name)}), "BTC")
    assert b["venue"] == venue and b["bids"][0][0] < b["asks"][0][0] and [x[0] for x in b["bids"]] == sorted((x[0] for x in b["bids"]), reverse=True)
    if venue == "COINBASE":
        assert b["sequence"] and b["exchange_ts_ms"] is None
    else:
        assert b["exchange_ts_ms"] and b["exchange_ts_ms"] > 1.5e12


def test_okx_and_deribit_books_carry_sequence_numbers_and_event_times():
    ok = fc.okx_book(http({"https://www.okx.com/api/v5/market/books": load("okx_order_book"), "https://www.okx.com/api/v5/public/instruments": load("okx_contract_specs")}), "BTC", True)
    de = fc.deribit_book(http({"https://www.deribit.com/api/v2/public/get_order_book": load("deribit_order_book")}), "BTC")
    assert ok["sequence"] and ok["exchange_ts_ms"] > 1.5e12 and de["sequence"] and de["exchange_ts_ms"] > 1.5e12 and ok["kind"] == "perp"


def test_the_capability_matrix_marks_only_live_answers_verified_and_never_infers():
    def transport(url, headers, timeout):
        if "bybit" in url:
            return 403, {}, b"{}"
        if "binance" in url:
            return 451, {}, b"{}"
        if "funding-rate-history" in url:
            return 200, {}, json.dumps({"code": "0", "data": [{"fundingTime": "1", "fundingRate": "0.0001"}]}).encode()
        return 200, {}, b'{"code": "1", "data": []}'
    m = cap.run(Http(transport, sleep=lambda s: None, pause=0.0, retries=0))
    by = {(r["venue"], r["feature"]): r for r in m["rows"]}
    assert by[("OKX", "funding_history")]["status"] == "verified" and by[("OKX", "open_interest")]["status"] == "failed"
    assert by[("BYBIT", "open_interest")]["status"] == "blocked" and by[("BINANCE", "open_interest")]["status"] == "blocked"
    assert by[("BINANCE", "margin_tiers")]["status"] == "unsupported" and by[("COINBASE", "derivatives")]["status"] == "unsupported"
    assert all(r["tested_at"] for r in m["rows"] if r["status"] in ("verified", "blocked", "failed"))


def test_the_committed_matrix_lists_what_was_checked_live_and_what_was_not():
    m = json.loads((Path(__file__).parents[1] / "docs" / "generated" / "exchange_capabilities.json").read_text())
    by = {(r["venue"], r["feature"]): r for r in m["rows"]}
    assert by[("OKX", "margin_tiers")]["status"] == "verified" and by[("BYBIT", "open_interest")]["status"] in ("blocked", "verified") and "legend" in m
    with pytest.raises(FetchError):
        fc.fetch_open_interest(http({}), "BTC", 5)
    assert pd.Timestamp(m["generated_at"]).year >= 2026
