"""Real crypto data from free exchange feeds: each venue's book normalised to coins, spread/depth/cost-of-size measured on real levels, coin histories from OKX (or the dollar venues), funding per day."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets import crypto_analytics as ca
from src.assets.free_data import crypto as fc
from src.assets.free_data.http import FetchError, Http
from tests import free_data_helpers as fh

DAYS = [str(d.date()) for d in pd.date_range("2025-11-01", periods=330)]                       # 330 daily candles to 2026-09-26
CLOSES = [100.0 + 0.1 * i for i in range(330)]


def _http(routes, **kw):
    return Http(fh.FakeFeeds(routes), sleep=lambda s: None, pause=0.0, retries=0, **kw)


def _books(coin="BTC", **kw):
    """Routes answering all six venues' books with ladders around 100."""
    pair = "XBT" if coin == "BTC" else coin
    return {f"https://api.exchange.coinbase.com/products/{coin}-USD/book": fh.coinbase_book(**kw), "https://api.kraken.com/0/public/Depth": fh.kraken_depth(f"X{pair}ZUSD", **kw),
            f"https://www.bitstamp.net/api/v2/order_book/{coin.lower()}usd/": fh.bitstamp_book(**kw), f"https://www.okx.com/api/v5/market/books?instId={coin}-USDT&": fh.okx_book(**kw),
            f"https://www.okx.com/api/v5/market/books?instId={coin}-USDT-SWAP": fh.okx_book(contracts=True, **kw), "https://www.okx.com/api/v5/public/instruments": fh.okx_instrument("0.01"),
            "https://www.deribit.com/api/v2/public/get_order_book": fh.deribit_book(**{k: v for k, v in kw.items() if k in ("mid", "n", "tick")})}


# ------------------------------------------------------------------------------------------------------------------ books
def test_every_venues_book_is_normalised_to_coins_best_first_in_the_unit_it_quotes_in():
    routes = _books(mid=100.0, n=30, tick=0.01, size=5.0)
    http = _http(routes)
    cb, kr, bs = fc.coinbase_book(http, "BTC"), fc.kraken_book(http, "BTC"), fc.bitstamp_book(http, "BTC")
    ok, okp, db = fc.okx_book(http, "BTC"), fc.okx_book(http, "BTC", swap=True), fc.deribit_book(http, "BTC")
    for v in (cb, kr, bs, ok, okp, db):
        bids, asks = v["bids"], v["asks"]
        assert bids[0][0] < asks[0][0] and [b[0] for b in bids] == sorted((b[0] for b in bids), reverse=True) and [a[0] for a in asks] == sorted(a[0] for a in asks), v["venue"]
        assert all(s > 0 for _, s in bids + asks)
    assert [(v["venue"], v["kind"], v["quote_asset"]) for v in (cb, kr, bs, ok, okp, db)] == [("COINBASE", "spot", "USD"), ("KRAKEN", "spot", "USD"), ("BITSTAMP", "spot", "USD"), ("OKX", "spot", "USDT"),
                                                                                              ("OKX-PERP", "perp", "USDT"), ("DERIBIT-PERP", "perp", "USD")]
    assert cb["bids"][0] == [99.995, 5.0] and kr["asks"][0] == [100.005, 5.0] and bs["pair"] == "btcusd" and kr["pair"] == "XBTUSD"
    assert okp["bids"][0][1] == pytest.approx(5.0), "perpetual contracts are converted to coins with the instrument's contract value (0.01 coin)"
    assert db["bids"][0][1] == pytest.approx(500.0 / db["bids"][0][0]), "Deribit's sizes are dollars: divided by the price they are coins"


def test_levels_far_from_the_mid_are_dropped_and_the_count_is_capped():
    p = fh.coinbase_book(mid=100.0, n=2000, tick=0.01, size=1.0)
    v = fc.coinbase_book(_http({"https://api.exchange.coinbase.com/products/BTC-USD/book": p}), "BTC")
    assert len(v["bids"]) == len(v["asks"]) == fc.MAX_LEVELS
    wide = fc.coinbase_book(_http({"https://api.exchange.coinbase.com/products/BTC-USD/book": fh.coinbase_book(mid=100.0, n=40, tick=1.0, size=1.0)}), "BTC")
    assert max(a[0] for a in wide["asks"]) <= 105.0 and min(b[0] for b in wide["bids"]) >= 95.0, "500 basis points either side of the mid and no further"


def test_the_book_summary_measures_spread_depth_and_walks_the_levels_for_the_cost_of_size():
    venue = {"venue": "COINBASE", "pair": "BTC-USD", "bids": [[99.9, 10.0], [99.8, 10.0], [99.0, 1000.0]], "asks": [[100.1, 10.0], [100.2, 10.0], [101.0, 1000.0]]}
    s = fc.book_summary(venue, "BTC", notionals=(500.0, 1501.0, 5e6))
    assert s["mid"] == 100.0 and s["spread_bps"] == pytest.approx(20.0) and s["source"] == "downloaded" and s["venue"] == "COINBASE"
    assert s["depth"]["10bp"]["bid"] == pytest.approx(99.9 * 10) and s["depth"]["10bp"]["ask"] == pytest.approx(100.1 * 10)
    assert s["depth"]["25bp"]["bid"] == pytest.approx(99.9 * 10 + 99.8 * 10) and s["depth"]["50bp"]["ask"] == pytest.approx(100.1 * 10 + 100.2 * 10)
    assert s["imbalance"] == pytest.approx((999.0 - 1001.0) / (999.0 + 1001.0))
    assert s["impact_bps"]["500"]["buy"] == pytest.approx(10.0) and s["impact_bps"]["500"]["sell"] == pytest.approx(10.0), "all inside the first level: pays the half-spread"
    coins = 10 + 500.0 / 100.2
    assert s["impact_bps"]["1,501"]["buy"] == pytest.approx((1501.0 / coins / 100.0 - 1.0) * 1e4)
    assert s["impact_bps"]["5,000,000"] == {"buy": None, "sell": None} and s["unfilled"] == ["5,000,000"] and "cannot fill" in s["note"]
    assert set(s) >= {"symbol", "venue", "mid", "spread_bps", "depth", "imbalance", "impact_bps", "source", "note"}, "the shape the modelled book has"


def test_a_venue_snapshot_keeps_what_worked_and_says_what_did_not_and_fails_when_nothing_did():
    routes = _books(mid=100.0)
    del routes["https://www.bitstamp.net/api/v2/order_book/btcusd/"]
    routes["https://www.deribit.com/api/v2/public/get_order_book"] = (400, {"error": "no such instrument"})
    messages = []
    d = fc.fetch_venues(_http(routes), "btc", progress=messages.append)
    p = d.payload
    assert d.kind == "crypto" and d.name == "venues_BTC" and d.item == "venues" and [v["venue"] for v in p["venues"]] == ["COINBASE", "KRAKEN", "OKX", "OKX-PERP"]
    assert set(p["failed"]) == {"BITSTAMP", "DERIBIT-PERP"} and p["taker_bps_assumed"]["COINBASE"] == 60.0 and all(v["bid"] < v["ask"] and v["label"] for v in p["venues"])
    assert any("USDT" in n for n in d.meta["notes"]) and any("Not available" in n and "Bitstamp" in n for n in d.meta["notes"]) and len(messages) == 4
    with pytest.raises(FetchError, match="no venue gave an order book for BTC"):
        fc.fetch_venues(_http({}), "BTC")
    crossed = fh.coinbase_book(mid=100.0)
    crossed["bids"][0][0] = "101"
    r = _books(mid=100.0)
    r["https://api.exchange.coinbase.com/products/BTC-USD/book"] = crossed
    assert "COINBASE" in fc.fetch_venues(_http(r), "BTC").payload["failed"], "a crossed book is not a quote"
    with pytest.raises(ValueError):
        fc.check_coin("btc usd")


# ------------------------------------------------------------------------------------------------------------------ histories
def test_a_coins_history_is_spot_perpetual_and_funding_from_okx_on_the_same_days():
    routes = {"https://www.okx.com/api/v5/market/history-candles?instId=SOL-USDT&": fh.okx_candles(DAYS, CLOSES), "https://www.okx.com/api/v5/market/history-candles?instId=SOL-USDT-SWAP&": fh.okx_candles(DAYS, [c + 0.05 for c in CLOSES]),
              "https://www.okx.com/api/v5/public/funding-rate-history": fh.okx_funding(DAYS[-130:], 0.0001)}
    messages = []
    d = fc.fetch_history(_http(routes), "sol", 260, messages.append)
    f = d.frame
    assert d.name == "SOL" and d.item == "history" and list(f.columns) == ["time", "spot", "perp", "funding_daily"] and len(f) == 260 and f["time"].is_monotonic_increasing
    assert str(f["time"].iloc[-1].date()) == DAYS[-2], "the unfinished last day is left out"
    assert f["perp"].iloc[-1] == pytest.approx(f["spot"].iloc[-1] + 0.05)
    funded = f["funding_daily"].dropna()
    assert len(funded) >= 100 and funded.iloc[-1] == pytest.approx(3 * 0.0001), "three settlements a day are summed to the day"
    assert d.meta["sources"] == {"spot": "OKX (USDT)", "perp": "OKX (USDT)", "funding": "OKX"} and any("USDT" in n for n in d.meta["notes"]) and len(messages) == 3


def test_the_workspaces_loader_reads_the_history_file_as_a_downloaded_coin_whose_funding_is_okxs(tmp_path):
    routes = {"https://www.okx.com/api/v5/market/history-candles?instId=SOL-USDT&": fh.okx_candles(DAYS, CLOSES), "https://www.okx.com/api/v5/market/history-candles?instId=SOL-USDT-SWAP&": fh.okx_candles(DAYS, CLOSES),
              "https://www.okx.com/api/v5/public/funding-rate-history": fh.okx_funding(DAYS[-150:], 0.0002)}
    d = fc.fetch_history(_http(routes), "SOL", 260)
    path = tmp_path / "SOL.csv"
    d.frame.to_csv(path, index=False)
    s = ca.user_series(path, "SOL", source="downloaded", notes=["n"], funding_venue="OKX")
    assert s.source == "downloaded" and s.funding_venue == "OKX" and s.perp is not None and s.funding.notna().sum() >= 100
    funding, source = ca.venue_funding(s, "OKX")
    assert source == "downloaded" and funding.equals(s.funding), "the coin's own funding series is the OKX venue's"
    simulated, source = ca.venue_funding(s, "BINANCE")
    assert source == "simulated" and len(simulated) == len(s.funding)
    other, source = ca.venue_funding(s, "DERIBIT")
    assert source == "simulated", "the series is not Deribit's, so Deribit is no longer the platform's real one"


def test_a_coin_okx_does_not_list_falls_back_to_the_dollar_venues_for_spot_alone():
    routes = {"https://www.okx.com/api/v5/market/history-candles": OKXU, "https://api.kraken.com/0/public/OHLC": fh.kraken_ohlc(DAYS, CLOSES, "XXDGUSD")}
    d = fc.fetch_history(_http(routes), "DOGE", 260)
    assert list(d.frame.columns) == ["time", "spot"] and d.meta["sources"] == {"spot": "Kraken (USD)"} and len(d.frame) == 260 and str(d.frame["time"].iloc[-1].date()) == DAYS[-1]
    assert any("OKX" not in n or "left out" in n for n in d.meta["notes"]) and not any("USDT" in n for n in d.meta["notes"]), "no USDT caveat when the prices are in dollars"
    routes = {"https://www.okx.com/api/v5/market/history-candles": OKXU, "https://api.kraken.com/0/public/OHLC": {"error": ["EQuery:Unknown asset pair"], "result": {}},
              "https://www.bitstamp.net/api/v2/ohlc/": fh.bitstamp_ohlc(DAYS, CLOSES)}
    assert fc.fetch_history(_http(routes), "DOGE", 260).meta["sources"] == {"spot": "Bitstamp (USD)"}
    routes["https://www.bitstamp.net/api/v2/ohlc/"] = (404, {})
    routes["https://api.exchange.coinbase.com/products/DOGE-USD/candles"] = fh.coinbase_candles(DAYS, CLOSES)
    assert fc.fetch_history(_http(routes), "DOGE", 260).meta["sources"] == {"spot": "Coinbase (USD)"}
    with pytest.raises(FetchError, match="no exchange gave a daily history for DOGE"):
        fc.fetch_history(_http({"https://www.okx.com/api/v5/market/history-candles": OKXU}), "DOGE", 260)
    with pytest.raises(ValueError, match="between 250 and 1500"):
        fc.fetch_history(_http({}), "DOGE", 100)


OKXU = fh.OKX_UNKNOWN


def test_okx_funding_is_summed_per_utc_day_with_the_part_day_at_the_start_dropped():
    routes = {"https://www.okx.com/api/v5/public/funding-rate-history": fh.okx_funding(DAYS[:30], 0.0001)}
    d = fc.fetch_funding(_http(routes), "BTC", 25)
    f = d.frame
    assert d.name == "funding_BTC_OKX" and d.item == "funding" and d.meta["venue"] == "OKX" and list(f.columns) == ["time", "funding_daily"] and len(f) == 25
    assert np.allclose(f["funding_daily"], 3 * 0.0001) and f["time"].is_monotonic_increasing
    with pytest.raises(FetchError, match="almost no funding history"):
        fc.fetch_funding(_http({"https://www.okx.com/api/v5/public/funding-rate-history": {"code": "0", "data": []}}), "BTC", 25)


# ------------------------------------------------------------------------------------------------------------------ real quotes in the analytics
def test_snapshot_quotes_books_and_cross_exchange_use_real_prices_and_compare_like_with_like():
    routes = _books(mid=100.0)
    routes["https://api.kraken.com/0/public/Depth"] = fh.kraken_depth("XXBTZUSD", mid=100.0, tick=0.01)
    snap = fc.fetch_venues(_http(routes), "BTC").payload
    quotes = ca.snapshot_quotes(snap)
    assert [q["venue"] for q in quotes] == ["COINBASE", "KRAKEN", "BITSTAMP", "OKX", "OKX-PERP", "DERIBIT-PERP"] and all(q["source"] == "downloaded" and q["spread_bps"] > 0 for q in quotes)
    books = ca.snapshot_books(snap, "BTC")
    assert list(books) == [q["venue"] for q in quotes] and all(b["source"] == "downloaded" and b["depth"]["10bp"]["bid"] > 0 for b in books.values())
    spot = ca.CryptoSeries("BTC", pd.Series([1.0] * 3, index=pd.date_range("2026-01-01", periods=3)), None, None, "platform")
    out = ca.cross_exchange(spot, 1.0, 1e6, snapshot=snap)
    assert {r["kind"] for r in out["pairs"]} == {"spot", "perp"} and all(next(q for q in quotes if q["venue"] == r["buy"])["kind"] == next(q for q in quotes if q["venue"] == r["sell"])["kind"] for r in out["pairs"])
    assert len(out["pairs"]) == 4 * 3 + 2 * 1, "ordered pairs among four spot venues and two perpetuals"
    assert any("USDT" in f for f in out["risk_flags"]) and any("Real quotes" in n for n in out["notes"]) and not any("simulated" in f for f in out["risk_flags"])
    fees = next(r for r in out["pairs"] if r["buy"] == "COINBASE" and r["sell"] == "KRAKEN")["fees_bps"]
    assert fees == 60.0 + 40.0, "the entry-level taker fees of the two venues"
