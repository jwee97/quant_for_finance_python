"""Historical crypto storage: Parquet round trips, sampling limits, retention, sequence reconciliation, gap detection and no look-ahead on reads."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assets.crypto_store import BookReconciler, CryptoStore, StoreConfig, record_once

T0 = int(pd.Timestamp("2026-11-20T12:00:00Z").timestamp() * 1000)
BIDS = [[100.0, 2.0], [99.5, 3.0]]
ASKS = [[100.5, 1.0], [101.0, 4.0]]


def test_a_book_row_keeps_event_and_receipt_times_sequences_depth_and_quality(tmp_path):
    s = CryptoStore(tmp_path, StoreConfig(depth=1))
    row = s.add_book("OKX", "BTC-USDT", BIDS, ASKS, exchange_ts_ms=T0 - 40, recv_ts_ms=T0, sequence=7)
    assert row["quality"] == "ok" and len(row["bids_px"]) == 1, "depth is configurable"
    df = s.books("OKX", "BTC-USDT")
    assert df.iloc[0]["exchange_ts_ms"] == T0 - 40 and df.iloc[0]["recv_ts_ms"] == T0 and df.iloc[0]["sequence"] == 7 and df.iloc[0]["kind"] == "snapshot" and list(df.iloc[0]["bids_px"]) == [100.0]
    assert list(next((tmp_path / "data" / "user" / "crypto_history" / "books" / "OKX" / "BTC-USDT").rglob("*.parquet")).suffixes) == [".parquet"]
    s.add_book("OKX", "BTC-USDT", [[101.0, 1.0]], [[100.0, 1.0]], recv_ts_ms=T0 + 60_000)
    s.add_book("OKX", "BTC-USDT", [], ASKS, recv_ts_ms=T0 + 120_000)
    assert list(s.books("OKX", "BTC-USDT")["quality"]) == ["ok", "crossed", "empty"]
    with pytest.raises(ValueError):
        s.add_book("OKX", "../x", BIDS, ASKS)
    with pytest.raises(ValueError, match="kind"):
        s.add_book("OKX", "BTC-USDT", BIDS, ASKS, kind="delta")


def test_the_sampling_interval_skips_rows_and_a_read_never_sees_the_future_or_a_stale_or_invalid_book(tmp_path):
    s = CryptoStore(tmp_path, StoreConfig(min_interval_s=10))
    assert s.add_book("OKX", "X", BIDS, ASKS, recv_ts_ms=T0) and s.add_book("OKX", "X", BIDS, ASKS, recv_ts_ms=T0 + 3000) is None and s.add_book("OKX", "X", BIDS, ASKS, recv_ts_ms=T0 + 12_000)
    s.add_book("OKX", "X", [[101.0, 1.0]], [[100.0, 1.0]], recv_ts_ms=T0 + 30_000)                                         # crossed: never returned
    assert s.book_asof("OKX", "X", T0 - 1) is None, "nothing was received before T0"
    assert s.book_asof("OKX", "X", T0 + 12_500, 5)["recv_ts_ms"] == T0 + 12_000
    assert s.book_asof("OKX", "X", T0 + 30_500, 5) is None, "the only newer row is invalid and the last valid one is 18 seconds old"
    assert s.book_asof("OKX", "X", T0 + 30_500, 60)["recv_ts_ms"] == T0 + 12_000
    assert s.book_asof("OKX", "X", pd.Timestamp(T0 + 12_500, unit="ms", tz="UTC"), 5)["recv_ts_ms"] == T0 + 12_000


def test_retention_and_size_limits_delete_the_oldest_partitions_only(tmp_path):
    s = CryptoStore(tmp_path, StoreConfig(retention_days=5, max_bytes=10**9))
    day = 86_400_000
    for i in range(10):
        s.add_book("OKX", "X", BIDS, ASKS, recv_ts_ms=T0 - i * day, force=True)
    s.add_funding("OKX", "X", [{"funding_time_ms": T0, "rate": 0.0001, "interval_hours": 8}])
    out = s.prune(pd.Timestamp(T0, unit="ms", tz="UTC"))
    assert out["partitions_removed"] == 4 and len(s.books("OKX", "X")) == 6 and len(s.funding("OKX", "X")) == 1
    small = CryptoStore(tmp_path, StoreConfig(retention_days=999, max_bytes=1))
    assert small.prune(pd.Timestamp(T0, unit="ms", tz="UTC"))["bytes"] == 0 and small.books("OKX", "X").empty


def test_coverage_says_when_recording_began_and_where_it_has_holes(tmp_path):
    s = CryptoStore(tmp_path, StoreConfig(min_interval_s=0))
    assert s.coverage("OKX", "X")["state"] == "empty" and "history before recording began does not exist" in s.coverage("OKX", "X")["reason"]
    for t in (0, 10, 20, 30, 200, 210):
        s.add_book("OKX", "X", BIDS, ASKS, recv_ts_ms=T0 + t * 1000)
    c = s.coverage("OKX", "X")
    assert c["rows"] == 6 and c["first_ms"] == T0 and c["time_gaps"] == [{"from": T0 + 30_000, "to": T0 + 200_000}] and c["by_quality"] == {"ok": 6}


def test_funding_open_interest_and_futures_are_deduplicated_and_keep_their_metadata(tmp_path):
    s = CryptoStore(tmp_path)
    rows = [{"funding_time_ms": T0 + i * 28_800_000, "rate": 0.0001 * i, "interval_hours": 8} for i in range(3)]
    assert s.add_funding("OKX", "BTC-USDT-SWAP", rows) == 3 and s.add_funding("OKX", "BTC-USDT-SWAP", rows + [{"funding_time_ms": T0 + 3 * 28_800_000, "rate": 0.0003, "interval_hours": 8}]) == 1
    f = s.funding("OKX", "BTC-USDT-SWAP")
    assert list(f["funding_time_ms"]) == sorted(f["funding_time_ms"]) and f["interval_hours"].eq(8).all() and "recv_ts_ms" in f
    assert s.add_open_interest("OKX", "BTC", [{"ts_ms": T0, "oi": 2e9}], "usd") == 1 and s.open_interest("OKX", "BTC").iloc[0]["unit"] == "usd"
    c = [{"instrument": "BTC-26JUN26", "expiry": "2026-06-26", "mark_price": 51_000.0, "underlying_price": 50_000.0, "open_interest": 5.0, "open_interest_unit": "usd_notional", "settlement_currency": "BTC"}]
    assert s.add_futures("DERIBIT", "BTC", c, T0) == 1 and s.add_futures("DERIBIT", "BTC", c, T0) == 0 and s.add_futures("DERIBIT", "BTC", c, T0 + 1000) == 1
    assert s.futures("DERIBIT", "BTC").iloc[0]["settlement_currency"] == "BTC"


def test_okx_and_deribit_updates_must_chain_by_previous_sequence_and_a_gap_invalidates_the_book():
    snap = {"bids": BIDS, "asks": ASKS, "sequence": 10}
    r = BookReconciler("OKX", snap)
    assert r.apply({"sequence": 11, "prev_sequence": 10, "bids": [[100.0, 5.0], [99.5, 0.0]], "asks": [[100.4, 2.0]]})
    b = r.book()
    assert b["valid"] and b["sequence"] == 11 and b["bids"] == [(100.0, 5.0)] and b["asks"][0] == (100.4, 2.0) and (99.5, 3.0) not in b["bids"]
    assert not r.apply({"sequence": 13, "prev_sequence": 12, "bids": [[100.0, 1.0]]})
    assert not r.valid and "sequence gap" in r.reason and r.book()["bids"] == [(100.0, 5.0)], "the missing update is not invented"
    assert not r.apply({"sequence": 12, "prev_sequence": 11, "bids": []}), "an invalid book ignores updates until a new snapshot"
    r.reset({"bids": BIDS, "asks": ASKS, "sequence": 20})
    assert r.valid and r.apply({"sequence": 21, "prev_sequence": 20, "bids": [], "asks": []})
    d = BookReconciler("DERIBIT", {"bids": BIDS, "asks": ASKS, "sequence": 5})
    assert d.apply({"sequence": 6, "prev_sequence": 5, "bids": [], "asks": []}) and not d.apply({"sequence": 8, "prev_sequence": 6, "bids": [], "asks": []}) is False


def test_coinbase_sequences_must_be_consecutive_crossed_books_and_unsequenced_snapshots_are_invalid():
    c = BookReconciler("COINBASE", {"bids": BIDS, "asks": ASKS, "sequence": 100})
    assert c.apply({"sequence": 101, "bids": [[100.0, 1.0]], "asks": []}) and not c.apply({"sequence": 103, "bids": [], "asks": []}) and "sequence gap" in c.reason
    x = BookReconciler("COINBASE", {"bids": BIDS, "asks": ASKS, "sequence": 1})
    assert not x.apply({"sequence": 2, "bids": [[101.0, 1.0]], "asks": []}) and "crossed" in x.reason
    none = BookReconciler("OKX", {"bids": BIDS, "asks": ASKS})
    assert not none.valid and not none.apply({"sequence": 1, "prev_sequence": 0, "bids": [], "asks": []})
    with pytest.raises(ValueError, match="no reconciliation profile"):
        BookReconciler("KRAKEN", {"bids": BIDS, "asks": ASKS, "sequence": 1})


def test_a_sampling_pass_stores_books_funding_open_interest_and_futures_and_reports_failures(tmp_path):
    import json
    from pathlib import Path

    from src.assets.free_data.http import Http
    from tests import free_data_helpers as fh

    fx = Path(__file__).parent / "fixtures" / "exchanges"
    ld = lambda n: json.loads((fx / f"{n}.json").read_text())                                                                   # noqa: E731
    routes = {"https://www.okx.com/api/v5/market/books": ld("okx_order_book"), "https://www.okx.com/api/v5/public/instruments": ld("okx_contract_specs"), "https://www.okx.com/api/v5/public/funding-rate-history": ld("okx_funding_history"),
              "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume": ld("okx_open_interest"), "https://www.deribit.com/api/v2/public/get_order_book": ld("deribit_order_book"),
              "https://www.deribit.com/api/v2/public/get_book_summary_by_currency": ld("deribit_dated_futures"), "https://api.exchange.coinbase.com/products/BTC-USD/book": ld("coinbase_order_book")}
    s = CryptoStore(tmp_path)
    rep = record_once(Http(fh.FakeFeeds(routes), sleep=lambda x: None, pause=0.0, retries=0), s, "BTC")
    assert set(rep["books"]) == {"COINBASE", "OKX", "OKX-PERP", "DERIBIT-PERP"} and {"KRAKEN", "BITSTAMP"} <= set(rep["failed"]) and rep["funding_new"] > 0 and rep["open_interest_new"] > 0 and rep["futures_new"] > 0
    assert s.books("DERIBIT-PERP", "BTC-PERPETUAL").iloc[0]["sequence"] > 0 and s.books("OKX-PERP", "BTC-USDT-SWAP").iloc[0]["exchange_ts_ms"] > 1.5e12
