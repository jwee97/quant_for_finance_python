"""The vendor-neutral market-data layer: schema, contracts, the point-in-time store (no look-ahead, revisions, streaming extension) and every loader."""

from __future__ import annotations

import http.server
import json
import threading

import numpy as np
import pandas as pd
import pytest

from src.marketdata import (DataContract, RestPollingAdapter, ReplayStream, UniverseHistory, WebSocketAdapter, apply_contract, concat_events, events_from_prices, fill_missing_bars,
                            http_fetcher, load_csv, load_events, load_sql, normalise_events, stale_flags, validate_events, write_arrow, write_csv,
                            write_jsonl, write_parquet, write_sql)
from src.marketdata.store import PointInTimeStore


def make_events(n=10, lag="30min"):
    idx = pd.bdate_range("2024-01-02", periods=n)
    prices = pd.DataFrame({"AAA": 100 + np.arange(n, dtype=float), "BBB": 50 + np.arange(n) * 0.5}, index=idx)
    return events_from_prices(prices, "bar", "16:00", lag=lag, spread_bps=2.0)


# ------------------------------------------------------------------------------------------------------------------------------------- schema
def test_available_at_must_be_stated_or_a_lag_given():
    raw = pd.DataFrame({"timestamp": ["2024-01-02 16:00"], "instrument_id": ["A"], "event_type": ["bar"], "close": [1.0]})
    with pytest.raises(ValueError, match="available_at"):
        normalise_events(raw)
    out = normalise_events(raw, lag="1h")
    assert out["available_at"].iloc[0] == pd.Timestamp("2024-01-02 17:00")
    assert out.attrs["assumed_available_at"] == 1                                    # the assumption stays visible


def test_lag_per_event_type():
    raw = pd.DataFrame({"timestamp": ["2024-01-02 16:00"] * 2, "instrument_id": ["A", "B"], "event_type": ["bar", "reference"], "close": [1.0, np.nan], "value": [np.nan, 3.0]})
    out = normalise_events(raw, lag={"bar": "0s", "reference": "1D"})
    got = dict(zip(out["instrument_id"], out["available_at"] - out["timestamp"]))
    assert got["A"] == pd.Timedelta(0) and got["B"] == pd.Timedelta(days=1)


def test_validate_flags_problems():
    ev = make_events()
    assert validate_events(ev).empty
    bad = ev.copy()
    bad.loc[bad.index[0], "available_at"] = bad.loc[bad.index[0], "timestamp"] - pd.Timedelta(hours=1)
    kinds = set(validate_events(bad)["kind"])
    assert "available_before_observed" in kinds
    crossed = ev[ev["event_type"] == "quote"].copy()
    crossed.loc[crossed.index[0], ["bid", "ask"]] = [101.0, 100.0]
    assert "crossed_quote" in set(validate_events(crossed)["kind"])


# ------------------------------------------------------------------------------------------------------------------------ the point-in-time store
def test_store_returns_only_available_data():
    store = PointInTimeStore(make_events(lag="2h"))
    day = pd.Timestamp("2024-01-05")
    assert store.latest("AAA", "bar", day + pd.Timedelta(hours=17)) is not None            # stamped 16:00 on the 5th, available 18:00 -> not yet
    assert pd.Timestamp(store.latest("AAA", "bar", day + pd.Timedelta(hours=17)).timestamp) == pd.Timestamp("2024-01-04 16:00")
    assert pd.Timestamp(store.latest("AAA", "bar", day + pd.Timedelta(hours=19)).timestamp) == pd.Timestamp("2024-01-05 16:00")
    store.assert_no_lookahead()
    assert store.violations == 0 and store.max_available_returned <= day + pd.Timedelta(hours=19)


def test_history_excludes_unpublished_points():
    store = PointInTimeStore(make_events(lag="2h"))
    h = store.history("AAA", "bar", "close", "2024-01-05 17:00")
    assert h.index[-1] == pd.Timestamp("2024-01-04 16:00") and len(h) == 3


def test_max_age_makes_stale_data_unavailable():
    store = PointInTimeStore(make_events(), max_age="2D")
    assert store.latest("AAA", "bar", "2024-01-03 17:00") is not None
    assert store.latest("AAA", "bar", "2024-01-20 17:00") is None
    snap = store.price("AAA", "2024-01-03 17:00")
    assert snap.age == pd.Timedelta(hours=1) and snap.source in ("quote", "bar")


def revised_events():
    return normalise_events(pd.DataFrame({
        "timestamp": ["2024-01-02 16:00", "2024-01-02 16:00"], "available_at": ["2024-01-02 17:00", "2024-01-05 09:00"], "instrument_id": ["CPI", "CPI"], "event_type": ["reference"] * 2,
        "value": [3.0, 3.4], "revision": [0, 1]}))


def test_revision_policies():
    latest = PointInTimeStore(revised_events(), "latest_known")
    first = PointInTimeStore(revised_events(), "first_release")
    t_before, t_after = pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-06")
    assert latest.latest("CPI", "reference", t_before).value == 3.0                       # the revision is not yet public
    assert latest.latest("CPI", "reference", t_after).value == 3.4
    assert first.latest("CPI", "reference", t_after).value == 3.0                         # the honest first-release view


def test_store_extend_appends_and_rejects_late_data():
    store = PointInTimeStore(make_events(5))
    last = store.events["available_at"].iloc[-1]
    new = normalise_events(pd.DataFrame({"timestamp": [last + pd.Timedelta(days=1)], "instrument_id": ["AAA"], "event_type": ["bar"], "close": [999.0]}), lag="0s")
    assert store.extend(new) == 1
    assert store.latest("AAA", "bar", last + pd.Timedelta(days=2)).close == 999.0
    late = new.copy()
    late["available_at"] = last - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="late data"):
        store.extend(late)


# ----------------------------------------------------------------------------------------------------------------------------------- contracts
def test_contract_converts_time_zone_and_raises_availability():
    raw = pd.DataFrame({"timestamp": ["2024-01-02 16:00"], "instrument_id": ["A"], "event_type": ["settlement"], "settlement": [1.0]})
    c = DataContract(timezone="America/New_York", calendar="US", publication_lag={"settlement": "90min"})
    out, rep = apply_contract(raw, c)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-02 21:00")                  # 16:00 New York is 21:00 UTC in winter
    assert out["available_at"].iloc[0] == pd.Timestamp("2024-01-02 22:30")
    assert rep.availability_raised == 1


def test_contract_dst_ambiguity_is_an_error_unless_shifted():
    raw = pd.DataFrame({"timestamp": ["2024-03-10 02:30"], "instrument_id": ["A"], "event_type": ["bar"], "close": [1.0]})        # does not exist in New York
    with pytest.raises(ValueError, match="daylight"):
        apply_contract(raw, DataContract(timezone="America/New_York", calendar="24x7"))
    out, _ = apply_contract(raw, DataContract(timezone="America/New_York", calendar="24x7", dst_policy="shift_forward"))
    assert len(out) == 1


def test_stale_quotes_and_gaps_are_flagged():
    q = pd.DataFrame({"timestamp": pd.date_range("2024-01-02 10:00", periods=10, freq="min"), "instrument_id": "A", "event_type": "quote", "bid": 1.0, "ask": 1.1})
    flags = stale_flags(q, max_repeat=3)
    assert flags.sum() == 7 and not flags.iloc[:3].any()
    gap = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:01", "2024-01-02 12:00"]), "instrument_id": "A", "event_type": "bar", "close": [1, 2, 3.0]})
    assert list(stale_flags(gap, max_gap="1h")) == [False, False, True]


def test_missing_data_policies():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08"])                    # 4th and 5th are missing business days
    bars = pd.DataFrame({"x": [1.0, 2.0, 5.0]}, index=idx)
    with pytest.raises(ValueError):
        fill_missing_bars(bars, "WEEKDAY", "error")
    ff = fill_missing_bars(bars, "WEEKDAY", "ffill", limit=1)
    assert ff.loc["2024-01-04", "x"] == 2.0 and np.isnan(ff.loc["2024-01-05", "x"])      # a stale price is not a price beyond the limit
    assert len(fill_missing_bars(bars, "WEEKDAY", "drop")) == 3


def test_universe_history_keeps_dead_instruments():
    uh = UniverseHistory(pd.DataFrame({"instrument_id": ["A", "B"], "listed": ["2020-01-01", "2020-01-01"], "delisted": [None, "2022-01-01"]}))
    assert uh.members("2021-06-01") == ["A", "B"] and uh.members("2023-01-01") == ["A"]
    assert uh.survivors_only_bias("2021-06-01") == 0.5


# ---------------------------------------------------------------------------------------------------------------------------------- loaders
def curve_events():
    return pd.concat([make_events(4), normalise_events(pd.DataFrame({"timestamp": ["2024-01-02 16:00"], "available_at": ["2024-01-02 16:30"], "instrument_id": ["USD-OIS"],
                                                                      "event_type": ["curve"], "curve_values": [{0.5: 0.05, 1.0: 0.048, 10.0: 0.04}]}))], ignore_index=True).pipe(normalise_events, lag="0s")


def assert_same(a: pd.DataFrame, b: pd.DataFrame):
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    assert len(a) == len(b)
    for col in ("timestamp", "available_at", "instrument_id", "event_type"):
        assert (a[col] == b[col]).all(), col
    for col in ("bid", "ask", "close"):
        np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True)
    ca = a[a["event_type"] == "curve"]["curve_values"].iloc[0]
    cb = b[b["event_type"] == "curve"]["curve_values"].iloc[0]
    assert {float(k): v for k, v in ca.items()} == {float(k): v for k, v in cb.items()}


@pytest.mark.parametrize("fmt", ["csv", "parquet", "arrow", "jsonl"])
def test_file_loaders_round_trip(tmp_path, fmt):
    if fmt in ("parquet", "arrow"):
        pytest.importorskip("pyarrow")
    ev = curve_events()
    path = tmp_path / f"e.{fmt}"
    {"csv": write_csv, "parquet": write_parquet, "arrow": write_arrow, "jsonl": write_jsonl}[fmt](ev, path)
    back = load_events(path, lag="0s") if fmt in ("csv",) else load_events(path)
    assert_same(ev, back)


def test_sql_loader_round_trip(tmp_path):
    ev = curve_events()
    db = tmp_path / "e.sqlite"
    write_sql(ev, db)
    assert_same(ev, load_sql(db, "select * from events", lag="0s"))
    assert len(load_sql(db, "select * from events where instrument_id = ?", params=("USD-OIS",), lag="0s")) == 1


def test_loader_mapping_renames_vendor_columns(tmp_path):
    raw = pd.DataFrame({"Date": ["2024-01-02 16:00"], "Symbol": ["X"], "Kind": ["bar"], "Close": [10.0]})
    p = tmp_path / "v.csv"
    raw.to_csv(p, index=False)
    out = load_csv(p, mapping={"Date": "timestamp", "Symbol": "instrument_id", "Kind": "event_type", "Close": "close"}, lag="15min")
    assert out["close"].iloc[0] == 10.0 and out["available_at"].iloc[0] == pd.Timestamp("2024-01-02 16:15")
    with pytest.raises(ValueError, match="available_at"):
        load_csv(p, mapping={"Date": "timestamp", "Symbol": "instrument_id", "Kind": "event_type", "Close": "close"})


def test_unknown_format_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        load_events(tmp_path / "x.xyz")


def test_loaders_with_a_contract(tmp_path):
    raw = pd.DataFrame({"timestamp": ["2024-01-02 16:00"], "instrument_id": ["A"], "event_type": ["bar"], "close": [1.0]})
    p = tmp_path / "c.csv"
    raw.to_csv(p, index=False)
    out = load_csv(p, contract=DataContract(timezone="America/New_York", calendar="US", publication_lag={"bar": "10min"}))
    assert out["available_at"].iloc[0] == pd.Timestamp("2024-01-02 21:10")


# ------------------------------------------------------------------------------------------------------------- REST polling and streaming
class _Handler(http.server.BaseHTTPRequestHandler):
    payload: list = []

    def do_GET(self):                                                                   # noqa: N802
        body = json.dumps({"data": {"events": type(self).payload}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_rest_polling_over_http_keeps_only_new_events_and_stamps_arrival():
    _Handler.payload = [{"timestamp": "2024-01-02T16:00:00", "instrument_id": "A", "event_type": "bar", "close": 1.0}]
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        ad = RestPollingAdapter(http_fetcher(url, records_path="data.events"))
        first = ad.poll("2024-01-02 16:05")
        assert len(first) == 1 and first["available_at"].iloc[0] == pd.Timestamp("2024-01-02 16:05")      # arrival time, not the vendor's stamp
        again = ad.poll("2024-01-02 16:10")
        assert len(again) == 0                                                            # nothing new
        _Handler.payload.append({"timestamp": "2024-01-03T16:00:00", "instrument_id": "A", "event_type": "bar", "close": 2.0})
        third = ad.poll("2024-01-03 16:07")
        assert len(third) == 1 and third["close"].iloc[0] == 2.0
    finally:
        server.shutdown()
        server.server_close()


def test_websocket_adapter_collects_messages_and_stamps_receipt_time():
    pytest.importorskip("websockets")
    from websockets.sync.server import serve

    def handler(ws):
        ws.recv()                                                                        # the subscription message
        for i in range(3):
            ws.send(json.dumps({"timestamp": f"2024-01-02T10:0{i}:00", "instrument_id": "BTC", "event_type": "trade", "trade": 100.0 + i}))

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        clock = iter(pd.date_range("2024-01-02 10:00:30", periods=10, freq="min"))
        ad = WebSocketAdapter(f"ws://127.0.0.1:{port}", parse=lambda m: json.loads(m), subscribe="sub", max_messages=3, timeout=3.0, clock=lambda: next(clock))
        got = ad.collect()
        server.shutdown()
    assert len(got) == 3 and list(got["trade"]) == [100.0, 101.0, 102.0]
    assert (got["available_at"] > got["timestamp"]).all()


def test_replay_stream_delivers_in_availability_order():
    ev = make_events(5)
    rows = list(ReplayStream(ev, start="2024-01-03", end="2024-01-05 23:59"))
    avail = [r["available_at"] for r in rows]
    assert avail == sorted(avail) and min(avail) > pd.Timestamp("2024-01-03") and max(avail) <= pd.Timestamp("2024-01-05 23:59")


def test_concat_events_keeps_order():
    a, b = make_events(3), make_events(3)
    out = concat_events(a, b)
    assert len(out) == len(a) + len(b) and out["available_at"].is_monotonic_increasing
