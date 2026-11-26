"""Which recorded book a historical decision may see. The newest row decides: if it is invalid, stale, crossed, locked, out of sequence or from the wrong instrument there is NO book, and an older valid one is never
substituted unless the policy asks for that, in which case it is marked and counted. Deterministic sequences with the answer worked out by hand."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assets.crypto_book_policy import BookPolicy, annotate, decide
from src.assets.crypto_replay import BookSeries, cross_exchange, execution_history, funding_carry
from src.assets.crypto_store import CryptoStore, StoreConfig

T0 = 1_700_000_000_000
GOOD = ([[100.0, 2.0], [99.0, 3.0]], [[101.0, 2.0], [102.0, 3.0]])


def store(tmp_path):
    return CryptoStore(tmp_path, StoreConfig(min_interval_s=0))


def add(s, t, venue="OKX", inst="X", bids=None, asks=None, **kw):
    return s.add_book(venue, inst, bids if bids is not None else GOOD[0], asks if asks is not None else GOOD[1], recv_ts_ms=t, **kw)


# ------------------------------------------------------------------------------------------------------------------------ the as-of rule in the store
def test_valid_then_invalid_then_decision_gives_no_book_not_the_older_valid_one(tmp_path):
    s = store(tmp_path)
    add(s, T0)
    add(s, T0 + 1000, bids=[[101.0, 1.0]], asks=[[100.0, 1.0]])                                     # crossed
    d = s.book_asof_decision("OKX", "X", T0 + 1500, 30)
    assert d["book"] is None and d["status"] == "none" and d["reason"] == "quality_crossed" and d["rejected"]["recv_ts_ms"] == T0 + 1000 and d["fallback_used"] is False
    assert s.book_asof("OKX", "X", T0 + 1500, 30) is None
    assert s.book_asof("OKX", "X", T0 + 900, 30)["recv_ts_ms"] == T0, "before the invalid row arrived the valid one was the state"


def test_valid_then_stale_then_decision(tmp_path):
    s = store(tmp_path)
    add(s, T0)
    d = s.book_asof_decision("OKX", "X", T0 + 6000, 5)
    assert d["book"] is None and d["reason"] == "stale" and d["rejected"]["age_s"] == 6.0
    assert s.book_asof("OKX", "X", T0 + 4999, 5)["age_s"] == pytest.approx(4.999)


@pytest.mark.parametrize("bids,asks,reason", [([[101.0, 1.0]], [[100.0, 1.0]], "quality_crossed"), ([[100.0, 1.0]], [[100.0, 1.0]], "quality_crossed"), ([], [[100.0, 1.0]], "quality_empty")])
def test_crossed_locked_and_empty_books_are_refused(tmp_path, bids, asks, reason):
    s = store(tmp_path)
    add(s, T0)
    add(s, T0 + 1000, bids=bids, asks=asks)
    assert s.book_asof_decision("OKX", "X", T0 + 1100, 30)["reason"] == reason


def test_a_locked_book_stored_as_ok_is_still_refused_at_read_time():
    df = pd.DataFrame([{"venue": "OKX", "instrument": "X", "kind": "snapshot", "quality": "ok", "exchange_ts_ms": None, "recv_ts_ms": T0, "sequence": None, "prev_sequence": None,
                        "bids_px": [100.0], "bids_qty": [1.0], "asks_px": [100.0], "asks_qty": [1.0]}])
    d = annotate(df, "OKX", "X", BookPolicy())
    assert not d.iloc[0]["valid"] and d.iloc[0]["reason"] == "locked"


@pytest.mark.parametrize("bids,reason", [([[100.0, 0.0]], "invalid_level"), ([[100.0, -1.0]], "invalid_level"), ([[float("nan"), 1.0]], "invalid_level"), ([[-5.0, 1.0]], "invalid_level")])
def test_missing_or_invalid_prices_and_quantities_are_detected(bids, reason):
    row = {"venue": "OKX", "instrument": "X", "kind": "snapshot", "quality": "ok", "exchange_ts_ms": None, "recv_ts_ms": T0, "sequence": None, "prev_sequence": None,
           "bids_px": [b[0] for b in bids], "bids_qty": [b[1] for b in bids], "asks_px": [101.0], "asks_qty": [1.0]}
    d = annotate(pd.DataFrame([row]), "OKX", "X", BookPolicy())
    assert d.iloc[0]["reason"] == reason


def test_sequence_gap_then_decision_and_a_new_valid_book_restores_the_market_state(tmp_path):
    s = store(tmp_path)
    add(s, T0, sequence=10, kind="snapshot")
    add(s, T0 + 1000, sequence=11, prev_sequence=10, kind="update")
    add(s, T0 + 2000, sequence=15, prev_sequence=14, kind="update")                                 # 12 to 14 never arrived
    assert s.book_asof_decision("OKX", "X", T0 + 2500, 30)["reason"] == "sequence_gap" and s.book_asof("OKX", "X", T0 + 2500, 30) is None
    add(s, T0 + 3000, sequence=16, kind="snapshot")                                                  # a fresh snapshot is a new valid state
    d = s.book_asof_decision("OKX", "X", T0 + 3500, 30)
    assert d["status"] == "ok" and d["book"]["recv_ts_ms"] == T0 + 3000 and d["book"]["sequence"] == 16
    assert s.book_asof("OKX", "X", T0 + 2500, 30) is None, "and the earlier decision is still refused: the rule is applied as of each time"


def test_invalid_then_a_new_valid_book_then_decision(tmp_path):
    s = store(tmp_path)
    add(s, T0, bids=[[101.0, 1.0]], asks=[[100.0, 1.0]])
    add(s, T0 + 1000)
    assert s.book_asof("OKX", "X", T0 + 500, 30) is None and s.book_asof("OKX", "X", T0 + 1500, 30)["recv_ts_ms"] == T0 + 1000


def test_out_of_order_arrival_cannot_move_a_row_into_the_past_and_a_regressing_sequence_is_refused(tmp_path):
    s = store(tmp_path)
    add(s, T0 + 2000, sequence=20, kind="snapshot")
    add(s, T0 + 1000, sequence=19, kind="snapshot")                                                  # written later, received earlier
    df = s.books("OKX", "X")
    assert list(df["recv_ts_ms"]) == [T0 + 1000, T0 + 2000], "ordered by receipt"
    assert s.book_asof("OKX", "X", T0 + 1500, 30)["sequence"] == 19 and s.book_asof("OKX", "X", T0 + 2500, 30)["sequence"] == 20
    add(s, T0 + 3000, sequence=5, kind="snapshot")                                                   # the sequence went backwards
    assert s.book_asof_decision("OKX", "X", T0 + 3500, 30)["reason"] == "out_of_order_sequence"


def test_exchange_event_time_and_local_receipt_time_are_different_and_a_stale_or_future_exchange_stamp_is_refused(tmp_path):
    s = store(tmp_path)
    add(s, T0, exchange_ts_ms=T0 - 50)                                                               # 50 ms of latency: fine
    assert s.book_asof("OKX", "X", T0 + 100, 30)["exchange_ts_ms"] == T0 - 50
    add(s, T0 + 5000, exchange_ts_ms=T0 + 5000 - 60_000)                                             # received now, produced a minute ago
    assert s.book_asof_decision("OKX", "X", T0 + 5100, 30)["reason"] == "stale_exchange_timestamp"
    add(s, T0 + 6000, exchange_ts_ms=T0 + 6000 + 10_000)                                             # the exchange clock is ahead of ours
    assert s.book_asof_decision("OKX", "X", T0 + 6100, 30)["reason"] == "exchange_timestamp_in_future"
    add(s, T0 + 7000, exchange_ts_ms=T0 - 10)                                                        # event time behind one already seen
    assert s.book_asof_decision("OKX", "X", T0 + 7100, 30)["reason"] in ("exchange_timestamp_regression", "stale_exchange_timestamp")


def test_a_row_for_another_instrument_is_refused():
    df = pd.DataFrame([{"venue": "OKX", "instrument": "Y", "kind": "snapshot", "quality": "ok", "exchange_ts_ms": None, "recv_ts_ms": T0, "sequence": None, "prev_sequence": None,
                        "bids_px": [100.0], "bids_qty": [1.0], "asks_px": [101.0], "asks_qty": [1.0]}])
    assert annotate(df, "OKX", "X", BookPolicy()).iloc[0]["reason"] == "identity_mismatch"


def test_nothing_received_after_the_decision_is_ever_visible(tmp_path):
    s = store(tmp_path)
    add(s, T0 + 1000)
    assert s.book_asof_decision("OKX", "X", T0 + 999, 30)["reason"] == "no_book_recorded", "nothing had been received by then"
    assert s.book_asof_decision("OKX", "X", T0 + 1000, 30)["status"] == "ok", "a book received at exactly the decision time is the state"


def test_two_venues_with_different_latencies_are_each_judged_on_their_own_rows(tmp_path):
    s = store(tmp_path)
    add(s, T0, venue="OKX", exchange_ts_ms=T0 - 20)
    add(s, T0 + 400, venue="BINANCE", exchange_ts_ms=T0 + 400 - 350)                                  # a slower feed, still within the allowed lag
    add(s, T0 + 2000, venue="OKX", bids=[[101.0, 1.0]], asks=[[100.0, 1.0]])
    t = T0 + 2200
    assert s.book_asof("OKX", "X", t, 30) is None
    b = s.book_asof("BINANCE", "X", t, 30)
    assert b["recv_ts_ms"] == T0 + 400 and b["exchange_ts_ms"] == T0 + 50, "the other venue's state is unaffected by this venue's invalid row"


# ------------------------------------------------------------------------------------------------------------------------ the explicit fallback
def test_the_last_valid_book_is_used_only_when_the_policy_asks_and_it_is_marked_and_bounded(tmp_path):
    s = store(tmp_path)
    add(s, T0)
    add(s, T0 + 1000, bids=[[101.0, 1.0]], asks=[[100.0, 1.0]])
    strict = BookPolicy(max_age_s=5)
    loose = BookPolicy(max_age_s=5, allow_last_valid_fallback=True, fallback_max_age_s=3)
    assert s.book_asof("OKX", "X", T0 + 1500, policy=strict) is None
    d = s.book_asof_decision("OKX", "X", T0 + 1500, policy=loose)
    assert d["status"] == "fallback" and d["fallback_used"] and d["book"]["fallback"] is True and d["book"]["recv_ts_ms"] == T0 and d["rejected"]["reason"] == "quality_crossed" and d["rejected"]["recv_ts_ms"] == T0 + 1000
    assert s.book_asof_decision("OKX", "X", T0 + 4500, policy=loose)["book"] is None, "even with the fallback, a book older than the fallback limit is not used"
    assert loose.describe()["strict"] is False and strict.describe()["strict"] is True


# ------------------------------------------------------------------------------------------------------------------------ the replay applies the same rules
def _record_pair(s, bad_b_at=None, n=8):
    for i in range(n):
        t = T0 + i * 1000
        add(s, t, venue="OKX", inst="X", bids=[[100.0, 5.0]], asks=[[100.5, 5.0]])
        if bad_b_at is not None and i == bad_b_at:
            add(s, t + 10, venue="BINANCE", inst="X", bids=[[104.0, 5.0]], asks=[[103.0, 5.0]])            # crossed
        else:
            add(s, t + 10, venue="BINANCE", inst="X", bids=[[102.0, 5.0]], asks=[[102.5, 5.0]])


def test_no_trade_is_executed_on_an_invalid_book_under_the_strict_policy(tmp_path):
    s = store(tmp_path)
    _record_pair(s, bad_b_at=3)
    out = cross_exchange(s, ("OKX", "X"), ("BINANCE", "X"), notional=100.0, fee_bps=(1.0, 1.0), max_age_s=30, max_skew_s=30, leg_delay_s=0.5)
    assert out["state"] == "ok" and out["trades"] > 0
    bad_ms = T0 + 3000 + 10
    for tr in out["trades_detail"]:
        assert tr["decision_books"]["b_recv"] != bad_ms and tr["decision_books"]["a_fallback"] is False and tr["decision_books"]["b_fallback"] is False
        assert tr.get("leg2_book_recv_ms") != bad_ms, "the second leg did not meet the crossed book either"
    assert out["coverage"]["skipped"]["invalid_book"] >= 1 and out["book_policy"]["b"]["rejected_by_reason"].get("quality_crossed", 0) >= 1
    assert out["book_policy"]["b"]["fallbacks_used"] == 0 and out["book_policy"]["b"]["policy"]["strict"] is True


def test_the_fallback_is_recorded_in_the_experiment_and_on_the_trades_that_used_it(tmp_path):
    s = store(tmp_path)
    _record_pair(s, bad_b_at=3)
    pol = BookPolicy(max_age_s=30, allow_last_valid_fallback=True)
    out = cross_exchange(s, ("OKX", "X"), ("BINANCE", "X"), notional=100.0, fee_bps=(1.0, 1.0), max_age_s=30, max_skew_s=30, leg_delay_s=0.5, policy=pol)
    audit = out["book_policy"]["b"]
    assert audit["policy"]["allow_last_valid_fallback"] is True and audit["fallbacks_used"] >= 1 and audit["fallbacks"][0]["reason"] == "quality_crossed"
    assert any(tr["decision_books"]["b_fallback"] for tr in out["trades_detail"]), "the trade that traded on a substituted book says so"


def test_funding_carry_and_execution_history_skip_invalid_books_and_report_them(tmp_path):
    s = store(tmp_path)
    for i in range(6):
        t = T0 + i * 3_600_000
        add(s, t, inst="SPOT", bids=[[100.0, 50.0]], asks=[[100.1, 50.0]])
        if i == 3:
            add(s, t + 1, inst="PERP", bids=[[101.0, 50.0]], asks=[[100.0, 50.0]])
        else:
            add(s, t + 1, inst="PERP", bids=[[100.4, 50.0]], asks=[[100.5, 50.0]])
    s.add_funding("OKX", "PERP", [{"funding_time_ms": T0 - h * 28_800_000, "rate": 0.001, "interval_hours": 8} for h in range(12)])
    out = funding_carry(s, ("OKX", "SPOT"), ("OKX", "PERP"), ("OKX", "PERP"), notional=1000.0, entry_annualised=0.0, window_settlements=3, max_age_s=3600)
    assert out["state"] == "ok" and out["coverage"]["skipped"]["no_valid_perp_book"] >= 1 and out["book_policy"]["perp"]["rejected_by_reason"].get("quality_crossed", 0) >= 1
    eh = execution_history(s, "OKX", "PERP", "buy", 1000.0, max_age_s=3600)
    assert eh["observations"] == 5, "the crossed book is not an observation of what an order would have met"
