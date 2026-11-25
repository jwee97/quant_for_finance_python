"""The replay engine on constructed recordings: only recorded data is used, no look-ahead, executable fills, funding timing, inventory, legging and honest gaps."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assets import crypto_replay as rp
from src.assets.crypto_store import CryptoStore, StoreConfig

T0 = int(pd.Timestamp("2026-11-01T00:00:00Z").timestamp() * 1000)
SEC, HOUR = 1000, 3_600_000


def store(tmp_path):
    return CryptoStore(tmp_path, StoreConfig(min_interval_s=0))


def put(s, venue, inst, t, bid, ask, size=5.0, bid2=None):
    s.add_book(venue, inst, [[bid, size], [bid - 0.5 if bid2 is None else bid2, size]], [[ask, size], [ask + 0.5, size]], recv_ts_ms=t, exchange_ts_ms=t - 20, sequence=t, force=True)


def test_a_cross_exchange_trade_fills_at_recorded_depth_and_the_second_leg_meets_the_later_book(tmp_path):
    s = store(tmp_path)
    for i in range(3):
        t = T0 + i * 10 * SEC
        put(s, "KRAKEN", "XBTUSD", t, 99.9, 100.0)                                                           # buy here at 100.0
        put(s, "COINBASE", "BTC-USD", t, 100.5, 100.6)                                                       # sell there at 100.5 ...
        put(s, "COINBASE", "BTC-USD", t + 1500, 99.0, 99.1)                                                  # ... but one and a half seconds later the bid has gone to 99.0
    out = rp.cross_exchange(s, ("KRAKEN", "XBTUSD"), ("COINBASE", "BTC-USD"), notional=250.0, fee_bps=(0.0, 0.0), max_age_s=5, leg_delay_s=1.0, min_edge_bps=10.0)
    assert out["state"] == "ok" and out["mode"]["id"] == "historical_replay" and out["mode"]["hypothetical"] is False and out["trades"] >= 1
    t = out["trades_detail"][0]
    assert t["decision_edge_bps"] == pytest.approx((100.5 / 100.0 - 1) * 1e4, rel=1e-6), "the decision saw the book as it was at the decision time"
    assert t["sell_vwap"] == pytest.approx(99.0) and t["gross"] == pytest.approx(t["qty"] * (99.0 - 100.0)), "the second leg met the LATER book: legging risk is realised, not assumed away"
    assert t["leg2_delay_s"] == pytest.approx(1.5) and t["decision_books"]["a_recv"] == T0 and out["gross_pnl"] < 0
    assert out["fees"] == 0 and out["max_drawdown"] < 0 and out["coverage"]["decision_times"] == 3


def test_stale_skewed_missing_and_edgeless_decisions_are_counted_not_traded(tmp_path):
    s = store(tmp_path)
    put(s, "KRAKEN", "XBTUSD", T0, 99.9, 100.0)
    put(s, "COINBASE", "BTC-USD", T0 - 60 * SEC, 100.5, 100.6)                                               # a minute old at the decision
    put(s, "KRAKEN", "XBTUSD", T0 + 120 * SEC, 99.9, 100.0)
    put(s, "COINBASE", "BTC-USD", T0 + 119 * SEC, 99.9, 100.0)                                               # no edge
    put(s, "COINBASE", "BTC-USD", T0 + 130 * SEC, 99.9, 100.0)
    out = rp.cross_exchange(s, ("KRAKEN", "XBTUSD"), ("COINBASE", "BTC-USD"), fee_bps=(10, 10), max_age_s=5, min_edge_bps=1.0)
    sk = out["coverage"]["skipped"]
    assert out["trades"] == 0 and sk["stale_or_skewed"] >= 1 and sk["no_edge"] == 1
    assert rp.cross_exchange(s, ("KRAKEN", "XBTUSD"), ("OKX", "NOPE"), fee_bps=(0, 0))["state"] == "unavailable"


def test_a_second_leg_with_no_book_is_left_unhedged_and_kept_out_of_the_pnl(tmp_path):
    s = store(tmp_path)
    put(s, "KRAKEN", "XBTUSD", T0, 99.9, 100.0)
    put(s, "COINBASE", "BTC-USD", T0, 100.5, 100.6)                                                          # no later COINBASE book exists
    out = rp.cross_exchange(s, ("KRAKEN", "XBTUSD"), ("COINBASE", "BTC-USD"), fee_bps=(0, 0), leg_delay_s=1.0)
    assert out["trades"] == 0 and len(out["unhedged"]) == 1 and out["unhedged"][0]["status"] == "open_unhedged" and out["gross_pnl"] == 0.0


def test_inventory_limits_the_size_of_a_repeated_spread_trade(tmp_path):
    s = store(tmp_path)
    for i in range(6):
        t = T0 + i * 10 * SEC
        put(s, "KRAKEN", "XBTUSD", t, 99.9, 100.0, size=1000)
        put(s, "COINBASE", "BTC-USD", t, 100.5, 100.6, size=1000)
        put(s, "COINBASE", "BTC-USD", t + 1100, 100.5, 100.6, size=1000)
    out = rp.cross_exchange(s, ("KRAKEN", "XBTUSD"), ("COINBASE", "BTC-USD"), notional=900.0, fee_bps=(0, 0), inventory_usd=1000.0, leg_delay_s=1.0)
    sold = sum(t["qty"] for t in out["trades_detail"])
    assert sold <= 1000.0 / 100.0 + 1e-9, "the selling venue held 10 coins and transfers are not modelled"
    assert out["trades"] == 2 and [round(x["qty"], 6) for x in out["trades_detail"]] == [9.0, 1.0] and out["coverage"]["skipped"]["no_edge"] + out["coverage"]["skipped"]["no_inventory"] == 4, "once the inventory is spent the rest are not trades"


def carry_recording(s, days=6, rate=0.0003, perp_bid=100.0, spot_gap=0.0):
    for h in range(0, days * 24 + 1):
        t = T0 + h * HOUR
        put(s, "OKX", "BTC-USDT", t, 100.0 + spot_gap - 0.1, 100.0 + spot_gap, size=50)
        put(s, "OKX-PERP", "BTC-USDT-SWAP", t, perp_bid, perp_bid + 0.1, size=50)
    s.add_funding("OKX", "BTC-USDT-SWAP", [{"funding_time_ms": T0 + (k * 8) * HOUR, "rate": rate, "interval_hours": 8} for k in range(0, days * 3 + 1)])


def test_funding_carry_enters_only_on_past_settlements_and_receives_each_recorded_settlement_once(tmp_path):
    s = store(tmp_path)
    carry_recording(s)
    out = rp.funding_carry(s, ("OKX", "BTC-USDT"), ("OKX-PERP", "BTC-USDT-SWAP"), ("OKX", "BTC-USDT-SWAP"), notional=10_000.0, entry_annualised=0.05, window_settlements=3, fee_bps=(0.0, 0.0), leverage=3.0)
    assert out["state"] == "ok" and out["trades"] == 1
    t = out["trades_detail"][0]
    assert t["open_ms"] >= T0 + 16 * HOUR + 1, "three settlements (at 0, 8 and 16 hours) must have PASSED strictly before the decision"
    n = t["settlements"]
    qty = t["qty"]
    assert n >= 10 and t["funding"] == pytest.approx(n * 0.0003 * qty * 100.05, rel=1e-6) and out["funding"] == pytest.approx(t["funding"])
    assert out["net_pnl"] == pytest.approx(out["gross_pnl"] - out["fees"] + out["funding"] - out["financing"]) and t["settlements_missed"] == 0
    assert out["coverage"]["funding_settlements"] == 19 and out["mode"]["id"] == "historical_replay"


def test_fees_slippage_financing_and_a_margin_call_are_charged_in_the_carry_replay(tmp_path):
    s = store(tmp_path)
    carry_recording(s, spot_gap=0.0)
    base = rp.funding_carry(s, ("OKX", "BTC-USDT"), ("OKX-PERP", "BTC-USDT-SWAP"), ("OKX", "BTC-USDT-SWAP"), entry_annualised=0.05, window_settlements=3, fee_bps=(0, 0))
    costly = rp.funding_carry(s, ("OKX", "BTC-USDT"), ("OKX-PERP", "BTC-USDT-SWAP"), ("OKX", "BTC-USDT-SWAP"), entry_annualised=0.05, window_settlements=3, fee_bps=(10, 5), financing_apr=0.10)
    assert costly["fees"] > 0 and costly["financing"] > 0 and costly["net_pnl"] < base["net_pnl"] and costly["slippage_included_in_gross"] != 0
    s2 = store(tmp_path / "squeeze")
    for h in range(0, 61):
        t = T0 + h * HOUR
        px = 100.0 if h < 30 else 100.0 + (h - 29) * 5.0                                                       # the perpetual rallies: the short loses its margin
        put(s2, "OKX", "BTC-USDT", t, px - 0.1, px, size=500)
        put(s2, "OKX-PERP", "BTC-USDT-SWAP", t, px, px + 0.1, size=500)
    s2.add_funding("OKX", "BTC-USDT-SWAP", [{"funding_time_ms": T0 + k * 8 * HOUR, "rate": 0.0003, "interval_hours": 8} for k in range(0, 9)])
    sq = rp.funding_carry(s2, ("OKX", "BTC-USDT"), ("OKX-PERP", "BTC-USDT-SWAP"), ("OKX", "BTC-USDT-SWAP"), entry_annualised=0.05, window_settlements=3, fee_bps=(0, 0), leverage=5.0, mmr=0.01)
    assert any(t["status"] == "margin_call" for t in sq["trades_detail"])


def test_carry_needs_recorded_funding_and_books_and_says_so_otherwise(tmp_path):
    s = store(tmp_path)
    assert rp.funding_carry(s, ("OKX", "A"), ("OKX-PERP", "B"), ("OKX", "B"))["state"] == "unavailable"
    for h in range(5):
        put(s, "OKX", "A", T0 + h * HOUR, 99.9, 100.0)
        put(s, "OKX-PERP", "B", T0 + h * HOUR, 100.0, 100.1)
    out = rp.funding_carry(s, ("OKX", "A"), ("OKX-PERP", "B"), ("OKX", "B"))
    assert out["state"] == "unavailable" and "no funding settlement is recorded" in out["reason"] and out["needs"]


def test_dated_futures_basis_settles_only_on_a_recorded_post_expiry_observation(tmp_path):
    s = store(tmp_path)
    exp = pd.Timestamp("2026-12-01").date().isoformat()
    exp_ms = int(pd.Timestamp("2026-12-01T08:00:00Z").timestamp() * 1000)
    c = lambda mark, idx: [{"instrument": "BTC-1DEC26", "expiry": exp, "mark_price": mark, "underlying_price": idx}]                          # noqa: E731
    s.add_futures("DERIBIT", "BTC", c(101.0, 100.0), T0)                                                                                   # 1% over index, 30 days out: about 12% a year
    s.add_futures("DERIBIT", "BTC", c(100.2, 99.0), T0 + 10 * 86_400_000)
    open_only = rp.futures_basis(s, "DERIBIT", "BTC", entry_annualised=0.05, fee_bps=3.5, notional=10_000.0)
    assert open_only["trades"] == 0 and len(open_only["open_positions"]) == 1 and "not settled" in open_only["open_positions"][0]["note"] and open_only["net_pnl"] == 0.0
    s.add_futures("DERIBIT", "BTC", c(97.0, 97.0), exp_ms + 3600_000)
    done = rp.futures_basis(s, "DERIBIT", "BTC", entry_annualised=0.05, fee_bps=3.5, notional=10_000.0)
    t = done["trades_detail"][0]
    assert t["status"] == "settled" and t["gross"] == pytest.approx(10_000.0 / 100.0 * 1.0) and t["fees"] == pytest.approx(10_000 * 3.5e-4 * 2) and done["net_pnl"] == pytest.approx(100.0 - 7.0)
    assert rp.futures_basis(s, "DERIBIT", "ETH")["state"] == "unavailable"


def test_historical_execution_costs_come_from_real_recorded_books(tmp_path):
    s = store(tmp_path)
    for i, size in enumerate((50.0, 50.0, 0.5)):
        put(s, "KRAKEN", "XBTUSD", T0 + i * 60 * SEC, 99.9, 100.0, size=size)
    out = rp.execution_history(s, "KRAKEN", "XBTUSD", "buy", 1000.0)
    assert out["observations"] == 3 and out["slippage_bps"]["median"] > 0 and out["unfilled_share_mean"] > 0 and out["fully_filled_share"] == pytest.approx(2 / 3)
    thin = rp.execution_history(s, "KRAKEN", "XBTUSD", "sell", 40.0)
    assert thin["slippage_bps"]["median"] >= 0
    with pytest.raises(ValueError):
        rp.execution_history(s, "KRAKEN", "XBTUSD", "hold", 1.0)
    assert rp.execution_history(s, "KRAKEN", "NOPE", "buy", 1.0)["state"] == "unavailable"
