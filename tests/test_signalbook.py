"""The signal book's engine and files: sides and rows from a rule board, how old the data is, what changed between two snapshots, the record of past signals scored only on later prices (checked
against returns worked out by hand), the option volatility rules, and the ledger that is only ever appended to."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import signalbook as sb
from src.signalbook.store import MAX_FOLLOWS, Book, BookError, check_webhook

FOLLOW = {"id": "f1", "market": "equity"}


def board(strengths: dict, assets=("A", "B")):
    return {"assets": list(assets), "signals": [{"id": rule, "label": rule.title(), "score": dict(zip(assets, vals)), "strength": dict(zip(assets, vals))} for rule, vals in strengths.items()]}


def snap(day: str, strengths: dict, issued: str | None = None) -> dict:
    rows, skipped = sb.board_rows(board(strengths))
    return sb.snapshot(FOLLOW, rows, skipped, sb.freshness("equity", day, day), issued or f"{day}T22:00:00Z", "test")


def test_a_side_is_a_view_only_beyond_the_limit_and_a_missing_strength_is_no_view():
    assert [sb.side(v) for v in (0.26, -0.26, 0.25, -0.25, 0.0, None, float("nan"), 3.0)] == ["long", "short", "flat", "flat", "flat", "flat", "flat", "long"]


def test_a_board_becomes_one_row_per_rule_and_asset_and_a_rule_that_failed_is_listed_with_why():
    b = board({"trend": (1.2, -0.5)})
    b["signals"].append({"id": "carry", "label": "Carry", "error": "needs a curve"})
    rows, skipped = sb.board_rows(b, lambda rule, asset: asset + "_X")
    assert [(r["rule"], r["asset"], r["side"], r["price_key"]) for r in rows] == [("trend", "A", "long", "A_X"), ("trend", "B", "short", "B_X")]
    assert skipped == [{"rule": "carry", "label": "Carry", "why": "needs a curve"}]


def test_old_data_and_simulated_data_are_said_and_crypto_counts_every_day():
    fresh = sb.freshness("equity", "2026-11-06", "2026-11-09")                                  # a Friday read on the Monday: one trading day
    assert (fresh["age_days"], fresh["stale"], fresh["problems"]) == (1, False, [])
    old = sb.freshness("equity", "2026-10-30", "2026-11-09")
    assert old["age_days"] == 6 and old["stale"] and "6 trading days ago" in old["problems"][0] and "not today's" in old["problems"][0]
    assert sb.freshness("crypto", "2026-11-06", "2026-11-09")["age_days"] == 3 and sb.freshness("crypto", "2026-11-06", "2026-11-09")["stale"]
    sim = sb.freshness("futures", "2026-11-09", "2026-11-09", simulated=True)
    assert sim["simulated"] and not sim["stale"] and "simulated data" in sim["problems"][0]
    assert sb.freshness("equity", "2026-11-10", "2026-11-09")["age_days"] == 0


def test_the_consensus_counts_the_sides_and_names_the_view():
    rows, _ = sb.board_rows(board({"r1": (1.0, -1.0), "r2": (1.0, 0.0), "r3": (0.5, 1.0), "r4": (-1.0, 0.1)}))
    a, b = sb.consensus(rows)
    assert (a["long"], a["short"], a["flat"], a["view"]) == (3, 1, 0, "long") and a["net"] == 0.5 and a["mean_strength"] == pytest.approx(0.375)
    assert (b["long"], b["short"], b["flat"], b["view"]) == (1, 1, 2, "mixed")


def test_changes_are_the_sides_that_flipped_and_a_first_snapshot_has_none():
    first, second = snap("2026-01-05", {"trend": (1.0, -1.0), "carry": (0.0, 0.5)}), snap("2026-01-06", {"trend": (1.0, 0.1), "carry": (0.4, 0.5)})
    assert sb.changes(None, first) == []
    got = sb.changes(first, second)
    assert [(c["rule"], c["asset"], c["from"], c["to"]) for c in got] == [("trend", "B", "short", "flat"), ("carry", "A", "flat", "long")]
    assert sb.changes(second, second) == []


def test_the_record_scores_each_signal_from_the_next_close_and_only_when_its_horizon_has_passed():
    days = pd.bdate_range("2026-01-05", periods=8)
    prices = pd.DataFrame({"A": [100, 101, 103, 102, 106, 107, 108, 110], "B": [50, 50, 49, 51, 50, 48, 47, 46]}, index=days, dtype=float)
    snaps = [snap(str(days[0].date()), {"trend": (1.0, -1.0)}), snap(str(days[1].date()), {"trend": (1.0, -1.0)}), snap(str(days[5].date()), {"trend": (-1.0, 0.0)})]
    rec = sb.track_record(snaps, prices, horizons=(1, 5))
    one = rec["overall"]["1"]
    # day 0 signals enter at day 1 and leave at day 2; day 1 signals enter at day 2 and leave at day 3; the day 5 signal (short A) enters at day 6 and leaves at day 7
    by_hand = [103 / 101 - 1, -(49 / 50 - 1), 102 / 103 - 1, -(51 / 49 - 1), -(110 / 108 - 1)]
    assert one["n"] == 5 and one["mean"] == pytest.approx(np.mean(by_hand)) and one["hit_rate"] == pytest.approx(2 / 5)
    assert one["baseline"] == pytest.approx(np.mean([103 / 101 - 1, 49 / 50 - 1, 102 / 103 - 1, 51 / 49 - 1, 110 / 108 - 1])) and one["edge"] == pytest.approx(one["mean"] - one["baseline"])
    five = rec["overall"]["5"]
    assert five["n"] == 4 and five["mean"] == pytest.approx(np.mean([108 / 101 - 1, -(47 / 50 - 1), 110 / 103 - 1, -(46 / 49 - 1)])), "the day 5 signal has no five days after it yet"
    assert rec["pending"] == 1 and rec["snapshots"] == 3 and (rec["first"], rec["last"]) == (str(days[0].date()), str(days[5].date())) and rec["rules"][0]["rule"] == "trend"


def test_the_record_cannot_see_the_future_or_count_a_day_twice():
    days = pd.bdate_range("2026-01-05", periods=6)
    prices = pd.DataFrame({"A": [100.0, 101, 102, 103, 104, 105], "B": [10.0] * 6}, index=days)
    s = snap(str(days[4].date()), {"trend": (1.0, 0.0)})
    assert sb.track_record([s], prices, (1,))["overall"] == {} and sb.track_record([s], prices, (1,))["pending"] == 1, "entered at the last close: no later price to score against"
    assert sb.track_record([snap(str(days[5].date()), {"trend": (1.0, 0.0)})], prices, (1,))["overall"] == {}
    twice = [snap(str(days[0].date()), {"trend": (1.0, 0.0)}), snap(str(days[0].date()), {"trend": (-1.0, 0.0)}, f"{days[0].date()}T23:00:00Z")]
    rec = sb.track_record(twice, prices, (1,))
    assert rec["snapshots"] == 1 and rec["overall"]["1"]["n"] == 1 and rec["overall"]["1"]["mean"] == pytest.approx(102 / 101 - 1), "the first snapshot of a data day is the one that counts"
    later = prices.copy()
    later.iloc[3:] *= 5.0                                                                        # changing prices after the exit changes nothing
    assert sb.track_record(twice, later, (1,))["overall"] == rec["overall"]
    assert sb.track_record([{**s, "rows": [{**s["rows"][0], "price_key": None}]}], prices, (1,))["pending"] == 0, "a signal with no price to score against is left out"


def test_a_snapshot_written_down_after_its_entry_close_is_not_scored():
    """A run on data that is already days old issues signals whose outcome exists. They are kept in the ledger and left out of the record."""
    days = pd.bdate_range("2026-01-05", periods=6)
    prices = pd.DataFrame({"A": [100.0, 101, 102, 103, 104, 105], "B": [10.0] * 6}, index=days)
    day0, day1, day2 = (str(d.date()) for d in days[:3])
    on_time = snap(day0, {"trend": (1.0, 0.0)}, f"{day1}T08:00:00Z")                          # read the next morning, before the close it is entered at
    assert sb.track_record([on_time], prices, (1,))["overall"]["1"]["n"] == 1
    late = snap(day0, {"trend": (1.0, 0.0)}, f"{day2}T08:00:00Z")                             # read two days on: the entry close has been and gone
    rec = sb.track_record([late], prices, (1,))
    assert rec["overall"] == {} and rec["late"] == 1 and rec["snapshots"] == 0 and rec["pending"] == 0
    both = sb.track_record([late, snap(day1, {"trend": (1.0, 0.0)})], prices, (1,))
    assert both["late"] == 1 and both["snapshots"] == 1 and both["overall"]["1"]["mean"] == pytest.approx(103 / 102 - 1)


def test_the_option_rules_read_the_overview_and_say_what_they_cannot_read():
    ov = {"symbol": "SPY", "iv_rank": 0.9, "atm_iv_30d": 0.30, "realized_vol_21d": 0.20, "term_inverted": True, "term_slope": -0.02}
    rows, skipped = sb.option_rows(ov)
    assert {r["rule"]: r["side"] for r in rows} == {"vol_rank": "short", "vol_premium": "short", "term_stress": "long"} and not skipped and all(r["price_key"] is None and r["asset"] == "SPY" for r in rows)
    rows, skipped = sb.option_rows({"symbol": "X", "iv_rank": float("nan"), "iv_rank_basis": "one snapshot", "atm_iv_30d": 0.15, "realized_vol_21d": 0.20, "term_inverted": False, "term_slope": 0.01})
    assert {r["rule"]: r["side"] for r in rows} == {"vol_premium": "long", "term_stress": "flat"} and skipped == [{"rule": "vol_rank", "label": sb.OPTION_RULES[0][1], "why": "one snapshot"}]
    assert {r["rule"]: r["side"] for r in sb.option_rows({"symbol": "X", "iv_rank": 0.1, "atm_iv_30d": 0.2, "realized_vol_21d": 0.2})[0]} == {"vol_rank": "long", "vol_premium": "flat", "term_stress": "flat"}


# ------------------------------------------------------------------------------------------------------------------------------ the files
def test_the_book_keeps_follows_and_a_ledger_that_is_only_appended_to(tmp_path):
    book = Book(tmp_path / "signals")
    assert book.follows() == [] and book.snapshots() == [] and book.latest() is None and not (tmp_path / "signals").exists(), "reading makes no files"
    f = book.add("  My ETFs ", "equity", {"tickers": ["SPY"]}, ["tsmom"])
    assert f["name"] == "My ETFs" and len(f["id"]) == 8 and book.follow(f["id"]) == f
    for bad, why in [(("My etfs", "equity"), "already exists"), (("", "equity"), "1 to 60 characters"), (("x" * 61, "equity"), "1 to 60"), (("<b>", "equity"), "1 to 60"), (("Other", "bonds"), "market must be")]:
        with pytest.raises(BookError, match=why):
            book.add(bad[0], bad[1], {}, [])
    book.record({"follow": f["id"], "as_of": "2026-01-05", "rows": []})
    book.record({"follow": "other", "as_of": "2026-01-05", "rows": []})
    with (tmp_path / "signals" / "ledger.jsonl").open("a") as handle:
        handle.write('{"follow": "cut sho')                                                      # a crash in the middle of a line
    assert len(book.snapshots()) == 2 and [s["follow"] for s in book.snapshots(f["id"])] == [f["id"]]
    assert book.remove(f["id"]) == {"removed": f["id"]} and book.follows() == [] and len(book.snapshots(f["id"])) == 1, "its snapshots stay"
    with pytest.raises(BookError, match="no follow"):
        book.remove(f["id"])
    book.save_latest({"ran_at": "now"})
    assert book.latest() == {"ran_at": "now"}


def test_the_book_holds_a_bounded_number_of_follows(tmp_path):
    book = Book(tmp_path)
    for i in range(MAX_FOLLOWS):
        book.add(f"f{i}", "crypto", {}, [])
    with pytest.raises(BookError, match="at most"):
        book.add("one more", "crypto", {}, [])


def test_settings_are_checked_and_the_alert_address_must_be_https(tmp_path):
    book = Book(tmp_path)
    assert book.settings()["schedule"] == {"enabled": False, "time": "18:30", "weekdays_only": True, "refresh": True} and book.settings()["webhook"] == ""
    s = book.save_settings({"enabled": True, "time": "07:05"}, "https://hooks.example.com/abc")
    assert s["schedule"]["enabled"] and s["schedule"]["time"] == "07:05" and s["schedule"]["weekdays_only"] and book.settings()["webhook"] == "https://hooks.example.com/abc"
    assert book.save_settings(last_scheduled="2026-01-05")["webhook"] == "https://hooks.example.com/abc", "saving one thing keeps the others"
    for bad, why in [({"time": "7:5"}, "HH:MM"), ({"time": "25:00"}, "HH:MM"), ("soon", "must be an object")]:
        with pytest.raises(BookError, match=why):
            book.save_settings(bad)
    for bad, why in [("http://example.com/x", "https://"), ("ftp://example.com", "https://"), ("https://", "no host"), ("https://a b.com", "without spaces"), (5, "one web address"), ("https://" + "a" * 600, "one web address")]:
        with pytest.raises(BookError, match=why):
            check_webhook(bad)
    assert check_webhook("") == "" and check_webhook(None) == "" and check_webhook("http://127.0.0.1:9/x") == "http://127.0.0.1:9/x"
    json.loads((tmp_path / "settings.json").read_text())
    book.alert({"at": "a", "text": "one"})
    book.alert({"at": "b", "text": "two"})
    assert [a["text"] for a in book.alerts()] == ["two", "one"]


# ------------------------------------------------------------------------------------------------------------------------------ the paper portfolio
def test_target_weights_are_the_net_share_of_rules_scaled_to_be_fully_invested():
    rows, _ = sb.board_rows(board({"r1": (1.0, -1.0), "r2": (1.0, -1.0), "r3": (1.0, 0.0), "r4": (0.0, 0.0)}))
    assert sb.target_weights(rows) == pytest.approx({"A": 0.6, "B": -0.4})                      # nets of 3/4 and -2/4, over 5/4
    assert sb.target_weights(rows, long_only=True) == {"A": 1.0, "B": 0.0}
    flat, _ = sb.board_rows(board({"r1": (0.0, 0.0)}))
    assert sb.target_weights(flat) == {"A": 0.0, "B": 0.0}, "no view holds cash"
    assert sb.target_weights([{**rows[0], "price_key": None}]) == {}, "a signal with nothing to trade is left out"


def test_the_paper_portfolio_is_bought_at_the_next_close_pays_for_every_trade_and_matches_a_hand_calculation():
    days = pd.bdate_range("2026-01-05", periods=6)
    prices = pd.DataFrame({"A": [100.0, 100, 110, 121, 121, 133.1], "B": [50.0, 50, 50, 55, 55, 55]}, index=days)
    d = [str(x.date()) for x in days]
    snaps = [snap(d[0], {"r": (1.0, 0.0)}), snap(d[2], {"r": (0.0, 1.0)})]                       # all in A from the close of day 1; all in B from the close of day 3
    p = sb.paper_portfolio(snaps, prices, cost_bps=10.0, capital=1000.0)
    by_hand = [1.0, 1.10 - 0.001, None, None, None]
    by_hand[2] = by_hand[1] * 1.10                                                             # day 3: still A (+10%); B is bought at that close
    by_hand[3] = by_hand[2] * (1.0 - 0.002)                                                    # day 4: B flat, and the switch traded 200% of the book at 10 bp
    by_hand[4] = by_hand[3] * 1.0                                                              # day 5: B flat
    assert p["curve"]["dates"] == d[1:] and p["curve"]["portfolio"] == pytest.approx(by_hand) and p["days"] == 4
    assert p["curve"]["baseline"] == pytest.approx([1.0, 1.05, 1.05 * 1.10, 1.05 * 1.10, 1.05 * 1.10 * 1.05]), "equal weights of A and B over the same days"
    st = p["stats"]
    assert st["turnover"] == pytest.approx(3.0) and st["costs"] == pytest.approx(0.003) and st["rebalances"] == 2 and st["total_return"] == pytest.approx(by_hand[4] - 1) and st["value"] == pytest.approx(1000 * by_hand[4])
    assert st["sharpe"] is None and st["sharpe_needs_days"] == 60, "four days are not a Sharpe ratio"
    assert st["max_drawdown"] == pytest.approx(-0.002)
    assert p["target"] == [{"asset": "A", "weight": 0.0, "amount": 0.0}, {"asset": "B", "weight": 1.0, "amount": 1000.0}] and p["trades"] == []
    free = sb.paper_portfolio(snaps, prices, cost_bps=0.0)["curve"]["portfolio"]
    assert free == pytest.approx([1.0, 1.10, 1.21, 1.21, 1.21])


def test_the_paper_portfolio_trades_only_what_could_have_been_traded_and_says_what_the_latest_snapshot_would_change():
    days = pd.bdate_range("2026-01-05", periods=6)
    prices = pd.DataFrame({"A": [100.0, 101, 102, 103, 104, 105], "B": [10.0] * 6}, index=days)
    d = [str(x.date()) for x in days]
    assert sb.paper_portfolio([], prices)["curve"] is None and sb.paper_portfolio([], prices)["target"] == []
    newest = sb.paper_portfolio([snap(d[5], {"r": (1.0, 0.0)})], prices, capital=1000.0)
    assert newest["curve"] is None and newest["days"] == 0 and newest["trades"] == [{"asset": "A", "from": 0.0, "to": 1.0, "change": 1.0, "amount": 1000.0}], "today's snapshot: an order to place, no result yet"
    late = snap(d[0], {"r": (1.0, 0.0)}, f"{d[3]}T08:00:00Z")
    assert sb.paper_portfolio([late], prices)["curve"] is None, "written down after its entry close: never traded on paper"
    mixed = sb.paper_portfolio([snap(d[1], {"r": (1.0, 0.0)}), snap(d[5], {"r": (-1.0, 0.0)})], prices, cost_bps=0.0)
    assert mixed["curve"]["portfolio"] == pytest.approx([1.0, 103 / 102, 104 / 102, 105 / 102]) and mixed["trades"][0]["from"] == 1.0 and mixed["trades"][0]["to"] == -1.0
    assert sb.paper_portfolio([snap(d[1], {"r": (-1.0, 0.0)})], prices, cost_bps=0.0, long_only=True)["curve"]["portfolio"] == pytest.approx([1.0] * 4), "long only: a short view holds cash"
    moved = prices.copy()
    moved.iloc[:2] *= 3.0                                                                      # prices before the entry change nothing
    assert sb.paper_portfolio([snap(d[1], {"r": (1.0, 0.0)})], moved, cost_bps=0.0)["curve"]["portfolio"] == pytest.approx([1.0, 103 / 102, 104 / 102, 105 / 102])


def test_the_paper_settings_are_checked(tmp_path):
    book = Book(tmp_path)
    assert book.settings()["paper"] == {"cost_bps": 5.0, "long_only": False, "capital": 100000.0}
    assert book.save_settings(paper={"cost_bps": 12, "long_only": True})["paper"] == {"cost_bps": 12.0, "long_only": True, "capital": 100000.0}
    for bad, why in [({"cost_bps": -1}, "between 0 and 200"), ({"cost_bps": "x"}, "must be numbers"), ({"capital": 5}, "between 100 and"), ("all", "must be an object")]:
        with pytest.raises(BookError, match=why):
            book.save_settings(paper=bad)
    assert book.save_settings({"enabled": False})["paper"]["cost_bps"] == 12.0
