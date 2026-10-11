"""Real futures curves from free sources: Yahoo's per-contract histories stacked into one contract table (with the expiry dates the feed does not give), kept across downloads, and Deribit's strip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assets import futures_analytics as fa
from src.assets.free_data import futures as ff
from src.assets.free_data.http import FetchError, Http
from tests.free_data_helpers import YAHOO_MISSING, FakeFeeds, yahoo_chart

FIXTURES = Path(__file__).parent / "fixtures" / "free_data"
TODAY = "2026-10-11"


def _http(routes):
    return Http(FakeFeeds(routes), sleep=lambda s: None, pause=0.0)


def _days(n: int, end="2026-10-09"):
    return [str(d.date()) for d in pd.bdate_range(end=end, periods=n)]


# ------------------------------------------------------------------------------------------------------------------ expiry rules and contract names
def test_the_last_trading_day_follows_each_exchanges_rule():
    cases = [("cl", 2026, 12, "2026-11-20"), ("cl", 2026, 11, "2026-10-20"),                       # the 25th of October 2026 is a Sunday: three business days before the Friday that precedes it
             ("ng", 2026, 11, "2026-10-28"), ("third_last_bday", 2026, 12, "2026-12-29"), ("third_friday", 2026, 12, "2026-12-18"), ("third_friday", 2027, 3, "2027-03-19"),
             ("treasury", 2026, 12, "2026-12-22"), ("euro_fx", 2026, 12, "2026-12-14"), ("grain", 2026, 12, "2026-12-14")]
    for rule, year, month, expected in cases:
        assert str(ff.last_trade_date(rule, year, month).date()) == expected, (rule, year, month)
    with pytest.raises(ValueError):
        ff.last_trade_date("nope", 2026, 1)


def test_the_next_contracts_skip_what_has_expired_and_name_the_yahoo_symbol():
    out = ff.upcoming_contracts(ff.FREE_ROOTS["CL"], TODAY, 4)
    assert [c["contract"] for c in out] == ["X26", "Z26", "F27", "G27"] and out[0]["symbol"] == "CLX26.NYM" and out[0]["expiry"] == pd.Timestamp("2026-10-20")
    es = ff.upcoming_contracts(ff.FREE_ROOTS["ES"], TODAY, 3)
    assert [c["symbol"] for c in es] == ["ESZ26.CME", "ESH27.CME", "ESM27.CME"]
    after_expiry = ff.upcoming_contracts(ff.FREE_ROOTS["ES"], "2026-12-19", 1)
    assert after_expiry[0]["contract"] == "H27", "a contract that expired yesterday is not asked for"
    assert all(r.multiplier > 0 and r.tick_size > 0 and r.suffix.startswith(".") and r.months for r in ff.FREE_ROOTS.values())
    assert {"CL", "NG", "GC", "HG", "ES"} <= set(ff.FREE_ROOTS), "every root of the simulated workspace can be downloaded"


# ------------------------------------------------------------------------------------------------------------------ Yahoo
def test_a_real_yahoo_chart_answer_gives_one_priced_bar_per_exchange_day():
    h = ff.yahoo_history(json.loads((FIXTURES / "yahoo_clz26_chart.json").read_text()))
    assert list(h.columns) == ["date", "price", "volume"] and len(h) == 12 and h["date"].is_monotonic_increasing
    assert str(h["date"].iloc[-1].date()) == "2026-10-09" and abs(h["price"].iloc[-1] - 91.01) < 1e-4 and h["volume"].iloc[-1] == 148317, "the last bar of the live feed"
    p = yahoo_chart("X", ["2026-10-08", "2026-10-09", "2026-10-12"], [10.0, None, 12.0])
    assert [str(d.date()) for d in ff.yahoo_history(p)["date"]] == ["2026-10-08", "2026-10-12"], "a bar with no price is dropped"
    with pytest.raises(FetchError, match="no history") as caught:
        ff.yahoo_history(YAHOO_MISSING)
    assert caught.value.status == 404
    with pytest.raises(FetchError, match="no prices"):
        ff.yahoo_history({"chart": {"result": [{"timestamp": [], "indicators": {"quote": [{}]}}], "error": None}})


def _yahoo_routes(prices: dict[str, float], days: int = 40):
    routes = {}
    for symbol, level in prices.items():
        dates = _days(days)
        routes[f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"] = yahoo_chart(symbol, dates, [level + 0.01 * i for i in range(len(dates))])
    return routes


def test_the_curve_is_the_contracts_stacked_with_their_computed_expiries_and_read_by_the_workspace_loader(tmp_path):
    prices = {"CLX26.NYM": 88.0, "CLZ26.NYM": 90.0, "CLF27.NYM": 91.0, "CLG27.NYM": 91.5}
    messages = []
    d = ff.fetch_yahoo_curve(_http(_yahoo_routes(prices)), "cl", n_contracts=4, history="6mo", today=TODAY, progress=messages.append)
    assert d.kind == "futures" and d.name == "CL" and d.item == "curve" and d.meta["source_id"] == "yahoo" and d.meta["contracts"] == 4 and d.meta["as_of"] == "2026-10-09" and d.meta["multiplier"] == 1000.0
    assert d.meta["root_name"] == "WTI crude oil" and "name" not in d.meta, "the root's long name has its own key: `name` is the file's"
    df = d.frame
    assert list(df.columns) == ["date", "contract", "expiry", "price", "volume"] and set(df["contract"]) == {"X26", "Z26", "F27", "G27"} and len(df) == 4 * 40
    assert df.groupby("contract")["expiry"].first().loc["Z26"] == pd.Timestamp("2026-11-20") and (df["date"] <= df["expiry"] + pd.Timedelta(days=3)).all()
    assert messages[0] == "CL: X26 (1 of 4)" and len(messages) == 4
    assert any("no open interest" in n for n in d.meta["notes"]) and any("merged" in n for n in d.meta["notes"])
    path = tmp_path / "CL.csv"
    df.to_csv(path, index=False)
    curve = fa.load_curve_file(path, "CL")                                                   # the file the Futures workspace reads
    snap = fa.curve_snapshot(curve)
    assert [r["contract"] for r in snap["rows"]] == ["X26", "Z26", "F27", "G27"] and snap["state"] in ("contango", "backwardation", "mixed") and snap["front_price"] > 88


def test_contracts_yahoo_does_not_have_are_skipped_and_reported_and_no_contract_at_all_is_an_error():
    routes = _yahoo_routes({"CLZ26.NYM": 90.0, "CLF27.NYM": 91.0})
    routes["https://query1.finance.yahoo.com/v8/finance/chart/CLX26.NYM"] = (404, YAHOO_MISSING)
    d = ff.fetch_yahoo_curve(_http(routes), "CL", n_contracts=3, today=TODAY)
    assert set(d.frame["contract"]) == {"Z26", "F27"} and any("no history for X26" in n for n in d.meta["notes"])
    with pytest.raises(FetchError, match="no price history for any of the next 3 CL contracts"):
        ff.fetch_yahoo_curve(_http({}), "CL", n_contracts=3, today=TODAY)
    with pytest.raises(ValueError, match="free curves are available"):
        ff.fetch_yahoo_curve(_http({}), "XX")
    with pytest.raises(ValueError, match="history must be"):
        ff.fetch_yahoo_curve(_http({}), "CL", history="forever")


def test_a_server_that_refuses_this_address_stops_the_download_instead_of_asking_twelve_times():
    feeds = FakeFeeds({"https://query1.finance.yahoo.com/": (429, "slow down")})
    http = Http(feeds, sleep=lambda s: None, pause=0.0, retries=0)
    with pytest.raises(FetchError, match="rate limiting"):
        ff.fetch_yahoo_curve(http, "CL", n_contracts=12, today=TODAY)
    assert len(feeds.calls) == 1


def test_merging_keeps_contracts_the_feed_has_dropped_and_lets_the_new_rows_win():
    old = pd.DataFrame({"date": ["2026-09-01", "2026-09-02", "2026-09-01", "2026-09-02"], "contract": ["U26", "U26", "Z26", "Z26"],
                        "expiry": ["2026-09-21", "2026-09-21", "2026-11-20", "2026-11-20"], "price": [80.0, 81.0, 90.0, 91.0], "volume": [1.0, 1.0, 1.0, 1.0]})
    new = pd.DataFrame({"date": ["2026-09-02", "2026-09-03"], "contract": ["Z26", "Z26"], "expiry": ["2026-11-20", "2026-11-20"], "price": [91.5, 92.0], "volume": [2.0, 2.0]})
    out = ff.merge_history(old, new)
    z = out[out["contract"] == "Z26"].set_index("date")["price"]
    assert list(out[out["contract"] == "U26"]["price"]) == [80.0, 81.0], "the expired contract's history stays"
    assert z[pd.Timestamp("2026-09-01")] == 90.0 and z[pd.Timestamp("2026-09-02")] == 91.5 and z[pd.Timestamp("2026-09-03")] == 92.0
    assert out.equals(out.sort_values(["date", "expiry", "contract"]).reset_index(drop=True)) and ff.merge_history(None, new) is new


# ------------------------------------------------------------------------------------------------------------------ Deribit
def test_deribits_strip_becomes_one_days_curve_with_open_interest_and_the_perpetual_beside_it():
    payload = json.loads((FIXTURES / "deribit_btc_futures.json").read_text())                # a live answer: 12 dated futures (daily, weekly, monthly, quarterly) and the perpetual
    d = ff.fetch_deribit_curve(_http({ff.DERIBIT_URL: payload}), "btc")
    df = d.frame
    assert d.name == "BTC" and d.meta["asset_class"] == "crypto" and d.meta["exchange"] == "Deribit" and d.meta["perpetual"]["mark"] > 0 and "current_funding" in d.meta["perpetual"]
    assert list(df.columns) == ["date", "contract", "expiry", "price", "open_interest", "volume", "spot"] and df["date"].nunique() == 1
    assert list(df["contract"]) == ["16OCT26", "23OCT26", "30OCT26", "27NOV26", "25DEC26", "26MAR27", "25JUN27", "24SEP27"], "Friday expiries at least five days out: the daily futures are not a curve"
    assert (df["expiry"].dt.weekday == 4).all()
    assert df["expiry"].is_monotonic_increasing and (df["expiry"] > df["date"]).all() and (df["price"] > 0).all() and (df["open_interest"] >= 0).all() and df["spot"].nunique() == 1
    assert "PERPETUAL" not in set(df["contract"]) and d.meta["as_of"] == str(df["date"].iloc[0].date())
    curve = pd.concat([df.assign(date=df["date"] - pd.Timedelta(days=1))]).pipe(lambda x: pd.concat([x, df], ignore_index=True))
    assert curve["date"].nunique() == 2
    with pytest.raises(FetchError, match="fewer than two"):
        ff.deribit_curve(payload["result"][:1], "BTC")
    daily_only = [f for f in payload["result"] if f["instrument_name"] in ("BTC-12OCT26", "BTC-13OCT26", "BTC-14OCT26", "BTC-PERPETUAL")]
    with pytest.raises(FetchError, match="fewer than two weekly, monthly or quarterly"):
        ff.deribit_curve(daily_only, "BTC")
    with pytest.raises(ValueError, match="BTC, ETH"):
        ff.fetch_deribit_curve(_http({}), "SOL")
    assert np.isfinite(df["price"]).all()
