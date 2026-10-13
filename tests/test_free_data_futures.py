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


# ------------------------------------------------------------------------------------------------------------------ the wider list of roots
def test_the_newer_expiry_rules_land_on_the_days_the_exchanges_publish():
    cases = [("last_bday", 2026, 12, "2026-12-31"), ("last_bday_prior", 2026, 12, "2026-11-30"), ("last_bday_prior", 2027, 1, "2026-12-31"),         # heating oil, gasoline and sugar stop at the end of the month before delivery
             ("brent", 2026, 12, "2026-10-30"), ("brent", 2027, 1, "2026-11-30"), ("tenth_bday", 2026, 12, "2026-12-14"), ("tenth_bday", 2026, 8, "2026-08-14"),
             ("last_thursday", 2026, 12, "2026-12-31"), ("last_thursday", 2026, 10, "2026-10-29"), ("cad", 2026, 12, "2026-12-15"), ("coffee", 2026, 12, "2026-11-19"),
             ("cocoa", 2026, 12, "2026-12-16"), ("cotton", 2026, 12, "2026-12-08"), ("oj", 2026, 11, "2026-11-10")]
    for rule, year, month, expected in cases:
        assert str(ff.last_trade_date(rule, year, month).date()) == expected, (rule, year, month)


def test_every_free_root_has_a_known_rule_real_terms_and_a_group_and_names_a_yahoo_symbol():
    assert len(ff.FREE_ROOTS) == 44 and all(ff.ROOT.match(r) for r in ff.FREE_ROOTS)
    groups = {r.group for r in ff.FREE_ROOTS.values()}
    assert groups == {"Energy", "Metals", "Grains", "Livestock", "Softs", "Equity indexes", "Rates", "Currencies"}
    today = pd.Timestamp(TODAY)
    for root, spec in ff.FREE_ROOTS.items():
        assert spec.root == root and spec.multiplier > 0 and spec.tick_size > 0 and spec.tick_value > 0 and spec.asset_class in fa.MARGIN_SHARE and spec.exchange and spec.name
        assert set(spec.months) <= set(range(1, 13)) and list(spec.months) == sorted(spec.months)
        out = ff.upcoming_contracts(spec, today, 6)
        assert len(out) == 6 and [c["expiry"] for c in out] == sorted(c["expiry"] for c in out), root
        assert all(c["expiry"] >= today and c["symbol"] == f"{root}{c['contract']}{spec.suffix}" and c["expiry"].weekday() < 5 for c in out), root
        assert not set(spec.terms()) & {"months", "rule", "suffix", "group"} and spec.terms()["root_name"] == spec.name
    assert {r.suffix for r in ff.FREE_ROOTS.values()} == {".NYM", ".CMX", ".CBT", ".CME", ".NYB"}
    # contract sizes spot-checked against the exchanges' specifications: 42,000 gallons of heating oil, 5,000 bushels of corn quoted in cents, $5 a point on the Dow, 12.5 million yen
    assert ff.FREE_ROOTS["HO"].multiplier == 42000 and ff.FREE_ROOTS["ZC"].multiplier == 50 and ff.FREE_ROOTS["YM"].multiplier == 5 and ff.FREE_ROOTS["6J"].multiplier == 12_500_000
    assert round(ff.FREE_ROOTS["6J"].tick_value, 2) == 6.25 and round(ff.FREE_ROOTS["ZT"].tick_value, 4) == 7.8125 and round(ff.FREE_ROOTS["LE"].tick_value, 2) == 10.0


@pytest.mark.parametrize("root", sorted(ff.FREE_ROOTS))
def test_each_root_downloads_through_the_same_code_path(root):
    seen = []

    def route(url):
        symbol = url.split("/chart/")[1].split("?")[0]
        seen.append(symbol)
        shift = sum(map(ord, symbol)) % 5
        return yahoo_chart(symbol, _days(30), [50.0 + shift + 0.1 * i for i in range(30)])

    d = ff.fetch_yahoo_curve(_http({"https://query1.finance.yahoo.com/v8/finance/chart/": route}), root, 3, "1y", today=TODAY)
    spec = ff.FREE_ROOTS[root]
    assert d.name == root and d.meta["contracts"] == 3 and d.meta["multiplier"] == spec.multiplier and d.meta["exchange"] == spec.exchange and d.meta["asset_class"] == spec.asset_class
    assert all(x.startswith(root) and x.endswith(spec.suffix) for x in seen) and d.frame["contract"].nunique() == 3


# ------------------------------------------------------------------------------------------------------------------ margins of a root the platform does not simulate
def test_a_downloaded_root_without_known_margins_gets_an_illustrative_share_of_its_value_and_says_so(tmp_path):
    d = ff.fetch_yahoo_curve(_http({"https://query1.finance.yahoo.com/v8/finance/chart/": lambda url: yahoo_chart(url.split("/chart/")[1].split("?")[0], _days(30), [2000.0 + i for i in range(30)])}), "PL", 3, "1y", today=TODAY)
    path = tmp_path / "PL.csv"
    d.frame.to_csv(path, index=False)
    spec = fa.RootSpec("PL", "Platinum", "NYMEX", "metal", 50.0, 0.10, tuple(range(1, 13)), "last_business_day", fa.SchwartzSmithParams(), 0.0, 0.0)
    data = fa.load_curve_file(path, "PL", spec=spec, source="downloaded")
    front = float(data.long[data.long["date"] == data.long["date"].max()].sort_values("expiry")["price"].iloc[0])
    assert data.spec.initial_margin == pytest.approx(0.06 * front * 50.0, rel=0.01) and data.spec.maintenance_margin == pytest.approx(0.9 * data.spec.initial_margin, rel=0.01)
    assert any("ILLUSTRATIVE 6%" in n for n in data.notes) and not any("shown as zero" in n for n in data.notes)
    unknown = fa.RootSpec("PL", "Platinum", "NYMEX", "mystery", 50.0, 0.10, tuple(range(1, 13)), "last_business_day", fa.SchwartzSmithParams(), 0.0, 0.0)
    again = fa.load_curve_file(path, "PL", spec=unknown, source="downloaded")
    assert again.spec.initial_margin == 0.0 and any("shown as zero" in n for n in again.notes), "a class the platform has no share for stays unknown, and says so"
    assert fa.illustrative_margin("rates", 100_000) == (2000.0, 1800.0) and fa.illustrative_margin("metal", 0) is None and fa.illustrative_margin("nope", 1e5) is None
