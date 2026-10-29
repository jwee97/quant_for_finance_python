"""The deeper downloads: every contract Yahoo lists (not only the nearest few), the text getter the CSV sources use, and Deribit's daily charts of the dated futures, expired ones included. The network is a
table of answers in the shape of the live feeds (Yahoo's chart answer, Deribit's ``get_tradingview_chart_data`` and ``get_index_chart_data``)."""

from __future__ import annotations

from urllib import parse

import pandas as pd
import pytest

from src.assets.free_data import futures as ff
from src.assets.free_data.http import FetchError, Http
from tests.free_data_helpers import YAHOO_MISSING, FakeFeeds, yahoo_chart

TODAY = "2026-10-11"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
CHART = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
INDEX = "https://www.deribit.com/api/v2/public/get_index_chart_data"


def _http(routes):
    feeds = FakeFeeds(routes)
    return Http(feeds, sleep=lambda s: None, pause=0.0, retries=0), feeds


def _listing(served: set[str], root: str = "CL", suffix: str = ".NYM", bars: int = 40, far_bars: int = 5, far_from: str = "Z31"):
    """Yahoo's chart route for a root whose served contracts are the codes in ``served`` (``Z31``); the others answer 404. Contracts at or after ``far_from`` have only ``far_bars`` bars."""
    far = ff.MONTH_CODES.index(far_from[0]), int(far_from[1:])

    def answer(url: str):
        symbol = parse.unquote(url.split("/chart/")[1].split("?")[0])
        code = symbol[len(root):-len(suffix)]
        if code not in served:
            return 404, YAHOO_MISSING
        n = far_bars if (int(code[1:]), ff.MONTH_CODES.index(code[0])) >= (far[1], far[0]) else bars
        dates = [str(d.date()) for d in pd.bdate_range(end="2026-10-09", periods=n)]
        return yahoo_chart(symbol, dates, [60.0 + 0.5 * i for i in range(n)])
    return {YAHOO + root: answer}


def _wti_cycle() -> set[str]:
    """What Yahoo lists for crude oil in this fixture: every month to December 2027, then June and December only to December 2033."""
    out = set()
    for year in range(2026, 2034):
        for month in range(1, 13):
            if (year, month) < (2026, 11) or (year > 2027 and month not in (6, 12)):
                continue
            out.add(f"{ff.MONTH_CODES[month - 1]}{year % 100:02d}")
    return out


# ------------------------------------------------------------------------------------------------------------------ every contract Yahoo lists
def test_a_deep_download_takes_every_contract_yahoo_lists_across_the_gaps_of_its_far_listing_and_stops_two_years_after_the_last():
    served = _wti_cycle()
    http, feeds = _http(_listing(served))
    d = ff.fetch_yahoo_curve(http, "CL", None, "max", today=TODAY)
    last_expiry = ff.last_trade_date("cl", 2033, 12).normalize()
    assert set(d.frame["contract"]) == served and d.meta["contracts"] == len(served) and d.meta["deep"] is True and d.meta["furthest"] == "Z33" and d.meta["furthest_expiry"] == str(last_expiry.date())
    candidates = ff.upcoming_contracts(ff.FREE_ROOTS["CL"], TODAY, ff.MAX_CONTRACTS)
    assert len(feeds.calls) == sum(1 for c in candidates if c["expiry"] <= last_expiry + pd.Timedelta(days=ff.GAP_DAYS)), "it asked for every month up to two years after the last contract it found, and no further"
    assert len(feeds.calls) < 140 and all("range=max" in u for u in feeds.calls)
    assert any(f"lists {len(served)} CL contracts" in n and "the far end of the curve has gaps" in n for n in d.meta["notes"])


def test_the_far_contracts_that_trade_rarely_are_named_in_the_notes_and_in_the_record():
    served = _wti_cycle()
    d = ff.fetch_yahoo_curve(*_http(_listing(served, far_from="Z31", far_bars=6))[:1], "CL", None, "5y", today=TODAY)
    sparse = set(d.meta["sparse_contracts"])
    assert "Z31" in sparse and "Z33" in sparse and "F27" not in sparse and sparse <= served
    assert any("trade rarely" in n and "fewer than 20 daily bars" in n for n in d.meta["notes"])


def test_a_gap_shorter_than_two_years_in_the_listing_does_not_end_the_search_and_a_longer_one_does():
    served = {f"{ff.MONTH_CODES[m - 1]}27" for m in range(1, 13)} | {"Z26", "X26"} | {f"{ff.MONTH_CODES[m - 1]}29" for m in range(10, 13)}      # nothing from January 2028 to September 2029: about 22 months
    d = ff.fetch_yahoo_curve(*_http(_listing(served))[:1], "CL", None, "1y", today=TODAY)
    assert set(d.frame["contract"]) == served, "the 22-month hole did not end the search"
    served_far = {"Z26", "X26", "F27"} | {"V31"}                                                                                           # a contract four years after the others
    d2 = ff.fetch_yahoo_curve(*_http(_listing(served_far))[:1], "CL", None, "1y", today=TODAY)
    assert set(d2.frame["contract"]) == {"Z26", "X26", "F27"}, "two years of nothing ended the search before it"


def test_a_deep_download_of_a_root_yahoo_does_not_serve_fails_after_a_year_of_looking_and_a_zero_means_the_same_as_none():
    http, feeds = _http({})
    with pytest.raises(FetchError, match="no price history"):
        ff.fetch_yahoo_curve(http, "CL", 0, "1y", today=TODAY)
    assert len(feeds.calls) <= 15, "it gave up after about a year of delivery months, not after 160 asks"
    served = {"Z26", "F27"}
    a = ff.fetch_yahoo_curve(*_http(_listing(served))[:1], "CL", 0, "1y", today=TODAY)
    b = ff.fetch_yahoo_curve(*_http(_listing(served))[:1], "CL", None, "1y", today=TODAY)
    assert set(a.frame["contract"]) == set(b.frame["contract"]) == served and a.meta["deep"] and b.meta["deep"]


def test_the_number_of_contracts_and_the_history_range_are_checked_and_the_old_behaviour_stays():
    http, feeds = _http(_listing({"X26", "Z26", "F27", "G27"}))
    d = ff.fetch_yahoo_curve(http, "CL", 4, "10y", today=TODAY)
    assert d.meta["deep"] is False and set(d.frame["contract"]) == {"X26", "Z26", "F27", "G27"} and len(feeds.calls) == 4 and all("range=10y" in u for u in feeds.calls)
    for bad in (ff.MAX_CONTRACTS + 1, 1000):
        with pytest.raises(ValueError, match="n_contracts must be"):
            ff.fetch_yahoo_curve(http, "CL", bad, "1y", today=TODAY)
    for bad in ("3y", "forever", ""):
        with pytest.raises(ValueError, match="history must be one of"):
            ff.fetch_yahoo_curve(http, "CL", 4, bad, today=TODAY)


def test_upcoming_contracts_looks_as_many_years_ahead_as_the_number_asked_for_needs():
    quarterly = ff.upcoming_contracts(ff.FREE_ROOTS["ES"], TODAY, 100)
    assert len(quarterly) == 100 and quarterly[0]["contract"] == "Z26" and quarterly[-1]["expiry"].year >= 2051
    assert [c["expiry"] for c in quarterly] == sorted(c["expiry"] for c in quarterly)
    monthly = ff.upcoming_contracts(ff.FREE_ROOTS["CL"], TODAY, 150)
    assert len(monthly) == 150 and monthly[0]["contract"] == "X26" and len(ff.upcoming_contracts(ff.FREE_ROOTS["CL"], TODAY, 12)) == 12


# ------------------------------------------------------------------------------------------------------------------ the text getter
def test_the_text_getter_returns_a_csv_decoded_without_its_byte_order_mark_and_shares_the_error_handling():
    feeds = FakeFeeds({"https://cdn.example/ok.csv": "﻿a,b\n1,2\n".encode("utf-8"), "https://cdn.example/refused": (403, "no"), "https://cdn.example/missing": (404, "no")})
    http = Http(feeds, sleep=lambda s: None, pause=0.0, retries=0)
    assert http.get_text("https://cdn.example/ok.csv") == "a,b\n1,2\n"
    with pytest.raises(FetchError) as refused:
        http.get_text("https://cdn.example/refused")
    assert refused.value.blocked and refused.value.status == 403
    with pytest.raises(FetchError, match="not asked again"):
        http.get_text("https://cdn.example/ok.csv")                                                  # the refusal tripped the host for the cool-down, as it does for JSON
    assert Http(FakeFeeds({"https://cdn.example/x.json": "<html>"}), sleep=lambda s: None, pause=0.0, retries=0).get_text("https://cdn.example/x.json") == "<html>"
    with pytest.raises(FetchError, match="not JSON"):
        Http(FakeFeeds({"https://cdn.example/x.json": "<html>"}), sleep=lambda s: None, pause=0.0, retries=0).get_json("https://cdn.example/x.json")


# ------------------------------------------------------------------------------------------------------------------ Deribit's daily charts
def _ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def _chart_answer(first_trade: str, expiry: str, last_price: float, lead: int = 3, tail: int = 4, today: str = TODAY) -> dict:
    """Deribit's daily chart of one future: ``lead`` flat days before the first trade, a ramp of traded days to the expiry (or to today, if the future is still alive), then, for an expired one, a flat line
    of ``tail`` days after it (the chart goes on being drawn)."""
    alive = pd.Timestamp(expiry) > pd.Timestamp(today)
    traded = [str(d.date()) for d in pd.date_range(first_trade, pd.Timestamp(today) - pd.Timedelta(days=1) if alive else expiry)]
    before = [str(d.date()) for d in pd.date_range(end=pd.Timestamp(first_trade) - pd.Timedelta(days=1), periods=lead)]
    after = [] if alive else [str(d.date()) for d in pd.date_range(pd.Timestamp(expiry) + pd.Timedelta(days=1), periods=tail)]
    days = before + traded + after
    price = [100.0] * len(before) + [100.0 + i for i in range(len(traded))] + [last_price] * len(after)
    volume = [0.0] * len(before) + [5.0 + i for i in range(len(traded))] + [0.0] * len(after)
    cost = [v * 1000.0 for v in volume]
    return {"jsonrpc": "2.0", "result": {"status": "ok", "ticks": [_ms(d) for d in days], "open": price, "high": price, "low": price, "close": price, "volume": volume, "cost": cost}}


def _deribit_routes(existing: dict, index: bool = True):
    def chart(url: str):
        name = parse.parse_qs(parse.urlsplit(url).query)["instrument_name"][0]
        if name not in existing:
            return 400, {"jsonrpc": "2.0", "error": {"code": 13020, "message": "instrument_not_found"}}
        return _chart_answer(*existing[name])

    routes = {CHART: chart}
    if index:
        days = [str(d.date()) for d in pd.date_range("2025-11-01", "2026-10-11")]
        routes[INDEX] = {"jsonrpc": "2.0", "result": [[_ms(d) + h * 3_600_000, 80000.0 + i + h] for i, d in enumerate(days) for h in (0, 6, 12, 18)]}
    return routes


def test_a_deribit_chart_loses_what_comes_before_the_first_trade_and_after_the_expiry():
    chart = _chart_answer("2026-03-02", "2026-03-27", 130.0)["result"]
    frame = ff.deribit_chart(chart, "2026-03-27")
    assert frame["date"].iloc[0] == pd.Timestamp("2026-03-02") and frame["date"].iloc[-1] == pd.Timestamp("2026-03-27") and len(frame) == 26
    assert frame["price"].iloc[0] == 100.0 and frame["price"].iloc[-1] == 125.0 and frame["volume"].iloc[0] == 5000.0, "the volume is the dollar cost of the day's trades"
    for bad in ({}, {"status": "no_data", "ticks": []}, {"status": "ok", "ticks": [_ms("2026-01-02")], "close": [], "volume": []}):
        with pytest.raises(FetchError, match="no daily prices"):
            ff.deribit_chart(bad, "2026-03-27")
    flat = {"status": "ok", "ticks": [_ms("2026-01-02"), _ms("2026-01-03")], "close": [100.0, 100.0], "volume": [0.0, 0.0], "cost": [0.0, 0.0]}
    with pytest.raises(FetchError, match="no trade"):
        ff.deribit_chart(flat, "2026-03-27")


def test_the_history_stacks_the_monthly_and_quarterly_futures_with_their_expiries_and_the_index_as_spot(tmp_path):
    existing = {"BTC-27FEB26": ("2025-12-01", "2026-02-27", 99.0), "BTC-27MAR26": ("2025-12-29", "2026-03-27", 130.0), "BTC-26JUN26": ("2026-01-30", "2026-06-26", 90.0),
                "BTC-25SEP26": ("2026-03-27", "2026-09-25", 95.0), "BTC-30OCT26": ("2026-04-24", "2026-10-30", 96.0)}
    http, feeds = _http(_deribit_routes(existing))
    messages = []
    d = ff.fetch_deribit_history(http, "btc", today=TODAY, first_year=2026, progress=messages.append)
    df = d.frame
    assert d.kind == "futures" and d.name == "BTC" and d.item == "curve" and d.meta["source_id"] == "deribit" and d.meta["history"] is True and d.meta["contracts"] == 5
    assert list(df.columns) == ["date", "contract", "expiry", "price", "volume", "spot"] and set(df["contract"]) == {"27FEB26", "27MAR26", "26JUN26", "25SEP26", "30OCT26"}
    assert (df["date"] <= df["expiry"]).all(), "nothing after an expiry"
    one = df[df["contract"] == "27MAR26"]
    assert one["expiry"].iloc[0] == pd.Timestamp("2026-03-27") and one["date"].min() == pd.Timestamp("2025-12-29") and df["spot"].notna().all()
    day = df[df["date"] == "2026-02-02"].sort_values("expiry")
    assert list(day["contract"]) == ["27FEB26", "27MAR26", "26JUN26"], "on a day in the past the strip is the contracts that were alive then, expired ones included"
    assert day["spot"].nunique() == 1 and day["spot"].iloc[0] == pytest.approx(80000.0 + (pd.Timestamp("2026-02-02") - pd.Timestamp("2025-11-01")).days + 18), "the last reading of the day"
    assert any("no chart for" in n and "no future of that name existed" in n for n in d.meta["notes"]), "the month-end Fridays Deribit never listed are said, not hidden"
    assert any("no open interest" in n for n in d.meta["notes"]) and messages[0].startswith("BTC: ") and d.meta["first_date"] == "2025-12-01"
    assert all(parse.parse_qs(parse.urlsplit(u).query).get("resolution", ["1D"])[0] == "1D" for u in feeds.calls if u.startswith(CHART))
    path = tmp_path / "BTC.csv"
    df.to_csv(path, index=False)
    from src.assets import futures_analytics as fa
    curve = fa.load_curve_file(path, "BTC")                                                                                      # the file the Futures workspace reads
    assert curve.long["contract"].nunique() >= 4 and curve.spot is not None


def test_a_deribit_history_without_the_index_still_downloads_and_says_it_has_no_spot():
    existing = {"BTC-27MAR26": ("2025-12-29", "2026-03-27", 130.0), "BTC-26JUN26": ("2026-01-30", "2026-06-26", 90.0)}
    d = ff.fetch_deribit_history(_http(_deribit_routes(existing, index=False))[0], "BTC", today=TODAY, first_year=2026)
    assert "spot" not in d.frame.columns and any("index price could not be read" in n for n in d.meta["notes"])


def test_deribit_refusing_this_address_stops_the_history_and_nothing_at_all_is_an_error():
    http, feeds = _http({CHART: (403, "no")})
    with pytest.raises(FetchError, match="refused"):
        ff.fetch_deribit_history(http, "ETH", today=TODAY, first_year=2026)
    assert len(feeds.calls) == 1, "one refusal ended the asking"
    with pytest.raises(FetchError, match="no daily chart for any ETH"):
        ff.fetch_deribit_history(_http(_deribit_routes({}))[0], "ETH", today=TODAY, first_year=2026)
    with pytest.raises(ValueError, match="available for BTC, ETH"):
        ff.fetch_deribit_history(_http({})[0], "SOL")


def test_the_month_end_fridays_are_the_days_deribits_monthly_futures_expire():
    days = ff.deribit_expiries("2026-10-11", first_year=2026, months_ahead=4)
    assert days[0] == pd.Timestamp("2026-01-30") and all(d.weekday() == 4 for d in days) and days[-1] == pd.Timestamp("2027-01-29")
    assert all((d + pd.Timedelta(days=7)).month != d.month for d in days), "each is the last Friday of its month"
    assert pd.Timestamp("2026-10-30") in days and pd.Timestamp("2026-12-25") in days


def test_the_index_series_is_the_last_reading_of_each_day():
    series = ff.deribit_index([[_ms("2026-03-02") + 0, 10.0], [_ms("2026-03-02") + 6 * 3_600_000, 11.0], [_ms("2026-03-03"), 12.0], [_ms("2026-03-03") + 1000, None]])
    assert list(series.index) == [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")] and list(series.values) == [11.0, 12.0]
    with pytest.raises(FetchError, match="empty"):
        ff.deribit_index([])
