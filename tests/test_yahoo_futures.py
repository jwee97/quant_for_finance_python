"""The futures Yahoo Finance lists, found by name: the bundled list, the groups it guesses, the lookup that asks Yahoo about the words typed, and the price history of one future.

Nothing here touches the network: Yahoo is a table of answers in the shape of the live service (``tests/free_data_helpers.py``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.assets.free_data import yahoo_futures as yf
from src.assets.free_data.futures import FREE_ROOTS
from src.assets.free_data.http import FetchError, Http
from tests import free_data_helpers as fh

LOOKUP = yf.LOOKUP


def http(routes):
    feeds = fh.FakeFeeds(routes)
    return Http(feeds, sleep=lambda s: None, pause=0.0, retries=0), feeds


# ------------------------------------------------------------------------------------------------------------------ names and groups
@pytest.mark.parametrize("raw, name", [
    ("Crude Oil Nov 26", "Crude Oil"), ("Euro FX Futures,Dec-2026", "Euro FX Futures"), ("Australian Dollar Futures,Dec-2", "Australian Dollar Futures"), ("Gold Dec 26", "Gold"),
    ("10-Year Yield Futures,Oct-2026", "10-Year Yield Futures"), ("BTCV Future OCT 2026 Trading Session", "BTCV"), ("Lumber Futures", "Lumber Futures"), ("", ""),
])
def test_a_name_loses_the_delivery_month_yahoo_puts_on_it(raw, name):
    assert yf.clean_name(raw) == name


@pytest.mark.parametrize("symbol, name, group", [
    ("CL=F", "Crude Oil", "Energy"), ("MZL=F", "Micro Soybean Oil Futures", "Grains and oilseeds"), ("GC=F", "Gold", "Metals"), ("6E=F", "Euro FX Futures", "Currencies"),
    ("M6A=F", "Micro AUD/USD Futures", "Currencies"), ("ES=F", "E-Mini S&P 500", "Equity indexes"), ("ZN=F", "10-Year T-Note Futures", "Rates"), ("SR3=F", "Three-Month SOFR Futures", "Rates"),
    ("LE=F", "Live Cattle Futures", "Livestock and dairy"), ("SB=F", "Sugar #11", "Softs"), ("BTC=F", "Bitcoin Futures", "Crypto"), ("SAAPL=F", "Apple Inc Stock Futures", "Single stocks"),
    ("BCH=F", "Chicago No1 Busheling Ferrous Scrap", "Metals"), ("ANAHM=F", "Micro Anaheim Ducks Index Futures", "Other"),
])
def test_the_group_is_guessed_from_the_name_and_not_from_the_symbol(symbol, name, group):
    assert yf.classify(symbol, name) == group


def test_the_symbols_this_app_reads_are_continuous_series_and_single_contracts_only():
    for ok in ("ES=F", "6E=F", "10Y=F", "CLZ26.NYM", "ZSX27.CBT", "6EZ26.CME"):
        assert yf.SYMBOL.match(ok), ok
    for bad in ("", "es=f", "ES", "ES=X", "^GSPC", "ES=F; rm -rf", "../ES=F", "CLZ26", "A" * 30 + "=F", "CL=F.NYM"):
        assert not yf.SYMBOL.match(bad), bad
    assert yf.kind_of("ES=F") == "continuous" and yf.kind_of("CLZ26.NYM") == "contract"
    assert yf.series_name("ES=F") == "YF_ES" and yf.series_name("6E=F") == "YF_6E" and yf.series_name("CLZ26.NYM") == "YF_CLZ26_NYM"


# ------------------------------------------------------------------------------------------------------------------ the bundled list
def test_the_bundled_list_holds_the_well_known_futures_and_nothing_this_app_cannot_read():
    listed = yf.bundled()
    assert 300 <= len(listed) <= 600 and all(yf.SYMBOL.match(p.symbol) and p.kind == "continuous" and p.name and p.exchange for p in listed)
    assert len({p.symbol for p in listed}) == len(listed)
    symbols = {p.symbol for p in listed}
    assert {"ES=F", "NQ=F", "CL=F", "GC=F", "ZC=F", "6E=F", "BTC=F", "LBR=F", "NKD=F", "SB=F", "KC=F"} <= symbols
    roots = [r for r in FREE_ROOTS if f"{r}=F" not in symbols]
    assert set(roots) <= {"TN", "DX"}, f"roots missing from the list (Yahoo lists these two only by contract): {roots}"
    assert set(yf.groups()) <= set(yf.GROUPS) and {"Energy", "Metals", "Currencies", "Rates", "Equity indexes", "Single stocks"} <= set(yf.groups())
    assert yf.find("es=f").name == "E-Mini S&P 500" and yf.find("nothing") is None


def test_the_search_ranks_exact_symbols_then_prefixes_then_words_and_puts_what_people_look_for_first():
    assert yf.search("es", limit=3)["results"][0]["item"] == "ES=F" and yf.search("ES=F")["results"][0]["item"] == "ES=F"
    crude = [r["item"] for r in yf.search("crude")["results"]]
    assert crude[:2] == ["BZ=F", "CL=F"] or "CL=F" in crude[:3]
    assert [r["item"] for r in yf.search("lumber")["results"]] == ["LBR=F"] and [r["item"] for r in yf.search("nikkei")["results"]][:4] == ["MNI=F", "MNK=F", "NIY=F", "NKD=F"]
    both = yf.search("nasdaq")
    assert "NQ=F" in [r["item"] for r in both["results"]] and both["total"] >= 3 and yf.search("micro nasdaq")["results"][0]["item"] in ("MNQ=F", "MCI=F")
    assert yf.search("zzzzqq") == {"results": [], "total": 0, "shown": 0}
    blank = [r["item"] for r in yf.search("", limit=8)["results"]]
    assert blank[:3] == ["CL=F", "NG=F", "HO=F"], "with nothing typed the roots this app builds curves for come first, in its order"
    only = yf.search("", group="Metals", limit=200)
    assert only["total"] >= 15 and {r["group"] for r in only["results"]} == {"Metals"}
    row = yf.search("gold", limit=1)["results"][0]
    assert set(row) == {"item", "code", "name", "exchange", "group", "kind", "region"} and row["region"] == "YF"


# ------------------------------------------------------------------------------------------------------------------ asking Yahoo
def test_a_lookup_asks_yahoo_for_the_words_typed_keeps_the_answer_and_the_search_then_has_it(tmp_path):
    path = tmp_path / "data" / "user" / "yahoo_futures.json"
    docs = [fh.yahoo_future_doc("OATZ=F", "Oat Futures,Dec-2026", "CBT"), fh.yahoo_future_doc("ES=F", "E-Mini S&P 500 Dec 26"), fh.yahoo_future_doc("OATZ26.CBT", "Oat Futures,Dec-2026", "CBT"),
            fh.yahoo_future_doc("AAPL", "Apple", "NMS", kind="equity"), fh.yahoo_future_doc("bad symbol", "Not a future", "CME")]
    h, feeds = http({LOOKUP: fh.yahoo_lookup(docs)})
    assert yf.search("oat")["results"] == [] or "OATZ=F" not in [r["item"] for r in yf.search("oat", path=path)["results"]]
    out = yf.lookup(h, "  oats ", path)
    assert [r["item"] for r in out["results"]] == ["ES=F", "OATZ=F", "OATZ26.CBT"] or {r["item"] for r in out["results"]} == {"ES=F", "OATZ=F", "OATZ26.CBT"}
    assert out["found"] == 3 and out["new"] == 2 and out["pages"] == 1 and not out["incomplete"], "a page of fewer than 100 is the last, ES=F was in the list already"
    asked = feeds.calls[0]
    assert "query=oats" in asked and "type=future" in asked and "count=100" in asked and "start=0" in asked
    saved = json.loads(path.read_text())
    assert {r["symbol"] for r in saved["products"]} == {"ES=F", "OATZ=F", "OATZ26.CBT"} and saved["fetched_at"].endswith("Z")
    found = yf.search("oat", path=path)
    assert [r["item"] for r in found["results"]][0] == "OATZ=F" and found["results"][0]["name"] == "Oat Futures" and found["results"][0]["exchange"] == "CBT"
    assert any(r["kind"] == "contract" for r in found["results"]), "single contracts are kept and shown as such"
    assert "OATZ=F" not in {p.symbol for p in yf.bundled()}, "the bundled list is not touched"
    assert yf.find("OATZ=F", path).name == "Oat Futures"


def test_a_lookup_pages_while_yahoo_returns_full_pages_and_keeps_what_it_read_when_a_page_fails(tmp_path):
    path = tmp_path / "yahoo.json"
    pages = {0: [fh.yahoo_future_doc(f"A{i}=F", f"Thing {i}") for i in range(100)], 100: [fh.yahoo_future_doc(f"B{i}=F", f"Other {i}") for i in range(100)]}

    def answer(url):
        start = int(url.split("start=")[1].split("&")[0])
        return (429, "slow down") if start >= 200 else fh.yahoo_lookup(pages.get(start, []))

    h, feeds = http({LOOKUP: answer})
    out = yf.lookup(h, "thing", path)
    assert out["found"] == 200 and out["pages"] == 2 and out["incomplete"] is True and len(feeds.calls) == 3
    assert len(json.loads(path.read_text())["products"]) == 200, "what was read before the refusal is kept"
    h, _ = http({LOOKUP: (429, "slow down")})
    with pytest.raises(FetchError):
        yf.lookup(h, "thing", tmp_path / "none.json")
    assert not (tmp_path / "none.json").exists(), "nothing read, nothing written"
    for bad in ("", "x" * 41, "   "):
        with pytest.raises(ValueError, match="a word or two"):
            yf.lookup(h, bad, path)


def test_a_damaged_list_file_is_ignored(tmp_path):
    path = tmp_path / "yahoo.json"
    path.write_text("{not json")
    assert len(yf.catalog(path)) == len(yf.bundled())
    path.write_text(json.dumps({"products": [{"symbol": "OK=F", "name": "Fine", "exchange": "CME", "group": "Other"}, {"nothing": 1}, "text"]}))
    assert {p.symbol for p in yf.catalog(path)} - {p.symbol for p in yf.bundled()} == {"OK=F"}


# ------------------------------------------------------------------------------------------------------------------ one future's history
def test_a_continuous_series_downloads_for_the_history_view_with_notes_that_say_what_it_is():
    days = [str(d.date()) for d in pd.bdate_range(end="2026-10-09", periods=400)]
    h, feeds = http({yf.CHART.format(symbol="LBR=F"): lambda url: fh.yahoo_chart("LBR=F", days, [500.0 + 0.5 * i for i in range(400)])})
    d = yf.fetch_series(h, "lbr=f", 5.0)
    assert d.kind == "futures" and d.item == "series" and d.name == "YF_LBR" and list(d.frame.columns) == ["date", "close", "high", "low", "volume"] and len(d.frame) == 400
    assert d.meta["region"] == "YF" and d.meta["code"] == "LBR=F" and d.meta["root_name"] == "Lumber Futures" and d.meta["source"] == "Yahoo Finance" and d.meta["bars"] == 400
    assert d.meta["first_date"] == days[0] and d.meta["as_of"] == days[-1] and d.meta["years"] == 5.0
    text = " ".join(d.meta["notes"])
    assert "rolled by Yahoo" in text and "not back-adjusted" in text and "no curve, roll or carry" in text and "no open interest" in text
    assert "range=5y" in feeds.calls[0] and "interval=1d" in feeds.calls[0] and len(feeds.calls) == 1, "one call"
    h, feeds = http({yf.CHART.format(symbol="CLZ26.NYM"): lambda url: fh.yahoo_chart("CLZ26.NYM", days, [90.0 + 0.1 * i for i in range(400)])})
    one = yf.fetch_series(h, "CLZ26.NYM", 1.0)
    assert one.name == "YF_CLZ26_NYM" and "one contract" in " ".join(one.meta["notes"]) and "range=1y" in feeds.calls[0]


def test_a_history_that_is_too_short_or_missing_or_not_a_future_is_refused_with_the_reason():
    days = [str(d.date()) for d in pd.bdate_range(end="2026-10-09", periods=10)]
    h, _ = http({yf.CHART.format(symbol="ES=F"): fh.yahoo_chart("ES=F", days, [5000.0] * 10), yf.CHART.format(symbol="ZZ=F"): (404, fh.YAHOO_MISSING)})
    with pytest.raises(FetchError, match="only 10 daily prices"):
        yf.fetch_series(h, "ES=F")
    with pytest.raises(FetchError):
        yf.fetch_series(h, "ZZ=F")
    for bad in ("AAPL", "ES=X", "../x"):
        with pytest.raises(ValueError, match="not a futures symbol"):
            yf.fetch_series(h, bad)
