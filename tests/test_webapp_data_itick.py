"""iTick as a source for the Futures, Crypto and Options tabs: the daily price history of any futures product, a coin's daily history and an option underlying's daily prices.

Nothing here touches the network: iTick is the fake transport of ``tests/fake_itick.py`` behind the same provider the Backtest tab uses (so the same rate limiter), on a fake clock. What cannot be
tested offline is that the real service answers the way its documentation says; the Backtest tab's "Test the iTick connection" button is how a user checks that with their own key.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.assets.free_data.http import Http
from src.webapp.api import ApiError
from src.webapp.providers import ITickProvider, RateLimiter
from src.webapp.universe import itick_source
from tests import free_data_helpers as fh
from tests.fake_itick import KEY, FakeClock, FakeITick
from tests.test_webapp_assets_ml import _app, wait
from tests.test_webapp_data import CBOE, CLOSES, DAYS, YAHOO, _crypto_routes, _safe, feeds, run
from tests.test_webapp_labs import _rich


def _provider(clock, fake, key=KEY, root=None, **kwargs):
    return ITickProvider(key=key, root=root, base_url="https://api-free.itick.org", limiter=RateLimiter(kwargs.pop("calls", 5), clock=clock.now, sleep=clock.sleep), transport=fake, sleep=clock.sleep, **kwargs)


@pytest.fixture
def ik(tmp_path):
    """An app whose iTick source talks to a fake service (futures bars stamped the evening before, like the real exchange's trading day) through a fake clock: ``(app, fake, clock)``."""
    app = _app(tmp_path, _rich, root=tmp_path)
    app.data.http = Http(fh.FakeFeeds({}), sleep=lambda s: None, pause=0.0, retries=0)
    clock, fake = FakeClock(), FakeITick(stamp="evening_prior")
    app.builder.sources["itick"] = itick_source(tmp_path, _provider(clock, fake), tmp_path / "itick")
    return app, fake, clock


def _use(app, provider, tmp_path):
    app.builder.sources["itick"] = itick_source(tmp_path, provider, tmp_path / "itick")


# ------------------------------------------------------------------------------------------------------------------ the status and the search
def test_the_status_reports_itick_without_the_key_and_the_search_costs_no_call(ik):
    app, fake, _ = ik
    st = _safe(app.extra_get["/api/data/status"]({}))
    assert st["itick"]["configured"] is True and st["itick"]["calls_per_minute"] == 5 and st["itick"]["calls_available_now"] == 5 and st["itick"]["page_size"] == 1000 and st["itick"]["setup"] == ""
    assert KEY not in json.dumps(st)
    for kind in ("options", "futures", "crypto"):
        assert "itick" in [s["id"] for s in st["kinds"][kind]["sources"]] and st["kinds"][kind]["itick_status"]["configured"], kind
    itick = next(s for s in st["kinds"]["futures"]["sources"] if s["id"] == "itick")
    assert itick["needs_key"] and "no option data" in itick["provides"] and "curves" in itick["provides"], "the card says plainly what iTick does not have"
    fut = st["kinds"]["futures"]
    assert len(fut["roots"]) == 45 and "VX" in [r["id"] for r in fut["roots"]] and fut["groups"][0] == "Energy" and {"Metals", "Grains", "Livestock", "Softs", "Equity indexes", "Rates", "Currencies"} <= set(fut["groups"])
    assert {r["id"] for r in fut["itick"]["regions"]} == {"US", "HK", "CN"} and fut["itick"]["products"] > 1500 and fut["itick"]["defaults"] == {"years": 5}
    assert st["kinds"]["crypto"]["itick"]["coins"] > 150 and st["kinds"]["options"]["prices"]["defaults"]["years"] == 3

    nasdaq = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "q": "nasdaq"}))
    assert nasdaq["results"][0]["item"] == "US:NQ" and nasdaq["results"][0]["featured"] and nasdaq["total"] >= 4
    hk = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "q": "hang seng", "region": "hk"}))
    assert [r["item"] for r in hk["results"]][:2] == ["HK:HSI", "HK:MHI"] and {r["region"] for r in hk["results"]} == {"HK"}
    metals = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "q": "", "group": "Metals", "limit": 10}))
    assert len(metals["results"]) == 10 and {r["group"] for r in metals["results"]} == {"Metals"} and metals["total"] > 50 and metals["results"][0]["item"] == "US:GC"
    featured = _safe(app.extra_get["/api/data/search"]({"kind": "futures", "limit": 100}))
    assert featured["results"][0]["item"] == "US:ES" and all(r["featured"] for r in featured["results"]) and "US:CL" in [r["item"] for r in featured["results"]]
    coins = _safe(app.extra_get["/api/data/search"]({"kind": "crypto", "q": "sol"}))
    assert coins["results"][0]["coin"] == "SOL"
    assert fake.calls == [], "the product list is bundled: searching spends none of the plan's calls"
    for bad, text in (({"kind": "options"}, "kind must be"), ({"kind": "futures", "region": "EU"}, "region must be"), ({"kind": "futures", "limit": "0"}, "between")):
        with pytest.raises(ApiError, match=text):
            app.extra_get["/api/data/search"](bad)


# ------------------------------------------------------------------------------------------------------------------ futures
def test_a_futures_product_downloads_from_itick_and_the_history_view_reads_it(ik):
    app, fake, _ = ik
    out = _safe(run(app, kind="futures", source="itick", items=["us:nq", "HK:HSI"], years=3))
    by = {r["item"]: r for r in out["results"]}
    assert out["ok"] == 2 and by["US:NQ"]["name"] == "US_NQ" and by["HK:HSI"]["name"] == "HK_HSI" and by["US:NQ"]["what"] == "series" and "daily bars" in by["US:NQ"]["message"]
    assert {(c["region"], c["code"]) for c in fake.calls} == {("US", "NQ"), ("HK", "HSI")} and {c["kType"] for c in fake.calls} == {"8"}
    assert all(u.startswith("https://api-free.itick.org/future/kline?") for u in fake.urls), "futures go to the futures endpoint"
    assert set(fake.tokens) == {KEY} and all(KEY not in u for u in fake.urls)
    rec = app.data.store.meta("futures", "US_NQ")
    assert rec["item"] == "series" and rec["source_id"] == "itick" and rec["root_name"].startswith("E-mini Nasdaq") and rec["exchange"] == "CME" and rec["group"] == "Indexes"
    assert rec["refresh"] == {"source": "itick", "items": ["US:NQ"], "years": 3} and KEY not in json.dumps(rec)
    assert any("no forward curve" in n for n in rec["notes"]) and any("joined" in n for n in rec["notes"]), "the record says what a product series is not"
    frame = pd.read_csv(app.data.store.path("futures", "US_NQ"), parse_dates=["date"])
    assert list(frame.columns) == ["date", "close", "high", "low", "volume"] and (frame["date"].dt.dayofweek < 5).all() and frame["date"].is_unique

    cat = _safe(app.extra_get["/api/assets/catalog"]({}))
    assert {s["id"] for s in cat["futures_series"]} == {"US_NQ", "HK_HSI"} and {s["id"] for s in cat["futures_roots"]}.isdisjoint({"US_NQ", "HK_HSI"}), "a price history is not a root with a curve"
    nq = next(s for s in cat["futures_series"] if s["id"] == "US_NQ")
    assert nq["code"] == "NQ" and nq["region"] == "US" and nq["bars"] == len(frame) and nq["provider"] == "iTick"

    view = _safe(app.extra_get["/api/futures/series"]({"name": "US_NQ"}))
    assert view["source"] == "downloaded" and view["sources"] == {"prices": "downloaded"} and view["code"] == "NQ" and view["label"].startswith("E-mini Nasdaq")
    st = view["stats"]
    assert st["bars"] == len(frame) and st["last_date"] == str(frame["date"].iloc[-1].date()) and st["max_drawdown"] < 0 and st["vol_1y"] > 0.05 and len(view["dates"]) == len(view["close"]) == len(view["drawdown"]) == len(view["vol_21"])
    assert len(view["moves"]) == 8 and abs(view["moves"][0]["change"]) >= abs(view["moves"][-1]["change"]) and any("iTick" in n for n in view["notes"])
    assert view["stats"]["first_date"] == str(frame["date"].iloc[0].date()) and len(view["seasonal"]) == 12 and {m["n"] for m in view["seasonal"]} <= {2, 3}, "three years give each calendar month two or three observations, and the table says how many"
    with pytest.raises(ApiError, match="no downloaded price history 'US_ZZ'"):
        app.extra_get["/api/futures/series"]({"name": "US_ZZ"})
    with pytest.raises(ApiError, match="no downloaded price history"):
        app.extra_get["/api/futures/series"]({"name": "../x"})
    curve_error = pytest.raises(ApiError, match="unknown root")
    with curve_error:
        app.extra_get["/api/futures/curve"]({"root": "US_NQ"})

    # the Update button posts the stored request back as it is, and the Delete button removes the file and the record
    again = run(app, kind="futures", **rec["refresh"])
    assert again["ok"] == 1 and again["results"][0]["name"] == "US_NQ"
    gone = app.extra_post["/api/data/delete"]({"kind": "futures", "name": "US_NQ", "confirm": True})
    assert [d["name"] for d in gone["downloads"]] == ["HK_HSI"]


def test_an_update_asks_only_for_the_newest_bars_and_a_history_that_changed_is_fetched_again(ik):
    app, fake, _ = ik
    run(app, kind="futures", source="itick", items=["US:CL"], years=10)
    first = len(fake.calls)
    assert first >= 3, "ten years of daily bars are several pages"
    again = run(app, kind="futures", source="itick", items=["US:CL"], years=10)
    assert len(fake.calls) - first == 1 and again["ok"] == 1, "an update is one call"
    assert len(pd.read_csv(app.data.store.path("futures", "US_CL"))) > 2000

    fake.restate = 1.5                                                    # iTick restated its history: the saved bars no longer match
    before = len(fake.calls)
    run(app, kind="futures", source="itick", items=["US:CL"], years=10)
    assert len(fake.calls) - before >= 3, "a history that does not line up with the saved one is downloaded again"
    frame = pd.read_csv(app.data.store.path("futures", "US_CL"))
    assert frame["close"].iloc[-1] / frame["close"].iloc[0] > 0 and frame["date"].is_unique

    shorter = run(app, kind="futures", source="itick", items=["US:CL"], years=20)
    assert shorter["ok"] == 1, "asking for more years than the saved file holds downloads the history again, not a top-up"


def test_the_plans_call_allowance_is_shared_and_a_long_download_waits_for_it(ik):
    app, fake, clock = ik
    items = ["US:ES", "US:NQ", "US:CL", "US:GC", "US:SI", "US:ZN", "US:ZB"]
    out = run(app, kind="futures", source="itick", items=items, years=1)
    assert out["ok"] == 7 and len(fake.calls) == 7
    assert sum(clock.slept) >= 55, "five calls a minute: the sixth and seventh wait for the first to age out"
    st = app.extra_get["/api/data/status"]({})["itick"]
    assert st["calls_per_minute"] == 5


def test_the_ceiling_on_calls_in_one_run_is_checked_before_anything_is_asked(tmp_path):
    clock, fake = FakeClock(), FakeITick()
    app = _app(tmp_path, _rich, root=tmp_path)
    _use(app, _provider(clock, fake, max_calls_per_run=3), tmp_path)
    with pytest.raises(ApiError, match=r"needs about 9 iTick calls.*more than the 3 allowed in one run"):
        app.extra_post["/api/data/fetch"]({"kind": "futures", "source": "itick", "items": ["US:ES", "US:NQ", "US:CL"], "years": 10})
    assert fake.calls == [] and app.data._running == set()


def test_without_a_key_nothing_is_started_and_the_message_says_how_to_fix_it(tmp_path, monkeypatch):
    monkeypatch.delenv("ITICK_API_KEY", raising=False)
    clock, fake = FakeClock(), FakeITick()
    app = _app(tmp_path, _rich, root=tmp_path)
    _use(app, _provider(clock, fake, key=None, root=tmp_path), tmp_path)
    st = app.extra_get["/api/data/status"]({})["itick"]
    assert st["configured"] is False and st["setup"] == "key" and "ITICK_API_KEY" in st["problem"] and "Secrets" in st["problem"]
    for body in ({"kind": "futures", "source": "itick", "items": ["US:NQ"]}, {"kind": "crypto", "source": "itick", "items": ["SOL"]},
                 {"kind": "options", "items": ["SPY"], "what": ["prices"], "prices_source": "itick"}):
        with pytest.raises(ApiError, match="ITICK_API_KEY"):
            app.extra_post["/api/data/fetch"](body)
    assert fake.calls == [] and app.data._running == set() and not app.jobs
    feeds(app, {CBOE + "SPY.json": fh.cboe_payload("SPY")})
    assert run(app, kind="options", items=["SPY"])["ok"] == 1, "the free sources do not need the key"


def test_a_rejected_key_stops_the_run_at_once_and_the_key_is_never_echoed(tmp_path):
    clock, fake = FakeClock(), FakeITick(key="THE-REAL-KEY-0123456789")
    app = _app(tmp_path, _rich, root=tmp_path)
    wrong = "WRONG-KEY-abcdefghij"
    _use(app, _provider(clock, fake, key=wrong), tmp_path)
    job = wait(app, app.extra_post["/api/data/fetch"]({"kind": "futures", "source": "itick", "items": ["US:ES", "US:NQ", "US:CL"], "years": 2}))
    assert job["status"] == "error" and "rejected the API key" in job["error"] and "ITICK_API_KEY" in job["error"]
    assert len(fake.calls) == 1, "once iTick refuses the key, the other products are not tried"
    assert wrong not in json.dumps(job) and "THE-REAL-KEY" not in json.dumps(job)
    assert app.data._running == set(), "the failed download released its place"


def test_a_product_iTick_does_not_have_fails_alone_and_a_bad_code_is_refused_before_it_is_queued(ik):
    app, fake, _ = ik
    fake.unknown_prefix = "ZQX"
    out = run(app, kind="futures", source="itick", items=["US:NQ", "US:ZQX"], years=1)
    by = {r["item"]: r for r in out["results"]}
    assert out["ok"] == 1 and out["failed"] == 1 and by["US:NQ"]["ok"] and not by["US:ZQX"]["ok"] and "does not have this symbol" in by["US:ZQX"]["message"], "one product failing does not lose the others"
    queued = len(app.jobs)
    for body, text in (({"items": ["US:"]}, "not an iTick futures product"), ({"items": ["EU:NQ"]}, "not an iTick futures product"), ({"items": ["US:N Q"]}, "not an iTick futures product"),
                       ({"items": ["../x"]}, "not an iTick futures product"), ({"items": []}, "items must be"), ({"items": ["US:NQ"], "years": 0}, "between"),
                       ({"items": ["US:NQ"], "years": 99}, "between"), ({"items": ["US:NQ"], "source": "stooq"}, "source must be one of")):
        with pytest.raises(ApiError, match=text):
            app.extra_post["/api/data/fetch"]({"kind": "futures", "source": "itick", **body})
    assert len(app.jobs) == queued
    out = run(app, kind="futures", source="itick", items=["nq"], years=1)                          # a bare code is a United States product
    assert out["results"][0]["item"] == "US:NQ"


def test_a_product_the_bundled_list_lacks_is_still_tried_and_the_download_says_so(ik):
    app, fake, _ = ik
    out = run(app, kind="futures", source="itick", items=["US:QQQ9"], years=1)
    rec = app.data.store.meta("futures", "US_QQQ9")
    assert out["ok"] == 1 and any("not in the product list bundled" in n for n in rec["notes"]) and rec["root_name"] == "QQQ9"


# ------------------------------------------------------------------------------------------------------------------ crypto
def test_a_coins_history_downloads_from_itick_as_a_spot_only_file_the_crypto_tab_reads(ik):
    app, fake, _ = ik
    out = _safe(run(app, kind="crypto", source="itick", items=["sol", "BTC", "NOTACOIN"], days=400))
    by = {r["item"]: r for r in out["results"]}
    assert out["ok"] == 2 and by["SOL"]["ok"] and by["BTC"].get("skipped") and "already holds Deribit" in by["BTC"]["message"] and not by["NOTACOIN"]["ok"]
    assert "does not have NOTACOIN" in by["NOTACOIN"]["message"] and {(c["region"], c["code"]) for c in fake.calls} == {("BA", "SOLUSDT")}
    assert all(u.startswith("https://api-free.itick.org/crypto/kline?") for u in fake.urls)
    rec = app.data.store.meta("crypto", "SOL")
    assert rec["item"] == "history" and rec["source_id"] == "itick" and rec["sources"] == {"spot": "iTick (USDT)"} and rec["refresh"] == {"source": "itick", "items": ["SOL"], "what": ["history"], "days": 400}
    assert any("spot only" in n for n in rec["notes"]) and any("USDT" in n for n in rec["notes"])
    frame = pd.read_csv(app.data.store.path("crypto", "SOL"))
    assert list(frame.columns) == ["time", "spot"] and len(frame) == 400 and (frame["spot"] > 0).all()
    cat = _safe(app.extra_get["/api/assets/catalog"]({}))
    assert {s["id"]: s["source"] for s in cat["crypto_symbols"]}["SOL"] == "downloaded"
    ov = _safe(app.extra_get["/api/crypto/overview"]({"symbol": "SOL"}))
    assert ov["sources"]["prices"] == "downloaded" and any("iTick" in n for n in ov["notes"])

    before = len(fake.calls)
    run(app, kind="crypto", **rec["refresh"])
    assert len(fake.calls) - before == 1, "an update is one call"

    for body, text in (({"what": ["history", "venues"]}, "only the daily history"), ({"what": ["funding"]}, "only the daily history"), ({"days": 100}, "between"), ({"source": "kraken"}, "source must be one of")):
        with pytest.raises(ApiError, match=text):
            app.extra_post["/api/data/fetch"]({"kind": "crypto", "source": "itick", "items": ["SOL"], **body})


def test_the_exchanges_still_work_as_before_and_a_later_download_from_them_replaces_the_itick_file(ik):
    app, fake, _ = ik
    run(app, kind="crypto", source="itick", items=["SOL"], days=300)
    feeds(app, _crypto_routes("SOL"))
    out = run(app, kind="crypto", items=["sol"], days=260, what=["history"])
    assert out["ok"] == 1
    rec = app.data.store.meta("crypto", "SOL")
    assert rec["source_id"] == "okx" and "OKX" in rec["source"], "one history per coin: the latest download is the file, and its record says where it came from"
    n = len(fake.calls)
    run(app, kind="crypto", source="itick", items=["SOL"], days=300)
    assert len(fake.calls) - n >= 1 and app.data.store.meta("crypto", "SOL")["source_id"] == "itick", "an exchange file is never extended with iTick's bars"


# ------------------------------------------------------------------------------------------------------------------ options
def _chain_feeds(app):
    return feeds(app, {CBOE + "SPY.json": fh.cboe_payload("SPY", spot=500.0, vol=0.2), YAHOO + "SPY": lambda url: fh.yahoo_chart("SPY", DAYS, CLOSES),
                       "https://www.deribit.com/api/v2/public/get_book_summary_by_currency": fh.deribit_options_payload("BTC")})


def test_iTick_has_no_options_but_it_can_download_the_underlyings_prices_for_the_realised_volatility(ik):
    app, fake, _ = ik
    _chain_feeds(app)
    run(app, kind="options", items=["SPY"])
    bare = _safe(app.extra_get["/api/options/chain"]({"underlying": "dl:SPY"}))
    assert bare["overview"]["realized_vol_21d"] is None and any("not in the platform's price data" in n for n in bare["notes"]), "without prices there is nothing to set beside the implied volatility, and the page says so"

    out = _safe(run(app, kind="options", items=["SPY"], what=["prices"], prices_source="itick", years=2))
    assert out["ok"] == 1 and out["results"][0]["what"] == "prices" and out["results"][0]["name"] == "SPY_prices" and {c["code"] for c in fake.calls} == {"SPY"}
    assert all(u.startswith("https://api-free.itick.org/stock/kline?") for u in fake.urls) and not any("option" in u for u in fake.urls), "iTick is never asked for an option"
    rec = app.data.store.meta("options", "SPY_prices")
    assert rec["item"] == "prices" and rec["underlying"] == "SPY" and rec["source_id"] == "itick" and rec["refresh"]["prices_source"] == "itick"
    assert {d["name"] for d in app.data.store.list("options")} == {"SPY", "SPY_prices"}
    cat = _safe(app.extra_get["/api/assets/catalog"]({}))
    assert {u["id"] for u in cat["underlyings"] if u["source"] == "downloaded"} == {"dl:SPY"}, "a price file is not an underlying to pick"

    chain = _safe(app.extra_get["/api/options/chain"]({"underlying": "dl:SPY"}))
    assert chain["overview"]["realized_vol_21d"] > 0.05 and any("daily prices of SPY downloaded from iTick" in n for n in chain["notes"])
    assert not any("not in the platform's price data" in n for n in chain["notes"])

    both = _safe(run(app, kind="options", items=["SPY", "deribit:btc"], what=["chain", "prices"], prices_source="yahoo", years=1))
    assert sorted((r["what"], bool(r.get("skipped"))) for r in both["results"]) == [("chain", False), ("chain", False), ("prices", False), ("prices", True)]
    skipped = next(r for r in both["results"] if r.get("skipped"))
    assert "already holds the BTC price history" in skipped["message"], "a Deribit chain's underlying is already in the platform's data"
    assert app.data.store.meta("options", "SPY_prices")["source_id"] == "yahoo", "the latest download of the prices is the file"


def test_an_index_underlying_is_asked_of_iTick_under_its_own_name(ik):
    app, fake, _ = ik
    run(app, kind="options", items=["SPX"], what=["prices"], prices_source="itick", years=1)
    assert fake.calls[-1]["code"] == "SPX" and fake.urls[-1].startswith("https://api-free.itick.org/indices/kline?"), "SPX is an index at iTick"
    for body, text in (({"what": ["nope"]}, "what must be"), ({"what": ["prices"], "prices_source": "stooq"}, "prices_source must be one of"), ({"what": ["prices"], "years": 50}, "between")):
        with pytest.raises(ApiError, match=text):
            app.extra_post["/api/data/fetch"]({"kind": "options", "items": ["SPY"], **body})


def test_a_price_history_with_too_few_days_is_reported_and_the_chain_is_unaffected(ik):
    app, fake, _ = ik
    _chain_feeds(app)
    fake.history_start["SPY"] = pd.Timestamp(fake.last) - pd.Timedelta(days=60)
    out = run(app, kind="options", items=["SPY"], what=["chain", "prices"], prices_source="itick", years=1)
    by = {r["what"]: r for r in out["results"]}
    assert by["chain"]["ok"] and by["prices"]["ok"]
    chain = _safe(app.extra_get["/api/options/chain"]({"underlying": "dl:SPY"}))
    assert chain["overview"]["realized_vol_21d"] is None, "forty bars are too few for the options tab (it needs 130): the volatility is left out, not invented"
