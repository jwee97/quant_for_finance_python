"""The Free data endpoints: a download runs as a job on its own worker, saves files with a record of where they came from, and the Options, Futures and Crypto workspaces then read them and label them
``downloaded``. The network is replaced by a table of answers in the shape of the live feeds."""

from __future__ import annotations

import json
import threading
from urllib import parse

import pandas as pd
import pytest

from src.assets.free_data import futures as ff
from src.assets.free_data import options as fo
from src.assets.free_data.http import Http
from src.webapp import api, server
from src.webapp.api import ApiError
from tests import free_data_helpers as fh
from tests.test_webapp import _http
from tests.test_webapp_assets_ml import _app, finished, wait
from tests.test_webapp_labs import _rich

CBOE = "https://cdn.cboe.com/api/global/delayed_quotes/options/"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
OKX = "https://www.okx.com/api/v5"
DAYS = [str(d.date()) for d in pd.date_range("2025-11-01", periods=330)]
CLOSES = [100.0 + 0.1 * i for i in range(330)]


def _safe(value):
    json.dumps(value, allow_nan=False)
    return value


@pytest.fixture
def app(tmp_path):
    """An app whose data folder is a temporary one (so nothing is written into the repository) and whose network is a table of answers."""
    a = _app(tmp_path, _rich, root=tmp_path)
    a.data.http = Http(fh.FakeFeeds({}), sleep=lambda s: None, pause=0.0, retries=0)
    a.data.yfinance = type("OfflineYahoo", (), {"Ticker": staticmethod(lambda symbol: (_ for _ in ()).throw(RuntimeError("offline")))})        # no test goes to the network
    return a


def feeds(app, routes: dict) -> fh.FakeFeeds:
    f = fh.FakeFeeds(routes)
    app.data.http = Http(f, sleep=lambda s: None, pause=0.0, retries=0)
    return f


def run(app, **body) -> dict:
    return finished(app, app.extra_post["/api/data/fetch"](body))


def yahoo_route(level: float = 90.0, days: int = 40):
    """Yahoo's chart answer for whichever contract symbol is asked: the same ramp, shifted by the contract's position so that curves have a shape."""
    def answer(url: str):
        symbol = parse.unquote(url.split("/chart/")[1].split("?")[0])
        shift = sum(map(ord, symbol)) % 7
        dates = [str(d.date()) for d in pd.bdate_range(end="2026-10-09", periods=days)]
        return fh.yahoo_chart(symbol, dates, [level + shift + 0.01 * i for i in range(days)])
    return answer


# ------------------------------------------------------------------------------------------------------------------ the page and its routes
def test_the_routes_the_version_and_the_status_describe_what_can_be_downloaded(app):
    assert api.API_VERSION == 11 and app.catalog()["api_version"] == 11
    assert "/api/data/status" in app.extra_get and {"/api/data/fetch", "/api/data/delete"} <= set(app.extra_post)
    st = _safe(app.extra_get["/api/data/status"]({}))
    assert set(st["kinds"]) == {"options", "futures", "crypto"} and st["running"] == [] and st["directory"] == "data/user/downloads/"
    assert "SPY" in st["kinds"]["options"]["symbols"] and st["kinds"]["options"]["crypto"] == ["BTC", "ETH"] and st["kinds"]["options"]["defaults"] == {"max_dte": 365}
    assert [r["id"] for r in st["kinds"]["futures"]["roots"]][:5] == ["CL", "NG", "HO", "RB", "BZ"] and "BTC" in [r["id"] for r in st["kinds"]["futures"]["crypto"]]
    assert st["kinds"]["crypto"]["defaults"]["what"] == ["history", "venues", "funding"] and "SOL" in st["kinds"]["crypto"]["coins"]
    for kind, part in st["kinds"].items():
        assert part["downloads"] == [] and part["sources"] and all(s["label"] and s["delay"] and s["url"].startswith("https://") for s in part["sources"]), kind
    assert "somebody else's service" in st["note"]


# ------------------------------------------------------------------------------------------------------------------ options
def test_option_chains_download_as_a_job_and_the_options_tab_reads_them_as_downloaded_data(app):
    f = feeds(app, {CBOE + "SPY.json": fh.cboe_payload("SPY", spot=500.0, vol=0.2), CBOE + "AAA.json": fh.cboe_payload("AAA", spot=100.0, step=1.0, vol=0.3),
                    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency": fh.deribit_options_payload("BTC")})
    resp = app.extra_post["/api/data/fetch"]({"kind": "options", "items": ["spy", "AAA", "ZZZZ", "deribit:btc"], "max_dte": 365})
    assert set(resp) == {"job"}
    out = _safe(finished(app, resp))
    by_item = {r["item"]: r for r in out["results"]}
    assert out["ok"] == 3 and out["failed"] == 1 and by_item["SPY"]["ok"] and by_item["AAA"]["ok"] and by_item["BTC-Deribit"]["ok"] and not by_item["ZZZZ"]["ok"]
    assert by_item["ZZZZ"]["message"].startswith("Cboe has no option chain for ZZZZ"), "a symbol with no listed options is reported as that"
    assert "contracts on" in by_item["SPY"]["message"] and by_item["SPY"]["source"] == "Cboe delayed quotes" and by_item["SPY"]["as_of"] == "2026-10-09"
    assert [d["name"] for d in out["downloads"]] and {d["name"] for d in app.extra_get["/api/data/status"]({})["kinds"]["options"]["downloads"]} == {"SPY", "AAA", "BTC-Deribit"}
    assert not (app.root / "data" / "user" / "options").exists(), "your own files folder is never written to"

    cat = _safe(app.extra_get["/api/assets/catalog"]({}))
    dl = {u["id"]: u for u in cat["underlyings"] if u["source"] == "downloaded"}
    assert set(dl) == {"dl:SPY", "dl:AAA", "dl:BTC-Deribit"} and "real chain, Cboe delayed quotes, 2026-10-09" in dl["dl:SPY"]["label"] and dl["dl:SPY"]["fetched_at"].endswith("Z")
    assert any(u["id"] == "AAA" and u["source"].startswith("simulated") for u in cat["underlyings"]), "the modelled chain of the same ticker is still there, labelled as a model"
    assert len(f.calls) >= 4

    chain = _safe(app.extra_get["/api/options/chain"]({"underlying": "dl:SPY"}))
    assert chain["sources"]["chain"] == "downloaded" and chain["sources"]["underlying_price"] == "downloaded" and chain["as_of"] == "2026-10-09"
    assert chain["source"] == "mixed" and chain["sources"]["reference_index"] == "platform", "the VIX beside SPY's chain is the platform's, and the page says so"
    assert any("Cboe delayed quotes" in n for n in chain["notes"]) and any("put-call parity" in n for n in chain["notes"])
    assert 0.17 < chain["overview"]["atm_iv_30d"] < 0.23, "the volatility that priced the payload comes back out of the file"
    assert chain["overview"]["snapshot"]["source"] == "downloaded" and "no history of implied volatility" in chain["overview"]["iv_rank_basis"]
    assert chain["overview"]["quote_ages"] is False and chain["chain"]["quote_ages"] is False, "a download has no quote times: the page must not claim that none is stale"
    assert app.extra_get["/api/options/chain"]({"underlying": "AAA"})["chain"]["quote_ages"] is True, "the modelled chain has quote ages"
    assert chain["settings"]["rate"] == 0.04 and 0.0 < chain["settings"]["dividend"] < 0.03
    with_rate = _safe(app.extra_get["/api/options/chain"]({"underlying": "dl:SPY", "rate": "0.02", "dividend": "0.05"}))
    assert with_rate["settings"]["rate"] == 0.04 and any("not editable here" in n for n in with_rate["notes"]), "the rate comes with the download"
    in_bundle = _safe(app.extra_get["/api/options/chain"]({"underlying": "dl:AAA"}))
    assert in_bundle["overview"]["realized_vol_21d"] > 0 and any("platform's prices of AAA" in n for n in in_bundle["notes"]), "an underlying the platform holds gets its realised volatility from the platform's prices"
    deribit = _safe(app.extra_get["/api/options/chain"]({"underlying": "dl:BTC-Deribit"}))
    assert deribit["source"] == "downloaded" and deribit["settings"]["rate"] == 0.0 and deribit["settings"]["dividend"] < 0

    surface = _safe(app.extra_get["/api/options/surface"]({"underlying": "dl:SPY"}))
    assert surface["source"] == "downloaded" and surface["arbitrage"] and any("within 5% of the forward" in n for n in surface["notes"])
    risk = _safe(app.extra_post["/api/options/portfolio-risk"]({"underlying": "dl:SPY", "positions": [{"type": "option", "right": "C", "strike": 500, "dte": 30, "quantity": 1}]}))
    assert risk["source"] == "downloaded" and risk["sources"]["volatilities"] == "downloaded"
    payoff = _safe(app.extra_post["/api/options/strategy/payoff"]({"underlying": "dl:SPY", "strategy": "long_straddle", "dte": 30}))
    assert payoff["sources"]["premiums"] == "downloaded"
    with pytest.raises(ApiError, match="no downloaded chain 'NVDA'"):
        app.extra_get["/api/options/chain"]({"underlying": "dl:NVDA"})
    with pytest.raises(ApiError, match="named like SPY"):
        app.extra_get["/api/options/chain"]({"underlying": "dl:../x"})


# ------------------------------------------------------------------------------------------------------------------ futures
def test_futures_curves_download_merge_with_the_last_download_and_the_futures_tab_reads_them(app):
    feeds(app, {YAHOO + "CL": yahoo_route(90.0), "https://www.deribit.com/api/v2/public/get_book_summary_by_currency": json.load(open(fh.__file__.replace("free_data_helpers.py", "fixtures/free_data/deribit_btc_futures.json")))})
    out = _safe(run(app, kind="futures", items=["CL", "btc"], contracts=3, history="1y"))
    by_item = {r["item"]: r for r in out["results"]}
    assert out["ok"] == 2 and by_item["CL"]["ok"] and "3 contracts, 40 days to 2026-10-09 (Yahoo Finance)" in by_item["CL"]["message"] and by_item["BTC"]["ok"]
    cat = _safe(app.extra_get["/api/assets/catalog"]({}))
    roots = {r["id"]: r for r in cat["futures_roots"]}
    assert roots["CL"]["source"] == "downloaded" and roots["NG"]["source"] == "simulated" and roots["BTC"]["source"] == "downloaded" and roots["BTC"]["asset_class"] == "crypto"
    assert roots["BTC"]["name"] == "Bitcoin futures" and roots["BTC"]["exchange"] == "Deribit", "a root the platform does not simulate keeps the long name the download recorded"

    curve = _safe(app.extra_get["/api/futures/curve"]({"root": "CL"}))
    assert curve["source"] == "mixed" and curve["sources"] == {"curve": "downloaded", "margins": "simulated"}, "no open interest in Yahoo's feed, so none is claimed"
    assert len(curve["curve"]["rows"]) == 3 and curve["spec"]["multiplier"] == 1000.0 and curve["curve"]["rows"][0]["open_interest"] is None
    assert any("Yahoo Finance" in n for n in curve["notes"]) and any("computed from the exchange's rule" in n for n in curve["notes"])
    roll = _safe(app.extra_get["/api/futures/roll"]({"root": "CL"}))
    assert roll["sources"] == {"curve": "downloaded"}
    btc = _safe(app.extra_get["/api/futures/curve"]({"root": "BTC"}))
    assert btc["sources"]["open_interest"] == "downloaded" and len(btc["curve"]["rows"]) == 8 and btc["curve"]["rows"][0]["contract"] == "16OCT26" and btc["spec"]["multiplier"] == 1.0
    margin = _safe(app.extra_post["/api/futures/margin"]({"positions": [{"root": "CL", "quantity": 2}, {"root": "NG", "quantity": 1}], "equity": 500000}))
    assert margin["sources"]["prices"] == "mixed", "one downloaded root and one simulated root"

    # a second download: Yahoo no longer serves one contract (it expired), and the file keeps its history
    before = pd.read_csv(app.data.store.path("futures", "CL"))
    gone = sorted(before["contract"].unique())[0]
    served = yahoo_route(95.0, days=10)

    def partial(url: str):
        return fh.YAHOO_MISSING if parse.unquote(url.split("/chart/")[1]).startswith(f"CL{gone}") else served(url)

    feeds(app, {YAHOO + "CL": lambda url: (404, partial(url)) if partial(url) is fh.YAHOO_MISSING else partial(url)})
    again = run(app, kind="futures", items=["CL"], contracts=3, history="6mo")
    after = pd.read_csv(app.data.store.path("futures", "CL"))
    assert gone in set(after["contract"]), "the contract the feed dropped keeps its rows"
    assert len(after[after["contract"] == gone]) == len(before[before["contract"] == gone]) and "Merged with the earlier download" in " ".join(app.data.store.meta("futures", "CL")["notes"])
    assert again["results"][0]["ok"] and after.duplicated(["date", "contract"]).sum() == 0


def test_your_own_futures_file_wins_over_a_download_of_the_same_root(app):
    feeds(app, {YAHOO + "CL": yahoo_route()})
    run(app, kind="futures", items=["CL"], contracts=3)
    folder = app.root / "data" / "user" / "futures"
    folder.mkdir(parents=True)
    dates = pd.bdate_range("2026-09-01", periods=30)
    rows = [{"date": d, "contract": c, "expiry": e, "price": p + 0.01 * i} for i, d in enumerate(dates) for c, e, p in (("Z26", "2026-11-20", 70.0), ("F27", "2026-12-21", 71.0))]
    pd.DataFrame(rows).to_csv(folder / "CL.csv", index=False)
    out = _safe(app.extra_get["/api/futures/curve"]({"root": "CL"}))
    assert out["sources"]["curve"] == "user_file" and {r["id"]: r["source"] for r in app.extra_get["/api/assets/catalog"]({})["futures_roots"]}["CL"] == "user_file"


# ------------------------------------------------------------------------------------------------------------------ crypto
def _crypto_routes(coin="SOL"):
    return {f"{OKX}/market/history-candles?instId={coin}-USDT&": fh.okx_candles(DAYS, CLOSES), f"{OKX}/market/history-candles?instId={coin}-USDT-SWAP&": fh.okx_candles(DAYS, [c + 0.05 for c in CLOSES]),
            f"{OKX}/public/funding-rate-history": fh.okx_funding(DAYS[-130:], 0.0001), f"{OKX}/market/books?instId={coin}-USDT&": fh.okx_book(), f"{OKX}/market/books?instId={coin}-USDT-SWAP": fh.okx_book(contracts=True),
            f"{OKX}/public/instruments": fh.okx_instrument(), f"https://api.exchange.coinbase.com/products/{coin}-USD/book": fh.coinbase_book(), "https://api.kraken.com/0/public/Depth": fh.kraken_depth(f"X{coin}ZUSD"),
            f"https://www.bitstamp.net/api/v2/order_book/{coin.lower()}usd/": fh.bitstamp_book()}


def test_crypto_downloads_feed_the_history_the_real_books_the_quotes_and_the_funding_of_the_crypto_tab(app):
    feeds(app, _crypto_routes("SOL"))
    out = _safe(run(app, kind="crypto", items=["sol"], days=260))
    by_what = {r["what"]: r for r in out["results"]}
    assert out["ok"] == 3 and out["failed"] == 0 and set(by_what) == {"history", "venues", "funding"} and "OKX (USDT)" in by_what["history"]["message"] and "quotes and books on 5 venues" in by_what["venues"]["message"]
    assert {r["item"] for r in out["results"]} == {"SOL"} and by_what["venues"]["name"] == "venues_SOL" and by_what["funding"]["name"] == "funding_SOL_OKX", "every row is named by the coin asked for; the file is a separate field"
    assert {d["name"] for d in app.data.store.list("crypto")} == {"SOL", "venues_SOL", "funding_SOL_OKX"}
    cat = _safe(app.extra_get["/api/assets/catalog"]({}))
    assert {s["id"]: s["source"] for s in cat["crypto_symbols"]}["SOL"] == "downloaded"

    ov = _safe(app.extra_get["/api/crypto/overview"]({"symbol": "SOL"}))
    assert ov["sources"]["prices"] == "downloaded" and ov["sources"]["funding"] == "downloaded" and ov["sources"]["order_book"] == "downloaded"
    assert list(ov["books"]) == ["COINBASE", "KRAKEN", "BITSTAMP", "OKX", "OKX-PERP"] and ov["snapshot"]["venues"] == list(ov["books"]) and ov["overview"]["funding_annualised"] is not None
    liquidity = [c for c in ov["overview"]["cards"] if c["name"] == "liquidity"][0]
    assert liquidity["source"] == "downloaded" and "real book" in liquidity["detail"]
    assert ov["spot_perp"]["source"] == "downloaded" and ov["basis"] is not None and any("Downloaded" in n for n in ov["notes"])
    cross = _safe(app.extra_get["/api/crypto/cross-exchange"]({"symbol": "SOL"}))
    assert cross["source"] == "downloaded" and {q["kind"] for q in cross["quotes"]} == {"spot", "perp"} and cross["pairs"] and all(p["kind"] in ("spot", "perp") for p in cross["pairs"])
    assert any("Real quotes" in n for n in cross["notes"]) and not any("Deribit's price is a daily mark" in n for n in cross["notes"])
    funding = _safe(app.extra_get["/api/crypto/funding"]({"symbols": "SOL"}))
    rows = {r["venue"]: r["source"] for r in funding["rows"]}
    assert rows["OKX"] == "downloaded" and rows["BINANCE"] == "simulated" and rows["DERIBIT"] == "simulated"
    assert any("Funding that is data: SOL on OKX" in n for n in funding["notes"]) and funding["sources"]["SOL@OKX"] == "downloaded"
    with pytest.raises(ApiError, match="unknown symbol 'DOGE'"):
        app.extra_get["/api/crypto/overview"]({"symbol": "DOGE"})


def test_bitcoin_and_ether_keep_the_platforms_history_and_take_downloaded_venues_and_funding(app):
    feeds(app, _crypto_routes("BTC"))
    out = run(app, kind="crypto", items=["BTC"], days=260)
    by_what = {r["what"]: r for r in out["results"]}
    assert by_what["history"].get("skipped") and "already holds Deribit" in by_what["history"]["message"] and by_what["venues"]["ok"] and by_what["funding"]["ok"]
    assert {d["name"] for d in app.data.store.list("crypto")} == {"venues_BTC", "funding_BTC_OKX"}, "no BTC history file: Deribit's is longer"


# ------------------------------------------------------------------------------------------------------------------ safety and failure
def test_bad_requests_are_refused_before_anything_is_queued_and_no_request_can_name_a_path_or_a_host(app):
    queued = len(app.jobs)
    fetch = app.extra_post["/api/data/fetch"]
    for body, text in (({}, "kind must be"), ({"kind": "prices", "items": ["SPY"]}, "kind must be"), ({"kind": "options"}, "items must be"), ({"kind": "options", "items": []}, "items must be"),
                       ({"kind": "options", "items": ["SPY"] * 3 + [f"T{i}" for i in range(20)]}, "at most|items must be"), ({"kind": "options", "items": ["../etc/passwd"]}, "not a ticker"),
                       ({"kind": "options", "items": ["http://evil.test/x"]}, "not a ticker"), ({"kind": "options", "items": ["deribit:SOL"]}, "BTC, ETH"), ({"kind": "options", "items": ["SPY"], "max_dte": 0}, "between"),
                       ({"kind": "options", "items": ["SPY"], "rate": 5}, "between"), ({"kind": "futures", "items": ["XX"]}, "free curves are available"), ({"kind": "futures", "items": ["CL"], "history": "forever"}, "history must be"),
                       ({"kind": "futures", "items": ["CL"], "contracts": ff.MAX_CONTRACTS + 1}, "between"), ({"kind": "futures", "items": ["CL"], "history": "3y"}, "history must be"),
                           ({"kind": "futures", "items": ["BTC"], "deribit": "weekly"}, "deribit must be"), ({"kind": "futures", "items": ["VXM"]}, "free curves are available"), ({"kind": "crypto", "items": ["btc/usd"]}, "not a coin"), ({"kind": "crypto", "items": ["SOL"], "what": ["nope"]}, "what must be"),
                       ({"kind": "crypto", "items": ["SOL"], "days": 100}, "between")):
        with pytest.raises(ApiError, match=text):
            fetch(body)
    assert len(app.jobs) == queued and app.data._running == set(), "nothing refused was queued, and nothing is marked as running"


def test_a_download_where_nothing_worked_is_an_error_with_the_reasons_and_can_be_asked_again(app):
    feeds(app, {})
    job = wait(app, app.extra_post["/api/data/fetch"]({"kind": "options", "items": ["SPY", "QQQ"]}))
    assert job["status"] == "error" and job["error"].startswith("Nothing could be downloaded.") and "Cboe has no option chain for SPY" in job["error"]
    assert app.data._running == set(), "the failed download released its place"
    feeds(app, {CBOE + "SPY.json": fh.cboe_payload("SPY")})
    assert run(app, kind="options", items=["SPY"])["ok"] == 1


def test_a_second_download_for_the_same_tab_is_refused_while_the_first_runs_and_another_tab_is_not(app):
    gate, started = threading.Event(), threading.Event()
    payload = fh.cboe_payload("SPY")

    def slow(url, headers, timeout):
        started.set()
        gate.wait(10)
        return 200, {}, json.dumps(payload).encode()

    app.data.http = Http(slow, sleep=lambda s: None, pause=0.0, retries=0)
    first = app.extra_post["/api/data/fetch"]({"kind": "options", "items": ["SPY"]})
    assert started.wait(10) and app.data._running == {"options"}
    assert app.extra_get["/api/data/status"]({})["running"] == ["options"]
    with pytest.raises(ApiError, match="already running"):
        app.extra_post["/api/data/fetch"]({"kind": "options", "items": ["QQQ"]})
    other = app.extra_post["/api/data/fetch"]({"kind": "futures", "items": ["CL"]})                    # another tab may queue its own download
    gate.set()
    assert wait(app, first)["status"] == "done"
    wait(app, other)
    assert app.data._running == set()


def test_deleting_a_download_needs_confirmation_and_never_touches_your_own_files(app):
    feeds(app, {CBOE + "SPY.json": fh.cboe_payload("SPY")})
    run(app, kind="options", items=["SPY"])
    own = app.root / "data" / "user" / "options"
    own.mkdir(parents=True)
    (own / "SPY.csv").write_text("date,expiry,strike,right,bid,ask,underlying\n")
    delete = app.extra_post["/api/data/delete"]
    with pytest.raises(ApiError, match="confirm: true"):
        delete({"kind": "options", "name": "SPY"})
    with pytest.raises(ApiError, match="no download called 'QQQ'"):
        delete({"kind": "options", "name": "QQQ", "confirm": True})
    with pytest.raises(ApiError, match="not a usable file name"):
        delete({"kind": "options", "name": "../x", "confirm": True})
    with pytest.raises(ApiError, match="kind must be"):
        delete({"kind": "prices", "name": "SPY", "confirm": True})
    out = delete({"kind": "options", "name": "SPY", "confirm": True})
    assert out["deleted"] == "SPY" and out["downloads"] == [] and (own / "SPY.csv").exists(), "your own file stays"
    assert {u["id"] for u in app.extra_get["/api/assets/catalog"]({})["underlyings"] if u["source"] == "downloaded"} == set()


def test_the_routes_need_the_token_and_a_download_runs_end_to_end_over_http(app):
    feeds(app, {CBOE + "SPY.json": fh.cboe_payload("SPY")})
    running = server.start(app, "127.0.0.1", 0)
    try:
        assert _http(running, "/api/data/status", token=False)[0] == 403
        assert _http(running, "/api/data/fetch", body={"kind": "options", "items": ["SPY"]}, token=False)[0] == 403
        assert _http(running, "/api/data/delete", body={"kind": "options", "name": "SPY", "confirm": True}, token=False)[0] == 403
        status, body, _ = _http(running, "/api/data/status")
        assert status == 200 and set(json.loads(body)["kinds"]) == {"options", "futures", "crypto"}
        status, body, _ = _http(running, "/api/data/fetch", body={"kind": "options", "items": ["../x"]})
        assert status == 400 and "not a ticker" in json.loads(body)["error"]
        status, body, _ = _http(running, "/api/data/fetch", body={"kind": "options", "items": ["SPY"]})
        assert status == 200
        job = json.loads(body)["job"]
        for _ in range(300):
            view = json.loads(_http(running, f"/api/job/{job}")[1])
            if view["status"] in ("done", "error"):
                break
            threading.Event().wait(0.1)
        assert view["status"] == "done" and view["result"]["ok"] == 1
        status, body, _ = _http(running, "/api/data/delete", body={"kind": "options", "name": "SPY"})
        assert status == 400 and "confirm: true" in json.loads(body)["error"]
        status, body, _ = _http(running, "/api/data/delete", body={"kind": "options", "name": "SPY", "confirm": True})
        assert status == 200 and json.loads(body)["downloads"] == []
    finally:
        running.stop()


def test_the_default_client_paces_the_hosts_that_rate_limit_and_retries_with_a_long_backoff():
    from src.webapp.data_api import HOSTS, DataApi

    client = DataApi.__new__(DataApi)
    client.http = None
    http = DataApi._client(client)
    assert http.retries == 3 and http.backoff == 3.0 and http.pauses == HOSTS and HOSTS["query1.finance.yahoo.com"] >= 0.5
    assert fo.MAX_DTE >= 730 and len(ff.FREE_ROOTS) == 44


# ------------------------------------------------------------------------------------------------------------------ the deeper futures downloads
def test_vix_futures_download_from_cboe_with_open_interest_and_an_update_fetches_only_the_live_contracts(app):
    from tests import test_free_data_cboe_futures as cx

    f = feeds(app, cx._routes())
    out = _safe(run(app, kind="futures", items=["VX"]))
    assert out["ok"] == 1 and out["results"][0]["ok"] and "5 contracts" in out["results"][0]["message"] and "Cboe Futures Exchange" in out["results"][0]["message"]
    cat = _safe(app.extra_get["/api/assets/catalog"]({}))
    roots = {r["id"]: r for r in cat["futures_roots"]}
    assert roots["VX"]["source"] == "downloaded" and roots["VX"]["asset_class"] == "volatility" and roots["VX"]["exchange"] == "CFE"
    curve = _safe(app.extra_get["/api/futures/curve"]({"root": "VX"}))
    assert curve["sources"]["open_interest"] == "downloaded" and curve["curve"]["rows"][0]["open_interest"] > 0 and curve["spec"]["multiplier"] == 1000.0
    assert any("Cboe Futures Exchange" in n for n in curve["notes"]) and curve["spec"]["initial_margin"] > 0, "a volatility future gets an illustrative margin, labelled as one"
    first_calls = len([u for u in f.calls if "VX_" in u])
    f2 = feeds(app, cx._routes())
    again = _safe(run(app, kind="futures", items=["VX"]))
    assert again["results"][0]["ok"] and len([u for u in f2.calls if "VX_" in u]) < first_calls, "an update does not ask again for the contracts that had expired"
    saved = app.data.store.meta("futures", "VX")
    assert saved["refresh"]["items"] == ["VX"] and saved["source_id"] == "cboe_cfe"
    after = pd.read_csv(app.data.store.path("futures", "VX"))
    assert after.duplicated(["date", "contract"]).sum() == 0 and after["contract"].nunique() == 5


def test_a_deep_download_asks_yahoo_for_every_contract_it_lists_and_the_record_remembers_that_it_did(app):
    from tests import test_free_data_deep_futures as dx

    served = dx._wti_cycle()
    expiries = {c["contract"]: c["expiry"] for c in ff.upcoming_contracts(ff.FREE_ROOTS["CL"], pd.Timestamp.today(), ff.MAX_CONTRACTS)}
    alive = {code for code in served if code in expiries}                                                    # the fixture lists fixed months; the app counts from the real date
    f = feeds(app, dx._listing(served))
    out = _safe(run(app, kind="futures", items=["CL"], contracts=0, history="max"))
    assert out["ok"] == 1 and f"{len(alive)} contracts" in out["results"][0]["message"]
    saved = app.data.store.meta("futures", "CL")
    assert saved["refresh"]["contracts"] == 0 and saved["refresh"]["history"] == "max" and saved["deep"] is True and saved["furthest"] == max(alive, key=expiries.get)
    assert all("range=max" in u for u in f.calls) and set(pd.read_csv(app.data.store.path("futures", "CL"))["contract"]) == alive
    out2 = _safe(run(app, kind="futures", items=["CL"], contracts="all", history="10y"))
    assert out2["results"][0]["ok"], "the word all means the same as 0"
    shallow = _safe(run(app, kind="futures", items=["CL"], contracts=24, history="2y"))
    assert shallow["results"][0]["ok"]


def test_bitcoin_futures_can_be_downloaded_as_their_whole_history_from_deribits_charts(app):
    from tests import test_free_data_deep_futures as dx

    existing = {"BTC-27FEB26": ("2025-12-01", "2026-02-27", 99.0), "BTC-27MAR26": ("2025-12-29", "2026-03-27", 130.0), "BTC-26JUN26": ("2026-01-30", "2026-06-26", 90.0), "BTC-25SEP26": ("2026-03-27", "2026-09-25", 95.0),
                "BTC-30OCT26": ("2026-04-24", "2026-10-30", 96.0), "BTC-27NOV26": ("2026-05-29", "2026-11-27", 97.0)}
    feeds(app, dx._deribit_routes(existing))
    out = _safe(run(app, kind="futures", items=["BTC"], deribit="history"))
    assert out["ok"] == 1 and "6 contracts" in out["results"][0]["message"] and "Deribit" in out["results"][0]["message"]
    saved = app.data.store.meta("futures", "BTC")
    assert saved["refresh"]["deribit"] == "history" and saved["history"] is True and saved["first_date"] == "2025-12-01"
    cat = _safe(app.extra_get["/api/assets/catalog"]({}))
    roots = {r["id"]: r for r in cat["futures_roots"]}
    assert roots["BTC"]["source"] == "downloaded" and roots["BTC"]["asset_class"] == "crypto"
    curve = _safe(app.extra_get["/api/futures/curve"]({"root": "BTC"}))
    assert len(curve["curve"]["rows"]) >= 2 and curve["spec"]["multiplier"] == 1.0


def test_the_status_lists_the_new_options_of_the_futures_card(app):
    futures = _safe(app.extra_get["/api/data/status"]({}))["kinds"]["futures"]
    assert futures["max_contracts"] == ff.MAX_CONTRACTS and "max" in futures["history"] and "10y" in futures["history"] and futures["deribit"] == ["curve", "history"]
    assert {r["id"] for r in futures["roots"]} >= {"CL", "VX"} and "Volatility" in futures["groups"]
    ids = {s["id"] for s in futures["sources"]}
    assert {"yahoo", "cboe_cfe", "deribit", "itick"} <= ids
