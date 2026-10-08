"""The dashboard backend: universes from tickers, strategy specs, background backtests, formulas, guides, and the HTTP layer's guards."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from src.framework import bundle_from_prices
from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import ApiError, App
from src.webapp.providers import ITickProvider, RateLimiter
from src.webapp.universe import TickerStore, UniverseBuilder, UniverseError, itick_source, normalise
from tests.fake_itick import KEY, FakeClock, FakeITick

N = 2300
IDX = pd.bdate_range("2012-01-02", periods=N)
CLASSES = {"AAA": "equity", "BBB": "equity", "CCC": "rates", "DDD": "commodity", "EEE": "credit", "FFF": "real_estate"}


def _prices(seed: int, columns, n=N, start=IDX[0]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (n, len(columns))), axis=0), index=idx, columns=list(columns))


def _default(config):
    macro = pd.DataFrame({"VIX": 20 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, N))}, index=IDX)
    return bundle_from_prices(_prices(1, CLASSES), asset_class=dict(CLASSES), macro=macro, name="fake-platform")


def _fetch(tickers, start):
    out = {}
    for t in tickers:
        if t.startswith("BAD"):
            out[t] = "no data found: check the symbol"
        elif t.startswith("NEW"):
            p = _prices(abs(hash(t)) % 1000, [t], n=200 if t == "NEWTINY" else 2000, start="2013-01-01")
            out[t] = pd.DataFrame({"close": p[t], "high": p[t] * 1.01, "low": p[t] * 0.99, "volume": 1e6})
        else:
            out[t] = "no data found: check the symbol"
    return out


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    store = TickerStore(tmp_path / "prices", fetch=_fetch)
    return App(config, builder=UniverseBuilder(config, store, default_loader=_default), root=config.root)


def _wait(app, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        view = app.job(job_id)
        if view["status"] in ("done", "error"):
            return view
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def _run(app, **body):
    return _wait(app, app.submit({"models": [{"name": "momentum"}], **body})["job"])


# ------------------------------------------------------------------------------------------------------ tickers
def test_tickers_are_normalised_deduplicated_and_validated():
    assert normalise("spy, qqq  SPY;tlt") == ["SPY", "QQQ", "TLT"]
    assert normalise(["btc-usd", "^GSPC", "EURUSD=X", "BRK.B"]) == ["BTC-USD", "^GSPC", "EURUSD=X", "BRK.B"]
    for bad in ["../etc", "A B$", "DROP;TABLE", "x" * 20, "-X", "a/b", "😀"]:
        with pytest.raises(UniverseError):
            normalise([bad])
    with pytest.raises(UniverseError, match="at most"):
        normalise([f"T{i}" for i in range(50)])


def test_check_reports_coverage_failures_and_short_histories(app):
    r = app.check_tickers({"tickers": ["AAA", "NEWONE", "BADX", "NEWTINY"]})["tickers"]
    assert r["AAA"]["source"] == "platform" and r["NEWONE"]["ok"] and r["NEWONE"]["days"] == 2000
    assert not r["BADX"]["ok"] and "no data" in r["BADX"]["error"] and not r["NEWTINY"]["ok"] and "only 200" in r["NEWTINY"]["error"]
    with pytest.raises(ApiError):
        app.check_tickers({"tickers": ["bad ticker!"]})


def test_downloads_are_cached_and_only_stale_tickers_hit_the_network(tmp_path):
    calls = []

    def fetch(tickers, start):
        calls.append(list(tickers))
        return _fetch(tickers, start)

    store = TickerStore(tmp_path, fetch=fetch)
    store.get(["NEWONE"], "2005-01-01")
    store.get(["NEWONE", "NEWTWO"], "2005-01-01")
    store.get(["NEWONE"], "2005-01-01", refresh=True)
    assert calls == [["NEWONE"], ["NEWTWO"], ["NEWONE"]]
    stale = TickerStore(tmp_path, fetch=fetch, ttl=-1)
    stale.get(["NEWONE"], "2005-01-01")
    assert calls[-1] == ["NEWONE"]


def test_resolve_uses_the_platform_bundle_for_its_own_tickers_and_merges_new_ones(app):
    bundle, info = app.builder.resolve(list(CLASSES), None, None)
    assert info["source"] == "platform dataset" and bundle.name == "fake-platform"
    mixed, info = app.builder.resolve(["AAA", "CCC", "NEWONE"], {"NEWONE": "equity"}, "2013-06-03")
    assert list(mixed.assets) == ["AAA", "CCC", "NEWONE"] and mixed.asset_class["NEWONE"] == "equity" and mixed.asset_class["CCC"] == "rates"
    assert info["has_macro"] and mixed.index[0] >= pd.Timestamp("2013-06-03")
    own, _ = app.builder.resolve(["NEWONE", "NEWTWO"], None, None)
    assert own.high is not None and own.volume is not None and own.name.startswith("user:")
    single, info = app.builder.resolve(["AAA"], None, None)                      # one ticker is a valid universe; whether a strategy can use it is checked separately
    assert list(single.assets) == ["AAA"] and info["tickers"] == ["AAA"]
    with pytest.raises(UniverseError, match="at least one"):
        app.builder.resolve([], None, None)
    with pytest.raises(UniverseError, match="asset class"):
        app.builder.resolve(["AAA", "NEWONE"], {"NEWONE": "meme"}, None)
    with pytest.raises(UniverseError, match="BADX"):
        app.builder.resolve(["AAA", "BADX"], None, None)
    with pytest.raises(UniverseError, match="too little history"):
        app.builder.resolve(["AAA", "NEWTINY"], None, None)


# ------------------------------------------------------------------------------------------------------ catalogue and specs
def test_catalog_lists_what_can_be_run_with_forms_for_every_parameter(app):
    c = app.catalog()
    names = {m["name"] for m in c["models"]}
    assert {"momentum", "expression", "ml_ridge", "online_ridge"} <= names and not any(m["family"] == "crypto" for m in c["models"])
    expr = next(m for m in c["models"] if m["name"] == "expression")
    assert {p["name"] for p in expr["params"]} == {"expr", "mode"} and next(p for p in expr["params"] if p["name"] == "mode")["kind"] == "choice"
    assert next(m for m in c["models"] if m["name"] == "deep_window")["slow"]
    assert "hrp" in next(a for a in c["allocators"] if a["name"] == "static")["params"][0]["choices"]
    assert c["default_tickers"] == list(CLASSES) and "composite" in {d["name"] for d in c["detectors"]} and "def score" in c["template"]
    json.dumps(c, allow_nan=False)


@pytest.mark.parametrize("body,message", [
    ({"models": []}, "between one and four"),
    ({"models": [{"name": "momentum"}] * 5}, "between one and four"),
    ({"models": [{"name": "nope"}]}, "unknown strategy"),
    ({"models": [{"name": "momentum", "params": {"bogus": 1}}]}, "momentum"),
    ({"models": [{"name": "expression", "params": {"expr": "eval('1')"}}]}, "expression"),
    ({"models": [{"name": "momentum"}], "allocator": "nope"}, "unknown allocator"),
    ({"models": [{"name": "momentum"}], "allocator": "bayesian", "allocator_params": {"kind": "x"}}, "bayesian"),
    ({"models": [{"name": "momentum"}], "regime": "nope"}, "unknown regime"),
    ({"models": [{"name": "momentum"}], "regime_risk": True}, "needs a regime"),
    ({"models": [{"name": "momentum"}, {"name": "mean_reversion"}], "combination": "magic"}, "unknown combination"),
    ({"models": [{"name": "momentum"}, {"name": "mean_reversion"}], "combination": "regime_conditional"}, "needs a regime"),
    ({"models": [{"name": "momentum"}], "aum": 5}, "between"),
])
def test_invalid_requests_are_refused_before_any_work_starts(app, body, message):
    with pytest.raises(ApiError, match=message):
        app.submit(body)


# ------------------------------------------------------------------------------------------------------ backtests
def test_a_backtest_runs_in_the_background_and_returns_everything_the_charts_need(app):
    view = _run(app, label="momentum test")
    assert view["status"] == "done", view
    r = view["result"]
    n = len(r["dates"])
    assert n > 500 and len(r["equity"]) == n == len(r["drawdown"]) == len(r["rolling_sharpe"])
    assert set(r["benchmarks"]) == {"equal_weight", "risk_parity"} and all(len(v) == n for v in r["benchmarks"].values())
    assert min(r["drawdown"]) <= 0 and r["equity"][0] > 0
    assert r["metrics"]["n_days"] == n and {"sharpe", "cagr", "ann_vol", "max_drawdown", "ann_turnover", "gross_sharpe"} <= set(r["metrics"])
    assert r["monthly"] and r["annual"] and r["weights"] and r["flags"] and "yaml" in r and "momentum" in r["yaml"]
    assert r["universe"]["source"] == "platform dataset" and r["validation"]["n_trials"] == 1
    json.dumps(r, allow_nan=False)


def test_the_trial_counter_rises_with_every_distinct_idea_and_not_with_repeats(app):
    first = _run(app)["result"]["validation"]["n_trials"]
    repeat = _run(app)["result"]["validation"]["n_trials"]
    other = _run(app, models=[{"name": "momentum", "params": {"lookback": 126}}])["result"]["validation"]["n_trials"]
    assert (first, repeat, other) == (1, 1, 2)
    app.reset_trials()
    assert _run(app)["result"]["validation"]["n_trials"] == 1


def test_a_universe_with_a_new_ticker_runs_and_reports_where_the_data_came_from(app):
    r = _run(app, tickers=["AAA", "BBB", "CCC", "NEWONE"], classes={"NEWONE": "equity"}, start="2013-06-03")
    assert r["status"] == "done", r
    assert r["result"]["universe"]["source"] == "platform dataset + Yahoo Finance" and "NEWONE" in {w["index"] for w in r["result"]["weights"]}


def test_a_formula_strategy_a_second_model_and_options_work_together(app):
    r = _run(app, models=[{"name": "expression", "params": {"expr": "rank(mom(126, 21)) - rank(vol(63))"}}, {"name": "mean_reversion"}], combination="confidence",
             regime="vol_state", regime_risk=True, aum=5e7, allocator="confidence", validate=True)
    assert r["status"] == "done", r
    v = r["result"]["validation"]["causality"]
    assert v["expression"]["ok"] and v["mean_reversion"]["ok"] and "regime:vol_state" in v
    assert "by_regime" in r["result"]["tables"] and r["result"]["spec"]["execution"]["aum"] == 5e7


def test_a_formula_traded_as_written_takes_positions_even_when_calibration_would_hold_nothing(app):
    calibrated = _run(app, models=[{"name": "expression", "params": {"expr": "-ret(5)"}}])
    written = _run(app, models=[{"name": "expression", "params": {"expr": "-ret(5)"}}], allocator="score_stack")
    assert written["status"] == "done" and written["result"]["metrics"]["ann_turnover"] > 1.0 and written["result"]["held_days"] > 500
    assert calibrated["status"] == "done" and calibrated["result"].get("warning", "").startswith("This strategy held") and "warning" not in written["result"]
    with pytest.raises(ApiError, match="exactly one"):
        app.submit({"models": [{"name": "momentum"}, {"name": "mean_reversion"}], "allocator": "score_stack"})


def test_errors_inside_a_run_reach_the_user_as_messages(app):
    r = _run(app, tickers=["AAA", "BADX"])
    assert r["status"] == "error" and "BADX" in r["error"]
    r = _run(app, models=[{"name": "yield_curve_regime"}])
    assert r["status"] == "error" and "needs macro series" in r["error"]
    with pytest.raises(ApiError):
        app.job("nope")


# ------------------------------------------------------------------------------------------------------ formulas and guides
def test_formula_check_previews_the_latest_scores_or_explains_the_problem(app):
    ok = app.check_formula({"expr": "mom(126, 21)", "mode": "cross_sectional"})
    assert ok["ok"] and len(ok["latest"]) == len(CLASSES) and ok["latest"][0]["score"] >= ok["latest"][-1]["score"] and 0 < ok["coverage"] <= 1
    for bad, text in [("__import__('os')", "unknown name"), ("mom(5, 10)", "larger than skip"), ("", "empty"), ("mom(9999)", "whole number"), ("1 + 1", "plain number")]:
        r = app.check_formula({"expr": bad})
        assert not r["ok"] and text in r["error"]
    assert not app.check_formula({"expr": "close", "mode": "sideways"})["ok"]


def test_guides_are_listed_and_readable_and_paths_cannot_escape(app):
    index = app.docs_index()["docs"]
    slugs = {d["slug"] for d in index}
    assert {"how_to_add_a_strategy", "glossary", "technique-momentum"} <= slugs and any(d["slug"].startswith("strategy-") for d in index)
    doc = app.doc("how_to_add_a_strategy")
    assert doc["title"] and "ForecastModel" in doc["markdown"] and not doc["markdown"].startswith("---")
    assert not app.doc("technique-momentum")["markdown"].startswith("---")
    for bad in ["../README", "technique-../../README", "technique-%2e%2e", "strategy-", "technique-nonexistent", "", "x" * 200, "etc/passwd", "technique-momentum/../.."]:
        with pytest.raises(ApiError):
            app.doc(bad)


def test_reload_picks_up_new_strategy_files(app, tmp_path):
    from src.framework import MODELS
    from src.strategies.user import new_strategy

    new_strategy(app.root, "web_test_idea")
    try:
        r = app.reload_strategies()
        assert r["files"]["web_test_idea.py"] == "" and "web_test_idea" in r["models"]
        entry = next(m for m in app.catalog()["models"] if m["name"] == "web_test_idea")
        assert entry["user"] and entry["params"][0]["name"] == "window"
    finally:
        MODELS.unregister("web_test_idea")
        (app.root / "user_strategies" / "web_test_idea.py").unlink()
        try:
            (app.root / "user_strategies").rmdir()
        except OSError:
            pass


# ------------------------------------------------------------------------------------------------------ HTTP
@pytest.fixture()
def live(app):
    running = server.start(app, "127.0.0.1", 0)
    yield running
    running.stop()


def _http(running, path, body=None, headers=None, method=None, token=True, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"X-Token": running.token} if token else {}
    if data is not None:
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    req = urllib.request.Request(running.url.rstrip("/") + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_the_page_carries_the_token_and_security_headers(live):
    status, body, headers = _http(live, "/", token=False)
    assert status == 200 and live.token.encode() in body and "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert _http(live, "/static/app.js", token=False)[0] == 200
    for path in ["/static/../../README.md", "/static/index.html", "/static/%2e%2e/api.py", "/nothing"]:
        assert _http(live, path, token=False)[0] == 404


def test_replit_preview_can_be_embedded_without_removing_api_token_checks(app, monkeypatch):
    monkeypatch.setenv("REPLIT_DEV_DOMAIN", "example.replit.dev")
    running = server.start(app, "0.0.0.0", 0)
    running.url = running.url.replace("0.0.0.0", "127.0.0.1")
    try:
        status, _, headers = _http(running, "/", token=False)
        assert status == 200
        policy = headers["Content-Security-Policy"]
        assert "frame-ancestors https://replit.com https://*.replit.com https://*.replit.dev" in policy
        assert _http(running, "/api/catalog", token=False)[0] == 403
        assert _http(running, "/api/catalog")[0] == 200
    finally:
        running.stop()


def test_exposed_server_outside_replit_still_blocks_embedding(app, monkeypatch):
    monkeypatch.delenv("REPLIT_DEV_DOMAIN", raising=False)
    running = server.start(app, "0.0.0.0", 0)
    running.url = running.url.replace("0.0.0.0", "127.0.0.1")
    try:
        assert "frame-ancestors 'none'" in _http(running, "/", token=False)[2]["Content-Security-Policy"]
    finally:
        running.stop()


def test_the_api_needs_the_token_the_right_host_and_a_json_body(live):
    assert _http(live, "/api/catalog", token=False)[0] == 403
    assert _http(live, "/api/catalog", headers={"X-Token": "wrong"})[0] == 403
    assert _http(live, "/api/catalog", headers={"Host": "evil.example.com"})[0] == 403
    assert _http(live, "/api/catalog", headers={"Origin": "http://evil.example.com"})[0] == 403
    assert _http(live, "/api/catalog")[0] == 200
    assert _http(live, "/api/tickers", raw=b"x=1", headers={"Content-Type": "text/plain"})[0] == 415
    assert _http(live, "/api/tickers", raw=b"{not json")[0] == 400
    assert _http(live, "/api/tickers", raw=b"[1, 2]")[0] == 400
    assert _http(live, "/api/tickers", raw=b"{" + b'"a":"' + b"x" * 300_000 + b'"}')[0] == 413
    assert _http(live, "/api/unknown", body={})[0] == 404
    status, body, _ = _http(live, "/api/tickers", body={"tickers": ["bad ticker!"]})
    assert status == 400 and "not a valid ticker" in json.loads(body)["error"]


def test_a_backtest_can_be_run_end_to_end_over_http(live):
    status, body, _ = _http(live, "/api/backtest", body={"models": [{"name": "momentum"}], "label": "http"})
    assert status == 200
    job = json.loads(body)["job"]
    for _ in range(600):
        status, body, _ = _http(live, f"/api/job/{job}")
        view = json.loads(body)
        if view["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert view["status"] == "done" and view["result"]["name"] == "http"
    assert _http(live, "/api/docs/glossary")[0] == 200 and _http(live, "/api/docs/..%2f..%2fREADME")[0] == 400


# ------------------------------------------------------------------------------------------------------ data sources: Yahoo and iTick
@pytest.fixture()
def itick(tmp_path):
    """An app whose iTick source talks to a fake service through a fake clock: ``(app, fake, clock, tmp_path)``."""
    config = load_config()
    fake, clock = FakeITick(), FakeClock()
    provider = ITickProvider(key=KEY, base_url="https://api-free.itick.org", limiter=RateLimiter(5, clock=clock.now, sleep=clock.sleep), transport=fake, sleep=clock.sleep)
    builder = UniverseBuilder(config, TickerStore(tmp_path / "yahoo", fetch=_fetch), default_loader=_default, sources={"itick": itick_source(tmp_path, provider, tmp_path / "itick")})
    return App(config, builder=builder, root=config.root), fake, clock, tmp_path


def test_the_catalog_lists_the_sources_without_ever_carrying_the_key(itick):
    app, fake, _, _ = itick
    sources = {s["name"]: s for s in app.catalog()["sources"]}
    assert set(sources) == {"yahoo", "itick"} and sources["yahoo"]["available"] and sources["itick"]["available"] and sources["itick"]["calls_per_minute"] == 5
    assert sources["itick"]["calls_available_now"] == 5 and "ITICK_API_KEY" in sources["itick"]["note"]
    assert KEY not in json.dumps(app.catalog()) and fake.calls == []
    assert sources["itick"]["setup"] == "" and "setup" not in sources["yahoo"]                       # nothing to set up once the key is there; Yahoo never needs it


def test_an_unset_key_shows_as_not_available_and_says_how_to_fix_it(tmp_path, monkeypatch):
    monkeypatch.delenv("ITICK_API_KEY", raising=False)
    config = load_config()
    builder = UniverseBuilder(config, TickerStore(tmp_path / "yahoo", fetch=_fetch), default_loader=_default, sources={"itick": itick_source(tmp_path)})
    app = App(config, builder=builder, root=config.root)
    entry = next(s for s in app.catalog()["sources"] if s["name"] == "itick")
    assert entry["available"] is False and "ITICK_API_KEY" in entry["problem"] and "Secrets" in entry["problem"]
    assert entry["setup"] == "key"                                                                  # the page shows its set-up guide for exactly this case
    rows = app.check_tickers({"tickers": ["AAA", "NVDA"], "source": "itick"})["tickers"]
    assert rows["AAA"]["ok"] and not rows["NVDA"]["ok"] and "not set up" in rows["NVDA"]["error"]
    with pytest.raises(ApiError, match="iTick is selected but iTick is not set up"):
        app.submit({"tickers": ["AAA", "NVDA"], "models": [{"name": "momentum"}], "source": "itick"})
    assert _run(app, tickers=["AAA", "BBB", "CCC"], source="itick")["status"] == "done"                  # nothing to download, so the source does not matter


def test_the_default_app_offers_both_sources(tmp_path, monkeypatch):
    monkeypatch.delenv("ITICK_API_KEY", raising=False)
    config = load_config()
    app = App(config, store=TickerStore(tmp_path / "prices", fetch=_fetch), root=tmp_path)
    assert {s["name"] for s in app.catalog()["sources"]} == {"yahoo", "itick"} and app.builder.sources["itick"].store.directory == tmp_path / "data" / "user" / "prices_itick"


def test_an_unknown_source_is_refused_everywhere(app):
    with pytest.raises(ApiError, match="unknown data source"):
        app.check_tickers({"tickers": ["NEWONE"], "source": "bloomberg"})
    with pytest.raises(ApiError, match="unknown data source"):
        app.submit({"tickers": ["AAA", "BBB"], "models": [{"name": "momentum"}], "source": "bloomberg"})
    with pytest.raises(ApiError, match="unknown data source"):
        app.test_source({"source": "bloomberg"})
    for odd in (["itick"], 5, {"a": 1}):
        with pytest.raises(ApiError, match="unknown data source"):
            app.check_tickers({"tickers": ["NEWONE"], "source": odd})
    assert app.test_source({"source": "yahoo"}) == {"ok": True, "message": "Yahoo Finance needs no key"}
    assert app.test_source({}) == {"ok": True, "message": "Yahoo Finance needs no key"}                       # no choice means the default


def test_checking_an_itick_ticker_spends_no_call_until_you_run(itick):
    app, fake, _, _ = itick
    rows = app.check_tickers({"tickers": ["NVDA", "0700.HK", "A$B".replace("$", ""), "AAA"], "source": "itick", "start": "2016-01-01"})["tickers"]
    assert fake.calls == []
    assert rows["NVDA"]["ok"] and rows["NVDA"]["pending"] and rows["NVDA"]["calls"] >= 2 and "not downloaded yet" in rows["NVDA"]["note"]
    assert rows["AAA"]["source"] == "platform"
    bad = app.check_tickers({"tickers": ["NVDA"], "source": "itick"})
    assert bad["ok"]
    with pytest.raises(ApiError, match="not a valid ticker"):
        app.check_tickers({"tickers": ["A$B"], "source": "itick"})


def test_itick_downloads_happen_in_the_run_are_cached_apart_from_yahoo_and_are_labelled(itick):
    app, fake, _, tmp = itick
    first = _run(app, tickers=["NVDA", "MSFT"], source="itick", start="2016-01-01", models=[{"name": "ma_crossover"}])
    assert first["status"] == "done", first
    r = first["result"]
    assert r["universe"]["source"] == "iTick" and r["universe"]["provider"] == "itick" and r["universe"]["tickers"] == ["NVDA", "MSFT"]
    calls = len(fake.calls)
    assert calls >= 4 and {c["code"] for c in fake.calls} == {"NVDA", "MSFT"}
    assert sorted(p.name for p in (tmp / "itick").glob("*.csv")) == ["MSFT.csv", "NVDA.csv"] and not list((tmp / "yahoo").glob("*.csv"))
    meta = json.loads((tmp / "itick" / "NVDA.json").read_text())
    assert meta["provider"] == "itick" and meta["days"] > 1000 and KEY not in json.dumps(meta)
    again = _run(app, tickers=["NVDA", "MSFT"], source="itick", start="2016-01-01", models=[{"name": "ma_crossover"}])
    assert again["status"] == "done" and len(fake.calls) == calls                                       # the second run is free
    checked = app.check_tickers({"tickers": ["NVDA", "MSFT"], "source": "itick", "start": "2016-01-01"})["tickers"]
    assert checked["NVDA"]["cached"] and checked["NVDA"]["days"] == meta["days"] and not checked["NVDA"].get("pending") and len(fake.calls) == calls
    mixed = _run(app, tickers=["AAA", "BBB", "NVDA"], source="itick", start="2016-01-01", models=[{"name": "ma_crossover"}])
    assert mixed["status"] == "done" and mixed["result"]["universe"]["source"] == "platform dataset + iTick" and len(fake.calls) == calls
    yahoo = _run(app, tickers=["AAA", "BBB", "NEWONE"], models=[{"name": "ma_crossover"}])                     # the same app can still download from Yahoo, into its own folder
    assert yahoo["status"] == "done" and yahoo["result"]["universe"]["source"] == "platform dataset + Yahoo Finance" and (tmp / "yahoo" / "NEWONE.csv").exists()


def test_a_top_up_costs_one_call_per_ticker_and_the_old_data_are_kept(itick):
    app, fake, clock, tmp = itick
    store = app.builder.sources["itick"].store
    store.get(["NVDA"], "2016-01-01")
    full_calls = len(fake.calls)
    before = pd.read_csv(tmp / "itick" / "NVDA.csv", parse_dates=["date"]).set_index("date")
    stale = TickerStore(tmp / "itick", fetch=app.builder.sources["itick"].provider, ttl=-1, provider="itick")
    frames, errors = stale.get(["NVDA"], "2016-01-01")
    assert not errors and len(fake.calls) - full_calls == 1
    pd.testing.assert_frame_equal(frames["NVDA"][["close"]], before[["close"]], check_freq=False)
    assert fake.calls[-1]["limit"] == "30" and "et" not in fake.calls[-1]


def test_a_failed_refresh_keeps_the_saved_copy_and_leaves_a_note(tmp_path):
    state = {"fail": False}

    def fetch(tickers, start):
        return {t: ("the service is down" if state["fail"] else _fetch(tickers, start)[t]) for t in tickers}

    store = TickerStore(tmp_path, fetch=fetch, ttl=-1)
    first, _ = store.get(["NEWONE"], "2013-01-01")
    state["fail"] = True
    frames, errors = store.get(["NEWONE"], "2013-01-01")
    assert not errors and len(frames["NEWONE"]) == len(first["NEWONE"])
    assert any("could not refresh (the service is down)" in w for w in store.notes(["NEWONE"])["NEWONE"])
    state["fail"] = False
    store.get(["NEWONE"], "2013-01-01")
    assert "NEWONE" not in store.notes(["NEWONE"])
    _, errors = store.get(["NEWNONE" if False else "BADONE"], "2013-01-01")
    assert "no data found" in errors["BADONE"]


def test_every_wait_for_the_rate_limit_is_applied_and_reported_while_the_run_downloads(itick):
    app, fake, clock, _ = itick
    messages = []
    app.builder.resolve(["NVDA", "MSFT", "AAPL"], None, "2012-01-01", False, "itick", messages.append)
    assert len(fake.calls) > 5 and sum(clock.slept) >= 60.0
    assert any("waiting" in m and "5 calls a minute" in m for m in messages) and all(m.startswith("iTick ") for m in messages)
    stamps = sorted(app.builder.sources["itick"].provider.limiter._stamps)
    assert all(stamps[i + 5] - stamps[i] >= 61.0 - 1e-6 for i in range(len(stamps) - 5))                 # never more than five calls in any minute


def test_a_job_shows_the_wait_in_its_progress_message(itick):
    app, fake, clock, _ = itick
    seen = []
    original = clock.sleep

    def watching(seconds):
        job = next(iter(app.jobs.values()), None)
        if job is not None and "waiting" in job.message:
            seen.append(job.message)
        original(seconds)

    app.builder.sources["itick"].provider.limiter._sleep = watching
    view = _run(app, tickers=["NVDA", "MSFT", "AAPL"], source="itick", start="2012-01-01", models=[{"name": "ma_crossover"}])
    assert view["status"] == "done" and seen and seen[0].startswith("iTick ") and "5 calls a minute" in seen[0]


def test_a_rejected_key_reaches_the_user_as_a_message_that_does_not_contain_the_key(tmp_path):
    config = load_config()
    fake, clock = FakeITick(key="the-real-key-is-different"), FakeClock()
    provider = ITickProvider(key=KEY, base_url="https://api-free.itick.org", limiter=RateLimiter(5, clock=clock.now, sleep=clock.sleep), transport=fake, sleep=clock.sleep)
    builder = UniverseBuilder(config, TickerStore(tmp_path / "yahoo", fetch=_fetch), default_loader=_default, sources={"itick": itick_source(tmp_path, provider, tmp_path / "itick")})
    app = App(config, builder=builder, root=config.root)
    view = _run(app, tickers=["NVDA", "MSFT"], source="itick", models=[{"name": "ma_crossover"}])
    assert view["status"] == "error" and "rejected the API key" in view["error"] and KEY not in json.dumps(view) and len(fake.calls) == 1
    out = app.test_source({"source": "itick"})
    assert not out["ok"] and "rejected" in out["message"] and KEY not in json.dumps(out)


def test_the_key_is_in_no_response_and_a_key_in_a_request_is_ignored(itick):
    app, fake, clock, tmp = itick
    seen = [app.catalog(), app.check_tickers({"tickers": ["NVDA", "NOPE1"], "source": "itick"}), app.test_source({"source": "itick"}), app.requirements({"models": [{"name": "momentum"}], "tickers": 1})]
    foreign = "SOMEONE-ELSES-KEY-0123456789"
    view = _run(app, tickers=["NVDA", "NOPE2"], source="itick", api_key=foreign, token=foreign, itick_api_key=foreign, models=[{"name": "ma_crossover"}])
    seen += [view]
    ok = _run(app, tickers=["NVDA", "MSFT"], source="itick", api_key=foreign, models=[{"name": "ma_crossover"}])
    seen += [ok, app.check_tickers({"tickers": ["MSFT"], "source": "itick", "api_key": foreign}), app.test_source({"source": "itick", "api_key": foreign, "token": foreign})]
    text = json.dumps(seen, default=str)
    assert KEY not in text and foreign not in text
    assert set(fake.tokens) == {KEY}                                                                       # the only key ever sent is the configured one
    for path in tmp.rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text(errors="ignore")                                            # nor is it written into the cache
    assert view["status"] == "error" and "NOPE2" in view["error"] and "does not have this symbol" in view["error"]


def test_an_unadjusted_split_is_flagged_on_the_ticker_and_in_the_result(tmp_path):
    config = load_config()
    fake, clock = FakeITick(split=("2022-06-01", 4.0)), FakeClock()
    provider = ITickProvider(key=KEY, base_url="https://api-free.itick.org", limiter=RateLimiter(5, clock=clock.now, sleep=clock.sleep), transport=fake, sleep=clock.sleep)
    builder = UniverseBuilder(config, TickerStore(tmp_path / "yahoo", fetch=_fetch), default_loader=_default, sources={"itick": itick_source(tmp_path, provider, tmp_path / "itick")})
    app = App(config, builder=builder, root=config.root)
    view = _run(app, tickers=["NVDA"], source="itick", start="2016-01-01", models=[{"name": "ma_crossover"}])
    assert view["status"] == "done"
    notes = view["result"]["universe"]["data_notes"]
    assert "-for-1 split that iTick did not adjust for" in notes["NVDA"][0] and notes["NVDA"][0].startswith("2022-06-01")
    row = app.check_tickers({"tickers": ["NVDA"], "source": "itick", "start": "2016-01-01"})["tickers"]["NVDA"]
    assert row["cached"] and "-for-1 split" in row["warnings"][0]


def test_an_itick_ticker_with_too_little_history_is_refused_with_the_count(itick):
    app, fake, _, _ = itick
    fake.history_start["TINYCO"] = fake.last - pd.Timedelta(days=200)
    view = _run(app, tickers=["TINYCO"], source="itick", models=[{"name": "ma_crossover"}])
    assert view["status"] == "error" and "too little history" in view["error"] and "TINYCO" in view["error"]


def test_the_test_endpoint_reports_the_connection_without_the_key(itick):
    app, fake, _, _ = itick
    out = app.test_source({"source": "itick"})
    assert out["ok"] and out["bars"] == 10 and "iTick works" in out["message"] and KEY not in json.dumps(out) and len(fake.calls) == 1


# ------------------------------------------------------------------------------------------------------ one ticker, strategies that need more
def test_the_catalog_states_its_version_and_that_the_code_has_not_changed_since_launch(app):
    from src.webapp import api

    catalog = app.catalog()
    assert catalog["api_version"] == api.API_VERSION == 2
    assert catalog["restart_needed"] is False                       # nothing under src/ was touched while this test process ran


def test_the_code_signature_changes_when_a_python_file_is_edited_added_or_removed(tmp_path):
    import os

    from src.webapp.api import code_signature

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__pycache__").mkdir()
    one, two = tmp_path / "one.py", tmp_path / "pkg" / "two.py"
    one.write_text("x = 1\n"), two.write_text("y = 2\n")
    (tmp_path / "pkg" / "__pycache__" / "two.cpython-310.pyc").write_bytes(b"compiled")      # not source: ignored
    (tmp_path / "notes.txt").write_text("not python")
    for p in (one, two):
        os.utime(p, (1_700_000_000, 1_700_000_000))
    before = code_signature(tmp_path)
    assert before == (2, 1_700_000_000.0)
    assert code_signature(tmp_path) == before                       # looking changes nothing

    os.utime(two, (1_700_000_500, 1_700_000_500))                   # edited (a git pull replaces the file: newer modification time)
    edited = code_signature(tmp_path)
    assert edited == (2, 1_700_000_500.0) and edited != before

    three = tmp_path / "three.py"
    three.write_text("z = 3\n"), os.utime(three, (1_600_000_000, 1_600_000_000))
    assert code_signature(tmp_path) == (3, 1_700_000_500.0) != edited       # added with an OLD modification time: the count still gives it away
    three.unlink()
    assert code_signature(tmp_path) == edited


def test_the_catalog_asks_for_a_restart_when_the_code_changed_after_the_app_started(app, monkeypatch):
    from src.webapp import api

    count, newest = api.LAUNCH_SIGNATURE
    monkeypatch.setattr(api, "LAUNCH_SIGNATURE", (count, newest - 60.0))        # as if a file had been replaced since launch
    assert app.catalog()["restart_needed"] is True
    monkeypatch.setattr(api, "LAUNCH_SIGNATURE", (count - 1, newest))           # or a file added
    assert app.catalog()["restart_needed"] is True


def test_the_catalog_says_how_many_tickers_each_strategy_needs(app):
    models = {m["name"]: m for m in app.catalog()["models"]}
    assert models["tsmom"]["min_assets"] == 1 and models["ma_crossover"]["min_assets"] == 1 and models["momentum"]["min_assets"] == 2 and models["bab"]["min_assets"] == 4
    assert models["expression"]["min_assets"] == 2 and all(isinstance(m["min_assets"], int) and m["min_assets"] >= 1 for m in models.values())


def test_requirements_tell_the_page_whether_the_chosen_tickers_are_enough(app):
    one = app.requirements({"models": [{"name": "momentum"}], "tickers": 1})
    assert one["need"] == 2 and one["have"] == 1 and not one["ok"] and "ranks the tickers against each other" in one["message"]
    assert app.requirements({"models": [{"name": "momentum"}], "tickers": 2})["ok"]
    assert app.requirements({"models": [{"name": "tsmom"}], "tickers": 1})["ok"]
    assert not app.requirements({"models": [{"name": "tsmom"}], "tickers": 0})["ok"]
    sized = app.requirements({"models": [{"name": "tsmom"}], "allocator": "min_variance", "tickers": 2})
    assert sized["need"] == 3 and not sized["ok"] and "min_variance" in sized["message"]
    with pytest.raises(ApiError, match="unknown strategy"):
        app.requirements({"models": [{"name": "nope"}], "tickers": 1})
    with pytest.raises(ApiError, match="count"):
        app.requirements({"models": [{"name": "tsmom"}], "tickers": "many"})


def test_a_strategy_that_works_per_ticker_runs_on_one_ticker_and_reports_buy_and_hold(app):
    view = _run(app, tickers=["AAA"], models=[{"name": "ma_crossover"}], allocator="sleeves")                    # the page picks this allocation for a per-ticker rule; the API takes what it is given
    assert view["status"] == "done", view
    r = view["result"]
    assert r["universe"]["tickers"] == ["AAA"] and set(r["benchmarks"]) == {"equal_weight"} and r["held_days"] > 200 and r["spec"]["execution"]["min_assets"] == 1
    assert "risk_parity" not in {b["benchmark"] for b in r["tables"]["benchmarks"]} and not any("error" in b for b in r["tables"]["benchmarks"])
    json.dumps(r, allow_nan=False)


def test_a_strategy_that_ranks_tickers_is_refused_on_one_ticker_before_anything_runs(app):
    with pytest.raises(ApiError, match="'momentum' ranks the tickers against each other, so it needs at least 2 tickers and you have 1"):
        app.submit({"tickers": ["AAA"], "models": [{"name": "momentum"}]})
    with pytest.raises(ApiError, match="ranks the tickers"):
        app.submit({"tickers": ["AAA"], "models": [{"name": "momentum"}], "allocator": "sleeves"})
    with pytest.raises(ApiError, match="needs at least 4 tickers"):
        app.submit({"tickers": ["AAA", "BBB", "CCC"], "models": [{"name": "bab"}]})
    with pytest.raises(ApiError, match="'static' allocation"):
        app.submit({"tickers": ["AAA"], "models": [{"name": "tsmom"}], "allocator": "static", "allocator_params": {"book": "hrp"}})
    assert _run(app, tickers=["AAA", "BBB"])["status"] == "done"                                     # two tickers are enough for a ranking
    with pytest.raises(ApiError, match="not a valid ticker"):
        app.submit({"tickers": ["A B$"], "models": [{"name": "tsmom"}]})


def test_the_earnings_numbers_add_up_on_a_real_run(app):
    view = _run(app, tickers=["AAA"], models=[{"name": "ma_crossover"}], allocator="sleeves", capital=250_000)
    r = view["result"]
    e, t = r["earnings"], r["trades"]
    assert r["capital"] == e["capital"] == 250_000
    assert e["end_value"] == pytest.approx(250_000 * r["equity"][-1], rel=2e-4)                          # the equity curve in the payload is rounded to four decimals
    assert e["net_profit"] == pytest.approx(e["end_value"] - 250_000) and e["total_return"] == pytest.approx(e["end_value"] / 250_000 - 1)
    assert sum(m["pnl"] for m in e["monthly"]) == pytest.approx(e["net_profit"], abs=1e-6 * 250_000) and sum(a["pnl"] for a in e["annual"]) == pytest.approx(e["net_profit"], abs=1e-6 * 250_000)
    assert e["costs_paid"] > 0 and e["gross_profit"] == pytest.approx(e["net_profit"] + e["costs_paid"])
    assert sum(trip["pnl"] for trip in t["round_trips"]) == pytest.approx(e["gross_profit"], rel=1e-6, abs=0.01)         # every dollar made before costs belongs to some round trip
    assert t["stats"]["round_trips"] + t["stats"]["open_positions"] == t["total"] > 3 and t["stats"]["buys"] > 0 and t["stats"]["sells"] > 0
    assert 0 < t["stats"]["days_in_market"] <= 1 and e["exposure"]["peak"] <= 1.0 + 1e-9                          # the crossover rule's score is bounded: no leverage on one ticker
    assert e["benchmarks"]["equal_weight"]["pnl"] == pytest.approx(250_000 * (r["benchmarks"]["equal_weight"][-1] - 1.0), rel=2e-4)
    json.dumps(r, allow_nan=False)


def test_the_starting_capital_defaults_to_the_aum_then_to_one_hundred_thousand(app):
    assert _run(app, tickers=["AAA", "BBB"])["result"]["capital"] == 100_000
    assert _run(app, tickers=["AAA", "BBB"], aum=5e7)["result"]["capital"] == 5e7
    assert _run(app, tickers=["AAA", "BBB"], aum=5e7, capital=20_000)["result"]["capital"] == 20_000
    for bad, message in [(5, "between"), ("abc", "must be a number"), (1e13, "between")]:
        with pytest.raises(ApiError, match=message):
            app.submit({"tickers": ["AAA", "BBB"], "models": [{"name": "momentum"}], "capital": bad})


def test_a_strategy_that_never_trades_still_gives_a_clean_result(app):
    view = _run(app, tickers=["AAA"], models=[{"name": "expression", "params": {"expr": "close * 0", "mode": "time_series"}}], allocator="score_stack", allocator_params={"mode": "time_series"})
    if view["status"] == "done":
        json.dumps(view["result"], allow_nan=False)
        assert view["result"]["trades"]["stats"]["round_trips"] == 0 or view["result"]["trades"]["total"] == 0


def test_new_routes_need_the_token_and_answer_in_json(live):
    assert _http(live, "/api/requirements", body={"models": [{"name": "momentum"}], "tickers": 1}, token=False)[0] == 403
    status, body, _ = _http(live, "/api/requirements", body={"models": [{"name": "momentum"}], "tickers": 1})
    assert status == 200 and json.loads(body)["need"] == 2 and not json.loads(body)["ok"]
    assert _http(live, "/api/source/test", body={"source": "yahoo"}, token=False)[0] == 403
    status, body, _ = _http(live, "/api/source/test", body={"source": "yahoo"})
    assert status == 200 and json.loads(body)["ok"]
    status, body, _ = _http(live, "/api/source/test", body={"source": "itick"})
    assert status == 400 and "unknown data source" in json.loads(body)["error"]
    status, body, _ = _http(live, "/api/requirements", body={"models": [{"name": "nope"}], "tickers": 1})
    assert status == 400 and "unknown strategy" in json.loads(body)["error"]


def test_without_a_start_date_a_rate_limited_source_downloads_ten_years_not_twenty(itick):
    app, fake, _, _ = itick
    row = app.check_tickers({"tickers": ["NVDA"], "source": "itick"})["tickers"]["NVDA"]
    with_start = app.check_tickers({"tickers": ["NVDA"], "source": "itick", "start": "2005-01-01"})["tickers"]["NVDA"]
    assert 2 <= row["calls"] < with_start["calls"] and fake.calls == []
    view = _run(app, tickers=["NVDA"], source="itick", models=[{"name": "ma_crossover"}])
    assert view["status"] == "done" and view["result"]["universe"]["first"] >= f"{pd.Timestamp.now().year - 10}-01-01"
    assert max(int(c["limit"]) for c in fake.calls) <= 1000
    assert _run(app, tickers=["AAA", "BBB", "CCC"], source="itick", models=[{"name": "ma_crossover"}])["result"]["universe"]["source"] == "platform dataset"     # nothing to download: the platform's own history is untouched
    yahoo = app.check_tickers({"tickers": ["NEWONE"]})["tickers"]["NEWONE"]
    assert yahoo["ok"] and yahoo["first"] < "2015-01-01"                                       # Yahoo keeps its 2005 default


def test_a_formula_is_not_checked_against_tickers_a_rate_limited_source_has_yet_to_download(itick):
    app, fake, _, _ = itick
    body = {"expr": "mom(60)", "mode": "cross_sectional", "tickers": ["NVDA", "MSFT"], "source": "itick"}
    out = app.check_formula(body)
    assert not out["ok"] and "NVDA, MSFT have not been downloaded from iTick yet" in out["error"] and fake.calls == []
    assert _run(app, tickers=["NVDA", "MSFT"], source="itick", models=[{"name": "ma_crossover"}])["status"] == "done"
    calls = len(fake.calls)
    ok = app.check_formula(body)
    assert ok["ok"] and {r["ticker"] for r in ok["latest"]} == {"NVDA", "MSFT"} and len(fake.calls) == calls
    assert app.check_formula({**body, "tickers": ["AAA", "BBB"]})["ok"]
