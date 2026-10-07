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
from src.webapp.universe import TickerStore, UniverseBuilder, UniverseError, normalise

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
    with pytest.raises(UniverseError, match="at least 2"):
        app.builder.resolve(["AAA"], None, None)
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
