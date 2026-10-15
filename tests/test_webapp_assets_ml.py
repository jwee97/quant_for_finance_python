"""The endpoints of the Options, Futures, Crypto and Machine learning workspaces: validated like any other request, labelled with where their numbers come from, and (for machine learning) never
training a model outside the walk-forward, costed, segment-split pipeline."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from src.ml import optimization as opt
from src.utils.config import load_config
from src.webapp import api, server
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch, _http
from tests.test_webapp_labs import _rich

CONFIG = load_config()
HAS_CRYPTO = (CONFIG.root / "data" / "raw" / "crypto" / "BTC_funding.csv.gz").exists()
needs_crypto = pytest.mark.skipif(not HAS_CRYPTO, reason="the Deribit crypto data is not in this checkout")


def _app(tmp_path, loader, root=None) -> App:
    return App(CONFIG, builder=UniverseBuilder(CONFIG, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=loader), root=root or CONFIG.root)


def _safe(value):
    json.dumps(value, allow_nan=False)                           # the server refuses NaN and infinity, so a payload that carried one would be a 500
    return value


def wait(app: App, response: dict, timeout: float = 600.0) -> dict:
    """The finished job behind a submitted task (the page polls ``/api/job/<id>`` the same way)."""
    t0 = time.time()
    while True:
        job = app.job(response["job"])
        if job["status"] in ("done", "error"):
            return job
        assert time.time() - t0 < timeout, job
        time.sleep(0.1)


def finished(app: App, response: dict) -> dict:
    job = wait(app, response)
    assert job["status"] == "done", job["error"]
    return _safe(job["result"])


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return _app(tmp_path_factory.mktemp("assets"), _rich)


@pytest.fixture(scope="module")
def mlapp(tmp_path_factory):
    """An app whose data is a synthetic market with a planted, persistent signal, and whose registry lives in a temporary folder."""
    tmp = tmp_path_factory.mktemp("ml")
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


# ------------------------------------------------------------------------------------------------------------------ the task queue
def test_a_task_runs_on_the_worker_and_reports_its_result_or_its_message(assets):
    ok = finished(assets, assets.submit_task("fine", lambda progress: (progress("halfway"), {"x": float("nan"), "n": np.int64(3)})[1]))
    assert ok == {"x": None, "n": 3}                                                    # NaN cannot be sent; numpy integers become integers
    bad = wait(assets, assets.submit_task("refused", lambda progress: (_ for _ in ()).throw(ApiError("choose fewer assets"))))
    assert bad["status"] == "error" and bad["error"] == "choose fewer assets" and bad["result"] is None
    crash = wait(assets, assets.submit_task("bug", lambda progress: 1 / 0))
    assert crash["status"] == "error" and "ZeroDivisionError" in crash["error"]


def test_the_version_announces_the_new_workspaces_and_the_routes_are_registered(assets):
    assert assets.catalog()["api_version"] == api.API_VERSION == 8
    for path in ("/api/options/chain", "/api/options/surface", "/api/futures/curve", "/api/futures/roll", "/api/crypto/funding", "/api/ml/catalog", "/api/ml/experiments"):
        assert path in assets.extra_get, path
    for path in ("/api/options/portfolio-risk", "/api/options/strategy/payoff", "/api/options/strategy/backtest", "/api/ml/train", "/api/ml/search", "/api/ml/combine"):
        assert path in assets.extra_post, path


# ------------------------------------------------------------------------------------------------------------------ options
def test_the_catalog_names_every_underlying_root_symbol_and_strategy_with_its_source(assets):
    cat = _safe(assets.extra_get["/api/assets/catalog"]({}))
    assert [u["id"] for u in cat["underlyings"]][:3] == ["AAA", "BBB", "CCC"] and {u["source"] for u in cat["underlyings"]} == {"simulated chain on real prices"}
    assert {r["id"] for r in cat["futures_roots"]} >= {"CL", "NG", "GC", "HG", "ES"} and {s["id"] for s in cat["crypto_symbols"]} >= {"BTC", "ETH", "SOL"}
    sources = {s["id"]: s["source"] for s in cat["crypto_symbols"]}
    assert sources["BTC"] == "platform" and sources["SOL"] == "simulated"
    assert {s["id"] for s in cat["option_strategies"]} >= {"long_straddle", "iron_condor", "bull_call_spread", "covered_call"}
    bt = {b["id"]: b for b in cat["option_backtests"]}
    from src.assets import options_analytics as oa

    assert set(bt) == set(oa.BACKTEST_STRATEGIES)                                                             # every rolling strategy is offered, each with its form fields
    assert {p["name"] for p in bt["iron_condor"]["params"]} >= {"short_delta", "wing_delta", "dte"} and all(p["text"] for p in bt["short_put"]["params"])


def test_the_option_chain_is_labelled_as_a_model_and_shows_both_sides_at_each_strike(assets):
    out = _safe(assets.extra_get["/api/options/chain"]({"underlying": "AAA"}))
    assert out["source"] == "mixed" and out["sources"]["chain"] == "simulated" and out["sources"]["underlying_price"] == "platform"
    assert any("MODEL" in n for n in out["notes"]) and out["as_of"] == str(assets.builder.default_bundle().index[-1].date())
    rows = out["chain"]["rows"]
    assert sum(r["atm"] for r in rows) == 1 and all({"bid", "ask", "iv", "delta", "volume", "oi", "stale"} <= set(r["call"]) for r in rows)
    assert [r["strike"] for r in rows] == sorted(r["strike"] for r in rows) and out["overview"]["iv_rank_basis"]
    nxt = out["chain"]["expiries"][3]["expiry"]
    wide = assets.extra_get["/api/options/chain"]({"underlying": "AAA", "expiry": nxt, "window": "0.2"})
    narrow = assets.extra_get["/api/options/chain"]({"underlying": "AAA", "expiry": nxt, "window": "0.05"})
    assert wide["chain"]["expiry"] == narrow["chain"]["expiry"] == nxt and len(narrow["chain"]["rows"]) < len(wide["chain"]["rows"])          # the moneyness filter narrows the strikes shown
    with pytest.raises(ApiError, match="unknown underlying"):
        assets.extra_get["/api/options/chain"]({"underlying": "NOPE"})
    with pytest.raises(ApiError, match="window"):
        assets.extra_get["/api/options/chain"]({"underlying": "AAA", "window": "9"})


def test_the_surface_endpoint_returns_a_grid_a_term_structure_and_the_no_arbitrage_checks(assets):
    out = _safe(assets.extra_get["/api/options/surface"]({"underlying": "BBB"}))
    s = out["surface"]
    assert len(s["iv"]) == len(s["expiries"]) and len(s["iv"][0]) == len(s["moneyness"]) and out["source"] == "simulated"
    assert len(out["term"]) >= 5 and out["term_metrics"] and out["skews"] and isinstance(out["arbitrage"], list)
    assert out["arbitrage"] and all(r["violations"] == 0 for r in out["arbitrage"]) and out["arbitrage_tick"] >= 0.01 and out["arbitrage_tolerance"] > 0, "the model chain is quoted without noise, so every static check is clean up to the rounding of a price to a tick"


def test_a_long_straddle_costs_its_premium_has_two_breakevens_and_no_upper_bound(assets):
    out = _safe(assets.extra_post["/api/options/strategy/payoff"]({"underlying": "AAA", "strategy": "long_straddle", "dte": 30}))
    a = out["analysis"]
    assert a["unlimited_profit"] and not a["unlimited_loss"] and a["max_loss"] == pytest.approx(-a["net_debit"]) and len(a["breakevens"]) == 2
    assert a["breakevens"][0] < a["spot"] < a["breakevens"][1] and 0.0 < a["prob_profit"] < 1.0 and a["margin"]["defined_risk"]
    assert {"expiry", "now"} <= set(a["curve"]) and "MODEL" not in json.dumps(a)                       # the curves are numbers; the provenance is in the envelope
    sold = assets.extra_post["/api/options/strategy/payoff"]({"underlying": "AAA", "strategy": "short_strangle", "dte": 30})["analysis"]
    assert sold["unlimited_loss"] and sold["net_credit"] > 0 and sold["max_loss"] is None and any("naked" in f.lower() or "unlimited" in f.lower() for f in sold["risk_flags"])
    custom = assets.extra_post["/api/options/strategy/payoff"]({"underlying": "AAA", "legs": [{"type": "option", "right": "C", "strike": 220, "quantity": 1, "dte": 45}, {"type": "stock", "quantity": 100}]})["analysis"]
    assert len(custom["legs"]) == 2
    for bad in ({"strategy": "nope"}, {"legs": [{"type": "option", "right": "X", "strike": 100}]}, {"legs": "all"}, {"dte": 0}):
        with pytest.raises((ApiError, ValueError)):
            assets.extra_post["/api/options/strategy/payoff"]({"underlying": "AAA", **bad})


def test_portfolio_risk_stresses_spot_and_volatility_and_refuses_a_malformed_book(assets):
    book = [{"type": "option", "right": "P", "strike": 190, "dte": 30, "quantity": -5}, {"type": "option", "right": "C", "strike": 230, "dte": 30, "quantity": -5}, {"type": "stock", "quantity": 100}]
    out = _safe(assets.extra_post["/api/options/portfolio-risk"]({"underlying": "AAA", "positions": book, "spot_shocks": [-0.1, 0, 0.1], "vol_shocks": [0, 0.1]}))
    assert out["source"] == "mixed" and len(out["stress"]) == 6 and out["worst"]["pnl"] == min(r["pnl"] for r in out["stress"]) and out["totals"]["vega_per_point"] < 0
    assert {(r["spot_shock"], r["vol_shock"]) for r in out["stress"]} == {(a, b) for a in (-0.1, 0.0, 0.1) for b in (0.0, 0.1)} and any("short vega" in f for f in out["risk_flags"])
    with pytest.raises(ApiError, match="positions"):
        assets.extra_post["/api/options/portfolio-risk"]({"underlying": "AAA", "positions": []})
    with pytest.raises(ApiError, match="spot shocks"):
        assets.extra_post["/api/options/portfolio-risk"]({"underlying": "AAA", "positions": book, "spot_shocks": [-5]})


def test_option_strategy_settings_are_whitelisted_cast_and_checked():
    from src.assets import options_analytics as oa

    assert oa.parse_backtest_params("short_put", {"delta": "0.2", "dte": 30.0}) == {"delta": 0.2, "dte": 30}
    assert oa.parse_backtest_params("short_strangle", {"hedge": "true"}) == {"hedge": True} and oa.parse_backtest_params("delta_hedged_straddle", {"side": 1}) == {"side": 1}
    assert oa.parse_backtest_params("short_put", None) == {}
    for name, given, text in (("short_put", {"bogus": 1}, "no setting"), ("short_put", {"delta": 3}, "between"), ("delta_hedged_straddle", {"side": 0}, "one of"), ("iron_condor", {"wing_delta": 0.3}, "smaller"),
                              ("calendar_spread", {"front_dte": 60, "back_dte": 62}, "longer"), ("nope", {}, "unknown strategy"), ("short_put", {"delta": "x"}, "number")):
        with pytest.raises(ValueError, match=text):
            oa.parse_backtest_params(name, given)


def test_an_option_strategy_backtest_is_a_job_on_the_simulated_market_and_says_so(assets):
    out = finished(assets, assets.extra_post["/api/options/strategy/backtest"]({"strategy": "short_put", "days": 90, "params": {"delta": 0.2}}))
    assert out["source"] == "simulated" and out["strategy"] == "short_put" and out["params"] == {"delta": 0.2} and len(out["equity"]) == len(out["dates"])
    assert any("simulated" in n for n in out["notes"]) and "sharpe" in out["summary"] and out["trades"] > 0
    for bad, text in (({"strategy": "nope"}, "strategy must"), ({"strategy": "short_put", "params": {"bogus": 1}}, "no setting"), ({"days": 10}, "days")):
        with pytest.raises(ApiError, match=text):
            assets.extra_post["/api/options/strategy/backtest"]({"strategy": "short_put", **bad} if "strategy" not in bad else bad)


# ------------------------------------------------------------------------------------------------------------------ futures
def test_the_futures_curve_roll_and_margin_endpoints_label_their_simulated_data(assets):
    cur = _safe(assets.extra_get["/api/futures/curve"]({"root": "CL"}))
    assert cur["source"] == "simulated" and cur["root"] == "CL" and cur["curve"]["state"] and cur["spec"]["multiplier"] == 1000 and cur["sources"]["open_interest"] == "simulated"
    assert len(cur["curve"]["rows"]) >= 6 and cur["carry"]["series"]["dates"] and cur["basis"] and any("simulated" in n.lower() for n in cur["notes"]) and len(cur["history"]) == 3
    roll = _safe(assets.extra_get["/api/futures/roll"]({"root": "CL", "roll_days": "3"}))
    assert roll["roll"]["current_contract"] != roll["roll"]["next_contract"] and roll["roll"]["suggested_roll_date"] and roll["roll"]["roll_days"] == 3 and roll["roll"]["past_rolls"]
    margin = _safe(assets.extra_post["/api/futures/margin"]({"positions": [{"root": "CL", "quantity": 20}, {"root": "ES", "quantity": -1}], "equity": 100_000}))
    assert margin["initial_margin"] > 0 and margin["margin_usage"] > 0.7 and any("margin use" in f for f in margin["risk_flags"]) and margin["source"] == "simulated" and set(margin["by_root"]) == {"CL", "ES"}
    for fn, arg, text in ((assets.extra_get["/api/futures/curve"], {"root": "ZZ"}, "unknown root"), (assets.extra_get["/api/futures/roll"], {"root": "CL", "roll_days": 999}, "roll days"),
                          (assets.extra_post["/api/futures/margin"], {"positions": []}, "positions"), (assets.extra_post["/api/futures/margin"], {"positions": [{"root": "ZZ", "quantity": 1}]}, "unknown root")):
        with pytest.raises(ApiError, match=text):
            fn(arg)


def test_a_futures_file_of_your_own_replaces_the_simulation(tmp_path):
    app = _app(tmp_path, _rich, root=tmp_path)
    d = tmp_path / "data" / "user" / "futures"
    d.mkdir(parents=True)
    dates = pd.bdate_range("2023-01-02", periods=120)
    rows = []
    for k, (code, expiry) in enumerate((("MYX1", "2023-09-15"), ("MYX2", "2023-12-15"), ("MYX3", "2024-03-15"))):
        for t in dates:
            rows.append({"date": t.date(), "contract": code, "expiry": expiry, "price": 100.0 + 2.0 * k + 0.01 * (t - dates[0]).days, "open_interest": 1000 * (3 - k)})
    pd.DataFrame(rows).to_csv(d / "MYX.csv", index=False)
    cur = app.extra_get["/api/futures/curve"]({"root": "MYX"})
    assert cur["source"] == "mixed" and cur["sources"]["curve"] == "user_file" and cur["sources"]["margins"] == "simulated" and len(cur["curve"]["rows"]) == 3 and cur["curve"]["state"]
    cat = app.extra_get["/api/assets/catalog"]({})
    assert {"id": "MYX", "source": "user_file"}.items() <= next(r for r in cat["futures_roots"] if r["id"] == "MYX").items()
    (d / "BAD.csv").write_text("date,contract\n2023-01-02,X\n")
    with pytest.raises(ApiError, match="BAD.csv"):
        app.extra_get["/api/futures/curve"]({"root": "BAD"})


# ------------------------------------------------------------------------------------------------------------------ crypto
@needs_crypto
def test_crypto_overview_funding_and_cross_exchange_say_which_numbers_are_data(assets):
    ov = _safe(assets.extra_get["/api/crypto/overview"]({"symbol": "BTC"}))
    assert ov["sources"]["prices"] == "platform" and ov["sources"]["order_book"] == "simulated" and len(ov["overview"]["cards"]) >= 5 and ov["spot_perp"] and set(ov["books"]) == {"DERIBIT", "BINANCE", "BYBIT", "OKX"}
    fund = _safe(assets.extra_get["/api/crypto/funding"]({"symbols": "BTC,ETH,SOL"}))
    assert fund["sources"]["BTC@DERIBIT"] == "platform" and fund["sources"]["SOL@DERIBIT"] == "simulated" and fund["sources"]["BTC@BINANCE"] == "simulated" and fund["sources"]["ETH@OKX"] == "simulated"
    assert any("Funding that is data: BTC on DERIBIT, ETH on DERIBIT" in n for n in fund["notes"]), "the page is told which rows are data and which are not, by name"
    cross = _safe(assets.extra_get["/api/crypto/cross-exchange"]({"symbol": "ETH", "slippage_bps": "2"}))
    assert cross["pairs"] and len(cross["quotes"]) == 4 and any("indicative" in n for n in cross["notes"]) and all("net_bps" in r for r in cross["pairs"])
    for fn, arg, text in ((assets.extra_get["/api/crypto/overview"], {"symbol": "DOGE!"}, "letters and digits"), (assets.extra_get["/api/crypto/funding"], {"symbols": ",".join("ABCDEFG")}, "at most"),
                          (assets.extra_get["/api/crypto/overview"], {"symbol": "ZZZ"}, "unknown symbol")):
        with pytest.raises(ApiError, match=text):
            fn(arg)


@needs_crypto
def test_a_crypto_strategy_backtest_runs_through_the_pipeline_and_refuses_data_it_lacks(assets):
    out = finished(assets, assets.extra_post["/api/crypto/strategy/backtest"]({"strategy": "funding_carry"}))
    assert out["source"] == "platform" and out["metrics"]["sharpe"] == pytest.approx(out["metrics"]["sharpe"]) and len(out["equity"]) == len(out["dates"]) and out["fees_bps"] == 5.0
    with pytest.raises(ApiError, match="exchange net-flow"):
        assets.extra_post["/api/crypto/strategy/backtest"]({"strategy": "exchange_flow"})
    with pytest.raises(ApiError, match="platform's data"):
        assets.extra_post["/api/crypto/strategy/backtest"]({"strategy": "spot_momentum", "assets": ["SOL"]})


def test_your_own_crypto_file_is_analysed_and_marked_as_yours(tmp_path):
    app = _app(tmp_path, _rich, root=tmp_path)
    d = tmp_path / "data" / "user" / "crypto"
    d.mkdir(parents=True)
    t = pd.date_range("2024-01-01", periods=400, freq="D")
    spot = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.02, 400))
    pd.DataFrame({"time": t, "spot": spot, "perp": spot * 1.001, "funding_daily": 0.0003}).to_csv(d / "MYC.csv", index=False)
    out = _safe(app.extra_get["/api/crypto/overview"]({"symbol": "MYC"}))
    assert out["sources"]["prices"] == "user_file" and out["spot_perp"] and out["basis"]


# ------------------------------------------------------------------------------------------------------------------ machine learning: catalogue, validation
def test_the_ml_catalog_lists_models_features_targets_rules_and_the_timing_convention(mlapp):
    cat = _safe(mlapp.extra_get["/api/ml/catalog"]({}))
    kinds = {m["kind"]: m for m in cat["models"]}
    assert {"ridge", "lasso", "random_forest", "hist_gb", "kalman", "online_ridge"} <= set(kinds) and kinds["ridge"]["available"] and kinds["random_forest"]["refit_every"] == 63
    assert [g["id"] for g in cat["feature_groups"]][:3] == ["price", "momentum", "mean_reversion"] and sum(g["n"] for g in cat["feature_groups"]) >= 80
    assert {t["kind"] for t in cat["targets"]} >= {"fwd_ret", "direction", "rank", "excess_ret"} and [s["id"] for s in cat["signals"]][0] == "top_n_long_short"
    assert cat["defaults"]["dataset"]["lag"] == 1 and cat["defaults"]["split"]["forward_fraction"] == 0.15 and cat["registry"]["experiments"] == 0
    assert "test and forward" in cat["timing"]["segments"] and {u["id"] for u in cat["universes"]} == {"etf", "crypto", "futures"}
    assert (mlapp.root / "artifacts").exists() is False, "reading the catalogue must not create the registry"


def test_a_booster_that_cannot_load_is_listed_as_unavailable_and_refused_with_its_reason(mlapp, boosters_without_openmp):
    """The tab used to fail to open with ``OSError: libgomp.so.1: cannot open shared object file`` on a machine without the OpenMP runtime, because the catalogue imported LightGBM to see whether it was there."""
    cat = _safe(mlapp.extra_get["/api/ml/catalog"]({}))
    kinds = {m["kind"]: m for m in cat["models"]}
    assert kinds["lightgbm"]["available"] is False and "libgomp.so.1" in kinds["lightgbm"]["reason"] and kinds["xgboost"]["available"] is False
    assert all(kinds[k]["available"] for k in ("ridge", "random_forest", "hist_gb")) and cat["defaults"]["model"]["kind"] == "ridge"
    queued = len(mlapp.jobs)
    for path, body, text in (("/api/ml/train", {"model": {"kind": "lightgbm"}}, r"LightGBM model cannot be used here: .*libgomp"), ("/api/ml/search", {"kinds": ["ridge", "xgboost"]}, "kinds must be"),
                             ("/api/ml/tune", {"model": {"kind": "xgboost"}}, r"XGBoost model cannot be used here")):
        with pytest.raises(ApiError, match=text):
            mlapp.extra_post[path](body)
    assert len(mlapp.jobs) == queued, "nothing refused may have been queued"


def test_a_training_request_is_whitelisted_ranged_and_validated_before_it_is_queued(mlapp):
    train = mlapp.extra_post["/api/ml/train"]
    for body, text in (({"signal": {"bogus": 1}}, "no setting"), ({"backtest": {"cost_bps": -1}}, "between 0 and 250"), ({"dataset": {"universe": "mars"}}, "universe must be one of"),
                       ({"dataset": {"feature_groups": ["price", "astrology"]}}, "feature_groups"), ({"dataset": {"target": "fwd_ret_x"}}, "unknown target"), ({"dataset": {"horizon": 500}}, "between 1 and 252"),
                       ({"dataset": {"universe": ["ONLYONE"]}}, "between 2"), ({"dataset": {"universe": ["bad ticker!", "X"]}}, "not a valid ticker"), ({"model": {"kind": "ridge", "params": {"alpha; drop": 1}}}, "model.params"), ({"model": "ridge"}, "object"),
                       ({"split": {"validation_fraction": 0.4, "test_fraction": 0.4, "forward_fraction": 0.4}}, "leave at least 30%"), ({"seed": -1}, "seed"), ({"signal": "x"}, "must be an object"),
                       ({"model": {"kind": "regime_ridge"}}, "regime_detector")):
        with pytest.raises(ApiError, match=text):
            train(body)
    assert not mlapp.jobs, "nothing refused may have been queued"


def test_trees_get_their_own_refit_interval_unless_the_request_sets_one(mlapp):
    ml = mlapp.extra_get["/api/ml/catalog"].__self__
    assert ml._overrides({"model": {"kind": "random_forest"}})["split"]["refit_every"] == 63
    assert ml._overrides({"model": {"kind": "random_forest"}, "split": {"refit_every": 21}})["split"]["refit_every"] == 21
    assert "split" not in ml._overrides({"model": {"kind": "no_such_model"}}) or ml._overrides({"model": {"kind": "no_such_model"}})["split"].get("refit_every") != 63


def test_identifiers_are_checked_so_a_request_cannot_reach_outside_the_registry(mlapp):
    for bad in ("../etc/passwd", "a/b", "..", "", "x" * 200, "id with space"):
        with pytest.raises(ApiError, match="valid identifier"):
            mlapp.extra_get["/api/ml/experiment"]({"id": bad})
    with pytest.raises(ApiError, match="unknown experiment"):
        mlapp.extra_get["/api/ml/experiment"]({"id": "20250101-000000-abcdef"})
    with pytest.raises(ApiError, match="between 2 and 8"):
        mlapp.extra_get["/api/ml/compare"]({"ids": "only-one"})
    with pytest.raises(ApiError, match="confirm"):
        mlapp.extra_post["/api/ml/delete"]({"id": "20250101-000000-abcdef"})


# ------------------------------------------------------------------------------------------------------------------ machine learning: the pipeline through the dashboard
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}}


@pytest.fixture(scope="module")
def runs(mlapp):
    """Three experiments trained through the endpoint: two linear models and a boosted tree, all on the same synthetic market."""
    train = mlapp.extra_post["/api/ml/train"]
    out = {}
    for name, model in (("ridge", {"kind": "ridge", "params": {"alpha": 30}}), ("enet", {"kind": "elasticnet", "params": {"alpha": 0.01}}), ("hgb", {"kind": "hist_gb", "params": {"max_depth": 3, "max_iter": 50}})):
        out[name] = finished(mlapp, train({**BASE, "model": model, "name": f"{name}-test"}))
    return out


def test_the_dataset_builder_describes_the_panel_and_measures_features_only_on_early_dates(mlapp):
    out = finished(mlapp, mlapp.extra_post["/api/ml/dataset/build"]({"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "name": "panel"}))
    assert out["dataset"]["n_assets"] == 8 and out["dataset"]["target"] == "fwd_ret_5" and out["timing"] == {"lag": 1, "horizon": 5, "purge": 6}
    assert out["target"]["scope"] in ("warm-up segment", "first half of the dates") and out["source"] == "platform"
    assert all(f["coverage"] > 0.5 and f["group"] in ("price", "momentum") for f in out["features"]) and any(f["ic"] is not None for f in out["features"])
    last = pd.Timestamp(out["dataset"]["end"])
    assert pd.Timestamp(out["plan"].get("train", {"end": str(last.date())})["end"]) <= last
    fsets = mlapp.extra_get["/api/ml/feature-sets"]({})["feature_sets"]
    assert out["feature_set"] in [f["id"] for f in fsets]


def test_training_through_the_dashboard_returns_segments_trials_and_overfitting_diagnostics(runs):
    rep = runs["ridge"]["report"]
    stats = rep["segments"]["stats"]
    assert list(stats) == ["train", "validation", "test", "forward"] and rep["id"] == runs["ridge"]["id"]
    assert rep["trials"]["dataset_trials"] >= 1 and rep["overfit"]["warnings"] is not None and rep["ic"]["decay"] and rep["equity"]["bounds"]["forward"]
    assert rep["segments"]["plan"]["forward"]["start"] > rep["segments"]["plan"]["test"]["end"], "the reserved segments come last, in order"
    assert runs["hgb"]["report"]["trials"]["dataset_trials"] == 3 and runs["ridge"]["report"]["trials"]["dataset_trials"] == 1, "each run on the same sample is another look at it"
    assert rep["config"]["model"]["kind"] == "ridge" and rep["config"]["dataset"]["lag"] == 1


def test_the_leaderboard_ranks_experiments_and_each_experiment_can_be_opened(mlapp, runs):
    board = _safe(mlapp.extra_get["/api/ml/experiments"]({"sort": "validation.sharpe"}))
    assert {r["id"] for r in board["rows"]} == {r["id"] for r in runs.values()} and board["n"] == 3 and "trials" in board["rows"][0]
    sharpes = [r["validation.sharpe"] for r in board["rows"] if r["validation.sharpe"] is not None]
    assert sharpes == sorted(sharpes, reverse=True)
    assert [r["model"] for r in mlapp.extra_get["/api/ml/experiments"]({"model": "hist_gb"})["rows"]] == ["hist_gb"] and mlapp.extra_get["/api/ml/experiments"]({"limit": "1"})["rows"].__len__() == 1
    with pytest.raises(ApiError, match="sort"):
        mlapp.extra_get["/api/ml/experiments"]({"sort": "x; drop table experiments"})
    one = _safe(mlapp.extra_get["/api/ml/experiment"]({"id": runs["ridge"]["id"]}))
    assert one["report"]["id"] == runs["ridge"]["id"] and one["record"]["stage"] == "none" and {a["name"] for a in one["artifacts"]} >= {"predictions", "positions", "returns", "report_json", "report_html", "model"}
    html = mlapp.extra_get["/api/ml/report-html"]({"id": runs["ridge"]["id"]})["html"]
    assert html.lstrip().lower().startswith("<!doctype html") and "<script" not in html.lower()


def test_comparing_experiments_aligns_their_curves_and_maps_how_alike_they_are(mlapp, runs):
    ids = [r["id"] for r in runs.values()]
    out = _safe(mlapp.extra_get["/api/ml/compare"]({"ids": ",".join(ids)}))
    assert [r["id"] for r in out["rows"]] == ids and set(out["curves"]) == set(ids) and all(c[0] == pytest.approx(1.0, abs=0.05) for c in out["curves"].values())
    assert len(out["dates"]) == len(out["segment"]) == len(out["curves"][ids[0]]) and np.allclose(np.diag(out["correlation"]["matrix"]), 1.0) and out["common_days"] > 100
    assert set(out["rows"][0]["segments"]) == {"train", "validation", "test", "forward"} and len(out["similarity"]["names"]) == 3


def test_a_prediction_is_explained_exactly_by_shapley_values(mlapp, runs):
    out = _safe(mlapp.extra_get["/api/ml/explain"]({"id": runs["ridge"]["id"], "permutations": "20"}))
    e = out["explanation"]
    assert out["asset"] in out["assets"] and e["steps"][-1]["feature"] == "all other features"
    assert e["base"] + sum(s["contribution"] for s in e["steps"]) == pytest.approx(e["prediction"], abs=1e-4), "the contributions add up to the score"
    assert [s["asset"] for s in out["scores"]] == sorted((s["asset"] for s in out["scores"]), key=lambda a: -next(x["score"] for x in out["scores"] if x["asset"] == a))
    again = mlapp.extra_get["/api/ml/explain"]({"id": runs["ridge"]["id"], "asset": out["assets"][1], "date": "2019-06-03"})
    assert again["asset"] == out["assets"][1] and again["date"] <= "2019-06-03"
    with pytest.raises(ApiError, match="unknown asset"):
        mlapp.extra_get["/api/ml/explain"]({"id": runs["ridge"]["id"], "asset": "ZZZ"})


def test_strategies_can_be_combined_by_every_method_and_the_segments_are_reported(mlapp, runs):
    ids = [r["id"] for r in runs.values()]
    out = _safe(mlapp.extra_post["/api/ml/combine"]({"ids": ids, "methods": ["equal", "inverse_vol", "bandit", "online_learning"], "min_history": 100}))
    assert {r["method"] for r in out["rows"]} == {"equal", "inverse_vol", "bandit", "online_learning"} and not out["errors"]
    assert all(set(r["segments"]) <= {"validation", "test", "forward"} and r["segments"] for r in out["rows"]) and set(out["strategy_sharpes"]) == set(ids)
    assert all(abs(sum(r["final_weights"].values()) - 1.0) < 1e-6 for r in out["rows"]) and "validation" in out["note"]
    for bad, text in (({"ids": ids[:1]}, "between 2"), ({"ids": ids, "methods": ["magic"]}, "methods"), ({"ids": ids, "lookback": 5}, "lookback")):
        with pytest.raises(ApiError, match=text):
            mlapp.extra_post["/api/ml/combine"](bad)


def test_a_saved_run_can_be_rebacktested_and_meta_labelled_without_refitting(mlapp, runs):
    parent = runs["ridge"]["id"]
    re = finished(mlapp, mlapp.extra_post["/api/ml/rebacktest"]({"id": parent, "backtest": {"cost_bps": 25, "rebalance": "monthly"}, "signal": {"holding_period": 20}}))
    assert re["parent"] == parent and re["report"]["config"]["backtest"]["cost_bps"] == 25 and re["report"]["config"]["model"] == runs["ridge"]["report"]["config"]["model"]
    assert re["report"]["trials"]["dataset_trials"] > runs["ridge"]["report"]["trials"]["dataset_trials"], "re-backtesting is another look at the data and is counted"
    with pytest.raises(ApiError, match="only the signal, backtest and robustness"):
        mlapp.extra_post["/api/ml/rebacktest"]({"id": parent, "model": {"kind": "lasso"}})
    meta = finished(mlapp, mlapp.extra_post["/api/ml/meta-label"]({"id": parent, "kind": "logistic"}))
    assert meta["parent"] == parent and 0.0 < meta["meta"]["coverage"] <= 1.0 and meta["report"]["config"]["signal"]["method"] == "proportional" and set(meta["parent_segments"]) >= {"validation", "test"}
    with pytest.raises(ApiError, match="classifier"):
        mlapp.extra_post["/api/ml/meta-label"]({"id": parent, "kind": "kalman"})
    kids = mlapp.extra_get["/api/ml/experiment"]({"id": parent})["children"]
    assert {re["id"], meta["id"]} <= {k["id"] for k in kids}
    board = {r["id"]: r for r in mlapp.extra_get["/api/ml/experiments"]({})["rows"]}
    assert board[parent]["has_model"] is True and board[parent]["parent"] is None
    assert board[re["id"]]["has_model"] is False and board[re["id"]]["parent"] == parent, "a child re-uses its parent's predictions and has no model of its own: the page must not offer it for a forward test"


def test_a_search_counts_every_trial_and_finalises_only_its_best(mlapp):
    out = finished(mlapp, mlapp.extra_post["/api/ml/search"]({**BASE, "kinds": ["ridge", "lasso"], "n_random": 4, "n_bayes": 2, "generations": 1, "population": 4, "top_k": 1}))
    assert out["n_trials"] >= 6 and out["stages"]["random"] == 4 and out["stages"]["bayes"] == 2 and len(out["finalists"]) == 1 and "validation-only" in out["note"]
    fin = out["finalists"][0]
    assert fin["trials"] >= out["n_trials"], "the finalist's deflated Sharpe counts every trial of the search"
    trials = _safe(mlapp.extra_get["/api/ml/trials"]({"study": "search"}))
    assert len(trials["trials"]) == out["n_trials"] and max(trials["n_total"].values()) >= out["n_trials"]
    for bad, text in (({"kinds": ["magic"]}, "kinds must"), ({"n_random": 0, "n_bayes": 1}, "at least two trials"), ({"n_random": 500}, "between 0 and 30")):
        with pytest.raises(ApiError, match=text):
            mlapp.extra_post["/api/ml/search"]({**BASE, **bad})


def test_tuning_one_model_searches_its_settings_and_can_finalise_the_best(mlapp):
    out = finished(mlapp, mlapp.extra_post["/api/ml/tune"]({**BASE, "model": {"kind": "ridge"}, "n_trials": 5, "finalize": True}))
    assert out["n_trials"] == 5 and out["best"] and "model.params.alpha" in out["space"] and out["final"]["report"]["trials"]["dataset_trials"] >= 5
    with pytest.raises(ApiError, match="no hyperparameters"):
        mlapp.extra_post["/api/ml/tune"]({**BASE, "model": {"kind": "linear"}})
    with pytest.raises(ApiError, match="method must be"):
        mlapp.extra_post["/api/ml/tune"]({**BASE, "method": "grid"})


def test_a_tuning_with_an_effort_starts_from_the_forms_settings_and_says_in_words_whether_the_gain_held_up(mlapp):
    catalog = mlapp.extra_get["/api/ml/catalog"]({})
    rows = catalog["tune_spaces"]["ridge"]
    assert [r["path"] for r in rows] == ["model.params.alpha", "dataset.horizon", "signal.method", "signal.n", "signal.holding_period", "backtest.rebalance"]
    assert rows[0]["kind"] == "log" and rows[0]["low"] == 0.1 and "held back from fitting noise" in rows[0]["text"] and rows[2]["kind"] == "categorical" and "linear" not in catalog["tune_spaces"]
    assert [e["id"] for e in catalog["tune_efforts"]] == ["quick", "standard", "thorough"] and catalog["tune_efforts"][0]["trials"] == 8
    out = finished(mlapp, mlapp.extra_post["/api/ml/tune"]({**BASE, "model": {"kind": "ridge", "params": {"alpha": 30}}, "effort": "quick"}))
    assert out["n_trials"] == 8 and out["start_number"] == 0 and out["effort"] == "quick" and out["start"]["model.params.alpha"] == 30 and out["start"]["signal.n"] == 2
    assert out["trials"][0]["params"] == out["start"], "the form's own settings are the first trial"
    assert out["order"] == [r["path"] for r in rows] and out["kind"] == "ridge" and out["texts"]["model.params.alpha"].startswith("How strongly")
    summary = out["summary"]
    assert summary["level"] in ("good", "mixed", "bad", "keep") and any("validation segment" in r for r in summary["reasons"]) and any("test and forward segments" in r or "nothing tried did better" in r.lower() for r in summary["reasons"])
    assert out["final"]["report"]["config"]["name"] == "ridge tuned"
    if out["best"]["number"] == 0:
        assert out["final_start"] is None and summary["level"] == "keep"
    else:
        assert out["final_start"]["report"]["config"]["name"] == "ridge starting settings" and out["final_start"]["id"] != out["final"]["id"]
    names = [r["name"] for r in mlapp.extra_get["/api/ml/experiments"]({"limit": 50})["rows"]]
    assert "ridge tuned" in names
    with pytest.raises(ApiError, match="effort must be one of"):
        mlapp.extra_post["/api/ml/tune"]({**BASE, "model": {"kind": "ridge"}, "effort": "forever"})
    with pytest.raises(ApiError, match="not running in this app"):
        mlapp.extra_post["/api/ml/tune/stop"]({"job": "000000000000"})


def test_stop_ends_a_tuning_after_the_trial_it_is_on_and_the_best_so_far_is_still_run_on_every_segment(mlapp):
    import threading

    gate = threading.Event()
    real = opt.TrialRunner.evaluate

    def slow(self, params, stage="random"):
        if len(self.records) >= 2:
            assert gate.wait(60)
        return real(self, params, stage)

    opt.TrialRunner.evaluate = slow
    try:
        started = mlapp.extra_post["/api/ml/tune"]({**BASE, "model": {"kind": "ridge"}, "effort": "standard"})
        t0 = time.time()
        while time.time() - t0 < 60 and "trial 3" not in mlapp.job(started["job"])["message"]:
            time.sleep(0.05)
        assert mlapp.extra_post["/api/ml/tune/stop"]({"job": started["job"]}) == {"ok": True}
        gate.set()
        out = finished(mlapp, started)
    finally:
        opt.TrialRunner.evaluate = real
        gate.set()
    assert out["stopped"] == "user" and 3 <= out["n_trials"] < 16 and out["final"] is not None and out["summary"] is not None


def test_forward_testing_records_paper_signals_settles_them_and_flags_replayed_history(mlapp, runs):
    exp = runs["ridge"]["id"]
    assert mlapp.extra_get["/api/ml/forward"]({"id": exp})["summary"]["n_signals"] == 0
    stepped = finished(mlapp, mlapp.extra_post["/api/ml/forward/step"]({"id": exp, "capital": 50_000}))
    sig = stepped["signal"]
    assert sig["date"] == str(mlapp.builder.default_bundle().index[-1].date()) and sig["capital"] == 50_000 and all(o["notional"] > 0 for o in sig["orders"])
    rec = _safe(mlapp.extra_get["/api/ml/forward"]({"id": exp}))
    assert rec["summary"]["n_signals"] == 1 and rec["signals"][0]["date"] == sig["date"] and rec["backtest_ic"] is not None
    assert rec["summary"]["n_backfilled"] == 1 and any("recorded after their date" in w for w in rec["warnings"]), "a signal for a past date is a replay, not a live forecast"
    again = finished(mlapp, mlapp.extra_post["/api/ml/forward/step"]({"id": exp}))
    assert again["signal"]["signal_id"] is None and mlapp.extra_get["/api/ml/forward"]({"id": exp})["summary"]["n_signals"] == 1, "the same date is not recorded twice"
    assert "drift" in again["summary"] and "live_ic" in again["summary"]
    with pytest.raises(ApiError, match="as_of"):
        mlapp.extra_post["/api/ml/forward/step"]({"id": exp, "as_of": "yesterday-ish"})


def test_models_move_through_stages_and_an_experiment_can_be_deleted_only_when_confirmed(mlapp, runs):
    models = mlapp.extra_get["/api/ml/models"]({})["models"]
    mid = next(m["id"] for m in models if m["experiment_id"] == runs["hgb"]["id"])
    assert mlapp.extra_post["/api/ml/promote"]({"model_id": mid, "stage": "production"})["stage"] == "production"
    assert [m["id"] for m in mlapp.extra_get["/api/ml/models"]({"stage": "production"})["models"]] == [mid]
    with pytest.raises(ApiError, match="stage must be"):
        mlapp.extra_post["/api/ml/promote"]({"model_id": mid, "stage": "live"})
    with pytest.raises(ApiError, match="unknown model"):
        mlapp.extra_post["/api/ml/promote"]({"model_id": "nope", "stage": "staging"})
    victim = runs["enet"]["id"]
    with pytest.raises(ApiError, match="confirm"):
        mlapp.extra_post["/api/ml/delete"]({"id": victim})
    assert mlapp.extra_post["/api/ml/delete"]({"id": victim, "confirm": True})["ok"]
    with pytest.raises(ApiError, match="unknown experiment"):
        mlapp.extra_get["/api/ml/experiment"]({"id": victim})
    assert not (mlapp.root / "artifacts" / "ml" / "experiments" / victim).exists()


def test_a_ticker_list_is_resolved_like_the_backtest_tab_and_a_failed_download_is_a_message(mlapp):
    job = wait(mlapp, mlapp.extra_post["/api/ml/train"]({**BASE, "dataset": {**BASE["dataset"], "universe": ["BAD1", "BAD2"]}}))
    assert job["status"] == "error" and "BAD1" in job["error"] and "no data found" in job["error"]


def test_a_model_that_cannot_run_on_the_data_is_reported_as_a_message_not_a_crash(mlapp):
    job = wait(mlapp, mlapp.extra_post["/api/ml/train"]({**BASE, "split": {"min_train_days": 3000}}))
    assert job["status"] == "error" and "warm-up" in job["error"] and "min_train_days" in job["error"]
    job = wait(mlapp, mlapp.extra_post["/api/ml/train"]({**BASE, "model": {"kind": "ridge", "params": {"no_such": 1}}}))
    assert job["status"] == "error" and "no_such" in job["error"]


# ------------------------------------------------------------------------------------------------------------------ over HTTP
def test_the_workspaces_are_served_with_query_strings_and_the_token(assets):
    running = server.start(assets, port=0)
    try:
        status, body, _ = _http(running, "/api/options/chain?underlying=AAA&window=0.1")
        assert status == 200 and json.loads(body)["chain"]["rows"]
        status, body, _ = _http(running, "/api/options/chain?underlying=NOPE")
        assert status == 400 and "unknown underlying" in json.loads(body)["error"]
        assert _http(running, "/api/options/chain?underlying=AAA", token=False)[0] == 403
        assert _http(running, "/api/nowhere")[0] == 404
        many = "&".join(f"k{i}=1" for i in range(60))
        status, body, _ = _http(running, f"/api/options/chain?underlying=AAA&{many}")
        assert status == 400 and "too many" in json.loads(body)["error"]
        status, body, _ = _http(running, "/api/options/portfolio-risk", {"underlying": "AAA", "positions": [{"type": "stock", "quantity": 10}]})
        assert status == 200 and json.loads(body)["kind"] == "options_portfolio_risk"
        status, body, _ = _http(running, "/api/options/strategy/backtest", {"strategy": "short_put", "days": 60})
        job = json.loads(body)["job"]
        assert status == 200
        for _ in range(300):
            _, body, _ = _http(running, f"/api/job/{job}")
            if json.loads(body)["status"] in ("done", "error"):
                break
            time.sleep(0.2)
        assert json.loads(body)["status"] == "done" and json.loads(body)["result"]["kind"] == "options_backtest"
    finally:
        running.stop()
