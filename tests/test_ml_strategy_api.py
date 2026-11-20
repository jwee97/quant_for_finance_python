"""The ML Strategy Explorer through the API: a decision_policy run saves its decisions, executions, round trips and closes; the routes read them back, filter them, and replay a decision date.

The numbers are checked against what the run itself reports, not against fixed values: the explorer must show exactly what the experiment recorded."""

import warnings

import pytest

from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504}, "model": {"kind": "ridge"}, "explain": False,
        "signal": {"method": "decision_policy", "decision_entry_quantile": 0.25, "decision_exit_quantile": 0.5, "decision_max_positions": 3, "decision_max_holding": 20, "decision_stop_loss_vol": 2.0,
                   "decision_allow_short": True, "decision_short_exposure": 0.5, "decision_long_exposure": 0.5},
        "backtest": {"rebalance": "on_change", "target_vol": None}}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("strategy")
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def run(app):
    return finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "policy", "mode": "development"}))


def get(app, path, **q):
    return _safe(app.extra_get[path](q))


def test_the_routes_exist_and_the_policy_settings_are_validated(app):
    for r in ("/api/ml/strategy", "/api/ml/strategy/decisions", "/api/ml/strategy/trades", "/api/ml/strategy/replay"):
        assert r in app.extra_get
    for body, text in (({"signal": {"decision_entry_quantile": 0.9}}, "decision_entry_quantile"), ({"signal": {"decision_sizing": "kelly"}}, "decision_sizing must be one of"),
                       ({"signal": {"decision_allow_short": "yes"}}, "must be true or false"), ({"signal": {"decision_stop_loss_vol": -1}}, "decision_stop_loss_vol"),
                       ({"signal": {"decision_made_up": 1}}, "has no setting 'decision_made_up'"), ({"backtest": {"rebalance": "hourly"}}, "rebalance must be one of")):
        with pytest.raises(ApiError, match=text):
            app.extra_post["/api/ml/train"]({**BASE, **body, "signal": {**BASE["signal"], **body.get("signal", {})}, "backtest": {**BASE["backtest"], **body.get("backtest", {})}})
    with pytest.raises(ApiError, match="not a valid identifier"):
        app.extra_get["/api/ml/strategy"]({"id": "../x"})


def test_a_run_without_the_policy_has_no_decisions_and_says_so(app):
    plain = finished(app, app.extra_post["/api/ml/train"]({"dataset": BASE["dataset"], "split": BASE["split"], "model": BASE["model"], "explain": False, "name": "plain", "mode": "development"}))
    with pytest.raises(ApiError, match="did not use the decision_policy"):
        app.extra_get["/api/ml/strategy"]({"id": plain["id"]})


def test_the_overview_carries_the_policy_the_analytics_and_a_chart_with_markers(app, run):
    out = get(app, "/api/ml/strategy", id=run["id"])
    s = out["summary"]
    assert s["policy"]["entry_quantile"] == 0.25 and s["policy"]["allow_short"] is True and s["rebalance"] == "on_change"
    assert s["analytics"]["n_round_trips"] > 0 and 0 <= s["analytics"]["win_rate"] <= 1 and s["n_executions"] > 0
    ch = out["chart"]
    assert len(ch["dates"]) == len(ch["close"]) == len(ch["weight"]) and ch["markers"]
    assert all(m["price"] is not None and m["date"] in set(ch["dates"]) and m["reason"] for m in ch["markers"]), "every marker sits on a close and says why"
    assert {"BUY", "SELL"} >= {m["side"] for m in ch["markers"]}
    b = out["book"]
    assert len(b["dates"]) == len(b["net"]) == len(b["benchmark"]) == len(b["drawdown"]) and min(b["drawdown"]) <= 0
    assert set(out["layers"]) == {"prediction", "decision", "execution"}
    other = [a["asset"] for a in out["assets"] if a["asset"] != ch["asset"]][0]
    assert get(app, "/api/ml/strategy", id=run["id"], asset=other)["chart"]["asset"] == other
    with pytest.raises(ApiError, match="asset must be one of"):
        app.extra_get["/api/ml/strategy"]({"id": run["id"], "asset": "NOPE"})


def test_the_decision_log_filters_by_action_asset_text_and_date_and_pages(app, run):
    allrows = get(app, "/api/ml/strategy/decisions", id=run["id"], limit=500)
    assert allrows["n"] > 0 and "HOLD" not in allrows["counts"], "holds are listed only on request"
    first = allrows["rows"][0]
    assert first["reason"] and first["rank_pct"] is not None and first["price"] is not None and first["action"] in {"BUY", "SHORT"}
    buys = get(app, "/api/ml/strategy/decisions", id=run["id"], action="buy", limit=500)
    assert buys["n"] == allrows["counts"].get("BUY", 0) and {r["action"] for r in buys["rows"]} == {"BUY"}
    one = get(app, "/api/ml/strategy/decisions", id=run["id"], asset=first["asset"], limit=500)
    assert one["n"] > 0 and {r["asset"] for r in one["rows"]} == {first["asset"]}
    stops = get(app, "/api/ml/strategy/decisions", id=run["id"], q="exit line", limit=500)
    assert all("exit line" in r["reason"] or "cover line" in r["reason"] or "exit line" in r["asset"].lower() for r in stops["rows"])
    late = get(app, "/api/ml/strategy/decisions", id=run["id"], **{"from": allrows["rows"][len(allrows["rows"]) // 2]["date"]}, limit=500)
    assert 0 < late["n"] < allrows["n"] and min(r["date"] for r in late["rows"]) >= allrows["rows"][len(allrows["rows"]) // 2]["date"]
    page = get(app, "/api/ml/strategy/decisions", id=run["id"], limit=5, offset=5)
    assert [r["date"] for r in page["rows"]] == [r["date"] for r in allrows["rows"][5:10]]
    with_hold = get(app, "/api/ml/strategy/decisions", id=run["id"], hold="1", limit=500)
    assert with_hold["n"] >= allrows["n"] and "HOLD" in with_hold["counts"]
    with pytest.raises(ApiError, match="does not take"):
        app.extra_get["/api/ml/strategy/decisions"]({"id": run["id"], "sql": "1"})


def test_trades_and_round_trips_agree_with_the_runs_own_report(app, run):
    trips = get(app, "/api/ml/strategy/trades", id=run["id"], kind="round_trips", limit=500)
    ex = get(app, "/api/ml/strategy/trades", id=run["id"], kind="executions", limit=500)
    s = run["report"]["trading"]
    assert ex["n"] == s["n_executions"] and trips["n"] == s["analytics"]["n_round_trips"] + s["analytics"]["n_open"]
    closed = [r for r in trips["rows"] if r["status"] == "closed"]
    assert closed and all(r["holding_days"] >= 1 and r["open_date"] < r["close_date"] and r["close_reason"] for r in closed)
    assert all(r["reason"] for r in ex["rows"]) and all(r["decision_date"] < r["date"] for r in ex["rows"] if r["decision_date"])
    with pytest.raises(ApiError, match="kind must be"):
        app.extra_get["/api/ml/strategy/trades"]({"id": run["id"], "kind": "fills"})


def test_replay_shows_only_what_was_known_on_the_date_and_what_the_policy_did(app, run):
    dec = get(app, "/api/ml/strategy/decisions", id=run["id"], action="BUY", limit=1)["rows"][0]
    rp = get(app, "/api/ml/strategy/replay", id=run["id"], date=dec["date"])
    assert rp["date"] == dec["date"] and rp["lag_days"] >= 1
    row = [a for a in rp["assets"] if a["asset"] == dec["asset"]][0]
    assert row["decision"] and "BUY" in row["decision"] and row["reason"] == dec["reason"] and row["rank_pct"] >= 0.75
    assert row["book_after"] == pytest.approx(dec["target_weight"], abs=1e-9)
    assert all(e["date"] > rp["date"] for e in rp["executed_later"]) and rp["executed_later"], "the executions that followed are later than the decision"
    assert "later and are not known" in rp["known"]
    ranks = [a["score"] for a in rp["assets"] if a["score"] is not None]
    assert ranks == sorted(ranks, reverse=True)


def test_the_random_signal_control_runs_as_a_job_and_places_the_model_among_random_scores(app, run):
    from tests.test_webapp import _wait

    with pytest.raises(ApiError, match="draws"):
        app.extra_post["/api/ml/strategy/controls"]({"id": run["id"], "draws": 1})
    with pytest.raises(ApiError, match="does not take"):
        app.extra_post["/api/ml/strategy/controls"]({"id": run["id"], "shell": "x"})
    view = _wait(app, app.extra_post["/api/ml/strategy/controls"]({"id": run["id"], "draws": 8})["job"], 280)
    assert view["status"] == "done", view
    r = view["result"]
    assert r["draws"] == 8 and 0.0 <= r["share_of_random_at_least_as_good"] <= 1.0 and r["random"]["p05"] <= r["random"]["p50"] <= r["random"]["p95"]
    assert r["rank_persistence"] > 0 and r["matched_benchmark"]["days"] > 100 and "reading" in r
    chk = r["replay_check"]
    assert chk["days"] > 100 and chk["reproduces"] and abs(chk["saved_sharpe"] - chk["replayed_sharpe"]) < 1e-9, "the replay reproduces the saved run exactly before anything is compared with it"
