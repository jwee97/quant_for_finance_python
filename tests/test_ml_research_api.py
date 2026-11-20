"""Research Evidence, the feature registry and study, model comparison and the scheduled simulation through the request handlers, on the synthetic ML dataset."""

from __future__ import annotations

import json
import warnings

import pytest

from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp import _wait
from tests.test_webapp_assets_ml import _app, _safe

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504, "refit_every": 126}, "model": {"kind": "ridge"}, "explain": False}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("research")
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


def get(app, path, **q):
    return _safe(app.extra_get[path](q))


def done(app, route, body, timeout=290):
    view = _wait(app, app.extra_post[route](body)["job"], timeout)
    assert view["status"] == "done", view
    json.dumps(view["result"], allow_nan=False)
    return view["result"]


def test_the_routes_exist_and_bad_requests_are_refused_before_a_job_is_queued(app):
    assert {"/api/ml/evidence", "/api/ml/features"} <= set(app.extra_get) and {"/api/ml/features/study", "/api/ml/compare-strategies", "/api/ml/simulate"} <= set(app.extra_post)
    for route, body, text in (("/api/ml/compare-strategies", {"models": []}, "1 to 6"), ("/api/ml/compare-strategies", {"models": ["ridge"]}, "object"), ("/api/ml/compare-strategies", {"models": [{"kind": "nope"}]}, "model.kind"),
                              ("/api/ml/simulate", {"grid": "huge"}, "grid must be"), ("/api/ml/simulate", {"grid": [{"id": "x"}]}, "grid is"), ("/api/ml/features/study", {"ablate_groups": ["nope"]}, "ablate_groups")):
        with pytest.raises(ApiError, match=text):
            app.extra_post[route](body)
    with pytest.raises(ApiError, match="group must be"):
        app.extra_get["/api/ml/features"]({"group": "nope"})
    with pytest.raises(ApiError, match="not a valid identifier"):
        app.extra_get["/api/ml/evidence"]({"id": "../x"})


def test_the_registry_lists_every_feature_with_its_hypothesis_and_reference(app):
    out = get(app, "/api/ml/features")
    assert len(out["features"]) > 50 and {g["group"] for g in out["groups"]} >= {"price", "momentum", "volatility"}
    f = out["features"][0]
    assert f["source"] and f["availability"] and f["hypothesis"] and f["reference"]
    assert {r["group"] for r in get(app, "/api/ml/features", group="momentum")["features"]} == {"momentum"}


def test_the_feature_study_is_a_job_that_measures_each_feature_and_can_ablate_groups(app):
    r = done(app, "/api/ml/features/study", {**BASE, "ablation": True, "ablate_groups": ["momentum"]})
    scored = [x for x in r["features"] if "mean_ic" in x]
    assert scored and r["dataset"]["n_features"] == len(r["features"]) and all(x["feature"] in r["registry"] for x in r["features"])
    names = [x["variant"] for x in r["ablation"]["rows"]]
    assert names[0] == "all features" and "without momentum" in names and r["ablation"]["noise"]["variant"] == "noise features only"


def test_the_comparison_is_a_job_that_trades_models_and_baselines_identically(app):
    r = done(app, "/api/ml/compare-strategies", {**BASE, "models": [{"kind": "ridge"}, {"kind": "lasso"}], "draws": 3, "signal": {"method": "long_only_top_quantile", "quantile": 0.3}, "backtest": {"rebalance": "monthly"}})
    assert [x["name"] for x in r["rows"]][:2] == ["ridge", "lasso"] and len({x["n"] for x in r["rows"]}) == 1 and r["design"]["position_rule"] == "long_only_top_quantile"


@pytest.fixture(scope="module")
def simulated(app):
    return done(app, "/api/ml/simulate", {**BASE, "grid": "narrow", "lookback_days": 126, "signal": {"decision_max_positions": 3, "decision_rebalance_days": 5}})


def test_the_simulation_saves_an_experiment_with_its_versions_choices_and_schedule(app, simulated):
    s = simulated["simulation"]
    assert len(s["model_versions"]) >= 3 and s["oos_violations"] == [] and len(s["choices"]) == len(s["model_versions"]) and s["selection"]["n_candidates"] == 2
    rep = _safe(app.extra_get["/api/ml/experiment"]({"id": simulated["id"]}))["report"]
    assert rep["simulation"]["comparison"]["scheduled"]["days"] > 100 and rep["config"]["signal"]["decision_schedule"]


def test_the_strategy_explorer_reads_a_simulated_run_like_any_other(app, simulated):
    out = get(app, "/api/ml/strategy", id=simulated["id"])
    assert out["summary"]["analytics"]["n_round_trips"] > 0 and out["chart"]["markers"]
    assert get(app, "/api/ml/strategy/decisions", id=simulated["id"], limit=5)["n"] > 0


def test_the_evidence_checklist_reads_the_record_applies_stated_rules_and_never_invents_a_pass(app, simulated):
    ev = get(app, "/api/ml/evidence", id=simulated["id"])
    by = {c["id"]: c for c in ev["checks"]}
    assert {"features_causal", "oos_lineage", "holdout", "multiple_testing", "out_of_sample", "execution", "random_signals", "simulation", "reproducible", "warnings"} <= set(by)
    assert all(c["rule"] and c["source"] and c["status"] in ("pass", "warn", "fail", "unknown") for c in ev["checks"])
    assert by["simulation"]["status"] == "pass" and by["random_signals"]["status"] == "unknown", "the control has not been run, so that line is unknown, not a pass"
    assert by["execution"]["status"] == "warn" and by["robustness"]["status"] == "unknown"
    assert sum(ev["counts"].values()) == len(ev["checks"]) and "no combined score" in ev["reading"]
    done(app, "/api/ml/strategy/controls", {"id": simulated["id"], "draws": 6})
    by2 = {c["id"]: c for c in get(app, "/api/ml/evidence", id=simulated["id"])["checks"]}
    assert by2["random_signals"]["status"] in ("pass", "warn", "fail") and "random-signal draws" in by2["random_signals"]["detail"]
