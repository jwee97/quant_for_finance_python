"""Release 3 through the API: the research-control routes, the request schemas that reject what they do not take, the development mode of training, and the whole holdout lifecycle as a client sees it.

The same service functions are exercised directly in ``test_ml_holdout_controls.py``; here the claim is that the routes call them, validate their input, and never let a request supply evidence or a status."""

import warnings

import pytest

from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished, wait

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("research")
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


def get(app, path, **q):
    return _safe(app.extra_get[path](q))


def post(app, path, body):
    return app.extra_post[path](body)


def job_error(app, started):
    job = wait(app, started)
    assert job["status"] == "error", job
    return job["error"]


def test_the_research_routes_are_registered(app):
    for path in ("/api/ml/lineage", "/api/ml/holdout", "/api/ml/promotion-status", "/api/ml/statistical-diagnostics", "/api/ml/null-controls"):
        assert path in app.extra_get, path
    for path in ("/api/ml/evaluation-plan", "/api/ml/evaluation-plan/freeze", "/api/ml/evaluate", "/api/ml/null-controls/run", "/api/ml/selection-tests/run", "/api/ml/promote"):
        assert path in app.extra_post, path


def test_requests_that_carry_anything_a_route_does_not_take_are_refused(app):
    for path, body, text in (("/api/ml/evaluate", {"plan_id": "plan-x", "idempotency_key": "k", "status": "passed"}, "does not take"),
                             ("/api/ml/evaluate", {"plan_id": "plan-x"}, "idempotency_key is required"),
                             ("/api/ml/evaluate", {"plan_id": "../etc", "idempotency_key": "k"}, "not a valid identifier"),
                             ("/api/ml/evaluation-plan", {"candidates": []}, "1 to 6"),
                             ("/api/ml/evaluation-plan", {"candidates": [BASE], "freeze_it": True}, "does not take"),
                             ("/api/ml/evaluation-plan", {"candidates": [BASE], "selection_adjustment": "magic"}, "selection_adjustment must be"),
                             ("/api/ml/evaluation-plan/freeze", {"plan_id": "plan-x", "force": True}, "does not take"),
                             ("/api/ml/selection-tests/run", {"study": "u-etf", "procedures": ["magic"]}, "procedures is a non-empty list"),
                             ("/api/ml/selection-tests/run", {"study": "bad study!"}, "a study name is"),
                             ("/api/ml/null-controls/run", {"replications": 50000}, "between 2 and 1000"),
                             ("/api/ml/null-controls/run", {"kinds": ["magic"]}, "kinds must be"),
                             ("/api/ml/null-controls/run", {"gate": "easy"}, "does not take"),
                             ("/api/ml/promote", {"model_id": "m", "stage": "staging", "status": "passed"}, "does not take"),
                             ("/api/ml/promote", {"model_id": "m", "stage": "production", "approval": {"actor": "a", "extra": "b"}}, "approval is"),
                             ("/api/ml/train", {**BASE, "mode": "final"}, "mode must be full or development")):
        with pytest.raises(ApiError, match=text):
            post(app, path, body)
    for path, q, text in (("/api/ml/lineage", {}, "give id"), ("/api/ml/lineage", {"id": "x", "study": "y"}, "not both"), ("/api/ml/statistical-diagnostics", {}, "give id"),
                          ("/api/ml/promotion-status", {}, "not a valid identifier"), ("/api/ml/holdout", {"study": "no good"}, "a study name is")):
        with pytest.raises(ApiError, match=text):
            app.extra_get[path](q)


def test_development_training_stops_at_validation_and_full_training_records_the_exposure(app):
    dev = finished(app, post(app, "/api/ml/train", {**BASE, "model": {"kind": "ridge"}, "mode": "development", "explain": False, "name": "dev run"}))
    assert dev["report"]["holdout"]["status"] == "development" and set(dev["report"]["segments"]["stats"]) == {"train", "validation"} and dev["report"]["lineage"]["study"] == "u-etf"
    full = finished(app, post(app, "/api/ml/train", {**BASE, "model": {"kind": "ridge"}, "explain": False, "name": "full run"}))
    assert full["report"]["holdout"]["status"] == "exposed_exploratory" and full["report"]["holdout"]["untouched"] is False and "reproducibility" in full["report"]
    rows = {r["id"]: r for r in get(app, "/api/ml/experiments", limit=20)["rows"]}
    assert rows[dev["id"]]["holdout"] == "development_or_unrecorded" and rows[full["id"]]["holdout"] == "exposed_exploratory"
    lin = get(app, "/api/ml/lineage", id=full["id"])
    assert lin["study"]["study_id"] == "u-etf" and lin["raw_looks"] >= 2 and any(o["experiment_id"] == full["id"] and o["state"] == "done" for o in lin["operations"])
    named = finished(app, post(app, "/api/ml/train", {**BASE, "model": {"kind": "ridge"}, "mode": "development", "explain": False, "study": "my-study"}))
    assert named["report"]["lineage"]["study"] == "my-study" and get(app, "/api/ml/lineage", study="my-study")["raw_looks"] == 1
    with pytest.raises(ApiError, match="a study name is"):
        post(app, "/api/ml/train", {**BASE, "model": {"kind": "ridge"}, "study": "no good"})


def test_the_dataset_page_registers_its_feature_screen_as_adaptive_discovery(app):
    out = finished(app, post(app, "/api/ml/dataset/build", {"name": "screened", "dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504}}))
    screen = out["feature_screen"]
    assert screen["status"] == "applied" and screen["method"] == "holm" and screen["n_tested"] >= 10 and 0 <= screen["n_selected"] <= screen["n_tested"] and screen["dependence"]["between_features"]
    lin = get(app, "/api/ml/lineage", study=screen["study"])
    assert any(o["kind"] == "feature_search" and o["op_key"] == screen["operation"] for o in lin["operations"]) and any(o["kind"] == "feature_choice" for o in lin["operations"])
    ev = get(app, "/api/ml/statistical-diagnostics", study=screen["study"])
    assert any(e["method"] == "feature_screen" and e["status"] == "applied" for e in ev["evidence"]) and any(r["subject"] for r in ev["evidence"] if r["method"] == "feature_screen")


def test_the_whole_holdout_lifecycle_as_a_client_sees_it(app):
    cands = [{**BASE, "model": {"kind": "ridge", "params": {"alpha": 5.0}}, "name": "c1"}, {**BASE, "model": {"kind": "ridge", "params": {"alpha": 50.0}}, "name": "c2"}]
    created = finished(app, post(app, "/api/ml/evaluation-plan", {"study": "lifecycle", "candidates": cands, "selection_adjustment": "holm", "metrics": ["sharpe", "ic"]}))
    plan = created["plan_id"]
    assert created["state"] == "reserved" and len(created["content"]["candidates"]) == 2 and created["reservation"]["untouched"] is False, "the history was read by the exploratory runs above, and the reservation says so"
    assert any(o["kind"] == "exposure" for o in created["reservation"]["overlaps"])
    with pytest.raises(ApiError, match="not a valid identifier"):
        post(app, "/api/ml/evaluate", {"plan_id": "bad id", "idempotency_key": "k"})
    err = job_error(app, post(app, "/api/ml/evaluate", {"plan_id": plan, "idempotency_key": "early"}))
    assert "reserved, not frozen" in err, "an unfrozen plan cannot be evaluated"
    err = job_error(app, post(app, "/api/ml/train", {**BASE, "model": {"kind": "ridge"}, "explain": False, "name": "blocked full run"}))
    assert "protects" in err and "development" in err, "a direct run on a reserved history is refused with the way forward"
    dev = finished(app, post(app, "/api/ml/train", {**BASE, "model": {"kind": "ridge"}, "mode": "development", "explain": False}))
    assert dev["report"]["holdout"]["status"] == "development"
    frozen = post(app, "/api/ml/evaluation-plan/freeze", {"plan_id": plan})
    assert frozen["state"] == "frozen"
    err = job_error(app, post(app, "/api/ml/evaluate", {"plan_id": plan, "idempotency_key": "first"}))
    assert "overlap it" in err and "allow_reuse" in err, "earlier exposures of the same history make an evaluation a re-use, which must be asked for"
    done = finished(app, post(app, "/api/ml/evaluate", {"plan_id": plan, "idempotency_key": "first", "allow_reuse": True}))
    assert done["status"] == "evaluated" and done["reuse"] is True and done["untouched"] is False and len(done["experiments"]) == 2
    assert all(e["holdout"]["status"] == "evaluated_under_plan" and e["holdout"]["untouched"] is False for e in done["experiments"])
    again = finished(app, post(app, "/api/ml/evaluate", {"plan_id": plan, "idempotency_key": "first", "allow_reuse": True}))
    assert again["status"] == "replayed" and [e["experiment_id"] for e in again["result"]["experiments"]] == [e["experiment_id"] for e in done["experiments"]]
    ho = get(app, "/api/ml/holdout", id=plan)
    assert ho["plans"][0]["state"] == "consumed" and ho["plans"][0]["evaluations"] == 1 and [a["access_kind"] for a in ho["accesses"] if a["status"] == "claimed"] == ["evaluate_reuse"]
    lin = get(app, "/api/ml/lineage", study="lifecycle")
    assert lin["raw_looks"] == 0 and {o["kind"] for o in lin["operations"]} == {"evaluation"} and lin["plans"][0]["plan_id"] == plan


def test_promotion_status_is_computed_on_the_server_and_a_request_cannot_set_it(app):
    exp = next(r["id"] for r in get(app, "/api/ml/experiments", limit=20)["rows"] if r["holdout"] == "evaluated_under_plan")
    status = get(app, "/api/ml/promotion-status", id=exp)
    assert set(status["gates"]) == {"A", "B", "C"} and status["policy_version"].startswith("gates-")
    assert status["gates"]["B"]["status"] in ("blocked", "pending") and status["gates"]["C"]["status"] == "blocked" and status["next"]["C"]
    assert any(i["id"] == "realistic_net_oos" and i["status"] == "blocked" for i in status["gates"]["B"]["items"])
    assert status["models"] and "history" in status
    mid = status["models"][0]["id"]
    with pytest.raises(ApiError, match="cannot move to production"):
        post(app, "/api/ml/promote", {"model_id": mid, "stage": "production", "approval": {"actor": "me", "note": "x"}})
    out = post(app, "/api/ml/promote", {"model_id": mid, "stage": "archived"})
    assert out["ok"] and out["stage"] == "archived" and get(app, "/api/ml/promotion-status", id=exp)["models"][0]["history"][-1]["to_stage"] == "archived"
    with pytest.raises(ApiError, match="unknown experiment"):
        app.extra_get["/api/ml/promotion-status"]({"id": "nope"})


def test_selection_tests_run_on_a_studys_recorded_trials_and_report_what_did_not_apply(app):
    tuned = finished(app, post(app, "/api/ml/tune", {**BASE, "model": {"kind": "ridge"}, "n_trials": 5, "study": "tuned-study"}))
    assert tuned["final"] is None and tuned["evaluation"] is None and tuned["finalize"]["requested"] is False
    lin = get(app, "/api/ml/lineage", study="tuned-study")
    assert lin["by_kind"].get("hpo_trial", 0) >= 5, "every tuning trial is in the study, registered before it ran"
    out = finished(app, post(app, "/api/ml/selection-tests/run", {"study": "tuned-study", "n_boot": 100}))
    methods = {e["method"]: e for e in out["evidence"]}
    assert set(methods) == {"dsr", "pbo", "reality_check", "spa", "fdr"} and out["n_candidates"] >= 5
    assert all(e["status"] in ("applied", "not_applicable", "unsupported", "insufficient", "failed") for e in methods.values()) and "procedures were applied" in out["statement"]
    assert methods["dsr"]["status"] == "applied" and methods["dsr"]["assumptions"] and methods["dsr"]["scope"]["tries_recorded"] >= 5 and methods["dsr"]["limitations"]
    diag = get(app, "/api/ml/statistical-diagnostics", study="tuned-study")
    assert {e["method"] for e in diag["latest"]} == set(methods) and diag["raw_looks"] >= 5 and "A procedure with status other than applied" in " ".join(diag["notes"])
    none = get(app, "/api/ml/statistical-diagnostics", study="empty-study")
    assert none["latest"] == [] and "no selection test has been run" in none["statement"]
    trials = get(app, "/api/ml/trials", study="tune-ridge")
    assert trials["trials"] and "lineage" in trials


def test_a_null_study_is_frozen_run_and_labelled_a_smoke_run_when_it_is_small(app):
    started = post(app, "/api/ml/null-controls/run", {**BASE, "model": {"kind": "ridge"}, "kinds": ["ridge"], "n_random": 1, "replications": 2, "min_replications": 100, "min_shift": 150, "study": "null-api"})
    out = finished(app, started)
    assert out["replications"] == 2 and out["sufficient"] is False and out["controlled"] is None and "plumbing only" in out["verdict"] and out["spec_hash"]
    listing = get(app, "/api/ml/null-controls")
    assert listing["studies"][0]["null_id"] == out["null_id"] and listing["studies"][0]["result"]["replications"] == 2
    one = get(app, "/api/ml/null-controls", id=out["null_id"])
    assert one["spec"]["generator"] == "circular_shift_returns" and len(one["replication_records"]) == 2 and one["state"] == "done"
    assert all(r["seed"] == 0 + i * 7919 for i, r in enumerate(one["replication_records"]))


def test_a_legacy_experiment_has_no_lineage_and_says_so(app):
    import json

    reg = app.extra_get["/api/ml/lineage"].__self__.registry                                   # the MlApi instance behind the route
    with reg._conn() as c:
        c.execute("INSERT INTO experiments (id, name, created, status, config_hash, dataset_key, universe, model_kind, trials_at_run, stage) VALUES ('old-run', 'old', 'x', 'done', 'h', 'd', 'etf', 'ridge', 0, 'none')")
    (reg.root / "experiments" / "old-run").mkdir(parents=True, exist_ok=True)
    (reg.root / "experiments" / "old-run" / "report.json").write_text(json.dumps({"id": "old-run"}))
    lin = get(app, "/api/ml/lineage", id="old-run")
    assert lin["operations"] == [] and "before study lineage was recorded" in lin["note"] and lin["study"]["study_id"] is None
    diag = get(app, "/api/ml/statistical-diagnostics", id="old-run")
    assert diag["study"] is None and "no selection test can be attributed" in diag["statement"]
    status = get(app, "/api/ml/promotion-status", id="old-run")
    assert status["gates"]["A"]["status"] == "pending" and any(i["id"] == "provenance" and i["status"] == "missing" for i in status["gates"]["A"]["items"])
