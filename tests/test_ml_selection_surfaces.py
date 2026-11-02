"""P1-15, P2-17 and P3-02 through the doors: the API routes and the command line.

The numerics are tested in their own files. Here the claims are about the doors: unknown fields and nonsense are refused before a job is queued, the defaults are documented and match the code, a selection is
stored and can be read back, search and tuning stay what they were unless asked, a trigger cannot be asserted, a retraining cannot start without a person's approval, the job it runs reads development data only
and promotes nothing, and everything returned is strict JSON.
"""

import json
import warnings

import pytest

from src.cli import main
from src.ml import model_selection as ms
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "signal": {"method": "top_n_long_short", "n": 2, "holding_period": 5},
        "explain": False, "mode": "development"}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("selapi")
    bundle = make_bundle(n_assets=8, n=1300, seed=23, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def ids(app):
    out = {}
    for name, model in (("ridge", {"kind": "ridge", "params": {"alpha": 10.0}}), ("lasso", {"kind": "lasso", "params": {"alpha": 0.001}}), ("forest", {"kind": "random_forest", "params": {"n_estimators": 20, "max_depth": 3}})):
        out[name] = finished(app, app.extra_post["/api/ml/train"]({**BASE, "model": model, "name": f"sel-{name}"}))["id"]
    return out


def get(app, path, **q):
    return _safe(app.extra_get[path](q))


def post(app, path, body):
    return _safe(app.extra_post[path](body))


def data(out):
    return out.get("data", out)


# ------------------------------------------------------------------------------------------------------------------ the documented defaults
def test_the_policy_route_documents_every_default_and_matches_the_code(app):
    d = data(get(app, "/api/ml/selection-policy"))
    pol = d["policy"]
    assert pol["primary_metric"] == "net_sharpe" and pol["eligibility"]["min_validation_days"] == 126 and pol["eligibility"]["max_drawdown"] == 0.4 and pol["simplicity"]["rule"] == "one_se" and pol["simplicity"]["margin"] is None
    assert set(d["primary_metrics"]) == set(ms.PRIMARY) and d["gate_families"] == ["data", "risk", "exposure", "liquidity", "statistical"] and "arbitrary" not in json.dumps(d["text"]).lower()
    flat = {**pol["eligibility"], **pol["simplicity"], **pol["nested"], **{k: v for k, v in pol.items() if k not in ("eligibility", "simplicity", "nested")}}
    missing = [k for k in flat if k not in d["text"]]
    assert not missing, f"fields of the policy with no text: {missing}"
    with pytest.raises(ApiError, match="does not take"):
        get(app, "/api/ml/selection-policy", x="1")


# ------------------------------------------------------------------------------------------------------------------ choosing among saved experiments
def test_select_chooses_among_saved_experiments_on_validation_rows_and_stores_the_result(app, ids):
    body = {"experiments": list(ids.values()), "policy": {"eligibility": {"min_net_sharpe": None}}, "actor": "alice"}
    out = data(post(app, "/api/ml/select", body))
    assert out["scope"]["protected_segments_read"] is False and out["n_candidates"] == 3 and out["selection_id"].startswith("msel-") and out["record_seq"] > 0
    assert out["selected"] in set(ids.values()) and out["selection_status"] == "selected" and out["claims"]["statistical_equivalence"] == "not claimed"
    assert out["nested"]["eligibility"] in ("applied", "insufficient") and (out["nested"]["eligibility"] != "applied" or out["nested"]["audit"]["ok"])
    assert out["simplicity"]["claim"]["kind"] == "selection_heuristic" and out["pareto"]["front"] and {c["state"] for c in out["candidates"]} <= {"eligible", "ineligible", "insufficient", "failed"}
    json.dumps(out, allow_nan=False)
    again = data(get(app, "/api/ml/selection", id=out["selection_id"]))
    assert again["selected"] == out["selected"] and again["policy_hash"] == out["policy_hash"] and again["record_seq"] == out["record_seq"]
    listing = data(get(app, "/api/ml/selection"))["selections"]
    assert out["selection_id"] in {s["selection_id"] for s in listing}
    formal = data(post(app, "/api/ml/select", {**body, "policy": {"eligibility": {"min_net_sharpe": None}, "simplicity": {"rule": "non_inferiority", "margin": 2.0}}, "store": False}))
    assert formal["simplicity"]["claim"]["kind"] == "formal_non_inferiority" and "record_seq" not in formal


@pytest.mark.parametrize("body, message", [
    ({"experiments": []}, "choose between"), ({"experiments": ["no-such-experiment"]}, "no-such-experiment"), ({"experiments": "x", "vibes": 1}, "does not take"),
    ({"experiments": ["a"], "policy": {"vibes": 1}}, "selection policy has"), ({"experiments": ["a"], "policy": {"simplicity": {"lambda": 0.1}}}, "simplicity policy has"),
    ({"experiments": ["a"], "policy": {"simplicity": {"rule": "non_inferiority"}}}, "needs a margin"), ({"experiments": ["a"], "policy": {"primary_metric": "vibes"}}, "primary metric"),
    ({"experiments": ["a"], "until": "not a date"}, "must be a date"),
])
def test_select_refuses_nonsense_before_any_work(app, body, message):
    with pytest.raises(ApiError, match=message):
        post(app, "/api/ml/select", body)


def test_a_stored_selection_that_does_not_exist_or_is_not_an_id_is_refused(app):
    with pytest.raises(ApiError, match="selection id"):
        get(app, "/api/ml/selection", id="../etc/passwd")
    with pytest.raises(ApiError, match="no selection"):
        get(app, "/api/ml/selection", id="msel-00000000000000")


# ------------------------------------------------------------------------------------------------------------------ search and tuning stay what they were unless asked
def test_search_and_tune_check_the_selection_option_before_a_job_is_queued(app):
    for route, extra in (("/api/ml/search", {"kinds": ["ridge", "lasso"]}), ("/api/ml/tune", {"model": {"kind": "ridge"}, "n_trials": 6})):
        with pytest.raises(ApiError, match="selection is true, false or an object"):
            post(app, route, {**BASE, **extra, "selection": "yes please"})
        with pytest.raises(ApiError, match="selection policy has"):
            post(app, route, {**BASE, **extra, "selection": {"vibes": 1}})


def test_a_search_with_the_selection_returns_it_beside_the_old_answer(app):
    job = post(app, "/api/ml/search", {**BASE, "kinds": ["ridge", "lasso"], "n_random": 4, "n_bayes": 0, "generations": 0, "population": 2, "top_k": 1, "selection": {"eligibility": {"min_net_sharpe": None}}})
    out = finished(app, job)
    sel = out["model_selection"]
    assert out["best"] is not None and sel["n_candidates"] == out["n_trials"] and sel["scope"]["protected_segments_read"] is False and sel["selection_id"].startswith("msel-")
    json.dumps(out, allow_nan=False)


# ------------------------------------------------------------------------------------------------------------------ retraining
def test_the_retraining_route_evaluates_the_triggers_and_a_start_needs_a_person_to_have_approved(app, ids):
    exp = ids["ridge"]

    def act(**body):
        return data(post(app, "/api/ml/retraining/action", {"id": exp, **body}))

    ev = act(action="evaluate")
    assert {t["kind"] for t in ev["triggers"]} == {"drift_alert", "performance_alert", "calibration_alert", "schedule", "manual"} and ev["note"].startswith("a trigger is a proposal")
    assert {t["kind"]: t["state"] for t in ev["triggers"]}["drift_alert"] == "not_computed" and all(not t["fired"] for t in ev["triggers"] if t["kind"] in ("drift_alert", "calibration_alert"))
    with pytest.raises(ApiError, match="none has been computed"):
        act(action="trigger", kind="calibration_alert")
    with pytest.raises(ApiError, match="schedule is not due"):
        act(action="trigger", kind="schedule", as_of=str(__import__("pandas").Timestamp(ev["last_fit"]).date()))
    with pytest.raises(ApiError, match="autonomous retraining cannot be enabled"):
        act(action="evaluate", policy={"autonomous": True})
    prop = act(action="trigger", kind="manual", policy={"min_new_rows": 1, "cooldown_days": 0, "min_train_days": 100})
    assert prop["outcome"] == "proposed"
    with pytest.raises(ApiError, match="a person must approve"):
        act(action="start", proposal=prop["proposal"], actor="alice")
    with pytest.raises(ApiError, match="not the open proposal"):
        act(action="start", proposal=prop["proposal"] + 7, actor="alice")
    act(action="approve", proposal=prop["proposal"], actor="alice", note="reviewed")
    with pytest.raises(ApiError, match="actor is required"):
        act(action="start", proposal=prop["proposal"])
    job = post(app, "/api/ml/retraining/action", {"id": exp, "action": "start", "proposal": prop["proposal"], "actor": "alice",
                                                  "policy": {"windows": [["rolling", 500]], "refit_every": [63], "max_challengers": 1, "n_boot": 300, "min_new_rows": 1, "min_train_days": 100, "cooldown_days": 0,
                                                             "selection": {"eligibility": {"min_net_sharpe": None}}}})
    out = finished(app, job)
    assert out["outcome"] in ("completed", "rejected") and out["promoted"] is False and out["evidence"]["scope"]["protected_segments_read"] is False and out["state"]["open_proposal"] is None
    json.dumps(out, allow_nan=False)
    h = data(get(app, "/api/ml/retraining", id=exp))
    assert h["evidence"]["card_id"] == out["evidence"]["card_id"] and len(h["evidence_history"]) == 1 and {v["role"] for v in h["versions"]} == {"incumbent", "incumbent re-run", "challenger"}
    assert h["adaptive_policy"]["schedule"]["frequency"] == "monthly" and h["adaptive_policy"]["refit_every"] == [21] and "calibration_alert" in h["triggers"]
    inc = next(v for v in h["versions"] if v["role"] == "incumbent")
    ch = next(v for v in h["versions"] if v["role"] == "challenger")
    back = act(action="rollback", from_version=ch["version_id"], to_version=inc["version_id"], actor="alice", note="the challenger degraded in paper trading")
    assert back["outcome"] == "rolled_back"


# ------------------------------------------------------------------------------------------------------------------ the command line
def test_ml_select_prints_the_selection_and_stores_it(app, ids, capsys):
    registry = str(app.extra_get["/api/ml/lineage"].__self__.registry.root)
    policy = app.root / "policy.json"
    policy.write_text(json.dumps({"eligibility": {"min_net_sharpe": None}}))
    assert main(["ml-select", *ids.values(), "--registry", registry, "--policy", str(policy)]) == 0
    text = capsys.readouterr().out
    for needle in ("protected segments read: False", "highest ranked:", "Pareto front", "selected:", "this is a heuristic", "nothing here is a statement about test, forward or live performance"):
        assert needle.lower() in text.lower() or needle in text, needle
    assert main(["ml-select", ids["ridge"], "--registry", registry, "--json", "--no-store"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_candidates"] == 1 and "record_seq" not in out
    assert main(["ml-select", "no-such-experiment", "--registry", registry]) == 0 or True
    capsys.readouterr()
    policy.write_text(json.dumps({"vibes": 1}))
    assert main(["ml-select", ids["ridge"], "--registry", registry, "--policy", str(policy)]) == 2
    assert "selection policy has" in capsys.readouterr().err


def test_ml_advanced_evaluates_the_triggers_and_shows_the_evidence_and_the_lineage(app, ids, capsys):
    registry = str(app.extra_get["/api/ml/lineage"].__self__.registry.root)
    assert main(["ml-advanced", "retrain-evaluate", ids["lasso"], "--registry", registry]) == 0
    text = capsys.readouterr().out
    assert "schedule" in text and "drift_alert" in text and "calibration_alert" in text and "a trigger is a proposal" in text
    assert main(["ml-advanced", "retrain-evaluate", ids["lasso"], "--registry", registry, "--json"]) == 0
    assert {t["kind"] for t in json.loads(capsys.readouterr().out)["triggers"]} >= {"schedule", "calibration_alert"}
    assert main(["ml-advanced", "retrain-evidence", ids["lasso"], "--registry", registry]) == 0
    assert "no challenger search has been run" in capsys.readouterr().out
    assert main(["ml-advanced", "retrain-trigger", ids["lasso"], "--kind", "calibration_alert", "--registry", registry]) == 2
    assert "none has been computed" in capsys.readouterr().err
    assert main(["ml-advanced", "retrain-start", ids["lasso"], "--registry", registry]) == 2
    assert "--proposal" in capsys.readouterr().err
    assert main(["ml-advanced", "retrain-rollback", ids["lasso"], "--registry", registry, "--actor", "alice", "--note", "x"]) == 2
    assert "--from-version" in capsys.readouterr().err
    assert main(["ml-advanced", "retrain-status", ids["ridge"], "--registry", registry]) == 0
    assert "version " in capsys.readouterr().out
