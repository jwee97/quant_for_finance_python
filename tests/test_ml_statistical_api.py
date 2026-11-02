"""Release 5 through the API: the validation policy, the stored statistical diagnostics, governance and the prospective ledger behind the forward test.

The service functions are tested directly elsewhere. Here the claims are about the routes: they validate what they take, nothing in a request can supply a classification, a creation time, a status or a
metric, diagnostics are read against the policy in force (and are stale under another), and every payload is strict JSON."""

import json
import warnings

import pytest

from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished, wait

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "ridge"}, "explain": False}
LENIENT = {"name": "lenient", "version": "v1", "criteria": {"segment": "validation", "min_segment_days": 60, "interval_method": "hac", "min_sharpe_ci_lower": -9.0}}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("statistical")
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def parent(app):
    return finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "parent", "mode": "development"}))


def get(app, path, **q):
    out = _safe(app.extra_get[path](q))
    json.dumps(out, allow_nan=False)
    return out


def test_the_routes_exist_and_nothing_is_adopted_by_default(app):
    for path in ("/api/ml/validation-policy", "/api/ml/governance", "/api/ml/statistical-diagnostics"):
        assert path in app.extra_get
    for path in ("/api/ml/validation-policy", "/api/ml/statistical-diagnostics/run", "/api/ml/governance/version", "/api/ml/governance/action"):
        assert path in app.extra_post
    p = get(app, "/api/ml/validation-policy")
    assert p["adopted"] is None and p["history"] == [] and p["example"]["illustrative"] is True and "max_pbo" in p["criteria"]
    assert p["criteria"]["segment"]["kind"] == "choice" and p["criteria"]["min_segment_days"]["min"] == 20


def test_adopting_a_policy_needs_a_person_a_reason_and_exactly_one_of_a_policy_or_the_example(app):
    post = app.extra_post["/api/ml/validation-policy"]
    for body, text in (({"policy": LENIENT, "example": True, "actor": "a", "note": "n"}, "either policy"), ({"actor": "a", "note": "n"}, "either policy"), ({"policy": LENIENT, "actor": "", "note": "n"}, "actor and a note"),
                       ({"policy": LENIENT, "actor": "a", "note": ""}, "actor and a note"), ({"policy": {**LENIENT, "criteria": {"max_pbo": 7}}, "actor": "a", "note": "n"}, "max_pbo"),
                       ({"policy": {**LENIENT, "criteria": {"nonsense": 1}}, "actor": "a", "note": "n"}, "unknown criterion"), ({"policy": {**LENIENT, "illustrative": True}, "actor": "a", "note": "n"}, "name, version, criteria"),
                       ({"policy": LENIENT, "actor": "a", "note": "n", "now": 1}, "does not take")):
        with pytest.raises(ApiError, match=text):
            post(body)
    assert get(app, "/api/ml/validation-policy")["adopted"] is None, "every refusal left the registry without a policy"
    made = _safe(post({"policy": LENIENT, "actor": "jo", "note": "declared before the evaluation"}))
    assert made["actor"] == "jo" and made["illustrative"] is False and made["name"] == "lenient"
    p = get(app, "/api/ml/validation-policy")
    assert p["adopted"]["policy"]["name"] == "lenient" and p["adopted"]["adoption"]["actor"] == "jo" and len(p["history"]) == 1


def test_diagnostics_are_computed_stored_and_read_against_the_policy_in_force(app, parent):
    run = app.extra_post["/api/ml/statistical-diagnostics/run"]
    for body, text in (({"id": parent["id"], "n_boot": 20}, "n_boot"), ({"id": parent["id"], "neighborhood": {"signal.n": 3}}, "neighborhood"), ({"id": parent["id"], "delays": [99]}, "delays"),
                       ({"id": parent["id"], "capitals": [10.0, 20.0]}, "capitals"), ({"id": parent["id"], "policy": "x"}, "does not take"), ({"id": "../x"}, "not a valid identifier")):
        with pytest.raises(ApiError, match=text):
            run(body)
    out = finished(app, run({"id": parent["id"], "n_boot": 300, "seed": 2, "delays": [0, 1]}))
    assert out["policy_hash"] and out["record_seq"] >= 1
    block = out["diagnostics"]
    assert block["stale"] is False and block["record"]["sharpe"]["hac"]["eligibility"] == "applied" and block["record"]["delay"]["rows"][0]["extra_sessions"] == 0
    statuses = {i["id"]: i["status"] for i in block["items"]}
    assert statuses["dx:segment_days"] == "ok" and statuses["dx:sharpe_interval"] == "ok", statuses
    again = get(app, "/api/ml/statistical-diagnostics", id=parent["id"])["diagnostics"]
    assert again["content_hash"] == block["content_hash"] and again["policy"]["name"] == "lenient"
    app.extra_post["/api/ml/validation-policy"]({"policy": {**LENIENT, "name": "other"}, "actor": "sam", "note": "a different declaration"})
    stale = get(app, "/api/ml/statistical-diagnostics", id=parent["id"])["diagnostics"]
    assert stale["stale"] is True and stale["items"] == [] and "compute them again" in stale["note"], "under another policy the stored numbers give no verdict"
    status = get(app, "/api/ml/promotion-status", id=parent["id"])
    sd = next(i for i in status["gates"]["B"]["items"] if i["id"] == "statistical_diagnostics")
    assert sd["status"] == "stale"
    finished(app, run({"id": parent["id"], "n_boot": 300, "seed": 2}))
    assert get(app, "/api/ml/statistical-diagnostics", id=parent["id"])["diagnostics"]["stale"] is False


def test_governance_registers_versions_and_records_every_action_with_a_name(app, parent):
    version = app.extra_post["/api/ml/governance/version"]
    with pytest.raises(ApiError, match="risk_limits is an object"):
        version({"id": parent["id"], "actor": "jo", "risk_limits": [1]})
    with pytest.raises(ApiError, match="unknown risk limit"):
        version({"id": parent["id"], "actor": "jo", "risk_limits": {"leverage": 3}})
    with pytest.raises(ApiError, match="who registered"):
        version({"id": parent["id"], "actor": "", "risk_limits": {}})
    made = _safe(version({"id": parent["id"], "actor": "jo", "risk_limits": {"max_gross": 1.5, "max_drawdown": 0.2}}))
    assert made["created"] is True and made["version_id"].startswith("sv-") and made["content"]["settings"]["execution"] == "vectorized_legacy"
    assert _safe(version({"id": parent["id"], "actor": "jo", "risk_limits": {"max_gross": 1.5, "max_drawdown": 0.2}}))["created"] is False
    vid = made["version_id"]
    act = app.extra_post["/api/ml/governance/action"]
    for body, text in (({"version": vid, "action": "halt", "actor": "jo"}, "who halted"), ({"version": vid, "action": "fly", "actor": "jo"}, "action is one of"), ({"version": vid, "action": "rollback", "actor": "jo", "reason": "r"}, "not a valid identifier"),
                       ({"version": vid, "action": "approve", "actor": "jo", "note": "ok", "stage": "none"}, "staging or production"), ({"version": vid, "action": "recovery", "actor": "jo", "procedure": "short"}, "substance"),
                       ({"version": vid, "action": "halt", "actor": "jo", "reason": "x", "state": "active"}, "does not take")):
        with pytest.raises(ApiError, match=text):
            act(body)
    assert _safe(act({"version": vid, "action": "halt", "actor": "jo", "reason": "reviewing the drawdown"}))["state"] == "halted"
    g = get(app, "/api/ml/governance", id=parent["id"])
    assert g["current"]["state"]["state"] == "halted" and g["current"]["risk_limits"]["limits"]["max_gross"] == 1.5 and g["versions"][-1]["version_id"] == vid
    assert _safe(act({"version": vid, "action": "resume", "actor": "jo", "note": "checked"}))["state"] == "active"
    assert _safe(act({"version": vid, "action": "recovery", "actor": "jo", "procedure": "Halt, flatten the paper book at the next close, return to the previous version, reconcile."}))["event_seq"] >= 1
    assert _safe(act({"version": vid, "action": "approve", "actor": "jo", "note": "reviewed", "stage": "staging"}))["stage"] == "staging"
    events = [e["kind"] for e in get(app, "/api/ml/governance", id=parent["id"])["current"]["events"]]
    assert {"halted", "resumed", "recovery_plan", "deployment_approved", "version_registered"} <= set(events)


def test_the_forward_step_commits_to_the_ledger_and_the_server_classifies_it(app, parent):
    final = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "with-model", "mode": "development"}))
    exp = final["id"]
    with pytest.raises(ApiError, match="does not take"):
        app.extra_post["/api/ml/forward/step"]({"id": exp, "classification": "prospective"})
    with pytest.raises(ApiError, match="does not take"):
        app.extra_post["/api/ml/forward/step"]({"id": exp, "now": 1.0})
    stepped = finished(app, app.extra_post["/api/ml/forward/step"]({"id": exp, "capital": 50_000}))
    c = stepped["signal"]["commitment"]
    assert c["status"] == "committed" and c["classification"] == "replay" and "not before the label's first return day" in c["why"], "simulated history committed today is a replay, whatever the request says"
    assert stepped["summary"]["ledger"]["counts"]["prospective"]["committed"] == 0 and stepped["summary"]["ledger"]["counts"]["replay"]["committed"] == 1
    assert stepped["summary"]["monitoring"]["status"] in ("ok", "alert") and any("replay" in a for a in stepped["summary"]["alerts"])
    rec = get(app, "/api/ml/forward", id=exp)
    assert rec["ledger"]["strategy_version"] == c["strategy_version"] and rec["ledger"]["commitments"][0]["classification"] == "replay" and rec["ledger"]["state"]["state"] == "active"
    assert "cumulative" not in rec["summary"] and "sharpe" not in rec["summary"], "no pseudo-equity is derived from horizon returns"
    again = finished(app, app.extra_post["/api/ml/forward/step"]({"id": exp}))
    assert again["signal"]["commitment"]["status"] == "unchanged" and get(app, "/api/ml/forward", id=exp)["ledger"]["counts"]["replay"]["committed"] == 1


def test_a_halted_version_refuses_the_forward_step_and_the_refusal_is_audited(app):
    final = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "halted-one", "mode": "development", "signal": {"n": 3}}))
    exp = final["id"]
    first = finished(app, app.extra_post["/api/ml/forward/step"]({"id": exp, "as_of": None} if False else {"id": exp}))
    vid = first["signal"]["commitment"]["strategy_version"]
    app.extra_post["/api/ml/governance/action"]({"version": vid, "action": "halt", "actor": "jo", "reason": "manual halt"})
    day = _safe(app.extra_get["/api/ml/forward"]({"id": exp}))["signals"][0]["date"]
    later = finished(app, app.extra_post["/api/ml/forward/step"]({"id": exp, "as_of": str((__import__("pandas").Timestamp(day) - __import__("pandas").Timedelta(days=3)).date())}))
    assert later["signal"]["commitment"]["status"] == "refused" and "halted" in later["signal"]["commitment"]["reason"] and later["signal"]["signal_id"] is None and later["signal"]["orders"] == []
    kinds = [e["kind"] for e in get(app, "/api/ml/governance", id=exp)["current"]["events"]]
    assert "commitment_refused" in kinds


def test_reading_results_and_computing_diagnostics_never_opens_a_protected_segment(app, parent):
    reg = app.extra_get["/api/ml/lineage"].__self__.registry

    def counts():
        with reg._conn() as c:
            return tuple(c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("holdout_access_log", "lineage_events", "plan_events", "holdout_reservations"))

    other = next(r["id"] for r in get(app, "/api/ml/experiments", limit=50)["rows"] if r["id"] != parent["id"])
    before = counts()
    compared = get(app, "/api/ml/compare", ids=f"{parent['id']},{other}")
    assert {r["holdout"] for r in compared["rows"]} <= {"development", "exposed_exploratory", "evaluated_under_plan", "not recorded"} and all(r["execution_mode"] for r in compared["rows"]), \
        "a comparison says which look each result cost and how it was produced"
    html = _safe(app.extra_get["/api/ml/report-html"]({"id": parent["id"]}))["html"]
    assert "development" in html or "Holdout" in html
    for path, q in (("/api/ml/experiments", {"limit": 50}), ("/api/ml/experiment", {"id": parent["id"]}), ("/api/ml/promotion-status", {"id": parent["id"]}), ("/api/ml/trading-diagnostics", {"id": parent["id"]}),
                    ("/api/ml/statistical-diagnostics", {"id": parent["id"]}), ("/api/ml/governance", {"id": parent["id"]}), ("/api/ml/forward", {"id": parent["id"]}), ("/api/ml/holdout", {})):
        get(app, path, **q)
    assert counts()[0] == before[0], "reading a computed result consumes no look and writes no access row"
    stats = reg.report(parent["id"])["segments"]["stats"]
    assert "test" not in stats and "forward" not in stats, "a development run never computed the sealed segments, and no read has computed them since"
    app.extra_post["/api/ml/validation-policy"]({"policy": {"name": "sealed", "version": "v1", "criteria": {"segment": "test", "min_segment_days": 60}}, "actor": "jo", "note": "judging the sealed segment"})
    refused = wait(app, app.extra_post["/api/ml/statistical-diagnostics/run"]({"id": parent["id"]}))
    assert refused["status"] == "error" and "the test segment of this experiment is protected" in refused["error"], "the job ends in an error that names the way forward, and computes nothing"
    assert counts()[0] == before[0] and "test" not in reg.report(parent["id"])["segments"]["stats"]
    app.extra_post["/api/ml/validation-policy"]({"policy": LENIENT, "actor": "jo", "note": "back to the validation segment"})


def test_multi_seed_runs_are_declared_in_advance_looks_of_the_study_and_never_pick_a_seed(app, parent):
    run = app.extra_post["/api/ml/multiseed/run"]
    for body, text in (({"id": parent["id"], "seeds": [1, 2]}, "3 to 12 distinct"), ({"id": parent["id"], "seeds": [1, 1, 2]}, "3 to 12 distinct"), ({"id": parent["id"], "seeds": [1, 2, 3], "select": "best"}, "does not take"),
                       ({"id": parent["id"], "seeds": [1, 2, "x"]}, "3 to 12 distinct"), ({"id": parent["id"], "seeds": [1, 2, 3], "n_boot": 5}, "n_boot")):
        with pytest.raises(ApiError, match=text):
            run(body)
    reg = app.extra_get["/api/ml/lineage"].__self__.registry
    from src.ml import research_controls as rc

    study = rc.study_of_experiment(reg, parent["id"])
    before = rc.raw_looks(reg, study)
    out = finished(app, run({"id": parent["id"], "seeds": [11, 12, 13], "n_boot": 200}))
    assert out["declaration"]["seeds"] == [11, 12, 13] and out["eligibility"] == "applied" and len(out["rows"]) == 3 and "no seed is chosen" in out["selection"]
    assert rc.raw_looks(reg, study) == before + 3, "each seed is a look at the study and is counted"
    shown = get(app, "/api/ml/statistical-diagnostics", id=parent["id"])["diagnostics"]
    assert shown["multiseed"]["declaration"]["hash"] == out["declaration"]["hash"]


def test_a_conflicting_commitment_is_not_logged_as_a_signal_and_is_reported(app):
    final = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "conflict-one", "mode": "development", "signal": {"n": 3}}))
    exp = final["id"]
    reg = app.extra_get["/api/ml/lineage"].__self__.registry
    first = finished(app, app.extra_post["/api/ml/forward/step"]({"id": exp}))
    vid = first["signal"]["commitment"]["strategy_version"]
    with reg._conn() as c:                                                                      # the same decision key arrives with other content (as after a data revision)
        key, content_hash = c.execute("SELECT commit_key, content_hash FROM commitments WHERE strategy_version=?", (vid,)).fetchone()
        n_signals = c.execute("SELECT COUNT(*) FROM forward_signals WHERE experiment_id=?", (exp,)).fetchone()[0]
    from src.ml import monitoring as mon, prospective as pr

    tracker = mon.ForwardTracker(reg, exp, app.config, app.builder)
    real_commit = pr.commit

    def altered(*args, **kw):
        kw["forecasts"] = {k: v + 1.0 for k, v in kw["forecasts"].items()}
        return real_commit(*args, **kw)

    pr.commit = altered
    try:
        again = tracker.step(None, None, 100_000.0)
    finally:
        pr.commit = real_commit
    assert again["commitment"]["status"] == "rejected_conflict" and again["signal_id"] is None and "never changed" in again["commitment"]["reason"]
    with reg._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM forward_signals WHERE experiment_id=?", (exp,)).fetchone()[0] == n_signals
        assert c.execute("SELECT content_hash FROM commitments WHERE commit_key=?", (key,)).fetchone()[0] == content_hash
        assert c.execute("SELECT COUNT(*) FROM commitment_audit WHERE commit_key=? AND event='rejected_conflict'", (key,)).fetchone()[0] == 1


def test_a_grid_over_the_models_own_hyper_parameters_is_accepted_by_the_route_and_refits_in_the_background(app, parent):
    run = app.extra_post["/api/ml/statistical-diagnostics/run"]
    with pytest.raises(ApiError, match="model.params"):
        run({"id": parent["id"], "neighborhood": {"signal.n": 3}})                                         # the message names the model settings too
    out = finished(app, run({"id": parent["id"], "n_boot": 200, "neighborhood": {"model.params.alpha": [1.0, 10.0, 100.0]}}))
    nb = out["diagnostics"]["record"]["neighborhood"]
    assert nb["refit"] is True and nb["declared_points"] == 3 and nb["evaluated"] == 3 and nb["eligibility"] == "applied" and nb["segment"] == "validation"
    assert all(r["experiment_id"] for r in nb["points"]) and any("REFITS" in n for n in nb["notes"])
    mixed = wait(app, run({"id": parent["id"], "n_boot": 200, "neighborhood": {"model.params.alpha": [1.0, 10.0, 100.0], "signal.n": [2, 3]}}))
    assert mixed["status"] == "error" and "not both" in json.dumps(mixed), "a grid that mixes refits with re-runs is refused with the reason"

