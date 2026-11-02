"""Release 8 through the API: the catalogue of the optional advanced modules and the routes for drift, retraining proposals, champion/challenger pairs, the uncertainty monitor and reproducible reports.

The modules are tested directly in their own files. Here the claims are about the doors: nothing is imported or run unless a request names it, unknown fields and nonsense are refused, a decision names a
person, an autonomous retraining cannot be switched on from a route, a trigger cannot be asserted without the alert behind it, a missing uncertainty estimate is never read as health, a stored report does
not move when its inputs do, and everything that is returned is strict JSON."""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml import prospective as pr
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "signal": {"method": "top_n_long_short", "n": 2, "holding_period": 1},
        "model": {"kind": "ridge"}, "explain": False, "mode": "development"}
R8_MODULES = ["drift", "retrain", "uncertainty_monitor", "champion", "repro_report", "bma", "dro", "horizon_combo", "transfer", "discovery", "offpolicy", "tax_lots"]
OPTIONAL = ("torch", "lightgbm", "xgboost", "catboost", "ngboost", "optuna", "gymnasium", "stable_baselines3", "cvxpy", "transformers")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("r8api")
    bundle = make_bundle(n_assets=8, n=1300, seed=23, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def parent(app):
    return finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "parent8"}))


@pytest.fixture(scope="module")
def ens(app):
    return finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "ens8", "model": {"kind": "model_ensemble", "params": {"members": "ridge,lasso", "weighting": "equal"}}}))


@pytest.fixture(scope="module")
def reg(app):
    return app.extra_get["/api/ml/lineage"].__self__.registry


def get(app, path, **q):
    return _safe(app.extra_get[path](q))


def post(app, path, body):
    return _safe(app.extra_post[path](body))


def data(out):
    return out.get("data", out)


def python(code: str) -> dict:
    run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------------------------------------------------------ off by default
def test_the_catalog_lists_the_twelve_modules_off_by_default_with_the_door_each_has(app):
    d = data(get(app, "/api/ml/advanced"))
    assert [m["id"] for m in d["modules"]] == [f"P3-{i:02d}" for i in range(1, 13)] and "off by default" in d["note"]
    assert all(m["default_state"] == "off" and m["available"] and m["reason"] is None for m in d["modules"])
    assert {m["key"] for m in d["modules"] if "api" in m["surface"]} == {"drift", "retrain", "champion", "uncertainty_monitor", "repro_report"}
    library = [m for m in d["modules"] if m["surface"] == ["library"]]
    assert {m["key"] for m in library} == {"bma", "dro", "horizon_combo", "transfer", "discovery", "offpolicy", "tax_lots"} and all(not m["routes"] and m["cli"] is None for m in library)
    for m in d["modules"]:                                                                                  # every route the catalogue names exists, and no unnamed route does
        for route in m["routes"]:
            verb, path = route.split(" ", 1)
            assert path in (app.extra_get if verb == "GET" else app.extra_post), route
    named = {r.split(" ", 1)[1] for m in d["modules"] for r in m["routes"]}
    assert {p for p in list(app.extra_get) + list(app.extra_post) if p in {"/api/ml/drift", "/api/ml/retraining", "/api/ml/retraining/action", "/api/ml/champion", "/api/ml/champion/register",
                                                                        "/api/ml/champion/commit", "/api/ml/champion/look", "/api/ml/uncertainty-monitor", "/api/ml/reports", "/api/ml/report",
                                                                        "/api/ml/report/build"}} == named
    with pytest.raises(ApiError, match="does not take"):
        get(app, "/api/ml/advanced", x="1")


def test_the_base_workflow_imports_none_of_the_advanced_modules_and_they_need_no_optional_package():
    out = python("import sys, json\nimport src.ml.experiments, src.ml.cli, src.ml.signals, src.ml.verification, src.ml.comparison, src.webapp.ml_api\n"
                 f"mods = {R8_MODULES!r}\nopt = {OPTIONAL!r}\n"
                 "print(json.dumps({'advanced': [m for m in mods if 'src.ml.' + m in sys.modules], 'optional': [m for m in opt if m in sys.modules]}))")
    assert out == {"advanced": [], "optional": []}, "the base ridge workflow must not depend on, or even load, the advanced modules or any accelerator, solver extra or foundation-model package"
    out = python("import sys, json\nfrom src.ml import advanced\nc = advanced.catalog()\n"
                 f"opt = {OPTIONAL!r}\n"
                 "print(json.dumps({'available': [m['key'] for m in c['modules'] if m['available']], 'optional': [m for m in opt if m in sys.modules]}))")
    assert len(out["available"]) == 12 and out["optional"] == [], "all twelve import and run on the base install alone"


# ------------------------------------------------------------------------------------------------------------------ P3-01
def test_the_drift_route_reports_input_and_performance_drift_links_each_look_to_the_last_and_is_recorded(app, parent, ens):
    with pytest.raises(ApiError, match="does not take"):
        get(app, "/api/ml/drift", id=parent["id"], nonsense="1")
    with pytest.raises(ApiError, match="reference_days"):
        get(app, "/api/ml/drift", id=parent["id"], reference_days="2")
    with pytest.raises(ApiError, match="psi_watch"):
        get(app, "/api/ml/drift", id=parent["id"], psi_watch="0.5", psi_alert="0.1")
    with pytest.raises(ApiError, match="correction"):
        get(app, "/api/ml/drift", id=parent["id"], correction="bonferroni")
    with pytest.raises(ApiError, match="not a valid identifier"):
        get(app, "/api/ml/drift", id="../x")
    first = data(get(app, "/api/ml/drift", id=parent["id"]))
    assert first["eligibility"] == "applied" and first["record_seq"] >= 1 and first["spec"]["reference_days"] == 504 and "alert is a proposal" in first["note"]
    feats = first["inputs"]["features"]
    assert feats and all(f["state"] in ("ok", "watch", "alert", "insufficient") for f in feats) and first["inputs"]["reference"]["fixed_at"] and first["inputs"]["reference"]["dates"] > 0 and first["lineage"]["previous_record"] is None
    assert first["performance"]["scope"].startswith("settled decisions on the validation segment") and first["performance"]["state"] in ("ok", "watch", "alert", "insufficient")
    again = data(get(app, "/api/ml/drift", id=parent["id"], reference_days="400", recent_days="40"))
    assert again["lineage"]["previous_record"] == first["record_seq"] and again["lineage"]["looks_so_far"] == 2 and again["spec_hash"] != first["spec_hash"]
    stored = data(get(app, "/api/ml/drift", id=parent["id"], stored="1"))
    assert stored["stored"] is True and stored["record_seq"] == again["record_seq"]
    untouched = data(get(app, "/api/ml/drift", id=ens["id"], stored="1"))
    assert untouched["eligibility"] == "not_computed" and "have not been computed" in untouched["reason"], "reading a stored record never computes one"


# ------------------------------------------------------------------------------------------------------------------ P3-02
def test_retraining_routes_record_proposals_and_decisions_and_cannot_switch_autonomy_on(app, ens):
    def act(**body):
        return data(post(app, "/api/ml/retraining/action", {"id": ens["id"], **body}))

    policy = {"name": "api-test", "min_new_rows": 1, "min_train_days": 100, "cooldown_days": 0}
    with pytest.raises(ApiError, match="does not take"):
        post(app, "/api/ml/retraining/action", {"id": ens["id"], "action": "trigger", "autonomous": True})
    with pytest.raises(ApiError, match="action is one of"):
        act(action="vibes")
    with pytest.raises(ApiError, match="actor is required"):
        act(action="start", proposal=1)
    with pytest.raises(ApiError, match="is not the open proposal"):
        act(action="start", proposal=1, actor="alice")
    with pytest.raises(ApiError, match="autonomous retraining cannot be enabled here"):
        act(action="trigger", policy={**policy, "autonomous": True})
    with pytest.raises(ApiError, match="a retraining policy has"):
        act(action="trigger", policy={**policy, "promote_automatically": True})
    with pytest.raises(ApiError, match="needs a recorded drift report"):
        act(action="trigger", kind="drift_alert", policy=policy)
    with pytest.raises(ApiError, match="a trigger is one of"):
        act(action="trigger", kind="gut_feeling", policy=policy)
    first = act(action="trigger", kind="manual", policy=policy)
    assert first["outcome"] == "proposed" and first["requires_approval"] is True and first["state"]["open_proposal"] == first["proposal"]
    dup = act(action="trigger", kind="manual", policy=policy)
    assert dup["outcome"] == "suppressed" and "duplicate" in dup["reason"] and dup["state"]["open_proposal"] == first["proposal"]
    with pytest.raises(ApiError, match="actor is required"):
        act(action="approve", proposal=first["proposal"], note="looks right")
    with pytest.raises(ApiError, match="note is required"):
        act(action="approve", proposal=first["proposal"], actor="alice")
    with pytest.raises(ApiError, match="not the open proposal"):
        act(action="approve", proposal=first["proposal"] + 99, actor="alice", note="x")
    ok = act(action="approve", proposal=first["proposal"], actor="alice", note="the data since the last fit is enough")
    assert ok["outcome"] == "approved" and ok["state"]["open_proposal"] == first["proposal"], "an approval is not the end of the proposal: nothing has been trained"
    no = act(action="reject", proposal=first["proposal"], actor="bob", note="not now")
    assert no["outcome"] == "rejected" and no["state"]["open_proposal"] is None
    cool = act(action="trigger", kind="manual", policy={**policy, "cooldown_days": 21})
    assert cool["outcome"] == "suppressed" and cool["reason"].startswith("cooldown")
    thin = act(action="trigger", kind="manual", policy={**policy, "min_train_days": 10**6})
    assert thin["outcome"] == "suppressed" and "days of history" in thin["reason"]
    with pytest.raises(ApiError, match="not suspended"):
        act(action="recover", actor="alice", note="checked")
    with pytest.raises(ApiError):
        act(action="rollback", from_version="no-such-version", to_version="other", actor="alice", note="back out")
    h = data(get(app, "/api/ml/retraining", id=ens["id"]))
    kinds = [e["event"] for e in h["events"]]
    assert kinds.count("proposal") == 1 and kinds.count("suppressed") == 3 and "approved" in kinds and "rejected" in kinds and h["events_total"] == len(kinds)
    assert h["state"]["open_proposal"] is None and h["default_policy"]["autonomous"] is False and "never promoted" in h["note"]
    with pytest.raises(ApiError, match="does not take"):
        get(app, "/api/ml/retraining", id=ens["id"], x="1")


def test_a_drift_trigger_needs_the_alert_it_names_in_a_recorded_report(app, ens, reg):
    from src.ml import diagnostics as dx

    policy = {"name": "api-drift", "min_new_rows": 1, "min_train_days": 100, "cooldown_days": 0}
    body = {"id": ens["id"], "action": "trigger", "kind": "drift_alert", "policy": policy}
    quiet = {"inputs": {"features": [{"feature": "momentum_21", "state": "ok"}]}, "performance": {"state": "ok", "reason": "fine"}}
    dx.store(reg, "drift_report", ens["id"], quiet, actor="test")
    with pytest.raises(ApiError, match="no alerting feature"):
        post(app, "/api/ml/retraining/action", body)
    with pytest.raises(ApiError, match="no performance alert"):
        post(app, "/api/ml/retraining/action", {**body, "kind": "performance_alert"})
    loud = {"inputs": {"features": [{"feature": "momentum_21", "state": "alert"}]}, "performance": {"state": "alert", "reason": "the recent IC fell"}}
    dx.store(reg, "drift_report", ens["id"], loud, actor="test")
    out = data(post(app, "/api/ml/retraining/action", body))
    assert out["outcome"] == "proposed" and out["state"]["open_proposal"] == out["proposal"]
    event = [e for e in data(get(app, "/api/ml/retraining", id=ens["id"]))["events"] if e["event"] == "proposal"][-1]
    assert event["trigger"] == "drift_alert" and event["trigger_detail"]["alerting_features"] == ["momentum_21"] and event["trigger_detail"]["drift_record"] >= 1
    again = data(post(app, "/api/ml/retraining/action", {**body, "kind": "performance_alert"}))
    assert again["outcome"] == "suppressed" and "duplicate" in again["reason"], "a second alert while a proposal is open adds a row, not a second proposal"


# ------------------------------------------------------------------------------------------------------------------ P3-10
def test_champion_routes_freeze_a_pair_commit_both_forecasts_together_and_run_the_predeclared_looks(app, reg):
    base = {"champion": "champ-v1", "challenger": "chall-v1", "universe_version": "u1", "horizon": 1, "looks": [10, 20], "alpha": 0.05, "rule": "obrien_fleming", "assumptions": {"cost_bps": 7},
            "data_cutoff": "2026-01-01", "actor": "alice"}
    for bad, text in (({**base, "looks": []}, "looks is a list"), ({**base, "looks": [10, "x"]}, "looks is a list"), ({**base, "looks": [20, 10]}, "strictly increasing"), ({**base, "looks": [5, 20]}, "first at least 10"),
                      ({**base, "challenger": "champ-v1"}, "different strategy versions"), ({**base, "rule": "peeking"}, "spending rule"), ({**base, "actor": ""}, "actor is required"),
                      ({**base, "assumptions": "x"}, "assumptions is an object"), ({**base, "extra": 1}, "does not take"), ({**base, "alpha": 0.9}, "alpha")):
        with pytest.raises(ApiError, match=text):
            post(app, "/api/ml/champion/register", bad)
    p = data(post(app, "/api/ml/champion/register", base))
    assert p["status"] == "registered" and p["pair_id"].startswith("cc-") and p["levels"] and sum(p["levels"]) <= 0.05 + 1e-9 and p["looks_remaining"] == 2
    assert data(post(app, "/api/ml/champion/register", base))["status"] == "unchanged" and data(post(app, "/api/ml/champion/register", base))["pair_id"] == p["pair_id"]
    assert data(post(app, "/api/ml/champion/register", {**base, "looks": [10, 30]}))["pair_id"] != p["pair_id"], "a different schedule is a different pair"
    assets = [f"A{i}" for i in range(8)]
    rng = np.random.default_rng(5)
    fc = {"champion": {a: 0.01 for a in assets}, "challenger": {a: 0.02 for a in assets}}
    commit = {"pair": p["pair_id"], "decision_ts": "2035-01-02", "forecasts": fc, "data_cutoff": "2026-01-01", "fit_cutoffs": {"champion": "2025-12-20", "challenger": "2025-12-28"},
              "model_created": {"champion": "2025-12-21", "challenger": "2025-12-29"}, "actor": "alice"}
    for bad, text in (({**commit, "forecasts": {"champion": fc["champion"]}}, "exactly 'champion' and 'challenger'"), ({**commit, "forecasts": {**fc, "challenger": {}}}, "forecasts.challenger is an object"),
                      ({**commit, "forecasts": {**fc, "challenger": {"A0": float("nan")}}}, "finite numbers"), ({**commit, "fit_cutoffs": {"champion": "2025-12-20"}}, "exactly 'champion' and 'challenger'"),
                      ({**commit, "decision_ts": "2020-01-02"}, "predates the registration"), ({**commit, "data_cutoff": "2026-01-02"}, "differs from the pair's declared"), ({**commit, "pair": "cc-unknown"}, "unknown champion/challenger pair"),
                      ({**commit, "actor": ""}, "actor is required"), ({**commit, "surplus": 1}, "does not take")):
        with pytest.raises(ApiError, match=text):
            post(app, "/api/ml/champion/commit", bad)
    with pytest.raises(ApiError, match="needs 10 settled decision pairs; there are 0"):
        post(app, "/api/ml/champion/look", {"pair": p["pair_id"], "actor": "alice"})
    commitments = []
    for i in range(10):
        day = (pd.Timestamp("2035-01-02") + pd.tseries.offsets.BDay(i)).date().isoformat()
        outcome = rng.normal(0, 0.02, 8)
        fcs = {"champion": dict(zip(assets, rng.normal(0, 0.02, 8))), "challenger": dict(zip(assets, 0.8 * outcome + rng.normal(0, 0.01, 8)))}
        out = data(post(app, "/api/ml/champion/commit", {**commit, "decision_ts": day, "forecasts": fcs}))
        assert out["champion"]["classification"] == "prospective" == out["challenger"]["classification"], out
        commitments.append((out, dict(zip(assets, outcome)), day))
    s0 = data(get(app, "/api/ml/champion", id=p["pair_id"]))
    assert s0["settled_pairs"] == 0 and s0["next_look_needs"] == 10 and s0["next_look_available"] is False and s0["verdict"] == "pending"
    for out, outcome, day in commitments:                                                                    # outcomes arrive later, through the ledger
        for role in ("champion", "challenger"):
            pr.settle(reg, out[role]["commitment_id"], outcome, actor="settler", now=pd.Timestamp(day).timestamp() + 5 * 86400)
    s1 = data(get(app, "/api/ml/champion", id=p["pair_id"]))
    assert s1["settled_pairs"] == 10 and s1["next_look_available"] is True and s1["looks_remaining"] == 2
    look = data(post(app, "/api/ml/champion/look", {"pair": p["pair_id"], "actor": "alice"}))
    assert look["look"] == 1 and look["of"] == 2 and look["pairs_used"] == 10 and look["promotes"] is False and look["verdict"] in ("challenger_better", "champion_better", "no_difference") and look["final"] is False
    assert look["level"] == p["levels"][0] and look["challenger_ic"] > look["champion_ic"]
    with pytest.raises(ApiError, match="needs 20 settled decision pairs; there are 10"):
        post(app, "/api/ml/champion/look", {"pair": p["pair_id"], "actor": "alice"})
    s2 = data(get(app, "/api/ml/champion", id=p["pair_id"]))
    assert len(s2["looks_taken"]) == 1 and s2["looks_remaining"] == 1 and s2["verdict"] == "pending"
    with pytest.raises(ApiError, match="does not take"):
        get(app, "/api/ml/champion", id=p["pair_id"], x="1")
    with pytest.raises(ApiError, match="unknown champion/challenger pair"):
        get(app, "/api/ml/champion", id="cc-0000")


# ------------------------------------------------------------------------------------------------------------------ P3-11
def test_the_uncertainty_monitor_checks_each_thing_apart_and_never_reads_a_missing_estimate_as_health(app, parent, ens, reg):
    one = data(get(app, "/api/ml/uncertainty-monitor", id=parent["id"]))
    assert one["eligibility"] == "not_applicable" and "not a clean bill of health" in one["reason"] and "record_seq" not in one
    assert data(get(app, "/api/ml/uncertainty-monitor", id=parent["id"], stored="1"))["eligibility"] == "not_computed"
    with pytest.raises(ApiError, match="does not take"):
        get(app, "/api/ml/uncertainty-monitor", id=ens["id"], nonsense="1")
    with pytest.raises(ApiError, match="as_of must be a date"):
        get(app, "/api/ml/uncertainty-monitor", id=ens["id"], as_of="yesterday")
    d = data(get(app, "/api/ml/uncertainty-monitor", id=ens["id"]))
    assert d["eligibility"] == "applied" and d["scope"].startswith("settled decisions on the validation segment") and d["record_seq"] >= 1
    checks = {c["check"]: c for c in d["checks"]}
    assert set(checks) == {"calibration", "distribution", "uncertainty", "disagreement"}
    assert checks["calibration"]["state"] in ("ok", "watch", "alert") and 0.0 <= checks["calibration"]["value"] <= 1.0
    assert checks["uncertainty"]["state"] in ("ok", "watch", "alert") and checks["distribution"]["state"] in ("ok", "watch", "alert")
    dis = checks["disagreement"]
    assert dis["state"] in ("ok", "watch", "alert") and dis["members"] == ["ridge", "lasso"] and dis["missing_members"] == [] and dis["value"] is not None and dis["value"] > 0, "the ensemble kept its members' forecasts"
    assert d["overall"]["state"] in ("ok", "watch", "alert", "degraded") and d["overall"]["state"] != "unavailable"
    assert checks["calibration"]["state"] != "not_applicable"
    solo = data(get(app, "/api/ml/uncertainty-monitor", id=parent["id"]))
    assert solo["eligibility"] == "not_applicable", "a single model states no distribution and has no members"
    assert all(c["level"] is None or c["state"] == "alert" for c in d["checks"])
    old = reg.root / "experiments" / ens["id"] / "extra_members.parquet"                                 # an ensemble saved before member forecasts were kept
    hidden = old.with_suffix(".hidden")
    old.rename(hidden)
    try:
        legacy = data(get(app, "/api/ml/uncertainty-monitor", id=ens["id"]))
    finally:
        hidden.rename(old)
    gd = {c["check"]: c for c in legacy["checks"]}["disagreement"]
    assert gd["state"] == "unavailable" and gd["value"] is None and "saved before its members' forecasts were kept" in gd["reason"] and legacy["overall"]["state"] != "ok"
    second = data(get(app, "/api/ml/uncertainty-monitor", id=ens["id"], reference_end="2100-01-01"))
    assert second["record_seq"] > d["record_seq"] and second["reference_end"] == "2100-01-01"
    assert second["overall"]["state"] == "unavailable" and "absence of evidence" in second["overall"]["reason"], "with no recent window nothing can be checked, and that is not reported as healthy"
    assert all(c["state"] in ("unavailable", "not_applicable") and c["value"] is None for c in second["checks"])
    assert data(get(app, "/api/ml/uncertainty-monitor", id=ens["id"], stored="1"))["record_seq"] == second["record_seq"]


# ------------------------------------------------------------------------------------------------------------------ P3-12
def test_reports_are_built_from_a_snapshot_stored_append_only_and_do_not_move_when_their_inputs_do(app, parent):
    with pytest.raises(ApiError, match="does not take"):
        post(app, "/api/ml/report/build", {"id": parent["id"], "nonsense": 1})
    with pytest.raises(ApiError, match="format is json or markdown"):
        get(app, "/api/ml/report", report="rp-x", format="pdf")
    with pytest.raises(ApiError, match="unknown report"):
        get(app, "/api/ml/report", report="rp-nothing")
    built = data(post(app, "/api/ml/report/build", {"id": parent["id"], "actor": "alice"}))
    assert built["status"] == "built" and built["report_id"].startswith("rp-") and len(built["report_hash"]) == 16
    assert data(post(app, "/api/ml/report/build", {"id": parent["id"]}))["status"] == "unchanged", "unchanged inputs are the same report"
    listing = data(get(app, "/api/ml/reports", id=parent["id"]))["reports"]
    assert [r["report_id"] for r in listing] == [built["report_id"]] and listing[0]["report_hash"] == built["report_hash"]
    rep = data(get(app, "/api/ml/report", report=built["report_id"], verify="1"))
    assert rep["report_hash"] == built["report_hash"] and rep["reproduction"]["reproduced"] is True and rep["inputs_now"]["stale"] is False
    sections = rep["content"]
    for key in ("header", "performance", "uncertainty_selection", "coverage", "costs_capacity", "attribution", "lineage", "gates", "limitations"):
        assert key in sections, key
    md = data(get(app, "/api/ml/report", report=built["report_id"], format="markdown"))
    assert md["markdown"].startswith("# Report on") and "Limitations and unresolved items" in md["markdown"] and "content" not in md
    get(app, "/api/ml/drift", id=parent["id"], recent_days="30")                                           # the registry moves on: a newer diagnostic record
    after = data(get(app, "/api/ml/report", report=built["report_id"], verify="1"))
    assert after["report_hash"] == built["report_hash"] and after["content"] == rep["content"], "the stored report is exactly as it was built"
    assert after["reproduction"]["reproduced"] is True and after["inputs_now"]["stale"] is True and any(c["input"] == "diagnostic:drift_report" for c in after["inputs_now"]["changed"])
    newer = data(post(app, "/api/ml/report/build", {"id": parent["id"]}))
    assert newer["status"] == "built" and newer["report_id"] != built["report_id"]
    assert {r["report_id"] for r in data(get(app, "/api/ml/reports", id=parent["id"]))["reports"]} == {built["report_id"], newer["report_id"]}


# ------------------------------------------------------------------------------------------------------------------ the command line
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from src.cli import main

    tmp = tmp_path_factory.mktemp("r8cli")
    b = make_bundle(8, 1300, seed=29, alpha=1.0, macro=False)
    csv = tmp / "toy.csv"
    b.prices.rename_axis("date").to_csv(csv)
    reg = str(tmp / "ml")
    base = ["ml-train", "--prices", str(csv), "--groups", "price", "momentum", "volatility", "--refit-every", "63", "--min-train-days", "400", "--registry", reg, "--no-explain", "--development"]
    assert main([*base, "--model", "ridge", "--name", "cli-ridge"]) == 0
    assert main([*base, "--model", "model_ensemble", "--param", "members=ridge,lasso", "--param", "weighting=equal", "--name", "cli-ens"]) == 0
    from src.ml.registry import Registry

    rows = {r["name"]: r["id"] for r in Registry(reg).list_experiments(10)}
    return {"reg": reg, "tmp": tmp, "ridge": rows["cli-ridge"], "ens": rows["cli-ens"]}


def run(capsys, *argv):
    from src.cli import main

    capsys.readouterr()
    code = main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_the_command_line_lists_the_catalogue_and_runs_the_same_services_as_the_api(world, capsys):
    reg = world["reg"]
    code, out, _ = run(capsys, "ml-advanced", "catalog", "--registry", "/nonexistent-registry-dir")
    assert code == 0 and "Drift monitoring" in out and "library" in out and "off by default" in out
    code, out, _ = run(capsys, "ml-advanced", "catalog", "--registry", reg, "--json")
    assert code == 0 and len(json.loads(out)["modules"]) == 12
    code, out, err = run(capsys, "ml-advanced", "drift", "--registry", reg)
    assert code == 2 and "needs --target" in err
    code, out, _ = run(capsys, "ml-advanced", "drift", world["ridge"], "--registry", reg, "--json")
    first = json.loads(out)
    assert code == 0 and first["eligibility"] == "applied" and first["record_seq"] >= 1 and first["inputs"]["features"]
    code, out, _ = run(capsys, "ml-advanced", "drift", world["ridge"], "--registry", reg, "--recent-days", "40")
    assert code == 0 and "settled-IC drift:" in out and "look 2" in out and "an alert is a proposal" in out
    code, out, err = run(capsys, "ml-advanced", "drift", "no-such-experiment", "--registry", reg)
    assert code == 2 and err.startswith("error:")


def test_the_command_line_monitor_says_not_applicable_for_a_model_without_a_distribution_and_checks_an_ensemble(world, capsys):
    reg = world["reg"]
    code, out, _ = run(capsys, "ml-advanced", "monitor", world["ridge"], "--registry", reg)
    assert code == 0 and "not_applicable" in out and "not a clean bill of health" in out
    code, out, _ = run(capsys, "ml-advanced", "monitor", world["ens"], "--registry", reg)
    assert code == 0 and "overall" in out and all(name in out for name in ("calibration", "distribution", "uncertainty", "disagreement"))
    code, out, _ = run(capsys, "ml-advanced", "monitor", world["ens"], "--registry", reg, "--reference-end", "2100-01-01", "--json")
    d = json.loads(out)
    assert code == 0 and d["overall"]["state"] == "unavailable" and all(c["value"] is None for c in d["checks"]), "nothing to compare is reported as unavailable, never as healthy"


def test_the_command_line_records_proposals_and_decisions_and_names_the_person(world, capsys):
    reg, tmp, exp = world["reg"], world["tmp"], world["ens"]
    pol = tmp / "policy.json"
    pol.write_text(json.dumps({"name": "cli", "min_new_rows": 1, "min_train_days": 100, "cooldown_days": 0}))
    bad = tmp / "auto.json"
    bad.write_text(json.dumps({"name": "cli", "autonomous": True}))
    code, _, err = run(capsys, "ml-advanced", "retrain-trigger", exp, "--registry", reg, "--policy", str(bad))
    assert code == 2 and "autonomous retraining cannot be enabled here" in err
    code, _, err = run(capsys, "ml-advanced", "retrain-trigger", exp, "--registry", reg, "--policy", str(pol), "--kind", "drift_alert")
    assert code == 2 and "needs a recorded drift report" in err
    code, out, _ = run(capsys, "ml-advanced", "retrain-trigger", exp, "--registry", reg, "--policy", str(pol), "--json")
    proposed = json.loads(out)
    assert code == 0 and proposed["outcome"] == "proposed" and proposed["requires_approval"] is True
    code, out, _ = run(capsys, "ml-advanced", "retrain-trigger", exp, "--registry", reg, "--policy", str(pol))
    assert code == 0 and "suppressed" in out and "duplicate" in out
    code, _, err = run(capsys, "ml-advanced", "retrain-approve", exp, "--registry", reg, "--proposal", str(proposed["proposal"]))
    assert code == 2 and "needs --actor" in err
    code, out, _ = run(capsys, "ml-advanced", "retrain-reject", exp, "--registry", reg, "--proposal", str(proposed["proposal"]), "--actor", "bob", "--note", "not now")
    assert code == 0 and "rejected" in out
    code, out, _ = run(capsys, "ml-advanced", "retrain-status", exp, "--registry", reg)
    assert code == 0 and "open proposal none" in out and "not suspended" in out and "rejected" in out and "proposal" in out
    code, _, err = run(capsys, "ml-advanced", "retrain-recover", exp, "--registry", reg, "--actor", "bob", "--note", "checked")
    assert code == 2 and "not suspended" in err


def test_the_command_line_registers_a_pair_and_refuses_a_look_before_the_pairs_have_settled(world, capsys):
    reg, tmp = world["reg"], world["tmp"]
    spec = tmp / "pair.json"
    spec.write_text(json.dumps({"champion": "cli-champ", "challenger": "cli-chall", "universe_version": "u1", "horizon": 5, "looks": [10, 20], "alpha": 0.05, "rule": "pocock"}))
    code, _, err = run(capsys, "ml-advanced", "champion-register", "--registry", reg, "--spec", str(spec))
    assert code == 2 and "needs --actor" in err
    code, out, _ = run(capsys, "ml-advanced", "champion-register", "--registry", reg, "--spec", str(spec), "--actor", "alice", "--json")
    pair = json.loads(out)
    assert code == 0 and pair["status"] == "registered" and pair["pair_id"].startswith("cc-") and pair["looks_remaining"] == 2
    code, out, _ = run(capsys, "ml-advanced", "champion-show", pair["pair_id"], "--registry", reg)
    assert code == 0 and "cli-champ (champion) against cli-chall (challenger)" in out and "settled pairs 0, verdict pending" in out
    commit = tmp / "commit.json"
    fc = {"champion": {"A0": 0.01, "A1": -0.02, "A2": 0.0}, "challenger": {"A0": 0.02, "A1": -0.01, "A2": 0.01}}
    body = {"decision_ts": "2035-01-02", "forecasts": fc, "data_cutoff": "2026-01-01", "fit_cutoffs": {"champion": "2025-12-20", "challenger": "2025-12-28"},
            "model_created": {"champion": "2025-12-21", "challenger": "2025-12-29"}}
    commit.write_text(json.dumps(body))
    code, _, err = run(capsys, "ml-advanced", "champion-commit", pair["pair_id"], "--registry", reg, "--spec", str(commit))
    assert code == 2 and "needs --actor" in err
    code, out, _ = run(capsys, "ml-advanced", "champion-commit", pair["pair_id"], "--registry", reg, "--spec", str(commit), "--actor", "alice")
    assert code == 0 and "champion committed (prospective), challenger committed (prospective)" in out
    code, out, _ = run(capsys, "ml-advanced", "champion-commit", pair["pair_id"], "--registry", reg, "--spec", str(commit), "--actor", "alice", "--json")
    again = json.loads(out)
    assert code == 0 and again["champion"]["status"] == "unchanged" and again["challenger"]["status"] == "unchanged", "the same commitment twice is the same commitment"
    sneaky = tmp / "sneaky.json"
    sneaky.write_text(json.dumps({**body, "now": 0}))
    code, _, err = run(capsys, "ml-advanced", "champion-commit", pair["pair_id"], "--registry", reg, "--spec", str(sneaky), "--actor", "alice")
    assert code == 2 and "a commitment spec has" in err, "the clock is not something a caller can set"
    early = tmp / "early.json"
    early.write_text(json.dumps({**body, "decision_ts": "2020-01-02"}))
    code, _, err = run(capsys, "ml-advanced", "champion-commit", pair["pair_id"], "--registry", reg, "--spec", str(early), "--actor", "alice")
    assert code == 2 and "predates the registration" in err
    code, _, err = run(capsys, "ml-advanced", "champion-look", pair["pair_id"], "--registry", reg, "--actor", "alice")
    assert code == 2 and "needs 10 settled decision pairs; there are 0" in err
    code, _, err = run(capsys, "ml-advanced", "champion-look", pair["pair_id"], "--registry", reg)
    assert code == 2 and "needs --actor" in err
    broken = tmp / "broken.json"
    broken.write_text("{not json")
    code, _, err = run(capsys, "ml-advanced", "champion-register", "--registry", reg, "--spec", str(broken), "--actor", "alice")
    assert code == 2 and "cannot read" in err


def test_the_command_line_builds_shows_and_verifies_a_report_that_does_not_move(world, capsys):
    reg, exp = world["reg"], world["ridge"]
    code, out, _ = run(capsys, "ml-advanced", "report-build", exp, "--registry", reg, "--actor", "alice", "--json")
    built = json.loads(out)
    assert code == 0 and built["status"] in ("built", "unchanged") and built["report_id"].startswith("rp-")
    code, out, _ = run(capsys, "ml-advanced", "report-show", built["report_id"], "--registry", reg, "--markdown")
    assert code == 0 and out.startswith("# Report on") and "Limitations and unresolved items" in out
    code, out, _ = run(capsys, "ml-advanced", "report-verify", built["report_id"], "--registry", reg)
    assert code == 0 and "reproduced" in out and "NOT reproduced" not in out and "unchanged" in out
    run(capsys, "ml-advanced", "drift", exp, "--registry", reg, "--recent-days", "30")                  # a newer diagnostic record
    code, out, _ = run(capsys, "ml-advanced", "report-verify", built["report_id"], "--registry", reg)
    assert code == 0 and "reproduced" in out and "have changed since" in out and "diagnostic:drift_report" in out
    code, out, _ = run(capsys, "ml-advanced", "report-show", built["report_id"], "--registry", reg, "--json")
    assert code == 0 and json.loads(out)["report_hash"] == built["report_hash"], "the stored report is the same after its inputs moved"
    code, _, err = run(capsys, "ml-advanced", "report-show", "rp-nothing", "--registry", reg)
    assert code == 2 and "unknown report" in err
