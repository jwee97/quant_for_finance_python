"""The research gates explained and made actionable: kinds of obstacle, next actions, the verification controls behind them, and the end-to-end run of requirement P0-08.

Why P0-08 was "never run" (audited before this module was written): the verification machinery worked (``ml-verify`` and ``POST /api/ml/verification/run`` ran its two test files, 21 tests, and persisted the
result), but nothing in the web interface called it, and a status exists only when a run wrote one. The tests here check that the page can now see why a gate is held, call the one registered action that
resolves it, follow it, and read the persisted outcome; and that a browser user can name only registered requirement ids, never a file or a command."""

from __future__ import annotations

import warnings

import pytest

from src.ml import experiments as xe, gate_guidance as gg, gates, registry as rg, verification as vf
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, finished, wait

warnings.filterwarnings("ignore")
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
BASE = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge", "params": {}}}


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    bundle = make_bundle(6, 1300, seed=21, alpha=1.0)
    reg = rg.Registry(tmp_path_factory.mktemp("guidance") / "ml")
    exp = xe.run_experiment({**BASE, "name": "exploratory"}, bundle=bundle, registry=reg, explain=False, robustness=False)
    return reg, exp


def annotated(reg, exp):
    res = gates.evaluate(reg, exp.id)
    return gg.annotate(res, reg, exp.id)


def fixture_checks(reg, kind="ridge", status="pass", skip=(), **kw):
    for req in gates.family_requirements(kind):
        if req not in skip:
            reg.log_check(req, "fixture", status, subject_hash=vf.source_hash(vf.REQUIREMENTS[req][1]), reason=kw.get("reason", "fixture row"), observed=kw.get("observed"))


@pytest.fixture
def fresh(shared, tmp_path):
    """The experiment's evidence with an empty verification history: a copy of the registry's other evidence is not needed, the checks live in their own table."""
    reg, exp = shared
    with reg._conn() as c:                                                                         # noqa: SLF001
        c.execute("DELETE FROM verification_checks")
    return reg, exp


# ------------------------------------------------------------------------------------------------------------------------------ the cards
def test_a_gate_with_no_evidence_names_the_kind_of_obstacle_and_the_one_action_that_resolves_it(fresh):
    reg, exp = fresh
    res = annotated(reg, exp)
    a = res["gates"]["A"]
    assert a["display_status"] == "pending" and a["title"] == "Algorithm Verified" and "profit" in a["not_implied"]
    kinds = {o["kind"] for o in a["obstacles"] if o["item"].startswith("verification:")}
    assert kinds == {"missing_evidence"}, "never run is missing evidence, not a failure"
    act = a["next_action"]
    assert act["id"] == "run_gate_a" and act["endpoint"] == "/api/ml/verification/run" and "P0-12" in act["body"]["requirements"] and "command" not in act["body"]
    assert set(act["body"]["requirements"]) == set(gates.family_requirements("ridge"))
    assert a["blocker"]["kind"] == "missing_evidence" and a["evidence_summary"].endswith("checks met") and a["last_verified"] is None
    assert all(i["action"]["body"]["requirements"] == [i["requirement_id"]] for i in a["items"] if i.get("requirement_id"))


def test_a_stale_pass_is_stale_evidence_and_a_current_pass_needs_nothing(fresh):
    reg, exp = fresh
    fixture_checks(reg)
    res = annotated(reg, exp)
    a = res["gates"]["A"]
    assert a["status"] == "passed" and a["next_action"]["id"] == "none" and a["display_status"] == "passed" and a["last_verified"]["ago"].endswith("ago")
    with reg._conn() as c:                                                                         # noqa: SLF001
        c.execute("UPDATE verification_checks SET subject_hash='0000' WHERE requirement_id='P0-12'")      # the source it checked has since changed
    a = annotated(reg, exp)["gates"]["A"]
    stale = [o for o in a["obstacles"] if o["kind"] == "stale_evidence"]
    assert [o["item"] for o in stale] == ["verification:P0-12"] and a["display_status"] == "stale" and a["next_action"]["body"]["requirements"] == ["P0-12"]


def test_an_execution_failure_is_told_apart_from_a_validation_failure(fresh):
    reg, exp = fresh
    fixture_checks(reg, skip=("P0-12", "P0-13"))
    reg.log_check("P0-12", "fixture", "fail", subject_hash=vf.source_hash(vf.REQUIREMENTS["P0-12"][1]), reason="tests/test_ml_serialization_matrix.py: timed out",
                  observed={"files": [{"file": "tests/test_ml_serialization_matrix.py", "passed": 0, "failed": 0, "skipped": 0, "errors": 1, "seconds": 900,
                                       "details": [{"test": "(timeout)", "message": "the tests of this file did not finish within 900 seconds and were ended", "trace": ""}]}]})
    reg.log_check("P0-13", "fixture", "fail", subject_hash=vf.source_hash(vf.REQUIREMENTS["P0-13"][1]), reason="tests/test_ml_interval_splits.py: test_embargo",
                  observed={"files": [{"file": "tests/test_ml_interval_splits.py", "passed": 5, "failed": 1, "skipped": 0, "errors": 0, "seconds": 3, "details": [{"test": "test_embargo", "message": "assert 3 == 4", "trace": "..."}]}]})
    a = annotated(reg, exp)["gates"]["A"]
    by = {o["item"]: o["kind"] for o in a["obstacles"]}
    assert by["verification:P0-12"] == "execution_failure" and by["verification:P0-13"] == "validation_failure"
    assert a["status"] == "failed" and a["blocker"]["kind"] == "validation_failure", "a real failed test outranks a run that could not finish"
    row = next(i for i in a["items"] if i["id"] == "verification:P0-13")["verification"]
    assert row["failure_details"][0]["message"] == "assert 3 == 4" and row["counts"]["failed"] == 1


def test_a_skipped_family_is_blocked_by_the_environment_not_failed(fresh):
    reg, exp = fresh
    fixture_checks(reg, skip=("P0-12",))
    reg.log_check("P0-12", "fixture", "blocked", subject_hash=vf.source_hash(vf.REQUIREMENTS["P0-12"][1]), reason="2 tests were skipped, so what they check was not verified here: lightgbm missing")
    a = annotated(reg, exp)["gates"]["A"]
    assert a["status"] == "blocked" and [o["kind"] for o in a["obstacles"]] == ["execution_failure"]


def test_when_the_runner_cannot_run_the_card_says_what_blocks_it_instead_of_offering_a_button(fresh, monkeypatch):
    reg, exp = fresh
    monkeypatch.setattr(vf, "preflight", lambda requirements=None: {"can_run": False, "blockers": [{"kind": "runner_missing", "message": "pytest is not installed", "fix": "pip install pytest"}]})
    a = annotated(reg, exp)["gates"]["A"]
    assert a["next_action"]["id"] == "fix_environment" and "pytest is not installed" in a["next_action"]["detail"] and "endpoint" not in a["next_action"]
    assert all(i["action"] is None for i in a["items"] if i.get("requirement_id"))


def test_a_running_verification_shows_as_running(fresh):
    reg, exp = fresh
    vf.LIVE.update(running=True, requirements=["P0-12"], requirement="P0-12", file="tests/x.py")
    try:
        a = annotated(reg, exp)["gates"]["A"]
    finally:
        vf.LIVE.clear()
        vf.LIVE["running"] = False
    assert a["display_status"] == "running" and a["next_action"]["id"] == "wait"


def test_gate_b_keeps_contaminated_evidence_apart_from_a_poor_strategy_and_never_offers_to_clean_it(fresh):
    reg, exp = fresh
    res = annotated(reg, exp)
    b = res["gates"]["B"]
    frozen = next(i for i in b["items"] if i["id"] == "frozen_plan")
    assert frozen["status"] == "blocked" and frozen["kind"] == "contaminated_evidence"
    assert b["next_action"]["id"] == "plan_new_look" and "cannot be undone" in b["next_action"]["detail"]
    assert b["blocker"]["kind"] == "contaminated_evidence" and "outside a frozen plan" in b["blocker"]["text"]
    assert not any(i.get("action") and "holdout" in str(i["action"]).lower() for i in b["items"]), "no control re-labels a read holdout as untouched"
    assert res["gates"]["B"]["status"] in ("blocked", "failed")


def test_gate_c_needs_a_policy_then_prospective_evidence_and_says_which_part_only_time_can_supply(fresh):
    reg, exp = fresh
    c = annotated(reg, exp)["gates"]["C"]
    assert c["status"] == "blocked" and c["next_action"]["id"] in ("adopt_policy", "earlier_gates_first")
    ready = gg.prospective_readiness(reg, exp.id)
    assert ready["policy_adopted"] is False and not any(s["done"] for s in ready["steps"][:2]) and "only by waiting" in ready["cannot_complete_now"]
    assert [s["step"] for s in ready["steps"]][:2] == ["Adopt a validation policy", "Register a strategy version"]


def test_plan_readiness_reads_the_ledger_and_creates_nothing(fresh):
    reg, exp = fresh
    before = len(reg.list_checks())
    r = gg.plan_readiness(reg, exp.id)
    assert r["experiment_id"] == exp.id and len(r["steps"]) == 4 and "cannot be undone" in r["cannot_complete_now"]
    assert r["n_consumed"] >= 1 and r["consumed_through"] and r["first_untouched_date"] > r["consumed_through"], "the exploratory run consumed its test and forward dates"
    assert len(reg.list_checks()) == before


# ------------------------------------------------------------------------------------------------------------------------------ the API
@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vapp")
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


def test_only_registered_requirement_ids_can_be_named_and_one_run_at_a_time(app):
    post = app.extra_post["/api/ml/verification/run"]
    for body in ({"requirements": ["../../etc/passwd"]}, {"requirements": ["tests/test_ml_cli.py"]}, {"requirements": ["P0-08"], "command": "rm -rf /"}, {"requirements": ["P0-08"], "path": "tests/x.py"},
                 {"requirements": ["P0-08"], "scope": "all_pending"}, {"scope": "bogus"}, {"requirements": [f"P0-{i}" for i in range(41)]}):
        with pytest.raises(ApiError):
            post(body)
    assert vf._RUN_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(vf.VerificationError, match="already running"):
            vf.run_verification(rg.Registry(app.root / "artifacts" / "ml-lock"), ["P0-08"])
        assert vf.preflight(["P0-08"])["blockers"][0]["kind"] == "busy"
    finally:
        vf._RUN_LOCK.release()


def test_status_reports_the_runner_and_what_cannot_run(app, monkeypatch):
    s = app.extra_get["/api/ml/verification/status"]({})
    assert s["preflight"]["can_run"] is True and s["live"] is None and s["preflight"]["timeout_per_file_seconds"] == vf.FILE_TIMEOUT
    monkeypatch.setattr(vf, "ROOT", vf.ROOT / "nowhere")                                           # a deployment without the tests folder
    s = app.extra_get["/api/ml/verification/status"]({})
    assert s["preflight"]["can_run"] is False and s["preflight"]["blockers"][0]["kind"] == "tests_missing"
    with pytest.raises(ApiError, match="verification cannot run"):
        app.extra_post["/api/ml/verification/run"]({"requirements": ["P0-08"]})


def test_p0_08_runs_through_the_job_api_is_persisted_and_turns_the_gate_item_green(app):
    """The whole path of the card's button: POST the registered requirement, follow the job, read the persisted check, history and the gate item."""
    resp = app.extra_post["/api/ml/verification/run"]({"requirements": ["P0-08"]})
    job = wait(app, resp)
    assert job["status"] == "done", job.get("error")
    (check,) = job["result"]["checks"]
    assert check["requirement_id"] == "P0-08" and check["status"] == "pass" and sum(f["passed"] for f in check["files"]) == 21 and all(f["failed"] == 0 for f in check["files"])
    rows = app.extra_get["/api/ml/verification"]({"id": "P0-08"})["checks"]
    row = rows[0]
    assert row["status"] == "pass" and row["stale"] is False and row["counts"]["passed"] == 21 and row["seconds"] > 0 and len(row["source_fingerprint"]) == 12
    assert row["tests"] == ["tests/test_ml_architecture_contracts.py", "tests/test_ml_architecture_catalogue.py"]
    hist = app.extra_get["/api/ml/verification/history"]({"id": "P0-08"})["history"]
    assert len(hist) == 1 and hist[0]["current_source"] is True and hist[0]["status"] == "pass"
    assert app.extra_get["/api/ml/verification/status"]({})["live"] is None
    assert vf.LIVE.get("running") is False
