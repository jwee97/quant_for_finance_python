"""Release 3 exit conditions: three independent gates computed from persisted evidence, evidence invalidation, descendant blockers and backend enforcement of promotion.

The evaluator is exercised on real experiments in a real registry. Where a test needs a check, a selection test or a null study to exist, it writes the row through the same function that production code uses
(``log_check``, ``selection.persist``, ``null_controls.run``); those rows are fixtures standing in for work that other tests and CI lanes do for real. Nothing here shows that any strategy deserves a gate:
the claim is that the gate logic reads persisted evidence, treats missing, stale, inapplicable and failed evidence as distinct, and cannot be told its own answer."""

import warnings

import pytest

from src.ml import experiments as xe, frozen_eval as fe, gates, null_controls as nc, provenance as pv, registry as rg, selection as sel, verification as vf
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
BASE = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge", "params": {}}}


@pytest.fixture(scope="module")
def bundle():
    return make_bundle(6, 1300, seed=21, alpha=1.0)


@pytest.fixture
def reg(tmp_path):
    return rg.Registry(tmp_path / "ml")


@pytest.fixture(scope="module")
def shared(bundle, tmp_path_factory):
    """One registry with an exploratory run, a development run and a plan-evaluated run, built once."""
    reg = rg.Registry(tmp_path_factory.mktemp("gates") / "ml")
    exploratory = xe.run_experiment({**BASE, "name": "exploratory"}, bundle=bundle, registry=reg, explain=False, robustness=False)
    development = xe.run_experiment({**BASE, "name": "development"}, bundle=bundle, registry=reg, explain=False, robustness=False, stop_after="validation")
    return reg, exploratory, development


def pass_all_checks(reg, kind="ridge", skip=()):
    """Fixture rows: a passing, current check for every requirement that applies to the family (the real ones are written by `ml-verify --run`)."""
    for req in gates.family_requirements(kind):
        if req not in skip:
            reg.log_check(req, "fixture", "pass", subject_hash=vf.source_hash(vf.REQUIREMENTS[req][1]), reason="fixture row for the evaluator's logic")


def item(res, gate, item_id):
    return next(i for i in res["gates"][gate]["items"] if i["id"] == item_id)


# --------------------------------------------------------------------------------------------------------------------- Gate A
def test_without_evidence_gate_a_is_pending_and_names_what_is_missing(shared):
    reg, exp, _ = shared
    res = gates.evaluate(reg, exp.id)
    a = res["gates"]["A"]
    assert a["status"] == "pending" and a["stale"] is False and res["policy_version"] == gates.POLICY_VERSION
    assert any("requirement P0-12" in r for r in a["reasons"]) and item(res, "A", "verification:P0-13")["status"] == "missing"
    assert {i["status"] for i in a["items"] if not i["id"].startswith("verification")} == {"ok"}, "provenance, lineage, contract and bundle exist for this experiment"
    assert res["gates"]["B"]["status"] in ("blocked", "pending") and res["gates"]["C"]["status"] == "blocked"


def test_gate_a_passes_on_current_evidence_and_says_nothing_about_profit(shared):
    reg, exp, _ = shared
    pass_all_checks(reg)
    res = gates.evaluate(reg, exp.id)
    assert res["gates"]["A"]["status"] == "passed" and res["gates"]["A"]["stale"] is False and res["gates"]["A"]["reasons"] == []
    assert "nothing about profit" in res["note"]


def test_a_failed_a_blocked_and_a_stale_check_are_three_different_things(shared):
    reg, exp, _ = shared
    pass_all_checks(reg)
    reg.log_check("P0-13", "fixture", "fail", subject_hash=vf.source_hash("intervals"), reason="2 tests failed")
    assert gates.evaluate(reg, exp.id)["gates"]["A"]["status"] == "failed"
    reg.log_check("P0-13", "fixture", "blocked", subject_hash=vf.source_hash("intervals"), reason="3 tests were skipped")
    blocked = gates.evaluate(reg, exp.id)["gates"]["A"]
    assert blocked["status"] == "blocked" and any("skipped" in r for r in blocked["reasons"])
    reg.log_check("P0-13", "fixture", "pass", subject_hash="hash-of-older-source", reason="passed once")
    stale = gates.evaluate(reg, exp.id)
    assert stale["gates"]["A"]["status"] == "pending" and stale["gates"]["A"]["stale"] is True and item(stale, "A", "verification:P0-13")["status"] == "stale"
    reg.log_check("P0-13", "fixture", "pass", subject_hash=vf.source_hash("intervals"))
    assert gates.evaluate(reg, exp.id)["gates"]["A"]["status"] == "passed"


def test_a_legacy_experiment_without_provenance_lineage_or_bundle_cannot_pass(shared):
    reg, _, _ = shared
    with reg._conn() as c:                                                                      # an experiment row as a pre-Release-3 registry holds it: no provenance, no lineage event, no files
        c.execute("INSERT INTO experiments (id, name, created, status, config_hash, dataset_key, universe, model_kind, trials_at_run, stage) VALUES ('legacy', 'old', 'x', 'done', 'h', 'd', 'etf', 'ridge', 0, 'none')")
    pass_all_checks(reg)
    res = gates.evaluate(reg, "legacy")
    a = res["gates"]["A"]
    assert a["status"] == "pending" and {item(res, "A", i)["status"] for i in ("provenance", "lineage", "output_contract", "inference_bundle")} == {"missing"}
    assert any("saved before" in r or "ran before" in r or "no provenance" in r for r in a["reasons"])


def test_a_change_to_the_checked_out_code_invalidates_the_provenance_item_and_the_history_shows_it(shared, monkeypatch):
    reg, exp, _ = shared
    pass_all_checks(reg)
    first = gates.decide(reg, exp.id, "tester")
    assert first["gates"]["A"]["status"] == "passed" and first["changed"] is True
    again = gates.decide(reg, exp.id, "tester")
    assert again["changed"] is False and again["gates"]["A"]["seq"] == first["gates"]["A"]["seq"], "an unchanged decision is not written twice"
    monkeypatch.setattr(gates, "source_tree_hash", lambda: {"sha256": "a-different-tree"})
    later = gates.decide(reg, exp.id, "tester")
    a = later["gates"]["A"]
    assert a["status"] == "pending" and a["stale"] is True and item(later, "A", "provenance")["status"] == "stale" and a["previous"]["status"] == "passed"
    hist = [h for h in gates.history(reg, exp.id) if h["gate"] == "A"]
    assert [h["status"] for h in hist][:2] == ["pending", "passed"] and hist[0]["previous_seq"] == hist[1]["seq"] and hist[0]["policy_version"] == gates.POLICY_VERSION and hist[0]["evidence"]["hash"]


def test_a_critical_failure_in_an_ancestor_blocks_its_descendants(shared):
    reg, exp, _ = shared
    pass_all_checks(reg)
    child = xe.rebacktest(reg, exp.id, {"signal": {"n": 2}})
    res = gates.evaluate(reg, child.id)
    assert res["gates"]["A"]["status"] == "passed" and "the fitted model of" in item(res, "A", "inference_bundle")["reason"], "a re-backtest uses its parent's fitted model and bundle"
    with reg._conn() as c:                                                                      # the parent is a reinforcement-learning family whose ledger check fails; the child is not
        c.execute("UPDATE experiments SET model_kind='ppo' WHERE id=?", (exp.id,))
    reg.log_check("P0-02", "fixture", "fail", subject_hash=vf.source_hash("rl"), reason="ledger reconciliation failed")
    inherited = gates.evaluate(reg, child.id)["gates"]["A"]
    assert inherited["status"] == "blocked" and any("ancestor" in r and "blocks its descendants" in r for r in inherited["reasons"])
    with reg._conn() as c:
        c.execute("UPDATE experiments SET model_kind='ridge' WHERE id=?", (exp.id,))
    assert gates.evaluate(reg, child.id)["gates"]["A"]["status"] == "passed", "the block follows the evidence: with the family's check no longer applying, it lifts"


# --------------------------------------------------------------------------------------------------------------------- Gate B and C
def test_exploratory_and_development_runs_cannot_earn_gate_b(shared):
    reg, exploratory, development = shared
    pass_all_checks(reg)
    e, d = gates.evaluate(reg, exploratory.id), gates.evaluate(reg, development.id)
    assert item(e, "B", "frozen_plan")["status"] == "blocked" and "exploratory" in item(e, "B", "frozen_plan")["reason"]
    assert item(d, "B", "frozen_plan")["status"] == "missing" and "development run" in item(d, "B", "frozen_plan")["reason"]
    assert e["gates"]["B"]["status"] == "blocked"


@pytest.fixture(scope="module")
def evaluated(bundle, tmp_path_factory):
    reg = rg.Registry(tmp_path_factory.mktemp("gates-b") / "ml")
    plan = fe.plan_and_reserve(reg, "u-etf", [BASE], freeze=True, bundle=bundle)
    out = fe.evaluate_plan(reg, plan["plan_id"], idempotency_key="gate-b", bundle=bundle)
    return reg, out["experiments"][0]["experiment_id"], plan


def selection_row(reg, status, passes=None, method="dsr"):
    stat = {} if passes is None else {"passes": passes}
    sel.persist(reg, "u-etf", [sel.Evidence(method, status, "fixture", statistic=stat)])


def null_study(reg, promoted, n=120):
    spec = nc.NullSpec(study_id="u-etf", replications=n, min_replications=100, tolerance=0.10, precision=0.08, seed=1)
    made = nc.register(reg, spec)
    nc.run(reg, made["null_id"], lambda s: {"promoted": (s // 7919) < promoted, "selected": None})
    return made["null_id"]


def test_gate_b_reads_the_plan_the_selection_controls_and_the_null_study(evaluated):
    reg, exp_id, plan = evaluated
    res = gates.evaluate(reg, exp_id)
    plan_item = item(res, "B", "frozen_plan")
    assert plan_item["status"] == "ok" and plan["plan_id"] in plan_item["reason"] and "evaluated once" in plan_item["reason"]
    assert item(res, "B", "selection")["status"] == "missing" and item(res, "B", "null_controls")["status"] == "missing"
    selection_row(reg, "not_applicable", method="pbo")
    assert item(gates.evaluate(reg, exp_id), "B", "selection")["status"] == "blocked", "a procedure that did not apply is not a pass"
    selection_row(reg, "applied", False, "dsr")
    assert item(gates.evaluate(reg, exp_id), "B", "selection")["status"] == "failed"
    selection_row(reg, "applied", True, "dsr")
    selection_row(reg, "applied", True, "pbo")
    ok = gates.evaluate(reg, exp_id)
    assert item(ok, "B", "selection")["status"] == "ok" and "dsr" in item(ok, "B", "selection")["reason"]
    null_study(reg, promoted=1, n=8)                                                            # too few replications: plumbing only
    assert item(gates.evaluate(reg, exp_id), "B", "null_controls")["status"] == "blocked"
    null_study(reg, promoted=90, n=300)                                                         # a false-promotion rate far above the tolerance
    assert item(gates.evaluate(reg, exp_id), "B", "null_controls")["status"] == "failed"
    null_study(reg, promoted=3, n=300)
    assert item(gates.evaluate(reg, exp_id), "B", "null_controls")["status"] == "ok"


def test_gate_b_stays_blocked_while_the_later_releases_evidence_has_no_provider(evaluated):
    reg, exp_id, _ = evaluated
    pass_all_checks(reg)
    res = gates.evaluate(reg, exp_id)
    b = res["gates"]["B"]
    assert item(res, "B", "realistic_net_oos")["status"] == "blocked" and "Release 4" in item(res, "B", "realistic_net_oos")["reason"]
    assert item(res, "B", "statistical_diagnostics")["status"] == "blocked" and "Release 5" in item(res, "B", "statistical_diagnostics")["reason"]
    assert b["status"] == "blocked" and res["gates"]["C"]["status"] == "blocked" and any("requires Gate B" in r for r in res["gates"]["C"]["reasons"])


@pytest.fixture
def seeded(evaluated):
    """The selection and null-control evidence of a study that was done properly, written in the order a real study would produce it."""
    reg, exp_id, plan = evaluated
    selection_row(reg, "applied", True, "dsr")
    selection_row(reg, "applied", True, "pbo")
    null_study(reg, promoted=3, n=300)
    return reg, exp_id, plan


def passing_policy():
    ok = lambda item_id: (lambda registry, experiment_id, report: gates.Item(item_id, item_id, "ok", "test provider", "x"))      # noqa: E731
    return gates.DEFAULT_POLICY.with_providers(b={"realistic_net_oos": ok("realistic_net_oos"), "statistical_diagnostics": ok("statistical_diagnostics")},
                                               c={"prospective_evidence": ok("prospective_evidence"), "operational_policy": ok("operational_policy")}, version="test-policy")


def test_all_three_gates_can_pass_when_every_kind_of_evidence_exists(seeded):
    reg, exp_id, _ = seeded
    pass_all_checks(reg)
    res = gates.evaluate(reg, exp_id, passing_policy())
    assert [res["gates"][g]["status"] for g in "ABC"] == ["passed", "passed", "passed"] and res["policy_version"] == "test-policy"


# --------------------------------------------------------------------------------------------------------------------- enforcement
def test_the_registry_refuses_a_stage_the_evidence_does_not_support(shared):
    reg, exp, _ = shared
    mid = reg.list_models()[0]["id"]
    reg.promote(mid, "candidate")                                                               # registering as a candidate makes no claim
    with pytest.raises(gates.PromotionBlocked, match="cannot move to staging: Gate [AB] is"):
        reg.promote(mid, "staging")
    with pytest.raises(gates.PromotionBlocked, match="cannot move to production"):
        reg.promote(mid, "production", approval={"actor": "me", "note": "ship it"})
    assert next(m for m in reg.list_models() if m["id"] == mid)["stage"] == "candidate"
    reg.promote(mid, "archived")
    reg.promote(mid, "none")
    stages = [(h["from_stage"], h["to_stage"]) for h in gates.stage_history(reg, mid)]
    assert stages[-3:] == [("candidate", "candidate"), ("candidate", "archived"), ("archived", "none")] or stages[-2:] == [("candidate", "archived"), ("archived", "none")]


def test_production_needs_the_gates_and_a_recorded_human_approval(seeded, monkeypatch):
    reg, exp_id, _ = seeded
    pass_all_checks(reg)
    mid = reg.list_models()[0]["id"]
    monkeypatch.setattr(gates, "DEFAULT_POLICY", passing_policy())
    with pytest.raises(gates.PromotionBlocked, match="production is a human decision"):
        reg.promote(mid, "production")
    with pytest.raises(gates.PromotionBlocked, match="human decision"):
        reg.promote(mid, "production", approval={"actor": "  ", "note": "x"})
    staged = reg.promote(mid, "staging")
    assert staged["allowed"] and set(staged["gate_seqs"]) == {"A", "B", "C"}
    done = reg.promote(mid, "production", approval={"actor": "a.person", "note": "reviewed the prospective record"})
    assert done["allowed"] and next(m for m in reg.list_models() if m["id"] == mid)["stage"] == "production"
    last = gates.stage_history(reg, mid)[-1]
    assert last["to_stage"] == "production" and last["actor"] == "a.person" and last["gate_seq"]
    assert reg.get_experiment(exp_id)["stage"] == "production"


def test_a_request_cannot_supply_its_own_gate_status(shared):
    reg, exp, _ = shared
    for forged in ({"gate": "A", "status": "passed"}, {"status": "passed"}, {"gates": {"B": "passed"}}):
        with pytest.raises(TypeError):
            gates.decide(reg, exp.id, "x", **forged)
        with pytest.raises(TypeError):
            gates.evaluate(reg, exp.id, **forged)
    assert gates.evaluate(reg, exp.id)["gates"]["C"]["status"] != "passed" and pv.load(reg, exp.id) is not None
