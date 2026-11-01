"""Release 3, P0-23 and P0-25 (and the shared infrastructure of P1-26): study lineage for every entry point, frozen evaluation plans, reservations, and the holdout access ledger.

The ledger is exercised without any model where only its rules are in question (synthetic reservations, real SQLite, real threads), and through the real pipeline where the claim is about the pipeline (a
reserved history cannot be read by a direct run, a development run cannot reach it, a replayed evaluation reads nothing again)."""

import dataclasses
import sqlite3
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import datasets as ds_mod, ensembles as en, experiments as xe, explainability as xp, frozen_eval as fe, optimization as opt, registry as rg, research_controls as rc, selection as sel, strategy_search as ss
from src.ml.config import experiment_config
from src.ops.hpo import Int
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


def states(reg, op_key):
    with reg._conn() as c:
        return [r["state"] for r in c.execute("SELECT state FROM lineage_events WHERE op_key=? ORDER BY seq", (op_key,))]


def identity(assets=("A", "B", "C"), start="2023-01-02", end="2023-12-29", outcome="h1"):
    return {"assets": list(assets), "start": start, "end": end, "outcome_hash": outcome}


def frozen_plan(reg, study="s1", ident=None, name="m", alpha=1.0):
    plan = rc.create_plan(reg, study, [{**BASE, "name": name, "model": {"kind": "ridge", "params": {"alpha": alpha}}}])
    rc.reserve(reg, plan["plan_id"], ident or identity())
    return rc.freeze_plan(reg, plan["plan_id"])


# --------------------------------------------------------------------------------------------------------------------- P0-23: lineage
def test_an_experiment_is_registered_before_it_runs_and_ends_as_done(bundle, reg):
    seen = {}

    def spy(message):
        if message == "training walk-forward":                                           # the work has started: the operation must already be in the record, and the experiment not
            with reg._conn() as c:
                seen["ops"] = [(r["state"], r["kind"]) for r in c.execute("SELECT state, kind FROM lineage_events ORDER BY seq")]
                seen["experiments"] = c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    exp = xe.run_experiment(BASE, bundle=bundle, registry=reg, progress=spy, explain=False, robustness=False)
    assert seen["ops"] == [("registered", "experiment"), ("running", "experiment")] and seen["experiments"] == 0
    key = exp.report["lineage"]["operation"]
    assert states(reg, key) == ["registered", "running", "done"] and rc.raw_looks(reg, "u-etf") == 1
    assert rc.lineage(reg, experiment_id=exp.id)["operations"][0]["experiment_id"] == exp.id


def test_a_failed_and_a_cancelled_experiment_are_looks_too(bundle, reg):
    with pytest.raises(xe.ExperimentError):
        xe.run_experiment({**BASE, "split": {"min_train_days": 4000, "refit_every": 126}}, bundle=bundle, registry=reg, explain=False)
    count_after_failure = rc.raw_looks(reg, "u-etf")

    def cancel(message):
        if message == "training walk-forward":
            raise rc.ResearchCancelled("stopped by the user")

    with pytest.raises(rc.ResearchCancelled):
        xe.run_experiment(BASE, bundle=bundle, registry=reg, progress=cancel, explain=False)
    lin = rc.lineage(reg, "u-etf")
    assert count_after_failure == 1 and lin["raw_looks"] == 2 and lin["failed"] == 1 and lin["cancelled"] == 1
    assert {o["state"] for o in lin["operations"]} == {"failed", "cancelled"} and "error" in next(o for o in lin["operations"] if o["state"] == "failed")["detail"]


def test_every_way_of_trying_something_lands_in_the_study(bundle, reg):
    base = xe.run_experiment(BASE, bundle=bundle, registry=reg, explain=False, robustness=False, save=True)
    opt.tune(BASE, {"model.params.alpha": __import__("src.ops.hpo", fromlist=["LogFloat"]).LogFloat(0.1, 100.0)}, n_trials=2, method="random", seed=1, bundle=bundle, registry=reg, study_name="t")
    opt.tune(BASE, {"signal.n": Int(1, 3)}, n_trials=2, method="random", seed=1, bundle=bundle, registry=reg, study_name="sig")
    ss.search(BASE, bundle=bundle, registry=reg, kinds=("ridge", "lasso"), n_random=2, n_bayes=0, generations=0, seed=2)
    xe.rebacktest(reg, base.id, {"signal": {"n": 2}})
    xe.rebacktest(reg, base.id, {"backtest": {"cost_bps": 9.0}})
    en.meta_label_experiment(reg, base.id, "logistic", min_rows=200)
    other = xe.run_experiment({**BASE, "model": {"kind": "ridge", "params": {"alpha": 20.0}}}, bundle=bundle, registry=reg, explain=False, robustness=False)
    en.combine_experiments(reg, [base.id, other.id], methods=("equal", "inverse_vol"), min_history=60)
    lin = rc.lineage(reg, "u-etf")
    kinds = lin["by_kind"]
    for kind in ("experiment", "hpo_trial", "signal_variation", "model_choice", "portfolio_variation", "meta_label", "ensemble"):
        assert kinds.get(kind, 0) >= 1, f"{kind} is not in the lineage: {kinds}"
    assert kinds["experiment"] == 2 and kinds["hpo_trial"] == 2 and lin["by_state"] == {"done": lin["raw_looks"]}
    meta = next(o for o in lin["operations"] if o["kind"] == "meta_label")
    assert meta["parent_key"] == rc.op_key_of_experiment(reg, base.id)                       # a derived run keeps its parent
    assert any(o["kind"] == "ensemble" and o["detail"]["experiments"] == [base.id, other.id] for o in lin["operations"])


def test_a_search_trial_and_its_finalisation_are_one_look(bundle, reg):
    out = opt.tune(BASE, {"model.params.alpha": __import__("src.ops.hpo", fromlist=["LogFloat"]).LogFloat(0.1, 100.0)}, n_trials=3, method="random", seed=1, bundle=bundle, registry=reg, study_name="t")
    assert rc.raw_looks(reg, "u-etf") == 3 and reg.trial_count(__import__("src.ml.config", fromlist=["dataset_key"]).dataset_key(experiment_config(BASE))) == 3
    done = fe.compat_finalize(reg, "u-etf", BASE, [out["best"]["params"]], idempotency_key="k-final", bundle=bundle)
    exp = done["_experiments"][0]
    assert rc.raw_looks(reg, "u-etf") == 3, "the finalisation re-runs a counted try; it must not be counted again"
    assert exp.report["trials"]["dataset_trials"] == 3 + 1, "the deflated Sharpe count is the 3 trials plus the experiment itself, not the trials twice"
    kinds = {o["kind"]: o["counts"] for o in rc.lineage(reg, "u-etf")["operations"] if o["kind"] in ("finalize", "evaluation")}
    assert kinds == {"evaluation": False}


def test_registering_the_same_operation_twice_is_refused_and_a_resume_is_not_a_second_look(reg):
    op = rc.begin(reg, "hpo_trial", "s1", op_key="same", start=True)
    with pytest.raises(rc.DuplicateOperation):
        rc.begin(reg, "hpo_trial", "s1", op_key="same")
    op.done()
    with pytest.raises(rc.DuplicateOperation):                                            # a finished operation cannot be resumed either
        rc.begin(reg, "hpo_trial", "s1", op_key="same", resume=True)
    crashed = rc.begin(reg, "hpo_trial", "s1", op_key="crashed")
    assert crashed.state == "running"                                                      # the process dies here: no terminal event
    again = rc.begin(reg, "hpo_trial", "s1", op_key="crashed", resume=True)
    again.done()
    assert states(reg, "crashed") == ["registered", "running", "running", "done"] and rc.raw_looks(reg, "s1") == 2


def test_history_survives_a_renamed_experiment_another_feature_set_and_another_cache_key(bundle, reg):
    a = xe.run_experiment({**BASE, "name": "first"}, bundle=bundle, registry=reg, explain=False, robustness=False)
    b = xe.run_experiment({**BASE, "name": "renamed", "dataset": {"feature_groups": ["price", "momentum"]}}, bundle=bundle, registry=reg, explain=False, robustness=False)
    c = xe.run_experiment({**BASE, "dataset": {"feature_groups": FEATURES, "end": "2019-06-28"}}, bundle=bundle, registry=reg, explain=False, robustness=False)
    assert len({r["dataset_key"] for r in (reg.get_experiment(i) for i in (a.id, b.id, c.id))}) == 2        # the legacy cache key differs
    lin = rc.lineage(reg, "u-etf")
    assert lin["raw_looks"] == 3 and len(lin["feature_sets"]) == 2 and len(lin["configurations"]) == 3
    named = xe.run_experiment({**BASE, "study": "alt"}, bundle=bundle, registry=reg, explain=False, robustness=False, dataset=None)
    assert rc.raw_looks(reg, "alt") == 1 and rc.raw_looks(reg, "u-etf") == 3 and named.report["lineage"]["study"] == "alt"
    rc.link_study(reg, "alt", "u-etf")                                                     # a study that continues another inherits its looks
    assert rc.raw_looks(reg, "alt") == 4
    with pytest.raises(rc.ControlError, match="not replaced"):
        rc.link_study(reg, "alt", "elsewhere")


def test_deleting_an_experiment_lowers_no_count_and_leaves_a_tombstone(bundle, reg):
    a = xe.run_experiment(BASE, bundle=bundle, registry=reg, explain=False, robustness=False)
    key = rg.Registry and __import__("src.ml.config", fromlist=["dataset_key"]).dataset_key(experiment_config(BASE))
    looks, trials = rc.raw_looks(reg, "u-etf"), reg.trial_count(key)
    reg.delete_experiment(a.id)
    assert rc.raw_looks(reg, "u-etf") == looks == 1 and reg.trial_count(key) == trials == 1
    with reg._conn() as c:
        tomb = c.execute("SELECT * FROM experiment_tombstones WHERE experiment_id=?", (a.id,)).fetchone()
        assert tomb["study_id"] == "u-etf" and tomb["status"] == "done" and tomb["report_sha256"]
        assert c.execute("SELECT COUNT(*) FROM provenance WHERE experiment_id=?", (a.id,)).fetchone()[0] == 1
        assert c.execute("SELECT COUNT(*) FROM holdout_access_log WHERE result_ref=?", (a.id,)).fetchone()[0] == 1
    assert rc.lineage(reg, "u-etf")["tombstones"][0]["experiment_id"] == a.id


def test_the_history_cannot_be_rewritten_through_the_registry_tables(reg):
    rc.begin(reg, "experiment", "s1", op_key="x").done()
    plan = frozen_plan(reg)
    with reg._conn() as c:                                                                     # rows must exist for a row trigger to have anything to refuse
        c.execute("INSERT INTO provenance (experiment_id, identity, record, created) VALUES ('e', 'i', '{}', 'now')")
        c.execute("INSERT INTO gate_decisions (subject_id, gate, policy_version, status, at) VALUES ('e', 'A', 'v', 'pending', 'now')")
    for sql in ("DELETE FROM lineage_events", "UPDATE lineage_events SET counts=0", "UPDATE evaluation_plans SET content='{}'", "DELETE FROM evaluation_plans", "DELETE FROM plan_events",
                "DELETE FROM provenance", "UPDATE provenance SET identity='z'", "DELETE FROM gate_decisions", "UPDATE gate_decisions SET status='passed'", "UPDATE holdout_access_log SET status='x'",
                "DELETE FROM holdout_access_log"):
        with pytest.raises(sqlite3.IntegrityError), reg._conn() as c:
            c.execute(sql)
    assert rc.raw_looks(reg, "s1") == 1 and rc.plan_state(reg, plan["plan_id"], brief=True)["state"] == "frozen"


# --------------------------------------------------------------------------------------------------------------------- P0-28: feature search
def test_an_adaptive_feature_search_registers_every_variant_and_its_choice_is_reproducible(reg):
    ic = pd.DataFrame({"t": [4.2, 3.9, 0.4, -0.2, 2.1], "n": 500}, index=["f1", "f2", "f3", "f4", "f5"])
    ev = sel.adjust_screen(ic, alpha=0.05, method="holm")
    assert ev.status == "applied" and ev.statistic["selected"] == ["f1", "f2"]
    variants = [{"features": [f], "score": float(t), "accepted": f in ev.statistic["selected"], "reason": "adjusted p" if f in ev.statistic["selected"] else "not significant after Holm"} for f, t in ic["t"].items()]
    out = rc.register_feature_search(reg, "s1", variants, "rank IC t-statistic", selected=ev.statistic["selected"], rule={"alpha": 0.05, "adjusted_p": ev.statistic["adjusted_p"]}, adjustment=ev.as_dict())
    lin = rc.lineage(reg, "s1")
    assert out["n_tried"] == 5 and out["n_rejected"] == 3 and lin["by_kind"] == {"feature_search": 5}, "rejected variants are looks too; the choice itself adds none"
    assert lin["raw_looks"] == 5 and len(lin["feature_sets"]) == 6                                         # five single-feature variants and the chosen pair
    again = rc.reproduce_feature_choice(reg, out["choice"], {}, sel.rule_select)
    assert again["reproduced"] and again["recorded"] == ["f1", "f2"] and again["adjustment"]["statistic"]["method"] == "holm"


# --------------------------------------------------------------------------------------------------------------------- P0-25: plans and the ledger
def test_a_plan_is_immutable_and_several_candidates_need_a_declared_selection_treatment(reg):
    one = rc.create_plan(reg, "s1", [{**BASE, "name": "a"}])
    assert rc.create_plan(reg, "s1", [{**BASE, "name": "a"}])["plan_id"] == one["plan_id"], "the same plan is the same plan"
    assert rc.create_plan(reg, "s1", [{**BASE, "name": "b"}])["plan_id"] != one["plan_id"]
    with pytest.raises(rc.PlanError, match="selection treatment"):
        rc.create_plan(reg, "s1", [{**BASE, "name": "a"}, {**BASE, "name": "b"}])
    two = rc.create_plan(reg, "s1", [{**BASE, "name": "a"}, {**BASE, "name": "b"}], selection_adjustment="holm")
    assert len(two["content"]["candidates"]) == 2 and two["content"]["selection_adjustment"] == "holm"
    assert one["content"]["candidates"][0]["config"]["model"]["kind"] == "ridge" and one["state"] == "draft"
    with pytest.raises(rc.PlanError, match="at least one"):
        rc.create_plan(reg, "s1", [])


def test_a_plan_cannot_be_evaluated_until_it_is_reserved_and_frozen(reg):
    plan = rc.create_plan(reg, "s1", [{**BASE, "name": "a"}])
    with pytest.raises(rc.PlanError, match="draft"):
        rc.begin_evaluation(reg, plan["plan_id"], "k1")
    with pytest.raises(rc.PlanError, match="reserve the plan"):
        rc.freeze_plan(reg, plan["plan_id"])
    rc.reserve(reg, plan["plan_id"], identity())
    with pytest.raises(rc.PlanError, match="reserved, not frozen"):
        rc.begin_evaluation(reg, plan["plan_id"], "k1")
    assert rc.freeze_plan(reg, plan["plan_id"])["state"] == "frozen" and rc.freeze_plan(reg, plan["plan_id"])["state"] == "frozen"
    with pytest.raises(rc.PlanError, match="a reservation needs"):
        rc.reserve(reg, rc.create_plan(reg, "s1", [{**BASE, "name": "z"}])["plan_id"], {"assets": ["A"]})
    rc.void_plan(reg, rc.create_plan(reg, "s1", [{**BASE, "name": "v"}])["plan_id"], "changed my mind")


def test_one_evaluation_event_per_plan_and_a_retry_with_the_same_key_gets_the_same_result(reg):
    plan = frozen_plan(reg)
    first = rc.begin_evaluation(reg, plan["plan_id"], "key-1")
    assert first["status"] == "claimed" and first["context"].reuse is False
    rc.complete_evaluation(reg, first["context"], "exp-1,exp-2", {"experiments": ["exp-1", "exp-2"]})
    again = rc.begin_evaluation(reg, plan["plan_id"], "key-1")
    assert again["status"] == "replayed" and again["result"] == {"experiments": ["exp-1", "exp-2"]} and again["result_ref"] == "exp-1,exp-2"
    with pytest.raises(rc.PlanError, match="already evaluated"):                           # a different key is a new look
        rc.begin_evaluation(reg, plan["plan_id"], "key-2")
    reused = rc.begin_evaluation(reg, plan["plan_id"], "key-2", allow_reuse=True)
    assert reused["status"] == "claimed" and reused["context"].reuse is True
    log = rc.access_log(reg, plan_id=plan["plan_id"])
    assert [r["access_kind"] for r in log if r["status"] == "claimed"][::-1] == ["evaluate", "evaluate_reuse"], "the ledger says which look was the deliberate re-use"
    assert rc.plan_state(reg, plan["plan_id"], brief=True)["evaluations"] == 2


def test_a_claim_that_never_finished_is_in_progress_until_it_is_resumed(reg):
    plan = frozen_plan(reg)
    assert rc.begin_evaluation(reg, plan["plan_id"], "crash")["status"] == "claimed"      # the process dies before complete_evaluation
    assert rc.begin_evaluation(reg, plan["plan_id"], "crash")["status"] == "in_progress"
    resumed = rc.begin_evaluation(reg, plan["plan_id"], "crash", resume=True)
    assert resumed["status"] == "claimed" and resumed["resumed"] is True
    rc.fail_evaluation(reg, resumed["context"], "RuntimeError: out of memory")
    failed = rc.begin_evaluation(reg, plan["plan_id"], "crash")
    assert failed["status"] == "failed" and "out of memory" in failed["detail"]["error"]
    with pytest.raises(rc.PlanError):                                                      # a failed evaluation still consumed the holdout
        rc.begin_evaluation(reg, plan["plan_id"], "another")
    with pytest.raises(rc.PlanError, match="belongs to another plan"):
        rc.begin_evaluation(reg, frozen_plan(reg, name="other")["plan_id"], "crash")


def test_concurrent_requests_cannot_both_get_the_one_look(reg):
    plan = frozen_plan(reg)
    results, errors = [], []

    def go(key):
        try:
            results.append(rc.begin_evaluation(reg, plan["plan_id"], key))
        except rc.PlanError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=go, args=(f"race-{i}",)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert [r["status"] for r in results] == ["claimed"] and len(errors) == 7, "exactly one of eight different keys wins the evaluation"
    plan2 = frozen_plan(reg, ident=identity(("X", "Y"), "2022-01-03", "2022-12-30", "h-other"), name="p2")          # another history: the first plan's evaluation overlaps nothing here
    same = []
    threads = [threading.Thread(target=lambda: same.append(rc.begin_evaluation(reg, plan2["plan_id"], "shared-key")["status"])) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert same.count("claimed") == 1 and same.count("in_progress") == 7, "the same key from eight workers is one claim"


def test_reserving_history_that_was_already_looked_at_is_detected_whatever_it_is_called(reg):
    a = frozen_plan(reg, "s1", identity(("A", "B", "C"), "2023-01-02", "2023-12-29", "h1"), name="a")
    clean = rc.plan_state(reg, a["plan_id"], brief=True)["reservation"]
    assert clean["untouched"] is True and clean["overlaps"] == []
    rc.complete_evaluation(reg, rc.begin_evaluation(reg, a["plan_id"], "ka")["context"], "e1")
    # another study, another dataset (a subset of the assets), a partly overlapping stretch of dates
    b = rc.create_plan(reg, "renamed-study", [{**BASE, "name": "b"}])
    res = rc.reserve(reg, b["plan_id"], identity(("B", "C", "Z"), "2023-07-03", "2024-03-29", "h2"))["reservation"]
    assert res["untouched"] is False and res["overlaps"][0]["plan_id"] == a["plan_id"] and res["overlaps"][0]["shared_assets"] == ["B", "C"] and res["overlaps"][0]["consumed"] is True
    rc.freeze_plan(reg, b["plan_id"])
    with pytest.raises(rc.PlanError, match="overlap it"):
        rc.begin_evaluation(reg, b["plan_id"], "kb")
    # disjoint assets or dates are untouched; the same outcome content under other names is caught by its hash
    c = rc.create_plan(reg, "s2", [{**BASE, "name": "c"}])
    assert rc.reserve(reg, c["plan_id"], identity(("X", "Y"), "2023-01-02", "2023-12-29", "h3"))["reservation"]["untouched"] is True
    d = rc.create_plan(reg, "s3", [{**BASE, "name": "d"}])
    assert rc.reserve(reg, d["plan_id"], identity(("A", "B", "C"), "2025-01-02", "2025-12-31", "h4"))["reservation"]["untouched"] is True
    e = rc.create_plan(reg, "s4", [{**BASE, "name": "e"}])
    res_e = rc.reserve(reg, e["plan_id"], identity(("Q1", "Q2"), "2031-01-01", "2031-12-31", "h1"))["reservation"]
    assert res_e["overlaps"][0]["same_outcome_content"] is True and res_e["untouched"] is False
    assert [r["access_kind"] for r in rc.access_log(reg)].count("reserve") == 5


# --------------------------------------------------------------------------------------------------------------------- P0-25: through the pipeline
def plan_for_data(reg, bundle, study="s1", freeze=True, candidates=None, **kw):
    return fe.plan_and_reserve(reg, study, candidates or [BASE], freeze=freeze, bundle=bundle, **kw)


def test_a_direct_run_cannot_read_a_reserved_history_but_a_development_run_can(bundle, reg):
    plan = plan_for_data(reg, bundle)
    assert plan["state"] == "frozen" and plan["reservation"]["untouched"] is True and sorted(plan["reservation"]["assets"]) == sorted(bundle.assets)
    with pytest.raises(xe.ExperimentError, match="reservation .* protects"):
        xe.run_experiment(BASE, bundle=bundle, registry=reg, explain=False, robustness=False)
    subset = dataclasses.replace(bundle, prices=bundle.prices.iloc[:, :4], returns=bundle.returns.iloc[:, :4], investable=bundle.investable.iloc[:, :4])
    with pytest.raises(xe.ExperimentError, match="protects"):                                  # a new universe, a new dataset key, the same history
        xe.run_experiment(BASE, bundle=subset, registry=reg, explain=False, robustness=False)
    dev = xe.run_experiment(BASE, bundle=bundle, registry=reg, explain=False, robustness=False, stop_after="validation")
    assert dev.report["holdout"]["status"] == "development" and set(dev.report["segments"]["stats"]) == {"train", "validation"}
    assert not [a for a in rc.access_log(reg) if a["access_kind"] in ("exposure", "evaluate")], "nothing read a protected outcome"
    with pytest.raises(rc.HoldoutAccessError, match="frozen evaluation plan"):
        opt.finalize(BASE, {}, bundle=bundle, registry=reg)
    with pytest.raises(rc.HoldoutAccessError, match="every date"):
        xp.feature_ic(dev.dataset)


def test_the_frozen_evaluation_reads_the_holdout_once_and_a_retry_reads_nothing(bundle, reg, monkeypatch):
    plan = plan_for_data(reg, bundle, candidates=[BASE, {**BASE, "name": "wider", "model": {"kind": "ridge", "params": {"alpha": 40.0}}}], selection_adjustment="holm")
    out = fe.evaluate_plan(reg, plan["plan_id"], idempotency_key="final-1", bundle=bundle)
    assert out["status"] == "evaluated" and len(out["experiments"]) == 2 and out["untouched"] is True
    for r in out["experiments"]:
        h = r["holdout"]
        assert h["status"] == "evaluated_under_plan" and h["plan_id"] == plan["plan_id"] and h["untouched"] is True and set(r["segments"]) == {"train", "validation", "test", "forward"}
    calls = []
    real = xe.run_experiment
    monkeypatch.setattr(fe, "run_experiment", lambda *a, **k: calls.append(1) or real(*a, **k))
    replay = fe.evaluate_plan(reg, plan["plan_id"], idempotency_key="final-1", bundle=bundle)
    assert replay["status"] == "replayed" and calls == [] and [e["experiment_id"] for e in replay["result"]["experiments"]] == [e["experiment_id"] for e in out["experiments"]]
    with pytest.raises(rc.PlanError, match="already evaluated"):
        fe.evaluate_plan(reg, plan["plan_id"], idempotency_key="final-2", bundle=bundle)
    spent = xe.run_experiment(BASE, bundle=bundle, registry=reg, explain=False, robustness=False)
    assert spent.report["holdout"]["status"] == "exposed_exploratory", "after the plan's evaluation the history is spent: a direct run is recorded as an exposure, not refused"
    lin = rc.lineage(reg, "s1")
    assert lin["raw_looks"] == 0, "an evaluation is not a new try: the candidates were the plan's"
    assert {o["kind"] for o in lin["operations"]} == {"evaluation"} and all(o["counts"] is False for o in lin["operations"])


def test_an_authorisation_covers_only_what_the_plan_froze(bundle, reg):
    plan = plan_for_data(reg, bundle)
    claim = rc.begin_evaluation(reg, plan["plan_id"], "k-scope")
    with pytest.raises(xe.ExperimentError, match="not a candidate or the baseline"):
        xe.run_experiment({**BASE, "model": {"kind": "ridge", "params": {"alpha": 7.0}}}, bundle=bundle, registry=reg, evaluation=claim["context"], explain=False, robustness=False)
    forged = rc.EvaluationContext(plan["plan_id"], plan["reservation"]["reservation_id"], "never-claimed", "me", "api")
    with pytest.raises(xe.ExperimentError, match="not backed by a claimed evaluation"):
        xe.run_experiment(BASE, bundle=bundle, registry=reg, evaluation=forged, explain=False, robustness=False)


def test_a_run_outside_any_plan_is_recorded_as_an_exposure_and_taints_a_later_reservation(bundle, reg):
    exp = xe.run_experiment(BASE, bundle=bundle, registry=reg, explain=False, robustness=False)
    h = exp.report["holdout"]
    assert h["status"] == "exposed_exploratory" and h["untouched"] is False and "cannot support the Strategy Validated gate" in h["note"]
    exposure = [a for a in rc.access_log(reg) if a["access_kind"] == "exposure"]
    assert len(exposure) == 1 and exposure[0]["result_ref"] == exp.id and sorted(exposure[0]["assets"]) == sorted(bundle.assets)
    plan = plan_for_data(reg, bundle, study="later")
    assert plan["reservation"]["untouched"] is False and plan["reservation"]["overlaps"][0]["kind"] == "exposure"
    with pytest.raises(rc.PlanError, match="overlap it"):
        rc.begin_evaluation(reg, plan["plan_id"], "late")
    ok = rc.begin_evaluation(reg, plan["plan_id"], "late", allow_reuse=True)
    assert ok["context"].reuse is True


def test_a_re_analysis_inherits_what_its_parent_read(bundle, reg):
    plan = plan_for_data(reg, bundle)
    out = fe.evaluate_plan(reg, plan["plan_id"], idempotency_key="k-parent", bundle=bundle)
    parent = out["experiments"][0]["experiment_id"]
    child = xe.rebacktest(reg, parent, {"signal": {"n": 2}})
    h = child.report["holdout"]
    assert h["status"] == "reanalysis_of_exposed_outcomes" and h["untouched"] is False and h["plan_id"] == plan["plan_id"]
    assert [a["access_kind"] for a in rc.access_log(reg, plan_id=plan["plan_id"]) if a["result_ref"] == child.id] == ["reanalysis"]
    dev = xe.run_experiment(BASE, bundle=bundle, registry=reg, explain=False, robustness=False, stop_after="validation")
    from_dev = en.meta_label_experiment(reg, dev.id, "logistic", min_rows=100)[0]
    assert from_dev.report["holdout"]["status"] == "development"                      # nothing protected was read by the parent, so nothing is inherited
    assert child.report["lineage"]["parent"] == parent and rc.lineage(reg, experiment_id=child.id)["parents"][1]["experiment_id"] == parent


def test_the_development_guard_stops_a_run_that_reaches_a_protected_date():
    dates = pd.bdate_range("2020-01-01", periods=300)
    plan = type("P", (), {"dates": dates, "bounds": {"validation": (100, 200)}})()
    ok = type("T", (), {"predictions": pd.Series(1.0, index=pd.MultiIndex.from_product([dates[100:200], ["A"]]))})()
    rc.guard_development(ok, plan)
    leaky = type("T", (), {"predictions": pd.Series(1.0, index=pd.MultiIndex.from_product([dates[100:201], ["A"]]))})()
    with pytest.raises(rc.HoldoutAccessError, match="protected outcomes were reached"):
        rc.guard_development(leaky, plan)


def test_default_studies_and_names_are_validated():
    cfg = experiment_config(BASE)
    assert rc.default_study_id(cfg) == "u-etf" and rc.study_of(cfg, "mine") == "mine" and rc.study_of({**cfg, "study": "from-config"}) == "from-config"
    assert rc.default_study_id({**cfg, "dataset": {**cfg["dataset"], "universe": ["SPY", "QQQ"]}}) == rc.default_study_id({**cfg, "dataset": {**cfg["dataset"], "universe": ["qqq", "spy"]}})
    with pytest.raises(rc.ControlError):
        rc.study_of(cfg, "bad name!")
    assert ds_mod.UNIVERSES and np.isfinite(1.0)
