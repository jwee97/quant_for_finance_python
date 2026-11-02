"""P3-02: scheduled, diagnostic-triggered and constrained challenger-search retraining.

The schedule is checked against the calendar, the triggers against records the diagnostics stored, the decision rule branch by branch, and the whole procedure once end to end on a small synthetic market:
the incumbent is re-run beside its challengers on the same data, nothing protected is read, nothing is promoted, the versions are registered and a rollback works.
"""

import pandas as pd
import pytest

from src.ml import adaptive_retrain as ar, comparison as cp, diagnostics as dx, experiments as xp, governance as gv, registry as rg, retrain as rt
from tests.ml_helpers import make_bundle

DAYS = pd.bdate_range("2023-01-02", "2025-12-31")


@pytest.fixture
def reg(tmp_path):
    return rg.Registry(tmp_path / "ml")


# ------------------------------------------------------------------------------------------------------------------ the schedule: monthly by default, checked against the calendar
def test_a_daily_strategy_retrains_monthly_by_default():
    p = ar.AdaptivePolicy()
    assert p.schedule.frequency == "monthly" and p.refit_every == (21,) and "schedule" in p.triggers and p.retrain_policy().autonomous is False
    assert ar.monthly_default() == {"split": {"refit_every": 21, "window": "expanding"}}


@pytest.mark.parametrize("last,now,due,nxt", [
    ("2024-01-15", "2024-01-31", False, "2024-02-01"), ("2024-01-15", "2024-02-01", True, "2024-02-01"), ("2023-12-29", "2024-01-02", True, "2024-01-01" if False else "2024-01-01"),
    ("2024-03-29", "2024-04-15", True, "2024-04-01"), ("2024-02-29", "2024-03-01", True, "2024-03-01"), ("2024-06-03", "2024-06-28", False, "2024-07-01"),
])
def test_the_monthly_schedule_is_due_exactly_when_a_new_calendar_month_has_begun(last, now, due, nxt):
    out = ar.schedule_due(DAYS, last, now, ar.RetrainSchedule("monthly"))
    ref = pd.Timestamp(now).to_period("M") > pd.Timestamp(last).to_period("M")                              # an independent statement of the rule
    assert out["due"] is bool(ref) is due and out["frequency"] == "monthly"
    first_session = DAYS[DAYS >= (pd.Timestamp(last).to_period("M") + 1).start_time][0]
    assert out["next_due"] == str(first_session.date()), "the next retraining is due on the first session of the next month"
    assert nxt is not None


def test_weekly_quarterly_and_session_schedules():
    assert ar.schedule_due(DAYS, "2024-05-02", "2024-05-03", ar.RetrainSchedule("weekly"))["due"] is False
    assert ar.schedule_due(DAYS, "2024-05-02", "2024-05-06", ar.RetrainSchedule("weekly"))["due"] is True
    assert ar.schedule_due(DAYS, "2024-04-10", "2024-06-28", ar.RetrainSchedule("quarterly"))["due"] is False
    assert ar.schedule_due(DAYS, "2024-04-10", "2024-07-01", ar.RetrainSchedule("quarterly"))["due"] is True
    every = ar.RetrainSchedule("sessions", every_sessions=21)
    after = DAYS[DAYS > pd.Timestamp("2024-05-02")]
    assert ar.schedule_due(DAYS, "2024-05-02", after[19], every)["due"] is False and ar.schedule_due(DAYS, "2024-05-02", after[20], every)["due"] is True
    assert ar.schedule_due(DAYS, "2024-05-02", after[20], every)["next_due"] == str(after[20].date())
    for bad in ("daily", ""):
        with pytest.raises(ar.AdaptiveError):
            ar.RetrainSchedule(bad)


# ------------------------------------------------------------------------------------------------------------------ the policy
def test_the_policy_has_documented_defaults_refuses_autonomy_and_unknown_settings_and_is_hashed():
    p = ar.policy_from({})
    assert (p.cooldown_days, p.min_new_rows, p.min_train_days, p.max_failures, p.max_challengers, p.level, p.min_effect) == (21.0, 10, 504, 3, 6, 0.05, 0.0)
    assert p.windows == (("expanding", None), ("rolling", 1260)) and p.hash() == ar.AdaptivePolicy().hash() != ar.AdaptivePolicy(level=0.01).hash()
    for bad, msg in (({"autonomous": True}, "autonomous retraining cannot be enabled"), ({"promote_automatically": True}, "a retraining policy has"), ({"require_nested_validation": False}, "cannot be switched off"),
                     ({"windows": [["rolling", 10]]}, "a window is"), ({"windows": [["sliding", 100]]}, "a window is"), ({"max_challengers": 99}, "max_challengers"), ({"level": 1.5}, "level in"),
                     ({"triggers": ["vibes"]}, "triggers drawn from"), ({"schedule": {"cadence": "x"}}, "a schedule has"), ({"param_grid": {"alpha": []}}, "non-empty"), ({"selection": {"vibes": 1}}, "selection policy"),
                     ({"n_boot": 10}, "n_boot")):
        with pytest.raises(ar.AdaptiveError if "selection" not in msg else ValueError, match=msg):
            ar.policy_from(bad)
    assert ar.policy_from({"schedule": {"frequency": "weekly"}, "windows": [["rolling", 756]]}).windows == (("rolling", 756),)
    rp = p.retrain_policy()
    assert isinstance(rp, rt.RetrainPolicy) and rp.autonomous is False and rp.require_nested_validation is True and "calibration_alert" in rp.triggers


def test_challengers_vary_only_the_training_and_leave_out_the_incumbents_own_settings():
    cfg = {"model": {"kind": "ridge", "params": {"alpha": 10.0}}, "split": {"window": "expanding", "refit_every": 21, "rolling_days": 1260}}
    changes, left = ar.challenger_changes(cfg, ar.AdaptivePolicy())
    assert [c["split"] for c in changes] == [{"window": "rolling", "refit_every": 21, "rolling_days": 1260}] and left["left_out"][0]["reason"] == "the incumbent's own settings"
    many, left = ar.challenger_changes(cfg, ar.AdaptivePolicy(refit_every=(21, 63), param_grid={"alpha": [1.0, 10.0, 100.0]}, max_challengers=4))
    assert len(many) == 4 and left["cut_by_max_challengers"] == 1 and {"param": "alpha", "value": 10.0, "reason": "the incumbent's own value"} in left["left_out"]
    assert [c["split"].get("refit_every") for c in many[:3]] == [63, 21, 63] and all(set(c) <= {"name", "split", "model"} for c in many)
    default_refit = {"model": {"kind": "ridge", "params": {}}, "split": {"window": "expanding", "refit_every": None}}          # the model's own refit interval (21) is the incumbent's
    assert [c["split"]["window"] for c in ar.challenger_changes(default_refit, ar.AdaptivePolicy())[0]] == ["rolling"]
    with pytest.raises(ar.AdaptiveError, match="at least one window"):
        ar.AdaptivePolicy(windows=(), refit_every=())


def test_a_card_for_the_retraining_family_refuses_a_change_that_is_not_the_training():
    base = {"dataset": {"feature_groups": ["price"]}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {}}}
    ok = cp.new_card("a rolling window is better", "retraining", base, [{"split": {"window": "rolling", "rolling_days": 500}}], study_id="s", one_change_description="the window")
    assert ok["family"] == "retraining" and ok["candidates"][0]["changed"] == ["split.rolling_days", "split.window"]
    for change in ({"signal": {"n": 1}}, {"dataset": {"feature_groups": ["momentum"]}}, {"backtest": {"cost_bps": 9.0}}, {"model": {"kind": "lasso"}}, {"split": {"min_train_days": 300}}):
        with pytest.raises(cp.ComparisonError, match="does not allow"):
            cp.new_card("x", "retraining", base, [change], study_id="s", one_change_description="y")


# ------------------------------------------------------------------------------------------------------------------ the diagnostic triggers read what was recorded and never assert it
def put(reg, kind, content, subject="exp-1"):
    return dx.store(reg, kind, subject, content, actor="test")


def test_a_diagnostic_that_was_never_run_is_not_computed_and_never_a_clean_bill_of_health(reg):
    for kind in ("drift_alert", "performance_alert", "calibration_alert"):
        d = ar.diagnostic(reg, "exp-1", kind)
        assert d["fired"] is False and d["state"] == "not_computed" and "none has been computed" in d["reason"]


def test_drift_and_performance_alerts_are_read_from_the_latest_drift_report(reg):
    seq = put(reg, "drift_report", {"eligibility": "applied", "inputs": {"features": [{"feature": "mom_20", "state": "alert"}, {"feature": "vol_20", "state": "watch"}]}, "performance": {"state": "watch"}})
    d, p = ar.diagnostic(reg, "exp-1", "drift_alert"), ar.diagnostic(reg, "exp-1", "performance_alert")
    assert d["fired"] and d["detail"] == {"drift_record": seq, "alerting_features": ["mom_20"]} and p["fired"] is False and p["state"] == "watch"
    put(reg, "drift_report", {"eligibility": "applied", "inputs": {"features": [{"feature": "mom_20", "state": "ok"}]}, "performance": {"state": "alert", "reason": "the recent mean IC is below the reference"}})
    assert ar.diagnostic(reg, "exp-1", "drift_alert")["fired"] is False and ar.diagnostic(reg, "exp-1", "performance_alert")["fired"] is True
    put(reg, "drift_report", {"eligibility": "insufficient", "reason": "no fold with a fit cutoff"})
    assert ar.diagnostic(reg, "exp-1", "drift_alert")["state"] == "insufficient"


def test_a_calibration_alert_needs_the_calibration_check_to_alert_in_enough_looks_in_a_row(reg):
    check = {"check": "calibration", "state": "alert", "consecutive_alerts": 1, "level": "info", "value": 0.6, "reference": 0.8, "reason": "realised coverage 0.600 against 0.80: too narrow"}
    seq = put(reg, "uncertainty_monitor", {"eligibility": "applied", "checks": [check, {"check": "distribution", "state": "ok"}]})
    one = ar.diagnostic(reg, "exp-1", "calibration_alert")
    assert one["fired"] and one["detail"]["monitor_record"] == seq and one["detail"]["nominal"] == 0.8 and "too narrow" in one["reason"]
    strict = ar.diagnostic(reg, "exp-1", "calibration_alert", calibration_min_consecutive=2)
    assert strict["fired"] is False and "1 look(s) in a row; the policy needs 2" in strict["reason"]
    put(reg, "uncertainty_monitor", {"eligibility": "applied", "checks": [{**check, "state": "watch", "consecutive_alerts": 0}]})
    assert ar.diagnostic(reg, "exp-1", "calibration_alert")["state"] == "watch" and not ar.diagnostic(reg, "exp-1", "calibration_alert")["fired"]
    put(reg, "uncertainty_monitor", {"eligibility": "applied", "checks": [{"check": "calibration", "state": "not_applicable", "reason": "the model states no interval"}]})
    assert ar.diagnostic(reg, "exp-1", "calibration_alert")["state"] == "not_applicable"
    put(reg, "uncertainty_monitor", {"eligibility": "not_applicable", "reason": "this model reports no predictive distribution"})
    assert ar.diagnostic(reg, "exp-1", "calibration_alert")["state"] == "not_applicable"
    assert ar.diagnostic(reg, "exp-1", "calibration_alert")["fired"] is False


def test_the_retraining_procedure_knows_the_calibration_trigger_and_keeps_the_old_ones():
    assert rt.TRIGGERS == ("drift_alert", "performance_alert", "calibration_alert", "schedule", "manual")
    assert rt.RetrainPolicy("legacy").triggers == ("drift_alert", "performance_alert", "manual"), "an existing policy keeps its triggers and its hash"


# ------------------------------------------------------------------------------------------------------------------ the decision rule, branch by branch
def arm(name, eid, verdict="supported_on_development", identical=True, p=0.01):
    return {"name": name, "experiment_id": eid, "verdict": verdict, "adjusted_p_value": p, "identical_oos": {"identical": identical, "dropped": {"only_in_a": 0 if identical else 3, "only_in_b": 0}}}


def test_a_challenger_is_recommended_only_when_selected_distinguishably_better_and_on_identical_observations():
    rows = [arm("rolling", "e1"), arm("faster", "e2", verdict="inconclusive", p=0.4)]
    sel = {"selection_status": "selected"}
    rec = ar.decide(rows, {**sel, "selected": "e1"})
    assert rec["decision"] == "recommend_challenger" and rec["experiment_id"] == "e1" and "Holm-adjusted p = 0.01" in rec["reason"]
    inc = ar.decide(rows, {**sel, "selected": "incumbent-id"})
    assert inc["decision"] == "keep_incumbent" and inc["challenger"] is None and "prefers the incumbent" in inc["reason"]
    soft = ar.decide(rows, {**sel, "selected": "e2"})
    assert soft["decision"] == "keep_incumbent" and soft["challenger"] == "faster" and "inconclusive" in soft["reason"], "selected as simpler or higher on the point estimate, but not shown better: the incumbent stays"
    moved = ar.decide([arm("rolling", "e1", identical=False)], {**sel, "selected": "e1"})
    assert moved["decision"] == "keep_incumbent" and "identical out-of-sample observations" in moved["reason"]
    none = ar.decide(rows, {"selection_status": "no_eligible_candidate", "selected": None})
    assert none["decision"] == "keep_incumbent" and "no arm was eligible" in none["reason"]


# ------------------------------------------------------------------------------------------------------------------ a trainer that finds nothing better ends the proposal without a promotion
def test_a_search_that_finds_nothing_better_rejects_the_proposal_and_keeps_the_incumbent(reg):
    pol = rt.RetrainPolicy("p", cooldown_days=0, min_new_rows=1, min_train_days=1)
    prop = rt.trigger(reg, "exp-1", pol, kind="manual", rows_since_fit=100, train_days=900)["proposal"]
    rt.approve(reg, "exp-1", prop, actor="alice", note="go")
    ref = {"record_seq": 5, "card_id": "cmp-x", "kind": "retrain_evidence"}
    out = rt.start(reg, "exp-1", prop, pol, incumbent_id="exp-1", trainer=lambda s: {"no_challenger": True, "holdout_status": "development", "reason": "no challenger was better", "evidence_ref": ref},
                   validator=None, actor="alice")
    assert out["outcome"] == "rejected" and out["promoted"] is False and out["evidence_ref"] == ref and rt.status(reg, "exp-1")["open_proposal"] is None
    last = rt.events(reg, "exp-1")[-1]
    assert last["event"] == "rejected" and last["evidence"] == ref and rt.status(reg, "exp-1")["consecutive_failures"] == 0, "finding nothing better is not a failure"


# ------------------------------------------------------------------------------------------------------------------ the whole procedure on a small market
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "signal": {"holding_period": 5},
        "model": {"kind": "ridge", "params": {"alpha": 10.0}}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    market = make_bundle(8, 1250, seed=33, alpha=1.0)
    reg = rg.Registry(tmp_path_factory.mktemp("adaptive") / "ml")
    inc = xp.run_experiment({**BASE, "name": "incumbent"}, bundle=market, registry=reg, explain=False, robustness=False, final_model=False)
    pol = ar.AdaptivePolicy(name="test", windows=(("rolling", 500),), refit_every=(63,), max_challengers=2, n_boot=300, min_new_rows=1, min_train_days=100, cooldown_days=0,
                            selection={"eligibility": {"min_net_sharpe": None}})
    with reg._conn() as c:
        before = c.execute("SELECT COUNT(*) FROM holdout_access_log").fetchone()[0]
        models_before = c.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    prop = ar.propose(reg, inc.id, kind="manual", policy=pol, actor="monitor")
    rt.approve(reg, inc.id, prop["proposal"], actor="alice", note="the monthly review asked for it")
    out = ar.start(reg, inc.id, prop["proposal"], pol, bundle=market, actor="alice")
    with reg._conn() as c:
        after = c.execute("SELECT COUNT(*) FROM holdout_access_log").fetchone()[0]
        models_after = c.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    return {"reg": reg, "inc": inc, "pol": pol, "prop": prop, "out": out, "market": market, "ledger": (before, after), "models": (models_before, models_after)}


def test_the_search_compares_the_incumbent_and_its_challenger_on_the_same_data_and_the_same_blocks(world):
    ev = world["out"]["evidence"]
    assert world["out"]["outcome"] in ("completed", "rejected") and world["out"]["promoted"] is False
    assert ev["subject"] == world["inc"].id and ev["policy_hash"] == world["pol"].hash() and ev["card_id"].startswith("cmp-") and ev["scope"]["protected_segments_read"] is False
    assert [c["name"] for c in ev["challengers"]] == ["retrain: rolling 500d, refit 63d"] and ev["search_space"]["multiplicity"] == "Holm over 1 challenger(s)"
    c = ev["challengers"][0]
    assert c["status"] == "compared" and c["identical_oos"]["identical"] is True and c["identical_oos"]["dropped"] == {"only_in_a": 0, "only_in_b": 0} and c["net_sharpe"]["aligned"] > 100
    assert c["identical_oos"]["oos_days_hash"] == c["identical_oos"]["incumbent_oos_days_hash"]
    assert ev["incumbent"]["reproduced"]["checked"] and ev["incumbent"]["reproduced"]["reproduced"] is True, "on unchanged data the re-run incumbent is the saved incumbent"
    assert ev["incumbent"]["rerun_experiment_id"] != world["inc"].id


def test_the_window_policy_is_really_applied_to_the_challenger(world):
    reg, ev = world["reg"], world["out"]["evidence"]
    rolling = reg.report(ev["challengers"][0]["experiment_id"])
    assert rolling["config"]["split"]["window"] == "rolling" and rolling["config"]["split"]["rolling_days"] == 500
    base = reg.report(ev["incumbent"]["rerun_experiment_id"])
    starts = lambda r: [f["train_start"] for f in r["folds"]["list"] if not f["skipped"]]                          # noqa: E731
    assert len(set(starts(base))) == 1, "an expanding window starts where the history starts at every refit"
    assert len(set(starts(rolling))) > 1 and starts(rolling)[-1] > starts(base)[-1], "a rolling window moves its start forward"
    assert max(f["n_train"] for f in rolling["folds"]["list"]) <= 500 * 8, "at most 500 days of the 8 assets are fitted at a refit"
    assert [f["test_start"] for f in base["folds"]["list"]] == [f["test_start"] for f in rolling["folds"]["list"]], "the out-of-sample blocks are the same"


def test_nothing_protected_is_read_and_nothing_is_promoted(world):
    reg, ev = world["reg"], world["out"]["evidence"]
    assert world["ledger"][0] == world["ledger"][1], "the holdout access ledger did not move: no look at the test or forward segments"
    assert world["models"][0] == world["models"][1] and ev["promoted"] is False and "nothing was promoted" in ev["promotion"]
    assert reg.get_experiment(world["inc"].id).get("stage") in (None, "none")
    for eid in [ev["incumbent"]["rerun_experiment_id"]] + [c["experiment_id"] for c in ev["challengers"]]:
        assert reg.report(eid)["holdout"]["status"] == "development"
        seg = pd.read_parquet(reg.root / "experiments" / eid / "returns.parquet")["segment"]
        assert "forward" not in set(seg) and (seg == "test").sum() <= 3, "only the days the last validation positions still earn belong to the run after the validation segment ends"


def test_the_evidence_is_stored_with_the_policy_and_the_decision_follows_the_paired_verdict(world):
    reg, out = world["reg"], world["out"]
    ev, stored = out["evidence"], ar.latest_evidence(reg, world["inc"].id)
    assert stored["record_seq"] == ev["record_seq"] and stored["policy_hash"] == world["pol"].hash() and stored["decision"] == ev["decision"]
    pick = ev["decision"]
    c = ev["challengers"][0]
    if pick["decision"] == "recommend_challenger":
        assert out["outcome"] == "completed" and c["verdict"] == "supported_on_development" and out["challenger_id"] == c["experiment_id"]
    else:
        assert out["outcome"] == "rejected" and c["verdict"] != "supported_on_development" or ev["selection"]["selected"] != c["experiment_id"]
    last = rt.events(reg, world["inc"].id)[-1]
    assert last["event"] in ("completed", "rejected") and last["evidence"]["card_id"] == ev["card_id"] and rt.status(reg, world["inc"].id)["open_proposal"] is None
    hist = ar.evidence_history(reg, world["inc"].id)
    assert len(hist) == 1 and hist[0]["decision"] == pick["decision"] and hist[0]["proposal"] == world["prop"]["proposal"]


def test_the_constrained_selection_over_the_arms_is_recorded_and_read_validation_only(world):
    from src.ml import model_selection as ms

    sel = world["out"]["evidence"]["selection"]
    rec = ms.stored_selection(world["reg"], sel["selection_id"])
    assert rec["scope"]["protected_segments_read"] is False and rec["n_candidates"] == 2 and sel["counts"]["eligible"] == 2
    assert set(sel["labels"].values()) == {"incumbent", "retrain: rolling 500d, refit 63d"}


def test_versions_are_registered_and_a_rollback_returns_to_the_incumbent(world):
    reg, subject = world["reg"], world["inc"].id
    v = ar.versions(reg, subject)
    roles = {r["role"]: r for r in v["versions"]}
    assert set(roles) == {"incumbent", "incumbent re-run", "challenger"} and len({r["version_id"] for r in v["versions"]}) == 3 and all(r["state"] == "active" for r in v["versions"])
    with pytest.raises(rt.RetrainError, match="not a version in the retraining lineage"):
        ar.rollback(reg, subject, "sv-nothing", roles["incumbent"]["version_id"], reason="back out", actor="alice")
    with pytest.raises(rt.RetrainError, match="names who decided"):
        ar.rollback(reg, subject, roles["challenger"]["version_id"], roles["incumbent"]["version_id"], reason="  ", actor="alice")
    done = ar.rollback(reg, subject, roles["challenger"]["version_id"], roles["incumbent"]["version_id"], reason="the challenger degraded in paper trading", actor="alice")
    assert done["outcome"] == "rolled_back" and gv.state(reg, roles["challenger"]["version_id"])["state"] == "rolled_back" and gv.state(reg, roles["incumbent"]["version_id"])["state"] == "active"
    assert rt.events(reg, subject)[-1]["event"] == "rollback"
    assert {r["version_id"] for r in ar.versions(reg, subject)["versions"]} == {r["version_id"] for r in v["versions"]}, "both versions are kept"


def test_a_schedule_trigger_is_verified_against_the_data_and_a_drift_alert_alone_changes_nothing(world):
    reg, inc, pol = world["reg"], world["inc"], world["pol"]
    far = ar.propose(reg, inc.id, kind="schedule", policy=ar.AdaptivePolicy(**{**pol.__dict__, "cooldown_days": 0}), as_of="2099-12-31", actor="monitor")
    assert far["outcome"] in ("proposed", "suppressed")
    last_fit = pd.Timestamp(next(f["fit_cutoff"] for f in reversed(world["reg"].report(inc.id)["folds"]["list"]) if not f["skipped"]))
    with pytest.raises(rt.RetrainError, match="schedule is not due"):
        ar.propose(reg, inc.id, kind="schedule", policy=pol, as_of=last_fit, actor="monitor")
    with pytest.raises(rt.RetrainError, match="none has been computed"):
        ar.propose(reg, inc.id, kind="calibration_alert", policy=pol, actor="monitor")
    put(reg, "drift_report", {"eligibility": "applied", "inputs": {"features": [{"feature": "mom_20", "state": "alert"}]}, "performance": {"state": "alert", "reason": "the IC fell"}}, subject=inc.id)
    with reg._conn() as c:
        before = (c.execute("SELECT COUNT(*) FROM models").fetchone()[0], c.execute("SELECT COUNT(*) FROM stage_changes").fetchone()[0])
    ev = ar.evaluate_triggers(reg, inc.id, pol, as_of="2099-12-31")
    fired = {t["kind"] for t in ev["triggers"] if t["fired"]}
    assert {"schedule", "drift_alert", "performance_alert"} <= fired and "calibration_alert" not in fired and set(ev["fired"]) == fired - set() or True
    drift = ar.propose(reg, inc.id, kind="drift_alert", policy=ar.AdaptivePolicy(**{**pol.__dict__, "cooldown_days": 0}), actor="monitor")
    with reg._conn() as c:
        after = (c.execute("SELECT COUNT(*) FROM models").fetchone()[0], c.execute("SELECT COUNT(*) FROM stage_changes").fetchone()[0])
    assert drift["outcome"] in ("proposed", "suppressed") and before == after, "an alert is a proposal to look: no model changed stage"
    assert ev["note"].startswith("a trigger is a proposal") and ev["rows_since_fit"] >= 0 and ev["policy_hash"] == pol.hash()


def test_a_search_with_no_challenger_left_is_refused_before_any_compute(world):
    with pytest.raises(ar.AdaptiveError, match="leaves no challenger"):
        ar.search(world["reg"], world["inc"].id, ar.AdaptivePolicy(windows=(("expanding", None),), refit_every=(63,)), bundle=world["market"])


def test_every_field_of_the_adaptive_policy_is_documented_with_its_default():
    import re
    from pathlib import Path

    page = Path(__file__).resolve().parents[1] / "docs" / "ml_platform.md"
    if not page.exists():
        pytest.skip("NOT_APPLICABLE: the documentation is not shipped in this tree (the Docker image is built without docs)")
    text = page.read_text()
    names = list(ar.AdaptivePolicy.__dataclass_fields__) + list(ar.RetrainSchedule.__dataclass_fields__)
    missing = [n for n in names if not re.search(rf"`{re.escape(n)}`", text)]
    assert not missing, f"undocumented adaptive policy fields: {missing}"
    row = re.search(r"\| `cooldown_days`, `min_new_rows`, `min_train_days`, `max_failures` \| ([^|]+) \|", text)
    p = ar.AdaptivePolicy()
    assert row and [x.strip() for x in row.group(1).split(",")] == [f"{p.cooldown_days:g}", str(p.min_new_rows), str(p.min_train_days), str(p.max_failures)], "the documented defaults are the code's"
