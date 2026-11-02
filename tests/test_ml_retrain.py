"""Release 8, P3-02: retraining triggers, cooldown, minimum data, nested validation and rollback.

A drift alert must be a proposal before it is anything else. The tests walk the procedure with a registry and a trainer they control: duplicate triggers are linked and not queued, a cooldown and a data
floor suppress, nothing trains without a person's approval unless the policy says so, a retraining that read the protected segments or is demonstrably worse is rejected, a failure leaves the incumbent
in force and ends in a suspension that only a documented recovery lifts, and a completed retraining is never a promotion.
"""

import pytest

from src.ml import governance as gv, registry as rg, retrain as rt

DAY = 86400.0
POLICY = rt.RetrainPolicy("weekly-review", cooldown_days=10, min_new_rows=50, min_train_days=300, max_failures=2)
AUTO = rt.RetrainPolicy("auto", cooldown_days=10, min_new_rows=50, min_train_days=300, autonomous=True, max_failures=2)


@pytest.fixture
def reg(tmp_path):
    return rg.Registry(tmp_path / "ml")


def trig(reg, policy=POLICY, kind="drift_alert", rows=100, days=800, now=0.0, subject="exp-1"):
    return rt.trigger(reg, subject, policy, kind=kind, detail={"seq": 7}, rows_since_fit=rows, train_days=days, actor="monitor", now=now)


def good_trainer(subject):
    return {"challenger_id": "exp-2", "holdout_status": "development"}


def ok_validator(incumbent, challenger):
    return {"eligibility": "applied", "difference": {"interval": [-0.1, 0.4], "value": 0.15}}


# ------------------------------------------------------------------------------------------------------------------ policy
def test_a_policy_names_its_triggers_and_its_limits():
    assert POLICY.hash() == rt.RetrainPolicy("weekly-review", cooldown_days=10, min_new_rows=50, min_train_days=300, max_failures=2).hash() != AUTO.hash()
    assert POLICY.autonomous is False and POLICY.require_nested_validation is True, "autonomous retraining is off and a nested validation is required unless a policy says otherwise"
    for kw in ({"name": ""}, {"name": "x", "triggers": ("vibes",)}, {"name": "x", "triggers": ()}, {"name": "x", "cooldown_days": -1}, {"name": "x", "max_failures": 0}, {"name": "x", "min_new_rows": 0}):
        with pytest.raises(rt.RetrainError):
            rt.RetrainPolicy(**kw)


# ------------------------------------------------------------------------------------------------------------------ triggers become proposals or are suppressed with a reason
def test_an_alert_is_a_proposal_and_every_other_trigger_is_suppressed_with_its_reason_and_recorded(reg):
    first = trig(reg)
    assert first["outcome"] == "proposed" and first["requires_approval"] is True and "nothing is trained" in first["reason"]
    dup = trig(reg, now=DAY)
    assert dup["outcome"] == "suppressed" and dup["reason"].startswith("duplicate") and str(first["proposal"]) in dup["reason"], "a second alert while a proposal is open is the same proposal, not a second one"
    assert rt.status(reg, "exp-1")["open_proposal"] == first["proposal"]
    assert trig(reg, kind="schedule")["reason"] == "the policy 'weekly-review' does not enable 'schedule' triggers"
    log = rt.events(reg, "exp-1")
    assert [e["event"] for e in log] == ["proposal", "suppressed", "suppressed"] and log[1]["duplicate_of"] == first["proposal"] and log[0]["policy_hash"] == POLICY.hash()
    with pytest.raises(rt.RetrainError, match="a trigger is one of"):
        trig(reg, kind="vibes")


def test_a_cooldown_a_data_floor_and_a_short_history_each_suppress(reg):
    p = trig(reg, now=0.0)["proposal"]
    rt.reject(reg, "exp-1", p, actor="alice", note="not now", now=1.0)
    cool = trig(reg, now=5 * DAY)
    assert cool["outcome"] == "suppressed" and cool["reason"].startswith("cooldown") and "5.0 days ago" in cool["reason"]
    assert trig(reg, now=11 * DAY, rows=10)["reason"] == "only 10 new dated rows since the last fit; the policy needs 50"
    assert trig(reg, now=11 * DAY, days=100)["reason"] == "100 days of history; the policy needs 300 to retrain at all"
    assert trig(reg, now=11 * DAY)["outcome"] == "proposed", "after the cooldown, with the data, a trigger is a proposal again"


# ------------------------------------------------------------------------------------------------------------------ nothing trains without approval, and a challenger is never a promotion
def test_nothing_is_trained_without_a_persons_approval_unless_the_policy_is_autonomous_and_a_challenger_is_never_promoted(reg):
    p = trig(reg)["proposal"]
    calls = []

    def trainer(subject):
        calls.append(subject)
        return good_trainer(subject)

    with pytest.raises(rt.RetrainError, match="autonomous retraining is off"):
        rt.start(reg, "exp-1", p, POLICY, incumbent_id="exp-0", trainer=trainer, validator=ok_validator)
    assert calls == []
    with pytest.raises(rt.RetrainError, match="names who approves and why"):
        rt.approve(reg, "exp-1", p, actor="alice", note=" ")
    with pytest.raises(rt.RetrainError, match="not the open proposal"):
        rt.approve(reg, "exp-1", p + 99, actor="alice", note="ok")
    rt.approve(reg, "exp-1", p, actor="alice", note="drift is real")
    out = rt.start(reg, "exp-1", p, POLICY, incumbent_id="exp-0", trainer=trainer, validator=ok_validator)
    assert out["outcome"] == "completed" and out["promoted"] is False and out["challenger_id"] == "exp-2" and calls == ["exp-1"]
    done = rt.events(reg, "exp-1")[-1]
    assert done["stage"] == "candidate" and done["promoted"] is False and "recorded human approval" in done["promotion"] and rt.status(reg, "exp-1")["open_proposal"] is None
    q = trig(reg, AUTO, subject="exp-9")["proposal"]
    assert rt.start(reg, "exp-9", q, AUTO, incumbent_id="exp-8", trainer=good_trainer, validator=ok_validator)["outcome"] == "completed", "an autonomous policy needs no approval, and still promotes nothing"


def test_a_retraining_that_read_the_protected_segments_or_is_demonstrably_worse_is_rejected_and_the_incumbent_stays(reg):
    p = trig(reg)["proposal"]
    rt.approve(reg, "exp-1", p, actor="alice", note="go")
    leaky = rt.start(reg, "exp-1", p, POLICY, incumbent_id="exp-0", trainer=lambda s: {"challenger_id": "exp-2", "holdout_status": "exposed_exploratory"}, validator=ok_validator)
    assert leaky["outcome"] == "failed" and "did not stay in development mode" in leaky["error"] and leaky["incumbent_kept"] is True
    unknown = trig(reg, now=20 * DAY)["proposal"]
    rt.approve(reg, "exp-1", unknown, actor="alice", note="again")
    assert rt.start(reg, "exp-1", unknown, POLICY, incumbent_id="exp-0", trainer=lambda s: {"challenger_id": "exp-2"}, validator=ok_validator)["error"].endswith("not recorded): a retraining cannot read the protected segments"), (
        "a challenger whose holdout status is unknown is treated as having consumed the segments")
    worse = trig(reg, now=40 * DAY)["proposal"]
    rt.approve(reg, "exp-1", worse, actor="alice", note="third")
    out = rt.start(reg, "exp-1", worse, AUTO, incumbent_id="exp-0", trainer=good_trainer, validator=lambda a, b: {"eligibility": "applied", "difference": {"interval": [-0.6, -0.1], "value": -0.35}})
    assert out["outcome"] == "rejected" and out["promoted"] is False and "worse" in out["reason"]
    last = rt.events(reg, "exp-1")[-1]
    assert last["event"] == "rejected" and last["challenger"] == "exp-2" and rt.status(reg, "exp-1")["consecutive_failures"] == 2, "a rejected challenger is not a failure; the two earlier ones were"


def test_a_validation_that_cannot_be_estimated_or_is_missing_fails_the_retraining_rather_than_passing_it(reg):
    p = trig(reg)["proposal"]
    rt.approve(reg, "exp-1", p, actor="a", note="go")
    none = rt.start(reg, "exp-1", p, POLICY, incumbent_id="exp-0", trainer=good_trainer, validator=None)
    assert none["outcome"] == "failed" and "requires a nested validation" in none["error"]
    q = trig(reg, now=20 * DAY)["proposal"]
    rt.approve(reg, "exp-1", q, actor="a", note="go")
    bad = rt.start(reg, "exp-1", q, POLICY, incumbent_id="exp-0", trainer=good_trainer, validator=lambda a, b: {"eligibility": "insufficient", "reason": "constant difference"})
    assert bad["outcome"] == "failed" and "could not be estimated" in bad["error"] and "constant difference" in bad["error"]
    skip = rt.RetrainPolicy("no-nested", require_nested_validation=False, cooldown_days=0, min_new_rows=1, min_train_days=1, autonomous=True)
    r = trig(reg, skip, subject="exp-5", rows=5, days=5)["proposal"]
    assert rt.start(reg, "exp-5", r, skip, incumbent_id="exp-0", trainer=good_trainer)["outcome"] == "completed", "a policy may waive the nested validation, and the record carries the policy hash that did"


# ------------------------------------------------------------------------------------------------------------------ failure, suspension and recovery
def test_repeated_failures_suspend_the_trigger_until_a_person_documents_a_recovery(reg):
    def broken(subject):
        raise RuntimeError("solver diverged")

    for i in range(2):
        p = trig(reg, AUTO, now=i * 20 * DAY)["proposal"]
        out = rt.start(reg, "exp-1", p, AUTO, incumbent_id="exp-0", trainer=broken, validator=ok_validator, now=i * 20 * DAY)
        assert out["outcome"] == "failed" and out["incumbent_kept"] is True and "solver diverged" in out["error"] and out["consecutive_failures"] == i + 1
    assert out["suspended"] is True and rt.status(reg, "exp-1")["suspended"] is True
    blocked = trig(reg, AUTO, now=60 * DAY)
    assert blocked["outcome"] == "suppressed" and "suspended after 2 consecutive failed retrainings" in blocked["reason"]
    with pytest.raises(rt.RetrainError, match="names who decided it"):
        rt.recover(reg, "exp-1", actor="alice", note="")
    rt.recover(reg, "exp-1", actor="alice", note="the solver setting was fixed", now=61 * DAY)
    assert rt.status(reg, "exp-1")["suspended"] is False and rt.status(reg, "exp-1")["consecutive_failures"] == 0
    again = trig(reg, AUTO, now=62 * DAY)
    assert again["outcome"] == "proposed", "recovery lets the next trigger through"
    assert rt.start(reg, "exp-1", again["proposal"], AUTO, incumbent_id="exp-0", trainer=good_trainer, validator=ok_validator, now=62 * DAY)["outcome"] == "completed"
    assert [e["event"] for e in rt.events(reg, "exp-1")].count("failed") == 2, "the failures stay in the history after the recovery"
    with pytest.raises(rt.RetrainError, match="is not suspended"):
        rt.recover(reg, "exp-1", actor="alice", note="no need")


def test_a_rollback_is_recorded_and_delegated_to_governance(reg, monkeypatch):
    seen = {}
    monkeypatch.setattr(gv, "rollback", lambda registry, a, b, reason, actor: seen.update(args=(a, b, reason, actor)) or {"rolled_back": True})
    out = rt.rollback(reg, "exp-1", "ver-2", "ver-1", reason="the challenger decayed in the first month", actor="alice")
    assert out["outcome"] == "rolled_back" and seen["args"] == ("ver-2", "ver-1", "the challenger decayed in the first month", "alice")
    assert rt.events(reg, "exp-1")[-1]["event"] == "rollback"
    with pytest.raises(rt.RetrainError, match="names who decided it and why"):
        rt.rollback(reg, "exp-1", "ver-2", "ver-1", reason=" ", actor="alice")
