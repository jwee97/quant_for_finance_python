"""A search trial or experiment whose process died stays 'running' in the research record for ever unless somebody closes it. ``reconcile_interrupted`` ends such operations as failed-with-interrupted, never as
done, and they keep counting as looks: dropping an unfinished try would flatter the search."""

from __future__ import annotations

import pytest

from src.ml import research_controls as rc
from src.ml.registry import Registry


@pytest.fixture()
def registry(tmp_path):
    return Registry(tmp_path / "registry")


def states(registry, study):
    return {o["op_key"]: o for o in rc._latest(registry, [study])}


def test_a_trial_left_running_by_a_dead_process_is_closed_as_interrupted_and_still_counted(registry):
    study = "study-a"
    done = rc.begin(registry, "experiment", study, op_key="t-done")
    done.done()
    rc._LIVE.discard("t-lost")
    lost = rc.begin(registry, "experiment", study, op_key="t-lost")                    # started and never finished: the process is "killed" here
    rc._LIVE.discard("t-lost")                                                         # it is not live in this process any more
    before = rc.raw_looks(registry, study)
    assert states(registry, study)["t-lost"]["state"] == "running" and before == 2
    assert rc.reconcile_interrupted(registry, older_than=3600) == []                    # too recent: it may belong to another process that is still working
    closed = rc.reconcile_interrupted(registry, older_than=0)
    assert closed == ["t-lost"]
    after = states(registry, study)
    assert after["t-lost"]["state"] == "failed" and after["t-lost"]["detail"]["interrupted"] is True
    assert after["t-done"]["state"] == "done"
    assert rc.raw_looks(registry, study) == before == 2                                 # still a look
    assert rc.reconcile_interrupted(registry, older_than=0) == []                       # idempotent


def test_an_operation_that_is_live_in_this_process_is_never_closed(registry):
    op = rc.begin(registry, "experiment", "study-b", op_key="t-live")
    assert "t-live" in rc._LIVE
    assert rc.reconcile_interrupted(registry, older_than=0) == []
    op.done()
    assert "t-live" not in rc._LIVE and states(registry, "study-b")["t-live"]["state"] == "done"


def test_a_resumed_operation_continues_without_being_counted_twice(registry):
    study = "study-c"
    rc.begin(registry, "experiment", study, op_key="t-1")
    rc._LIVE.discard("t-1")
    rc.reconcile_interrupted(registry, older_than=0)
    with pytest.raises(rc.DuplicateOperation):
        rc.begin(registry, "experiment", study, op_key="t-1")                          # a retry of the same key must say it resumes
    assert rc.raw_looks(registry, study) == 1
