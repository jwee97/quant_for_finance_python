"""A run does not vanish when the page is reloaded or the app restarts: every job leaves a record on disk, a reloaded page can list its runs and read a finished one, a job whose process is gone is marked
as interrupted instead of running for ever, and a training goes on with nobody watching it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from src.webapp import server
from src.webapp.api import ApiError, Job
from src.webapp import jobs as jobs_module
from src.webapp.jobs import PROCESS, JobLog, _alive
from tests.ml_helpers import make_bundle
from tests.test_webapp import _http
from tests.test_webapp_assets_ml import _app, _safe, finished, wait
from tests.test_webapp_labs import _rich


def _record(**kw):
    return {"id": "0123456789ab", "label": "ml: ridge on etf", "kind": "ml", "status": "running", "message": "fold 3 of 9", "error": None, "created": time.time() - 60, "finished": None, "ref": None, **kw}


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


# ------------------------------------------------------------------------------------------------------------------ the log
def test_a_record_and_the_result_of_a_finished_job_are_kept_and_read_back(tmp_path):
    log = JobLog(tmp_path / "jobs")
    assert log.records() == [] and log.get("0123456789ab") is None and log.mark_interrupted() == 0, "an empty or missing folder is not an error"
    log.save(_record(status="done", message="finished", finished=time.time(), ref="20260101-120000-ab12cd"), {"id": "20260101-120000-ab12cd", "report": {"sharpe": 0.8}})
    got = log.get("0123456789ab")
    assert got["status"] == "done" and got["result"] == {"id": "20260101-120000-ab12cd", "report": {"sharpe": 0.8}} and got["ref"] == "20260101-120000-ab12cd" and got["elapsed"] >= 59
    assert [r["id"] for r in log.records()] == ["0123456789ab"] and log.records()[0]["pid"] == os.getpid() and log.records()[0]["has_result"] is True
    (tmp_path / "jobs" / "0123456789ab.result.json").unlink()
    again = log.get("0123456789ab")
    assert again["status"] == "done" and again["result"] is None and "result was not kept" in again["message"], "a lost result is said to be lost, not shown as empty"


def test_an_id_is_checked_before_a_path_is_built_from_it_and_damaged_files_are_skipped(tmp_path):
    log = JobLog(tmp_path / "jobs")
    log.save(_record())
    outside = tmp_path / "secret.json"
    outside.write_text(json.dumps(_record(id="ffffffffffff")))
    for bad in ("../secret", "..", "", "x" * 40, "0123456789AB", "../../etc/passwd", "0123456789ab/../../secret", None, 12):
        assert log.get(bad) is None, bad
    log.save({**_record(), "id": "../evil"})
    assert [p.name for p in (tmp_path / "jobs").iterdir()] == ["0123456789ab.json"], "a record with an unsafe id is not written"
    (tmp_path / "jobs" / "111111111111.json").write_text("{not json")
    (tmp_path / "jobs" / "222222222222.json").write_text(json.dumps([1, 2]))
    (tmp_path / "jobs" / "333333333333.json").write_text(json.dumps({"id": "not-an-id"}))
    assert [r["id"] for r in log.records()] == ["0123456789ab"] and log.get("111111111111") is None


def test_the_log_keeps_the_newest_jobs_and_never_raises_for_a_result_it_cannot_write(tmp_path):
    log = JobLog(tmp_path / "jobs", keep=5)
    for i in range(9):
        log.save(_record(id=f"{i:012x}", created=1000.0 + i, status="done", finished=1001.0 + i), {"n": i})
    assert [r["id"] for r in log.records()] == [f"{i:012x}" for i in (8, 7, 6, 5, 4)]
    assert sorted(p.name for p in (tmp_path / "jobs").iterdir()) == sorted([f"{i:012x}{ext}" for i in (4, 5, 6, 7, 8) for ext in (".json", ".result.json")]), "the results of the jobs that were dropped go with them"
    log.save(_record(id="aaaaaaaaaaaa", status="done"), {"bad": float("nan")})
    assert log.get("aaaaaaaaaaaa") is None, "a result that cannot be written as JSON is dropped, and the run is not stopped"
    blocked = JobLog(tmp_path / "file")
    (tmp_path / "file").write_text("a file where the folder should be")
    blocked.save(_record())
    assert blocked.records() == [] and blocked.get("0123456789ab") is None and blocked.mark_interrupted() == 0


@pytest.mark.skipif(os.name == "nt", reason="there is no way to ask whether a process exists on Windows")
def test_a_job_whose_process_is_gone_is_marked_interrupted_and_anothers_running_job_is_left_alone(tmp_path):
    log = JobLog(tmp_path / "jobs")
    folder = tmp_path / "jobs"
    folder.mkdir()
    other_app = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])      # another app, alive while this test runs (this process's parent cannot stand for it: in a container this is process 1)
    try:
        dead = _record(id="aaaaaaaaaaaa", pid=_dead_pid(), message="training fold 3 of 9")
        queued = _record(id="bbbbbbbbbbbb", status="queued", pid=_dead_pid(), message="waiting for a free worker")
        foreign = _record(id="cccccccccccc", pid=other_app.pid, process="another app", updated=time.time())   # alive and beating: another app's run, not ours to end
        own = _record(id="dddddddddddd", pid=os.getpid(), process=PROCESS)               # written by this very process (another app in it)
        done = _record(id="eeeeeeeeeeee", status="done", pid=_dead_pid(), message="finished")
        reused = _record(id="999999999999", pid=os.getpid(), process="an earlier run of the interpreter", message="training fold 5 of 9")   # a container restarts the app with the same process id (1)
        for r in (dead, queued, foreign, own, done, reused):
            (folder / f"{r['id']}.json").write_text(json.dumps(r))
        assert log.mark_interrupted() == 3
        status = {r["id"]: r["status"] for r in log.records()}
        assert status == {"aaaaaaaaaaaa": "interrupted", "bbbbbbbbbbbb": "interrupted", "cccccccccccc": "running", "dddddddddddd": "running", "eeeeeeeeeeee": "done", "999999999999": "interrupted"}, \
            "a record from an earlier run of the app is lost even when its process id is ours again"
        lost = log.get("aaaaaaaaaaaa")
        assert "the app stopped while this was running" in lost["message"] and "training fold 3 of 9" in lost["message"] and lost["finished"] is not None and lost["result"] is None
        assert log.mark_interrupted() == 0, "a second start finds nothing more to end"
        assert _alive(other_app.pid) and _alive(os.getpid()) and not _alive(_dead_pid()) and not _alive(0) and not _alive(-5)
    finally:
        other_app.kill()
        other_app.wait()


def test_a_progress_message_is_saved_at_most_every_couple_of_seconds_and_a_change_of_state_at_once(monkeypatch):
    saved = []
    job = Job("ml: ridge", "ml")
    job.sink = lambda j, final: saved.append((j.status, j.message, final))
    job.save()
    for i in range(200):
        job.message = f"step {i}"
    assert len(saved) <= 2, "a training that reports every few milliseconds does not write a file every few milliseconds"
    assert job.message == "step 199" and job.view()["message"] == "step 199"
    monkeypatch.setattr(Job, "MESSAGE_GAP", 0.0)
    before = len(saved)
    job.message = "again"
    assert len(saved) == before + 1
    job.result = {"id": "20260101-120000-ab12cd", "report": {}}
    assert job.ref == "20260101-120000-ab12cd" and job.record()["ref"] == "20260101-120000-ab12cd" and "result" not in job.record()
    other = Job("backtest", "backtest")
    other.result = {"metrics": {}}
    assert other.ref is None and other.record()["kind"] == "backtest"


# ------------------------------------------------------------------------------------------------------------------ the app
@pytest.fixture
def two_apps(tmp_path):
    """A factory of apps that share one folder: the second stands for the app after a restart."""
    def make():
        return _app(tmp_path / "prices", _rich, root=tmp_path)

    return make


def test_a_finished_job_can_be_read_after_the_app_restarts_and_the_list_keeps_what_was_lost(two_apps, tmp_path):
    app = two_apps()
    ok = app.submit_task("fine", lambda progress: (progress("halfway"), {"x": float("nan"), "n": np.int64(3), "id": "20260101-120000-ab12cd"})[1], "ml")
    bad = app.submit_task("refused", lambda progress: (_ for _ in ()).throw(ApiError("choose fewer assets")))
    assert wait(app, ok)["status"] == "done" and wait(app, bad)["status"] == "error"
    time.sleep(0.05)
    listing = _safe(app.jobs_list())
    assert [r["label"] for r in listing["jobs"]][:2] == ["refused", "fine"] and listing["active"] == 0 and all("result" not in r for r in listing["jobs"]), "the list is newest first and carries no results"
    assert {r["label"]: r["kind"] for r in listing["jobs"]} == {"fine": "ml", "refused": "task"} and listing["jobs"][1]["ref"] == "20260101-120000-ab12cd"

    # a job that was running when the app died (its process is gone)
    (tmp_path / "data" / "user" / "jobs" / "aaaaaaaaaaaa.json").write_text(json.dumps(_record(id="aaaaaaaaaaaa", pid=_dead_pid(), created=time.time() - 30)))
    restarted = two_apps()
    again = restarted.job(ok["job"])
    assert again["status"] == "done" and again["result"] == {"x": None, "n": 3, "id": "20260101-120000-ab12cd"} and again["label"] == "fine" and again["kind"] == "ml", "a finished run's result is still there after a restart"
    assert restarted.job(bad["job"])["status"] == "error" and restarted.job(bad["job"])["error"] == "choose fewer assets"
    lost = restarted.job("aaaaaaaaaaaa")
    assert lost["status"] == "interrupted" and "fold 3 of 9" in lost["message"]
    rows = {r["id"]: r for r in restarted.jobs_list()["jobs"]}
    assert set(rows) >= {ok["job"], bad["job"], "aaaaaaaaaaaa"} and rows["aaaaaaaaaaaa"]["status"] == "interrupted" and restarted.jobs_list()["active"] == 0
    for bad_id in ("nope", "../../etc/passwd", "ffffffffffff", ""):
        with pytest.raises(ApiError, match="unknown job"):
            restarted.job(bad_id)


def test_the_jobs_still_in_the_table_win_over_their_records_and_a_running_job_is_listed_as_active(two_apps):
    app = two_apps()
    gate, started = threading.Event(), threading.Event()

    def slow(progress):
        progress("training fold 2 of 9")
        started.set()
        gate.wait(30)
        return {"done": True}

    job = app.submit_task("ml: slow", slow, "ml")
    assert started.wait(10)
    rows = app.jobs_list()
    mine = next(r for r in rows["jobs"] if r["id"] == job["job"])
    assert rows["active"] == 1 and mine["status"] == "running" and mine["message"] == "training fold 2 of 9" and mine["kind"] == "ml" and mine["finished"] is None
    queued = app.submit_task("ml: waiting", lambda progress: {"ok": 1}, "ml")            # the single worker is busy: this one waits
    assert next(r for r in app.jobs_list()["jobs"] if r["id"] == queued["job"])["status"] == "queued" and app.jobs_list()["active"] == 2
    gate.set()
    assert wait(app, job)["status"] == "done" and wait(app, queued)["status"] == "done"
    assert app.jobs_list()["active"] == 0 and next(r for r in app.jobs_list()["jobs"] if r["id"] == job["job"])["finished"] is not None


def test_the_routes_need_the_token_and_serve_the_list_and_a_job_from_before_a_restart(two_apps):
    app = two_apps()
    done = finished(app, app.submit_task("fine", lambda progress: {"a": 1}))
    assert done == {"a": 1}
    first = list(app.jobs)[0]
    running = server.start(two_apps(), "127.0.0.1", 0)
    try:
        assert _http(running, "/api/jobs", token=False)[0] == 403
        status, body, _ = _http(running, "/api/jobs")
        listing = json.loads(body)
        assert status == 200 and [r["id"] for r in listing["jobs"]] == [first] and listing["active"] == 0
        status, body, _ = _http(running, f"/api/job/{first}")
        assert status == 200 and json.loads(body)["result"] == {"a": 1}, "the new app has never run this job and still knows it"
        assert _http(running, "/api/job/..%2f..%2fetc%2fpasswd")[0] == 400 and _http(running, "/api/job/ffffffffffff")[0] == 400
    finally:
        running.stop()


# ------------------------------------------------------------------------------------------------------------------ a training with nobody watching
def test_a_training_goes_on_with_no_page_asking_after_it_and_its_experiment_is_there_afterwards(tmp_path):
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    app = _app(tmp_path, lambda config: bundle, root=tmp_path)
    body = {"dataset": {"feature_groups": ["price", "momentum"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "ridge", "params": {"alpha": 30}}, "name": "left alone", "explain": False}
    started = app.extra_post["/api/ml/train"](body)
    job_id = started["job"]                                                                # ... and nothing ever calls /api/job/<id>: the tab was closed
    deadline = time.time() + 600
    while time.time() < deadline:
        rows = app.jobs_list()["jobs"]
        if rows and rows[0]["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    row = app.jobs_list()["jobs"][0]
    assert row["id"] == job_id and row["status"] == "done" and row["kind"] == "ml" and row["label"].startswith("ml: left alone") and row["finished"] is not None
    saved = app.extra_get["/api/ml/experiments"]({})["rows"]
    assert [r["name"] for r in saved] == ["left alone"] and row["ref"] == saved[0]["id"], "the experiment is in the registry, and the job says which one it made"
    reread = _app(tmp_path / "other", lambda config: bundle, root=tmp_path)               # the app restarted: the run and its experiment are still found
    assert reread.job(job_id)["status"] == "done" and reread.job(job_id)["result"]["id"] == saved[0]["id"] and reread.extra_get["/api/ml/experiments"]({})["rows"][0]["id"] == saved[0]["id"]


# ------------------------------------------------------------------------------------------------------------------ a crash must not leave a run "running"
def test_a_process_id_that_exists_again_is_not_taken_for_the_app_that_died(tmp_path, monkeypatch):
    """After a crash the numbers are reused: some other process has the id the dead app had. Only the app's own signs of life tell, so a record that stops beating is lost after STALE seconds."""
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        log = JobLog(tmp_path / "jobs")
        folder = tmp_path / "jobs"
        folder.mkdir()
        now = time.time()
        fresh = _record(id="aaaaaaaaaaaa", pid=other.pid, process="the app before the crash", updated=now - 5)
        silent = _record(id="bbbbbbbbbbbb", pid=other.pid, process="the app before the crash", updated=now - jobs_module.STALE - 5, message="training fold 4 of 9")
        for r in (fresh, silent):
            (folder / f"{r['id']}.json").write_text(json.dumps(r))
        assert not JobLog.gone(fresh) and JobLog.gone(silent)
        assert log.mark_interrupted() == 1
        assert {r["id"]: r["status"] for r in log.records()} == {"aaaaaaaaaaaa": "running", "bbbbbbbbbbbb": "interrupted"}, "a fresh record is left alone: its app may be right there"
        assert "fold 4 of 9" in log.get("bbbbbbbbbbbb")["message"]
        monkeypatch.setattr(jobs_module, "STALE", 0.01)
        time.sleep(0.05)
        assert log.mark_interrupted() == 1 and log.get("aaaaaaaaaaaa")["status"] == "interrupted", "and when it falls silent it is lost too"
    finally:
        other.kill()
        other.wait()


def test_a_beat_refreshes_a_running_record_and_never_brings_back_a_job_that_has_ended(tmp_path):
    log = JobLog(tmp_path / "jobs")
    log.save(_record(id="aaaaaaaaaaaa"))
    first = json.loads((tmp_path / "jobs" / "aaaaaaaaaaaa.json").read_text())["updated"]
    time.sleep(0.02)
    log.beat(["aaaaaaaaaaaa", "ffffffffffff", "../x"])                       # an unknown id and an unsafe one are ignored
    second = json.loads((tmp_path / "jobs" / "aaaaaaaaaaaa.json").read_text())["updated"]
    assert second > first
    log.save(_record(id="aaaaaaaaaaaa", status="done", finished=time.time()), {"ok": 1})
    ended = json.loads((tmp_path / "jobs" / "aaaaaaaaaaaa.json").read_text())
    log.beat(["aaaaaaaaaaaa"])
    after = json.loads((tmp_path / "jobs" / "aaaaaaaaaaaa.json").read_text())
    assert after == ended and after["status"] == "done", "a beat that read the record a moment earlier must not put a finished job back to running"
    elsewhere = _record(id="bbbbbbbbbbbb", process="another app", updated=1.0)
    (tmp_path / "jobs" / "bbbbbbbbbbbb.json").write_text(json.dumps(elsewhere))
    log.beat(["bbbbbbbbbbbb"])
    assert json.loads((tmp_path / "jobs" / "bbbbbbbbbbbb.json").read_text())["updated"] == 1.0, "only this process's own jobs are vouched for"


def test_the_app_beats_while_a_job_runs_and_stops_when_none_does(two_apps):
    app = two_apps()
    app.beat_every = 0.05
    gate, started = threading.Event(), threading.Event()

    def slow(progress):
        started.set()
        gate.wait(30)
        return {"done": True}

    job = app.submit_task("ml: slow", slow, "ml")
    assert started.wait(10)
    path = app.joblog.directory / f"{job['job']}.json"
    first = json.loads(path.read_text())["updated"]
    deadline = time.time() + 10
    while time.time() < deadline and json.loads(path.read_text())["updated"] <= first:
        time.sleep(0.05)
    assert json.loads(path.read_text())["updated"] > first, "the record is refreshed while the job runs, with no progress message at all"
    gate.set()
    assert wait(app, job)["status"] == "done"
    deadline = time.time() + 10
    while time.time() < deadline and app._beating:
        time.sleep(0.05)
    assert not app._beating, "the beat stops when nothing is queued or running"
    again = app.submit_task("ml: again", lambda progress: {"ok": 1}, "ml")           # and starts again with the next job
    assert wait(app, again)["status"] == "done"


def test_a_crashed_apps_run_is_listed_as_stopped_and_can_be_cleared(two_apps, tmp_path, monkeypatch):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])         # a process that has the id the crashed app had
    try:
        app = two_apps()
        monkeypatch.setattr(jobs_module, "STALE", 0.05)
        app.joblog.directory.mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "user" / "jobs" / "aaaaaaaaaaaa.json").write_text(json.dumps(_record(id="aaaaaaaaaaaa", pid=other.pid, process="before the crash", updated=time.time() - 3, message="training fold 3 of 9")))
        listing = _safe(app.jobs_list())
        row = next(r for r in listing["jobs"] if r["id"] == "aaaaaaaaaaaa")
        assert row["status"] == "interrupted" and listing["active"] == 0 and "fold 3 of 9" in row["message"], "read, not restarted: the sign does not hang"
        assert app.job("aaaaaaaaaaaa")["status"] == "interrupted"
        assert app.jobs_clear({"id": "aaaaaaaaaaaa"}) == {"cleared": 1}
        assert all(r["id"] != "aaaaaaaaaaaa" for r in app.jobs_list()["jobs"]) and not (app.joblog.directory / "aaaaaaaaaaaa.json").exists()
        with pytest.raises(ApiError, match="unknown job"):
            app.job("aaaaaaaaaaaa")
        assert app.jobs_clear({"id": "aaaaaaaaaaaa"}) == {"cleared": 0}, "clearing what is not there is not an error"
    finally:
        other.kill()
        other.wait()


def test_a_run_that_is_really_going_cannot_be_cleared_and_clearing_the_ended_ones_leaves_it(two_apps):
    app = two_apps()
    gate, started = threading.Event(), threading.Event()

    def slow(progress):
        started.set()
        gate.wait(30)
        return {"done": True}

    running = app.submit_task("ml: going", slow, "ml")
    assert started.wait(10)
    with pytest.raises(ApiError, match="still going in this app"):
        app.jobs_clear({"id": running["job"]})
    old_ml = app.submit_task("ml: old", lambda progress: {"ok": 1}, "ml")            # queued behind the running one
    gate.set()
    assert wait(app, running)["status"] == "done" and wait(app, old_ml)["status"] == "done"
    data = app.submit_task("free data", lambda progress: {"ok": 1}, "data")
    assert wait(app, data)["status"] == "done"
    assert app.jobs_clear({"ended": True, "kind": "ml"}) == {"cleared": 2}
    assert [r["label"] for r in app.jobs_list()["jobs"]] == ["free data"], "only the ml runs went"
    assert app.jobs_clear({"ended": True}) == {"cleared": 1} and app.jobs_list()["jobs"] == []
    with pytest.raises(ApiError, match="say which run to clear"):
        app.jobs_clear({})
    with pytest.raises(ApiError, match="say which run to clear"):
        app.jobs_clear({"ended": "yes"})
