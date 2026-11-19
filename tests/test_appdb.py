"""The app's own database (src/store/appdb.py) and its routes: the training of a network filed epoch by epoch (so a crash keeps what happened before it), jobs kept beyond the newest sixty, kept answers that
are only used for the data they were made from, a refusal to touch a database from a newer version, and a backup that is a real, readable copy made while the app runs."""

from __future__ import annotations

import io
import sqlite3
import time
import zipfile

import pytest

from src.ml import live
from src.store import appdb as adb
from src.store.appdb import AppDb, TrainingLog
from src.utils.config import load_config
from src.webapp.api import ApiError, App, Job
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch, _wait


@pytest.fixture()
def db(tmp_path):
    d = AppDb(tmp_path / "app.db")
    yield d
    d.close()


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


def _train(telemetry, epochs=3, model="lstm", fold=1):
    with live.use(telemetry):
        live.set_context(model=model, fold=fold, folds=2)
        live.start(epochs=5, patience=2, device="cpu", n_params=99, n_train=100, n_val=20, batch_size=16)
        for e in range(epochs):
            live.epoch(e, 1.0 - 0.1 * e, 1.1 - 0.05 * e, lr=1e-3, grad=0.2, seconds=0.5, best=1.1 - 0.05 * e, best_epoch=e, bad=0, improved=True, epochs=5, patience=2, peak_mb=42.0)


def test_a_fit_is_filed_epoch_by_epoch_and_a_crash_keeps_what_happened_before_it(db):
    job = Job("Train the lstm", "ml")
    job.telemetry.store = TrainingLog(db, job.id, job.label)
    _train(job.telemetry, epochs=3)                                                      # the process dies here: the fit never finished
    fit = db.training_fits()[0]
    assert fit["status"] == "running" and fit["job_id"] == job.id and fit["model"] == "lstm" and fit["fold"] == 1 and fit["n_params"] == 99 and fit["epochs_run"] == 3
    detail = db.training_fit(fit["id"])["epochs_detail"]
    assert [e["epoch"] for e in detail] == [0, 1, 2] and detail[2]["train"] == pytest.approx(0.8) and detail[2]["val"] == pytest.approx(1.0) and detail[0]["improved"] == 1
    assert db.mark_interrupted() == 0, "a fit of this very run is not interrupted"
    db._run("UPDATE training_fits SET process = 'another-run', updated = ?", (time.time() - 1000,))
    assert db.mark_interrupted() == 1 and db.training_fits()[0]["status"] == "interrupted" and db.training_fits(status="interrupted")[0]["epochs_run"] == 3


def test_a_finished_fit_has_its_summary_and_a_cancelled_one_is_not_left_running(db):
    done, stopped = Job("a", "ml"), Job("b", "ml")
    for j in (done, stopped):
        j.telemetry.store = TrainingLog(db, j.id, j.label)
    _train(done.telemetry, epochs=3)
    with live.use(done.telemetry):
        live.finish(best_epoch=3, best_val=0.95, stopped_early=True, final_train=0.8, peak_mb=42.0)
    _train(stopped.telemetry, epochs=2, model="gru", fold=2)
    stopped.telemetry.abort("cancelled")
    fits = {f["model"]: f for f in db.training_fits()}
    assert fits["lstm"]["status"] == "done" and fits["lstm"]["stopped_early"] == 1 and fits["lstm"]["best_epoch"] == 3 and fits["lstm"]["best_val"] == pytest.approx(0.95) and fits["lstm"]["peak_mb"] == 42.0
    assert fits["gru"]["status"] == "cancelled"
    assert [f["model"] for f in db.training_fits(model="gru")] == ["gru"] and len(db.training_fits(job_id=done.id)) == 1


def test_a_database_that_cannot_be_written_never_stops_a_training(tmp_path):
    bad = AppDb(tmp_path / "missing" / "deeper" / "app.db")
    bad.close()                                                                           # closed: every write now fails
    job = Job("t", "ml")
    job.telemetry.store = TrainingLog(bad, job.id, job.label)
    _train(job.telemetry, epochs=2)                                                       # no exception
    assert job.telemetry.snapshot()["run"]["current"]["epoch"] == 2 and bad.training_fits() == []


def test_jobs_are_kept_beyond_the_newest_sixty_and_can_be_searched(db):
    for i in range(130):
        db.upsert_job({"id": f"{i:012x}", "label": f"Backtest {i}", "kind": "backtest" if i % 2 else "ml", "status": "error" if i % 10 == 0 else "done", "message": "finished", "error": "out of memory" if i % 10 == 0 else None,
                       "created": 1000.0 + i, "finished": 1005.0 + i})
    db.upsert_job({"id": f"{0:012x}", "label": "Backtest 0", "kind": "ml", "status": "interrupted", "message": "the app stopped", "created": 1000.0})
    out = db.jobs(limit=10)
    assert out["total"] == 130 and len(out["jobs"]) == 10 and out["jobs"][0]["label"] == "Backtest 129", "newest first, and all of them kept"
    assert db.jobs(q="out of memory")["total"] == 12 and db.jobs(status="interrupted")["jobs"][0]["id"] == f"{0:012x}" and db.jobs(kind="ml", status="done")["total"] == 52
    assert db.job_counts()["interrupted"] == 1 and db.jobs(limit=5, offset=125)["jobs"][-1]["label"] == "Backtest 0"


def test_a_kept_answer_is_used_only_for_the_data_it_was_made_from(db):
    assert db.cache_get("valuation", "AAPL|0.09", "v1") is None
    assert db.cache_put("valuation", "AAPL|0.09", "v1", {"fair": [1.5, 2.5]}) and db.cache_get("valuation", "AAPL|0.09", "v1") == {"fair": [1.5, 2.5]}
    assert db.cache_get("valuation", "AAPL|0.09", "v2") is None, "the statements changed: the answer is not trusted"
    db.cache_put("valuation", "MSFT|0.09", "v1", {"x": 1}, ttl=-1)
    assert db.cache_get("valuation", "MSFT|0.09", "v1") is None, "an expired answer is not returned"
    assert not db.cache_put("valuation", "bad", "v1", {"x": float("nan")}), "what cannot be stored as JSON is refused, not half stored"
    stats = db.cache_stats()
    assert stats[0]["kind"] == "valuation" and stats[0]["entries"] == 2 and stats[0]["hits"] == 1
    assert db.cache_clear("valuation") == 2 and db.cache_stats() == []
    db.set("landing", {"tab": "overview"})
    assert db.get("landing") == {"tab": "overview"} and db.get("absent", 7) == 7


def test_a_database_from_a_newer_version_is_left_alone_and_the_file_is_migrated_once(tmp_path):
    path = tmp_path / "app.db"
    first = AppDb(path)
    assert first.ok and first._one("PRAGMA user_version")["user_version"] == adb.SCHEMA
    first.close()
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {adb.SCHEMA + 5}")
    conn.commit()
    conn.close()
    newer = AppDb(path)
    assert not newer.ok and "newer version" in newer.error and newer.training_fits() == [] and newer.jobs()["total"] == 0, "nothing is read from it and nothing is written to it"


def test_the_backup_is_a_consistent_copy_made_while_the_database_is_in_use(db, tmp_path):
    for i in range(50):
        db.upsert_job({"id": f"{i:012x}", "label": f"job {i}", "kind": "ml", "status": "done", "created": float(i)})
    out = tmp_path / "copy" / "app.db"
    assert db.backup(out)
    copy = sqlite3.connect(out)
    assert copy.execute("SELECT COUNT(*) FROM job_history").fetchone()[0] == 50 and copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    copy.close()


def test_the_routes_report_training_jobs_and_storage_and_the_backup_is_a_zip_of_every_database(app):
    for path in ("/api/training/fits", "/api/training/fit", "/api/jobs/history", "/api/storage/status", "/api/storage/backups"):
        assert path in app.extra_get, path
    assert {"/api/storage/backup", "/api/storage/cache/clear"} <= set(app.extra_post)
    view = _wait(app, app.submit({"models": [{"name": "momentum"}]})["job"])
    assert view["status"] == "done"
    history = app.extra_get["/api/jobs/history"]({"limit": "5"})
    assert history["total"] >= 1 and any(j["id"] == view["id"] and j["status"] == "done" for j in history["jobs"]) and history["counts"]["done"] >= 1
    assert app.extra_get["/api/training/fits"]({})["fits"] == [] and app.extra_get["/api/training/fits"]({})["ok"] is True
    for bad in ({"limit": "0"}, {"limit": "x"}, {"offset": "-1"}):
        with pytest.raises(ApiError):
            app.extra_get["/api/training/fits"](bad)
    with pytest.raises(ApiError, match="unknown training fit"):
        app.extra_get["/api/training/fit"]({"id": "999"})
    status = app.extra_get["/api/storage/status"]({})
    assert status["app_db"]["ok"] and status["app_db"]["tables"]["job_history"] >= 1 and any(d["name"] == "app.db" for d in status["databases"]) and "advice" in status
    made = app.extra_post["/api/storage/backup"]({})
    assert made["name"].startswith("quant-lab-backup-") and made["bytes"] > 0 and "app.db" in made["files"]
    import base64
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(made["zip_base64"])))
    assert "app.db" in z.namelist() and z.testzip() is None
    raw = z.read("app.db")
    assert raw[:15] == b"SQLite format 3"
    assert app.extra_get["/api/storage/backups"]({})["backups"][0]["name"] == made["name"]


def test_a_training_run_inside_a_job_is_filed_under_that_job_and_a_stopped_app_marks_it_interrupted(app, tmp_path):
    job = app.submit_task("Train", lambda progress: (_train(live.current(), epochs=4), {"ok": True})[1], kind="ml")["job"]
    view = _wait(app, job)
    assert view["status"] == "done" and view["live"]["run"]["current"]["epoch"] == 4
    fit = app.extra_get["/api/training/fits"]({"job": job})["fits"][0]
    assert fit["job_label"] == "Train" and fit["epochs_run"] == 4 and fit["status"] == "running"        # the loop never said it finished: that is how an interrupted fit looks
    app.appdb._run("UPDATE training_fits SET process = 'earlier-run', updated = ?", (time.time() - 1000,))
    restarted = App(app.config, builder=app.builder, root=app.root)                                        # the app is started again over the same files
    assert restarted.extra_get["/api/training/fits"]({"job": job})["fits"][0]["status"] == "interrupted"
