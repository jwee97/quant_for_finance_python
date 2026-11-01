"""Versioned, additive registry migrations, and the verification evidence they hold: legacy registries keep their history, a failed migration leaves nothing behind, evidence is append-only, a stale green is flagged."""

import sqlite3

import pytest

from src.ml import registry as rg, verification as vf
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished


def test_a_fresh_registry_is_at_the_latest_version_and_reopening_changes_nothing(tmp_path):
    r = rg.Registry(tmp_path / "ml")
    assert r.schema_version() == max(v for v, _, _ in rg.MIGRATIONS)
    eid = r.new_id("kept")
    r.log_experiment(eid, {"name": "kept", "model": {"kind": "ridge"}, "dataset": {"target": "fwd_ret_5", "horizon": 5, "universe": "etf"}, "signal": {"method": "top_n_long_short"}}, "h", "k", 0)
    again = rg.Registry(tmp_path / "ml")
    assert again.schema_version() == r.schema_version() and again.get_experiment(eid)["name"] == "kept"
    with again._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == len(rg.MIGRATIONS)


def test_a_registry_from_before_migrations_existed_is_upgraded_in_place_and_keeps_every_experiment(tmp_path):
    root = tmp_path / "old"
    (root).mkdir()
    with sqlite3.connect(root / "registry.sqlite") as c:
        c.executescript(rg.SCHEMA)                                                       # the tables as they were, with no schema_migrations and no verification table
        c.execute("INSERT INTO experiments (id, name, created, status, model_kind, stage) VALUES ('legacy-1', 'old run', '2026-01-01', 'done', 'ridge', 'production')")
    r = rg.Registry(root)
    assert r.get_experiment("legacy-1")["stage"] == "production" and r.schema_version() >= 1 and r.list_checks() == []
    assert rg.Registry(root).schema_version() == r.schema_version()


def test_a_migration_that_fails_is_rolled_back_whole_and_the_version_does_not_move(tmp_path, monkeypatch):
    r = rg.Registry(tmp_path / "ml")
    before = r.schema_version()
    monkeypatch.setattr(rg, "MIGRATIONS", [*rg.MIGRATIONS, (before + 1, "broken", "CREATE TABLE half_made (x INTEGER); INSERT INTO no_such_table VALUES (1)")])
    with pytest.raises(sqlite3.OperationalError):
        rg.Registry(tmp_path / "ml")
    with r._conn() as c:
        assert c.execute("SELECT name FROM sqlite_master WHERE name='half_made'").fetchone() is None and r.schema_version() == before


def test_evidence_is_appended_never_rewritten_and_a_status_must_be_one_of_the_five(tmp_path):
    r = rg.Registry(tmp_path / "ml")
    first = r.log_check("P0-01", "t", "fail", reason="broken", observed={"n": 1})
    second = r.log_check("P0-01", "t", "pass", observed={"n": 2})
    rows = r.list_checks("P0-01")
    assert [x["id"] for x in rows] == [second, first] and [x["status"] for x in rows] == ["pass", "fail"] and rows[1]["reason"] == "broken"
    for bad in ("passed", "ok", "green", ""):
        with pytest.raises(rg.RegistryError):
            r.log_check("P0-01", "t", bad)


def test_deleting_an_experiment_leaves_the_verification_evidence(tmp_path):
    r = rg.Registry(tmp_path / "ml")
    eid = r.new_id("gone")
    r.log_experiment(eid, {"name": "gone", "model": {"kind": "ridge"}, "dataset": {"target": "fwd_ret_5", "horizon": 5, "universe": "etf"}, "signal": {"method": "top_n_long_short"}}, "h", "k", 0)
    r.log_check("P0-05", "t", "pass")
    r.delete_experiment(eid)
    assert len(r.list_checks("P0-05")) == 1


def test_status_rules_are_strict_about_skips_failures_and_empty_runs():
    ran = {"file": "a", "passed": 5, "failed": 0, "skipped": 0, "errors": 0, "failures": [], "skips": []}
    assert vf.status_of([ran])[0] == "pass"
    assert vf.status_of([ran, {**ran, "passed": 3, "skipped": 2, "skips": ["x: gymnasium is not available"]}])[0] == "blocked"
    na = vf.status_of([ran, {**ran, "skipped": 1, "skips": ["test_x[ngboost]: NOT_APPLICABLE: needs pandas below 3"]}])
    assert na[0] == "pass" and "1 not applicable" in na[1], "a justified gap is counted and named, not hidden and not a block"
    assert vf.status_of([ran, {**ran, "failed": 1, "failures": ["test_x"]}])[0] == "fail"
    assert vf.status_of([{**ran, "passed": 0}])[0] == "not_run" and vf.status_of([])[0] == "not_run"
    assert vf.status_of([{**ran, "errors": 1, "failures": ["boom"]}])[0] == "fail"


def test_a_recorded_pass_goes_stale_when_the_source_it_checked_changes(tmp_path, monkeypatch):
    r = rg.Registry(tmp_path / "ml")
    r.log_check("P0-05", "tests:ranker", "pass", subject_hash=vf.source_hash("ranker"))
    fresh = {x["requirement_id"]: x for x in vf.current(r)}
    assert fresh["P0-05"]["status"] == "pass" and fresh["P0-05"]["stale"] is False and fresh["P0-01"]["status"] == "not_run" and fresh["P0-01"]["reason"] == "never run"
    monkeypatch.setattr(vf, "source_hash", lambda group: "something-else")
    assert {x["requirement_id"]: x for x in vf.current(r)}["P0-05"]["stale"] is True


@pytest.fixture(scope="module")
def mlapp(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("verify_api")
    data = make_bundle(n_assets=6, n=700, seed=2)
    return _app(tmp, lambda config: data, root=tmp)


def test_the_verification_endpoint_lists_every_requirement_and_only_a_run_can_write_a_status(mlapp):
    out = _safe(mlapp.extra_get["/api/ml/verification"]({}))
    assert {c["requirement_id"] for c in out["checks"]} == set(vf.REQUIREMENTS) and {c["status"] for c in out["checks"]} == {"not_run"}
    assert _safe(mlapp.extra_get["/api/ml/verification"]({"id": "P0-05"}))["checks"][0]["requirement_id"] == "P0-05"
    with pytest.raises(ApiError, match="unknown requirement"):
        mlapp.extra_get["/api/ml/verification"]({"id": "P9-99"})
    for body, text in (({"requirements": "P0-05"}, "list of requirement ids"), ({"requirements": ["P0-05", 7]}, "list of requirement ids"), ({"requirements": ["P7-01"]}, "unknown requirement"),
                       ({"status": "pass"}, None)):
        if text:
            with pytest.raises(ApiError, match=text):
                mlapp.extra_post["/api/ml/verification/run"](body)
    assert not mlapp.jobs, "nothing refused may have been queued"


def test_a_verification_job_runs_the_tests_and_the_endpoint_then_shows_the_recorded_result(mlapp):
    done = finished(mlapp, mlapp.extra_post["/api/ml/verification/run"]({"requirements": ["P0-05"]}))
    assert done["checks"][0]["requirement_id"] == "P0-05" and done["checks"][0]["status"] in ("pass", "blocked")
    shown = {c["requirement_id"]: c for c in _safe(mlapp.extra_get["/api/ml/verification"]({}))["checks"]}
    assert shown["P0-05"]["status"] == done["checks"][0]["status"] and shown["P0-05"]["stale"] is False and shown["P0-05"]["artifact_hash"] and shown["P0-01"]["status"] == "not_run"
