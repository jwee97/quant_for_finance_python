"""Registry migrations (spec v3, 3.4 and the legacy-preservation rules): a registry written before any migration existed opens, keeps every row, and gains the research-control tables additively; the same registry
opened again changes nothing; a migration that fails halfway leaves no trace; the audit tables refuse rewriting."""

import json
import sqlite3

import pytest

from src.ml import registry as rg, research_controls as rc


def make_legacy_registry(root):
    """A registry as the platform wrote it before the migration table existed: the original tables only, with an experiment, its trials, a model and a forward signal."""
    (root / "experiments" / "legacy-1").mkdir(parents=True)
    (root / "experiments" / "legacy-1" / "report.json").write_text(json.dumps({"id": "legacy-1", "config": {"dataset": {"universe": "etf"}}, "trials": {"dataset_trials": 3}}))
    (root / "models").mkdir(parents=True)
    (root / "models" / "ridge-legacy-1.pkl").write_bytes(b"model-bytes")
    con = sqlite3.connect(root / "registry.sqlite")
    con.executescript(rg.SCHEMA)
    con.execute("INSERT INTO experiments (id, name, created, status, config_hash, dataset_key, universe, model_kind, target, horizon, signal_method, trials_at_run, notes, artifact_dir, parent_id, stage, seconds, source) "
                "VALUES ('legacy-1', 'old run', '2025-01-01 00:00:00', 'done', 'abc', 'dkey', 'etf', 'ridge', 'fwd_ret_5', 5, 'proportional', 2, '', 'x', NULL, 'production', 3.0, 'platform')")
    con.execute("INSERT INTO metrics VALUES ('legacy-1', 'validation', 'sharpe', 1.25)")
    for i in range(2):
        con.execute("INSERT INTO trials (study, dataset_key, experiment_id, number, params, value, seconds, created, stage) VALUES ('s', 'dkey', NULL, ?, '{}', 0.5, 1.0, '2025-01-01', 'random')", (i,))
    con.execute("INSERT INTO models VALUES ('ridge-legacy-1', 'legacy-1', 'ridge', ?, '2025-01-01', 'production', '[]', '')", (str(root / "models" / "ridge-legacy-1.pkl"),))
    con.execute("INSERT INTO forward_signals (experiment_id, signal_date, created, weights, scores, turnover) VALUES ('legacy-1', '2025-02-03', 'now', '{}', '{}', 0.1)")
    con.commit()
    con.close()


def test_a_registry_from_before_the_migrations_opens_keeps_every_row_and_gains_the_new_tables(tmp_path):
    make_legacy_registry(tmp_path)
    reg = rg.Registry(tmp_path)
    assert reg.schema_version() == max(v for v, _, _ in rg.MIGRATIONS) >= 9
    with reg._conn() as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"research_studies", "lineage_events", "experiment_tombstones", "provenance", "evaluation_plans", "plan_events", "holdout_reservations", "holdout_access_log", "selection_evidence", "null_studies",
                "null_replications", "gate_decisions", "stage_changes", "verification_checks", "analysis_records", "validation_policies", "policy_adoptions", "commitments", "commitment_audit", "settlements",
                "strategy_versions", "risk_limits", "governance_events"} <= tables
        assert c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1 and c.execute("SELECT stage FROM models WHERE id='ridge-legacy-1'").fetchone()[0] == "production"
        assert c.execute("SELECT COUNT(*) FROM forward_signals").fetchone()[0] == 1 and c.execute("SELECT value FROM metrics").fetchone()[0] == 1.25
    assert reg.get_experiment("legacy-1")["stage"] == "production" and reg.report("legacy-1")["trials"]["dataset_trials"] == 3
    assert reg.trial_count("dkey") == 1 + 2, "the legacy trial count is unchanged: one finished experiment and two search trials"
    assert rc.raw_looks(reg, "u-etf") == 0, "legacy experiments have no lineage and say so; none is invented"


def test_opening_the_registry_again_applies_nothing_twice(tmp_path):
    make_legacy_registry(tmp_path)
    first = rg.Registry(tmp_path)
    with first._conn() as c:
        before = [dict(r) for r in c.execute("SELECT * FROM schema_migrations ORDER BY version")]
    again = rg.Registry(tmp_path)
    with again._conn() as c:
        after = [dict(r) for r in c.execute("SELECT * FROM schema_migrations ORDER BY version")]
    assert before == after and [r["version"] for r in after] == sorted(v for v, _, _ in rg.MIGRATIONS)


def test_a_migration_that_fails_halfway_leaves_no_trace_and_the_registry_still_opens(tmp_path, monkeypatch):
    make_legacy_registry(tmp_path)
    rg.Registry(tmp_path)                                                                       # everything up to the current version is applied
    broken = list(rg.MIGRATIONS) + [(99, "broken", ["CREATE TABLE half_done (x INTEGER)", "CREATE TABLE half_done (x INTEGER)"])]    # the second statement fails: the table exists already
    monkeypatch.setattr(rg, "MIGRATIONS", broken)
    with pytest.raises(sqlite3.OperationalError):
        rg.Registry(tmp_path)
    monkeypatch.setattr(rg, "MIGRATIONS", broken[:-1])
    reg = rg.Registry(tmp_path)
    with reg._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='half_done'").fetchone()[0] == 0, "the first statement of the failed migration was rolled back"
        assert c.execute("SELECT COUNT(*) FROM schema_migrations WHERE version=99").fetchone()[0] == 0
    assert reg.get_experiment("legacy-1")["name"] == "old run"


def test_a_trigger_migration_is_split_by_statement_not_by_semicolon(tmp_path):
    reg = rg.Registry(tmp_path)
    with reg._conn() as c:
        triggers = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    assert {"lineage_events_no_update", "lineage_events_no_delete", "evaluation_plans_frozen_content", "access_log_no_update", "gate_decisions_no_delete", "provenance_no_update"} <= triggers


def test_the_audit_survives_the_deletion_of_the_experiment_it_describes(tmp_path):
    make_legacy_registry(tmp_path)
    reg = rg.Registry(tmp_path)
    reg.delete_experiment("legacy-1")
    with reg._conn() as c:
        tomb = c.execute("SELECT * FROM experiment_tombstones WHERE experiment_id='legacy-1'").fetchone()
        assert tomb["status"] == "done" and tomb["dataset_key"] == "dkey" and tomb["stage"] == "production" and tomb["report_sha256"] and tomb["deleted_at"]
        assert c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0
    assert reg.trial_count("dkey") == 1 + 2, "a deleted experiment is still a look; the count does not fall"
    with pytest.raises(rg.RegistryError):
        reg.delete_experiment("legacy-1")


def test_legacy_stages_can_be_left_without_evidence_and_every_change_is_recorded(tmp_path):
    make_legacy_registry(tmp_path)
    reg = rg.Registry(tmp_path)
    out = reg.promote("ridge-legacy-1", "archived")                                              # taking a model out of production needs nothing
    assert out["allowed"] and reg.list_models()[0]["stage"] == "archived"
    with reg._conn() as c:
        row = c.execute("SELECT * FROM stage_changes").fetchone()
        assert (row["from_stage"], row["to_stage"], row["model_id"]) == ("production", "archived", "ridge-legacy-1")
    with pytest.raises(rg.RegistryError, match="unknown model"):
        reg.promote("nope", "archived")
    with pytest.raises(rg.RegistryError, match="stage must be one of"):
        reg.promote("ridge-legacy-1", "launched")


def test_a_registry_at_the_end_of_release_3_gains_the_release_5_tables_and_keeps_every_row(tmp_path):
    """Migrations 7 to 9 on a registry that already holds verification, lineage, gate and stage rows: they are added, the earlier rows are untouched, and the append-only tables refuse rewriting."""
    make_legacy_registry(tmp_path)
    con = sqlite3.connect(tmp_path / "registry.sqlite")
    con.executescript(rg.SCHEMA)
    con.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT, applied TEXT)")
    for version, name, script in [m for m in rg.MIGRATIONS if m[0] <= 6]:
        for statement in (list(script) if isinstance(script, (list, tuple)) else [x.strip() for x in script.split(";") if x.strip()]):
            con.execute(statement)
        con.execute("INSERT INTO schema_migrations VALUES (?,?,?)", (version, name, "2026-01-01 00:00:00"))
    con.execute("INSERT INTO gate_decisions (subject_id, gate, policy_version, status, stale, evidence, reasons, actor, previous_seq, at) VALUES ('legacy-1', 'A', 'gates-v2', 'pending', 0, '{}', '[]', 'x', NULL, 'then')")
    con.commit()
    con.close()
    reg = rg.Registry(tmp_path)
    assert reg.schema_version() == 9
    with reg._conn() as c:
        assert [r[0] for r in c.execute("SELECT version FROM schema_migrations ORDER BY version")] == list(range(1, 10))
        assert c.execute("SELECT policy_version FROM gate_decisions").fetchone()[0] == "gates-v2" and c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
        c.execute("INSERT INTO policy_adoptions (policy_hash, actor, note, at) VALUES ('h', 'a', 'n', 't')")
        c.execute("INSERT INTO governance_events (kind, subject_id, detail, actor, at) VALUES ('halted', 'v', '{}', 'a', 't')")
        c.execute("INSERT INTO analysis_records (kind, subject_id, study_id, content, content_hash, policy_hash, actor, created) VALUES ('diagnostics', 'e', NULL, '{}', 'h', NULL, 'a', 't')")
        c.execute("INSERT INTO validation_policies (policy_hash, name, version, illustrative, content, created) VALUES ('h', 'p', 'v', 0, '{}', 't')")
        c.execute("INSERT INTO commitments (commitment_id, commit_key, content_hash, strategy_version, decision_ts, universe_version, horizon, created_at, created_epoch, artifact_hashes, classification, why, forecasts, actor) "
                  "VALUES ('cm-1', 'k', 'h', 'sv', '2024-01-01', 'u', 5, 't', 0.0, '{}', 'replay', 'w', '{}', 'a')")
        c.execute("INSERT INTO settlements (commitment_id, version, outcomes, outcome_hash, vintage, settled_at, actor) VALUES ('cm-1', 1, '{}', 'h', 'first', 't', 'a')")
        c.execute("INSERT INTO strategy_versions (version_id, experiment_id, content, created, actor) VALUES ('sv-1', 'legacy-1', '{}', 't', 'a')")
        c.execute("INSERT INTO risk_limits (limits_hash, content, created, actor) VALUES ('h', '{}', 't', 'a')")
        for stmt, text in (("UPDATE policy_adoptions SET actor='x'", "append-only"), ("DELETE FROM governance_events", "append-only"), ("UPDATE analysis_records SET kind='x'", "append-only"),
                           ("UPDATE settlements SET outcomes='{}'", "append-only"), ("DELETE FROM commitments", "cannot be deleted"), ("UPDATE strategy_versions SET actor='x'", "immutable"),
                           ("UPDATE risk_limits SET actor='x'", "immutable"), ("DELETE FROM validation_policies", "cannot be deleted")):
            with pytest.raises(sqlite3.DatabaseError, match=text):
                c.execute(stmt)
