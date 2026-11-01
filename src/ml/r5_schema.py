"""The registry tables of Release 5 (migrations 7 to 9), kept apart from ``registry.py`` so that the DDL of each release can be read, tested and reviewed on its own. ``registry.MIGRATIONS`` applies them in order.

* 7  ``analysis_records`` (append-only results of the statistical diagnostics, neighbourhood, delay, calibration and multi-seed analyses), ``validation_policies`` (the content of every policy ever
  registered, keyed by its hash) and ``policy_adoptions`` (who adopted which policy, when and why; the latest row is the policy in force);
* 8  the prospective ledger: immutable ``commitments``, append-only ``settlements`` and an audit of every attempt to commit (``prospective.py``);
* 9  governance: immutable ``strategy_versions`` (the identity of what is deployed: model artifact, signal, execution and risk settings), ``risk_limits`` and an append-only ``governance_events`` audit.

Every table is additive and append-only where it records a decision; triggers refuse updates and deletes.
"""

ANALYSIS = [
    """CREATE TABLE IF NOT EXISTS analysis_records (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, subject_id TEXT NOT NULL, study_id TEXT, content TEXT NOT NULL, content_hash TEXT NOT NULL, policy_hash TEXT, actor TEXT, created TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_analysis_subject ON analysis_records (subject_id, kind, seq)",
    "CREATE TRIGGER IF NOT EXISTS analysis_no_update BEFORE UPDATE ON analysis_records BEGIN SELECT RAISE(ABORT, 'analysis records are append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS analysis_no_delete BEFORE DELETE ON analysis_records BEGIN SELECT RAISE(ABORT, 'analysis records are append-only'); END",
    """CREATE TABLE IF NOT EXISTS validation_policies (
        policy_hash TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL, illustrative INTEGER NOT NULL, content TEXT NOT NULL, created TEXT NOT NULL)""",
    "CREATE TRIGGER IF NOT EXISTS policies_no_update BEFORE UPDATE ON validation_policies BEGIN SELECT RAISE(ABORT, 'a policy is immutable: register a new version'); END",
    "CREATE TRIGGER IF NOT EXISTS policies_no_delete BEFORE DELETE ON validation_policies BEGIN SELECT RAISE(ABORT, 'a policy cannot be deleted'); END",
    """CREATE TABLE IF NOT EXISTS policy_adoptions (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, policy_hash TEXT NOT NULL, actor TEXT NOT NULL, note TEXT NOT NULL, at TEXT NOT NULL)""",
    "CREATE TRIGGER IF NOT EXISTS adoptions_no_update BEFORE UPDATE ON policy_adoptions BEGIN SELECT RAISE(ABORT, 'adoptions are append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS adoptions_no_delete BEFORE DELETE ON policy_adoptions BEGIN SELECT RAISE(ABORT, 'adoptions are append-only'); END",
]

PROSPECTIVE = TABLES = [
    """CREATE TABLE IF NOT EXISTS commitments (
        commitment_id TEXT PRIMARY KEY, commit_key TEXT NOT NULL UNIQUE, content_hash TEXT NOT NULL, strategy_version TEXT NOT NULL, decision_ts TEXT NOT NULL, universe_version TEXT NOT NULL,
        horizon INTEGER NOT NULL, created_at TEXT NOT NULL, created_epoch REAL NOT NULL, data_cutoff TEXT, fit_cutoff TEXT, model_created TEXT, artifact_hashes TEXT NOT NULL,
        classification TEXT NOT NULL, why TEXT NOT NULL, forecasts TEXT NOT NULL, weights TEXT, actor TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS commitment_audit (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, commit_key TEXT NOT NULL, event TEXT NOT NULL, content_hash TEXT, existing_hash TEXT, actor TEXT, detail TEXT, at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS settlements (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, commitment_id TEXT NOT NULL, version INTEGER NOT NULL, outcomes TEXT NOT NULL, outcome_hash TEXT NOT NULL, vintage TEXT NOT NULL,
        correction_reason TEXT, settled_at TEXT NOT NULL, actor TEXT NOT NULL, UNIQUE (commitment_id, version))""",
    "CREATE TRIGGER IF NOT EXISTS commitments_no_update BEFORE UPDATE ON commitments BEGIN SELECT RAISE(ABORT, 'a commitment is immutable'); END",
    "CREATE TRIGGER IF NOT EXISTS commitments_no_delete BEFORE DELETE ON commitments BEGIN SELECT RAISE(ABORT, 'a commitment cannot be deleted'); END",
    "CREATE TRIGGER IF NOT EXISTS settlements_no_update BEFORE UPDATE ON settlements BEGIN SELECT RAISE(ABORT, 'a settlement is append-only: record a new version'); END",
    "CREATE TRIGGER IF NOT EXISTS settlements_no_delete BEFORE DELETE ON settlements BEGIN SELECT RAISE(ABORT, 'a settlement cannot be deleted'); END",
    "CREATE TRIGGER IF NOT EXISTS commitment_audit_no_update BEFORE UPDATE ON commitment_audit BEGIN SELECT RAISE(ABORT, 'the audit is append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS commitment_audit_no_delete BEFORE DELETE ON commitment_audit BEGIN SELECT RAISE(ABORT, 'the audit is append-only'); END",
]

GOVERNANCE = [
    """CREATE TABLE IF NOT EXISTS strategy_versions (
        version_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, model_id TEXT, content TEXT NOT NULL, risk_limits_hash TEXT, created TEXT NOT NULL, actor TEXT NOT NULL)""",
    "CREATE TRIGGER IF NOT EXISTS versions_no_update BEFORE UPDATE ON strategy_versions BEGIN SELECT RAISE(ABORT, 'a strategy version is immutable'); END",
    "CREATE TRIGGER IF NOT EXISTS versions_no_delete BEFORE DELETE ON strategy_versions BEGIN SELECT RAISE(ABORT, 'a strategy version cannot be deleted'); END",
    """CREATE TABLE IF NOT EXISTS risk_limits (
        limits_hash TEXT PRIMARY KEY, content TEXT NOT NULL, created TEXT NOT NULL, actor TEXT NOT NULL)""",
    "CREATE TRIGGER IF NOT EXISTS limits_no_update BEFORE UPDATE ON risk_limits BEGIN SELECT RAISE(ABORT, 'risk limits are immutable: register new ones'); END",
    "CREATE TRIGGER IF NOT EXISTS limits_no_delete BEFORE DELETE ON risk_limits BEGIN SELECT RAISE(ABORT, 'risk limits cannot be deleted'); END",
    """CREATE TABLE IF NOT EXISTS governance_events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, subject_id TEXT NOT NULL, detail TEXT, actor TEXT NOT NULL, at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_governance_subject ON governance_events (subject_id, seq)",
    "CREATE TRIGGER IF NOT EXISTS governance_no_update BEFORE UPDATE ON governance_events BEGIN SELECT RAISE(ABORT, 'the governance audit is append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS governance_no_delete BEFORE DELETE ON governance_events BEGIN SELECT RAISE(ABORT, 'the governance audit is append-only'); END",
]

MIGRATIONS = [(7, "analysis_and_policies", ANALYSIS), (8, "prospective_ledger", PROSPECTIVE), (9, "governance", GOVERNANCE)]
