"""One SQLite file for what the app learns while it runs.

The fundamentals database, the experiment registry and the research workspace keep their own files; this one holds the rest that used to be lost or buried in many small files:

``training_fits`` / ``training_epochs``  every network that was trained: its settings, the loss of every epoch, and how it ended (finished, stopped early, or interrupted: the app died while it was training,
                                          which is what an out-of-memory kill looks like afterwards). Written epoch by epoch, so a crash keeps what happened before it.
``job_history``                           every job the dashboard ever ran, not only the last 60 whose files are kept: when, what, how it ended, and the message it died with.
``analysis_cache``                        a computed answer kept under a name and a *version* (what it was computed from), so that asking again for the same thing is a read, not a recomputation.
``kv``                                    small settings the app wants to keep.

Every method is safe from any thread and none lets a database problem reach the caller as anything but a ``False``/empty answer where that is harmless (a log that cannot be written must never stop a training).
The file is in WAL mode, so a page reading the history does not wait for a training that is writing it, and it can be copied while the app runs with :meth:`backup` (SQLite's own online backup).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = 2
STALE_FIT = 90.0                                    # seconds without a sign of life before a fit of another process counts as interrupted
PROCESS = f"{os.getpid()}-{int(time.time())}"        # names this run of the app: a fit left 'running' by an earlier run was interrupted

MIGRATIONS = {
    1: """
    CREATE TABLE IF NOT EXISTS training_fits (
        id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, job_label TEXT, process TEXT, model TEXT, fold INTEGER, folds INTEGER, n INTEGER, epochs INTEGER, patience INTEGER, device TEXT,
        n_params INTEGER, n_train INTEGER, n_val INTEGER, batch_size INTEGER, status TEXT NOT NULL, started REAL NOT NULL, updated REAL NOT NULL, finished REAL,
        epochs_run INTEGER, best_epoch INTEGER, best_val REAL, final_train REAL, stopped_early INTEGER, seconds REAL, peak_mb REAL);
    CREATE INDEX IF NOT EXISTS fits_job ON training_fits(job_id);
    CREATE INDEX IF NOT EXISTS fits_started ON training_fits(started);
    CREATE TABLE IF NOT EXISTS training_epochs (
        fit_id INTEGER NOT NULL, epoch INTEGER NOT NULL, train REAL, val REAL, lr REAL, grad REAL, seconds REAL, best REAL, bad INTEGER, improved INTEGER, at REAL NOT NULL, PRIMARY KEY (fit_id, epoch));
    CREATE TABLE IF NOT EXISTS job_history (
        id TEXT PRIMARY KEY, label TEXT, kind TEXT, status TEXT, message TEXT, error TEXT, created REAL, finished REAL, updated REAL, session TEXT, retry_of TEXT, process TEXT);
    CREATE INDEX IF NOT EXISTS jobs_created ON job_history(created);
    """,
    2: """
    CREATE TABLE IF NOT EXISTS analysis_cache (kind TEXT NOT NULL, key TEXT NOT NULL, version TEXT NOT NULL, value TEXT NOT NULL, created REAL NOT NULL, expires REAL, hits INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (kind, key));
    CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated REAL NOT NULL);
    """,
}


class AppDb:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.error: str | None = None
        self._open()

    # ------------------------------------------------------------------------------------------------ connection
    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=15, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 15000")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA:
                self.error = f"app.db is from a newer version of the app (schema {version}, this one knows {SCHEMA}); it is left alone"
                conn.close()
                return
            for v in range(version + 1, SCHEMA + 1):
                conn.executescript(MIGRATIONS[v])
                conn.execute(f"PRAGMA user_version = {v}")
            self._conn = conn
        except (sqlite3.Error, OSError) as error:
            self.error = f"{type(error).__name__}: {error}"
            self._conn = None

    @property
    def ok(self) -> bool:
        return self._conn is not None

    def _run(self, sql: str, params=(), many: bool = False):
        if self._conn is None:
            return None
        with self._lock:
            try:
                return self._conn.executemany(sql, params) if many else self._conn.execute(sql, params)
            except sqlite3.Error as error:
                self.error = f"{type(error).__name__}: {error}"
                return None

    def _all(self, sql: str, params=()) -> list[dict]:
        with self._lock:
            cur = self._run(sql, params)
            return [] if cur is None else [dict(r) for r in cur.fetchall()]

    def _one(self, sql: str, params=()):
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------------------------------------ training
    def fit_started(self, job_id: str | None, job_label: str | None, run: dict) -> int | None:
        ctx = run.get("context") or {}
        now = time.time()
        cur = self._run("INSERT INTO training_fits (job_id, job_label, process, model, fold, folds, n, epochs, patience, device, n_params, n_train, n_val, batch_size, status, started, updated) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (job_id, job_label, PROCESS, ctx.get("model") or run.get("name"), ctx.get("fold"), ctx.get("folds"), run.get("n"), run.get("epochs"), run.get("patience"), run.get("device"), run.get("n_params"),
                         run.get("n_train"), run.get("n_val"), run.get("batch_size"), "running", now, now))
        return None if cur is None else cur.lastrowid

    def fit_epoch(self, fit_id: int | None, row: dict) -> None:
        if fit_id is None:
            return
        now = time.time()
        self._run("INSERT OR REPLACE INTO training_epochs (fit_id, epoch, train, val, lr, grad, seconds, best, bad, improved, at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (fit_id, int(row["epoch"]), row.get("train"), row.get("val"), row.get("lr"), row.get("grad"), row.get("seconds"), row.get("best"), row.get("bad"), 1 if row.get("improved") else 0, now))
        self._run("UPDATE training_fits SET updated = ?, epochs_run = ? WHERE id = ?", (now, int(row["epoch"]) + 1, fit_id))

    def fit_touch(self, fit_id: int | None) -> None:
        if fit_id is not None:
            self._run("UPDATE training_fits SET updated = ? WHERE id = ?", (time.time(), fit_id))

    def fit_finished(self, fit_id: int | None, summary: dict) -> None:
        if fit_id is None:
            return
        now = time.time()
        self._run("UPDATE training_fits SET status = 'done', finished = ?, updated = ?, epochs_run = ?, best_epoch = ?, best_val = ?, final_train = ?, stopped_early = ?, seconds = ?, peak_mb = ? WHERE id = ?",
                  (now, now, summary.get("epochs_run"), summary.get("best_epoch"), summary.get("best_val"), summary.get("final_train"), 1 if summary.get("stopped_early") else 0, summary.get("seconds"), summary.get("peak_mb"), fit_id))

    def fit_aborted(self, fit_id: int | None, status: str = "cancelled") -> None:
        if fit_id is not None:
            now = time.time()
            self._run("UPDATE training_fits SET status = ?, finished = ?, updated = ? WHERE id = ? AND status = 'running'", (status, now, now, fit_id))

    def mark_interrupted(self) -> int:
        """Fits another run of the app left 'running' and that have shown no sign of life for a while: that app died with them (out of memory, a restart, the host going idle)."""
        cur = self._run("UPDATE training_fits SET status = 'interrupted', finished = updated WHERE status = 'running' AND process != ? AND updated < ?", (PROCESS, time.time() - STALE_FIT))
        return 0 if cur is None else cur.rowcount

    def training_fits(self, job_id: str | None = None, model: str | None = None, status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
        where, args = [], []
        for col, val in (("job_id", job_id), ("model", model), ("status", status)):
            if val:
                where.append(f"{col} = ?")
                args.append(val)
        return self._all("SELECT * FROM training_fits" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY started DESC LIMIT ? OFFSET ?", (*args, int(limit), int(offset)))

    def training_fit(self, fit_id: int) -> dict | None:
        fit = self._one("SELECT * FROM training_fits WHERE id = ?", (int(fit_id),))
        if fit is None:
            return None
        fit["epochs_detail"] = self._all("SELECT epoch, train, val, lr, grad, seconds, best, bad, improved FROM training_epochs WHERE fit_id = ? ORDER BY epoch", (int(fit_id),))
        return fit

    # ------------------------------------------------------------------------------------------------ jobs
    def upsert_job(self, record: dict) -> None:
        self._run("INSERT INTO job_history (id, label, kind, status, message, error, created, finished, updated, session, retry_of, process) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                  "ON CONFLICT(id) DO UPDATE SET label=excluded.label, kind=excluded.kind, status=excluded.status, message=excluded.message, error=excluded.error, finished=excluded.finished, updated=excluded.updated, "
                  "session=excluded.session, retry_of=excluded.retry_of",
                  (record.get("id"), record.get("label"), record.get("kind"), record.get("status"), record.get("message"), record.get("error"), record.get("created"), record.get("finished"),
                   record.get("updated") or time.time(), record.get("session"), record.get("retry_of"), record.get("process") or PROCESS))

    def jobs(self, kind: str | None = None, status: str | None = None, q: str | None = None, limit: int = 100, offset: int = 0) -> dict:
        where, args = [], []
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if status:
            where.append("status = ?")
            args.append(status)
        if q:
            where.append("(label LIKE ? OR message LIKE ? OR error LIKE ?)")
            args += [f"%{q}%"] * 3
        clause = " WHERE " + " AND ".join(where) if where else ""
        total = (self._one("SELECT COUNT(*) AS n FROM job_history" + clause, tuple(args)) or {"n": 0})["n"]
        rows = self._all("SELECT * FROM job_history" + clause + " ORDER BY created DESC LIMIT ? OFFSET ?", (*args, int(limit), int(offset)))
        return {"total": total, "jobs": rows}

    def job_counts(self) -> dict:
        return {r["status"]: r["n"] for r in self._all("SELECT status, COUNT(*) AS n FROM job_history GROUP BY status")}

    # ------------------------------------------------------------------------------------------------ cache
    def cache_get(self, kind: str, key: str, version: str):
        """The kept answer, or ``None`` when there is none, it was made from other data (``version``), or it has expired."""
        row = self._one("SELECT value, version, expires FROM analysis_cache WHERE kind = ? AND key = ?", (kind, key))
        if row is None or row["version"] != str(version) or (row["expires"] is not None and row["expires"] < time.time()):
            return None
        self._run("UPDATE analysis_cache SET hits = hits + 1 WHERE kind = ? AND key = ?", (kind, key))
        try:
            return json.loads(row["value"])
        except ValueError:
            return None

    def cache_put(self, kind: str, key: str, version: str, value, ttl: float | None = None) -> bool:
        try:
            text = json.dumps(value, allow_nan=False, default=str)
        except (TypeError, ValueError):
            return False
        now = time.time()
        return self._run("INSERT OR REPLACE INTO analysis_cache (kind, key, version, value, created, expires, hits) VALUES (?,?,?,?,?,?,0)", (kind, key, str(version), text, now, None if ttl is None else now + ttl)) is not None

    def cache_clear(self, kind: str | None = None) -> int:
        cur = self._run("DELETE FROM analysis_cache" + (" WHERE kind = ?" if kind else ""), (kind,) if kind else ())
        return 0 if cur is None else cur.rowcount

    def cache_stats(self) -> list[dict]:
        return self._all("SELECT kind, COUNT(*) AS entries, SUM(LENGTH(value)) AS bytes, SUM(hits) AS hits, MAX(created) AS newest FROM analysis_cache GROUP BY kind ORDER BY kind")

    # ------------------------------------------------------------------------------------------------ kv
    def get(self, key: str, default=None):
        row = self._one("SELECT value FROM kv WHERE key = ?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except ValueError:
            return default

    def set(self, key: str, value) -> None:
        self._run("INSERT OR REPLACE INTO kv (key, value, updated) VALUES (?,?,?)", (key, json.dumps(value, default=str), time.time()))

    # ------------------------------------------------------------------------------------------------ storage
    def table_counts(self) -> dict:
        names = [r["name"] for r in self._all("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {n: (self._one(f'SELECT COUNT(*) AS n FROM "{n}"') or {"n": 0})["n"] for n in names}

    def size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                total += os.path.getsize(str(self.path) + suffix)
            except OSError:
                pass
        return total

    def backup(self, dest) -> bool:
        """A consistent copy of the database to ``dest`` while it is in use (SQLite's online backup)."""
        if self._conn is None:
            return False
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            target = sqlite3.connect(str(dest))
            with self._lock:
                self._conn.backup(target)
            target.close()
            return True
        except sqlite3.Error as error:
            self.error = f"{type(error).__name__}: {error}"
            return False


class TrainingLog:
    """The bridge from a job's :class:`~src.ml.live.Telemetry` to the database: one instance per job, so that every fit it trains is filed under that job."""

    TOUCH = 10.0

    def __init__(self, db: AppDb, job_id: str, job_label: str):
        self.db, self.job_id, self.job_label = db, job_id, job_label
        self.fit_id: int | None = None
        self._touched = 0.0

    def start(self, run: dict) -> None:
        self.fit_id = self.db.fit_started(self.job_id, self.job_label, run)

    def epoch(self, row: dict) -> None:
        self.db.fit_epoch(self.fit_id, row)

    def touch(self) -> None:
        now = time.time()
        if now - self._touched >= self.TOUCH:
            self._touched = now
            self.db.fit_touch(self.fit_id)

    def finish(self, summary: dict) -> None:
        self.db.fit_finished(self.fit_id, summary)
        self.fit_id = None

    def abort(self, status: str) -> None:
        self.db.fit_aborted(self.fit_id, status)
        self.fit_id = None
