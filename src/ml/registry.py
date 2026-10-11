"""The experiment and model registry: SQLite for the records, plain files for the artifacts.

Layout under the registry root (default ``artifacts/ml``)::

    registry.sqlite                      tables: experiments, trials, metrics, artifacts, models, feature_sets, predictions, forward_signals
    experiments/<id>/config.yaml         the merged configuration the experiment ran with
                     metrics.json        the flat metrics (also in the ``metrics`` table)
                     predictions.parquet out-of-sample scores with their segment and the realised target
                     positions.parquet   target weights (date x asset)
                     returns.parquet     net / gross returns, costs, turnover, benchmark, segment
                     model.pkl           the final model, fitted to all labelled data
                     report.json         everything the dashboard shows
                     report.html         the same as one self-contained page
    models/<model_id>.pkl                registered models (a copy of an experiment's final model, with a stage)
    features/<dataset_key>/              a saved dataset (features, targets, prices, returns)
    predictions/<id>.parquet             forward-test signals
    reports/                             exports

The registry is also where the TRIAL COUNT lives. Every experiment run and every search trial on a dataset is a recorded look at that data; ``trial_count`` is what the deflated Sharpe ratio of the next
result is deflated by, so a result cannot be reported as if it were the first thing tried.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import yaml

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, name TEXT, created TEXT, status TEXT, config_hash TEXT, dataset_key TEXT, universe TEXT, model_kind TEXT, target TEXT, horizon INTEGER,
    signal_method TEXT, trials_at_run INTEGER, notes TEXT, artifact_dir TEXT, parent_id TEXT, stage TEXT DEFAULT 'none', seconds REAL, source TEXT);
CREATE TABLE IF NOT EXISTS metrics (experiment_id TEXT, segment TEXT, name TEXT, value REAL, PRIMARY KEY (experiment_id, segment, name));
CREATE TABLE IF NOT EXISTS trials (id INTEGER PRIMARY KEY AUTOINCREMENT, study TEXT, dataset_key TEXT, experiment_id TEXT, number INTEGER, params TEXT, value REAL, seconds REAL, created TEXT, stage TEXT);
CREATE TABLE IF NOT EXISTS artifacts (experiment_id TEXT, name TEXT, path TEXT, kind TEXT, bytes INTEGER, sha256 TEXT, PRIMARY KEY (experiment_id, name));
CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, experiment_id TEXT, kind TEXT, path TEXT, created TEXT, stage TEXT, features TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS feature_sets (id TEXT PRIMARY KEY, dataset_key TEXT, created TEXT, path TEXT, n_rows INTEGER, n_features INTEGER, start TEXT, end TEXT, universe TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS predictions (experiment_id TEXT, segment TEXT, n INTEGER, path TEXT, ic REAL, PRIMARY KEY (experiment_id, segment));
CREATE TABLE IF NOT EXISTS forward_signals (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, signal_date TEXT, created TEXT, weights TEXT, scores TEXT, turnover REAL, realised REAL, realised_date TEXT);
"""
STAGES = ("none", "candidate", "staging", "production", "archived")
SUBDIRS = ("experiments", "models", "features", "predictions", "reports")


class RegistryError(ValueError):
    """A request the user can fix (an unknown experiment, a bad stage)."""


class Registry:
    def __init__(self, root: str | Path = "artifacts/ml"):
        self.root = Path(root)
        for d in SUBDIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.sqlite"
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------------------------------------------------ experiments
    @staticmethod
    def new_id(name: str) -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha1(f"{name}{time.time_ns()}".encode()).hexdigest()[:6]

    def experiment_dir(self, exp_id: str) -> Path:
        d = self.root / "experiments" / exp_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log_experiment(self, exp_id: str, cfg: dict, config_hash: str, dataset_key: str, trials_at_run: int, status: str = "done", seconds: float = 0.0, source: str = "platform",
                       parent_id: str | None = None) -> None:
        d = self.experiment_dir(exp_id)
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        uni = cfg["dataset"]["universe"]
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO experiments (id, name, created, status, config_hash, dataset_key, universe, model_kind, target, horizon, signal_method, trials_at_run, notes, artifact_dir, parent_id, "
                      "seconds, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (exp_id, cfg["name"], time.strftime("%Y-%m-%d %H:%M:%S"), status, config_hash, dataset_key, uni if isinstance(uni, str) else f"{len(uni)} tickers", cfg["model"]["kind"],
                       cfg["dataset"]["target"], int(cfg["dataset"]["horizon"]), cfg["signal"]["method"], int(trials_at_run), cfg.get("notes", ""), str(d), parent_id, float(seconds), source))

    def log_metrics(self, exp_id: str, metrics: dict) -> None:
        """``metrics`` is ``{segment: {name: value}}`` (segment ``all`` for the whole out-of-sample run); non-numeric and non-finite values are skipped."""
        rows = [(exp_id, seg, name, float(v)) for seg, d in metrics.items() for name, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")]
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM metrics WHERE experiment_id=?", (exp_id,))
            c.executemany("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?)", rows)
        (self.experiment_dir(exp_id) / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))

    def log_artifact(self, exp_id: str, name: str, path: Path, kind: str) -> None:
        data = Path(path).read_bytes()
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?)", (exp_id, name, str(path), kind, len(data), hashlib.sha256(data).hexdigest()[:16]))

    def log_prediction_summary(self, exp_id: str, segment: str, n: int, path: Path, ic: float | None) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?)", (exp_id, segment, int(n), str(path), None if ic is None else float(ic)))

    def set_status(self, exp_id: str, status: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE experiments SET status=? WHERE id=?", (status, exp_id))

    def get_experiment(self, exp_id: str) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
            if row is None:
                raise RegistryError(f"unknown experiment '{exp_id}'")
            metrics = c.execute("SELECT segment, name, value FROM metrics WHERE experiment_id=?", (exp_id,)).fetchall()
            arts = c.execute("SELECT name, path, kind, bytes FROM artifacts WHERE experiment_id=?", (exp_id,)).fetchall()
        out = dict(row)
        out["metrics"] = {}
        for m in metrics:
            out["metrics"].setdefault(m["segment"], {})[m["name"]] = m["value"]
        out["artifacts"] = [dict(a) for a in arts]
        return out

    def report(self, exp_id: str) -> dict:
        """The saved ``report.json`` of an experiment."""
        self.get_experiment(exp_id)                                         # an unknown id is reported as unknown, not as a missing file
        p = self.root / "experiments" / exp_id / "report.json"
        if not p.exists():
            raise RegistryError(f"experiment '{exp_id}' has no report")
        return json.loads(p.read_text())

    def list_experiments(self, limit: int = 200, universe: str | None = None, model_kind: str | None = None, stage: str | None = None, status: str = "done") -> list[dict]:
        q, args = "SELECT * FROM experiments WHERE status=?", [status]
        for col, val in (("universe", universe), ("model_kind", model_kind), ("stage", stage)):
            if val:
                q += f" AND {col}=?"
                args.append(val)
        q += " ORDER BY created DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def leaderboard(self, sort_by: str = "validation.sharpe", limit: int = 50, universe: str | None = None, descending: bool = True) -> list[dict]:
        """Experiments ranked by one metric (``segment.name``), each with its headline metrics and the trial count at the time it ran."""
        seg, _, name = sort_by.partition(".")
        rows = self.list_experiments(1000, universe)
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        with self._conn() as c:
            marks = ",".join("?" * len(ids))
            mets = c.execute(f"SELECT experiment_id, segment, name, value FROM metrics WHERE experiment_id IN ({marks})", ids).fetchall()
        by: dict[str, dict] = {}
        for m in mets:
            by.setdefault(m["experiment_id"], {})[f"{m['segment']}.{m['name']}"] = m["value"]
        out = [{**r, "metrics": by.get(r["id"], {})} for r in rows]
        out = [r for r in out if sort_by in r["metrics"]] + [r for r in out if sort_by not in r["metrics"]]
        out.sort(key=lambda r: (sort_by not in r["metrics"], -(r["metrics"].get(sort_by, 0.0)) if descending else r["metrics"].get(sort_by, 0.0)))
        return out[: int(limit)]

    def delete_experiment(self, exp_id: str) -> None:
        self.get_experiment(exp_id)
        with self._lock, self._conn() as c:
            for t in ("metrics", "artifacts", "predictions"):
                c.execute(f"DELETE FROM {t} WHERE experiment_id=?", (exp_id,))
            c.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
        shutil.rmtree(self.root / "experiments" / exp_id, ignore_errors=True)

    # ------------------------------------------------------------------------------------------------------------ trials
    def trial_count(self, dataset_key: str) -> int:
        """How many looks at this dataset have been recorded: finished experiments plus search trials. The next result's deflated Sharpe ratio counts them all."""
        with self._conn() as c:
            n_exp = c.execute("SELECT COUNT(*) FROM experiments WHERE dataset_key=? AND status='done'", (dataset_key,)).fetchone()[0]
            n_trials = c.execute("SELECT COUNT(*) FROM trials WHERE dataset_key=?", (dataset_key,)).fetchone()[0]
        return int(n_exp + n_trials)

    def log_trial(self, study: str, dataset_key: str, number: int, params: dict, value: float, seconds: float, stage: str = "random", experiment_id: str | None = None) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO trials (study, dataset_key, experiment_id, number, params, value, seconds, created, stage) VALUES (?,?,?,?,?,?,?,?,?)",
                      (study, dataset_key, experiment_id, int(number), json.dumps(params, default=str, sort_keys=True), None if value != value else float(value), float(seconds), time.strftime("%Y-%m-%d %H:%M:%S"), stage))

    def list_trials(self, study: str | None = None, dataset_key: str | None = None, limit: int = 500) -> list[dict]:
        q, args = "SELECT * FROM trials WHERE 1=1", []
        for col, val in (("study", study), ("dataset_key", dataset_key)):
            if val:
                q += f" AND {col}=?"
                args.append(val)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as c:
            return [{**dict(r), "params": json.loads(r["params"])} for r in c.execute(q, args).fetchall()]

    # ------------------------------------------------------------------------------------------------------------ models
    def register_model(self, exp_id: str, model_bytes: bytes, kind: str, features: list, stage: str = "candidate", notes: str = "") -> str:
        if stage not in STAGES:
            raise RegistryError(f"stage must be one of {STAGES}")
        self.get_experiment(exp_id)
        model_id = f"{kind}-{exp_id}"
        path = self.root / "models" / f"{model_id}.pkl"
        path.write_bytes(model_bytes)
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?,?)", (model_id, exp_id, kind, str(path), time.strftime("%Y-%m-%d %H:%M:%S"), stage, json.dumps(list(features)), notes))
        return model_id

    def list_models(self, stage: str | None = None) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM models" + (" WHERE stage=?" if stage else "") + " ORDER BY created DESC", ([stage] if stage else [])).fetchall()
        return [{**dict(r), "features": json.loads(r["features"])} for r in rows]

    def load_model_bytes(self, model_id: str) -> bytes:
        with self._conn() as c:
            row = c.execute("SELECT path FROM models WHERE id=?", (model_id,)).fetchone()
        if row is None:
            raise RegistryError(f"unknown model '{model_id}'")
        return Path(row["path"]).read_bytes()

    def promote(self, model_id: str, stage: str) -> None:
        """Move a model to a stage; at most one model per experiment universe may be ``production`` (the previous one is archived)."""
        if stage not in STAGES:
            raise RegistryError(f"stage must be one of {STAGES}")
        with self._lock, self._conn() as c:
            row = c.execute("SELECT experiment_id FROM models WHERE id=?", (model_id,)).fetchone()
            if row is None:
                raise RegistryError(f"unknown model '{model_id}'")
            if stage == "production":
                uni = c.execute("SELECT universe FROM experiments WHERE id=?", (row["experiment_id"],)).fetchone()["universe"]
                c.execute("UPDATE models SET stage='archived' WHERE stage='production' AND experiment_id IN (SELECT id FROM experiments WHERE universe=?)", (uni,))
            c.execute("UPDATE models SET stage=? WHERE id=?", (stage, model_id))
            c.execute("UPDATE experiments SET stage=? WHERE id=?", (stage, row["experiment_id"]))

    # ------------------------------------------------------------------------------------------------------------ feature sets
    def register_feature_set(self, dataset, dataset_key: str) -> str:
        fid = f"{dataset_key}-{dataset.meta.get('data_fingerprint', 'x')}"
        path = self.root / "features" / fid
        if not (path / "features.parquet").exists():
            dataset.save(path)
        s = dataset.summary()
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO feature_sets VALUES (?,?,?,?,?,?,?,?,?,?)", (fid, dataset_key, time.strftime("%Y-%m-%d %H:%M:%S"), str(path), s["n_rows"], s["n_features"], s["start"], s["end"],
                                                                                          str(dataset.meta.get("universe")), json.dumps({k: v for k, v in dataset.meta.items() if k != "skipped"}, default=str)))
        return fid

    def list_feature_sets(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT id, dataset_key, created, path, n_rows, n_features, start, end, universe FROM feature_sets ORDER BY created DESC").fetchall()]

    # ------------------------------------------------------------------------------------------------------------ forward signals
    def log_forward_signal(self, exp_id: str, signal_date, weights: dict, scores: dict, turnover: float) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute("INSERT INTO forward_signals (experiment_id, signal_date, created, weights, scores, turnover) VALUES (?,?,?,?,?,?)",
                            (exp_id, str(pd.Timestamp(signal_date).date()), time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(weights), json.dumps(scores), float(turnover)))
            return int(cur.lastrowid)

    def forward_signals(self, exp_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM forward_signals WHERE experiment_id=? ORDER BY signal_date", (exp_id,)).fetchall()
        return [{**dict(r), "weights": json.loads(r["weights"]), "scores": json.loads(r["scores"])} for r in rows]

    def set_realised(self, signal_id: int, realised: float, realised_date) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE forward_signals SET realised=?, realised_date=? WHERE id=?", (float(realised), str(pd.Timestamp(realised_date).date()), int(signal_id)))
