"""A content-addressed artifact store and a model registry.

**ArtifactStore.** Objects are stored under the SHA-256 of their serialised bytes (``root/ab/cdef...``), so identical content is stored once, a reference is a proof of what was saved
(``get`` re-hashes the bytes and refuses a mismatch, which also detects tampering before anything is deserialised), and a result can be reproduced from its recorded hash alone. Kinds:
``json`` (dicts and lists), ``npz`` (dict of arrays), ``frame`` (a pandas DataFrame/Series, pickled: only open stores you trust, the hash check protects integrity, not provenance) and ``bytes``.

**ModelRegistry.** A SQLite table of versioned models: what was trained (``params``), on which data (``data_key``) with which code (``git_commit``) and settings (``config_fingerprint``),
how it scored (``metrics``), where its artifact is, what it was derived from (``parent``), and its lifecycle ``stage`` (``staging`` -> ``production`` -> ``archived``). Promoting a version to production
archives the previous production version, so there is never more than one. ``register_run`` records a recorded experiment (a pipeline spec) as a model version, which makes strategies
first-class registered models next to trained ones.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("json", "npz", "frame", "bytes")
STAGES = ("staging", "production", "archived")


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    kind: str
    size: int

    def to_dict(self) -> dict:
        return {"sha256": self.sha256, "kind": self.kind, "size": self.size}


def _serialise(obj, kind: str) -> bytes:
    if kind == "json":
        return json.dumps(obj, sort_keys=True, default=_json_default).encode()
    if kind == "npz":
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: np.asarray(v) for k, v in obj.items()})
        return buf.getvalue()
    if kind == "frame":
        buf = io.BytesIO()
        pickle.dump(obj, buf, protocol=4)
        return buf.getvalue()
    if kind == "bytes":
        return bytes(obj)
    raise ValueError(f"kind must be one of {KINDS}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha: str) -> Path:
        return self.root / sha[:2] / sha[2:]

    def put(self, obj, kind: str = "json") -> ArtifactRef:
        data = _serialise(obj, kind)
        sha = hashlib.sha256(data).hexdigest()
        path = self._path(sha)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)                                               # atomic: a reader never sees a half-written artifact
        return ArtifactRef(sha, kind, len(data))

    def get(self, ref: ArtifactRef | dict | str, kind: str | None = None):
        sha = ref if isinstance(ref, str) else (ref["sha256"] if isinstance(ref, dict) else ref.sha256)
        kind = kind or (ref["kind"] if isinstance(ref, dict) else getattr(ref, "kind", None))
        if kind is None:
            raise ValueError("pass kind when loading by hash string")
        path = self._path(sha)
        if not path.exists():
            raise KeyError(f"artifact {sha[:12]} is not in the store")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"artifact {sha[:12]} is corrupted: its content no longer matches its hash")
        if kind == "json":
            return json.loads(data)
        if kind == "npz":
            with np.load(io.BytesIO(data)) as z:
                return {k: z[k] for k in z.files}
        if kind == "frame":
            return pickle.loads(data)       # noqa: S301 - integrity verified by the hash above; the store is trusted local storage
        return data

    def exists(self, ref) -> bool:
        sha = ref if isinstance(ref, str) else (ref["sha256"] if isinstance(ref, dict) else ref.sha256)
        return self._path(sha).exists()

    def verify(self) -> list[str]:
        """Re-hash every stored object and return the hashes that no longer match (an empty list means the store is intact)."""
        bad = []
        for p in self.root.glob("*/*"):
            if p.suffix == ".tmp":
                continue
            sha = p.parent.name + p.name
            if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
                bad.append(sha)
        return bad

    def list(self) -> list[str]:
        return sorted(p.parent.name + p.name for p in self.root.glob("*/*") if p.suffix != ".tmp")

    def gc(self, keep: set[str]) -> int:
        """Delete every object whose hash is not in ``keep``; returns the number removed."""
        n = 0
        for sha in self.list():
            if sha not in keep:
                self._path(sha).unlink()
                n += 1
        return n


DDL = """
CREATE TABLE IF NOT EXISTS models (
    name TEXT NOT NULL, version INTEGER NOT NULL, stage TEXT NOT NULL, created TEXT NOT NULL, description TEXT, params TEXT, metrics TEXT, data_key TEXT,
    config_fingerprint TEXT, git_commit TEXT, parent_version INTEGER, artifact TEXT, tags TEXT, PRIMARY KEY (name, version));
CREATE TABLE IF NOT EXISTS stage_log (name TEXT, version INTEGER, stage TEXT, at TEXT, note TEXT);
"""


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class ModelRegistry:
    def __init__(self, db_path: str | Path, store: ArtifactStore | None = None, root: Path | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = store or ArtifactStore(self.db_path.parent / "artifacts")
        self.root = root or self.db_path.parent
        with self._con() as con:
            con.executescript(DDL)

    def _con(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def register(self, name: str, params: dict | None = None, artifact=None, artifact_kind: str = "json", metrics: dict | None = None, data_key: str = "",
                 config_fingerprint: str = "", parent_version: int | None = None, tags: list[str] | None = None, description: str = "", stage: str = "staging") -> int:
        """Record a new version of ``name`` (the next integer) and return it. ``artifact`` (a fitted model's parameters, weights, a table...) goes into the store."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        ref = self.store.put(artifact, artifact_kind).to_dict() if artifact is not None else None
        with self._con() as con:
            last = con.execute("SELECT COALESCE(MAX(version), 0) FROM models WHERE name = ?", (name,)).fetchone()[0]
            if parent_version is not None and not con.execute("SELECT 1 FROM models WHERE name = ? AND version = ?", (name, parent_version)).fetchone():
                raise KeyError(f"parent version {parent_version} of '{name}' does not exist")
            version = int(last) + 1
            con.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (name, version, "staging", datetime.now(timezone.utc).isoformat(timespec="seconds"), description, json.dumps(params or {}, sort_keys=True, default=_json_default),
                         json.dumps(metrics or {}, sort_keys=True, default=_json_default), data_key, config_fingerprint, _git_commit(self.root), parent_version,
                         json.dumps(ref) if ref else None, json.dumps(tags or [])))
            con.execute("INSERT INTO stage_log VALUES (?,?,?,?,?)", (name, version, "staging", datetime.now(timezone.utc).isoformat(timespec="seconds"), "registered"))
        if stage != "staging":
            self.promote(name, version, stage)
        return version

    def register_run(self, manager, run_id: str, name: str | None = None, **kwargs) -> int:
        """Register a recorded experiment run as a model version: the spec as ``params``, the headline metrics, the data key and the config fingerprint of the run."""
        import sqlite3 as _sq

        spec = manager.spec_of(run_id)
        with _sq.connect(manager.db_path) as con:
            row = con.execute("SELECT sharpe, cagr, ann_vol, max_drawdown, ann_turnover, deflated_sharpe_probability, n_trials, data_version, config_fingerprint FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        keys = ("sharpe", "cagr", "ann_vol", "max_drawdown", "ann_turnover", "deflated_sharpe_probability", "n_trials")
        metrics = {k: v for k, v in zip(keys, row[:7])} | {"run_id": run_id}
        return self.register(name or spec.name, params=spec.to_dict(), metrics=metrics, data_key=str(row[7]), config_fingerprint=str(row[8]), description=f"recorded run {run_id}", **kwargs)

    def _row(self, con, name, version):
        return con.execute("SELECT name, version, stage, created, description, params, metrics, data_key, config_fingerprint, git_commit, parent_version, artifact, tags FROM models "
                           "WHERE name = ? AND version = ?", (name, version)).fetchone()

    def resolve(self, name: str, version: int | str | None = None) -> int:
        """A version number from an explicit number, a stage name (``production`` / ``staging``) or ``None`` (production, else the latest)."""
        with self._con() as con:
            if isinstance(version, int):
                return version
            if version in STAGES:
                row = con.execute("SELECT MAX(version) FROM models WHERE name = ? AND stage = ?", (name, version)).fetchone()[0]
            else:
                row = (con.execute("SELECT MAX(version) FROM models WHERE name = ? AND stage = 'production'", (name,)).fetchone()[0]
                       or con.execute("SELECT MAX(version) FROM models WHERE name = ?", (name,)).fetchone()[0])
        if row is None:
            raise KeyError(f"no model '{name}'" + (f" at stage '{version}'" if version in STAGES else ""))
        return int(row)

    def get(self, name: str, version: int | str | None = None) -> dict:
        v = self.resolve(name, version)
        with self._con() as con:
            r = self._row(con, name, v)
        if r is None:
            raise KeyError(f"model '{name}' has no version {v}")
        keys = ("name", "version", "stage", "created", "description", "params", "metrics", "data_key", "config_fingerprint", "git_commit", "parent_version", "artifact", "tags")
        d = dict(zip(keys, r))
        for k in ("params", "metrics", "artifact", "tags"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def load(self, name: str, version: int | str | None = None):
        """The artifact of a version (production by default), integrity-checked by the store."""
        rec = self.get(name, version)
        if rec["artifact"] is None:
            raise ValueError(f"{name} v{rec['version']} has no artifact")
        return self.store.get(rec["artifact"])

    def promote(self, name: str, version: int, stage: str, note: str = "") -> None:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._con() as con:
            if self._row(con, name, version) is None:
                raise KeyError(f"model '{name}' has no version {version}")
            if stage == "production":
                for (old,) in con.execute("SELECT version FROM models WHERE name = ? AND stage = 'production' AND version != ?", (name, version)).fetchall():
                    con.execute("UPDATE models SET stage = 'archived' WHERE name = ? AND version = ?", (name, old))
                    con.execute("INSERT INTO stage_log VALUES (?,?,?,?,?)", (name, old, "archived", now, f"superseded by v{version}"))
            con.execute("UPDATE models SET stage = ? WHERE name = ? AND version = ?", (stage, name, version))
            con.execute("INSERT INTO stage_log VALUES (?,?,?,?,?)", (name, version, stage, now, note))

    def lineage(self, name: str, version: int | None = None) -> list[dict]:
        """The chain of versions this one was derived from, newest first."""
        v = self.resolve(name, version)
        out = []
        while v is not None:
            rec = self.get(name, v)
            out.append({"version": rec["version"], "stage": rec["stage"], "created": rec["created"], "metrics": rec["metrics"]})
            v = rec["parent_version"]
        return out

    def versions(self, name: str | None = None) -> pd.DataFrame:
        """All versions with their stage and headline metrics, as a table."""
        with self._con() as con:
            rows = con.execute("SELECT name, version, stage, created, metrics, data_key, git_commit, parent_version, description FROM models" + (" WHERE name = ?" if name else "")
                               + " ORDER BY name, version", (name,) if name else ()).fetchall()
        out = pd.DataFrame(rows, columns=["name", "version", "stage", "created", "metrics", "data_key", "git_commit", "parent_version", "description"])
        if len(out):
            metric_frame = pd.DataFrame([json.loads(m) for m in out["metrics"]])
            out = pd.concat([out.drop(columns="metrics"), metric_frame], axis=1)
        return out

    def history(self, name: str) -> pd.DataFrame:
        with self._con() as con:
            return pd.read_sql_query("SELECT version, stage, at, note FROM stage_log WHERE name = ? ORDER BY at, rowid", con, params=(name,))
