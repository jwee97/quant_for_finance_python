"""The experiment manager: one call runs a strategy, validates it, and records it where it can be compared.

    spec -> pipeline (walk-forward by construction) -> metrics + validation -> experiment database -> comparison

The database (``data/experiments.db``, local and persistent) is what makes the multiple-testing arithmetic honest: it knows how
many distinct ideas you have tried in a group, and uses that count as the number of trials in the deflated Sharpe ratio.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import Pipeline, PipelineResult, PipelineSpec

DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, name TEXT, group_name TEXT, created TEXT, git_commit TEXT, spec_hash TEXT, spec_json TEXT,
    data_name TEXT, data_version TEXT, config_fingerprint TEXT, start TEXT, end TEXT, n_days INTEGER,
    sharpe REAL, cagr REAL, ann_vol REAL, max_drawdown REAL, ann_turnover REAL, gross_sharpe REAL, ann_cost_bps REAL,
    deflated_sharpe_probability REAL, n_trials INTEGER, causality_ok INTEGER, runtime_s REAL, status TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS run_metrics (run_id TEXT, key TEXT, value REAL);
CREATE TABLE IF NOT EXISTS run_tables (run_id TEXT, name TEXT, json TEXT);
CREATE INDEX IF NOT EXISTS idx_runs_group ON runs (group_name);
CREATE INDEX IF NOT EXISTS idx_runs_spec ON runs (spec_hash);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON run_metrics (run_id);
"""


def spec_hash(spec: PipelineSpec, data_key: str) -> str:
    body = {k: v for k, v in spec.to_dict().items() if k != "notes"}
    body["evaluation"] = {k: v for k, v in (body.get("evaluation") or {}).items() if k != "n_trials"}       # the count is an output of the database, not an input
    payload = json.dumps({"spec": body, "data": data_key}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _scalar(value):
    return float(value) if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) else None


class ExperimentManager:
    def __init__(self, config, db_path: str | Path | None = None, runs_dir: str | Path | None = None):
        self.config = config
        node = config.get("framework.experiment_db", {}) or {}
        self.db_path = Path(db_path) if db_path else config.root / str(node.get("path", "data/experiments.db"))
        self.runs_dir = Path(runs_dir) if runs_dir else config.root / str(node.get("runs_dir", "data/experiments"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(DDL)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------------------------------------ accounting
    def n_trials(self, group: str, extra_hash: str | None = None) -> int:
        """Distinct specs tried in the group, counting the one about to be recorded."""
        with self._connect() as con:
            hashes = {r[0] for r in con.execute("SELECT DISTINCT spec_hash FROM runs WHERE group_name = ?", (group,))}
        if extra_hash:
            hashes.add(extra_hash)
        return max(len(hashes), 1)

    # ------------------------------------------------------------------------------------------------- running
    def run(self, spec: PipelineSpec | dict, bundle, tearsheet: bool = False, force: bool = False, validate: bool = True,
            tearsheet_dir: str | Path | None = None) -> tuple[str, PipelineResult | None]:
        spec = spec if isinstance(spec, PipelineSpec) else PipelineSpec.from_dict(spec)
        data_key = f"{bundle.name}|{bundle.prices.shape}|{bundle.index[0].date()}|{bundle.index[-1].date()}|{bundle.as_market().data_version}"
        h = spec_hash(spec, data_key)
        group = str(spec.evaluation.get("group") or spec.name)
        if not force:
            with self._connect() as con:
                row = con.execute("SELECT run_id FROM runs WHERE spec_hash = ? AND config_fingerprint = ? ORDER BY created DESC LIMIT 1",
                                  (h, self.config.fingerprint("gen5"))).fetchone()
            if row and not tearsheet:
                return row[0], None
        spec.evaluation = {**spec.evaluation, "n_trials": self.n_trials(group, h)}
        started = time.perf_counter()
        result = Pipeline(spec, self.config, bundle).run(validate=validate)
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{h[:8]}"
        self.record(run_id, spec, result, bundle, h, group, data_key, time.perf_counter() - started)
        if tearsheet:
            from .tearsheet import make_tearsheet
            make_tearsheet(result, Path(tearsheet_dir) if tearsheet_dir else self.config.root / "reports" / "tearsheets" / run_id, run_id)
        return run_id, result

    def record(self, run_id: str, spec: PipelineSpec, result: PipelineResult, bundle, h: str, group: str, data_key: str, runtime: float) -> None:
        m, v = result.metrics, result.validation
        causality = v.get("causality", {})
        ok = None if not causality else int(all(c["ok"] for c in causality.values()))
        with self._connect() as con:
            con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (run_id, spec.name, group, datetime.now(timezone.utc).isoformat(timespec="seconds"), git_commit(self.config.root), h,
                         json.dumps(spec.to_dict(), default=str, sort_keys=True), bundle.name, bundle.as_market().data_version, self.config.fingerprint("gen5"),
                         m.get("start"), m.get("end"), m.get("n_days"), _scalar(m.get("sharpe")), _scalar(m.get("cagr")), _scalar(m.get("ann_vol")),
                         _scalar(m.get("max_drawdown")), _scalar(m.get("ann_turnover")), _scalar(m.get("gross_sharpe")), _scalar(m.get("ann_cost_bps")),
                         _scalar(v.get("deflated_sharpe_probability")), v.get("n_trials"), ok, runtime, "ok", spec.notes))
            con.executemany("INSERT INTO run_metrics VALUES (?,?,?)", [(run_id, k, _scalar(x)) for k, x in m.items() if _scalar(x) is not None])
            for name, table in result.tables.items():
                frame = table if isinstance(table, pd.DataFrame) else table.to_frame()
                con.execute("INSERT INTO run_tables VALUES (?,?,?)", (run_id, name, frame.reset_index().to_json(orient="records", date_format="iso")))
            con.execute("INSERT INTO run_tables VALUES (?,?,?)", (run_id, "causality", json.dumps(causality)))
        folder = self.runs_dir / run_id
        folder.mkdir(parents=True, exist_ok=True)
        result.net_returns.rename("net").to_frame().assign(**{f"bench_{k}": v for k, v in result.benchmarks.items()}).to_csv(folder / "returns.csv.gz", index_label="date")
        result.weights.round(6).to_csv(folder / "weights.csv.gz", index_label="date")
        if result.regimes is not None:
            result.regimes.probabilities.round(6).to_csv(folder / "regimes.csv.gz", index_label="date")

    # ------------------------------------------------------------------------------------------------ queries
    def leaderboard(self, group: str | None = None, top: int = 20, by: str = "sharpe") -> pd.DataFrame:
        cols = {"sharpe", "deflated_sharpe_probability", "cagr", "ann_vol", "max_drawdown", "ann_turnover", "created"}
        if by not in cols:
            raise ValueError(f"sort by one of {sorted(cols)}")
        sql = ("SELECT run_id, name, group_name, start, end, sharpe, cagr, ann_vol, max_drawdown, ann_turnover, gross_sharpe, deflated_sharpe_probability, "
               "n_trials, causality_ok FROM runs" + (" WHERE group_name = ?" if group else "") + f" ORDER BY {by} DESC LIMIT {int(top)}")
        with self._connect() as con:
            return pd.read_sql_query(sql, con, params=(group,) if group else ())

    def compare(self, run_ids: list[str]) -> pd.DataFrame:
        marks = ",".join("?" * len(run_ids))
        with self._connect() as con:
            frame = pd.read_sql_query(f"SELECT * FROM runs WHERE run_id IN ({marks})", con, params=run_ids)
        keep = ["run_id", "name", "start", "end", "sharpe", "gross_sharpe", "cagr", "ann_vol", "max_drawdown", "ann_turnover", "ann_cost_bps", "deflated_sharpe_probability", "n_trials"]
        return frame[keep].set_index("run_id")

    def spec_of(self, run_id: str) -> PipelineSpec:
        with self._connect() as con:
            row = con.execute("SELECT spec_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no run '{run_id}'")
        payload = json.loads(row[0])
        return PipelineSpec.from_dict(payload)

    def returns_of(self, run_id: str) -> pd.DataFrame:
        return pd.read_csv(self.runs_dir / run_id / "returns.csv.gz", index_col="date", parse_dates=["date"])

    def table_of(self, run_id: str, name: str) -> pd.DataFrame:
        with self._connect() as con:
            row = con.execute("SELECT json FROM run_tables WHERE run_id = ? AND name = ?", (run_id, name)).fetchone()
        return pd.read_json(row[0], orient="records") if row else pd.DataFrame()
