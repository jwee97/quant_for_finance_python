"""The experiment manager: one call runs a strategy, validates it, and records it where it can be compared.

    spec -> pipeline (walk-forward by construction) -> metrics + validation -> experiment database -> comparison

The database (``data/experiments.db``, local and persistent) is what makes the multiple-testing arithmetic honest: it knows how
many distinct ideas you have tried in a group, and uses that count as the number of trials in the deflated Sharpe ratio.
"""

from __future__ import annotations

from io import StringIO
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


def _forecast_long(result) -> pd.DataFrame:
    """Month-end forecasts (combined and per model) in long form: the model outputs a run produced, small enough to keep for every run."""
    frames = []
    panels = {**({"combined": result.forecasts} if result.forecasts is not None else {}), **(result.model_forecasts or {})}
    for name, panel in panels.items():
        if panel is None:
            continue
        idx = panel.mean.index
        month_end = idx.to_series().groupby(idx.to_period("M")).max()
        mean, std, conf = (f.reindex(month_end.to_numpy()).stack().rename(n) for f, n in ((panel.mean, "mean"), (panel.std, "std"), (panel.confidence, "confidence")))
        frame = pd.concat([mean, std, conf, panel.probability_up().reindex(month_end.to_numpy()).stack().rename("p_up")], axis=1).dropna(subset=["mean"]).reset_index()
        frame.columns = ["date", "asset", "mean", "std", "confidence", "p_up"]
        frame.insert(0, "model", name)
        frame[["mean", "std", "confidence", "p_up"]] = frame[["mean", "std", "confidence", "p_up"]].round(8)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _sweep_task(payload: tuple, config, bundle, validate: bool):
    """One sweep job: runs in a worker process, returns the result to the parent, which is the only writer of the database."""
    from . import load_library

    load_library()
    spec_dict, n_trials = payload
    spec = PipelineSpec.from_dict(spec_dict)
    spec.evaluation = {**spec.evaluation, "n_trials": n_trials}
    started = time.perf_counter()
    return Pipeline(spec, config, bundle).run(validate=validate), time.perf_counter() - started


def expand_grid(spec: dict, grid: dict[str, list]) -> list[dict]:
    """Cartesian product of dotted-path overrides, e.g. ``{"models.0.params.lookback": [63, 126], "combination.rule": ["equal", "ic_weighted"]}``.

    Each variant is named ``base[path=value,...]`` (the last path component and value), which is what the dashboard's parameter-sensitivity chart parses.
    """
    import copy
    import itertools

    keys = list(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        variant = copy.deepcopy(spec)
        for key, value in zip(keys, values):
            node = variant
            parts = key.split(".")
            for part in parts[:-1]:
                node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
            last = parts[-1]
            if isinstance(node, list):
                node[int(last)] = value
            else:
                node[last] = value
        label = ",".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, values))
        variant["name"] = f"{spec.get('name', 'strategy')}[{label}]"
        out.append(variant)
    return out


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
        run_id = self._new_run_id(h)
        self.record(run_id, spec, result, bundle, h, group, data_key, time.perf_counter() - started)
        if tearsheet:
            from .tearsheet import make_tearsheet
            make_tearsheet(result, Path(tearsheet_dir) if tearsheet_dir else self.config.root / "reports" / "tearsheets" / run_id, run_id)
        return run_id, result

    def sweep(self, specs: list, bundle, validate: bool = True, backend: str = "joblib", n_jobs: int = -1, force: bool = False) -> list[str]:
        """Run many specifications in parallel (the Generation 4 executor), then record them one by one.

        Workers compute; only the parent writes to the database, so SQLite never sees concurrent writers. Every spec in a group counts as a trial for the
        deflated Sharpe ratio of all of them, including the ones in this sweep, so a grid of 20 variants is deflated as 20 trials.
        """
        from functools import partial

        from ..distributed.executor import run_tasks

        prepared = []
        for spec in specs:
            spec = spec if isinstance(spec, PipelineSpec) else PipelineSpec.from_dict(spec)
            data_key = f"{bundle.name}|{bundle.prices.shape}|{bundle.index[0].date()}|{bundle.index[-1].date()}|{bundle.as_market().data_version}"
            prepared.append((spec, spec_hash(spec, data_key), str(spec.evaluation.get("group") or spec.name), data_key))
        batch_hashes: dict[str, set] = {}
        for _, h, group, _ in prepared:
            batch_hashes.setdefault(group, set()).add(h)
        with self._connect() as con:
            known = {(r[0], r[1]): r[2] for r in con.execute("SELECT spec_hash, config_fingerprint, run_id FROM runs")}
        todo = [(i, p) for i, p in enumerate(prepared) if force or (p[1], self.config.fingerprint("gen5")) not in known]
        run_ids = {i: known[(p[1], self.config.fingerprint("gen5"))] for i, p in enumerate(prepared) if (p[1], self.config.fingerprint("gen5")) in known and not force}
        payloads = []
        for _, (spec, h, group, _) in todo:
            with self._connect() as con:
                existing = {r[0] for r in con.execute("SELECT DISTINCT spec_hash FROM runs WHERE group_name = ?", (group,))}
            payloads.append((spec.to_dict(), len(existing | batch_hashes[group])))
        results = run_tasks(partial(_sweep_task, config=self.config, bundle=bundle, validate=validate), payloads, backend, n_jobs, 1)
        for (i, (spec, h, group, data_key)), (result, runtime) in zip(todo, results):
            run_id = self._new_run_id(h)
            spec.evaluation = {**spec.evaluation, "n_trials": result.validation["n_trials"]}
            self.record(run_id, spec, result, bundle, h, group, data_key, runtime)
            run_ids[i] = run_id
        return [run_ids[i] for i in range(len(prepared))]

    def _new_run_id(self, h: str) -> str:
        """``<UTC timestamp>-<spec hash>``; a forced re-run inside the same second gets a ``-2``, ``-3`` ... suffix instead of colliding."""
        base = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{h[:8]}"
        with self._connect() as con:
            taken = {r[0] for r in con.execute("SELECT run_id FROM runs WHERE run_id LIKE ?", (base + "%",))}
        run_id, n = base, 1
        while run_id in taken:
            n += 1
            run_id = f"{base}-{n}"
        return run_id

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
        forecasts = _forecast_long(result)
        if len(forecasts):
            forecasts.to_csv(folder / "forecasts.csv.gz", index=False)
        (folder / "meta.json").write_text(json.dumps({
            "run_id": run_id, "name": spec.name, "group": group, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(self.config.root),
            "spec_hash": h, "data_key": data_key, "config_fingerprints": {scope: self.config.fingerprint(scope) for scope in ("core", "all", "gen3", "gen4", "gen5")},
            "spec": spec.to_dict(), "metrics": {k: _scalar(x) for k, x in m.items() if _scalar(x) is not None}, "validation": {k: x for k, x in v.items() if k != "causality"},
            "causality_ok": ok, "runtime_s": runtime, "files": sorted(p.name for p in folder.iterdir())}, indent=1, default=str))

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
        order = {r: i for i, r in enumerate(run_ids)}
        return frame.assign(_o=frame["run_id"].map(order)).sort_values("_o")[keep].set_index("run_id")      # in the order asked, not the database's

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
        return pd.read_json(StringIO(row[0]), orient="records") if row else pd.DataFrame()
