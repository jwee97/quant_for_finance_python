"""The research database (Generation 4, Priority 14).

The experiment record already exists as files: a JSON-lines registry, one CSV per table, one PNG and caption per
figure. This module loads it into SQLite (standard library; no server) and verifies that nothing was lost or
altered in the move. The database is derived: it is rebuilt from the files on every run and is not committed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
STANDARD_METRICS = ("sharpe", "cagr", "ann_vol", "max_drawdown", "ann_turnover", "sortino", "calmar")

DDL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE experiments (experiment_id TEXT PRIMARY KEY, stage TEXT, hypothesis TEXT, decision TEXT, test_period TEXT, train_period TEXT,
                          config_fingerprint TEXT, data_version TEXT, date TEXT, notes TEXT, parameters TEXT);
CREATE TABLE metrics (experiment_id TEXT, key TEXT, value REAL);
CREATE TABLE result_files (path TEXT PRIMARY KEY, stage TEXT, n_rows INTEGER, n_columns INTEGER, sha256 TEXT);
CREATE TABLE figures (number INTEGER, file TEXT PRIMARY KEY, question TEXT, stage TEXT);
CREATE TABLE books (source TEXT, stage TEXT, name TEXT, sharpe REAL, cagr REAL, ann_vol REAL, max_drawdown REAL, ann_turnover REAL, sortino REAL, calmar REAL);
CREATE TABLE grid_results (id INTEGER PRIMARY KEY, family TEXT, variant TEXT, lookback INTEGER, skip INTEGER, vol_lookback INTEGER, sign INTEGER,
                           rebalance TEXT, signal_lag INTEGER, cross_sectional INTEGER, gross_sharpe REAL, ann_turnover REAL, mean_ic_21d REAL,
                           sharpe REAL, cagr REAL, ann_vol REAL, max_drawdown REAL);
CREATE INDEX idx_metrics ON metrics (experiment_id);
CREATE INDEX idx_books_sharpe ON books (sharpe);
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_of(name: str) -> str | None:
    match = re.match(r"(stage\d+)", name)
    return match.group(1) if match else None


def load_registry(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _numeric(value) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value)
    return None


def build_database(root: str | Path, db_path: str | Path) -> dict:
    """Create the database from the files under ``root`` and return the row counts."""
    root, db_path = Path(root), Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    tables_dir, figures_dir = root / "reports" / "tables", root / "reports" / "figures"
    connection = sqlite3.connect(db_path)
    connection.executescript(DDL)
    connection.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    entries = load_registry(root / "experiments" / "registry.jsonl")
    for e in entries:
        connection.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (e["experiment_id"], e["stage"], e["hypothesis"], e["decision"], e.get("test_period"), e.get("train_period"),
                            e.get("config_fingerprint"), e.get("data_version"), e.get("date"), e.get("notes"), json.dumps(e.get("parameters", {}), sort_keys=True)))
        for key, value in (e.get("results") or {}).items():
            number = _numeric(value)
            if number is not None:
                connection.execute("INSERT INTO metrics VALUES (?,?,?)", (e["experiment_id"], key, number))

    sources = {p.stem: p for p in (root / "experiments").glob("stage*.py")}
    for csv in sorted(tables_dir.glob("*.csv")):
        frame = pd.read_csv(csv)
        connection.execute("INSERT INTO result_files VALUES (?,?,?,?,?)", (csv.name, stage_of(csv.name), len(frame), frame.shape[1], sha256_file(csv)))
        if "sharpe" in frame.columns and frame.shape[0] and frame.columns[0] != "sharpe":
            first = frame.columns[0]
            if frame[first].map(lambda v: isinstance(v, str)).all():
                for _, row in frame.iterrows():
                    sharpe = _numeric(row["sharpe"])
                    if sharpe is None:
                        continue
                    values = [_numeric(row[m]) if m in frame.columns else None for m in STANDARD_METRICS]
                    connection.execute("INSERT INTO books VALUES (?,?,?,?,?,?,?,?,?,?)", (csv.name, stage_of(csv.name), str(row[first]), *values))

    for caption in sorted(figures_dir.glob("*.txt")):
        text = caption.read_text(encoding="utf-8").strip()
        match = re.match(r"Figure (\d+)\.\s*(.*)", text, flags=re.DOTALL)
        number, question = (int(match.group(1)), match.group(2)) if match else (0, text)
        png = caption.with_suffix(".png").name
        owner = next((stage for stage, src in sources.items() if f"{caption.stem[:5]}_" in src.read_text(encoding="utf-8") or caption.name[:-4] in src.read_text(encoding="utf-8")), None)
        connection.execute("INSERT INTO figures VALUES (?,?,?,?)", (number, png, question, owner))

    grid = tables_dir / "stage26_grid_results.csv"
    if grid.exists():
        g = pd.read_csv(grid)
        columns = ["id", "family", "variant", "lookback", "skip", "vol_lookback", "sign", "rebalance", "signal_lag", "cross_sectional", "gross_sharpe",
                   "ann_turnover", "mean_ic_21d", "sharpe", "cagr", "ann_vol", "max_drawdown"]
        g = g.reindex(columns=columns)
        g["cross_sectional"] = g["cross_sectional"].astype(float).astype("Int64")
        connection.executemany(f"INSERT INTO grid_results VALUES ({','.join('?' * len(columns))})",
                               [tuple(None if pd.isna(v) else (v.item() if hasattr(v, 'item') else v) for v in row) for row in g.itertuples(index=False)])
    connection.commit()
    counts = {t: connection.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("experiments", "metrics", "result_files", "figures", "books", "grid_results")}
    connection.close()
    return counts


def verify_database(root: str | Path, db_path: str | Path) -> dict[str, bool]:
    """The declared acceptance checks. Every value must be True."""
    root, db_path = Path(root), Path(db_path)
    from ..assistant.sqlguard import open_readonly

    connection = open_readonly(db_path)
    checks: dict[str, bool] = {}
    entries = load_registry(root / "experiments" / "registry.jsonl")
    ids = {r[0] for r in connection.execute("SELECT experiment_id FROM experiments")}
    checks["every registry entry is present"] = ids == {e["experiment_id"] for e in entries}
    stored = {(r[0], r[1]): r[2] for r in connection.execute("SELECT experiment_id, key, value FROM metrics")}
    expected = {(e["experiment_id"], k): _numeric(v) for e in entries for k, v in (e.get("results") or {}).items() if _numeric(v) is not None}
    checks["every numeric result is in metrics with the same value"] = (set(stored) == set(expected) and all(stored[k] == expected[k] for k in expected))
    files = {r[0]: r[1] for r in connection.execute("SELECT path, sha256 FROM result_files")}
    disk = {p.name: sha256_file(p) for p in (root / "reports" / "tables").glob("*.csv")}
    checks["every CSV is indexed and its sha256 matches"] = files == disk
    figs = {r[0] for r in connection.execute("SELECT file FROM figures")}
    checks["every figure caption is in figures"] = figs == {p.with_suffix(".png").name for p in (root / "reports" / "figures").glob("*.txt")}
    grid_file = root / "reports" / "tables" / "stage26_grid_results.csv"
    if grid_file.exists():
        direct = pd.read_csv(grid_file)
        direct = direct[(direct["family"] == "momentum") & (direct["ann_turnover"] < 8)].sort_values("sharpe", ascending=False, kind="stable")
        sql = pd.read_sql_query("SELECT * FROM grid_results WHERE family='momentum' AND ann_turnover<8 ORDER BY sharpe DESC, id", connection)
        direct = direct.sort_values(["sharpe", "id"], ascending=[False, True])
        checks["the roadmap query equals a direct pandas filter, in order"] = (list(sql["id"]) == list(direct["id"]) and bool(np.allclose(sql["sharpe"], direct["sharpe"])))
    try:
        connection.execute("INSERT INTO meta VALUES ('x', 'y')")
        checks["a write through the read-only connection fails"] = False
    except sqlite3.OperationalError:
        checks["a write through the read-only connection fails"] = True
    connection.close()
    return checks
