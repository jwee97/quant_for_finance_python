"""The research database: nothing lost or altered in the move, and the verifier notices when something is (Generation 4)."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.assistant.sqlguard import ResearchAssistant
from src.research_db.builder import build_database, load_registry, sha256_file, stage_of, verify_database


def _repo(root):
    (root / "experiments").mkdir()
    (root / "reports" / "tables").mkdir(parents=True)
    (root / "reports" / "figures").mkdir()
    entries = [
        {"experiment_id": "EXP-001", "stage": "stage06_backtest", "hypothesis": "h1", "decision": "reject", "test_period": "p", "config_fingerprint": "f", "data_version": "d",
         "date": "2026-01-01", "notes": "n", "parameters": {"a": 1}, "results": {"sharpe": 0.31, "label": "text", "ok": True, "bad": float("nan")}},
        {"experiment_id": "EXP-002", "stage": "stage26_distributed", "hypothesis": "h2", "decision": "retain", "parameters": {}, "results": {"p": 0.04}},
    ]
    (root / "experiments" / "registry.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    (root / "experiments" / "stage26_distributed.py").write_text('save_figure("fig52_search.png")')
    pd.DataFrame({"name": ["M1", "M3"], "sharpe": [0.85, 0.31], "cagr": [0.05, 0.02], "ann_turnover": [0.5, 9.1]}).to_csv(root / "reports" / "tables" / "stage12_final.csv", index=False)
    pd.DataFrame({"id": [0, 1, 2, 3], "family": ["momentum"] * 3 + ["mean_reversion"], "variant": ["raw"] * 4, "lookback": [21, 63, 126, 5], "skip": [1, 1, 1, None],
                  "vol_lookback": [None] * 4, "sign": [None, None, None, -1], "rebalance": ["monthly"] * 4, "signal_lag": [1] * 4, "cross_sectional": [True, True, False, True],
                  "gross_sharpe": [0.4, 0.5, 0.2, 0.1], "ann_turnover": [5.0, 12.0, 6.0, 20.0], "mean_ic_21d": [0.01] * 4, "sharpe": [0.3, 0.5, 0.3, -0.2],
                  "cagr": [0.01] * 4, "ann_vol": [0.06] * 4, "max_drawdown": [-0.1] * 4}).to_csv(root / "reports" / "tables" / "stage26_grid_results.csv", index=False)
    (root / "reports" / "figures" / "fig52_search.txt").write_text("Figure 52. What does the search show?\n")
    (root / "reports" / "figures" / "fig52_search.png").write_bytes(b"png")


def test_build_and_verify_a_small_repository(tmp_path):
    _repo(tmp_path)
    counts = build_database(tmp_path, tmp_path / "db" / "r.db")
    assert counts == {"experiments": 2, "metrics": 3, "result_files": 2, "figures": 1, "books": 2, "grid_results": 4}
    checks = verify_database(tmp_path, tmp_path / "db" / "r.db")
    assert checks and all(checks.values()), checks
    connection = sqlite3.connect(tmp_path / "db" / "r.db")
    assert connection.execute("select stage from figures").fetchone()[0] == "stage26_distributed"
    assert connection.execute("select value from metrics where key='ok'").fetchone()[0] == 1.0
    assert connection.execute("select count(*) from metrics where key in ('label','bad')").fetchone()[0] == 0        # only finite numbers are stored
    ids = [r[0] for r in connection.execute("SELECT id FROM grid_results WHERE family='momentum' AND ann_turnover<8 ORDER BY sharpe DESC, id")]
    assert ids == [0, 2]
    connection.close()


def test_verifier_notices_a_changed_table_a_new_entry_and_a_missing_figure(tmp_path):
    _repo(tmp_path)
    db = tmp_path / "r.db"
    build_database(tmp_path, db)
    (tmp_path / "reports" / "tables" / "stage12_final.csv").write_text("name,sharpe\nM1,0.9\n")
    assert not verify_database(tmp_path, db)["every CSV is indexed and its sha256 matches"]
    build_database(tmp_path, db)
    with (tmp_path / "experiments" / "registry.jsonl").open("a") as handle:
        handle.write(json.dumps({"experiment_id": "EXP-003", "stage": "stage01_data", "hypothesis": "h", "decision": "record", "results": {}}) + "\n")
    assert not verify_database(tmp_path, db)["every registry entry is present"]
    build_database(tmp_path, db)
    (tmp_path / "reports" / "figures" / "fig99_new.txt").write_text("Figure 99. New?\n")
    assert not verify_database(tmp_path, db)["every figure caption is in figures"]


def test_rebuild_is_idempotent_and_the_assistant_reads_the_result(tmp_path):
    _repo(tmp_path)
    db = tmp_path / "r.db"
    first, second = build_database(tmp_path, db), build_database(tmp_path, db)
    assert first == second
    assistant = ResearchAssistant(db)
    assert list(assistant.ask("which hypotheses were retained?").rows["experiment_id"]) == ["EXP-002"]
    assert list(assistant.ask("momentum rules with turnover below 8").rows["id"]) == [0, 2]


def test_helpers():
    assert stage_of("stage26_grid_results.csv") == "stage26" and stage_of("notes.csv") is None
    assert len(sha256_file(__import__("pathlib").Path(__file__))) == 64
    assert load_registry.__name__ == "load_registry"
