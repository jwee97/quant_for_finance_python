"""The experiment manager and the `quant` command line: recording, idempotence, trial counting, stored outputs, parallel sweeps and guarded queries."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.cli import build_parser, main as cli_main
from src.framework import PipelineSpec, bundle_from_prices
from src.framework.experiments import ExperimentManager, expand_grid, spec_hash
from src.utils.config import load_config


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(21)
    n = 2600
    idx = pd.bdate_range("2009-01-01", periods=n)
    r = pd.DataFrame(rng.normal(0.0003, 0.01, (n, 6)), index=idx, columns=list("ABCDEF"))
    return bundle_from_prices(100 * (1 + r).cumprod(), min_history=60, name="mgr")


@pytest.fixture()
def manager(tmp_path):
    return ExperimentManager(load_config(), db_path=tmp_path / "runs.db", runs_dir=tmp_path / "runs")


SPEC = {"name": "mom", "models": [{"name": "momentum", "params": {"lookback": 63}}], "evaluation": {"group": "g", "benchmarks": ["equal_weight"]}}


def test_a_run_is_recorded_with_metadata_metrics_returns_weights_forecasts_and_commit(manager, bundle):
    run_id, result = manager.run(SPEC, bundle)
    assert result is not None and np.isfinite(result.metrics["sharpe"])
    with sqlite3.connect(manager.db_path) as con:
        row = con.execute("SELECT name, group_name, git_commit, config_fingerprint, spec_json, sharpe, n_trials, causality_ok FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        n_metrics = con.execute("SELECT COUNT(*) FROM run_metrics WHERE run_id = ?", (run_id,)).fetchone()[0]
        tables = {r[0] for r in con.execute("SELECT name FROM run_tables WHERE run_id = ?", (run_id,))}
    assert row[0] == "mom" and row[1] == "g" and row[2] and row[3] == load_config().fingerprint("gen5") and json.loads(row[4])["models"][0]["params"]["lookback"] == 63
    assert row[5] == pytest.approx(result.metrics["sharpe"]) and row[6] == 1 and n_metrics > 8 and {"attribution", "benchmarks", "by_sample", "causality"} <= tables
    folder = manager.runs_dir / run_id
    assert {"returns.csv.gz", "weights.csv.gz", "forecasts.csv.gz", "meta.json"} <= {p.name for p in folder.iterdir()}
    forecasts = pd.read_csv(folder / "forecasts.csv.gz")
    assert set(forecasts["model"]) >= {"combined", "momentum"} and forecasts["confidence"].between(0, 1).all() and forecasts["p_up"].between(0, 1).all()
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["git_commit"] == row[2] and meta["config_fingerprints"]["gen5"] == row[3] and meta["spec"]["name"] == "mom"
    assert pd.read_csv(folder / "returns.csv.gz", index_col="date").shape[0] == len(result.net_returns)


def test_rerunning_the_same_spec_is_idempotent_unless_forced(manager, bundle):
    first, result = manager.run(SPEC, bundle)
    again, skipped = manager.run(SPEC, bundle)
    assert again == first and skipped is None
    forced, fresh = manager.run(SPEC, bundle, force=True)
    assert forced != first or fresh is not None
    with sqlite3.connect(manager.db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2


def test_trial_counting_deflates_every_variant_in_a_group(manager, bundle):
    ids = [manager.run({**SPEC, "name": f"m{k}", "models": [{"name": "momentum", "params": {"lookback": k}}]}, bundle)[0] for k in (21, 63, 126)]
    board = manager.leaderboard("g", 10)
    assert len(board) == 3 and set(board["run_id"]) == set(ids) and board["n_trials"].max() == 3
    assert manager.n_trials("g") == 3 and manager.n_trials("other") == 1
    assert manager.n_trials("g", "newhash") == 4
    compare = manager.compare(ids)
    assert list(compare.index) == ids and {"sharpe", "deflated_sharpe_probability"} <= set(compare.columns)
    with pytest.raises(ValueError, match="sort by one of"):
        manager.leaderboard("g", 5, "nonsense")


def test_stored_returns_spec_and_tables_can_be_read_back(manager, bundle):
    run_id, result = manager.run(SPEC, bundle)
    returns = manager.returns_of(run_id)
    assert np.allclose(returns["net"].to_numpy(), result.net_returns.to_numpy(), equal_nan=True)
    assert manager.spec_of(run_id).models[0]["name"] == "momentum"
    assert len(manager.table_of(run_id, "attribution")) > 2 and manager.table_of(run_id, "no_such_table").empty
    with pytest.raises(KeyError):
        manager.spec_of("missing")


def test_spec_hash_ignores_notes_and_trial_count_but_not_parameters(bundle):
    a = PipelineSpec.from_dict(SPEC)
    b = PipelineSpec.from_dict({**SPEC, "notes": "different", "evaluation": {**SPEC["evaluation"], "n_trials": 9}})
    c = PipelineSpec.from_dict({**SPEC, "models": [{"name": "momentum", "params": {"lookback": 64}}]})
    assert spec_hash(a, "d") == spec_hash(b, "d") and spec_hash(a, "d") != spec_hash(c, "d") and spec_hash(a, "d") != spec_hash(a, "other data")


def test_expand_grid_builds_named_variants_and_rejects_nothing_silently():
    base = {"name": "s", "models": [{"name": "momentum", "params": {"lookback": 21}}], "combination": {"rule": "equal"}}
    out = expand_grid(base, {"models.0.params.lookback": [63, 126], "combination.rule": ["equal", "confidence"]})
    assert len(out) == 4 and out[0]["name"] == "s[lookback=63,rule=equal]" and out[3]["models"][0]["params"]["lookback"] == 126 and out[3]["combination"]["rule"] == "confidence"
    assert base["models"][0]["params"]["lookback"] == 21                                          # the base is not mutated
    with pytest.raises((IndexError, KeyError, ValueError)):
        expand_grid(base, {"models.5.params.x": [1]})


@pytest.mark.parametrize("backend", ["serial", "joblib"])
def test_a_parallel_sweep_equals_a_serial_run_and_counts_every_variant_as_a_trial(manager, bundle, backend):
    variants = expand_grid({"name": "sw", "models": [{"name": "momentum"}], "evaluation": {"group": "sweep_g", "causality": False}}, {"models.0.params.lookback": [42, 84, 168]})
    ids = manager.sweep(variants, bundle, validate=False, backend=backend, n_jobs=2)
    assert len(ids) == 3 and len(set(ids)) == 3
    board = manager.compare(ids)
    assert (board["n_trials"] == 3).all()
    solo_id, solo = ExperimentManager(load_config(), db_path=manager.db_path.with_name("solo.db"), runs_dir=manager.runs_dir / "solo").run(variants[1], bundle, validate=False)
    assert board.loc[ids[1], "sharpe"] == pytest.approx(solo.metrics["sharpe"])                    # the sweep gave the number a plain run gives
    assert manager.sweep(variants, bundle, validate=False, backend="serial") == ids                 # a second sweep records nothing new


def test_cli_parser_and_commands_reach_the_manager(bundle, tmp_path, monkeypatch, capsys):
    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "momentum", "--allocator", "static", "--alloc-param", "book=hrp", "--aum", "1e8"])
    from src.cli import _spec_from_args
    spec = _spec_from_args(args)
    assert spec["allocation"] == {"allocator": "static", "params": {"book": "hrp"}} and spec["execution"] == {"aum": 1e8}
    with pytest.raises(SystemExit):
        _spec_from_args(parser.parse_args(["backtest", "--model", "momentum", "--alloc-param", "book=hrp"]))
    with pytest.raises(SystemExit):
        parser.parse_args(["sql"])                                                                   # a query is required
    assert cli_main(["list", "allocators"]) == 0 and "regime_switch" in capsys.readouterr().out


def test_sql_over_runs_is_read_only_and_whitelisted(manager, bundle, monkeypatch, tmp_path):
    from src.assistant.sqlguard import UnsafeSQL, check_select

    manager.run(SPEC, bundle)
    runs_only = ("runs", "run_metrics", "run_tables")
    with pytest.raises(UnsafeSQL):
        check_select("DROP TABLE runs", runs_only)
    with pytest.raises(UnsafeSQL):
        check_select("SELECT * FROM experiments", runs_only)                                         # a table of the other database
    from src.assistant.sqlguard import open_readonly
    con = open_readonly(manager.db_path)
    rows = pd.read_sql_query(check_select("SELECT name, round(sharpe, 3) AS sharpe FROM runs", runs_only), con)
    assert rows["name"].tolist() == ["mom"]
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO runs (run_id) VALUES ('x')")                                       # the connection itself cannot write
