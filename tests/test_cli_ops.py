"""``quant tune``, ``quant benchmark`` and ``quant registry``: the research-operations tools reachable from the command line."""

from __future__ import annotations

import pytest

from src import cli
from src.ops.hpo import Categorical, Float, Int, LogFloat
from src.ops.store import ModelRegistry


def test_search_space_syntax():
    assert (cli._param_from_text("int:21:504").kind, cli._param_from_text("float:0:1").kind, cli._param_from_text("log:1e-4:1").kind) == ("int", "float", "log")
    cat = cli._param_from_text("cat:a,3,2.5")
    assert cat.kind == "categorical" and cat.choices == ("a", 3, 2.5)
    for bad in ("int:5", "wat:1:2", "int::"):
        with pytest.raises(SystemExit):
            cli._param_from_text(bad)
    assert Int and Float and LogFloat and Categorical


def test_tune_runs_counted_trials_and_prints_the_selection_report(capsys):
    assert cli.main(["tune", "--model", "tsmom", "--space", "models.0.params.lookback=int:63:252", "--trials", "4", "--method", "random", "--blocks", "2", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "best of 4 trials" in out and "deflated_sharpe_probability" in out and "lookback" in out


def test_benchmark_reports_time_and_writes_a_csv(tmp_path, capsys):
    csv = tmp_path / "bench.csv"
    assert cli.main(["benchmark", "--models", "tsmom", "--csv", str(csv)]) == 0
    assert csv.exists() and "tsmom" in csv.read_text() and "tsmom" in capsys.readouterr().out


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda: type("C", (), {"root": tmp_path})())
    monkeypatch.setattr(cli, "load_library", lambda: None)
    return tmp_path


def test_registry_lists_promotes_and_traces_lineage(root, capsys):
    registry = ModelRegistry(root / "data" / "registry" / "models.db")
    v1 = registry.register("alpha", params={"lookback": 126}, artifact={"w": [1, 2]}, metrics={"sharpe": 0.5})
    registry.register("alpha", params={"lookback": 252}, metrics={"sharpe": 0.7}, parent_version=v1)
    assert cli.main(["registry", "list"]) == 0
    assert "alpha" in capsys.readouterr().out
    assert cli.main(["registry", "promote", "--name", "alpha", "--version", "2", "--stage", "production"]) == 0
    assert cli.main(["registry", "list", "--name", "alpha"]) == 0
    assert "production" in capsys.readouterr().out
    assert cli.main(["registry", "lineage", "--name", "alpha", "--version", "2"]) == 0
    assert "version" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["registry", "promote", "--name", "alpha"])


def test_an_empty_registry_says_so(root, capsys):
    assert cli.main(["registry", "list"]) == 0
    assert "no registered models" in capsys.readouterr().out
