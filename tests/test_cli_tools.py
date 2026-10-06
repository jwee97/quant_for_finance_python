"""``quant causal`` and ``quant ask``: planted truths, refusals, no live model call."""

from __future__ import annotations

import sqlite3

import pytest

from src import cli
from src.causal.check import recovery_table


def test_causal_check_shows_the_naive_estimate_missing_and_the_right_estimator_covering():
    t = recovery_table(n=3000, seed=7).set_index(["world", "method"])
    naive = t.loc[("confounded (nonlinear controls)", "naive OLS")]
    strong = t.loc[("confounded (nonlinear controls)", "DML, larger boosted trees")]
    small = t.loc[("confounded (nonlinear controls)", "DML, small boosted trees")]
    assert not naive["covers_truth"] and abs(strong["estimate"] - 0.5) < abs(small["estimate"] - 0.5) < abs(naive["estimate"] - 0.5)
    assert t.loc[("unobserved confounder + instrument", "2SLS"), "covers_truth"]
    assert t.loc[("two groups, two periods (parallel trends)", "difference in differences"), "covers_truth"]
    assert not t.loc[("trends NOT parallel (assumption violated)", "difference in differences"), "covers_truth"]


def test_causal_check_rejects_a_tiny_sample():
    with pytest.raises(ValueError):
        recovery_table(n=50)


def test_causal_command_prints_the_table(capsys):
    assert cli.main(["causal", "--n", "1000"]) == 0
    out = capsys.readouterr().out
    assert "naive OLS" in out and "2SLS" in out and "covers_truth" in out


@pytest.fixture
def research_db(tmp_path, monkeypatch):
    path = tmp_path / "data" / "processed" / "research.db"
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE experiments (experiment_id TEXT, stage TEXT, hypothesis TEXT, decision TEXT)")
    connection.executemany("INSERT INTO experiments VALUES (?,?,?,?)", [("EXP-001", "stage01_data", "panel is fit", "retain"), ("EXP-002", "stage02_eda", "returns are fat tailed", "reject")])
    connection.commit()
    connection.close()
    monkeypatch.setattr(cli, "load_config", lambda: type("C", (), {"root": tmp_path, "path": lambda self, k: tmp_path, "get": lambda self, *a, **k: None})())
    monkeypatch.setattr(cli, "load_library", lambda: None)
    return path


def test_ask_answers_a_recognised_question_from_the_database(research_db, capsys):
    assert cli.main(["ask", "which", "hypotheses", "were", "retained"]) == 0
    out = capsys.readouterr().out
    assert "template" in out and "EXP-001" in out and "EXP-002" not in out


def test_ask_refuses_instead_of_guessing(research_db, capsys):
    assert cli.main(["ask", "drop", "the", "experiments", "table"]) == 1
    assert "I will not guess" in capsys.readouterr().out


def test_ask_says_how_to_build_a_missing_database(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: type("C", (), {"root": tmp_path})())
    monkeypatch.setattr(cli, "load_library", lambda: None)
    assert cli.main(["ask", "which", "hypotheses", "were", "retained"]) == 1
    assert "stage29_research_db" in capsys.readouterr().out


def test_ask_with_a_model_never_runs_without_a_client(research_db, monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert cli.main(["ask", "--llm", "some-model", "which", "hypotheses", "were", "retained"]) == 1
    assert "anthropic" in capsys.readouterr().out
