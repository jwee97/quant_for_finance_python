"""The real-data study script: a finished evaluation is saved and reused, and the tables it prints do not depend on whether they were computed or loaded."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from experiments import forecast_study as study
from tests import forecast_worlds as W


@pytest.fixture()
def market(monkeypatch):
    """One ticker of simulated prices in place of the platform's cached ETFs."""
    h = W.to_history(W.garch(1400, seed=5))
    prices = pd.Series(h.close, index=h.index)
    monkeypatch.setattr(study, "load_default_bundle", lambda cfg, with_macro: SimpleNamespace(prices={"SPY": prices}, high=None, low=None))
    monkeypatch.setattr(study, "load_config", lambda: {})
    return prices


def _tables(text: str) -> str:
    return text.split("\n\n", 1)[1]


def test_a_saved_evaluation_is_reused_and_prints_the_same_tables(market, tmp_path, capsys):
    args = ["--tickers", "SPY", "--methods", "garch", "--effort", "quick", "--cache", str(tmp_path)]
    assert study.main(args) == 0
    first = capsys.readouterr().out
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert "from the saved run" not in first
    assert study.main(args) == 0
    second = capsys.readouterr().out
    assert "SPY at 21 days: from the saved run" in second
    assert _tables(first) == _tables(second)
    assert "random walk" in _tables(first) and "garch" in _tables(first)


def test_methods_that_describe_only_the_spread_say_so_instead_of_failing(market, tmp_path, capsys):
    assert study.main(["--tickers", "SPY", "--methods", "garch", "--effort", "quick", "--cache", str(tmp_path)]) == 0
    assert "none of the chosen methods forecasts the average return" in capsys.readouterr().out


def test_a_different_set_of_methods_or_horizon_is_not_taken_for_a_saved_one(market, tmp_path):
    history = study.History.from_frame(market)
    study._saved(tmp_path, "SPY", history, ["garch"], 5, "quick", 0)
    study._saved(tmp_path, "SPY", history, ["garch", "har"], 5, "quick", 0)
    study._saved(tmp_path, "SPY", history, ["garch"], 21, "quick", 0)
    names = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(names) == 3 and len(set(names)) == 3
    assert all(n.startswith("SPY_") for n in names)
    assert any("_5d_" in n for n in names) and any("_21d_" in n for n in names)


def test_nothing_is_written_without_a_folder(market, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history = study.History.from_frame(market)
    result, reused = study._saved(None, "SPY", history, ["garch"], 5, "quick", 0)
    assert not reused and result["methods"]
    assert list(tmp_path.iterdir()) == []
