"""``quant lab``: the four studies in the terminal, on prices supplied by a stand-in for the data layer."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.cli import main

TICKERS = [f"T{i}" for i in range(8)]


@pytest.fixture()
def prices(monkeypatch):
    rng = np.random.default_rng(2)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (1700, 8)), axis=0), index=pd.bdate_range("2015-01-01", periods=1700), columns=TICKERS)

    class StubApp:
        def __init__(self, config):
            self.builder = SimpleNamespace(resolve=lambda tickers, *a, **k: (SimpleNamespace(prices=px), {}))

    monkeypatch.setattr("src.webapp.api.App", StubApp)
    return px


def test_alpha_prints_the_ic_the_half_life_the_calibration_and_the_breadth(prices, capsys, tmp_path):
    out_file = tmp_path / "alpha.json"
    assert main(["lab", "alpha", *TICKERS, "--signal", "trend", "--horizon", "5", "--json", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert "IC " in out and "information half-life" in out and "calibration slope" in out and "breadth" in out and "transfer coefficient" in out
    saved = json.loads(out_file.read_text())
    assert saved["signal"] == "trend" and saved["ic"]["n"] > 1000


def test_risk_prints_each_factors_contribution(prices, capsys):
    assert main(["lab", "risk", *TICKERS, "--kind", "time_series"]) == 0
    out = capsys.readouterr().out
    assert "time_series factor model" in out and "market" in out and "specific" in out and "bias statistic" in out


def test_portfolio_prints_one_line_per_method_and_says_the_differences_are_noise(prices, capsys):
    assert main(["lab", "portfolio", *TICKERS, "--methods", "equal_weight", "nco", "cdar", "--step", "63"]) == 0
    out = capsys.readouterr().out
    assert "Equal weight" in out and "NCO (minimum variance)" in out and "CDaR (95%)" in out and "inside the noise" in out and "fallbacks" in out


def test_a_request_that_cannot_run_says_why_and_exits_two(prices, capsys):
    assert main(["lab", "alpha", "T0", "T1", "T2"]) == 2
    out = capsys.readouterr().out
    assert "cannot run" in out and "at least 6" in out
    assert main(["lab", "portfolio", *TICKERS, "--methods", "oracle"]) == 2
    assert "unknown method" in capsys.readouterr().out
