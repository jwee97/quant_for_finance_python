"""``quant forecast``: the forecast and its test in the terminal, on a price file of your own."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.cli import main
from src.forecast.registry import METHODS
from tests import forecast_worlds as W


@pytest.fixture(scope="module")
def prices(tmp_path_factory) -> str:
    """A market whose volatility comes in spells, as a wide CSV of one column of prices."""
    h = W.to_history(W.garch(1900, seed=4))
    path = tmp_path_factory.mktemp("prices") / "prices.csv"
    pd.DataFrame({"date": h.index, "Zed": h.close}).to_csv(path, index=False)
    return str(path)


def test_the_list_names_every_method_and_needs_no_ticker(capsys):
    assert main(["forecast", "--list"]) == 0
    out = capsys.readouterr().out
    for name in METHODS:
        assert name in out
    assert "blend" in out


def test_a_forecast_prints_the_verdict_the_scoreboard_and_what_each_method_says_now_and_can_be_saved(prices, tmp_path, capsys):
    saved = tmp_path / "forecast.json"
    code = main(["forecast", "zed", "--prices", prices, "--horizon", "5", "--methods", "riskmetrics", "garch", "--effort", "quick", "--json", str(saved)])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("Forecast for Zed, as of ") and "5 trading days ahead" in out, "the ticker is found whatever its case"
    assert any(mark in out for mark in ("✓ ", "! ", "✗ "))
    assert "How each method scored at 5 trading days" in out and "yardstick" in out and "GJR-GARCH" in out and "RiskMetrics" in out
    assert "What each method says now: the price in 5 trading days" in out and "5% worst" in out and "chance of a rise" in out
    assert "not a strategy" in out
    data = json.loads(saved.read_text())
    assert data["ticker"] == "Zed" and data["horizon"] == 5 and data["verdict"]["level"] in ("good", "mixed", "bad")
    assert {m["id"] for m in data["evaluation"]["methods"]} == {"random_walk", "riskmetrics", "garch"} and set(data["now"]["methods"]) == {"random_walk", "riskmetrics", "garch"}
    assert f"written to {saved}" in out


@pytest.mark.parametrize("argv, message", [
    (["forecast"], "choose a ticker"),
    (["forecast", "NOPE", "--prices", "PRICES"], "'NOPE' is not in the data"),
    (["forecast", "Zed", "--prices", "PRICES", "--methods", "astrology"], "unknown forecasting method 'astrology'"),
    (["forecast", "Zed", "--prices", "PRICES", "--methods", "random_walk"], "choose at least one method besides the random walk"),
    (["forecast", "Zed", "--prices", "PRICES", "--methods", "riskmetrics", "--horizon", "500"], "the horizon must be between 1 and 63"),
])
def test_a_request_that_cannot_run_says_why_on_stderr_and_exits_with_2(prices, capsys, argv, message):
    assert main([prices if a == "PRICES" else a for a in argv]) == 2
    captured = capsys.readouterr()
    assert message in captured.err and "Forecast for" not in captured.out, "the reason goes to stderr and no report is printed (stdout may carry the log lines of a first load of the platform's data)"


def test_a_history_too_short_to_test_on_says_how_long_it_must_be(tmp_path, capsys):
    h = W.to_history(W.garch(620, seed=1))                      # 600 days to learn from and fewer than 30 to test on
    path = tmp_path / "short.csv"
    pd.DataFrame({"date": h.index, "Zed": h.close}).to_csv(path, index=False)
    assert main(["forecast", "Zed", "--prices", str(path), "--methods", "garch", "--effort", "quick"]) == 2
    assert "too short" in capsys.readouterr().err
