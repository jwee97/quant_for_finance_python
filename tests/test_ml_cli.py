"""The ML command line: build a dataset, train, re-backtest and report, on a small price file."""

import json

import pytest

from src.cli import main
from tests.ml_helpers import make_bundle

GROUPS = ["price", "momentum", "volatility", "mean_reversion", "relative"]


@pytest.fixture(scope="module")
def prices_csv(tmp_path_factory):
    b = make_bundle(10, 1600, seed=31, alpha=1.0, macro=False)
    path = tmp_path_factory.mktemp("prices") / "toy.csv"
    b.prices.rename_axis("date").to_csv(path)
    return str(path)


def test_the_four_commands_work_end_to_end(prices_csv, tmp_path, capsys):
    reg = str(tmp_path / "ml")
    assert main(["ml-build-dataset", "--prices", prices_csv, "--groups", *GROUPS, "--registry", reg, "--extra-targets", "fwd_ret_1"]) == 0
    out = capsys.readouterr().out
    assert "saved as feature set" in out and "features" in out and "starts 1 day(s) after the decision" in out
    assert main(["ml-train", "--prices", prices_csv, "--groups", *GROUPS, "--model", "ridge", "--param", "alpha=20", "--refit-every", "63", "--min-train-days", "500", "--registry", reg,
                 "--name", "cli_ridge", "--n", "2", "--no-explain"]) == 0
    out = capsys.readouterr().out
    assert "cli_ridge" in out and "validation" in out and "generalisation score" in out and "saved to" in out
    exp_id = out.split("experiment ")[1].split(":")[0]
    assert main(["ml-report", "--registry", reg]) == 0
    board = capsys.readouterr().out
    assert exp_id in board and "cli_ridge" in board
    assert main(["ml-report", exp_id, "--registry", reg]) == 0
    text = capsys.readouterr().out
    assert "rank IC" in text and "top features" in text
    assert main(["ml-report", exp_id, "--registry", reg, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == exp_id
    assert main(["ml-report", exp_id, "--registry", reg, "--html"]) == 0
    assert capsys.readouterr().out.strip().endswith("report.html")
    assert main(["ml-backtest", exp_id, "--registry", reg, "--cost-bps", "30", "--holding-period", "20"]) == 0
    out = capsys.readouterr().out
    assert "against the original run" in out and "original sharpe" in out
    assert main(["ml-report", "--registry", reg, "--limit", "5"]) == 0
    assert capsys.readouterr().out.count("cli_ridge") >= 2                                          # the original and the re-backtest


def test_errors_are_messages_and_exit_codes(prices_csv, tmp_path, capsys):
    reg = str(tmp_path / "ml")
    assert main(["ml-train", "--prices", prices_csv, "--groups", *GROUPS, "--min-train-days", "5000", "--registry", reg, "--no-save"]) == 2
    assert "warm-up" in capsys.readouterr().err
    assert main(["ml-report", "nonexistent", "--registry", reg]) == 2
    assert "unknown experiment" in capsys.readouterr().err
    assert main(["ml-backtest", "nonexistent", "--registry", reg]) == 2
    assert main(["ml-report", "--registry", reg]) == 0
    assert "no experiments recorded" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["ml-train", "--groups", "astrology"])
