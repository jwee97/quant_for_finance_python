"""Release 4 on the command line: ``quant ml-realistic`` re-runs a saved experiment through orders, fills and the ledger as a child run, and the report commands print the execution mode."""

import json

import pytest

from src.cli import main
from src.ml.registry import Registry
from tests.ml_helpers import make_bundle

GROUPS = ["price", "momentum", "volatility", "mean_reversion", "relative"]


@pytest.fixture(scope="module")
def prices_csv(tmp_path_factory):
    b = make_bundle(8, 1300, seed=33, alpha=1.0, macro=False)
    path = tmp_path_factory.mktemp("prices") / "toy.csv"
    b.prices.rename_axis("date").to_csv(path)
    return str(path)


def train(prices, reg, *extra):
    return main(["ml-train", "--prices", prices, "--groups", *GROUPS, "--model", "ridge", "--refit-every", "126", "--min-train-days", "400", "--registry", reg, "--no-explain", *extra])


def test_the_cli_runs_a_realistic_child_and_prints_what_it_assumes_and_what_it_could_not_do(prices_csv, tmp_path, capsys):
    reg = str(tmp_path / "ml")
    assert train(prices_csv, reg, "--name", "base") == 0
    out = capsys.readouterr().out
    assert "execution mode: vectorized_legacy" in out
    base = out.split("experiment ")[1].split(":")[0]
    assert main(["ml-realistic", base, "--registry", reg, "--prices", prices_csv, "--capital", "500000", "--participation", "0.05", "--commission-bps", "2", "--name", "real"]) == 0
    text = capsys.readouterr().out
    assert f"execution mode: realistic_cash_equity (a re-run of {base})" in text and "gross P&L" in text and "reconciled" in text and "the lag is applied once" in text
    assert "no volume: participation limits and impact were not applied" in text and "capacity: not established" in text, "a price file has no volume, and the report says capacity was not established instead of assuming it"
    assert "claim: blocked" in text and "no volume" in text
    child = text.split("experiment ")[1].split(":")[0]
    assert Registry(reg).get_experiment(child)["parent_id"] == base
    assert main(["ml-report", child, "--registry", reg]) == 0
    assert "execution mode: realistic_cash_equity" in capsys.readouterr().out
    assert main(["ml-report", base, "--registry", reg]) == 0
    assert "execution mode: vectorized_legacy" in capsys.readouterr().out
    assert main(["ml-realistic", base, "--registry", reg, "--prices", prices_csv, "--json", "--no-save"]) == 0
    block = json.loads(capsys.readouterr().out)
    assert block["mode"] == "realistic_cash_equity" and block["reconciliation"]["reconciled"] and block["timeline"]["delay_sessions"] == 1
    html = (Registry(reg).root / "experiments" / child / "report.html").read_text()
    assert "Execution mode:" in html and "realistic (orders, fills and the ledger)" in html


def test_the_cli_refuses_settings_that_would_lag_twice_or_cannot_mean_anything(prices_csv, tmp_path, capsys):
    reg = str(tmp_path / "ml")
    assert train(prices_csv, reg, "--name", "base") == 0
    base = capsys.readouterr().out.split("experiment ")[1].split(":")[0]
    assert main(["ml-realistic", base, "--registry", reg, "--prices", prices_csv, "--capital", "-5"]) == 2
    assert "capital must be positive" in capsys.readouterr().err
    assert main(["ml-realistic", "no-such-experiment", "--registry", reg, "--prices", prices_csv]) == 2
    capsys.readouterr()
