"""Release 3 on the command line: the same service functions as the API, reached through ``quant ml-*``. Checks that the commands register their tries, enforce the same protections, and print the same limits."""

import json

import pytest
import yaml

from src.cli import main
from src.ml.registry import Registry
from tests.ml_helpers import make_bundle

GROUPS = ["price", "momentum", "volatility", "mean_reversion", "relative"]


@pytest.fixture(scope="module")
def prices_csv(tmp_path_factory):
    b = make_bundle(8, 1300, seed=31, alpha=1.0, macro=False)
    path = tmp_path_factory.mktemp("prices") / "toy.csv"
    b.prices.rename_axis("date").to_csv(path)
    return str(path)


def train(prices, reg, *extra):
    return main(["ml-train", "--prices", prices, "--groups", *GROUPS, "--model", "ridge", "--refit-every", "126", "--min-train-days", "400", "--registry", reg, "--no-explain", *extra])


def test_the_cli_registers_its_tries_and_tells_development_from_a_full_run(prices_csv, tmp_path, capsys):
    reg = str(tmp_path / "ml")
    assert train(prices_csv, reg, "--development", "--name", "dev") == 0
    dev = capsys.readouterr().out
    assert "holdout: development" in dev and "forward" not in dev.split("holdout:")[0], "a development run reports no test or forward segment"
    assert train(prices_csv, reg, "--name", "full") == 0
    full = capsys.readouterr().out
    assert "holdout: exposed_exploratory" in full
    exp_id = full.split("experiment ")[1].split(":")[0]
    assert main(["ml-lineage", exp_id, "--registry", reg, "--json"]) == 0
    lin = json.loads(capsys.readouterr().out)
    assert lin["raw_looks"] == 2 and lin["by_kind"] == {"experiment": 2}
    assert main(["ml-lineage", exp_id, "--registry", reg]) == 0
    text = capsys.readouterr().out
    assert "2 tries recorded" in text and "experiment" in text and "counted" in text
    assert main(["ml-report", exp_id, "--registry", reg]) == 0
    report = capsys.readouterr().out
    assert "holdout: exposed_exploratory" in report and "provenance: identity" in report
    html = (Registry(reg).root / "experiments" / exp_id / "report.html").read_text()
    assert "Protected outcomes:" in html and "exposed_exploratory" in html and "Provenance identity" in html


def test_the_cli_plan_and_evaluate_follow_the_ledger(prices_csv, tmp_path, capsys):
    reg = str(tmp_path / "ml")
    cand = tmp_path / "cand.yaml"
    cand.write_text(yaml.safe_dump({"dataset": {"feature_groups": GROUPS}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge", "params": {"alpha": 10.0}}, "name": "plan-ridge"}))
    assert main(["ml-plan", "create", "--candidate", str(cand), "--study", "u-toy", "--registry", reg, "--prices", prices_csv, "--freeze", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["state"] == "frozen" and plan["reservation"]["untouched"] is True
    pid = plan["plan_id"]
    assert train(prices_csv, reg, "--name", "blocked") == 2
    assert "protects" in capsys.readouterr().err
    assert main(["ml-evaluate", pid, "--key", "cli-1", "--registry", reg, "--prices", prices_csv, "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "evaluated" and first["experiments"][0]["holdout"]["status"] == "evaluated_under_plan"
    assert main(["ml-evaluate", pid, "--key", "cli-1", "--registry", reg, "--prices", prices_csv, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "replayed"
    assert main(["ml-evaluate", pid, "--key", "cli-2", "--registry", reg, "--prices", prices_csv]) == 2
    assert "already evaluated" in capsys.readouterr().err
    assert main(["ml-holdout", "--plan", pid, "--registry", reg]) == 0
    assert "consumed" in capsys.readouterr().out
    assert main(["ml-plan", "show", pid, "--registry", reg]) == 0
    assert pid in capsys.readouterr().out
    assert main(["ml-plan", "freeze", "plan-unknown", "--registry", reg]) == 2
    capsys.readouterr()


def test_the_cli_reports_gates_and_refuses_a_promotion_the_evidence_does_not_support(prices_csv, tmp_path, capsys):
    reg = str(tmp_path / "ml")
    assert train(prices_csv, reg, "--name", "gated") == 0
    out = capsys.readouterr().out
    exp_id = out.split("experiment ")[1].split(":")[0]
    assert main(["ml-promotion", exp_id, "--registry", reg]) == 0
    text = capsys.readouterr().out
    assert "Gate A:" in text and "Gate B:" in text and "Gate C:" in text and "nothing about profit" in text
    assert main(["ml-promotion", exp_id, "--registry", reg, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["gates"]["C"]["status"] == "blocked"
    model = Registry(reg).list_models()[0]["id"]
    assert main(["ml-promote", model, "production", "--registry", reg, "--approval-actor", "me", "--approval-note", "go"]) == 1
    assert "cannot move to production" in capsys.readouterr().err
    assert main(["ml-promote", model, "archived", "--registry", reg]) == 0
    assert "is now archived" in capsys.readouterr().out
    assert main(["ml-promotion", "nope", "--registry", reg]) == 2
    capsys.readouterr()


def test_the_cli_selection_tests_and_null_controls_say_what_they_could_not_do(prices_csv, tmp_path, capsys):
    reg = str(tmp_path / "ml")
    assert main(["ml-selection-tests", "empty", "--registry", reg, "--n-boot", "50"]) == 0
    text = capsys.readouterr().out
    assert "0 of 5 procedures were applied" in text and "unsupported" in text
    assert main(["ml-null-controls", "--list", "--registry", reg]) == 0
    assert capsys.readouterr().out.strip() == ""
    cfg = tmp_path / "base.yaml"
    cfg.write_text(yaml.safe_dump({"dataset": {"feature_groups": GROUPS}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge", "params": {}}}))
    assert main(["ml-null-controls", "--registry", reg, "--prices", prices_csv, "--config", str(cfg), "--kinds", "ridge", "--n-random", "1", "--replications", "2", "--min-replications", "50"]) == 0
    out = capsys.readouterr().out
    assert "plumbing only" in out and "false promotion" in out
    assert main(["ml-null-controls", "--list", "--registry", reg]) == 0
    assert "done" in capsys.readouterr().out
    assert main(["ml-build-dataset", "--prices", prices_csv, "--groups", *GROUPS, "--registry", reg]) == 0
    assert "feature screen: not made (the warm-up training segment would have" in capsys.readouterr().out, "a timeline too short for the default split has no warm-up segment to screen on, and says so"
