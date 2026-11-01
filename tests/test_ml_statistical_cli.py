"""Release 5 on the command line: the validation policy, the diagnostics, governance and the forward-test ledger have the same rules as the API because they call the same service functions."""

import json

import pytest

from src.cli import main
from src.ml.registry import Registry
from tests.ml_helpers import make_bundle

GROUPS = ["price", "momentum", "volatility", "mean_reversion", "relative"]
POLICY = {"name": "cli-policy", "version": "v1", "criteria": {"segment": "validation", "min_segment_days": 60, "interval_method": "hac", "min_sharpe_ci_lower": -9.0}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli5")
    b = make_bundle(8, 1300, seed=33, alpha=1.0, macro=False)
    csv = tmp / "toy.csv"
    b.prices.rename_axis("date").to_csv(csv)
    reg = str(tmp / "ml")
    assert main(["ml-train", "--prices", str(csv), "--groups", *GROUPS, "--model", "ridge", "--refit-every", "126", "--min-train-days", "400", "--registry", reg, "--no-explain", "--name", "base", "--development"]) == 0
    policy = tmp / "policy.json"
    policy.write_text(json.dumps(POLICY))
    exp = Registry(reg).list_experiments(5)[0]["id"]
    return str(csv), reg, exp, str(policy)


def test_no_policy_is_in_force_until_a_named_person_adopts_one(world, capsys):
    _, reg, _, policy = world
    capsys.readouterr()
    assert main(["ml-policy", "--registry", reg]) == 0
    assert "no validation policy is adopted" in capsys.readouterr().out
    assert main(["ml-policy", "--registry", reg, "--file", policy]) == 2
    assert "actor and a note" in capsys.readouterr().err
    assert main(["ml-policy", "--registry", reg, "--file", policy, "--adopt-example"]) == 2
    assert main(["ml-policy", "--registry", reg, "--adopt-example", "--actor", "jo", "--note", "to see what a policy looks like"]) == 0
    out = capsys.readouterr().out
    assert "ILLUSTRATIVE" in out and "adopted by jo" in out
    assert main(["ml-policy", "--registry", reg, "--file", policy, "--actor", "jo", "--note", "predeclared"]) == 0
    assert "policy in force: cli-policy v1" in capsys.readouterr().out
    assert main(["ml-policy", "--registry", reg, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["adopted"]["policy"]["name"] == "cli-policy"


def test_diagnostics_print_three_sharpe_intervals_matched_benchmarks_and_the_policy_items(world, capsys):
    csv, reg, exp, _ = world
    capsys.readouterr()
    assert main(["ml-diagnostics", exp, "--registry", reg, "--n-boot", "300", "--prices", csv, "--delays", "0,1", "--neighborhood", '{"signal.n": [2, 3, 4]}']) == 0
    out = capsys.readouterr().out
    assert "naive interval" in out and "assumes independent days" in out and "| HAC" in out and "stationary bootstrap" in out
    assert "vs beta_matched" in out and "UNMATCHED (descriptive)" in out and "regimes:" in out and "neighbourhood: applied" in out and "delay: applied" in out
    assert "evidence length on the validation segment" in out
    assert main(["ml-diagnostics", exp, "--registry", reg, "--n-boot", "300", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sharpe"]["hac"]["eligibility"] == "applied" and payload["policy_hash"]
    assert main(["ml-diagnostics", "no-such-experiment", "--registry", reg]) == 2
    assert main(["ml-diagnostics", exp, "--registry", reg, "--neighborhood", "not json"]) == 2
    capsys.readouterr()


def test_governance_registers_a_version_and_every_action_needs_a_name(world, capsys):
    _, reg, exp, _ = world
    capsys.readouterr()
    assert main(["ml-governance", exp, "--registry", reg]) == 0
    assert "no strategy version is registered" in capsys.readouterr().out
    assert main(["ml-governance", exp, "--registry", reg, "--register", "--limits", '{"max_gross": 1.5}']) == 2
    assert "who registered" in capsys.readouterr().err
    assert main(["ml-governance", exp, "--registry", reg, "--register", "--actor", "jo", "--limits", '{"max_gross": 1.5, "max_drawdown": 0.2}']) == 0
    out = capsys.readouterr().out
    assert "registered" in out and "risk limits: max_drawdown=0.2, max_gross=1.5" in out and "recovery procedure: not recorded" in out
    assert main(["ml-governance", exp, "--registry", reg, "--halt", "reviewing", "--actor", "jo"]) == 0
    assert "halted since" in capsys.readouterr().out
    assert main(["ml-governance", exp, "--registry", reg, "--resume", "reviewed", "--actor", "jo"]) == 0
    assert ", active" in capsys.readouterr().out
    assert main(["ml-governance", exp, "--registry", reg, "--recovery", "short", "--actor", "jo"]) == 2
    assert "substance" in capsys.readouterr().err
    assert main(["ml-governance", exp, "--registry", reg, "--approve", "reviewed", "--actor", "jo", "--stage", "staging"]) == 0
    assert "deployment approvals recorded: 1" in capsys.readouterr().out


def test_a_forward_step_commits_to_the_ledger_and_calls_simulated_history_a_replay(world, capsys):
    csv, reg, exp, _ = world
    capsys.readouterr()
    assert main(["ml-train", "--prices", csv, "--groups", *GROUPS, "--model", "ridge", "--refit-every", "126", "--min-train-days", "400", "--registry", reg, "--no-explain", "--name", "withmodel", "--development"]) == 0
    capsys.readouterr()
    other = [r for r in Registry(reg).list_experiments(10) if r["name"] == "withmodel"][0]["id"]
    assert main(["ml-forward", other, "--registry", reg, "--prices", csv]) == 0
    out = capsys.readouterr().out
    assert ": committed as replay:" in out and "replays 1 (never counted)" in out and "prospective 0/0 settled" in out
    assert main(["ml-forward", other, "--registry", reg, "--prices", csv, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signal"]["commitment"]["status"] == "unchanged" and payload["summary"]["ledger"]["counts"]["replay"]["committed"] == 1
    assert main(["ml-forward", "no-such-experiment", "--registry", reg, "--prices", csv]) == 2
    capsys.readouterr()
