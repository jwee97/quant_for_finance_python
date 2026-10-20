"""``quant fundamentals``: the Fundamentals tab in the terminal. The demo and the real database are separate files and the real one is never touched by a simulated world, a key is read from the environment and
never printed or accepted as an argument, statements are fetched only within a budget that must be stated, a study is printed in words with the same numbers as the page, a search counts every formula, a status
is never changed without a reason, and each failure of input is a sentence with exit code 2 and not a traceback."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import signal

import pandas as pd
import pytest

from src import cli
from src.fundamentals import cli as fcli
from src.fundamentals import expr as X
from src.fundamentals.keys import KeyPool
from src.fundamentals.providers.openfilings import OpenFilingsProvider
from src.fundamentals.service import FundamentalsService
from src.fundamentals.usage import Allowance
from src.fundamentals.workbench import KEY_NAMES, Workbench
from src.utils.config import load_config
from tests.test_fund_data import KEY_1, FakeOpenFilings

SECRETS = ("of_zzzzSECRETzzzz_one_0123456789", "of_zzzzSECRETzzzz_two_9876543210")
WORLD = ["fundamentals", "simulate", "--demo", "--companies", "40", "--from-year", "2010", "--to-year", "2018", "--seed", "5", "--strong"]


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    """A folder the command treats as the project root, with one demo world made through the command itself and one finished search in it."""
    root = tmp_path_factory.mktemp("fund_cli_home")
    patch = pytest.MonkeyPatch()
    for name in KEY_NAMES:
        patch.delenv(name, raising=False)                                                   # a key in the developer's own environment would change what an empty setup looks like
    config = dataclasses.replace(load_config(), root=root)
    patch.setattr(cli, "load_config", lambda *a, **k: config)
    assert cli.main(WORLD) == 0
    assert cli.main(["fundamentals", "discover", "--demo", "--n", "30", "--seed", "4", "--finalists", "3", "--formulas", "div(fcf,market_cap)", "delta(roic,12)"]) == 0
    yield root
    patch.undo()


def run(capsys, *argv):
    code = cli.main(["fundamentals", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def alpha_ids(capsys) -> list[str]:
    _, out, _ = run(capsys, "library", "--demo")
    return re.findall(r"^(A[0-9a-f]+-[0-9a-f]+)\s", out, re.M)


def add_weak_alpha(home) -> str:
    """A candidate whose evidence is thin (a locked-dates t-statistic of 0.2): the library must not call it validated."""
    wb = Workbench(service=FundamentalsService(home, db_path=home / "data" / "user" / "fundamentals_demo.db"))
    settings = {"universe": {}, "freq": "M", "horizon": "1m", "entry_lag": 1, "min_assets": 20, "direction": 1.0, "steps": [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}]}
    return wb.library.add_candidate(X.parse("div(fcf,enterprise_value)"), settings, {"ic": 0.08, "direction": 1.0, "n_tested": 40, "oos_t": 0.2})


# ------------------------------------------------------------------------------------------------------------ the two databases
def test_the_simulated_world_is_made_only_in_the_demo_file_and_the_real_database_is_left_alone(home, capsys):
    code, out, err = run(capsys, "simulate")
    assert code == 2 and "made in the demo database: add --demo" in err and out == ""
    assert (home / "data" / "user" / "fundamentals_demo.db").exists() and not (home / "data" / "user" / "fundamentals.db").exists()
    code, out, _ = run(capsys, "status", "--demo")
    assert code == 0 and "demo: a simulated world" in out and "40 companies" in out and "ready to study: yes" in out and "no provider is called here" in out
    code, out, _ = run(capsys, "status")
    assert code == 0 and "real data" in out and "0 companies" in out and "ready to study: no" in out and "openfilings: not set up" in out
    for action in ("add", "plan", "sync", "prices"):
        code, out, err = run(capsys, action, "--demo")
        assert code == 2 and "is for real data" in err and "never calls a provider" in err, action


def test_a_demo_that_has_not_been_made_says_how_to_make_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: dataclasses.replace(load_config(), root=tmp_path))
    code, _, err = run(capsys, "status", "--demo")
    assert code == 2 and "run `quant fundamentals simulate`" in err and not (tmp_path / "data" / "user" / "fundamentals_demo.db").exists()


# ------------------------------------------------------------------------------------------------------------ keys
def test_the_status_shows_the_slots_and_the_allowance_and_never_a_key(home, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("OPENFILINGS_API_KEY_1", SECRETS[0])
    monkeypatch.setenv("OPENFILINGS_API_KEY_2", SECRETS[1])
    saved = tmp_path / "status.json"
    code, out, err = run(capsys, "status", "--json", str(saved))
    assert code == 0 and "OPENFILINGS_API_KEY_1" in out and "OPENFILINGS_API_KEY_2" in out and "active" in out and "standby" in out
    assert "today 0 of 100" in out and "this month 0 of 1,000" in out and "does not allow rotating accounts or API keys" in out and "typed into the app or this command" in out
    blob = out + err + saved.read_text()
    assert not any(secret in blob or secret[:12] in blob or secret[-10:] in blob for secret in SECRETS)
    parser = cli.build_parser()
    option_strings = {o for action in parser._subparsers._group_actions[0].choices["fundamentals"]._actions for o in action.option_strings}
    assert not [o for o in option_strings if re.search(r"key|secret|token|password", o, re.I)]            # nothing on the command line takes a key


# ------------------------------------------------------------------------------------------------------------ statements and prices
@pytest.fixture()
def fake_provider(monkeypatch):
    """The command with OpenFilings replaced by a stand-in that behaves as its documentation says (a daily allowance of 100 credits, ten for statements and one for a lookup)."""
    fake = FakeOpenFilings({KEY_1: "a"})
    real = fcli._bench

    def bench(config, demo, action):
        wb = real(config, demo, action)
        if not demo:
            pool = KeyPool("openfilings", [("OPENFILINGS_API_KEY_1", KEY_1)], wb.service.usage, wb.store.salt(), Allowance("credits", 100, 1000, "test"))
            wb.service.register("openfilings", OpenFilingsProvider(pool, transport=fake, sleep=lambda s: None))
        return wb
    monkeypatch.setattr(fcli, "_bench", bench)
    return fake


def test_companies_are_registered_for_free_planned_and_fetched_only_within_a_stated_budget(home, fake_provider, capsys):
    code, out, _ = run(capsys, "add", "AAPL", "jp:7203", "us:MSFT")
    assert code == 0 and "3 companies registered" in out and fake_provider.requests == []
    code, out, _ = run(capsys, "plan", "--start-year", "2015")
    assert code == 0 and "costing 33 credits" in out and "Left today: 100 credits" in out and "AAPL" in out and fake_provider.requests == []
    code, out, err = run(capsys, "sync", "--start-year", "2015")
    assert code == 2 and "--budget" in err and "how many credits" in err and fake_provider.requests == []      # no budget, no spending
    code, out, _ = run(capsys, "sync", "--start-year", "2015", "--budget", "22")
    assert code == 0 and "Fetched 2 companies" in out and "spent 22 credits" in out and "budget of 22" in out
    code, out, _ = run(capsys, "status")
    assert "today 22 of 100" in out and "key set" in out and re.search(r"OPENFILINGS_API_KEY_1\s+active", out)
    code, out, _ = run(capsys, "sync", "--start-year", "2015", "--budget", "50")
    assert code == 0 and "Fetched 1 companies" in out and "spent 11 credits" in out


def test_prices_are_loaded_from_the_market_data_source(home, monkeypatch, capsys):
    asked = []

    def fetch(symbols, start):
        asked.append(list(symbols))
        idx = pd.bdate_range("2020-01-01", periods=10)
        return {s: pd.DataFrame({"close": 10.0, "adj_close": 9.5, "volume": 100.0}, index=idx) for s in symbols}

    monkeypatch.setattr("src.fundamentals.prices.yahoo_download", fetch)
    run(capsys, "add", "us:AAA", "jp:7203")
    code, out, _ = run(capsys, "prices", "--price-start", "2020-01-01")
    assert code == 0 and re.search(r"Prices for \d+ of \d+ companies", out) and asked and "7203.T" in asked[0]


def test_a_company_is_looked_up_by_name_and_the_command_says_what_it_cost_and_what_to_do_next(home, fake_provider, capsys, tmp_path):
    saved = tmp_path / "found.json"
    fake_provider.skipped = ["jp"]
    code, out, _ = run(capsys, "search", "Nvidia", "--json", str(saved))
    assert "did not search these markets this time: JP" in out
    assert code == 0 and re.search(r"read as a company name: 1 match \(1 credit spent; \d+ left today\)", out)             # what is left depends on what the earlier tests of this module spent
    assert re.search(r"NVIDIA Corp\s+us:NVDA\s+US\s+US67066G1040\s+not added", out) and "us:NVDA is also listed as mx:NVDA" in out
    assert "quant fundamentals add us:NVDA" in out and "--budget 11" in out and json.loads(saved.read_text())["results"][0]["symbol"] == "us:NVDA"
    code, out, _ = run(capsys, "search", "nvidia")                                              # the same words again: the saved answer, for nothing
    assert code == 0 and "saved answer" in out and "for nothing" in out and len(fake_provider.requests) == 1
    assert run(capsys, "add", "us:NVDA")[0] == 0
    code, out, _ = run(capsys, "search", "Nvidia")
    assert "in the database, no statements yet" in out and "quant fundamentals add" not in out
    code, out, _ = run(capsys, "search", "Zzyzx")
    assert code == 0 and "0 matches" in out and "no issuer" in out and re.search(r"could also be a ticker: quant fundamentals search --kind symbol \"Zzyzx\"", out)
    code, out, _ = run(capsys, "search", "--kind", "symbol", "nvda", "--market", "us", "--refresh")
    assert code == 0 and "read as a ticker on the us market" in out and "us:NVDA" in out
    code, out, _ = run(capsys, "search", "Deutsche", "Bank")
    assert code == 0 and "no statements in this market" in out and "quant fundamentals add" not in out
    for argv, message in ((["search"], "say what to look for"), (["search", "x"], "say what to look for"), (["search", "Nvidia", "--demo"], "is for real data")):
        code, out, err = run(capsys, *argv)
        assert code == 2 and message in err and out == "", argv
    assert all(KEY_1 not in r["url"] for r in fake_provider.requests)


# ------------------------------------------------------------------------------------------------------------ the research
def test_the_factor_library_is_listed_and_a_scan_sets_factors_side_by_side(home, capsys):
    code, out, _ = run(capsys, "factors", "--demo")
    assert code == 0 and "fcf_yield" in out and "d_roic" in out and re.search(r"\b\d\d factors\b", out)
    code, out, _ = run(capsys, "factors", "--scan", "fcf_yield", "roic", "--demo")
    assert code == 0 and "2 factors under one set of conditions" in out and "map of where to look" in out and out.index("fcf_yield") < out.index("roic")


def test_a_study_is_printed_in_words_with_the_audit_and_saved(home, capsys, tmp_path):
    saved = tmp_path / "study.json"
    code, out, _ = run(capsys, "evaluate", "fcf_yield", "--demo", "--lookahead", "--json", str(saved))
    assert code == 0 and re.search(r"fcf_yield: (Strong in this sample|Some evidence)", out) and "Interesting is not validated" in out
    assert "Mean rank IC +0." in out and "How fast the signal fades" in out and "IC by year (positive in" in out and "By country:" in out and "Look-ahead audit:" in out
    data = json.loads(saved.read_text())
    assert data["study"]["ic"]["mean_ic"] > 0.03 and {"honest", "naive", "flattered"} <= set(data["lookahead"]) and f"written to {saved}" in out
    code, out, _ = run(capsys, "evaluate", "--formula", "div( fcf , market_cap )", "--demo", "--horizon", "3m")
    assert code == 0 and "div(fcf,market_cap)" in out


@pytest.mark.parametrize("argv, message", [
    (["evaluate", "--demo"], "name one factor"),
    (["evaluate", "fcf_yield", "--formula", "roic", "--demo"], "not both"),
    (["evaluate", "fcf_yield", "--direction", "up", "--demo"], "direction is theory"),
    (["evaluate", "not_a_factor", "--demo"], "not_a_factor"),
    (["evaluate", "--formula", "div(fcf,", "--demo"], "formula"),
    (["evaluate", "fcf_yield", "--horizon", "9y", "--demo"], "horizon must be"),
    (["evaluate", "fcf_yield", "--quantiles", "1", "--demo"], "quantiles"),
    (["discover", "--demo", "--n", "0"], "between"),
    (["library", "--demo", "--set-status", "Anope", "--to", "Rejected"], "needs a reason"),
    (["library", "--demo", "--set-status", "Anope"], "say the status"),
    (["backtest", "--demo"], "one of --alpha"),
    (["backtest", "--demo", "--factor", "fcf_yield", "--alpha", "x"], "one of --alpha"),
])
def test_bad_input_is_a_sentence_and_exit_code_two_never_a_traceback(home, capsys, argv, message):
    code, out, err = run(capsys, *argv)
    assert code == 2 and message in err and "Traceback" not in err and out == ""


def test_a_search_counts_every_formula_keeps_the_survivors_as_candidates_and_does_not_reopen_a_finished_run(home, capsys, tmp_path):
    saved = tmp_path / "run.json"
    code, out, err = run(capsys, "discover", "--demo", "--n", "20", "--seed", "8", "--finalists", "3", "--formulas", "div(net_income,market_cap)", "delta(roic,12)", "--json", str(saved))
    run_id = re.search(r"run (R[0-9a-f]+)", err).group(1)
    assert code == 0 and f"Discovery run {run_id}: finished, 20 of 20 formulas" in out and "Where the formulas fell" in out and "sanity" in out and "locked_test" in out
    assert "Validated" not in out and "Nothing here is validated" in out
    summary = json.loads(saved.read_text())
    assert summary["counts"]["tested"] >= 5 and all(r["status"] in ("candidate", "survivor") for r in summary["leaderboard"])
    code, _, err = run(capsys, "discover", "--demo", "--resume", run_id)
    assert code == 2 and "that run has finished" in err and "looking again would spend it" in err
    code, _, err = run(capsys, "discover", "--demo", "--resume", "Rnothere")
    assert code == 2 and "no discovery run called" in err


def test_the_library_keeps_the_failures_and_a_status_is_never_changed_without_a_reason(home, capsys):
    code, out, _ = run(capsys, "library", "--demo")
    ids = alpha_ids(capsys)
    assert code == 0 and ids and "Candidate" in out and "Validated" not in out and re.search(r"\d+ failed ideas in the graveyard", out) and "higher" in out
    tried = int(re.search(r"(\d+) ideas tried in all", out).group(1))
    code, grave, _ = run(capsys, "library", "--demo", "--graveyard", "--top", "500")
    failed = int(re.search(r"^(\d+) failed ideas", grave).group(1))
    assert code == 0 and "failed at" in grave and "too weak" in grave and tried == failed + len(ids)
    weak = add_weak_alpha(home)
    code, _, err = run(capsys, "library", "--demo", "--validate", weak)
    assert code == 2 and "not validated" in err and "out-of-sample t-statistic" in err            # thin evidence is not validated
    code, _, err = run(capsys, "library", "--demo", "--set-status", weak, "--to", "Rejected")
    assert code == 2 and "needs a reason" in err
    code, out, _ = run(capsys, "library", "--demo", "--set-status", weak, "--to", "Rejected", "--reason", "does not hold up in the second half")
    assert code == 0 and f"{weak} is now Rejected" in out
    code, out, _ = run(capsys, "library", "--demo", "--status", "Rejected")
    assert code == 0 and weak in out and "Rejected" in out                                         # a rejected alpha stays in the library
    code, _, err = run(capsys, "library", "--demo", "--set-status", weak, "--to", "Production", "--reason", "why not")
    assert code == 2 and "cannot go straight to Production" in err


def test_the_monitor_judges_the_alphas_in_use_and_a_backtest_reads_in_words(home, capsys, tmp_path):
    ids = alpha_ids(capsys)
    code, out, _ = run(capsys, "monitor", "--demo")
    assert code == 0 and "alpha(s) checked" in out and re.search(r"\b(Healthy|Weakening|Decaying|Broken)\b", out)
    saved = tmp_path / "bt.json"
    code, out, _ = run(capsys, "backtest", "--demo", "--alpha", ids[0], "-n", "8", "--max-position", "0.2", "--json", str(saved))
    assert code == 0 and "Backtest:" in out and "Deflated Sharpe" in out and "not that it will keep doing so" in out and re.search(r"Recorded as experiment bt-[0-9a-f]+", out)
    assert "its status is Candidate" in out and "Latest book" in out and "One-way cost against net return" in out
    assert json.loads(saved.read_text())["experiment_id"].startswith("bt-")
    code, out, _ = run(capsys, "backtest", "--demo", "--formula", "div(fcf,market_cap)", "--side", "long_short", "--percentile", "20", "--cost-bps", "25")
    assert code == 0 and "long and short" in out and "top 20.0%" in out
    code, out, _ = run(capsys, "backtest", "--demo", "--factor", "fcf_yield")
    assert code == 0 and "long only, top 30, equal weights" in out                                 # the default size of a leg is its own, not the number of formulas a search tries
    code, out, _ = run(capsys, "discover", "--demo", "--n", "6", "--seed", "2", "--finalists", "1")
    assert code == 0 and "of 6 formulas" in out


def test_an_alpha_is_validated_only_when_the_evidence_holds_after_the_number_of_ideas_tried(home, capsys):
    """Last of the tests that share one library: it validates the best alpha."""
    best = alpha_ids(capsys)[0]
    code, out, _ = run(capsys, "library", "--demo", "--validate", best)
    assert code == 0 and f"{best} is now Validated" in out and "evidence holds after the correction for the number of ideas tried" in out
    code, out, _ = run(capsys, "library", "--demo", "--status", "Validated")
    assert code == 0 and best in out


# ------------------------------------------------------------------------------------------------------------ Ctrl-C
def test_the_first_ctrl_c_asks_a_job_to_stop_and_the_second_ends_the_command():
    before = signal.getsignal(signal.SIGINT)
    with fcli._Stop() as stop:
        assert stop() is False
        os.kill(os.getpid(), signal.SIGINT)
        assert stop() is True
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) is before                                                  # the handler is put back
