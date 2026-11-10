"""Scenario generators side by side: the report (what runs, what is scored, what a failing generator does to the others, the refusals), the Scenarios tab's endpoints (plan, a run as a job on the app's prices,
every refusal as a message) and ``quant scenarios`` in the terminal."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.cli import main
from src.scenarios import GENERATORS, report
from src.utils.config import load_config
from src.webapp import api
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch

ROOT = Path(__file__).resolve().parents[1]
FAST = ["historical", "bootstrap", "copula", "risk_factor"]


def world(n: int = 900, assets: int = 3, seed: int = 4) -> pd.DataFrame:
    """Daily returns with fat tails and one common factor."""
    rng = np.random.default_rng(seed)
    common = rng.standard_t(5, (n, 1)) * 0.008
    R = 0.6 * common + rng.standard_t(5, (n, assets)) * 0.007
    return pd.DataFrame(R, columns=[f"X{i}" for i in range(assets)], index=pd.bdate_range("2019-01-01", periods=n))


@pytest.fixture(scope="module")
def fast():
    return report.run(world(), FAST, 200, 21, 3)


def plain(value):
    json.dumps(value, allow_nan=False)                                                          # the server refuses to send anything else
    return value


# ------------------------------------------------------------------------------------------------------------------------------ the report
def test_every_generator_has_a_description_and_the_catalogue_matches_the_library():
    assert [g["id"] for g in report.catalog()] == list(GENERATORS)
    assert all(g["label"] and g["text"] and g["assumes"] and g["fails"] and g["family"] in report.FAMILIES for g in report.catalog())
    assert set(report.DEFAULT) <= set(GENERATORS) and set(report.PRESETS) <= set(GENERATORS) and all(m["target"] in (0.0, 1.0) for m in report.METRICS.values())


def test_a_run_scores_each_generator_against_the_history_and_summarises_the_portfolio(fast):
    d = plain(fast)
    assert [g["id"] for g in d["generators"]] == FAST and d["assets"] == ["X0", "X1", "X2"] and d["days"] == 900 and d["horizon"] == 21 and d["n"] == 200 and d["levels"] == list(report.LEVELS)
    for g in d["generators"]:
        assert g["error"] is None and set(g["flags"].values()) <= {"close", "far", "n/a"} and set(g["flags"]) == set(g["quality"])
        q = g["terminal"]["quantiles"]
        assert q == sorted(q) and g["terminal"]["es5"] >= g["terminal"]["var5"] > 0 and g["terminal"]["std"] > 0
    assert d["history"]["quantiles"] == sorted(d["history"]["quantiles"]) and d["history"]["count"] == 900 - 21 + 1
    assert {"marginal_ks", "vol_error", "corr_error", "kurtosis_ratio", "es_ratio", "horizon_vol_ratio"} <= set(d["generators"][0]["quality"])


def test_resampling_the_history_is_close_to_it_and_the_flags_say_so(fast):
    boot = {g["id"]: g for g in fast["generators"]}["bootstrap"]
    assert boot["quality"]["marginal_ks"] < 0.05 and boot["quality"]["vol_error"] < 0.15 and boot["quality"]["corr_error"] < 0.1
    assert boot["flags"]["marginal_ks"] == "close" and boot["flags"]["corr_error"] == "close"
    assert report._flag("vol_error", 0.9) == "far" and report._flag("kurtosis_ratio", 0.5) == "far" and report._flag("kurtosis_ratio", 1.1) == "close" and report._flag("es_ratio", float("nan")) == "n/a"


def test_the_same_seed_gives_the_same_scenarios_and_another_seed_gives_others():
    r = world(700)
    a, b, c = (report.run(r, ["bootstrap"], 100, 10, s)["generators"][0]["terminal"] for s in (1, 1, 2))
    assert a == b and a != c


def test_a_generator_that_fails_is_reported_and_the_others_still_run(monkeypatch):
    real = report.generate

    def broken(name, *args, **kw):
        if name == "copula":
            raise RuntimeError("the copula did not converge")
        return real(name, *args, **kw)

    monkeypatch.setattr(report, "generate", broken)
    d = report.run(world(700), ["bootstrap", "copula"], 100, 5, 0)
    by = {g["id"]: g for g in d["generators"]}
    assert by["copula"]["error"] == "the copula did not converge" and by["copula"]["quality"] is None and by["bootstrap"]["error"] is None
    assert any("did not converge" in n for n in d["notes"])
    monkeypatch.setattr(report, "generate", lambda *a, **k: (_ for _ in ()).throw(ValueError("no luck")))
    with pytest.raises(report.ScenarioError, match="no generator could make scenarios: no luck"):
        report.run(world(700), ["bootstrap"], 100, 5, 0)


def test_a_generator_whose_scenarios_are_not_finite_is_refused(monkeypatch):
    monkeypatch.setattr(report, "generate", lambda name, R, n, horizon, seed, **kw: np.full((n, horizon, R.shape[1]), np.nan))
    with pytest.raises(report.ScenarioError, match="not finite"):
        report.run(world(700), ["bootstrap"], 100, 5, 0)


def test_a_stop_keeps_the_generators_already_made():
    r = world(700)
    calls = []
    d = report.run(r, FAST, 100, 5, 0, stop=lambda: len(calls) >= 1 or calls.append(1) is not None)
    assert [g["id"] for g in d["generators"]] == [FAST[0]] and any("Stopped before" in n for n in d["notes"])
    with pytest.raises(report.ScenarioError, match="stopped before the first generator"):
        report.run(r, FAST, 100, 5, 0, stop=lambda: True)


def test_every_unusable_request_is_a_message():
    r = world(700)
    for kw, why in [({"names": []}, "at least one generator"), ({"names": ["nope"]}, "unknown generator 'nope'"), ({"n": 5}, "between 100 and 2000"), ({"n": 99999}, "between 100 and 2000"), ({"horizon": 0}, "between 1 and 126"),
                    ({"horizon": 500}, "between 1 and 126")]:
        with pytest.raises(report.ScenarioError, match=why):
            report.run(r, **{"names": ["bootstrap"], "n": 100, "horizon": 5, **kw})
    with pytest.raises(report.ScenarioError, match="at least 500 days"):
        report.run(r.iloc[:300], ["bootstrap"], 100, 5)
    with pytest.raises(report.ScenarioError, match="between 1 and 12 assets"):
        report.run(world(700, assets=13), ["bootstrap"], 100, 5)


def test_one_asset_has_no_correlation_column_and_a_day_horizon_has_no_clustering_column():
    d = report.run(world(700, assets=1), ["bootstrap"], 100, 1, 0)
    q = d["generators"][0]["quality"]
    assert "corr_error" not in q and "clustering_error" not in q and "tail_dep_error" not in q and "es_ratio" in q


def test_the_learned_generators_make_finite_scenarios_and_are_scored_like_the_others(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(report, "PRESETS", {"wgan_gp": {"steps": 40, "hidden": 16, "layers": 2, "latent": 4}, "factor_vae": {"epochs": 10, "hidden": 16}, "diffusion": {"epochs": 8, "hidden": 16, "layers": 2, "steps": 20}})
    d = plain(report.run(world(600, assets=2), ["wgan_gp", "factor_vae", "diffusion"], 100, 5, 0))
    assert [g["error"] for g in d["generators"]] == [None, None, None]
    assert all(np.isfinite(list(g["quality"].values())).all() and g["terminal"]["std"] > 0 for g in d["generators"])
    assert {g["family"] for g in d["generators"]} == {"Learned"}


# ------------------------------------------------------------------------------------------------------------------------------ the endpoints
def make_app(tmp_path, n=1000, assets=4):
    config = load_config()
    bundle = make_bundle(n_assets=assets, n=n, seed=3, alpha=0.0)
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=lambda c: bundle), root=tmp_path)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return make_app(tmp_path_factory.mktemp("scenarios_app"))


def _wait(app, job, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        view = app.job(job)
        if view["status"] in ("done", "error", "interrupted"):
            return view
        time.sleep(0.05)
    raise AssertionError("the job did not end")


def _run(app, **body):
    return app.extra_post["/api/scenarios/run"](body)


def test_the_routes_and_the_version_say_the_tab_is_there(app):
    assert "/api/scenarios/plan" in app.extra_get and "/api/scenarios/run" in app.extra_post and "/api/scenarios/stop" in app.extra_post
    assert api.API_VERSION >= 16 and app.catalog()["api_version"] >= 16 and api.DOC_ROOT_FILES["scenarios"] == "scenarios.md" and (ROOT / "docs" / "scenarios.md").exists()


def test_the_plan_lists_the_generators_and_what_the_form_chooses_among(app):
    plan = plain(app.extra_get["/api/scenarios/plan"]({}))
    assert [g["id"] for g in plan["generators"]] == list(GENERATORS) and plan["horizons"] == [5, 10, 21, 63] and plan["n_limits"] == [100, 2000] and plan["default"] == report.DEFAULT
    assert {g["family"] for g in plan["generators"]} <= set(plan["families"]) and plan["min_days"] == 500 and plan["max_assets"] == 12 and [m["id"] for m in plan["metrics"]] == list(report.METRICS)
    assert any(g["heavy"] for g in plan["generators"]) and not {g["id"]: g for g in plan["generators"]}["bootstrap"]["heavy"]


def test_a_run_is_a_job_on_the_apps_prices_and_touches_no_trial_count(app):
    before = {k: set(v) for k, v in app.trials.items()}
    view = _wait(app, _run(app, tickers=["A01", "A02", "A03"], generators=["bootstrap", "copula"], horizon=10, n=100, seed=2)["job"])
    assert view["status"] == "done", view
    d = plain(view["result"])
    assert d["assets"] == ["A01", "A02", "A03"] and [g["id"] for g in d["generators"]] == ["bootstrap", "copula"] and d["horizon"] == 10 and d["n"] == 100 and d["universe"] and d["stopped"] is False
    assert {k: set(v) for k, v in app.trials.items()} == before
    assert any(j["kind"] == "scenarios" for j in app.jobs_list()["jobs"])
    again = _wait(app, _run(app, tickers=["A01", "A02", "A03"], generators=["bootstrap", "copula"], horizon=10, n=100, seed=2)["job"])["result"]
    assert again["generators"][0]["terminal"] == d["generators"][0]["terminal"], "the same request gives the same scenarios"


def test_without_tickers_the_apps_default_universe_is_used_and_a_history_too_short_is_a_message(app, tmp_path):
    view = _wait(app, _run(app, generators=["bootstrap"], horizon=5, n=100)["job"])
    assert view["status"] == "done" and len(view["result"]["assets"]) == 4
    short = make_app(tmp_path, n=300)
    bad = _wait(short, _run(short, generators=["bootstrap"], horizon=5, n=100)["job"])
    assert bad["status"] == "error" and "at least 500 days" in bad["error"]


def test_every_refusal_is_a_message_before_a_job_starts(app):
    jobs = len(app.jobs_list()["jobs"])
    for body, why in [({"tickers": []}, "at least one ticker"), ({"tickers": "SPY"}, "at least one ticker"), ({"tickers": [f"T{i}" for i in range(13)]}, "at most 12"), ({"generators": "bootstrap"}, "list of generator names"),
                      ({"generators": ["nope"]}, "unknown generator 'nope'"), ({"generators": []}, "at least one generator"), ({"n": 50}, "between 100 and 2000"), ({"n": "many"}, "must be a number"), ({"n": 150.5}, "whole number"),
                      ({"horizon": 7}, "must be one of 5, 10, 21, 63"), ({"seed": -1}, "between 0 and")]:
        with pytest.raises(ApiError, match=why):
            _run(app, **{"generators": ["bootstrap"], **body})
    assert len(app.jobs_list()["jobs"]) == jobs, "a refused request starts nothing"
    with pytest.raises(ApiError, match="not going in this app"):
        app.extra_post["/api/scenarios/stop"]({"job": "nope"})


def test_the_new_files_run_in_the_lanes_that_read_their_dependencies():
    if not (ROOT / ".github/workflows/ci.yml").exists():                                         # the Docker image is built without `.github`
        pytest.skip("the workflow files are not in this tree")
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())

    def text(job):
        return " ".join(str(step.get("run", "")) for step in ci["jobs"][job]["steps"])

    assert "tests/test_scenarios_report.py" in text("replit-python"), "the endpoint test runs in the dashboard lane"


# ------------------------------------------------------------------------------------------------------------------------------ the terminal
@pytest.fixture(scope="module")
def prices(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("prices") / "prices.csv"
    r = world(800, assets=3)
    (100 * (1 + r).cumprod()).rename_axis("date").reset_index().to_csv(path, index=False)
    return str(path)


def test_the_list_names_every_generator_and_needs_no_data(capsys):
    assert main(["scenarios", "--list"]) == 0
    out = capsys.readouterr().out
    for name in GENERATORS:
        assert name in out


def test_scenarios_print_the_table_the_portfolio_and_can_be_saved(prices, tmp_path, capsys):
    saved = tmp_path / "scenarios.json"
    code = main(["scenarios", "x0", "X1", "--prices", prices, "--generators", "bootstrap", "copula", "--horizon", "5", "--n", "100", "--json", str(saved)])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("2 tickers (X0, X1), 799 days") and "100 scenarios of 5 days" in out, "tickers are found whatever their case"
    assert "Block bootstrap" in out and "Copula" in out and "perfect copy" in out and "the history" in out and "ES" in out
    data = json.loads(saved.read_text())
    assert data["assets"] == ["X0", "X1"] and [g["id"] for g in data["generators"]] == ["bootstrap", "copula"]


def test_the_terminal_says_what_is_wrong_and_returns_two(prices, capsys):
    for argv, why in [(["scenarios", "ZZZ", "--prices", prices], "'ZZZ' is not in the data"), (["scenarios", "--prices", prices, "--generators", "nope"], "unknown generator 'nope'"),
                      (["scenarios", "--prices", prices, "--n", "7"], "between 100 and 2000")]:
        assert main(argv) == 2
        assert why in capsys.readouterr().err
