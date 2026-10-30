"""The model families through the whole pipeline: an experiment on each kind of model saves what the model reports beside its scores, a re-backtest restores it, the report's series line up with their axes and
survive JSON, a missing optional package takes away only its own models, and the endpoints validate what they add."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.ml import experiments as xe, models as md, registry as rg
from src.utils.optional import unavailable_reason
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
BASE = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 400, "refit_every": 300}, "signal": {"n": 2}}
API_BASE = {**BASE, "split": {"min_train_days": 400, "refit_every": 252}}                  # the endpoint caps a refit interval at a year
OPTIONAL = ("torch", "gymnasium", "stable_baselines3", "sb3_contrib", "ngboost", "optuna", "hmmlearn", "lightgbm", "xgboost")


def _need(*packages):
    for pkg in packages:
        reason = unavailable_reason(pkg)
        if reason:
            pytest.skip(f"{pkg} is not available here: {reason}")


@pytest.fixture(scope="module")
def bundle():
    return make_bundle(8, 1200, seed=5, alpha=1.0)


def _plotted(x: dict) -> list[dict]:
    """The groups of lists the page draws against one axis."""
    groups = list(x["intervals"]["series"].values()) if "intervals" in x else []
    groups += [x[k][sub] for k, sub in (("ranking", "series"), ("portfolio", "series"), ("rewards", "in_sample"), ("rewards", "curve")) if k in x]
    groups += [{"dates": x[k]["dates"], "rows": x[k][rows]} for k, rows in (("regimes", "probabilities"), ("weights", "weights")) if k in x]
    return groups


def _checked(report: dict) -> dict:
    """The model section of a report as the page receives it: JSON with no NaN, and every list plotted against an axis as long as that axis."""
    x = report["model_extras"]
    json.dumps(report, allow_nan=False)
    for group in _plotted(x):
        lengths = {len(v) for v in group.values() if isinstance(v, list)}
        assert len(lengths) == 1 and lengths.pop() > 1, {k: len(v) for k, v in group.items() if isinstance(v, list)}
    assert x["resources"]["train_seconds"] >= 0 and x["reproducibility"]["seed"] == report["config"]["seed"] and x["reproducibility"]["config_hash"] == report["config_hash"]
    return x


# ------------------------------------------------------------------------------------------------------------ optional packages
def test_importing_the_machine_learning_modules_loads_none_of_the_optional_packages():
    code = ("import sys, json\nimport src.ml.models, src.ml.training, src.ml.experiments, src.ml.signals, src.webapp.ml_api\n"
            f"print(json.dumps([m for m in {OPTIONAL!r} if m in sys.modules]))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)}, check=False)
    assert out.returncode == 0, out.stderr[-800:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [], "an optional package loaded at import: the tab would fail to open without it"


@pytest.fixture
def without_extras(tmp_path, monkeypatch):
    """The packages that are not in requirements.txt as a machine without them behaves: importing each raises, and PyTorch is reported missing to the catalogue."""
    for name in ("ngboost", "gymnasium", "stable_baselines3", "sb3_contrib", "optuna"):
        (tmp_path / f"{name}.py").write_text(f"raise ModuleNotFoundError(\"No module named '{name}'\", name='{name}')\n")
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    real = md.unavailable_reason
    monkeypatch.setattr(md, "unavailable_reason", lambda name: "the 'torch' package is not installed (pip install torch)" if name == "torch" else real(name))


def test_a_missing_package_marks_only_its_own_models_unavailable_and_the_rest_still_trains(without_extras, bundle):
    cat = {m["kind"]: m for m in md.model_catalog()}
    gone = {"ngboost": "ngboost", "ppo": "gymnasium", "sac": "gymnasium", "recurrent_ppo": "gymnasium", "lstm": "torch", "tft": "torch", "gcn": "torch", "vae": "torch"}
    for kind, pkg in gone.items():
        assert cat[kind]["available"] is False and pkg in cat[kind]["reason"], (kind, cat[kind]["reason"])
    for kind in ("ridge", "lasso", "random_forest", "hist_gb", "hmm_regime", "model_ensemble"):
        assert cat[kind]["available"] is True, kind
    for kind in ("ngboost", "ppo", "lstm"):
        with pytest.raises(md.ModelUnavailable, match=gone[kind]):
            md.make_model(kind)
    assert xe.run_experiment({**BASE, "model": {"kind": "ridge", "params": {"alpha": 10.0}}}, bundle=bundle, final_model=False, explain=False).report["ic"]["all"]["n"] > 100


# ------------------------------------------------------------------------------------------------------------ experiments end to end
@pytest.fixture(scope="module")
def quantile(bundle, tmp_path_factory):
    _need("lightgbm")
    reg = rg.Registry(tmp_path_factory.mktemp("q") / "ml")
    run = xe.run_experiment({**BASE, "model": {"kind": "lgbm_quantile", "params": {"n_estimators": 30}}, "signal": {"method": "uncertainty_weighted", "n": 2}}, bundle=bundle, registry=reg, explain=False)
    return reg, run


def test_a_quantile_experiment_reports_intervals_ranking_and_provenance_and_saves_them(quantile):
    reg, run = quantile
    x = _checked(run.report)
    assert x["family"] == "probabilistic" and {"intervals", "ranking", "resources", "reproducibility"} <= set(x)
    iv = x["intervals"]
    assert 0.5 < iv["coverage"] <= 1.0 and iv["nominal"] == pytest.approx(0.8) and len(iv["by_uncertainty"]) == 5 and len(iv["series"]) == 5
    assert x["ranking"]["ndcg"] > 0 and set(x["ranking"]["by_segment"]) <= {"validation", "test", "forward"}
    assert x["reproducibility"]["model_artifact"]["bytes"] > 0 and x["reproducibility"]["versions"]["lightgbm"] and "python" in x["reproducibility"]["versions"]
    d = reg.root / "experiments" / run.id
    assert (d / "extra_distribution.parquet").exists() and {a["name"] for a in reg.get_experiment(run.id)["artifacts"]} >= {"extra_distribution", "model", "report_json"}
    assert json.loads((d / "report.json").read_text())["model_extras"]["intervals"]["coverage"] == pytest.approx(iv["coverage"])


@pytest.mark.parametrize("rule", ["uncertainty_weighted", "uncertainty_mean_variance", "top_n_long_short"])
def test_a_rebacktest_restores_the_saved_distribution_and_can_use_it_in_a_new_position_rule(quantile, rule):
    reg, run = quantile
    cfg, ds, train, info = xe.load_run(reg, run.id)
    assert "distribution" in train.extras and set(train.extras["distribution"].columns) >= {"mean", "std", "lower", "upper"} and train.extras["distribution"].index.equals(train.predictions.index)
    child = xe.rebacktest(reg, run.id, {"signal": {"method": rule, "n": 2}}, save=False)
    assert child.report["config"]["signal"]["method"] == rule and child.report["segments"]["stats"]["test"]["n_days"] > 50
    assert child.report["model_extras"]["intervals"]["coverage"] == pytest.approx(run.report["model_extras"]["intervals"]["coverage"]), "the child reports the intervals the saved run produced"
    json.dumps(child.report, allow_nan=False)


def test_a_ranker_and_a_regime_experiment_report_ranking_quality_and_filtered_regime_probabilities(bundle):
    _need("lightgbm", "hmmlearn")
    ranker = xe.run_experiment({**BASE, "model": {"kind": "lgbm_ranker", "params": {"n_estimators": 30}}}, bundle=bundle, final_model=False, explain=False)
    xr = _checked(ranker.report)
    assert xr["family"] == "ranking" and xr["ranking"]["ndcg"] > xr["ranking"]["ndcg_random"] and "intervals" not in xr
    regime = xe.run_experiment({**BASE, "model": {"kind": "hmm_regime", "params": {"n_states": 2}}}, bundle=bundle, final_model=False, explain=False)
    x = _checked(regime.report)
    probs = pd.DataFrame(x["regimes"]["probabilities"])
    assert x["family"] == "regime" and probs.shape[1] == 2 and (probs.sum(axis=1) - 1).abs().max() < 1e-6 and sum(x["regimes"]["share"]) == pytest.approx(1.0, abs=1e-6)


def test_an_agent_that_is_ranked_by_another_rule_is_warned_and_one_that_trades_its_weights_is_not(bundle):
    _need("gymnasium", "stable_baselines3")
    agent = {"kind": "ppo", "params": {"total_timesteps": 256, "n_steps": 64, "batch_size": 32, "episode_days": 80, "hidden": 8}}
    ranked = xe.run_experiment({**BASE, "model": agent, "signal": {"method": "top_n_long_short", "n": 2}}, bundle=bundle, final_model=False, explain=False)
    own = xe.run_experiment({**BASE, "model": agent, "signal": {"method": "target_weights"}}, bundle=bundle, final_model=False, explain=False)
    assert any("sizes its own positions" in w for w in ranked.report["warnings"]) and not any("sizes its own positions" in w for w in own.report["warnings"])
    x = _checked(own.report)
    assert x["family"] == "reinforcement" and x["rewards"]["algo"] == "ppo" and x["rewards"]["curve"]["timesteps"] and len(x["rewards"]["in_sample"]["dates"]) > 100
    assert x["resources"]["devices"] == ["cpu"] or "cuda" in x["resources"]["devices"]


# ------------------------------------------------------------------------------------------------------------ the endpoints
@pytest.fixture(scope="module")
def mlapp(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models_api")
    data = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    return _app(tmp, lambda config: data, root=tmp)


def test_the_catalog_carries_every_models_settings_family_and_outputs_and_the_search_engines(mlapp):
    cat = _safe(mlapp.extra_get["/api/ml/catalog"]({}))
    kinds = {m["kind"]: m for m in cat["models"]}
    new = {"lgbm_ranker", "xgb_ranker", "lgbm_quantile", "ngboost", "hmm_regime", "markov_switching", "lstm", "gru", "tcn", "patchtst", "nbeats", "nhits", "tft", "autoencoder", "vae", "gcn", "gat", "model_ensemble", "ppo", "sac", "recurrent_ppo"}
    assert new <= set(kinds) and {"ridge", "lasso", "random_forest", "hist_gb"} <= set(kinds) and kinds["ridge"]["available"]
    assert all({"family", "specs", "needs", "outputs", "params", "available"} <= set(kinds[k]) for k in new)
    assert {kinds[k]["family"] for k in new} == {"ranking", "probabilistic", "regime", "deep sequence", "forecasting", "representation", "graph", "ensemble", "reinforcement"}
    assert kinds["lgbm_quantile"]["outputs"] == ["distribution"] and "rewards" in kinds["ppo"]["outputs"] and kinds["xgb_ranker"]["specs"]["objective"]["choices"][0] == "rank:ndcg"
    assert [e["id"] for e in cat["engines"]] == ["builtin", "optuna"] and cat["engines"][0]["available"] is True
    assert {"uncertainty_weighted", "uncertainty_mean_variance", "target_weights"} <= {s["id"] for s in cat["signals"]}
    assert all(s["text"] for s in cat["signals"]), "every position rule says what it does"


def test_an_agent_is_given_the_position_rule_and_the_costs_it_trains_against_unless_the_request_says_otherwise(mlapp):
    _need("gymnasium", "stable_baselines3")
    ml = mlapp.extra_get["/api/ml/catalog"].__self__
    over = ml._overrides({"model": {"kind": "ppo"}, "backtest": {"cost_bps": 4.0, "slippage_bps": 2.0}})
    assert over["signal"]["method"] == "target_weights" and over["model"]["params"]["cost_bps"] == 6.0
    custom = ml._overrides({"model": {"kind": "ppo", "params": {"cost_bps": 1.5}}, "signal": {"method": "top_n_long_short"}})
    assert custom["signal"]["method"] == "top_n_long_short" and custom["model"]["params"]["cost_bps"] == 1.5
    plain = ml._overrides({"model": {"kind": "ridge"}})
    assert "cost_bps" not in plain["model"]["params"] and plain.get("signal", {}).get("method") != "target_weights"
    with pytest.raises(ApiError, match="model.params"):
        ml._overrides({"model": {"kind": "ppo", "params": {"hidden; drop": 1}}})


def test_the_tune_endpoint_checks_the_engine_and_says_when_optuna_is_missing(mlapp, monkeypatch):
    tune = mlapp.extra_post["/api/ml/tune"]
    with pytest.raises(ApiError, match="engine must be builtin or optuna"):
        tune({"model": {"kind": "ridge"}, "engine": "hyperopt"})
    monkeypatch.setattr("src.webapp.ml_api.unavailable_reason", lambda name: "the 'optuna' package is not installed (pip install optuna)")
    with pytest.raises(ApiError, match=r"Optuna engine is not available here: .*pip install optuna"):
        tune({"model": {"kind": "ridge"}, "engine": "optuna"})
    engines = {e["id"]: e for e in mlapp.extra_get["/api/ml/catalog"]({})["engines"]}
    assert engines["optuna"]["available"] is False and "pip install optuna" in engines["optuna"]["reason"] and engines["builtin"]["available"] is True


def test_a_tuning_job_can_use_optuna_and_a_new_model_trains_through_the_endpoint(mlapp):
    _need("optuna", "lightgbm")
    out = finished(mlapp, mlapp.extra_post["/api/ml/tune"]({**API_BASE, "model": {"kind": "ridge"}, "engine": "optuna", "n_trials": 5, "finalize": False}))
    assert out["n_trials"] == 5 and out["best"] and "model.params.alpha" in out["space"] and out["final"] is None
    run = finished(mlapp, mlapp.extra_post["/api/ml/train"]({**API_BASE, "model": {"kind": "lgbm_quantile", "params": {"n_estimators": 20}}, "name": "api-quantile"}))
    extras = run["report"]["model_extras"]
    assert extras["intervals"]["coverage"] > 0.4 and extras["family"] == "probabilistic" and run["report"]["config"]["model"]["kind"] == "lgbm_quantile"
