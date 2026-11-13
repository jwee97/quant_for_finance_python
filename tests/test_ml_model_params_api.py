"""A training request's model.params through the web API: validated against each model's own schema, laid over the model's defaults, and recorded in full.

The form used to send every setting of the chosen model and the API refused more than twenty, so PatchTST (23 settings), the Temporal Fusion Transformer (21) and the recurrent PPO agent (22, 21 sent) could not be
trained from the Machine learning tab at all. These tests pin the behaviour that replaces the count: every model the catalogue lists accepts its full default object and every valid customisation, a
request that names only what it changes is trained with the defaults for the rest, the experiment records every setting it ran with, an experiment saved earlier with only some of its settings still opens, and
a wrong value, an unknown setting or a malformed request is refused before a job is queued, with a message that names the setting."""

from __future__ import annotations

import json
import warnings

import pytest

from src.ml import experiments as xe, models as md, param_schema as ps
from src.webapp import server
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.ml_param_cases import invalid_values, request_params, valid_values
from tests.test_ml_integration import API_BASE, _need
from tests.test_webapp import _http
from tests.test_webapp_assets_ml import _app, _safe, finished

warnings.filterwarnings("ignore")
CATALOGUE = {m["kind"]: m for m in md.model_catalog()}
AVAILABLE = [k for k, m in CATALOGUE.items() if m["available"]]
RELATED = {"patchtst": ("lookback", "patch_len", "d_model", "n_heads"), "itransformer": ("d_model", "n_heads"), "tft": ("hidden", "n_heads"), "gat": ("hidden", "heads")}                      # settings that must agree with each other


@pytest.fixture(scope="module")
def market():
    return make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)


@pytest.fixture(scope="module")
def mlapp(tmp_path_factory, market):
    tmp = tmp_path_factory.mktemp("model_params_api")
    return _app(tmp, lambda config: market, root=tmp)


@pytest.fixture(scope="module")
def ml(mlapp):
    return mlapp.extra_get["/api/ml/catalog"].__self__


def _req(kind: str, params: dict | None = None, **more) -> dict:
    """A request body for a model; the regime ridge needs the regime feature that its configuration asks for."""
    return {"model": {"kind": kind, **({} if params is None else {"params": params})}, **({"dataset": {"regime_detector": "vol_state"}} if kind == "regime_ridge" else {}), **more}


def _custom(kind: str) -> dict:
    """Every setting of the model at a valid value other than its default (the settings that must agree with each other are left at theirs)."""
    out = {}
    for name, s in ps.schema(kind).items():
        if s.extra or name in RELATED.get(kind, ()):
            continue
        other = [v for v in valid_values(s) if s.check(v, name) != s.default and not (s.type == ps.LIST and v == [])]
        if other:
            out[name] = other[0]
    return out


# ---------------------------------------------------------------------------------------------------------------------- the full default object, the original failure
@pytest.mark.parametrize("kind", sorted(CATALOGUE))
def test_a_request_with_every_default_setting_is_accepted_and_recorded_in_full_or_refused_for_its_package_never_for_its_size(ml, kind):
    body = _req(kind, request_params(kind))
    if not CATALOGUE[kind]["available"]:
        with pytest.raises(ApiError, match=r"model cannot be used here: "):
            ml._overrides(body)
        return
    over = ml._overrides(body)
    assert over["model"]["kind"] == kind and over["model"]["params"] == ps.effective_params(kind, {}) and list(over["model"]["params"]) == list(CATALOGUE[kind]["params"])


def test_the_models_the_limit_of_twenty_settings_refused_are_accepted(ml):
    ran = []
    for kind in ("patchtst", "tft", "ppo", "recurrent_ppo"):
        if not CATALOGUE[kind]["available"]:
            continue
        over = ml._overrides({"model": {"kind": kind, "params": request_params(kind)}})
        assert len(over["model"]["params"]) == len(CATALOGUE[kind]["params"]) > 20
        ran.append(kind)
    if not ran:
        pytest.skip("neither PyTorch nor the agents' packages are installed here")


@pytest.mark.parametrize("kind", sorted(CATALOGUE))
def test_a_request_that_names_only_what_it_changes_is_laid_over_the_defaults(ml, kind):
    if not CATALOGUE[kind]["available"]:
        with pytest.raises(ApiError, match=r"model cannot be used here: "):
            ml._overrides(_req(kind, {}))
        return
    for name, s in ps.schema(kind).items():
        if s.extra:
            continue
        for value in valid_values(s)[:2]:
            try:
                expected = ps.effective_params(kind, {name: value})
            except ps.ParamError:
                continue
            got = ml._overrides(_req(kind, {name: value}))["model"]["params"]
            if kind in xe.RL_KINDS and name != "cost_bps":
                assert got.pop("cost_bps") == 7.0, "an agent that is not told its cost trains against the backtest's (the default 5 + 2 bps)"
                expected.pop("cost_bps")
            assert got == expected, (kind, name, value)


@pytest.mark.parametrize("kind", sorted(CATALOGUE))
def test_a_request_that_customises_every_setting_at_once_is_accepted_and_the_model_is_built_from_it(ml, kind):
    if not CATALOGUE[kind]["available"]:
        with pytest.raises(ApiError, match=r"model cannot be used here: "):
            ml._overrides(_req(kind, _custom(kind)))
        return
    params = _custom(kind)
    over = ml._overrides(_req(kind, params))["model"]["params"]
    assert len(over) == len(CATALOGUE[kind]["params"]) and all(over[k] == ps.effective_params(kind, {k: v})[k] for k, v in params.items())
    try:
        md.make_model(kind, over, CATALOGUE[kind]["tasks"][0], 0)
    except md.ModelUnavailable as exc:
        pytest.skip(str(exc))


def test_related_settings_are_checked_together_and_the_message_names_both(ml):
    _need("torch")
    for kind, bad, why in (("patchtst", {"d_model": 30, "n_heads": 4}, r"d_model \(30\) must be a multiple of .*n_heads \(4\)"), ("patchtst", {"lookback": 6, "patch_len": 8}, r"lookback \(6\) must be at least .*patch_len \(8\)"),
                           ("tft", {"hidden": 30, "n_heads": 4}, r"hidden \(30\) must be a multiple of"), ("gat", {"hidden": 30, "heads": 4}, r"hidden \(30\) must be a multiple of")):
        with pytest.raises(ApiError, match=why):
            ml._overrides({"model": {"kind": kind, "params": bad}})


# ---------------------------------------------------------------------------------------------------------------------- the agent's cost
def test_an_agent_that_is_not_told_its_cost_trains_against_the_backtests_and_one_that_is_keeps_its_own(ml):
    for kind in ("ppo", "recurrent_ppo", "sac"):
        if not CATALOGUE[kind]["available"]:
            continue
        auto = ml._overrides({"model": {"kind": kind}, "backtest": {"cost_bps": 4.0, "slippage_bps": 2.0}})
        assert auto["model"]["params"]["cost_bps"] == 6.0 and auto["signal"]["method"] == "target_weights" and len(auto["model"]["params"]) == len(CATALOGUE[kind]["params"])
        own = ml._overrides({"model": {"kind": kind, "params": {"cost_bps": 1.5}}, "backtest": {"cost_bps": 4.0, "slippage_bps": 2.0}})
        assert own["model"]["params"]["cost_bps"] == 1.5
        every = ml._overrides({"model": {"kind": kind, "params": request_params(kind)}, "backtest": {"cost_bps": 4.0, "slippage_bps": 2.0}})
        assert every["model"]["params"]["cost_bps"] == 7, "a setting that is sent is the person's, even at the model's default"


# ---------------------------------------------------------------------------------------------------------------------- what is refused
def test_a_wrong_value_of_any_setting_of_any_model_is_refused_with_the_setting_named(ml):
    for kind in AVAILABLE:
        for name, s in ps.schema(kind).items():
            if s.extra:
                continue
            for value in invalid_values(s)[:2]:
                with pytest.raises(ApiError, match=f"model.params.{name}"):
                    ml._overrides({"model": {"kind": kind, "params": {name: value}}})


@pytest.mark.parametrize("body, text", [
    ({"model": "ridge"}, "model must be an object"), ({"model": []}, "model must be an object"), ({"model": {"kind": 5}}, "model.kind must be one of"), ({"model": {"kind": ["ridge"]}}, "model.kind must be one of"),
    ({"model": {"kind": "oracle"}}, "model.kind must be one of"), ({"model": {"kind": "ridge", "params": []}}, "model.params must be an object"), ({"model": {"kind": "ridge", "params": "alpha=1"}}, "model.params must be an object"),
    ({"model": {"kind": "ridge", "params": {"alpha": "5"}}}, r"model\.params\.alpha must be a number, not text"), ({"model": {"kind": "ridge", "params": {"alpha": True}}}, r"model\.params\.alpha must be a number, not true/false"),
    ({"model": {"kind": "ridge", "params": {"alpha": -1}}}, r"model\.params\.alpha must be a number, at least 0 \(got -1\)"), ({"model": {"kind": "ridge", "params": {"alpha": float("nan")}}}, "finite"),
    ({"model": {"kind": "ridge", "params": {"alpha": 10 ** 400}}}, "between -1e18 and 1e18"), ({"model": {"kind": "ridge", "params": {"depth": 3}}}, r"model\.params\.depth is not a setting of the Ridge model"),
    ({"model": {"kind": "ridge", "params": {"alpha; drop": 1}}}, r"model\.params\.'alpha; drop' is not a setting"), ({"model": {"kind": "random_forest", "params": {"max_depth": 2.5}}}, "must be a whole number"),
    ({"model": {"kind": "random_forest", "params": {"max_depth": 0}}}, "at least 1"), ({"model": {"kind": "random_forest", "params": {"max_features": 1.5}}}, "above 0 and at most 1"),
    ({"model": {"kind": "elasticnet", "params": {"l1_ratio": -0.1}}}, "at least 0 and at most 1"), ({"model": {"kind": "online_ridge", "params": {"forgetting": 0}}}, "above 0 and at most 1"),
    ({"model": {"kind": "ar", "params": {"columns": ["bad name!"]}}}, "cannot be a feature name"), ({"model": {"kind": "ar", "params": {"columns": 5}}}, "list of feature names"),
    ({"model": {"kind": "kalman", "params": {"r": 0}}}, "above 0"), ({"model": {"kind": "regime_ridge", "params": {"min_rows": 0}}}, "at least 1")])
def test_an_invalid_setting_or_a_malformed_model_is_refused_before_a_job_is_queued(mlapp, body, text):
    queued = len(mlapp.jobs)
    for route in ("/api/ml/train", "/api/ml/tune"):
        with pytest.raises(ApiError, match=text):
            mlapp.extra_post[route]({**API_BASE, **body})
    assert len(mlapp.jobs) == queued, "nothing refused may have been queued"


def test_a_network_s_wrong_setting_is_refused_before_a_job_is_queued(mlapp):
    _need("torch")
    queued = len(mlapp.jobs)
    for params, text in (({"dropout": 1}, r"model\.params\.dropout must be a number, at least 0 and below 1 \(got 1\)"), ({"device": "tpu"}, r"model\.params\.device must be one of \['auto', 'cpu', 'cuda'\]"),
                         ({"resume": 2}, "true or false"), ({"horizons": "1,x"}, "whole numbers separated by commas"), ({"batch_size": 4}, "at least 8"), ({"d_model": 30, "n_heads": 4}, "must be a multiple of"),
                         ({"patch_len": 99}, "must be at least")):
        with pytest.raises(ApiError, match=text):
            mlapp.extra_post["/api/ml/train"]({**API_BASE, "model": {"kind": "patchtst", "params": params}})
    assert len(mlapp.jobs) == queued


def test_the_endpoints_answer_400_with_the_message_over_http(mlapp):
    running = server.start(mlapp, "127.0.0.1", 0)
    try:
        for body, text in (({"model": {"kind": "ridge", "params": {"alpha": -1}}}, "model.params.alpha must be a number"), ({"model": {"kind": "ridge", "params": []}}, "model.params must be an object"),
                           ({"model": {"kind": "ridge", "params": {"nope": 1}}}, "is not a setting of the Ridge model"), ({"model": {"kind": "nope"}}, "model.kind must be one of")):
            status, raw, _ = _http(running, "/api/ml/train", body={**API_BASE, **body})
            assert status == 400 and text in json.loads(raw)["error"], (body, status, raw[:200])
        status, raw, _ = _http(running, "/api/ml/train", raw=b'{"model": {"kind": "ridge", "params": {"alpha": NaN}}}')
        assert status == 400, raw[:200]
    finally:
        running.stop()


# ---------------------------------------------------------------------------------------------------------------------- trained through the endpoint
def _config_of(mlapp, exp_id: str) -> dict:
    return _safe(mlapp.extra_get["/api/ml/experiment"]({"id": exp_id}))["report"]["config"]


def test_a_model_trained_from_the_settings_it_changed_runs_with_the_defaults_for_the_rest_and_records_all_of_them(mlapp):
    run = finished(mlapp, mlapp.extra_post["/api/ml/train"]({**API_BASE, "model": {"kind": "random_forest", "params": {"n_estimators": 20, "max_depth": 3}}, "name": "forest-changes-only"}))
    assert _config_of(mlapp, run["id"])["model"] == {"kind": "random_forest", "params": {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 200, "max_features": 0.5}}
    full = finished(mlapp, mlapp.extra_post["/api/ml/train"]({**API_BASE, "model": {"kind": "random_forest", "params": request_params("random_forest") | {"n_estimators": 20, "max_depth": 3}}, "name": "forest-every-setting"}))
    assert _config_of(mlapp, full["id"])["model"] == _config_of(mlapp, run["id"])["model"] and full["report"]["config_hash"] == run["report"]["config_hash"], "the same settings are one configuration however they were sent"


def test_the_autoregression_takes_its_columns_as_text_or_as_its_default_list(mlapp):
    plain = finished(mlapp, mlapp.extra_post["/api/ml/train"]({**API_BASE, "model": {"kind": "ar"}, "name": "ar-default"}))
    assert _config_of(mlapp, plain["id"])["model"]["params"]["columns"] == list(md.AR_FEATURES)
    text = finished(mlapp, mlapp.extra_post["/api/ml/train"]({**API_BASE, "model": {"kind": "ar", "params": {"columns": "ret_1, ret_lag_2"}}, "name": "ar-text"}))
    assert _config_of(mlapp, text["id"])["model"]["params"]["columns"] == ["ret_1", "ret_lag_2"]


TINY = {"epochs": 1, "max_rows": 1000, "hidden": 8, "layers": 1, "lookback": 10}                         # a network that trains in a few seconds
NETWORKS = [("patchtst", {"epochs": 1, "max_rows": 2000, "d_model": 16, "n_heads": 2, "ff_dim": 32}), ("tft", {"epochs": 1, "max_rows": 2000, "hidden": 16, "n_heads": 2}), ("nhits", {"epochs": 1, "max_rows": 2000}),
            ("lstm", {"epochs": 1, "max_rows": 2000}), ("gru", TINY), ("tcn", TINY), ("nbeats", {**TINY, "blocks": 2, "width": 16}), ("autoencoder", {**TINY, "latent_dim": 4}), ("vae", {**TINY, "latent_dim": 4}),
            ("gcn", {"epochs": 1, "hidden": 8, "graph": "correlation"}), ("gat", {"epochs": 1, "hidden": 8, "heads": 2, "graph": "correlation"})]


@pytest.mark.parametrize("kind, quick", NETWORKS)
def test_a_network_is_trained_through_the_endpoint_from_a_few_settings_and_records_every_one(mlapp, kind, quick):
    _need("torch")
    run = finished(mlapp, mlapp.extra_post["/api/ml/train"]({**API_BASE, "dataset": {**API_BASE["dataset"], "feature_groups": ["price", "momentum", "volatility"]}, "model": {"kind": kind, "params": quick},
                                                             "name": f"{kind}-few-settings", "explain": False}))
    recorded = _config_of(mlapp, run["id"])["model"]["params"]
    assert list(recorded) == list(CATALOGUE[kind]["params"]) and recorded["epochs"] == 1 and all(recorded[k] == v for k, v in quick.items())


@pytest.mark.parametrize("kind, quick", [("ppo", {"total_timesteps": 300, "n_steps": 64, "episode_days": 60, "hidden": 8}),
                                         ("recurrent_ppo", {"total_timesteps": 300, "n_steps": 64, "episode_days": 60, "hidden": 8, "lstm_hidden": 8})])
def test_an_agent_is_trained_through_the_endpoint_from_a_few_settings_and_records_every_one_including_its_cost(mlapp, kind, quick):
    _need("gymnasium", "stable_baselines3", "sb3_contrib")
    run = finished(mlapp, mlapp.extra_post["/api/ml/train"]({**API_BASE, "dataset": {**API_BASE["dataset"], "feature_groups": ["price", "momentum", "volatility"]}, "model": {"kind": kind, "params": quick},
                                                            "name": f"{kind}-few-settings", "explain": False}))
    recorded = _config_of(mlapp, run["id"])["model"]["params"]
    assert list(recorded) == list(CATALOGUE[kind]["params"]) and len(recorded) > 20 and all(recorded[k] == v for k, v in quick.items())
    assert recorded["cost_bps"] == 7.0, "the agent trained against the costs the backtest charges (the default 5 + 2 bps)"


# ---------------------------------------------------------------------------------------------------------------------- what was saved before
def test_an_experiment_saved_with_only_some_of_its_settings_still_opens_and_runs_again(mlapp, ml, market):
    """An experiment made before this change (or by a script or the command line) holds exactly the settings it was given. Nothing reads a saved configuration through the request validation, so it is read as it is."""
    legacy = xe.run_experiment({**API_BASE, "name": "saved-before", "model": {"kind": "random_forest", "params": {"max_depth": 3, "n_estimators": 10}}}, bundle=market, registry=ml.registry, explain=False, robustness=False,
                               final_model=False, stop_after="validation")
    assert _config_of(mlapp, legacy.id)["model"]["params"] == {"max_depth": 3, "n_estimators": 10}, "stored exactly as it was given"
    child = finished(mlapp, mlapp.extra_post["/api/ml/rebacktest"]({"id": legacy.id, "signal": {"method": "top_n_long_short", "n": 3}}))
    assert child["report"]["config"]["model"] == {"kind": "random_forest", "params": {"max_depth": 3, "n_estimators": 10}}
    assert _safe(mlapp.extra_get["/api/ml/retraining"]({"id": legacy.id}))["state"]["open_proposal"] is None
    rerun = md.make_model("random_forest", _config_of(mlapp, legacy.id)["model"]["params"], "regression", 0)
    assert rerun.params == {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 200, "max_features": 0.5}, "a model built from a partial configuration merges the defaults itself, as before"
    assert any(r["id"] == legacy.id for r in mlapp.extra_get["/api/ml/experiments"]({})["rows"])
