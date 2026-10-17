"""Deep-window and foundation-model plug-ins: causal, usable, parameter-checked, pipeline-ready; optional models skip cleanly when their dependency is absent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, Pipeline, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.strategies.deep import KINDS
from src.utils.config import load_config

load_library()


def _bundle(n: int, k: int = 5, seed: int = 4, planted: bool = True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    r = rng.normal(0.0002, 0.01, (n, k))
    if planted:
        for t in range(22, n):
            r[t] += 0.06 * np.tanh(r[t - 21:t].sum(axis=0) / 0.2) / 21
    return bundle_from_prices(pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=[f"A{i}" for i in range(k)]), name="deep")


@pytest.fixture(scope="module")
def bundle():
    return _bundle(2000)


@pytest.mark.parametrize("kind", KINDS)
def test_every_network_kind_gives_a_finite_causal_forecast(kind, bundle):
    model = MODELS.create("deep_window", kind=kind, min_train=900, max_epochs=4)
    assert check_causality(model, bundle, cutoff=bundle.index[1500])["ok"]
    f = model.forecast(bundle)
    values = f.mean.stack().dropna()
    assert len(values) > 500 and np.isfinite(values).all() and f.confidence.stack().dropna().between(0, 1).all()


def test_no_forecast_before_the_first_training_window(bundle):
    f = MODELS.create("deep_window", min_train=1000, max_epochs=3).forecast(bundle).mean
    assert f.dropna(how="all").index[0] >= bundle.index[1000]


def test_the_same_seed_gives_the_same_forecast(bundle):
    a = MODELS.create("deep_window", kind="tsmixer", min_train=1000, max_epochs=3, seeds=(3,)).forecast(bundle).mean
    b = MODELS.create("deep_window", kind="tsmixer", min_train=1000, max_epochs=3, seeds=(3,)).forecast(bundle).mean
    assert np.allclose(a.fillna(0.0), b.fillna(0.0))


@pytest.mark.parametrize("kwargs", [{"kind": "mamba"}, {"min_train": 300}, {"refit_every": 5}, {"embargo": -1}, {"max_epochs": 0}, {"patience": 0}, {"seeds": ()}, {"d_model": 2}])
def test_deep_window_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        MODELS.create("deep_window", **kwargs)


def test_deep_window_runs_through_the_pipeline_with_the_same_costs_as_any_model(bundle):
    spec = {"models": [{"name": "deep_window", "params": {"kind": "nbeats", "min_train": 1000, "max_epochs": 3}}], "allocation": {"allocator": "forecast_stack"},
            "evaluation": {"causality": False, "benchmarks": []}}
    out = Pipeline(spec, load_config(), bundle).run(validate=False)
    assert out.metrics["n_days"] > 200 and np.isfinite(out.metrics["sharpe"]) and out.metrics["ann_cost_bps"] > 0


@pytest.mark.parametrize("name", ["chronos", "timesfm"])
@pytest.mark.parametrize("kwargs", [{"batch": 0}, {"min_history": 100}])
def test_foundation_models_reject_invalid_parameters(name, kwargs):
    with pytest.raises(ValueError):
        MODELS.create(name, **kwargs)


def _cached(repo: str) -> bool:
    return (Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}").exists()


@pytest.mark.skipif(importlib.util.find_spec("chronos") is None or not _cached("amazon/chronos-bolt-small"), reason="chronos-forecasting or its cached weights are not available")
def test_chronos_zero_shot_is_causal_and_forecasts_after_enough_history():
    b = _bundle(700, k=3, planted=False)
    model = MODELS.create("chronos")
    assert check_causality(model, b, cutoff=b.index[550])["ok"]
    f = model.forecast(b).mean
    assert f.dropna(how="all").index[0] >= b.index[315] and np.isfinite(f.stack().dropna()).all() and f.stack().notna().any()


def test_chronos_runs_a_forecast_with_a_pretrained_pipeline(monkeypatch, bundle):
    import sys
    import types
    import torch

    calls = {}

    class FakePipeline:
        def predict_quantiles(self, context, prediction_length, quantile_levels):
            calls["prediction"] = (tuple(context.shape), prediction_length, quantile_levels)
            quantiles = torch.ones((len(context), prediction_length, len(quantile_levels)))
            return quantiles, None

    def from_pretrained(model, **kwargs):
        calls["load"] = (model, kwargs)
        return FakePipeline()

    monkeypatch.setitem(
        sys.modules,
        "chronos",
        types.SimpleNamespace(
            ChronosBoltPipeline=types.SimpleNamespace(from_pretrained=from_pretrained)
        ),
    )
    forecast = MODELS.create("chronos", batch=64).forecast(bundle).mean
    values = forecast.stack().dropna()

    assert calls["load"][0] == "amazon/chronos-bolt-small"
    assert calls["load"][1]["device_map"] == "cpu"
    assert calls["prediction"][1:] == (21, [0.5])
    assert values.any() and np.isfinite(values).all()


@pytest.mark.skipif(importlib.util.find_spec("timesfm") is None or not _cached("google/timesfm-2.5-200m-pytorch"), reason="timesfm or its cached weights are not available")
def test_timesfm_zero_shot_is_causal_and_forecasts_after_enough_history():
    b = _bundle(700, k=3, planted=False)
    model = MODELS.create("timesfm")
    assert check_causality(model, b, cutoff=b.index[550])["ok"]
    f = model.forecast(b).mean
    assert f.dropna(how="all").index[0] >= b.index[315] and np.isfinite(f.stack().dropna()).all() and f.stack().notna().any()


def test_foundation_models_say_what_to_install_when_the_dependency_is_missing(monkeypatch, bundle):
    import sys
    monkeypatch.setitem(sys.modules, "chronos", None)
    monkeypatch.setitem(sys.modules, "timesfm", None)
    with pytest.raises(ImportError, match=r"\.\[chronos\]"):
        MODELS.create("chronos").forecast(bundle)
    with pytest.raises(ImportError) as info:
        MODELS.create("timesfm").forecast(bundle)
    message = str(info.value)
    assert str(info.value.__cause__) in message and f"{sys.executable} -m pip install 'timesfm[torch]" in message       # why it failed, and the command for the interpreter that is running


def test_timesfm_names_pytorch_when_pytorch_is_what_cannot_be_imported(monkeypatch, bundle):
    import sys

    class NeedsTorch:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "timesfm":
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return None

    monkeypatch.delitem(sys.modules, "timesfm", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NeedsTorch(), *sys.meta_path])
    with pytest.raises(ImportError, match=r"No module named 'torch'.*download\.pytorch\.org/whl/cpu torch"):
        MODELS.create("timesfm").forecast(bundle)


def test_an_old_timesfm_without_the_2_5_model_is_reported_and_not_an_attribute_error(monkeypatch, bundle):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "timesfm", types.SimpleNamespace(__version__="1.3.0"))
    with pytest.raises(ImportError, match=r"timesfm 1\.3\.0 has no TimesFM 2\.5"):
        MODELS.create("timesfm").forecast(bundle)


def _fake_timesfm(calls):
    """A stand-in for the ``timesfm`` package with the three calls the plug-in makes: ``from_pretrained``, ``compile`` and ``forecast``. Its point forecast is the mean of the last 21 inputs, so it varies by window."""
    import types

    class Net:
        def compile(self, config):
            calls["config"] = config

        def forecast(self, horizon, inputs):
            calls.setdefault("batches", []).append((horizon, len(inputs), {w.dtype.name for w in inputs}, {len(w) for w in inputs}))
            return np.array([[float(np.mean(w[-21:]))] * horizon for w in inputs]), None

    def from_pretrained(model):
        calls["load"] = model
        return Net()

    return types.SimpleNamespace(TimesFM_2p5_200M_torch=types.SimpleNamespace(from_pretrained=from_pretrained), ForecastConfig=lambda **kwargs: kwargs)


def test_timesfm_runs_a_forecast_with_the_loaded_model(monkeypatch, bundle):
    """The plug-in loads the 2.5 weights, compiles them for a 21-day horizon and feeds the 252 normalised returns of every month-end: what the app does when the package is installed."""
    import sys
    calls = {}
    monkeypatch.setitem(sys.modules, "timesfm", _fake_timesfm(calls))
    values = MODELS.create("timesfm", batch=64).forecast(bundle).mean.stack().dropna()
    assert calls["load"] == "google/timesfm-2.5-200m-pytorch"
    assert calls["config"]["max_horizon"] == 21 and calls["config"]["max_context"] == 256 and calls["config"]["normalize_inputs"] is True
    assert all(h == 21 and n <= 64 and kinds == {"float32"} and lengths == {252} for h, n, kinds, lengths in calls["batches"])
    assert len(values) > 500 and np.isfinite(values).all() and values.nunique() > 100


def test_timesfm_runs_through_the_pipeline_like_any_other_model(monkeypatch, bundle):
    import sys
    monkeypatch.setitem(sys.modules, "timesfm", _fake_timesfm({}))
    spec = {"models": [{"name": "timesfm"}], "allocation": {"allocator": "forecast_stack"}, "evaluation": {"causality": False, "benchmarks": []}}
    out = Pipeline(spec, load_config(), bundle).run(validate=False)
    assert out.metrics["n_days"] > 200 and np.isfinite(out.metrics["ann_cost_bps"])
