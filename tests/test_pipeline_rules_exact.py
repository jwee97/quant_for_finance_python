"""What the pipeline hands to each combination rule, adjustment and calibration, and what it keeps of what they return.

The rules themselves (trust weights, decay, the information-coefficient combinations, the calibration of confidence) are tested where they are defined. What is left to ``framework/pipeline.py`` is plumbing:
which function a specification's rule names, with which of its numbers and which defaults, what is remembered for the report, and how the tables are built from what comes back. A run of the real rules takes
minutes on a real market; stand-ins that record their arguments and return a known answer let each decision be stated exactly, in a fraction of a second.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.framework import ForecastModel, Pipeline, bundle_from_prices
from src.framework import pipeline as module
from src.framework.pipeline import PipelineSpec
from src.framework.registry import MODELS, register_model
from src.framework.types import ForecastPanel
from src.utils.config import load_config
from src.utils.dates import rebalance_dates


class Config:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class Spy:
    """Records every call and returns what it was given to return."""

    def __init__(self, result=None):
        self.result, self.calls = result, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def bare(spec=None, config=None, bundle=None) -> Pipeline:
    p = object.__new__(Pipeline)
    p.spec, p.config, p.bundle = spec or PipelineSpec(), config or Config(), bundle
    p._decay_table = p._ic_table = p._rule = None
    return p


def panel(level: float) -> ForecastPanel:
    index = pd.bdate_range("2021-01-04", periods=5)
    mean = pd.DataFrame(level, index=index, columns=["A", "B"])
    return ForecastPanel(mean, mean * 0 + 0.02, mean * 0 + 0.5)


PANELS = {"m": panel(0.01), "n": panel(0.03)}
MODELS_TWO = [SimpleNamespace(name="m"), SimpleNamespace(name="n")]


# ------------------------------------------------------------------------------------------------------------------ the forecast calibration
def test_the_calibration_of_a_forecast_is_read_from_the_configuration_with_its_defaults():
    class Recorder:
        name = "m"

        def __init__(self):
            self.calls = []

        def forecast(self, bundle, *args):
            self.calls.append((bundle, args))
            return f"panel for {bundle}"

    one = Recorder()
    assert bare(bundle="data")._forecasts([one]) == {"m": "panel for data"} and one.calls == [("data", (504, False, 40.0))], "the platform's calibration when the configuration has none"
    asked = Recorder()
    config = Config(**{"framework.forecast": {"calibration": {"min_observations": "300", "allow_negative_slope": 1}, "volatility_halflife_days": 20}})
    bare(config=config, bundle="data")._forecasts([asked])
    (_, args), = asked.calls
    assert args == (300, True, 20.0) and [type(a) for a in args] == [int, bool, float], "numbers written as text are numbers"
    for empty in (Config(**{"framework.forecast": None}), Config(**{"framework.forecast": {"calibration": None}})):
        again = Recorder()
        bare(config=empty, bundle="data")._forecasts([again])
        assert again.calls[0][1] == (504, False, 40.0), "an empty section is no choice"


# ------------------------------------------------------------------------------------------------------------------ the combination rules
@pytest.mark.parametrize("rule", ["equal", "confidence", "precision"])
def test_the_rules_that_need_nothing_but_the_forecasts_combine_them_and_keep_no_weights(monkeypatch, rule):
    combined = Spy("combined")
    monkeypatch.setattr(module, "combine_forecasts", combined)
    assert bare(PipelineSpec(combination={"rule": rule}))._combine(PANELS, MODELS_TWO, None) == ("combined", None) and combined.calls == [((PANELS, rule), {})]


def test_the_combination_by_trailing_information_coefficients_uses_the_returns_and_keeps_its_weights(monkeypatch):
    weights, combined = pd.DataFrame({"m": [0.7], "n": [0.3]}), Spy("combined")
    trusted = Spy(weights)
    monkeypatch.setattr(module, "combine_forecasts", combined)
    monkeypatch.setattr(module, "ic_trust_weights", trusted)
    data = SimpleNamespace(returns="returns")
    got = bare(PipelineSpec(combination={"rule": "ic_weighted"}), bundle=data)._combine(PANELS, MODELS_TWO, None)
    assert got == ("combined", weights) and got[1] is weights and trusted.calls == [((PANELS, "returns"), {})] and combined.calls == [((PANELS, "given", weights), {})]


def test_the_combination_by_regime_trusts_each_model_by_its_own_standalone_run_in_the_regime_of_the_day(monkeypatch):
    index = pd.bdate_range("2021-01-04", periods=30)
    labels, standalone, weights = pd.Series("Calm", index=index), pd.DataFrame({"m": 0.001, "n": 0.002}, index=index), pd.DataFrame({"m": [0.5], "n": [0.5]})
    trust, combined = Spy(weights), Spy("combined")
    monkeypatch.setattr(module, "regime_trust_weights", trust)
    monkeypatch.setattr(module, "combine_forecasts", combined)
    regimes, engine, data = SimpleNamespace(hard_labels=lambda: labels), BacktestEngine(), SimpleNamespace(index=index)
    p = bare(PipelineSpec(combination={"rule": "regime_conditional"}), bundle=data)
    p._standalone = Spy(standalone)
    assert p._combine(PANELS, MODELS_TWO, engine, regimes) == ("combined", weights)
    assert p._standalone.calls == [((PANELS, MODELS_TWO, {}, engine), {})], "the net returns of each model on its own"
    (args, kwargs), = trust.calls
    assert args[0] is standalone and args[1] is labels and list(args[2]) == list(rebalance_dates(index, "monthly")) and args[3:] == (126, 0.5) and kwargs == {}, "the defaults: 126 days in a regime, half shrunk"
    assert combined.calls == [((PANELS, "given", weights), {})]
    custom = bare(PipelineSpec(combination={"rule": "regime_conditional", "min_regime_days": "40", "shrink": 0.2}), bundle=data)
    custom._standalone = Spy(standalone)
    custom._combine(PANELS, MODELS_TWO, engine, regimes)
    assert trust.calls[-1][0][3:] == (40, 0.2) and isinstance(trust.calls[-1][0][3], int)


def test_the_decay_weighted_combination_passes_its_lags_window_and_holding_and_keeps_the_decay_table(monkeypatch):
    weights, decay, combined = pd.DataFrame({"m": [0.4], "n": [0.6]}), pd.DataFrame({"ic0": [0.1]}), Spy("combined")
    spy = Spy((weights, decay))
    monkeypatch.setattr(module, "decay_trust_weights", spy)
    monkeypatch.setattr(module, "combine_forecasts", combined)
    data = SimpleNamespace(returns="returns")
    p = bare(PipelineSpec(combination={"rule": "decay_weighted"}), bundle=data)
    assert p._combine(PANELS, MODELS_TWO, None) == ("combined", weights) and p._decay_table is decay and combined.calls == [((PANELS, "given", weights), {})]
    assert spy.calls == [((PANELS, "returns", (1, 5, 10, 21, 42, 63), 756, 252, 21), {})], "the defaults"
    custom = bare(PipelineSpec(combination={"rule": "decay_weighted", "lags": [2, 4], "window": "300", "min_obs": 100, "holding": 10}), bundle=data)
    custom._combine(PANELS, MODELS_TWO, None)
    assert spy.calls[-1] == ((PANELS, "returns", (2, 4), 300, 100, 10), {})


def test_the_optimal_information_coefficient_combination_hands_on_the_weights_as_shares_of_the_day(monkeypatch):
    index = pd.bdate_range("2021-01-04", periods=3)
    weights, expected, combined = pd.DataFrame({"m": [3.0, 0.0, 1.0], "n": [1.0, 0.0, 1.0]}, index=index), pd.Series([0.1, 0.2, 0.3], index=index), Spy("combined")
    spy = Spy((weights, expected))
    monkeypatch.setattr(module, "optimal_ic_weights", spy)
    monkeypatch.setattr(module, "combine_forecasts", combined)
    data = SimpleNamespace(returns="returns")
    p = bare(PipelineSpec(combination={"rule": "optimal_ic"}), bundle=data)
    panel_, shares = p._combine(PANELS, MODELS_TWO, None)
    assert panel_ == "combined" and combined.calls == [((PANELS, "given", weights), {})], "the combination uses the weights as they are"
    assert shares["m"].tolist() == [0.75, 0.5, 0.5] and shares["n"].tolist() == [0.25, 0.5, 0.5], "the shares: each weight over the day's total, equal when the total is nothing"
    assert p._ic_table[0] is shares and p._ic_table[1] is expected and spy.calls == [((PANELS, "returns", 504, 252, 0.3), {})]
    custom = bare(PipelineSpec(combination={"rule": "optimal_ic", "window": 100, "min_obs": "50", "shrink": 0.1}), bundle=data)
    custom._combine(PANELS, MODELS_TWO, None)
    assert spy.calls[-1] == ((PANELS, "returns", 100, 50, 0.1), {})
    orthogonal = Spy(("orthogonal panel", weights, expected))
    monkeypatch.setattr(module, "orthogonal_ic_forecast", orthogonal)
    o = bare(PipelineSpec(combination={"rule": "orthogonal_ic"}), bundle=data)
    assert o._combine(PANELS, MODELS_TWO, None) == ("orthogonal panel", weights) and o._ic_table == (weights, expected) and spy.calls[-1] == ((PANELS, "returns", 100, 50, 0.1), {}), "it does not use the optimal weights"
    assert orthogonal.calls == [((PANELS, "returns", "gram_schmidt", 504, 252, 0.3), {})]
    pca = bare(PipelineSpec(combination={"rule": "orthogonal_ic", "orthogonalize": "pca", "window": 100, "min_obs": 50, "shrink": 0.1}), bundle=data)
    pca._combine(PANELS, MODELS_TWO, None)
    assert orthogonal.calls[-1] == ((PANELS, "returns", "pca", 100, 50, 0.1), {})


def test_the_cost_aware_combination_trusts_the_trailing_sharpe_of_each_model_net_of_its_costs(monkeypatch):
    index = pd.bdate_range("2021-01-04", periods=30)
    standalone, weights, combined = pd.DataFrame({"m": 0.001, "n": 0.002}, index=index), pd.DataFrame({"m": [0.5], "n": [0.5]}), Spy("combined")
    sharpe = Spy(pd.Series([1.0, 2.0], index=index[:2]))
    trust = Spy(weights)
    monkeypatch.setattr(module, "trailing_sharpe", sharpe)
    monkeypatch.setattr(module, "trust_weights", trust)
    monkeypatch.setattr(module, "combine_forecasts", combined)
    engine, data = BacktestEngine(), SimpleNamespace(index=index)
    p = bare(PipelineSpec(combination={"rule": "cost_aware"}), bundle=data)
    p._standalone = Spy(standalone)
    assert p._combine(PANELS, MODELS_TWO, engine) == ("combined", weights) and p._standalone.calls == [((PANELS, MODELS_TWO, {}, engine), {})]
    assert [c[0][1:] for c in sharpe.calls] == [(504, 252), (504, 252)] and [c[0][0].name for c in sharpe.calls] == ["m", "n"], "each model's own stream, over the defaults"
    (args, kwargs), = trust.calls
    assert list(args[0].columns) == ["m", "n"] and list(args[1]) == list(rebalance_dates(index, "monthly")) and kwargs == {"shrink": 0.5}
    assert combined.calls == [((PANELS, "given", weights), {})]
    custom = bare(PipelineSpec(combination={"rule": "cost_aware", "window": "200", "min_history": 60, "shrink": 0.25}), bundle=data)
    custom._standalone = Spy(standalone)
    custom._combine(PANELS, MODELS_TWO, engine)
    assert sharpe.calls[-1][0][1:] == (200, 60) and trust.calls[-1][1] == {"shrink": 0.25}


def test_a_model_on_its_own_is_run_through_the_forecast_stack_in_its_position_mode_and_reports_net_or_gross(monkeypatch):
    class Allocators:
        def __init__(self):
            self.created = []

        def create(self, name, **params):
            self.created.append((name, params))
            return SimpleNamespace(build=lambda ctx: f"weights of {ctx.forecasts}")

    class Engine:
        def __init__(self):
            self.calls = []

        def run(self, weights, returns, name, investable, apply_vol_target=None):
            self.calls.append((weights, returns, name, investable, apply_vol_target))
            index = pd.bdate_range("2021-01-04", periods=3)
            return SimpleNamespace(gross_returns=pd.Series([0.02, 0.01, 0.0], index=index), net_returns=pd.Series([0.01, 0.0, -0.01], index=index))

    allocators, engine = Allocators(), Engine()
    monkeypatch.setattr(module, "ALLOCATORS", allocators)
    data = SimpleNamespace(returns="returns", investable="investable")
    models = [SimpleNamespace(name="m"), SimpleNamespace(name="n", position_mode="time_series")]
    p = bare(bundle=data)
    net = p._standalone({"m": "forecast m", "n": "forecast n"}, models, {}, engine)
    assert allocators.created == [("forecast_stack", {"mode": "cross_sectional"}), ("forecast_stack", {"mode": "time_series"})]
    assert engine.calls == [("weights of forecast m", "returns", "m", "investable", True), ("weights of forecast n", "returns", "n", "investable", True)], "volatility-targeted, as a book on its own would be"
    assert list(net.columns) == ["m", "n"] and net["m"].tolist() == [0.01, 0.0, -0.01], "net of costs unless asked for the gross"
    assert p._standalone({"m": "forecast m", "n": "forecast n"}, models, {}, engine, gross=True)["m"].tolist() == [0.02, 0.01, 0.0]


def test_the_adjustments_of_a_forecast_pass_their_options_in_order_and_keep_the_scale(monkeypatch):
    scaled, calibrated, scale = panel(0.02), panel(0.03), pd.Series([1.0, 2.0])
    spread, confidence = Spy((scaled, scale)), Spy(calibrated)
    monkeypatch.setattr(module, "regime_spread_scale", spread)
    monkeypatch.setattr(module, "calibrate_confidence", confidence)
    data = SimpleNamespace(returns="returns")
    start = panel(0.01)
    p = bare(PipelineSpec(forecast={"regime_spread": {}, "confidence": {}}), bundle=data)
    out, kept = p._adjust_forecast(start, "regimes")
    assert out is calibrated and list(kept) == ["regime_spread_scale"] and kept["regime_spread_scale"] is scale
    assert spread.calls == [((start, "returns", "regimes", 1000, 0.5), {})], "the defaults: a thousand days, half shrunk"
    assert confidence.calls == [((scaled, "returns", "platt", 2000, 21), {})], "the calibration is applied to the forecast that was rescaled, with its defaults"
    custom = bare(PipelineSpec(forecast={"regime_spread": {"min_obs": "300", "shrink": 0.2}, "confidence": {"method": "isotonic", "min_obs": 400, "refit_every": "5"}}), bundle=data)
    custom._adjust_forecast(start, "regimes")
    assert spread.calls[-1] == ((start, "returns", "regimes", 300, 0.2), {}) and confidence.calls[-1] == ((scaled, "returns", "isotonic", 400, 5), {})
    only = bare(PipelineSpec(forecast={"confidence": {}}), bundle=data)
    assert only._adjust_forecast(start, None) == (calibrated, {}) and confidence.calls[-1][0][0] is start and len(spread.calls) == 2, "the spread is not asked for, so it is not run"


# ------------------------------------------------------------------------------------------------------------------ what a run keeps of them
class TrendUp(ForecastModel):
    name = "test_rules_up"

    def score(self, data):
        return data.prices.pct_change(40)


class TrendDown(ForecastModel):
    name = "test_rules_down"

    def score(self, data):
        return -data.prices.pct_change(5)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def bundle():
    steps = np.random.default_rng(5).normal(0.0, 0.01, (1300, 6)) * np.linspace(0.6, 1.6, 6) + np.linspace(-0.0006, 0.0010, 6)
    index = pd.bdate_range("2017-01-02", periods=1300)
    return bundle_from_prices(pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=list("ABCDEF")), name="trending", min_history=40)


@pytest.fixture(autouse=True)
def registered():
    register_model("test_rules_up")(TrendUp)
    register_model("test_rules_down")(TrendDown)
    yield
    MODELS.unregister("test_rules_up")
    MODELS.unregister("test_rules_down")


def spec(**parts) -> dict:
    base = {"name": "t", "models": [{"name": "test_rules_up"}, {"name": "test_rules_down"}],
            "evaluation": {"benchmarks": [], "n_trials": 1, "causality": False, "brinson": False, "model_attribution": False}}
    for key, value in parts.items():
        base[key] = {**base[key], **value} if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return base


def test_the_tables_of_the_information_coefficient_rules_are_built_from_the_window_of_the_run(monkeypatch, bundle, config):
    index = bundle.index
    up = pd.Series(np.linspace(0.6, 0.9, len(index)), index=index)
    weights = pd.DataFrame({"test_rules_up": up, "test_rules_down": 1.0 - up})
    expected = pd.Series(np.where(np.arange(len(index)) < 700, np.nan, 0.2), index=index)
    expected.iloc[-100:] = 0.5
    monkeypatch.setattr(module, "optimal_ic_weights", Spy((weights, expected)))
    out = Pipeline(spec(combination={"rule": "optimal_ic"}), config, bundle).run(validate=False)
    table = out.tables["ic_combination"]
    assert table.index.tolist() == ["test_rules_up", "test_rules_down"] and table.columns.tolist() == ["mean_weight", "last_weight"]
    late = up.loc[out.start:]
    assert table["mean_weight"].tolist() == pytest.approx([late.mean(), 1.0 - late.mean()]) and table["last_weight"].tolist() == pytest.approx([0.9, 0.1]), "the average over the window, and the last day"
    window = expected.loc[out.start:]
    assert window.iloc[0] != window.iloc[0] and window.notna().any(), "the window starts before the expected ratio is known"
    assert out.metrics["expected_ic_ir"] == pytest.approx(float(window.dropna().mean())) and 0.2 < out.metrics["expected_ic_ir"] < 0.5
    monkeypatch.setattr(module, "optimal_ic_weights", Spy((weights, expected * np.nan)))
    nothing = Pipeline(spec(combination={"rule": "optimal_ic"}), config, bundle).run(validate=False)
    assert "ic_combination" in nothing.tables and "expected_ic_ir" not in nothing.metrics, "no expected information ratio in the window, none reported"
    helper = Pipeline(spec(models=[{"name": "test_rules_up"}]), config, bundle)
    real = helper._forecasts(helper._models())["test_rules_up"]
    monkeypatch.setattr(module, "orthogonal_ic_forecast", Spy((real, weights, expected)))
    ortho = Pipeline(spec(combination={"rule": "orthogonal_ic"}), config, bundle).run(validate=False)
    mean_up = up.loc[ortho.start:].mean()                                    # its own window: the forecast of one model starts a little before the combined one
    assert ortho.start != out.start and ortho.tables["ic_combination"]["mean_weight"].tolist() == pytest.approx([mean_up, 1.0 - mean_up])
    assert ortho.metrics["expected_ic_ir"] == pytest.approx(float(expected.loc[ortho.start:].dropna().mean()))
    assert ortho.forecasts.mean.equals(real.mean), "the orthogonalised forecast is the combined one"


def test_the_decay_table_is_one_row_a_model_and_the_spread_scale_is_described_over_the_window(monkeypatch, bundle, config):
    dates = bundle.index[[100, 200]]
    decay = pd.DataFrame({"ic0": [0.1, 0.3, 0.5, 0.7], "tau": [5.0, 15.0, 20.0, 40.0]},
                         index=pd.MultiIndex.from_product([dates, ["test_rules_up", "test_rules_down"]], names=["date", "model"]))
    weights = pd.DataFrame({"test_rules_up": 0.5, "test_rules_down": 0.5}, index=bundle.index)
    scale = pd.Series(np.linspace(0.5, 2.0, len(bundle.index)), index=bundle.index)
    monkeypatch.setattr(module, "decay_trust_weights", Spy((weights, decay)))
    monkeypatch.setattr(module, "regime_spread_scale", lambda combined, returns, regimes, min_obs, shrink: (combined, scale))
    out = Pipeline(spec(combination={"rule": "decay_weighted"}, forecast={"regime_spread": {}}), config, bundle).run(validate=False)
    table = out.tables["alpha_decay"]
    assert table.index.tolist() == ["test_rules_down", "test_rules_up"] and table.columns.tolist() == ["ic0_mean", "tau_median"]
    assert table.loc["test_rules_up"].tolist() == pytest.approx([0.3, 12.5]) and table.loc["test_rules_down"].tolist() == pytest.approx([0.5, 27.5]), "the mean first-day information coefficient and the median decay time of each model"
    spread = out.tables["regime_spread_scale"]
    assert spread.columns.tolist() == ["regime_spread_scale"] and spread.index.tolist() == scale.loc[out.start:].describe().index.tolist()
    assert spread["regime_spread_scale"].tolist() == pytest.approx(scale.loc[out.start:].describe().tolist())
    assert "regime_spread_scale" not in Pipeline(spec(combination={"rule": "decay_weighted"}), config, bundle).run(validate=False).tables
