"""The pieces of the research pipeline that decide what it runs, checked one at a time.

``framework/pipeline.py`` turns a specification into a run: which allocator, which rebalance frequency and lag and cost, which date the evaluation starts from, and how forecasts are combined. A whole run takes
seconds to minutes, which is why a planted-bug audit could not be pointed at it; the decisions themselves are small functions of a specification, and each is stated here with the choice it makes and why.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import LinearCostModel
from src.backtest.engine import BacktestEngine
from src.framework.allocation import ALLOCATORS, Sleeves
from src.framework.pipeline import Pipeline, PipelineSpec, _engine, _position_mode, allocator_choice
from src.framework.registry import MODELS, register_model
from src.framework.types import ForecastPanel


class Config:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


# ------------------------------------------------------------------------------------------------------------------ the specification
def test_a_specification_has_its_defaults_refuses_a_key_it_does_not_know_and_round_trips():
    spec = PipelineSpec()
    assert (spec.name, spec.models, spec.combination, spec.regime, spec.forecast, spec.allocation, spec.risk, spec.execution, spec.evaluation, spec.notes) == (
        "strategy", [], {"rule": "equal"}, None, {}, {}, {}, {}, {}, "")
    other = PipelineSpec()
    spec.models.append("momentum")
    spec.combination["rule"] = "confidence"
    assert other.models == [] and other.combination == {"rule": "equal"}, "two specifications never share a list or a dictionary"
    payload = {"name": "trend", "models": [{"name": "momentum", "params": {"lookback": 60}}], "combination": {"rule": "precision"}, "regime": {"detector": "hmm"}, "forecast": {"confidence": {"method": "platt"}},
               "allocation": {"allocator": "sleeves"}, "risk": {"mode": "constant"}, "execution": {"rebalance": "weekly"}, "evaluation": {"start": "2015-01-01"}, "notes": "a note"}
    built = PipelineSpec.from_dict(payload)
    assert built.to_dict() == payload and list(built.to_dict()) == ["name", "models", "combination", "regime", "forecast", "allocation", "risk", "execution", "evaluation", "notes"]
    with pytest.raises(ValueError, match=r"unknown spec keys \['alocation', 'risk_mode'\]; allowed: \['allocation', 'combination', 'evaluation', 'execution', 'forecast', 'models', 'name', 'notes', 'regime', 'risk'\]"):
        PipelineSpec.from_dict({"name": "x", "alocation": {}, "risk_mode": 1})


def test_a_specification_can_be_read_from_a_file(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text("name: from_file\nmodels:\n  - name: momentum\n    params: {lookback: 21}\nexecution:\n  rebalance: daily\n  signal_lag: 2\n")
    spec = PipelineSpec.from_yaml(path)
    assert spec.name == "from_file" and spec.models == [{"name": "momentum", "params": {"lookback": 21}}] and spec.execution == {"rebalance": "daily", "signal_lag": 2} and spec.combination == {"rule": "equal"}
    assert PipelineSpec.from_yaml(str(path)).name == "from_file", "a path or its text"


# ------------------------------------------------------------------------------------------------------------------ which allocator, and in which mode
def test_the_allocator_is_the_one_named_else_the_models_own_weights_else_the_forecast_stack_else_a_static_book():
    stack = SimpleNamespace(position_mode="time_series")
    structured = SimpleNamespace(structured=True)
    plain = SimpleNamespace()
    assert allocator_choice(None, []) == ("static", {"book": "risk_parity"}) and allocator_choice({}, []) == ("static", {"book": "risk_parity"}), "nothing to go on: risk parity"
    assert allocator_choice({"params": {"book": "hrp"}}, []) == ("static", {"book": "hrp"}) and allocator_choice({"allocator": "static"}, []) == ("static", {"book": "risk_parity"})
    assert allocator_choice({}, [plain]) == ("forecast_stack", {"mode": "cross_sectional"}) and allocator_choice(None, [stack]) == ("forecast_stack", {"mode": "time_series"})
    assert allocator_choice({}, [structured]) == ("model_weights", {}), "one structured model brings its own weights"
    assert allocator_choice({}, [structured, plain])[0] == "forecast_stack" and allocator_choice({}, [SimpleNamespace(structured=False)])[0] == "forecast_stack", "two models, or one that is not structured, are forecasts"
    assert allocator_choice({"allocator": "sleeves", "params": {"gross": 2.0}}, [plain]) == ("sleeves", {"gross": 2.0}), "a named allocator wins, with its own parameters"
    assert allocator_choice({"allocator": "score_stack"}, [stack]) == ("score_stack", {"mode": "time_series"}) and allocator_choice({"allocator": "confidence"}, [stack]) == ("confidence", {}), "only the two stacks take a mode"
    assert allocator_choice({"allocator": "forecast_stack", "params": {"mode": "cross_sectional"}}, [stack]) == ("forecast_stack", {"mode": "cross_sectional"}), "a mode given is not overruled"
    given = {"allocator": "static", "params": {"book": "hrp"}}
    assert allocator_choice(given, []) == ("static", {"book": "hrp"}) and given == {"allocator": "static", "params": {"book": "hrp"}}, "the specification is left as it was"


def test_models_that_disagree_about_their_position_mode_are_stacked_across_assets():
    ts, xs = SimpleNamespace(position_mode="time_series"), SimpleNamespace(position_mode="cross_sectional")
    assert _position_mode([ts]) == "time_series" and _position_mode([ts, ts]) == "time_series" and _position_mode([xs]) == "cross_sectional"
    assert _position_mode([ts, xs]) == "cross_sectional" and _position_mode([SimpleNamespace()]) == "cross_sectional" and _position_mode([]) == "cross_sectional"


# ------------------------------------------------------------------------------------------------------------------ the engine a specification asks for
def test_the_engine_is_read_from_the_configuration():
    engine = BacktestEngine.from_config(Config())
    assert (engine.signal_lag, engine.rebalance, engine.weight_drift, engine.annualisation, engine.min_assets, engine.target_vol, engine.vol_lookback, engine.max_leverage) == (1, "monthly", True, 252, 5, None, 63, 1.5)
    assert engine.cost_model.cost_bps == 10.0 and engine.cost_model.per_asset_bps == {}
    custom = BacktestEngine.from_config(Config(**{"backtest.costs.cost_bps": 4, "backtest.costs.per_asset_bps": {"SPY": 1}, "backtest.engine.signal_lag": 2, "backtest.engine.rebalance": "weekly", "backtest.engine.weight_drift": False,
                                                  "backtest.engine.annualisation": 260, "backtest.engine.min_assets": 3, "portfolio.volatility.vol_target_enabled": True, "portfolio.volatility.target_vol": 0.08,
                                                  "portfolio.volatility.lookback": 40, "portfolio.volatility.max_leverage_from_vol_target": 2.0}))
    assert (custom.signal_lag, custom.rebalance, custom.weight_drift, custom.annualisation, custom.min_assets, custom.target_vol, custom.vol_lookback, custom.max_leverage) == (2, "weekly", False, 260, 3, 0.08, 40, 2.0)
    assert custom.cost_model.cost_bps == 4.0 and custom.cost_model.per_asset_bps == {"SPY": 1}
    assert BacktestEngine.from_config(Config(**{"portfolio.volatility.target_vol": 0.2})).target_vol is None, "a target is used only when the configuration enables it"


def test_a_specification_overrides_the_engine_and_the_fastest_hint_of_a_model_is_used_unless_it_says_otherwise():
    config = Config()
    base = _engine(config, PipelineSpec())
    assert (base.rebalance, base.signal_lag, base.min_assets, base.cost_model.cost_bps) == ("monthly", 1, 5, 10.0), "nothing asked for: the platform's engine"
    slow, fast, middle = SimpleNamespace(rebalance="monthly"), SimpleNamespace(rebalance="daily"), SimpleNamespace(rebalance="weekly")
    assert _engine(config, PipelineSpec(), [slow, fast, middle]).rebalance == "daily" and _engine(config, PipelineSpec(), [slow, middle]).rebalance == "weekly", "the fastest rule sets the pace"
    assert _engine(config, PipelineSpec(), [slow]).rebalance == "monthly" and _engine(config, PipelineSpec(), [SimpleNamespace(rebalance="hourly"), SimpleNamespace()]).rebalance == "monthly", "a hint the engine does not know is ignored"
    assert _engine(config, PipelineSpec(execution={"rebalance": "weekly"}), [fast]).rebalance == "weekly", "the specification wins over a hint"
    assert _engine(config, PipelineSpec(execution={"rebalance": ""}), [fast]).rebalance == "daily", "an empty one is no choice"
    tuned = _engine(config, PipelineSpec(execution={"signal_lag": 0, "min_assets": 2, "cost_bps": 7}))
    assert tuned.signal_lag == 0 and tuned.min_assets == 2 and tuned.rebalance == "monthly", "a lag of zero is a choice"
    assert tuned.cost_model == LinearCostModel(cost_bps=7.0) and isinstance(tuned.cost_model.cost_bps, float), "one cost for the whole book, in place of the platform's per-asset costs"
    assert _engine(Config(**{"backtest.costs.per_asset_bps": {"SPY": 1}}), PipelineSpec(execution={"cost_bps": 3})).cost_model.per_asset_bps == {}
    assert _engine(Config(**{"backtest.costs.per_asset_bps": {"SPY": 1}}), PipelineSpec()).cost_model.per_asset_bps == {"SPY": 1}
    assert _engine(config, PipelineSpec(execution={"signal_lag": "3", "min_assets": "4"})).signal_lag == 3, "numbers written as text are numbers"


# ------------------------------------------------------------------------------------------------------------------ the evaluation start, the models and the combination
def pipeline(spec=None, config=None, bundle=None) -> Pipeline:
    p = object.__new__(Pipeline)
    p.spec, p.config, p.bundle = spec or PipelineSpec(), config or Config(), bundle
    p._decay_table = p._ic_table = p._rule = None
    return p


def test_the_evaluation_starts_when_the_returns_the_forecasts_and_the_book_all_exist_and_not_before_a_date_asked_for():
    index = pd.bdate_range("2021-01-04", periods=40)
    net = pd.Series(0.001, index=index[5:])
    mean = pd.DataFrame(np.nan, index=index, columns=["A", "B"])
    mean.iloc[12:, 0] = 0.01
    forecasts = ForecastPanel.from_mean_std(mean, mean * 0 + 0.02)
    targets = pd.DataFrame(np.nan, index=index, columns=["A", "B"])
    targets.iloc[20:] = 0.5
    p = pipeline()
    assert p._evaluation_start(forecasts, targets, net) == index[20], "the latest of the first return, the first forecast and the first day the allocator has a view"
    assert p._evaluation_start(None, targets, net) == index[20] and p._evaluation_start(forecasts, targets.assign(A=0.0, B=0.0), net) == index[12] and p._evaluation_start(None, targets.iloc[:0], net) == index[5]
    assert p._evaluation_start(forecasts, pd.DataFrame(np.nan, index=index, columns=["A", "B"]), net) == index[12], "a book that never has a view does not hold the start back"
    flat = targets.copy()
    flat.iloc[8:20] = 0.0
    assert p._evaluation_start(None, flat, net) == index[8], "a flat book earns zero: it is not missing"
    assert pipeline(PipelineSpec(evaluation={"start": "2021-02-15"}))._evaluation_start(forecasts, targets, net) == pd.Timestamp("2021-02-15"), "a start asked for after all of those is used"
    assert pipeline(PipelineSpec(evaluation={"start": "2021-01-01"}))._evaluation_start(forecasts, targets, net) == index[20], "and one before them is not"
    quiet = ForecastPanel.from_mean_std(mean.iloc[:, :0], mean.iloc[:, :0])
    assert p._evaluation_start(quiet, targets, net) == index[20]


class PipeModel:
    def __init__(self, scale: float = 1.0):
        self.scale, self.required = scale, []

    def require(self, bundle):
        self.required.append(bundle)


def test_models_are_made_from_their_names_with_their_parameters_and_each_checks_the_data_it_needs():
    register_model("test_pipe_model")(PipeModel)
    try:
        bundle = object()
        p = pipeline(PipelineSpec(models=["test_pipe_model", {"name": "test_pipe_model", "params": {"scale": 2.5}}, {"name": "test_pipe_model", "params": None}]), bundle=bundle)
        models = p._models()
        assert [m.scale for m in models] == [1.0, 2.5, 1.0] and all(m.required == [bundle] for m in models), "a name alone, a name with parameters, and a name with none"
        with pytest.raises(KeyError, match="unknown forecast model 'nope'"):
            pipeline(PipelineSpec(models=["nope"]))._models()
        assert pipeline()._models() == []
        name, allocator = pipeline(PipelineSpec(allocation={"allocator": "sleeves", "params": {"gross": 2.0}}))._allocator([])
        assert name == "sleeves" and isinstance(allocator, Sleeves) and allocator.gross == 2.0
        assert pipeline()._allocator([])[0] == "static"
    finally:
        MODELS.unregister("test_pipe_model")
    assert "sleeves" in ALLOCATORS


def forecast_panel(level: float, confidence: float = 0.5) -> ForecastPanel:
    index = pd.bdate_range("2021-01-04", periods=5)
    mean = pd.DataFrame(level, index=index, columns=["A", "B"])
    return ForecastPanel(mean, mean * 0 + 0.02, mean * 0 + confidence)


def test_a_single_forecast_is_used_as_it_is_and_the_rules_that_need_more_say_what_they_need():
    one = {"m": forecast_panel(0.01)}
    p = pipeline()
    combined, weights = p._combine(one, [object()], BacktestEngine())
    assert combined is one["m"] and weights is None and p._rule == "equal", "nothing to combine"
    two = {"m": forecast_panel(0.01), "n": forecast_panel(0.03)}
    equal, weights = p._combine(two, [object(), object()], BacktestEngine())
    assert weights is None and equal.mean.iloc[0].tolist() == pytest.approx([0.02, 0.02]), "the plain average"
    with pytest.raises(ValueError, match="unknown combination rule 'vote'"):
        pipeline(PipelineSpec(combination={"rule": "vote"}))._combine(two, [object(), object()], BacktestEngine())
    with pytest.raises(ValueError, match="the regime_conditional rule needs a regime detector in the spec"):
        pipeline(PipelineSpec(combination={"rule": "regime_conditional"}))._combine(two, [object(), object()], BacktestEngine(), None)
    named = pipeline(PipelineSpec(combination={"rule": "confidence"}))
    named._combine(two, [object(), object()], BacktestEngine())
    assert named._rule == "confidence", "remembered for the attribution of the combined forecast"


def test_forecast_adjustments_are_the_regime_spread_then_a_calibrated_confidence_and_nothing_else_is_accepted():
    panel = forecast_panel(0.01)
    assert pipeline()._adjust_forecast(panel, None) == (panel, {}) and pipeline(PipelineSpec(forecast={}))._adjust_forecast(panel, None)[0] is panel
    with pytest.raises(ValueError, match=r"unknown forecast options \['leverage'\]; allowed: regime_spread, confidence"):
        pipeline(PipelineSpec(forecast={"leverage": 2}))._adjust_forecast(panel, None)
    assert pipeline(PipelineSpec(forecast={"regime_spread": None, "confidence": None}))._adjust_forecast(panel, None) == (panel, {}), "an option left empty is not asked for"
