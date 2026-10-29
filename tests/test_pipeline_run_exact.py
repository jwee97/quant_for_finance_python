"""What a pipeline run decides, checked on a small market where every run takes a fraction of a second.

``Pipeline.run`` chains the models, the combination, the allocator, the risk policy, the engine and the evaluation. The existing end-to-end tests run it on the full synthetic market for the sake of its
tables; these run it on six assets and a trivial model, to state each decision it takes: which risk mode a specification gets and what it refuses, what a limit adds and when it reaches the book, what a
size does to the net return, which tables appear and when, and that a forecast computed beforehand gives the same run as one computed inside it.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from src.framework import ForecastModel, Pipeline, bundle_from_prices
from src.framework.adaptive import RegimeRiskLimits
from src.framework.allocation import Context
from src.framework.registry import DETECTORS, MODELS, register_detector, register_model
from src.framework.risk import RegimeRiskPolicy
from src.framework.types import ForecastPanel, RegimeSeries
from src.utils.config import load_config
from src.utils.dates import rebalance_dates

DAYS = 1300


class RunTrend(ForecastModel):
    name = "test_run_trend"

    def __init__(self, lookback: int = 40):
        self.lookback = lookback

    def score(self, data):
        return data.prices.pct_change(self.lookback)

    def explain(self, data):
        return pd.DataFrame({"importance": [0.7, 0.3], "share": [0.7, 0.3]}, index=["momentum", "noise"])


class RunReversal(ForecastModel):
    name = "test_run_reversal"

    def score(self, data):
        return -data.prices.pct_change(5)


class RunBlank(ForecastModel):
    name = "test_run_blank"

    def score(self, data):
        return data.prices.pct_change(10)

    def explain(self, data):
        return pd.DataFrame()


def _prices(steps: np.ndarray) -> pd.DataFrame:
    index = pd.bdate_range("2017-01-02", periods=DAYS)
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=list("ABCDEF"))


@pytest.fixture(scope="module")
def bundle():
    """No edge: independent daily returns, the six assets' volatilities running from 0.6% to 1.6% a day."""
    steps = np.random.default_rng(1).normal(0.0003, 0.01, (DAYS, 6)) * np.linspace(0.6, 1.6, 6)
    return bundle_from_prices(_prices(steps), name="mini", min_history=40)


@pytest.fixture(scope="module")
def trending():
    """An edge: the assets drift apart, from -0.06% to +0.10% a day, so the recent winners keep winning."""
    steps = np.random.default_rng(5).normal(0.0, 0.01, (DAYS, 6)) * np.linspace(0.6, 1.6, 6) + np.linspace(-0.0006, 0.0010, 6)
    return bundle_from_prices(_prices(steps), name="trending", min_history=40)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(autouse=True)
def models():
    register_model("test_run_trend")(RunTrend)
    register_model("test_run_reversal")(RunReversal)
    register_model("test_run_blank")(RunBlank)
    yield
    MODELS.unregister("test_run_trend")
    MODELS.unregister("test_run_reversal")
    MODELS.unregister("test_run_blank")


@contextlib.contextmanager
def fixed_regime(probabilities: pd.DataFrame):
    """A detector that reports the probabilities it was given."""
    @register_detector("test_fixed_regime")
    class Fixed:
        def detect(self, data):
            return RegimeSeries(probabilities.reindex(data.index), "fixed")

    try:
        yield "test_fixed_regime"
    finally:
        DETECTORS.unregister("test_fixed_regime")


def spec(**parts) -> dict:
    base = {"name": "t", "models": [{"name": "test_run_trend", "params": {"lookback": 40}}],
            "evaluation": {"benchmarks": [], "n_trials": 1, "causality": False, "brinson": False, "model_attribution": False}}
    for key, value in parts.items():
        base[key] = {**base[key], **value} if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return base


def run(data, config, **parts):
    return Pipeline(spec(**parts), config, data).run(validate=False)


def gross(result) -> pd.Series:
    return result.weights.abs().sum(axis=1)


def targets_of(pipeline: Pipeline, regimes=None) -> pd.DataFrame:
    """The weights the allocator hands the risk step: the first lines of ``run``."""
    models = pipeline._models()
    combined, _ = pipeline._combine(pipeline._forecasts(models), models, None, regimes)
    combined, _ = pipeline._adjust_forecast(combined, regimes)
    _, allocator = pipeline._allocator(models)
    return allocator.build(Context(pipeline.bundle, pipeline.config, forecasts=combined, regimes=regimes, models=models))


# ------------------------------------------------------------------------------------------------------------------ the risk mode
def test_a_forecast_book_is_scaled_to_a_volatility_target_unless_told_not_to_and_the_target_is_the_one_asked_for(bundle, config):
    default = run(bundle, config)
    constant = run(bundle, config, risk={"mode": "constant"})
    none = run(bundle, config, risk={"mode": "none"})
    assert default.net_returns.equals(constant.net_returns) and not default.net_returns.equals(none.net_returns), "a forecast book targets a volatility unless it is told not to"
    assert default.risk_scalar is None, "the engine scales it, so there is no scalar to keep"
    assert gross(none).max() == pytest.approx(1.0), "unscaled, the stack holds a gross of one"
    assert default.metrics["ann_vol"] == pytest.approx(0.10, abs=0.005) and none.metrics["ann_vol"] < 0.08, "the configured 10%, against what the book is without it"
    quiet = run(bundle, config, risk={"mode": "constant", "target_vol": 0.03})
    assert quiet.metrics["ann_vol"] == pytest.approx(0.03, abs=0.002), "the target asked for"
    ceiling = run(bundle, config, risk={"mode": "constant", "target_vol": 0.20})
    assert gross(ceiling).max() == pytest.approx(1.5) and ceiling.metrics["ann_vol"] < 0.13, "a target the leverage cap cannot reach is not reached"
    book = {"models": [], "allocation": {"allocator": "static", "params": {"book": "equal_weight"}}}
    static = run(bundle, config, **book)
    assert static.net_returns.equals(run(bundle, config, risk={"mode": "none"}, **book).net_returns), "a static book is held as built"
    scaled = run(bundle, config, risk={"mode": "constant"}, **book)
    assert not static.net_returns.equals(scaled.net_returns) and static.metrics["ann_vol"] < 0.08 and scaled.metrics["ann_vol"] == pytest.approx(0.10, abs=0.005)


def test_a_risk_specification_with_a_word_it_does_not_know_or_a_mode_that_does_not_exist_is_refused(bundle, config):
    with pytest.raises(ValueError, match=r"unknown risk options \['leverage'\]"):
        run(bundle, config, risk={"leverage": 2})
    with pytest.raises(ValueError, match="risk.mode must be none, constant or regime"):
        run(bundle, config, risk={"mode": "kelly"})
    with pytest.raises(ValueError, match=r"unknown risk limits \['stop'\]; allowed: gross_caps, drawdown"):
        run(bundle, config, risk={"mode": "constant", "limits": {"stop": 1}})
    none = run(bundle, config, risk={"mode": "none"})
    assert run(bundle, config, risk={"mode": "none", "lookback": 21, "max_leverage": 1.0, "default_target": 0.5}).net_returns.equals(none.net_returns), "the words it does know, unused here"
    regime = run(bundle, config, risk={"mode": "regime"})
    assert regime.net_returns.equals(run(bundle, config).net_returns) and regime.risk_scalar is not None, "with no detector the regime mode is the default target, scaled before the engine instead of in it"


def test_limits_are_applied_after_an_explicit_target_and_the_table_says_what_they_did(bundle, config):
    default = run(bundle, config)
    free = run(bundle, config, risk={"mode": "constant", "target_vol": 0.10, "limits": {"gross_caps": {}}})
    assert free.net_returns.equals(default.net_returns), "the volatility target applied here, so the limits can act on the final weights, is the one the engine applies"
    assert free.risk_scalar.between(0.0, 1.5).all() and free.risk_scalar.max() > 1.0, "kept, as the engine keeps none"
    assert list(free.tables["risk_limits"].index) == ["gross cap binding", "drawdown limit on", "held book above the gross cap"]
    assert list(free.tables["risk_limits"].columns) == ["share_of_days", "mean_exposure_factor"]
    assert free.tables["risk_limits"]["share_of_days"].eq(0.0).all() and free.tables["risk_limits"]["mean_exposure_factor"].eq(1.0).all() and free.metrics["risk_limit_days_share"] == 0.0
    assert "risk_limits" not in default.tables and "risk_limit_days_share" not in default.metrics
    capped = run(bundle, config, risk={"mode": "constant", "target_vol": 0.10, "limits": {"drawdown": {"trigger": 0.02, "derisk": 0.5, "recover": 0.01}}})
    table = capped.tables["risk_limits"]
    on = table.loc["drawdown limit on", "share_of_days"]
    assert 0.5 < on < 1.0, "a book that has no edge is under its peak by two per cent most days"
    assert table.loc["drawdown limit on", "mean_exposure_factor"] == pytest.approx(1.0 - 0.5 * on), "half the exposure on those days, all of it on the rest"
    assert table.loc["gross cap binding"].tolist() == [0.0, 1.0] and table.loc["held book above the gross cap"].tolist() == [0.0, 1.0], "no regime caps were given"
    assert capped.metrics["risk_limit_days_share"] == pytest.approx(on)
    assert gross(capped).mean() < gross(default).mean() and not capped.net_returns.equals(default.net_returns)


def test_limits_applied_with_no_target_in_the_specification_or_the_configuration_target_ten_per_cent(bundle, config):
    class WithoutTarget:
        def __getattr__(self, name):
            return getattr(config, name)

        def get(self, key, default=None):
            return default if key == "portfolio.volatility.target_vol" else config.get(key, default)

    limits = {"mode": "constant", "limits": {"gross_caps": {}}}
    assert run(bundle, WithoutTarget(), risk=limits).net_returns.equals(run(bundle, config, risk={**limits, "target_vol": 0.10}).net_returns)


def test_a_regime_cap_reaches_the_held_book_at_the_next_rebalance_unless_the_book_is_rebalanced_daily(bundle, config):
    marks = rebalance_dates(bundle.index, "monthly")
    first, next_mark = marks[40], marks[41]
    crisis_start = bundle.index[bundle.index.get_loc(first) + 1]                      # the day after a month end: the cap cannot be met until the next one
    probabilities = pd.DataFrame({"Calm": 1.0, "Crisis": 0.0}, index=bundle.index)
    probabilities.loc[crisis_start:, :] = [0.0, 1.0]
    risk = {"mode": "constant", "target_vol": 0.10, "limits": {"gross_caps": {"Crisis": 0.5}}}
    with fixed_regime(probabilities) as detector:
        monthly = run(bundle, config, regime={"detector": detector}, risk=risk)
        daily = run(bundle, config, regime={"detector": detector}, risk=risk, execution={"rebalance": "daily"})
    window = bundle.index[bundle.index >= monthly.start]
    in_crisis = float((window >= crisis_start).mean())
    for result in (monthly, daily):
        assert result.start == daily.start and result.tables["risk_limits"].loc["gross cap binding", "share_of_days"] == pytest.approx(in_crisis), "the cap binds the targets on every crisis day"
    late = bundle.index.get_loc(next_mark) - bundle.index.get_loc(crisis_start) - 1             # the days between the first capped decision and the rebalance that carries it out
    assert late > 15
    row = monthly.tables["risk_limits"].loc["held book above the gross cap"]
    assert row["share_of_days"] == pytest.approx(late / len(window)) and row["mean_exposure_factor"] < 1.0, "the held book is above the cap until the next rebalance, and says so"
    held = gross(monthly)
    assert held.loc[crisis_start:next_mark - pd.Timedelta(days=1)].min() > 0.5 + 1e-6 and held.loc[next_mark] == pytest.approx(0.5) and held.loc[next_mark:].max() == pytest.approx(0.5)
    assert held.loc[:crisis_start].max() > 1.0, "before the crisis the book was levered by the volatility target"
    assert daily.tables["risk_limits"].loc["held book above the gross cap"].tolist() == [0.0, 1.0], "rebalanced daily, the cap acts at once"
    assert gross(daily).loc[crisis_start + pd.Timedelta(days=1):].max() == pytest.approx(0.5) and daily.metrics["ann_turnover"] > monthly.metrics["ann_turnover"]


def test_the_regime_mode_takes_its_target_and_its_window_and_leverage_from_the_specification(trending, config):
    probabilities = pd.DataFrame({"LowVol": 0.5, "HighVol": 0.3, "Crisis": 0.2}, index=trending.index)
    probabilities.iloc[:100] = np.nan
    asked = {"regime_targets": {"LowVol": 0.10, "HighVol": 0.05}, "default_target": 0.2, "lookback": 30, "max_leverage": 1.2}
    with fixed_regime(probabilities) as detector:
        out = run(trending, config, regime={"detector": detector}, risk={"mode": "regime", **asked})
        plain = run(trending, config, regime={"detector": detector}, risk={"mode": "regime"})
        regimes = DETECTORS.create(detector).detect(trending)
    targets = targets_of(Pipeline(spec(regime={"detector": "test_fixed_regime"}), config, trending), regimes)
    _, scalar, target = RegimeRiskPolicy(targets=asked["regime_targets"], default_target=0.2, lookback=30, max_leverage=1.2).apply(targets, trending.returns, regimes)
    assert out.risk_scalar.equals(scalar) and out.risk_scalar.max() == pytest.approx(1.2), "the policy the specification describes, and no other"
    assert target.iloc[0] == 0.2 and target.iloc[-1] == pytest.approx(0.5 * 0.10 + 0.3 * 0.05 + 0.2 * 0.2), "the default until there is a view; a regime without a target takes the default"
    assert plain.risk_scalar.equals(RegimeRiskPolicy().apply(targets, trending.returns, regimes)[1]), "and the policy's own defaults when it describes nothing"
    assert not plain.risk_scalar.equals(out.risk_scalar) and out.regimes is not None and "by_regime" in out.tables
    assert out.tables["by_regime"].loc["LowVol", "share_of_days"] == 1.0, "the most probable regime is the label of the day"


# ------------------------------------------------------------------------------------------------------------------ execution at a size
def test_an_assets_under_management_charges_market_impact_and_reports_where_it_goes(bundle, config):
    plain = run(bundle, config)
    sized = run(bundle, config, execution={"aum": 1e11})
    assert sized.costs.sum() > 5 * plain.costs.sum() and sized.net_returns.sum() < plain.net_returns.sum(), "impact is a cost on top of the linear one"
    assert sized.gross_returns.equals(plain.gross_returns) and sized.weights.equals(plain.weights), "what the book holds and earns before costs does not depend on its size"
    pd.testing.assert_series_equal(sized.net_returns, (sized.gross_returns - sized.costs).dropna(), check_names=False)
    table = sized.tables["cost_breakdown"]
    assert list(table.index) == ["spread and commission (linear)", "market impact (square root)", "total"] and list(table.columns) == ["annual_bps", "mean_participation"]
    assert table.loc["total", "annual_bps"] == pytest.approx(table.loc["spread and commission (linear)", "annual_bps"] + table.loc["market impact (square root)", "annual_bps"])
    assert table.loc["market impact (square root)", "annual_bps"] > table.loc["spread and commission (linear)", "annual_bps"] > 0.0 and 0.0 < table.loc["total", "mean_participation"] < 1.0
    assert "cost_breakdown" not in plain.tables and "capacity" not in sized.tables, "a capacity curve was not asked for"
    harder = run(bundle, config, execution={"aum": 1e11, "impact_coefficient": 3.0})
    assert harder.tables["cost_breakdown"].loc["market impact (square root)", "annual_bps"] == pytest.approx(3.0 * table.loc["market impact (square root)", "annual_bps"])
    assert harder.tables["cost_breakdown"].loc["spread and commission (linear)", "annual_bps"] == pytest.approx(table.loc["spread and commission (linear)", "annual_bps"])
    assert harder.net_returns.sum() < sized.net_returns.sum()


def test_a_capacity_request_makes_a_curve_and_either_a_figure_or_the_reason_there_is_none(bundle, trending, config):
    grid = [1e7, 1e9, 1e11, 1e13]
    edge = run(trending, config, execution={"capacity": grid})
    curve = edge.tables["capacity"]
    assert list(curve.columns) == ["net_sharpe"] and curve.index.tolist() == grid and curve["net_sharpe"].is_monotonic_decreasing
    assert curve["net_sharpe"].iloc[0] == pytest.approx(edge.metrics["sharpe"], abs=0.02), "at a size that costs nothing the curve starts where the run does"
    assert 1e11 < edge.metrics["capacity_usd"] < 1e13 and edge.metrics["capacity_note"] == "", "half the Sharpe is gone between the third and the fourth size"
    assert curve["net_sharpe"].iloc[2] > 0.5 * curve["net_sharpe"].iloc[0] > curve["net_sharpe"].iloc[3]
    default = run(trending, config, execution={"capacity": True})
    assert default.tables["capacity"].index.tolist() == [1e7, 1e8, 1e9, 1e10, 1e11], "the platform's grid when none is given"
    assert np.isnan(default.metrics["capacity_usd"]) and default.metrics["capacity_note"] == "above grid"
    no_edge = run(bundle, config, execution={"capacity": grid})
    assert np.isnan(no_edge.metrics["capacity_usd"]) and no_edge.metrics["capacity_note"] == "n/a", "a book that loses before impact has no capacity to find"
    assert "cost_breakdown" not in edge.tables, "a size to cost was not asked for"


# ------------------------------------------------------------------------------------------------------------------ what a run reports
def test_a_run_starts_when_it_has_a_view_reports_what_each_model_explains_and_times_each_step(trending, config):
    result = Pipeline(spec(evaluation={"explain": True}), config, trending).run(validate=False)
    assert result.start >= result.forecasts.first_valid() and result.start >= result.net_returns.index[0] and result.metrics["start"] == str(result.start.date())
    assert result.window.index[0] >= result.start and result.metrics["n_days"] == len(result.window.dropna())
    assert result.tables["explain_test_run_trend"].index.tolist() == ["momentum", "noise"]
    assert "explain_test_run_trend" not in run(trending, config).tables
    three = [{"name": "test_run_trend"}, {"name": "test_run_reversal"}, {"name": "test_run_blank"}]
    mixed = Pipeline(spec(models=three, evaluation={"explain": True}), config, trending).run(validate=False)
    assert [k for k in mixed.tables if k.startswith("explain_")] == ["explain_test_run_trend"], "a model that explains nothing, or an empty table, leaves no table"
    assert list(result.timings) == ["regime", "forecast", "allocation", "backtest", "total"] and list(result.timings.values()) == sorted(result.timings.values())
    assert 0.0 < result.timings["total"] < 120.0, "seconds since the run began"
    late = Pipeline(spec(evaluation={"start": "2021-06-01"}), config, trending).run(validate=False)
    assert late.start == pd.Timestamp("2021-06-01") and late.metrics["start"] == "2021-06-01"
    assert late.metrics["n_days"] == int((late.net_returns.index >= "2021-06-01").sum()) < result.metrics["n_days"]


def test_a_run_checks_that_its_models_use_no_future_data_unless_told_not_to(trending, config):
    asked = {"name": "t", "models": [{"name": "test_run_trend"}], "evaluation": {"benchmarks": [], "n_trials": 1, "brinson": False, "model_attribution": False}}
    checked = Pipeline(asked, config, trending).run()
    assert list(checked.validation["causality"]) == ["test_run_trend"] and checked.validation["causality"]["test_run_trend"]["ok"], "validated by default"
    assert "causality" not in Pipeline(asked, config, trending).run(validate=False).validation and "causality" not in Pipeline({**asked, "evaluation": {**asked["evaluation"], "causality": False}}, config, trending).run().validation


def test_a_forecast_computed_beforehand_is_the_forecast_the_run_uses(trending, config):
    pipeline = Pipeline(spec(), config, trending)
    panels = pipeline._forecasts(pipeline._models())
    assert list(panels) == ["test_run_trend"] and panels["test_run_trend"].horizon == 21
    inside = Pipeline(spec(), config, trending).run(validate=False)
    again = Pipeline(spec(), config, trending).run(validate=False, precomputed=panels)
    assert again.net_returns.equals(inside.net_returns) and again.weights.equals(inside.weights)
    panel = panels["test_run_trend"]
    flipped = Pipeline(spec(), config, trending).run(validate=False, precomputed={"test_run_trend": ForecastPanel(-panel.mean, panel.std, panel.confidence, panel.horizon, panel.name)})
    assert np.allclose(flipped.weights.fillna(0.0), -inside.weights.fillna(0.0), atol=1e-12) and flipped.metrics["sharpe"] < 0.0 < inside.metrics["sharpe"], "the supplied forecast, not a recomputed one"


def test_two_models_are_combined_by_the_rule_and_each_gets_a_share_of_the_return_unless_the_attribution_is_switched_off(trending, config):
    models = [{"name": "test_run_trend", "params": {"lookback": 40}}, {"name": "test_run_reversal"}]
    both = Pipeline(spec(models=models, combination={"rule": "equal"}, evaluation={"model_attribution": True}), config, trending).run(validate=False)
    table = both.tables["attribution_by_model"]
    assert table.index.tolist() == ["model: test_run_trend", "model: test_run_reversal", "interaction (stack, risk scaling, limits)", "costs", "NET RETURN (sum of daily net returns)"]
    assert table["return_points"].iloc[:-1].sum() == pytest.approx(table["return_points"].iloc[-1]) and table.loc["costs", "return_points"] < 0.0, "the rows add up to the net return"
    window = both.net_returns.loc[both.start:].dropna()
    assert table.loc["NET RETURN (sum of daily net returns)", "return_points"] == pytest.approx(100.0 * window.sum())
    window = window.to_frame()
    alone = [Pipeline(spec(models=[m]), config, trending).run(validate=False) for m in models]
    for row, model in zip(table.index[:2], alone):
        assert table.loc[row, "return_points"] == pytest.approx(50.0 * model.gross_returns.reindex(window.index).sum()), "half of the model's own gross return: gross, before the costs that are a row of their own"
    single = [model.forecasts.mean for model in alone]
    valid = single[0].notna() & single[1].notna()
    assert valid.to_numpy().sum() > 1000 and (both.forecasts.mean.where(valid) - ((single[0] + single[1]) / 2).where(valid)).abs().max().max() < 1e-12, "the equal rule is the plain average"
    assert "attribution_by_model" not in Pipeline(spec(models=models), config, trending).run(validate=False).tables, "switched off"
    assert "attribution_by_model" not in Pipeline(spec(evaluation={"model_attribution": True}), config, trending).run(validate=False).tables, "one model: nothing to split"
    trust = Pipeline(spec(models=models, combination={"rule": "ic_weighted"}, evaluation={"model_attribution": True}), config, trending).run(validate=False)
    weights = trust.tables["trust_weights"]
    assert weights.columns.tolist() == ["test_run_trend", "test_run_reversal"] and weights.sum(axis=1).sub(1.0).abs().max() < 1e-9 and weights.index.is_month_end.all()
    assert "attribution_by_model" in trust.tables and "trust_weights" not in both.tables


def test_the_benchmark_attribution_is_a_table_for_a_window_of_a_year_or_more_and_nothing_for_less(trending, config):
    on = Pipeline(spec(evaluation={"brinson": True}), config, trending).run(validate=False)
    table = on.tables["attribution_vs_benchmark"]
    assert table.index[-1] == "TOTAL (cumulative active return, points)" and list(table.columns) == ["allocation", "selection", "interaction", "total"]
    assert table.loc["TOTAL (cumulative active return, points)", "total"] == pytest.approx(table["total"].iloc[:-1].sum())
    assert "attribution_vs_benchmark" not in run(trending, config).tables, "switched off"
    short = Pipeline(spec(evaluation={"brinson": True, "start": "2021-06-01"}), config, trending).run(validate=False)
    assert "attribution_vs_benchmark" not in short.tables, "seven months of months say nothing"
    unasked = {"name": "t", "models": [{"name": "test_run_trend"}, {"name": "test_run_reversal"}], "evaluation": {"benchmarks": [], "causality": False}}
    assert {"attribution_vs_benchmark", "attribution_by_model"} <= set(Pipeline(unasked, config, trending).run(validate=False).tables), "both attributions are on unless a specification switches them off"


# ------------------------------------------------------------------------------------------------------------------ the regime limits themselves
def test_the_gross_cap_is_the_probability_weighted_cap_of_the_regimes():
    index = pd.bdate_range("2021-01-04", periods=7)
    weights = pd.DataFrame({"A": 1.0}, index=index)
    probs = RegimeSeries(pd.DataFrame({"Calm": [0.5, 1.0, 0.0, 0.2, np.nan, 0.0, 0.0], "Stress": [0.5, 0.0, 1.0, 0.2, np.nan, 0.0, 0.0]}, index=index))
    limits = RegimeRiskLimits(gross_caps={"Stress": 0.4, "Other": 9.0})
    cap = limits.cap_series(probs, index)
    assert cap.iloc[0] == pytest.approx(0.5 * 0.4 + 0.5 * 1e6) and cap.iloc[1] == pytest.approx(1e6) and cap.iloc[2] == pytest.approx(0.4) and cap.iloc[3] == pytest.approx(0.2 * 0.4 + 0.8 * 1e6)
    assert cap.iloc[4] == pytest.approx(1e6) and cap.iloc[5] == pytest.approx(1e6), "no view is no cap"
    assert RegimeRiskLimits().cap_series(probs, index).eq(np.inf).all() and RegimeRiskLimits(gross_caps={"Stress": 0.4}).cap_series(None, index).eq(np.inf).all()
    scaled, diag = limits.apply(weights, pd.DataFrame({"A": 0.0}, index=index), probs)
    assert scaled["A"].iloc[2] == pytest.approx(0.4) and scaled["A"].iloc[1] == 1.0 and scaled["A"].iloc[0] == 1.0, "a cap below the gross scales the book down to it, never up"
    assert diag["cap_factor"].iloc[2] == pytest.approx(0.4) and diag["cap_factor"].iloc[1] == 1.0 and diag["drawdown_limit_on"].eq(0.0).all()
    two = pd.DataFrame({"A": 1.0, "B": -1.0}, index=index)
    assert limits.apply(two, pd.DataFrame(0.0, index=index, columns=["A", "B"]), probs)[0].iloc[2].tolist() == pytest.approx([0.2, -0.2]), "gross is the sum of the absolute weights"
    nothing = limits.apply(pd.DataFrame({"A": 0.0}, index=index), pd.DataFrame({"A": 0.0}, index=index), probs)
    assert nothing[1]["cap_factor"].eq(1.0).all() and nothing[0]["A"].eq(0.0).all(), "nothing held, nothing to cap"


def test_the_drawdown_rule_cuts_exposure_until_the_limited_book_has_recovered():
    index = pd.bdate_range("2021-01-04", periods=7)
    weights = pd.DataFrame({"A": 1.0}, index=index)
    returns = pd.DataFrame({"A": [0.0, -0.06, -0.06, 0.0, 0.05, 0.05, 0.05]}, index=index)
    drawn, state = RegimeRiskLimits(drawdown={"trigger": 0.10, "derisk": 0.5, "recover": 0.05}).apply(weights, returns)
    assert drawn["A"].tolist() == [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0], "on at a drawdown of 11.6% (not at 6%), off again at 4.8% (the cut book earns half of the 5%)"
    assert state["drawdown_limit_on"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0] and state["drawdown_factor"].tolist() == drawn["A"].tolist() and state["cap_factor"].eq(1.0).all()
    assert RegimeRiskLimits(drawdown={"trigger": 0.10}).apply(weights, returns)[0]["A"].tolist() == drawn["A"].tolist(), "a cut of a half and a recovery at half the trigger, by default"
    steady, exact = pd.DataFrame({"A": 1.0}, index=index[:3]), pd.DataFrame({"A": [0.0, -0.25, 0.25]}, index=index[:3])               # binary fractions: a drawdown of exactly 1/4, then exactly 5/32
    assert RegimeRiskLimits(drawdown={"trigger": 0.25, "derisk": 0.5, "recover": 0.15625}).apply(steady, exact)[0]["A"].tolist() == [1.0, 0.5, 1.0], "on at a drawdown equal to the trigger, off at one equal to the recovery"
    for bad in ({"trigger": 0.1, "recover": 0.1}, {"trigger": 0.1, "recover": 0.2}, {"trigger": 1.0}, {"trigger": 0.1, "derisk": 1.0}, {"trigger": 0.1, "derisk": -0.1}, {"trigger": 0.1, "recover": 0.0}):
        with pytest.raises(ValueError, match="drawdown limits need 0 < recover < trigger < 1 and 0 <= derisk < 1"):
            RegimeRiskLimits(drawdown=bad).apply(weights, returns)
    final, stuck = RegimeRiskLimits(drawdown={"trigger": 0.10, "derisk": 0.0}).apply(weights, returns)
    assert final["A"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] and stuck["drawdown_limit_on"].tolist()[2:] == [1.0] * 5, "a book that holds nothing cannot earn its way back: a cut to nothing is final"
