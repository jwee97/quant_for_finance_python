"""The research framework: types, calibration, combination, regimes, allocation, risk and the pipeline (Generation 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.execution import vol_target_scaling
from src.framework import (ALLOCATORS, DETECTORS, MODELS, Forecast, ForecastModel, ForecastPanel, Pipeline, RegimeSeries, bundle_from_prices,
                           combine_forecasts, register_allocator, register_model)
from src.framework.allocation import Allocator, Context
from src.framework.forecasting import calibration_slope
from src.framework.regimes import ramp
from src.framework.registry import Registry
from src.framework.risk import RegimeRiskPolicy, vol_target_series
from src.framework.types import confidence_from
from src.framework.validate import check_causality
from src.utils.config import load_config


@pytest.fixture(scope="module")
def bundle(synthetic_prices):
    macro = pd.DataFrame({"CPI_YOY": np.linspace(0.01, 0.06, len(synthetic_prices))}, index=synthetic_prices.index)
    return bundle_from_prices(synthetic_prices, macro=macro, name="synthetic", min_history=60)


@pytest.fixture(scope="module")
def config():
    return load_config()


# ------------------------------------------------------------------------------------------------ types
def test_confidence_is_zero_for_a_coin_flip_and_grows_with_the_signal_to_noise():
    assert confidence_from(0.0, 0.04) == pytest.approx(0.0)
    assert 0 < confidence_from(0.01, 0.04) < confidence_from(0.04, 0.04) < confidence_from(0.2, 0.04) <= 1.0
    assert confidence_from(0.02, 0.04) == confidence_from(-0.02, 0.04)
    assert np.isnan(confidence_from(0.02, 0.0))
    f = Forecast.from_mean_std(0.02, 0.04)
    assert f.probability_up == pytest.approx(0.6915, abs=1e-3) and f.confidence == pytest.approx(0.3829, abs=1e-3)


def test_regime_series_reports_the_most_probable_regime_and_rejects_bad_probabilities():
    idx = pd.date_range("2020-01-01", periods=3)
    probs = pd.DataFrame({"LowVol": [0.9, 0.2, np.nan], "HighVol": [0.1, 0.8, np.nan]}, index=idx)
    series = RegimeSeries(probs, "test", pd.DataFrame({"trend": ["Bull", "Bear", None]}, index=idx))
    series.validate()
    assert series.at(idx[0]).name == "LowVol" and series.at(idx[1]).label == "HighVolBear" and series.at(idx[2]).name == "Unknown"
    assert str(series.at(idx[1])) == "Regime(name='HighVol', probability=0.80)"
    with pytest.raises(ValueError):
        RegimeSeries(pd.DataFrame({"A": [0.5], "B": [0.2]}, index=idx[:1])).validate()


def test_registry_rejects_a_duplicate_name_and_lists_choices():
    reg = Registry("thing")
    reg.register("a")(lambda: 1)
    with pytest.raises(ValueError):
        reg.register("a")(lambda: 2)
    with pytest.raises(KeyError, match="available"):
        reg.create("b")


# --------------------------------------------------------------------------------------------- calibration
def _planted(n=2500, k=6, seed=3, slope=0.5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    score = pd.DataFrame(rng.normal(size=(n, k)), index=idx)
    shock = rng.normal(0, 0.01, (n, k))
    returns = pd.DataFrame(shock, index=idx)
    # the score at s predicts the NEXT day's return: slope * score / 21 per day for 21 days would be heavy; use the one-day version
    returns.iloc[1:] = shock[1:] + slope * 0.01 * score.to_numpy()[:-1] / 21
    return score, returns


def test_calibration_slope_recovers_a_planted_relationship_and_is_zero_for_noise():
    score, returns = _planted()
    slope = calibration_slope(score, returns, 21, 252).dropna()
    assert len(slope) > 500 and slope.iloc[-1] > 0
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(size=score.shape), index=score.index)
    flat = calibration_slope(noise, returns * 0 + rng.normal(0, 0.01, returns.shape), 21, 252).dropna()
    assert flat.abs().max() < 0.05


def test_calibration_uses_only_matured_labels():
    """Changing returns after date t must not change the slope on or before t."""
    score, returns = _planted(n=1500)
    base = calibration_slope(score, returns, 21, 252)
    cut = returns.index[1000]
    shocked = returns.copy()
    shocked.loc[shocked.index > cut] *= 10.0
    other = calibration_slope(score, shocked, 21, 252)
    assert np.allclose(base.loc[:cut].fillna(-1), other.loc[:cut].fillna(-1))
    assert not np.allclose(base.fillna(-1), other.fillna(-1))


def test_negative_slopes_are_clipped_unless_allowed():
    score, returns = _planted(slope=-1.0)
    clipped = calibration_slope(score, returns, 21, 252).dropna()
    assert clipped.min() >= 0.0 and clipped.iloc[-500:].max() == 0.0          # early sampling noise can be positive; the settled value is zero
    assert calibration_slope(score, returns, 21, 252, allow_negative=True).dropna().iloc[-1] < 0


# --------------------------------------------------------------------------------------------- combination
def _panel(mean, std, name=""):
    return ForecastPanel.from_mean_std(pd.DataFrame(mean), pd.DataFrame(std), 21, name)


def test_combining_identical_forecasts_changes_nothing_and_disagreement_widens_the_std():
    a = _panel({"X": [0.02, 0.02]}, {"X": [0.05, 0.05]})
    same = combine_forecasts({"a": a, "b": a}, "equal")
    assert np.allclose(same.mean, a.mean) and np.allclose(same.std, a.std)
    b = _panel({"X": [-0.02, -0.02]}, {"X": [0.05, 0.05]})
    split = combine_forecasts({"a": a, "b": b}, "equal")
    assert np.allclose(split.mean, 0.0) and (split.std > a.std).all().all()
    assert (split.confidence < a.confidence).all().all()


def test_combination_rules_weights_and_missing_models():
    a = _panel({"X": [0.04, np.nan]}, {"X": [0.05, 0.05]})
    b = _panel({"X": [0.0, 0.01]}, {"X": [0.05, 0.05]})
    eq = combine_forecasts({"a": a, "b": b}, "equal")
    assert eq.mean.iloc[0, 0] == pytest.approx(0.02) and eq.mean.iloc[1, 0] == pytest.approx(0.01)       # renormalised over what is present
    given = combine_forecasts({"a": a, "b": b}, "given", pd.DataFrame({"a": [1.0, 1.0], "b": [0.0, 0.0]}))
    assert given.mean.iloc[0, 0] == pytest.approx(0.04)
    conf = combine_forecasts({"a": a, "b": b}, "confidence")
    assert conf.mean.iloc[0, 0] > eq.mean.iloc[0, 0]                                                     # the more confident model gets more weight
    with pytest.raises(ValueError):
        combine_forecasts({"a": a}, "nonsense")


# --------------------------------------------------------------------------------------------- regimes
@pytest.mark.parametrize("name", ["vol_state", "macro", "composite", "static"])
def test_detectors_return_probabilities_that_sum_to_one_and_are_causal(name, bundle):
    detector = DETECTORS.create(name)
    series = detector.detect(bundle)
    series.validate()
    assert series.probabilities.dropna().shape[0] > 100
    check = check_causality(detector, bundle)
    assert check["ok"], check


def test_hmm_detector_is_causal(bundle):
    detector = DETECTORS.create("hmm", min_train=500, refit_every=250, n_init=1)
    series = detector.detect(bundle)
    series.validate()
    assert check_causality(detector, bundle)["ok"]


def test_ramp_is_a_soft_threshold():
    x = pd.Series([0.0, 0.5, 1.0, 2.0])
    assert list(ramp(x, 0.5, 1.5)) == [0.0, 0.0, 0.5, 1.0]


def test_macro_detector_flags_inflation_when_cpi_is_high(bundle):
    from dataclasses import replace

    probs = DETECTORS.create("macro").detect(bundle).probabilities
    assert probs["Inflation"].iloc[-1] == pytest.approx(1.0) and probs["Inflation"].iloc[0] == 0.0 and probs["Deflation"].iloc[0] == 0.0
    falling = replace(bundle, macro=pd.DataFrame({"CPI_YOY": -0.005}, index=bundle.index))
    assert DETECTORS.create("macro").detect(falling).probabilities["Deflation"].iloc[-1] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------- risk
def test_variable_target_reproduces_the_engine_scalar_for_a_constant_target():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=400)
    r = pd.DataFrame(rng.normal(0, 0.01, (400, 3)), index=idx, columns=list("ABC"))
    w = pd.DataFrame(1 / 3, index=idx, columns=r.columns)
    a, s1 = vol_target_scaling(w, r, 0.1, 63, 1.5)
    b, s2 = vol_target_series(w, r, 0.1, 63, 1.5)
    assert np.allclose(s1, s2) and np.allclose(a, b)


def test_regime_risk_policy_cuts_the_target_as_crisis_becomes_probable():
    idx = pd.bdate_range("2020-01-01", periods=3)
    regimes = RegimeSeries(pd.DataFrame({"LowVol": [1.0, 0.5, 0.0], "Crisis": [0.0, 0.5, 1.0]}, index=idx))
    target = RegimeRiskPolicy(targets={"LowVol": 0.10, "Crisis": 0.04}).target_series(regimes, idx)
    assert list(np.round(target, 3)) == [0.10, 0.07, 0.04]


# ---------------------------------------------------------------------------------------------- pipeline
@register_model("test_trend", "time-series", "sign of the trailing return, for tests")
class _Trend(ForecastModel):
    name, family, position_mode = "test_trend", "time-series", "time_series"

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def score(self, data):
        return np.sign(data.prices.pct_change(self.lookback)).where(data.investable)


@register_model("test_peek", "bad", "a model that looks one day ahead: must fail the causality test")
class _Peek(ForecastModel):
    name = "test_peek"

    def score(self, data):
        return data.returns.shift(-1)


def test_a_model_that_peeks_fails_the_causality_check_and_an_honest_one_passes(bundle):
    assert check_causality(MODELS.create("test_trend"), bundle)["ok"]
    bad = check_causality(MODELS.create("test_peek"), bundle)
    assert not bad["ok"] and bad["max_abs_difference"] > 0


def test_pipeline_end_to_end_on_synthetic_data(bundle, config):
    spec = {"name": "t", "models": [{"name": "test_trend", "params": {"lookback": 40}}], "regime": {"detector": "vol_state"},
            "evaluation": {"benchmarks": ["equal_weight"], "n_trials": 3}}
    result = Pipeline(spec, config, bundle).run()
    m = result.metrics
    assert result.start is not None and np.isfinite(m["sharpe"]) and m["n_days"] > 300
    assert result.validation["causality"]["test_trend"]["ok"] and result.validation["causality"]["regime:vol_state"]["ok"]
    assert 0.0 <= result.validation["deflated_sharpe_probability"] <= 1.0
    assert set(result.tables) >= {"attribution", "attribution_by_asset", "by_sample", "benchmarks", "by_regime", "reliability"}
    # attribution closes: the components sum to the net return in return points (sum of daily net returns)
    attribution = result.tables["attribution"]["return_points"]
    assert attribution.sum() == pytest.approx(100 * result.window.sum(), abs=1e-6)
    assert result.forecasts.first_valid() is not None and (result.forecasts.confidence.stack().dropna().between(0, 1)).all()


def test_regime_switch_blends_books_by_probability(bundle, config):
    @register_allocator("test_const_a", "all in the first asset")
    class A(Allocator):
        def build(self, ctx):
            return pd.DataFrame({c: float(i == 0) for i, c in enumerate(ctx.bundle.assets)}, index=ctx.bundle.index)

    @register_allocator("test_const_b", "all in the last asset")
    class B(Allocator):
        def build(self, ctx):
            return pd.DataFrame({c: float(i == len(ctx.bundle.assets) - 1) for i, c in enumerate(ctx.bundle.assets)}, index=ctx.bundle.index)

    idx = bundle.index
    probs = pd.DataFrame({"Calm": 0.25, "Stress": 0.75}, index=idx)
    ctx = Context(bundle, config, regimes=RegimeSeries(probs))
    rule = {"Calm": {"allocator": "test_const_a"}, "Stress": {"allocator": "test_const_b"}}
    blend = ALLOCATORS.create("regime_switch", rules=rule, mode="soft").build(ctx)
    assert blend.iloc[100, 0] == pytest.approx(0.25) and blend.iloc[100, -1] == pytest.approx(0.75)
    hard = ALLOCATORS.create("regime_switch", rules=rule, mode="hard").build(ctx)
    assert hard.iloc[100, -1] == pytest.approx(1.0) and hard.iloc[100, 0] == pytest.approx(0.0)
    tilted = ALLOCATORS.create("regime_switch", rules={"Calm": {"allocator": "test_const_a", "tilt": {bundle.assets[1]: 1.0}}, "Stress": {"allocator": "test_const_b"}},
                               mode="hard").build(Context(bundle, config, regimes=RegimeSeries(pd.DataFrame({"Calm": 1.0, "Stress": 0.0}, index=idx))))
    assert tilted.iloc[100, 0] == pytest.approx(0.5) and tilted.iloc[100, 1] == pytest.approx(0.5)      # tilt added, gross restored to 1


def test_spec_validation_rejects_unknown_keys():
    from src.framework import PipelineSpec
    with pytest.raises(ValueError, match="unknown spec keys"):
        PipelineSpec.from_dict({"name": "x", "modles": []})


# --------------------------------------------------------------------------------------- data quality
def test_data_quality_finds_each_planted_problem_and_passes_clean_data(synthetic_prices):
    from src.framework.dataquality import check_prices, format_report

    clean = check_prices(synthetic_prices)
    assert not clean["errors"]
    broken = synthetic_prices.copy()
    broken.iloc[300:312, 0] = broken.iloc[299, 0]                      # stale run
    broken.iloc[500, 1] = broken.iloc[499, 1] * 1.8                      # a 80% jump
    broken.iloc[700, 2] = np.nan                                         # a missing price
    report = check_prices(broken)
    text = "\n".join(report["warnings"])
    assert "identical consecutive prices" in text and "daily return" in text and "missing price" in text
    dup = pd.concat([synthetic_prices.iloc[:5], synthetic_prices.iloc[:5]])
    assert any("duplicate" in e for e in check_prices(dup)["errors"])
    assert any("below zero" in e or "at or below zero" in e for e in check_prices(synthetic_prices.iloc[:, :5] * 0 - 1)["errors"])
    assert any("at least" in e for e in check_prices(synthetic_prices.iloc[:, :2])["errors"])
    assert "ERROR" in format_report(check_prices(dup))


def test_a_bundle_from_your_own_prices_never_fills_or_alters_them(synthetic_prices):
    prices = synthetic_prices.copy()
    prices.iloc[:100, 0] = np.nan
    b = bundle_from_prices(prices, min_history=60)
    assert b.prices.iloc[:100, 0].isna().all() and not b.investable.iloc[:100, 0].any() and b.investable.iloc[160:, 0].all()
    assert np.allclose(b.returns.iloc[200:, 1], prices.iloc[:, 1].pct_change().iloc[200:])
    with pytest.raises(ValueError):
        bundle_from_prices(pd.concat([prices.iloc[:3], prices.iloc[:3]]))


# ------------------------------------------------------------------------------ the adaptive signal
def test_regime_trust_weights_follow_the_model_that_works_in_the_current_regime_and_use_only_the_past():
    from src.framework.forecasting import regime_trust_weights

    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2015-01-01", periods=1600)
    labels = pd.Series(np.where((np.arange(1600) // 200) % 2 == 0, "Calm", "Stress"), index=idx)
    good_in_calm = rng.normal(0, 0.01, 1600) + np.where(labels == "Calm", 0.002, -0.002)
    good_in_stress = rng.normal(0, 0.01, 1600) + np.where(labels == "Stress", 0.002, -0.002)
    streams = pd.DataFrame({"calm_model": good_in_calm, "stress_model": good_in_stress}, index=idx)
    updates = idx[::21]
    w = regime_trust_weights(streams, labels, updates, min_regime_days=100, shrink=0.0)
    late_calm, late_stress = idx[1218], idx[1512]          # update dates (multiples of 21) inside a Calm and a Stress block
    assert labels.loc[late_calm] == "Calm" and w.loc[late_calm, "calm_model"] > 0.8
    assert labels.loc[late_stress] == "Stress" and w.loc[late_stress, "stress_model"] > 0.8
    assert np.allclose(w.sum(axis=1), 1.0)
    # causality: replacing everything after a date leaves the earlier weights unchanged
    cut = idx[1000]
    shocked = streams.copy()
    shocked.loc[shocked.index > cut] = rng.normal(0, 0.05, shocked.loc[shocked.index > cut].shape)
    w2 = regime_trust_weights(shocked, labels, updates, min_regime_days=100, shrink=0.0)
    assert np.allclose(w.loc[:cut], w2.loc[:cut])
    # with too little history in the regime the weights are equal
    assert np.allclose(regime_trust_weights(streams.iloc[:150], labels.iloc[:150], idx[:150][::21], 200, 0.0).iloc[-1], 0.5)
