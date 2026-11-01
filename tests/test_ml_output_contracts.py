"""P0-09 and P0-11: what a model's output means is checked where it is used, and the validators of each output kind."""

import numpy as np
import pandas as pd
import pytest

from src.ml import experiments as xe
from src.ml.contracts import (OutputContractError, build_contract, check_signal_use, output_kind, pinball_loss, validate_distribution, validate_probabilities, validate_quantiles)
from tests.ml_helpers import make_bundle

FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
BASE = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 400, "refit_every": 250}, "signal": {"n": 2}}


# ------------------------------------------------------------------------------------------------------------ meaning and use
def test_each_model_kind_says_what_its_score_is():
    assert output_kind("ridge", "regression", "fwd_ret_5") == "expected_return"
    assert output_kind("lgbm_ranker", "regression", "fwd_ret_5") == "ranking_score" and output_kind("xgb_ranker", "regression", "rank_5") == "ranking_score"
    assert output_kind("lgbm_quantile", "regression", "fwd_ret_5") == "quantile_forecast" and output_kind("tft", "regression", "fwd_ret_5") == "quantile_forecast"
    assert all(output_kind(k, "regression", "fwd_ret_5") == "portfolio_weights" for k in ("ppo", "sac", "recurrent_ppo"))
    assert output_kind("logistic", "classification", "direction_5") == "class_probability"
    assert output_kind("ridge", "regression", "fwd_vol_20") == "volatility_forecast" and output_kind("ridge", "regression", "fwd_dd_20") == "drawdown_forecast"
    assert output_kind("ppo", "regression", "fwd_vol_20") == "portfolio_weights", "an agent's output is a book whatever target it was trained against"


@pytest.mark.parametrize("kind, task, target, method, text", [
    ("ppo", "regression", "fwd_ret_5", "top_n_long_short", "sizes its own positions"),
    ("ppo", "regression", "fwd_ret_5", "vol_scaled", "target_weights"),
    ("ridge", "regression", "fwd_ret_5", "target_weights", "outputs a expected return"),
    ("lgbm_ranker", "regression", "fwd_ret_5", "target_weights", "ranking score"),
    ("ridge", "regression", "fwd_vol_20", "top_n_long_short", "volatility forecast"),
    ("ridge", "regression", "fwd_dd_20", "proportional", "drawdown forecast"),
])
def test_a_position_rule_that_would_misread_an_output_is_refused_with_what_to_do_instead(kind, task, target, method, text):
    with pytest.raises(OutputContractError, match=text) as err:
        check_signal_use(kind, task, target, {"method": method})
    assert "reinterpret_outputs" in str(err.value)
    assert check_signal_use(kind, task, target, {"method": method, "reinterpret_outputs": True}) == output_kind(kind, task, target)


@pytest.mark.parametrize("kind, task, target, method", [("ridge", "regression", "fwd_ret_5", "top_n_long_short"), ("lgbm_ranker", "regression", "fwd_ret_5", "rank_normalised"),
                                                          ("lgbm_quantile", "regression", "fwd_ret_5", "uncertainty_weighted"), ("ppo", "regression", "fwd_ret_5", "target_weights"),
                                                          ("logistic", "classification", "direction_5", "confidence_weighted")])
def test_the_pairings_that_make_sense_pass(kind, task, target, method):
    assert check_signal_use(kind, task, target, {"method": method}) == output_kind(kind, task, target)


def test_an_experiment_with_a_misread_output_is_refused_before_any_training():
    with pytest.raises(xe.ExperimentError, match="sizes its own positions"):
        xe.run_experiment({**BASE, "model": {"kind": "ppo", "params": {}}, "signal": {"method": "top_n_long_short"}}, bundle=make_bundle(6, 600, seed=1))
    with pytest.raises(xe.ExperimentError, match="forecasts volatility forecast"):
        xe.run_experiment({**BASE, "dataset": {**BASE["dataset"], "target": "fwd_vol_5"}, "model": {"kind": "ridge", "params": {}}}, bundle=make_bundle(6, 600, seed=1))


def test_a_report_carries_the_versioned_output_contract():
    run = xe.run_experiment({**BASE, "model": {"kind": "ridge", "params": {"alpha": 10.0}}}, bundle=make_bundle(8, 1300, seed=2, alpha=1.0), final_model=False, explain=False)
    c = run.report["output_contract"]
    assert c["schema_version"] == 1 and c["output_kind"] == "expected_return" and c["horizon"] == run.dataset.horizon and c["horizon_unit"] == "trading_days"
    assert c["asset_order"] == list(run.dataset.assets) and len(c["feature_schema_hash"]) == 16 and c["model_version_id"] == run.report["config_hash"] and c["state_schema_version"] is None
    assert c == build_contract("ridge", "regression", run.cfg["dataset"]["target"], run.dataset, run.cfg, run.report["config_hash"])


# ------------------------------------------------------------------------------------------------------------ validators
def test_probabilities_must_be_finite_bounded_and_normalised():
    ok = validate_probabilities([[0.2, 0.8], [1.0, 0.0]])
    assert ok.shape == (2, 2)
    for bad in ([[0.5, 0.6]], [[np.nan, 1.0]], [[-0.2, 1.2]]):
        with pytest.raises(OutputContractError):
            validate_probabilities(bad)
    assert validate_probabilities([0.0, 0.5, 1.0]).shape == (3,)


def test_quantiles_are_sorted_and_the_crossing_rate_is_reported_not_hidden():
    values = np.array([[0.1, 0.3, 0.2], [0.0, 0.1, 0.2], [0.5, 0.4, 0.6], [0.0, 0.1, 0.2]])
    fixed, rate = validate_quantiles([0.1, 0.5, 0.9], values)
    assert np.all(np.diff(fixed, axis=1) >= 0) and rate == pytest.approx(0.5)
    with pytest.raises(OutputContractError, match="2 rows"):
        validate_quantiles([0.1, 0.5, 0.9], values, repair=False)
    for levels, shape in (([0.5, 0.1], (2, 2)), ([0.0, 0.5], (2, 2)), ([0.1, 0.5, 0.9], (2, 2))):
        with pytest.raises(OutputContractError):
            validate_quantiles(levels, np.zeros(shape))
    with pytest.raises(OutputContractError, match="finite"):
        validate_quantiles([0.5], np.array([[np.inf]]))


def test_pinball_loss_matches_a_hand_calculation_and_is_minimised_by_the_true_quantile():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pinball_loss(y, np.full(4, 2.0), 0.9) == pytest.approx((0.9 * (-1 + 0 + 1 + 2) * 0 + 0.1 * 1 + 0.0 + 0.9 * 1 + 0.9 * 2) / 4)
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000)
    grid = np.linspace(-2, 2, 81)
    best = grid[np.argmin([pinball_loss(z, g, 0.8) for g in grid])]
    assert best == pytest.approx(np.quantile(z, 0.8), abs=0.06)


def test_a_predictive_distribution_needs_a_positive_scale_and_a_mean_inside_its_interval():
    good = pd.DataFrame({"mean": [0.0, 0.1], "std": [0.2, 0.3], "lower": [-0.3, -0.4], "upper": [0.3, 0.5]})
    assert validate_distribution(good) is good
    for column, value in (("std", 0.0), ("std", np.nan), ("mean", 9.0), ("lower", np.inf)):
        bad = good.copy()
        bad.loc[0, column] = value
        with pytest.raises(OutputContractError):
            validate_distribution(bad)
    with pytest.raises(OutputContractError, match="columns"):
        validate_distribution(good[["mean", "std"]])
