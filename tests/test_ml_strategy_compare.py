"""Models and baselines compared like for like: shared data and splits, one engine and cost, the same days, predictive quality kept apart from trading quality."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from src.ml import strategy_compare as sc
from src.utils.config import load_config
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
OVER = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504, "refit_every": 126},
        "signal": {"method": "long_only_top_quantile", "quantile": 0.3}, "backtest": {"rebalance": "monthly"}}


@pytest.fixture(scope="module")
def result():
    bundle = make_bundle(n_assets=10, n=1500, seed=4, alpha=1.0)
    return sc.compare([{"kind": "ridge"}, {"kind": "lasso"}], OVER, bundle=bundle, config=load_config(), draws=4)


def test_models_and_baselines_share_the_same_days_and_every_row_is_present(result):
    names = [r["name"] for r in result["rows"]]
    assert names[:2] == ["ridge", "lasso"]
    assert {"equal weight, rebalanced monthly", "equal weight, held (no rebalance)", "12-1 momentum, top 30%", "5-day reversal, top 30%", "trend: 50-day above 200-day average",
            "equal weight, 10% volatility target"} <= set(names) and any(n.startswith("random-signal books") for n in names)
    n_days = {r["n"] for r in result["rows"]}
    assert n_days == {result["period"]["days"]}, "every row is measured on exactly the same days"
    assert {r["kind"] for r in result["rows"]} == {"model", "baseline", "control"}
    assert len(result["equity"]["dates"]) == len(result["equity"][names[0]]) and len(result["correlation"]["names"]) == len(names)


def test_predictive_quality_and_trading_quality_are_reported_apart(result):
    models = [r for r in result["rows"] if r["kind"] == "model"]
    for r in models:
        p = r["predictive"]
        assert p["rank_ic"] is not None and p["mae_zero"] > 0 and p["mse_zero"] > 0 and "calibration_slope" in p
        assert {"cagr", "sharpe", "sortino", "max_drawdown", "beta", "information_ratio", "active_over_beta_matched", "ann_turnover", "cost_drag", "gross_exposure"} <= set(r)
    assert all(r["predictive"] is None for r in result["rows"] if r["kind"] != "model")


def test_a_model_with_planted_alpha_has_positive_rank_ic_and_the_long_only_book_is_not_dollar_neutral(result):
    m = next(r for r in result["rows"] if r["name"] == "ridge")
    assert m["predictive"]["rank_ic"] > 0.01 and m["net_exposure"] > 0.8 and m["gross_exposure"] > 0.8


def test_costs_and_exposure_make_the_baselines_differ_as_they_should(result):
    by = {r["name"]: r for r in result["rows"]}
    held, monthly = by["equal weight, held (no rebalance)"], by["equal weight, rebalanced monthly"]
    assert held["max_drawdown"] <= 0 and monthly["max_drawdown"] <= 0
    assert by["equal weight, 10% volatility target"]["ann_vol"] < monthly["ann_vol"] + 0.05
    assert by["equal weight, rebalanced monthly"]["beta"] == pytest.approx(1.0, abs=0.15)


def test_the_result_is_json_ready_and_the_design_is_stated(result):
    json.dumps(result, allow_nan=False, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x))
    assert result["design"]["models"] == ["ridge", "lasso"] and result["design"]["position_rule"] == "long_only_top_quantile" and any("multiple comparison" in n for n in result["notes"])


def test_too_many_models_or_none_are_refused():
    bundle = make_bundle(n_assets=8, n=1300, seed=5)
    with pytest.raises(ValueError, match="between 1 and"):
        sc.compare([], OVER, bundle=bundle, config=load_config())
    with pytest.raises(ValueError, match="between 1 and"):
        sc.compare([{"kind": "ridge"}] * 7, OVER, bundle=bundle, config=load_config())
