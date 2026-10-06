"""CFTC positioning model and the explainability table: planted answers, missing data, invalid input, pipeline and tear-sheet integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, Pipeline, bundle_from_prices, load_library
from src.framework.tearsheet import make_tearsheet
from src.framework.validate import check_causality
from src.utils.config import load_config

load_library()


@pytest.fixture(scope="module")
def world():
    """Eight assets whose next-month return follows 21-day momentum (so the ridge has something to find), plus two positioning series."""
    rng = np.random.default_rng(8)
    n, k = 3300, 8
    idx = pd.bdate_range("2004-01-01", periods=n)
    r = rng.normal(0.0002, 0.01, (n, k))
    for t in range(22, n):
        r[t] += 0.06 * np.tanh(r[t - 21:t].sum(axis=0) / 0.2) / 21
    prices = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=["SPY", "QQQ", "IEF", "GLD", "SLV", "DBC", "EEM", "VNQ"])
    macro = pd.DataFrame({"cftc_es": 0.2 + np.cumsum(rng.normal(0, 0.004, n)), "cftc_gold": 0.4 + np.cumsum(rng.normal(0, 0.004, n))}, index=idx)
    macro["VIX"] = 20 + np.cumsum(rng.normal(0, 0.2, n))
    return bundle_from_prices(prices, macro=macro, name="world")


def test_positioning_fades_a_crowded_long_and_follows_when_asked(world):
    fade, follow = MODELS.create("cftc_positioning", sign=-1).score(world), MODELS.create("cftc_positioning", sign=1).score(world)
    assert np.allclose(fade["SPY"].dropna(), -follow["SPY"].dropna())
    assert fade["EEM"].fillna(0.0).abs().max() == 0.0                           # no positioning series for it: flat
    z = (world.macro["cftc_es"] - world.macro["cftc_es"].expanding(756).mean()) / world.macro["cftc_es"].expanding(756).std()
    both = pd.concat([z, fade["SPY"]], axis=1).dropna()
    assert len(both) > 1000 and both.corr().iloc[0, 1] < -0.9                      # the score moves against the positioning z-score


def test_positioning_is_causal_and_needs_its_series(world):
    assert check_causality(MODELS.create("cftc_positioning"), world)["ok"]
    bare = bundle_from_prices(world.prices, macro=world.macro[["VIX"]], name="bare")
    with pytest.raises(KeyError, match="cftc_positioning needs"):
        MODELS.create("cftc_positioning").score(bare)


@pytest.mark.parametrize("kwargs", [{"sign": 0}, {"sign": 2}, {"min_periods": 10}])
def test_positioning_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        MODELS.create("cftc_positioning", **kwargs)


def test_ridge_explanation_finds_the_planted_momentum_feature_and_shares_sum_to_one(world):
    table = MODELS.create("ml_ridge", min_train=1260).explain(world)
    assert table["share"].sum() == pytest.approx(1.0)
    trend_like = {"mom_21", "mom_63", "zscore_5", "zscore_21", "zscore_63", "rsi_14"}                    # all measure the planted 21-day trend
    assert table.index[0] in trend_like and table["share"].iloc[0] > 0.2
    assert not set(world.assets) & set(table.index)


def test_explain_defaults_to_none_for_models_without_features(world):
    assert MODELS.create("momentum").explain(world) is None


def test_explain_is_wired_into_the_pipeline_and_the_tearsheet(world, tmp_path):
    spec = {"models": [{"name": "ml_ridge", "params": {"min_train": 1260}}], "allocation": {"allocator": "forecast_stack"},
            "evaluation": {"explain": True, "causality": False, "benchmarks": []}}
    out = Pipeline(spec, load_config(), world).run(validate=False)
    assert "explain_ml_ridge" in out.tables and out.tables["explain_ml_ridge"]["importance"].notna().all()
    path = make_tearsheet(out, tmp_path)
    assert "What drives the forecast: ml_ridge" in path.read_text()


def test_ridge_with_macro_features_runs_end_to_end_through_the_pipeline(world):
    spec = {"models": [{"name": "ml_ridge", "params": {"features": "price_macro", "min_train": 1260}}], "allocation": {"allocator": "forecast_stack"},
            "evaluation": {"explain": True, "causality": False, "benchmarks": []}}
    out = Pipeline(spec, load_config(), world).run(validate=False)
    assert any(i.startswith("macro_") for i in out.tables["explain_ml_ridge"].index) and np.isfinite(out.metrics["sharpe"])
