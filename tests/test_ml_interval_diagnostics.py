"""P1-22 on a saved experiment: when the model stored a predictive distribution, the statistical diagnostics score its interval (coverage with an interval, width, interval score, pinball) for the diagnosed segment
and put the two time-series conformal procedures beside it, calibrated only on outcomes that had ended; a model with no distribution says so. The coverage is recomputed here from the stored files."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import diagnostics as dx, experiments as xe, registry as rg
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
BASE = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 400, "refit_every": 126}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    bundle = make_bundle(6, 1300, seed=21, alpha=1.0)
    reg = rg.Registry(tmp_path_factory.mktemp("interval") / "ml")
    ens = xe.run_experiment({**BASE, "name": "ens", "model": {"kind": "model_ensemble", "params": {"members": "ridge,lasso", "weighting": "equal"}}}, bundle=bundle, registry=reg, explain=False, robustness=False)
    ridge = xe.run_experiment({**BASE, "name": "ridge", "model": {"kind": "ridge", "params": {}}}, bundle=bundle, registry=reg, explain=False, robustness=False)
    return reg, ens, ridge


def test_the_diagnostics_score_the_stored_interval_for_the_diagnosed_segment_and_match_a_recomputation(world):
    reg, ens, _ = world
    out = dx.run(reg, ens.id, n_boot=100, save=False)
    ic = out["interval_calibration"]
    assert ic["eligibility"] == "applied" and ic["segment"] == "validation" and ic["model"] == "model_ensemble"
    assert set(ic["quantiles"]["segments"]) == {"validation"}, "only the diagnosed segment is scored: the protected ones are not read"
    v = ic["quantiles"]["segments"]["validation"]
    folder = reg.root / "experiments" / ens.id
    preds, dist = pd.read_parquet(folder / "predictions.parquet"), pd.read_parquet(folder / "extra_distribution.parquet")
    rows = (preds["segment"] == "validation").to_numpy()
    d, y = dist[rows], preds["target"][rows]
    ok = y.notna().to_numpy()
    inside = ((y >= d["lower"]) & (y <= d["upper"]))[ok]
    assert v["eligibility"] == "applied" and v["n"] == int(ok.sum()) and v["nominal"] == pytest.approx(0.8)
    assert v["coverage"] == pytest.approx(float(inside.mean())) and v["width"] == pytest.approx(float((d["upper"] - d["lower"])[ok].mean()))
    assert v["coverage_interval"][0] <= v["coverage"] <= v["coverage_interval"][1] and v["pinball"]["centre"] > 0
    cp = ic["conformal"]
    horizon = int(ens.dataset.horizon)
    assert cp["delay_sessions"] == int(ens.cfg["dataset"]["lag"]) + int(ens.cfg["backtest"].get("extra_lag", 0)) + horizon and cp["nominal"] == pytest.approx(0.8)
    for name in ("split", "adaptive"):
        r = cp["methods"][name]["segments"]["validation"]
        assert r["eligibility"] == "applied" and 0.0 < r["coverage"] <= 1.0 and r["n"] <= v["n"], "the first forecasts have no matured residuals to calibrate on, so they are unscored"
    assert "none for time series" in cp["methods"]["split"]["guarantee"] and "not conditional" in cp["methods"]["adaptive"]["guarantee"]
    assert np.isfinite(v["interval_score"])


def test_a_point_forecast_model_has_no_interval_to_score_and_the_record_says_why(world):
    reg, _, ridge = world
    out = dx.run(reg, ridge.id, n_boot=100, save=False)["interval_calibration"]
    assert out["eligibility"] == "not_applicable" and "no predictive distribution" in out["reason"]
