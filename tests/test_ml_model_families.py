"""Fixes found by running every model family through the dashboard's training route with the recommended set-up (docs/ml_pipeline_audit.md).

* LightGBM grows trees leaf by leaf with no depth limit; its trees are capped at 7 leaves (the size class of XGBoost's depth 3) because deeper trees fit noise.
* The LightGBM ranker bags whole days: sampling single rows breaks each day's query apart.
* scikit-learn models fit and predict on one native thread, like the boosters' n_jobs=1 (histogram gradient boosting took 76 s instead of 0.7 s on a shared machine).
* regime_ridge cannot run without a regime detector; a new run that names none gets the volatility-state detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from src.ml.config import experiment_config, with_recommended
from src.ml.models import SklearnModel, make_model, model_catalog

AVAILABLE = {m["kind"]: m["available"] for m in model_catalog()}


def _need(kind):
    if not AVAILABLE.get(kind):
        pytest.skip(f"{kind} is not installed here")


def _frame(n=400, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=n // 10), [f"A{i}" for i in range(10)]], names=["date", "asset"])
    X = pd.DataFrame(rng.normal(size=(len(idx), k)), index=idx, columns=[f"f{i}" for i in range(k)])
    y = pd.Series(X["f0"].to_numpy() * 0.5 + rng.normal(size=len(idx)), index=idx)
    return X, y


@pytest.mark.parametrize("kind", ["lightgbm", "lgbm_ranker", "lgbm_quantile"])
def test_lightgbm_trees_default_to_seven_leaves(kind):
    _need(kind)
    info = {m["kind"]: m for m in model_catalog()}[kind]
    assert info["params"]["num_leaves"] == 7


def test_the_lightgbm_ranker_bags_whole_days():
    _need("lgbm_ranker")
    X, y = _frame()
    m = make_model("lgbm_ranker", {"n_estimators": 20}).fit(X, y)
    assert m.est_.get_params()["bagging_by_query"] is True


class _RecordsThreads(BaseEstimator, RegressorMixin):
    """Records how many native threads the pools allow while it fits and predicts."""

    def fit(self, X, y, sample_weight=None):
        from threadpoolctl import threadpool_info

        self.fit_threads_ = max([p["num_threads"] for p in threadpool_info()] or [1])
        self.coef_ = np.zeros(X.shape[1])
        return self

    def predict(self, X):
        from threadpoolctl import threadpool_info

        self.predict_threads_ = max([p["num_threads"] for p in threadpool_info()] or [1])
        return np.zeros(len(X))


def test_scikit_learn_models_fit_and_predict_on_one_native_thread():
    pytest.importorskip("threadpoolctl")
    X, y = _frame()
    est = _RecordsThreads()
    m = SklearnModel("custom", "regression", {}, 0, est, False, True).fit(X, y)
    m.raw_predict(X)
    assert m.pipe_.named_steps["model"].fit_threads_ == 1 and m.pipe_.named_steps["model"].predict_threads_ == 1


def test_histogram_gradient_boosting_still_learns_on_one_thread():
    X, y = _frame(n=2000)
    m = make_model("hist_gb", {"max_iter": 50}).fit(X, y)
    assert np.corrcoef(m.raw_predict(X), X["f0"])[0, 1] > 0.5


def test_regime_ridge_gets_a_regime_detector_unless_one_is_asked_for():
    assert experiment_config(with_recommended({"model": {"kind": "regime_ridge"}}))["dataset"]["regime_detector"] == "vol_state"
    assert experiment_config(with_recommended({"model": {"kind": "regime_ridge"}, "dataset": {"regime_detector": "hmm"}}))["dataset"]["regime_detector"] == "hmm"
    assert experiment_config(with_recommended({"model": {"kind": "ridge"}}))["dataset"]["regime_detector"] is None
