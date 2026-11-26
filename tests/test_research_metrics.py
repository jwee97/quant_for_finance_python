"""Which way is better, and when two numbers may be ranked at all: the metric registry and the rankings built on it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research import compare as rc
from src.research import metrics as mr
from src.research import store as rs


@pytest.fixture()
def store(tmp_path):
    return rs.ResearchStore(tmp_path / "research")


def test_every_error_and_loss_metric_is_lower_is_better_and_every_score_is_higher_is_better():
    lower = {"rmse", "mae", "mape", "crps", "pinball", "log_loss", "brier", "ann_vol", "turnover_annual", "cost_drag_annual"}
    higher = {"rank_ic", "best_combination_test_ic", "best_skill_common", "directional_accuracy", "sharpe", "sortino", "calmar", "cagr", "total_return", "max_drawdown"}
    for k in lower:
        assert mr.get(k).direction == "lower", k
    for k in higher:
        assert mr.get(k).direction == "higher", k
    assert mr.get("best_coverage90").direction == "target"
    with pytest.raises(mr.MetricError, match="unknown metric"):
        mr.get("vibes")


@pytest.mark.parametrize("key,values,best_first", [
    ("rmse", {"a": 0.9, "b": 0.4, "c": 0.6}, ["b", "c", "a"]),
    ("crps", {"a": 0.02, "b": 0.05}, ["a", "b"]),
    ("pinball", {"a": 3.0, "b": 1.0, "c": None}, ["b", "a", "c"]),
    ("log_loss", {"a": 0.7, "b": 0.69}, ["b", "a"]),
    ("rank_ic", {"a": 0.01, "b": 0.05, "c": -0.02}, ["b", "a", "c"]),
    ("directional_accuracy", {"a": 0.51, "b": 0.58}, ["b", "a"]),
    ("max_drawdown", {"a": -0.35, "b": -0.10}, ["b", "a"]),                                    # less negative is better
    ("ann_vol", {"a": 0.30, "b": 0.12}, ["b", "a"]),
    ("best_coverage90", {"a": 0.97, "b": 0.88, "c": 0.90}, ["c", "b", "a"]),                    # nearest to the stated 90% is best
])
def test_ranking_follows_the_metrics_own_direction_with_missing_values_last(key, values, best_first):
    assert mr.rank(values, key) == best_first


def _study(store, s, kind, title, metrics, **cfg):
    return store.add_artifact(s, {"kind": kind, "title": title, "evidence": "walk_forward", "data_kind": "platform", "context": {"assets": ["A"], "start": "2020-01-01", "end": "2022-01-01", "dataset_version": "v1"},
                                  "config": {"horizon": 5, **cfg}, "metrics": metrics})


def test_a_metric_is_refused_for_kinds_horizons_and_targets_it_cannot_compare(store):
    s = store.create_session("m")["id"]
    f5 = _study(store, s, "forecast", "f5", {"rmse": 0.02}, target="log_return")
    f21 = _study(store, s, "forecast", "f21", {"rmse": 0.05}, target="log_return", horizon=21)
    fx = _study(store, s, "factor_study", "factor", {"best_combination_test_ic": 0.04})
    assert mr.applicability([f5, f21], "rmse") and "horizon" in mr.applicability([f5, f21], "rmse")[0]
    assert any("not defined for" in w for w in mr.applicability([f5, fx], "rmse"))
    assert any("not defined for" in w for w in mr.applicability([f5], "best_combination_test_ic"))
    g = _study(store, s, "forecast", "g", {"rmse": 0.03}, target="price")
    assert any("units" in w for w in mr.applicability([f5, g], "rmse")), "an RMSE in log returns and one in prices are not comparable"
    h = _study(store, s, "forecast", "h", {}, target="log_return")
    assert any("was not saved" in w for w in mr.applicability([f5, h], "rmse"))
    assert mr.applicability([f5, _study(store, s, "forecast", "f5b", {"rmse": 0.03}, target="log_return")], "rmse") == []


def test_forecast_studies_are_ranked_by_skill_highest_first_and_a_horizon_difference_blocks_it(store):
    s = store.create_session("m")["id"]
    a, b = _study(store, s, "forecast", "a", {"best_skill_common": 0.02}), _study(store, s, "forecast", "b", {"best_skill_common": 0.07})
    out = rc.study_comparison([a, b])
    assert out["predictive"]["ranking"] == [b["id"], a["id"]] and out["predictive"]["ranked_by"]["direction"] == "higher"
    c = _study(store, s, "forecast", "c", {"best_skill_common": 0.5}, horizon=21)
    out = rc.study_comparison([a, c])
    assert out["predictive"]["ranking"] is None and out["predictive"]["blocked_because"]


def _trading(store, s, title, r, idx):
    lv = np.concatenate([[1.0], np.cumprod(1 + r)])
    return store.add_artifact(s, {"kind": "backtest", "title": title, "evidence": "historical_simulation", "data_kind": "platform", "context": {"assets": ["A"], "dataset_version": "v1"}, "metrics": {"sharpe": 1.0},
                                  "series": {"dates": [str(d.date()) for d in idx], "equity": [float(x) for x in lv]}})


def test_the_trading_ranking_can_be_by_volatility_lowest_first_and_by_drawdown_and_says_so(store):
    s = store.create_session("m")["id"]
    idx = pd.bdate_range("2021-01-04", periods=201)
    rng = np.random.default_rng(4)
    calm, wild = _trading(store, s, "calm", rng.normal(0.0004, 0.004, 200), idx), _trading(store, s, "wild", rng.normal(0.0004, 0.02, 200), idx)
    out = rc.compare([wild, calm], rank_by="ann_vol")
    assert out["ranking"] == [calm["id"], wild["id"]] and out["rank_by"]["direction"] == "lower" and "lowest first" in out["reason"]
    out = rc.compare([wild, calm], rank_by="max_drawdown")
    assert out["ranking"][0] == max((calm, wild), key=lambda a: out["metrics"][a["id"]]["max_drawdown"])["id"]
    assert rc.compare([wild, calm])["rank_by"]["key"] == "sharpe"


def test_a_predictive_metric_cannot_rank_trading_results_and_nothing_is_ranked_instead(store):
    s = store.create_session("m")["id"]
    idx = pd.bdate_range("2021-01-04", periods=121)
    rng = np.random.default_rng(6)
    a, b = _trading(store, s, "a", rng.normal(0.0004, 0.01, 120), idx), _trading(store, s, "b", rng.normal(0.0004, 0.01, 120), idx)
    out = rc.compare([a, b], rank_by="rmse")
    assert out["ranking"] is None and out["rankable"] is False and "not defined for" in out["reason"] and out["metrics"]
    with pytest.raises(mr.MetricError):
        rc.compare([a, b], rank_by="nonsense")


def test_the_registry_is_sent_with_the_comparison_so_the_page_never_guesses_a_direction(store):
    s = store.create_session("m")["id"]
    idx = pd.bdate_range("2021-01-04", periods=121)
    rng = np.random.default_rng(6)
    a = _trading(store, s, "a", rng.normal(0.0004, 0.01, 120), idx)
    reg = {m["key"]: m for m in rc.compare([a])["metric_registry"]}
    assert reg["rmse"]["direction"] == "lower" and reg["sharpe"]["direction"] == "higher" and reg["crps"]["family"] == "calibration"
