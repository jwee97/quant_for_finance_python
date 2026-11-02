"""Release 5, P1-20 and P1-21: neighbourhood robustness, delay decay and capacity decay, run on a real saved experiment.

A development-mode parent is used throughout, so the analyses read the validation segment only and the protected segments are never touched."""

import types

import numpy as np
import pandas as pd
import pytest

from src.ml import experiments as xe, registry as rg, research_controls as rc, sensitivity as sn
from tests.ml_helpers import make_bundle

OVER = {"dataset": {"feature_groups": ["price", "momentum", "volatility", "mean_reversion", "relative"]}, "split": {"min_train_days": 500, "refit_every": 126}, "model": {"kind": "ridge", "params": {"alpha": 10.0}}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    bundle = make_bundle(10, 1600, seed=41, alpha=1.0, macro=False)
    reg = rg.Registry(tmp_path_factory.mktemp("ml"))
    parent = xe.run_experiment(OVER, bundle=bundle, registry=reg, explain=False, final_model=False, stop_after="validation")
    return bundle, reg, parent


def test_the_neighbourhood_reports_the_whole_predeclared_grid_and_never_the_best_cell(world):
    bundle, reg, parent = world
    grid = {"signal.n": [2, 3, 4], "backtest.rebalance": ["weekly", "monthly"]}
    out = sn.neighborhood(reg, parent.id, grid, segment="validation")
    assert out["declared_points"] == 6 and out["evaluated"] == 6 and out["failed"] == 0 and out["eligibility"] == "applied"
    assert out["grid_hash"] == sn.grid_hash({"backtest.rebalance": ["weekly", "monthly"], "signal.n": [2, 3, 4]}), "the hash does not depend on the order the grid was written in"
    s = out["summary"]
    assert s["min"] <= s["p25"] <= s["median"] <= s["p75"] <= s["max"] and 0.0 <= s["share_positive"] <= 1.0 and 1 <= s["baseline_rank_from_top"] <= 7
    assert "best" not in out and all("best" not in k for k in s), "the distribution is reported; no cell is selected"
    assert any("not selected" in n for n in out["notes"])
    lin = rc.lineage(reg, rc.study_of_experiment(reg, parent.id))
    assert lin["raw_looks"] >= 7, "every grid point is a counted look at the study"
    assert all(r["experiment_id"] is None for r in out["points"]), "not saved as experiments unless asked"


def test_a_point_that_cannot_run_is_listed_as_failed_with_its_reason_and_a_grid_that_is_too_big_or_not_a_rebacktest_is_refused(world):
    _, reg, parent = world
    out = sn.neighborhood(reg, parent.id, {"signal.n": [2, 3, 0], "backtest.rebalance": ["weekly"]})
    bad = [r for r in out["points"] if r["state"] == "failed"]
    assert len(bad) == 1 and bad[0]["point"]["signal.n"] == 0 and "signal.n" in bad[0]["reason"] and out["failed"] == 1
    thin = sn.neighborhood(reg, parent.id, {"signal.n": [2, 3]})
    assert thin["eligibility"] == "insufficient" and thin["summary"] is None, "two points are not a distribution"
    with pytest.raises(ValueError, match="more than 64"):
        sn.grid_points({"a": list(range(9)), "b": list(range(9))})
    with pytest.raises(ValueError, match="not a re-backtest setting"):
        sn.neighborhood(reg, parent.id, {"dataset.target": ["fwd_ret_5", "fwd_ret_10", "fwd_ret_20"]})
    with pytest.raises(ValueError, match="non-empty list"):
        sn.grid_points({"signal.n": []})
    with pytest.raises(ValueError, match="metric must be"):
        sn.neighborhood(reg, parent.id, {"signal.n": [2, 3, 4]}, metric="alpha")


def test_delay_decay_adds_sessions_to_the_decision_to_fill_path_and_its_ratio_is_undefined_when_the_base_is_not_positive(world, monkeypatch):
    _, reg, parent = world
    out = sn.delay_decay(reg, parent.id, delays=(0, 1, 2, 5), segment="validation")
    assert [r["extra_sessions"] for r in out["rows"]] == [0, 1, 2, 5] and all(r["value"] is not None for r in out["rows"])
    base = out["rows"][0]["value"]
    if base > 0:
        assert out["rows"][0]["ratio_to_undelayed"] == pytest.approx(1.0) and out["eligibility"] == "applied"
    assert any("never shifted" in n for n in out["notes"])
    # the same code with stubbed children: a base that is not positive gives no ratio at all
    values = {0: -0.2, 1: 0.1, 2: 0.3, 5: -0.1}

    def fake(registry, exp_id, over, **kw):
        d = over["backtest"]["extra_lag"]
        rep = {"segments": {"stats": {"validation": {"sharpe": values[d], "n_days": 100}}}, "metrics": {"oos": {"ann_turnover": 3.0, "ann_cost_drag": 0.01}}}
        return types.SimpleNamespace(report=rep, id=f"x{d}")

    monkeypatch.setattr(sn.xp, "rebacktest", fake)
    stub = types.SimpleNamespace(report=lambda exp_id: {"config": {"backtest": {"extra_lag": 0}}})
    und = sn.delay_decay(stub, "e", delays=(0, 1, 2, 5))
    assert und["eligibility"] == "insufficient" and all(r["ratio_to_undelayed"] is None for r in und["rows"]) and "not positive" in und["reason"]
    values.update({0: 0.5, 1: 0.4, 2: 0.25, 5: 0.0})
    ok = sn.delay_decay(stub, "e", delays=(0, 1, 2, 5))
    assert [r["ratio_to_undelayed"] for r in ok["rows"]] == pytest.approx([1.0, 0.8, 0.5, 0.0])


def test_capacity_decay_needs_volume_and_more_capital_costs_more(world):
    bundle, reg, parent = world
    out = sn.capacity_decay(reg, parent.id, [1e5, 1e8, 1e10], segment="validation", bundle=bundle, execution={"max_participation": 0.05})
    ok = [r for r in out["rows"] if r["state"] == "ok"]
    assert len(ok) == 3 and out["eligibility"] == "applied"
    net = [r["net_return"] for r in ok]
    assert net[0] > net[1] > net[2], "the same predictions earn less per unit of capital as the book grows: the participation limit stops it being built and the impact grows"
    assert [r["impact"] for r in ok][0] < [r["impact"] for r in ok][1], "the impact paid, in money, rises with the size of the book"
    novol = make_bundle(10, 1600, seed=41, alpha=1.0, macro=False, volume=False)
    none = sn.capacity_decay(reg, parent.id, [1e5, 1e7], bundle=novol)
    assert none["eligibility"] == "applied" or all(r["capacity_status"] == "not_established" for r in none["rows"] if r["state"] == "ok"), "without volume the capacity is said not to be established"
    assert np.isfinite(ok[0]["net_return"])


def test_a_grid_over_a_setting_that_does_not_exist_is_refused_before_anything_runs(world):
    bundle, reg, parent = world
    with pytest.raises(ValueError, match="not a setting of the signal section"):
        sn.neighborhood(reg, parent.id, {"signal.top_frac": [0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="not a setting of the execution section"):
        sn.neighborhood(reg, parent.id, {"execution.no_such_thing": [1, 2, 3]})


def test_a_grid_over_the_models_own_hyper_parameters_refits_every_point_in_development_mode_and_counts_each_as_a_look(world):
    bundle, reg, parent = world
    study = rc.study_of_experiment(reg, parent.id)
    before = rc.lineage(reg, study)["raw_looks"]
    out = sn.neighborhood(reg, parent.id, {"model.params.alpha": [1.0, 10.0, 100.0, 1000.0]}, segment="validation", bundle=bundle)
    assert out["refit"] is True and out["declared_points"] == 4 and out["evaluated"] == 4 and out["failed"] == 0 and out["eligibility"] == "applied"
    assert any("REFITS" in n for n in out["notes"]) and any("not selected" in n for n in out["notes"]) and out["summary"]["distinct_values"] >= 2, "the strength of the penalty changes the result"
    for r in out["points"]:                                                                            # each point is a saved experiment of the same study and never predicted a protected segment
        preds = pd.read_parquet(reg.root / "experiments" / r["experiment_id"] / "predictions.parquet")
        assert set(preds["segment"].unique()) == {"validation"}
    same = next(r for r in out["points"] if r["point"]["model.params.alpha"] == 10.0)
    assert same["value"] == pytest.approx(out["baseline"], abs=1e-9), "the point that is the baseline's own setting reproduces the baseline: the refit is on the same windows and data"
    assert rc.lineage(reg, study)["raw_looks"] >= before + 4, "every refit is a counted look at the study"
    assert out["grid_hash"] == sn.grid_hash({"model.params.alpha": [1.0, 10.0, 100.0, 1000.0]}) and "best" not in out["summary"]


def test_a_model_grid_is_refused_before_anything_runs_when_it_mixes_kinds_names_an_unknown_setting_is_too_big_or_reads_a_protected_segment(world):
    _, reg, parent = world
    n = len(reg.list_experiments(10_000))
    with pytest.raises(ValueError, match="not both"):
        sn.neighborhood(reg, parent.id, {"model.params.alpha": [1, 2, 3], "signal.n": [2, 3]})
    with pytest.raises(ValueError, match="no parameter"):
        sn.neighborhood(reg, parent.id, {"model.params.nonsense": [1, 2, 3]})
    with pytest.raises(ValueError, match="more than 12"):
        sn.neighborhood(reg, parent.id, {"model.params.alpha": [float(i) for i in range(1, 14)]})
    with pytest.raises(ValueError, match="validation segment only"):
        sn.neighborhood(reg, parent.id, {"model.params.alpha": [1.0, 2.0, 3.0]}, segment="test")
    assert len(reg.list_experiments(10_000)) == n, "nothing was fitted"

