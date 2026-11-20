"""Scheduled walk-forward simulation: dated policy schedules, selection on prior data only, the model-version ledger and the whole loop end to end on a synthetic bundle."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import decisions as D
from src.ml import simulation as sim
from src.ml.config import MLConfigError, experiment_config
from src.utils.config import load_config
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
COLS = list("ABCDEFGH")


def world(n=400, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    r = pd.DataFrame(rng.normal(0.0003, 0.01, (n, 8)), idx, COLS)
    return r, r.shift(-2).rolling(5).mean().fillna(0.0) + rng.normal(0, 0.001, r.shape)


def pol(**kw):
    return D.policy_from({f"decision_{k}": v for k, v in kw.items()})


def test_a_dated_schedule_changes_the_policy_from_its_date_and_carries_the_book_over():
    r, s = world()
    switch = r.index[200]
    base = pol(max_positions=1, entry_quantile=0.2, exit_quantile=0.4, rebalance_days=5)
    sched = D.schedule_from({**{f"decision_{k}": v for k, v in base.items() if v is not None}, "decision_schedule": [{"from": str(switch.date()), "policy": {"max_positions": 4, "entry_quantile": 0.5, "exit_quantile": 0.75}}]})
    w, dec = D.run(s, r, base, vol=r.rolling(20).std(), lag=1, schedule=sched)
    held = (w > 1e-9).sum(axis=1)
    assert held.loc[:switch - pd.Timedelta(days=1)].max() <= 1, "before the date the base policy (one position) is in force"
    assert held.loc[switch:].max() >= 3, "from the date the scheduled policy (four positions) is in force"
    carried = w.loc[switch - pd.Timedelta(days=3)]
    assert (carried > 1e-9).sum() >= 1 and not dec[(dec["date"] == switch) & (dec["action"].isin(["SELL", "COVER"])) & (dec["reason"].str.contains("schedule"))].shape[0]
    base_only, _ = D.run(s, r, base, vol=r.rolling(20).std(), lag=1)
    assert np.allclose(w.loc[:switch - pd.Timedelta(days=1)].to_numpy(), base_only.loc[:switch - pd.Timedelta(days=1)].to_numpy()), "nothing before the date differs from running the base policy alone"


def test_a_bad_schedule_is_refused_by_the_configuration():
    with pytest.raises(MLConfigError, match="unknown settings"):
        experiment_config({"signal": {"method": "decision_policy", "decision_schedule": [{"from": "2020-01-01", "policy": {"nonsense": 1}}]}})
    with pytest.raises(MLConfigError, match="entry_quantile"):
        experiment_config({"signal": {"method": "decision_policy", "decision_schedule": [{"from": "2020-01-01", "policy": {"entry_quantile": 0.9}}]}})
    cfg = experiment_config({"signal": {"method": "decision_policy", "decision_schedule": [{"from": "2020-01-01", "policy": {"entry_quantile": 0.3}}]}})
    assert cfg["signal"]["decision_schedule"][0]["from"] == "2020-01-01"


def test_the_policy_is_chosen_on_days_before_each_refit_and_never_on_the_days_after():
    idx = pd.bdate_range("2020-01-01", periods=700)
    rng = np.random.default_rng(0)
    good_early = pd.Series(np.where(np.arange(700) < 350, 0.004, -0.004) + rng.normal(0, 0.006, 700), idx)      # A earns early and loses late
    flat = pd.Series(rng.normal(0, 0.006, 700), idx)
    nets = {"A": good_early, "B": flat}
    grid = [{"id": "B", "policy": {}}, {"id": "A", "policy": {}}]
    starts = [idx[300], idx[600]]
    c = sim.select_schedule(nets, starts, grid, "B", lookback_days=252, min_days=63, embargo=5)
    assert c[0]["chosen"] == "A" and c[0]["basis"] == "selected on prior data"
    assert c[1]["chosen"] == "B", "by the second refit A had been losing for 250 days"
    rewritten = {k: v.copy() for k, v in nets.items()}
    rewritten["A"].iloc[300 - 5:] = -0.05                                            # rewrite A from the embargo edge onward: the first choice cannot see it
    assert sim.select_schedule(rewritten, starts[:1], grid, "B", lookback_days=252, min_days=63, embargo=5)[0]["chosen"] == "A"
    short = sim.select_schedule(nets, [idx[20]], grid, "B", lookback_days=252, min_days=63, embargo=5)[0]
    assert short["chosen"] == "B" and "default" in short["basis"]


def test_a_grid_that_is_too_small_or_has_duplicate_ids_is_refused():
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    with pytest.raises(ValueError, match="between 2 and"):
        sim.simulate({}, [{"id": "only", "policy": {}}], bundle=bundle, config=load_config(), save=False)
    with pytest.raises(ValueError, match="unique"):
        sim.simulate({}, [{"id": "x", "policy": {}}, {"id": "x", "policy": {}}], bundle=bundle, config=load_config(), save=False)


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    from src.ml.registry import Registry

    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    reg = Registry(tmp_path_factory.mktemp("sim"))
    over = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504, "refit_every": 63}, "model": {"kind": "ridge"},
            "signal": {"decision_max_positions": 3, "decision_rebalance_days": 5}}
    exp, block = sim.simulate(over, sim.default_grid()[:4], bundle=bundle, config=load_config(), registry=reg, lookback_days=126, min_days=40)
    return exp, block, reg


def test_every_refit_is_a_recorded_model_version_whose_labels_had_settled(simulated):
    exp, block, _ = simulated
    v = block["model_versions"]
    assert len(v) >= 3 and [x["version"] for x in v] == sorted(x["version"] for x in v)
    for a in v:
        assert a["label_end_max"] <= a["fit_cutoff"] <= a["predicts_from"], "the labels of the training rows had settled by the fitting day, which is not after the first prediction"
        assert len(a["train_rows_hash"]) >= 8
    assert block["oos_violations"] == [], "the framework's own check finds no prediction made by a model that had seen its date"
    assert len({a["train_rows_hash"] for a in v}) == len(v), "each refit trained on its own rows"
    cuts = [a["fit_cutoff"] for a in v]
    assert cuts == sorted(cuts) and len(set(cuts)) == len(cuts)


def test_the_run_records_the_choice_at_each_refit_and_what_it_was_chosen_on(simulated):
    exp, block, _ = simulated
    ch = block["choices"]
    assert len(ch) == len(block["model_versions"]) and ch[0]["from"] == block["model_versions"][0]["predicts_from"]
    assert all(c["chosen"] in {g["id"] for g in block["policy_grid"]} for c in ch)
    assert any(c["basis"].startswith("default") for c in ch) and block["selection"]["n_candidates"] == 4
    assert block["schedule"][0]["from"] == ch[0]["from"] and all(s["policy"] for s in block["schedule"])
    assert block["reproducibility"]["refits"] == len(block["model_versions"]) and len(block["reproducibility"]["config_hash"]) >= 8


def test_the_simulation_is_a_saved_experiment_the_explorer_can_read_and_is_counted_as_a_search(simulated):
    exp, block, reg = simulated
    report = reg.report(exp.id)
    assert report["simulation"]["comparison"]["scheduled"]["days"] > 100 and len(report["simulation"]["comparison"]["fixed"]) == 4
    assert report["trading"]["n_executions"] > 0 and report["config"]["signal"]["decision_schedule"], "the schedule is part of the saved configuration"
    d = reg.experiment_dir(exp.id)
    assert (d / "decisions.parquet").exists() and (d / "executions.parquet").exists()
    assert report["trials"]["dataset_trials"] >= 1
    dec = pd.read_parquet(d / "decisions.parquet")
    assert dec["action"].isin(D.ACTIONS).all()
