"""Bootstrap intervals, walk-forward choice and the realism checklist for backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets import options_analytics as oa
from src.derivatives import validation as val


def series(mean, sd=0.01, n=400, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, n), index=pd.bdate_range("2023-01-02", periods=n))


def test_the_stationary_bootstrap_keeps_blocks_and_gives_wider_intervals_for_shorter_samples():
    rng = np.random.default_rng(0)
    idx = val.stationary_indices(10_000, 20.0, rng)
    runs = np.diff(idx) == 1
    assert 0.9 < runs.mean() < 0.98, "about 1 - 1/20 of steps continue a block"
    long = val.bootstrap(series(0.0008, n=1000), n=300)
    short = val.bootstrap(series(0.0008, n=100), n=300)
    wl, ws = long["interval"]["sharpe"][1] - long["interval"]["sharpe"][0], short["interval"]["sharpe"][1] - short["interval"]["sharpe"][0]
    assert ws > wl and long["point"]["sharpe"] == pytest.approx(val._sharpe(series(0.0008, n=1000).to_numpy(), 252.0))
    assert val.bootstrap(series(0.0, n=200), n=300)["sharpe_interval_contains_zero"] and not val.bootstrap(series(0.004, n=500), n=300)["sharpe_interval_contains_zero"]
    assert val.bootstrap(series(0.0, n=10))["available"] is False


def test_the_interval_has_close_to_its_nominal_coverage_on_independent_data():
    hits = 0
    for seed in range(40):
        r = series(0.0006, sd=0.01, n=500, seed=seed)
        b = val.bootstrap(r, n=200, seed=seed)
        truth = 0.0006 / 0.01 * np.sqrt(252)
        hits += b["interval"]["sharpe"][0] <= truth <= b["interval"]["sharpe"][1]
    assert hits / 40 > 0.8, hits


def test_walk_forward_charges_the_cost_of_choosing_among_settings_that_are_all_noise():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=700)
    runs = {f"s{i}": pd.Series(rng.normal(0.0, 0.01, 700), index=idx) for i in range(12)}
    wf = val.walk_forward(runs, train=126, test=42)
    assert len(wf["folds"]) >= 10 and all(f["test"][0] > f["train"][1] for f in wf["folds"]), "every test window is after its training window"
    assert wf["in_sample_sharpe"] > wf["oos_sharpe"] and wf["gap"] > 0.15 and "cost of choosing" in wf["reading"], "the best of twelve noise strategies looks good in sample and not out of it"
    with pytest.raises(ValueError, match="at least two settings"):
        val.walk_forward({"a": runs["s0"]})
    with pytest.raises(ValueError, match="not enough days"):
        val.walk_forward({k: v.iloc[:100] for k, v in list(runs.items())[:2]})


def test_walk_forward_finds_a_setting_that_is_really_better():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2022-01-03", periods=700)
    runs = {"good": pd.Series(rng.normal(0.0015, 0.01, 700), index=idx), "bad": pd.Series(rng.normal(-0.0005, 0.01, 700), index=idx), "flat": pd.Series(rng.normal(0.0, 0.01, 700), index=idx)}
    wf = val.walk_forward(runs, train=126, test=42)
    assert wf["settings_chosen"].get("good", 0) >= 0.7 * len(wf["folds"]) and wf["oos_sharpe"] > 0.8


def test_a_backtest_is_called_simplified_until_every_essential_friction_is_modelled():
    r = val.realism("options", {"commissions": True, "bid-ask spread crossed on every fill": True})
    assert r["label"] == "simplified" and "assignment and early exercise (American options)" in r["not_modelled"] and r["statement"].startswith("Not a realistic simulation")
    full = val.realism("futures", {i: True for i in val.ESSENTIAL["futures"]})
    assert full["label"] == "covers the listed frictions" and not full["not_modelled"]
    assert "daily variation margin" in val.ESSENTIAL["futures"] and "contract multiplier and tick size" in val.ESSENTIAL["futures"]


def test_an_options_backtest_carries_its_bootstrap_and_its_realism_label_and_never_calls_itself_realistic():
    out = oa.run_strategy_backtest("short_put", n_days=252, seed=1)
    v = out["validation"]
    assert v["bootstrap"]["available"] and len(v["bootstrap"]["interval"]["sharpe"]) == 2 and v["realism"]["label"] == "simplified"
    assert "margin requirements and funding" in v["realism"]["not_modelled"] and "realistic" not in v["realism"]["label"].replace("Not a realistic", "")
    zero_cost = oa.run_strategy_backtest("short_put", n_days=252, seed=1, commission=0.0, fill_fraction=0.0)["validation"]["realism"]
    assert "commissions" in zero_cost["not_modelled"] and "bid-ask spread crossed on every fill" in zero_cost["not_modelled"], "a free, mid-price backtest says so"
