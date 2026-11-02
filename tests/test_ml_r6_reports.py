"""Release 6: the alpha and portfolio diagnostics of a saved experiment.

They are run on experiments that were really trained and saved on a small synthetic market. The numbers are checked against what the experiment itself recorded and against independent calculations from the
saved positions, and the guards (sealed segments, mismatched experiments, a rank score that is not a return forecast) are exercised.
"""

import json

import pandas as pd
import pytest

from src.ml import experiments as xp, registry as rg, r6_reports as rr
from tests.ml_helpers import make_bundle

BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 500, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}}}
FAST = dict(explain=False, final_model=False, robustness=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    reg = rg.Registry(tmp_path_factory.mktemp("ml_r6"))
    b = make_bundle(12, 1500, seed=51, alpha=1.0)
    full = xp.run_experiment(BASE, bundle=b, registry=reg, **FAST)
    lasso = xp.run_experiment({**BASE, "model": {"kind": "lasso", "params": {"alpha": 0.02}}}, bundle=b, registry=reg, stop_after="validation", **FAST)
    ten = xp.run_experiment({**BASE, "dataset": {**BASE["dataset"], "target": "fwd_ret_10", "horizon": 10}}, bundle=b, registry=reg, stop_after="validation", **FAST)
    rank = xp.run_experiment({**BASE, "dataset": {**BASE["dataset"], "target": "rank_5"}}, bundle=b, registry=reg, stop_after="validation", **FAST)
    return reg, full, lasso, ten, rank


def test_the_alpha_diagnostics_agree_with_what_the_experiment_recorded_and_say_what_is_missing(world):
    reg, full, *_ = world
    out = rr.alpha_diagnostics(reg, full.id)
    assert out["eligibility"] == "applied" and out["segment"] == "validation" and out["horizon"] == 5 and "raw 5-session forward return" in out["label"] and out["output_kind"] == "expected_return"
    own = reg.report(full.id)["ic"]["by_segment"]["validation"]["mean"]
    mine = out["ic"]["mean"]["value"]
    assert mine == pytest.approx(own, abs=2e-3), (mine, own)
    assert out["ic"]["mean"]["settings"]["lags"] >= 4, "five-session labels overlap: the Newey-West lag is at least horizon - 1"
    assert out["ic"]["n_with_ic"] > 100 and set(out["ic"]["unavailable_by_reason"]) <= {"too_few", "tied_signal", "tied_target"}
    cells = {(c["extra_lag"], c["horizon"]): c for c in out["decay"]["cells"]}
    assert set(cells) == {(k, h) for k in (0, 1, 2, 5) for h in (1, 5, 10, 20)} and cells[(0, 5)]["mean_ic"] == pytest.approx(mine, abs=1e-9), "the undelayed cell at the experiment's horizon is the IC itself"
    assert out["coverage"]["eligibility"] != "insufficient" and out["stability"]["eligibility"] == "applied" and len(out["stability"]["blocks"]) == 4
    assert out["rolling"]["eligibility"] == "applied" and out["rolling"]["series"] and out["redundancy"]["rows_scope"].startswith("train and validation rows only")
    assert out["redundancy"]["eligibility"] == "applied" and out["redundancy"]["n_features"] > 5
    assert json.dumps(out, allow_nan=False)


def test_a_segment_that_drives_choices_is_read_only_where_it_may_be(world):
    reg, full, lasso, *_ = world
    for exp, status in ((full, "exposed"), (lasso, "development")):
        for seg in ("test", "forward"):
            with pytest.raises(ValueError, match="protected"):
                rr.alpha_diagnostics(reg, exp.id, segment=seg)
            with pytest.raises(ValueError, match="protected"):
                rr.portfolio_diagnostics(reg, exp.id, segment=seg)
        assert status in (reg.report(exp.id)["holdout"]["status"] or "")
    with pytest.raises(ValueError, match="segment must be one of"):
        rr.alpha_diagnostics(reg, full.id, segment="holiday")
    assert rr.alpha_diagnostics(reg, lasso.id)["eligibility"] == "applied", "a development run still has its validation segment"


def test_a_second_experiment_is_judged_on_what_it_adds_to_the_first_and_a_mismatch_is_named(world):
    reg, full, lasso, ten, _ = world
    out = rr.alpha_diagnostics(reg, full.id, baseline_id=lasso.id)["incremental"]
    assert out["eligibility"] == "applied" and out["baseline_id"] == lasso.id
    assert out["conditional_ic"]["n_with_ic"] > 50 and out["plain_ic"]["n_with_ic"] >= out["conditional_ic"]["n_with_ic"]
    assert out["blend"]["eligibility"] in ("applied", "insufficient") and any("same dates and assets" in a for a in out["assumptions"])
    pair = out["correlation"]["pairs"][0]
    assert -1.0 <= pair["return_correlation"] <= 1.0 and pair["aligned_dates"] > 100 and set(pair["dropped_dates"]) == {full.id, lasso.id} and 0.0 <= pair["exposure_overlap"] <= 1.0
    assert -1.0 <= pair["prediction_rank_correlation"] <= 1.0 and set(out["correlation"]["concentration"]) == {full.id, lasso.id} and "factor overlap is not reported" in out["correlation"]["limitations"][0]
    assert pair["blend"]["sharpe_blend"] is not None and pair["blend"]["diversification_ratio"] >= 1.0 - 1e-9, "a blend of two imperfectly correlated books never diversifies less than 1"
    bad = rr.alpha_diagnostics(reg, full.id, baseline_id=ten.id)["incremental"]
    assert bad["eligibility"] == "not_comparable" and "10 sessions ahead" in bad["reason"]


def test_the_portfolio_diagnostics_describe_the_book_the_experiment_held(world):
    reg, full, *_ = world
    out = rr.portfolio_diagnostics(reg, full.id, max_dates=10)
    assert out["eligibility"] == "applied" and out["book"]["eligibility"] == "applied"
    folder = reg.root / "experiments" / full.id
    pos = pd.read_parquet(folder / "positions.parquet")
    seg = pd.read_parquet(folder / "returns.parquet")["segment"]
    val = pos[pos.index.isin(seg[seg == "validation"].index)]
    live = val[val.abs().sum(axis=1) > 1e-12]
    assert out["book"]["gross"]["mean"] == pytest.approx(float(live.abs().sum(axis=1).mean()), abs=1e-9) and out["book"]["days_with_a_book"] == len(live)
    est = out["risk_models"]["estimators"]
    assert len(est) == 4 and all(v["windows"] > 0 for v in est.values()) and all(v["failed"] == 0 for v in est.values())
    assert all(v["worst_condition"] is not None for v in est.values())
    rc = out["risk_contributions"]
    assert rc["estimator"] in est and rc["fit"]["units"] == "per day" and abs(sum(r["share"] for r in rc["table"]) - 1.0) < 1e-6, "risk shares sum to one"
    assert "not a selection of an estimator" in rc["chosen_because"]
    assert json.dumps(out, allow_nan=False)


def test_the_allocation_sensitivity_runs_for_a_return_forecast_and_is_not_applicable_to_a_rank_target(world):
    reg, full, _, _, rank = world
    al = rr.portfolio_diagnostics(reg, full.id, max_dates=6)["allocation"]
    assert al["eligibility"] == "applied" and al["holding_days"] == 5 and al["grid_hash"] and len(al["points"]) == 54 and al["failed"] == 0
    assert al["base_status"] == "optimal" and al["summary"]["max_weight_change"] >= 0.0 and "absolute expected return" in al["interpretation"]
    assert {p["status"] for p in al["points"]} == {"optimal"}
    na = rr.portfolio_diagnostics(reg, rank.id, max_dates=6)["allocation"]
    assert na["eligibility"] == "not_applicable" and "not a return forecast in return units" in na["reason"]
