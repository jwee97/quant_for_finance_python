"""Release 5, P1-23: evaluation over predeclared seeds."""

import numpy as np
import pytest

from src.ml import multiseed as ms, registry as rg, research_controls as rc
from tests.ml_helpers import make_bundle

OVER = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 500, "refit_every": 252}, "model": {"kind": "random_forest", "params": {"n_estimators": 15, "max_depth": 3}}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return make_bundle(8, 1500, seed=51, alpha=1.0, macro=False), rg.Registry(tmp_path_factory.mktemp("ml"))


def test_every_predeclared_seed_is_run_on_the_same_windows_and_reported_with_the_two_uncertainties_apart(world):
    bundle, reg = world
    out = ms.run_seeds(OVER, [1, 2, 3, 4], bundle=bundle, registry=reg, stop_after="validation")
    assert out["declaration"]["seeds"] == [1, 2, 3, 4] and len(out["rows"]) == 4 and len({r["experiment_id"] for r in out["rows"]}) == 4 and out["eligibility"] == "applied"
    assert out["same_market_windows"] and out["selection"].startswith("median over the predeclared seeds")
    assert len({r["n_days"] for r in out["rows"]}) == 1, "identical market windows: every seed is scored on the same days"
    s = out["summary"]
    assert s["min"] <= s["median"] <= s["max"] and s["range"] == pytest.approx(s["max"] - s["min"]) and s["std_across_seeds"] > 0, "a model that uses its seed does not give the same result for every seed"
    assert s["mean_time_series_se"] > 0 and s["seed_spread_to_time_series_se"] == pytest.approx(s["std_across_seeds"] / s["mean_time_series_se"])
    assert all(r["sharpe_hac_interval"] and r["time_series_se_per_period"] for r in out["rows"]) and out["flag"]
    assert rc.raw_looks(reg, "u-etf") >= 4, "each seed is a counted look"


def test_a_seed_is_never_chosen_by_its_result_and_too_few_or_repeated_seeds_are_refused(world):
    bundle, reg = world
    with pytest.raises(ValueError, match="never selected by its result"):
        ms.run_seeds(OVER, [1, 2, 3], select="best", bundle=bundle)
    with pytest.raises(ValueError, match="at least 3"):
        ms.declare([1, 2])
    with pytest.raises(ValueError, match="distinct"):
        ms.declare([1, 1, 2, 3])
    assert ms.declare([3, 1, 2])["hash"] == ms.declare([1, 2, 3])["hash"], "the declaration is the set of seeds, not their order"
    assert np.isfinite(len(ms.declare([5, 6, 7])["hash"]))
