"""Release 7, P2-12: meta-labels built from out-of-sample base signals and settled outcomes that include the declared costs.

The labels are recomputed here by hand from the saved predictions and the prices (a second implementation), the purge and the nesting are read off the lineage of every secondary model, and the complete
base-plus-filter strategy is compared with the base signal on the validation days both have.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml import comparison as cp, ensembles as en, experiments as xp, registry as rg
from src.ml.targets import forward_return
from tests.ml_helpers import make_bundle

BASE = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}},
        "signal": {"method": "top_n_long_short", "n": 3, "holding_period": 1}, "backtest": {"cost_bps": 5.0, "slippage_bps": 2.0, "rebalance": "weekly"}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    reg = rg.Registry(tmp_path_factory.mktemp("meta") / "ml")
    bundle = make_bundle(10, 1150, seed=97, alpha=1.0)
    base = xp.run_experiment({**BASE, "name": "primary-audit"}, bundle=bundle, registry=reg, explain=False, robustness=False)
    cfg, ds, train, _ = xp.load_run(reg, base.id)
    return reg, base, cfg, ds, train


def reference_hit_rate(ds, train, cost: float) -> float:
    """The share of the primary's bets that earned more than the cost of entering and leaving them, from the saved predictions and the prices (not from the module under test)."""
    fwd = forward_return(ds.prices, ds.horizon, ds.lag).reindex(index=ds.dates, columns=ds.assets)
    pred = train.predictions
    d, a = pred.index.get_level_values(0), pred.index.get_level_values(1)
    f = fwd.to_numpy()[ds.dates.get_indexer(d), ds.assets.get_indexer(a)] if hasattr(ds.assets, "get_indexer") else fwd.to_numpy()[ds.dates.get_indexer(d), pd.Index(ds.assets).get_indexer(a)]
    p = pred.to_numpy()
    ok = np.isfinite(f) & (p != 0)
    won = (np.sign(p[ok]) * f[ok] - 2.0 * cost) > 0
    return float(won.mean())


def test_the_labels_are_the_settled_outcomes_net_of_the_declared_cost_and_match_a_second_implementation(world):
    reg, base, cfg, ds, train = world
    cost = (cfg["backtest"]["cost_bps"] + cfg["backtest"]["slippage_bps"]) / 1e4
    for c in (0.0, cost, 0.01):
        meta = en.meta_label_scores(ds, train, "logistic", None, 200, 0, c)
        assert meta["label_cost"] == c and meta["primary_hit_rate"] == pytest.approx(reference_hit_rate(ds, train, c), abs=1e-12), f"cost {c}"
    free, declared, dear = (en.meta_label_scores(ds, train, "logistic", None, 200, 0, c)["primary_hit_rate"] for c in (0.0, cost, 0.01))
    assert free > declared > dear, "a bet has to earn back what it cost to enter and leave it: fewer bets are right as the cost rises"


def test_every_secondary_model_was_fitted_on_purged_settled_bets_from_folds_fitted_before_them(world):
    reg, base, cfg, ds, train = world
    meta = en.meta_label_scores(ds, train, "logistic", None, 200, 0, 7e-4)
    purge = ds.lag + ds.horizon
    assert meta["n_models"] >= 3 and len(meta["lineage"]) == meta["n_models"]
    previous, prev_rows = None, 0
    for block in meta["lineage"]:
        cutoff = pd.Timestamp(block["meta_fit_cutoff"])
        last_decision, label_end = pd.Timestamp(block["meta_train_last_decision"]), pd.Timestamp(block["meta_label_end_max"])
        assert block["labels_settled_at_cutoff"] is True and label_end <= cutoff, "every training label had settled before the block it scores began"
        assert ds.dates.get_loc(last_decision) + purge <= ds.dates.get_loc(cutoff), "the last training decision is a whole label horizon (plus the lag) before the cutoff"
        assert block["producer_fit_cutoffs"] and max(pd.Timestamp(c) for c in block["producer_fit_cutoffs"]) <= cutoff, "the base predictions the secondary model learned from came from folds fitted before the cutoff"
        assert "out of sample" in block["base_predictions"] and block["producer_lineage"] == "recorded"
        if previous is not None:
            assert cutoff > previous and block["meta_train_rows"] >= prev_rows, "nested chronology: each later model sees at least what the earlier one saw"
        previous, prev_rows = cutoff, block["meta_train_rows"]


def test_the_complete_base_plus_filter_strategy_is_evaluated_against_the_base_and_its_lineage_is_kept(world):
    reg, base, cfg, ds, train = world
    child, meta = en.meta_label_experiment(reg, base.id, "logistic", min_rows=200)
    assert meta["label_cost"] == pytest.approx(7e-4) and meta["lineage"] and meta["coverage"] > 0.2
    vs = meta["versus_base"]
    assert vs["eligibility"] == "applied" and vs["baseline"] == base.id and vs["candidate"] == child.id and vs["segment"] == "validation"
    assert vs["metrics"]["net_sharpe"]["baseline"] is not None and vs["metrics"]["net_sharpe"]["candidate"] is not None and "not a controlled one-change test" in vs["note"]
    again = cp.compare_saved(reg, base.id, child.id)
    assert again["metrics"]["net_sharpe"]["difference"]["value"] == vs["metrics"]["net_sharpe"]["difference"]["value"], "the comparison is reproducible from the two saved experiments"
    rep = reg.report(child.id)
    assert rep["config"]["signal"]["method"] == "proportional" and "meta-labelled version of" in rep["config"]["notes"] and base.id in rep["config"]["notes"]
    row = reg.get_experiment(child.id)
    assert row.get("parent_id") == base.id or base.id in str(row), "the child records its parent"
    legacy, legacy_meta = en.meta_label_experiment(reg, base.id, "logistic", min_rows=200, include_costs=False)
    assert legacy_meta["label_cost"] == 0.0, "the earlier, cost-free labelling is still available, and says it was used"
