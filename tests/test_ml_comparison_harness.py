"""Release 6 (and 7): the frozen-comparison harness.

The statistics are checked on series built with a known difference, the card rules by refusing what they must refuse, and one real card is run end to end on a small synthetic market: development data only,
matched arms, the verdict and its reason stored, negative findings kept, and the second call a no-op.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml import comparison as cp, registry as rg
from tests.ml_helpers import make_bundle

BASE = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}}}


@pytest.fixture
def reg(tmp_path):
    return rg.Registry(tmp_path / "ml")


def card(**over):
    kw = dict(hypothesis="a market-neutral target ranks assets better than the raw return", family="target", baseline=BASE,
              candidates=[{"dataset": {"target": "excess_ret_5"}}], study_id="study-cmp", one_change_description="the target: raw forward return to the return minus the equal-weight benchmark's")
    kw.update(over)
    return cp.new_card(**kw)


# ------------------------------------------------------------------------------------------------------------------ the card rules
def test_the_card_refuses_what_a_controlled_comparison_must_not_contain():
    with pytest.raises(cp.ComparisonError, match="unknown experiment family"):
        card(family="vibes")
    with pytest.raises(cp.ComparisonError, match="changes nothing"):
        card(candidates=[{}])
    with pytest.raises(cp.ComparisonError, match=r"backtest\.cost_bps.*does not allow"):
        card(candidates=[{"dataset": {"target": "excess_ret_5"}, "backtest": {"cost_bps": 1.0}}])
    with pytest.raises(cp.ComparisonError, match="hypothesis and the one change"):
        card(hypothesis="  ")
    with pytest.raises(cp.ComparisonError, match="at least one candidate"):
        card(candidates=[])
    with pytest.raises(cp.ComparisonError, match="primary metric"):
        card(primary_metric="sortino")
    with pytest.raises(Exception, match="k"):
        card(parsimony={"rule": "one_se", "complexity": "n_features"})
    assert cp.diff_paths({"a": {"b": 1, "c": 2}, "name": "x"}, {"a": {"b": 1, "c": 3}, "name": "y", "d": 1}) == ["a.c", "d"]


def test_the_card_carries_every_field_of_the_protocol_and_its_identity_is_its_content(reg):
    c = card()
    for field in ("hypothesis", "study_id", "baseline_version", "candidate_version", "one_change_description", "execution_cost_risk_capital_assumptions", "primary_metric", "secondary_metrics", "selection_rule",
                  "uncertainty_method", "multiple_testing_scope", "compute_search_budget", "minimum_coverage", "robustness_grid", "final_evaluation_policy"):
        assert field in c, field
    assert c["primary_metric"] == "ic" and c["candidates"][0]["changed"] == ["dataset.target"] and c["card_id"].startswith("cmp-") and "none: this card compares on development data" in c["final_evaluation_policy"]
    assert card()["card_id"] == c["card_id"] and card(min_effect=0.01)["card_id"] != c["card_id"], "the same declaration is the same card; changing a threshold makes another"
    frozen = cp.freeze(reg, c)
    assert frozen["already_frozen"] is False and frozen["frozen_at"] and cp.freeze(reg, c)["already_frozen"] is True
    full = cp.get(reg, c["card_id"])
    for field in ("frozen_at", "results", "negative_findings", "gate_decision", "evidence_artifacts"):
        assert field in full
    assert full["state"] == "frozen" and full["results"] is None and full["gate_decision"]["status"] == "not_evaluated"
    with pytest.raises(cp.ComparisonError, match="unknown comparison card"):
        cp.get(reg, "cmp-nope")


def test_holm_adjustment_by_hand():
    assert cp.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06]), "ordered 0.01, 0.03, 0.04 times 3, 2, 1, made monotone"
    assert cp.holm([0.5]) == [0.5] and cp.holm([None, 0.2, 0.01]) == [None, 0.2, 0.02]
    assert cp.holm([0.3, 0.4, 0.45]) == pytest.approx([0.9, 0.9, 0.9]), "0.3 x 3 = 0.9, then 0.4 x 2 = 0.8 and 0.45 x 1 are lifted to the running maximum"
    assert max(cp.holm([0.6, 0.7, 0.9])) == 1.0, "never above one"


# ------------------------------------------------------------------------------------------------------------------ the paired statistics
def arm(shift_ic=0.0, shift_ret=0.0, seed=0, n=300, cover=0.95, noise=0):
    """A development arm: what both arms share (from ``seed``), a known shift, and noise of its own (from ``noise``) so that the paired difference is not exactly constant."""
    rng, own = np.random.default_rng(seed), np.random.default_rng(10_000 + noise)
    ix = pd.bdate_range("2020-01-01", periods=n)
    common_ic, common_ret = rng.normal(0.02, 0.05, n), rng.normal(0.0004, 0.01, n)
    return {"ic": pd.Series(common_ic + shift_ic + own.normal(0, 0.01, n), index=ix), "ic_own": pd.Series(common_ic, index=ix),
            "net": pd.Series(common_ret + shift_ret + own.normal(0, 0.002, n), index=ix), "turnover": pd.Series(0.2, index=ix), "costs": pd.Series(0.0001, index=ix), "coverage": cover, "dev_dates": n}


def test_a_known_gain_is_supported_a_loss_is_worse_and_no_difference_is_inconclusive():
    base = arm(0.0, 0.0, seed=1, noise=0)
    results = []
    for k, (shift_ic, shift_ret) in enumerate(((0.03, 0.0008), (-0.03, -0.0008), (0.0, 0.0))):
        cand = arm(shift_ic, shift_ret, seed=1, noise=k + 1)
        results.append({"name": f"c{shift_ic}", "status": "compared", "metrics": cp.compare_series(base, cand, label_horizon=5, hold=5, n_boot=400, seed=3, scope="test")})
    for primary in ("ic", "net_sharpe"):
        c = {"primary_metric": primary, "settings": {"level": 0.05, "min_effect": 0.0}}
        out = cp.decide(c, [dict(r) for r in results])
        assert [o["verdict"] for o in out] == ["supported_on_development", "worse", "inconclusive"], (primary, [o["reason"] for o in out])
    d = results[0]["metrics"]["ic"]
    assert d["difference"]["value"] == pytest.approx(0.03, abs=0.002) and d["aligned"] == 300 and d["dropped"] == {"only_baseline": 0, "only_candidate": 0}
    assert d["difference"]["settings"]["lags"] >= 4, "five-session labels overlap: the Newey-West lag is at least horizon - 1"
    assert results[0]["metrics"]["ic_own_horizon"]["note"].startswith("each arm's IC against its own horizon")


def test_the_minimum_effect_and_the_holm_correction_both_have_to_be_met():
    base = arm(0.0, 0.0, seed=4, noise=0)
    cand = arm(0.004, 0.0, seed=4, noise=1)                                                             # a real but small gain
    res = [{"name": "small", "status": "compared", "metrics": cp.compare_series(base, cand, label_horizon=1, hold=1, n_boot=300, seed=1, scope="t")}]
    big_bar = cp.decide({"primary_metric": "ic", "settings": {"level": 0.05, "min_effect": 0.01}}, [dict(r) for r in res])[0]
    assert big_bar["verdict"] == "inconclusive" and "below the declared minimum 0.01" in big_bar["reason"], big_bar["reason"]
    low_bar = cp.decide({"primary_metric": "ic", "settings": {"level": 0.05, "min_effect": 0.0}}, [dict(r) for r in res])[0]
    assert low_bar["verdict"] == "supported_on_development"
    many = [dict(res[0], name=f"c{i}") for i in range(40)]                                                # forty looks at the same small gain: the adjusted p-value is forty times larger
    adj = cp.decide({"primary_metric": "ic", "settings": {"level": 0.05, "min_effect": 0.0}}, many)
    assert adj[0]["adjusted_p_value"] == pytest.approx(min(1.0, 40 * res[0]["metrics"]["ic"]["difference"]["p_value"]), rel=1e-9)
    assert adj[0]["verdict"] == ("supported_on_development" if adj[0]["adjusted_p_value"] < 0.05 else "inconclusive")


def test_a_difference_that_cannot_be_estimated_is_inconclusive_and_says_so_not_zero():
    base, cand = arm(seed=5, n=300, noise=0), arm(seed=5, n=300, noise=1)
    cand["ic"] = cand["ic"].iloc[:10]                                                                   # too few aligned dates for a dependence-aware estimate
    m = cp.compare_series(base, cand, label_horizon=1, hold=1, n_boot=300, seed=1, scope="t")
    assert m["ic"]["difference"]["eligibility"] == "insufficient" and m["ic"]["difference"]["value"] is None and m["ic"]["dropped"]["only_baseline"] == 290
    out = cp.decide({"primary_metric": "ic", "settings": {"level": 0.05, "min_effect": 0.0}}, [{"name": "x", "status": "compared", "metrics": m}])[0]
    assert out["verdict"] == "inconclusive" and "could not be estimated" in out["reason"]
    failed = cp.decide({"primary_metric": "ic", "settings": {"level": 0.05, "min_effect": 0.0}}, [{"name": "y", "status": "failed", "reason": "boom"}])[0]
    assert failed["verdict"] == "not_compared" and failed["reason"] == "boom"


# ------------------------------------------------------------------------------------------------------------------ a card, run
@pytest.fixture(scope="module")
def bundle():
    return make_bundle(10, 1150, seed=41, alpha=1.0)


def test_a_card_is_run_once_on_development_data_with_matched_arms_and_every_outcome_is_stored(reg, bundle):
    c = card(parsimony={"rule": "one_se", "k": 1.0, "complexity": "n_features"})
    cp.freeze(reg, c)
    out = cp.run(reg, c["card_id"], bundle=bundle)
    assert out["reused"] is False and out["state"] == "run" and out["gate_decision"]["status"] == "not_evaluated" and "nothing here promotes" in out["gate_decision"]["reason"]
    res = out["results"]
    assert res["segment"] == "validation" and res["label_horizon"] == 5 and len(res["candidates"]) == 1
    cand = res["candidates"][0]
    assert cand["status"] == "compared" and cand["verdict"] in ("supported_on_development", "worse", "inconclusive") and cand["reason"]
    d = cand["metrics"]["ic"]["difference"]
    assert d["eligibility"] == "applied" and d["interval"][0] <= d["value"] <= d["interval"][1] and cand["metrics"]["ic"]["aligned"] > 100
    assert cand["adjusted_p_value"] == pytest.approx(d["p_value"]), "one candidate: Holm leaves the p-value as it is"
    assert cand["manifest"]["fold_manifest_hash"] and out["observed"]["baseline"]["fold_manifest_hash"] == cand["manifest"]["fold_manifest_hash"], "identical folds"
    assert cand["manifest"]["feature_schema_hash"] == out["observed"]["baseline"]["feature_schema_hash"]
    assert res["parsimony"]["eligibility"] == "applied" and res["parsimony"]["winner"] in ("baseline", cand["name"])
    if cand["verdict"] != "supported_on_development":
        assert any(f["candidate"] == cand["name"] and f["what"] == cand["verdict"] for f in out["negative_findings"]), "a result that did not go the candidate's way stays in the record"
    ids = out["evidence_artifacts"]["experiments"]
    assert set(ids) == {"baseline", cand["name"]}
    for exp_id in ids.values():
        assert reg.report(exp_id)["holdout"]["status"] == "development", "the test and forward outcomes were never predicted"
    n_before = len(reg.list_experiments())
    again = cp.run(reg, c["card_id"], bundle=bundle)
    assert again["reused"] is True and len(reg.list_experiments()) == n_before, "a second call runs nothing: another look needs another card"
    assert cp.get(reg, c["card_id"])["results"]["candidates"][0]["verdict"] == cand["verdict"]


def test_arms_on_different_folds_are_not_compared_and_a_failing_candidate_is_recorded(reg, bundle, monkeypatch):
    nogroups = make_bundle(10, 1150, seed=41, alpha=1.0)
    nogroups.group = {}                                                                                  # no sectors: the sector-relative target cannot be built
    bundle = nogroups
    c = card(candidates=[{"dataset": {"target": "excess_ret_5"}}, {"dataset": {"target": "sector_ret_5"}}], study_id="study-cmp2")
    cp.freeze(reg, c)
    real = cp.manifest
    calls = {"n": 0}

    def fake(exp):
        m = real(exp)
        calls["n"] += 1
        if calls["n"] == 2:                                                                             # the first candidate: shift one fold boundary
            m["fold_manifest"][0][4] = "2099-01-01"
        return m

    monkeypatch.setattr(cp, "manifest", fake)
    out = cp.run(reg, c["card_id"], bundle=bundle)
    first, second = out["results"]["candidates"]
    assert first["status"] == "not_comparable" and "fold boundaries differ" in first["reason"] and first["verdict"] == "not_compared"
    assert second["status"] == "failed" and "group map" in second["reason"]
    assert {f["what"] for f in out["negative_findings"]} >= {"not_comparable", "failed"}


def test_a_feature_card_compares_the_simple_motivated_sets_first_and_records_what_the_selection_chose(reg, bundle):
    stable = {"model": {"kind": "stable_select", "params": {"base": "ridge", "alpha": 10.0, "pi": 0.5, "top_k": 4, "block_days": 21, "subsamples": 20}}}
    c = card(family="feature", candidates=[stable, {"dataset": {"feature_groups": ["momentum"]}}], study_id="study-feat", hypothesis="a stability-selected set adds rank IC over the existing features",
             one_change_description="the feature set", parsimony={"rule": "one_se", "k": 1.0, "complexity": "n_features"})
    assert [c["candidates"][i]["change"] for i in (0, 1)] == [{"dataset": {"feature_groups": ["momentum"]}}, stable], "the single-group candidate is first although it was declared second"
    assert c["ladder"]["simple_first"] is True and c["ladder"]["has_low_complexity_candidate"] is True and c["ladder"]["ranks"] == [0, 2] and c["family"] == "feature"
    assert card(family="feature", candidates=[stable], study_id="study-feat")["ladder"]["has_low_complexity_candidate"] is False
    with pytest.raises(cp.ComparisonError, match="does not allow"):
        card(family="feature", candidates=[{"dataset": {"target": "excess_ret_5"}}])
    cp.freeze(reg, c)
    out = cp.run(reg, c["card_id"], bundle=bundle)
    simple, selected = out["results"]["candidates"]
    assert simple["status"] == "compared" and selected["status"] == "compared"
    assert simple["n_features"] < out["results"]["baseline"]["n_features"], "one group has fewer features than two"
    assert selected["kind"] == "stable_select" and selected["selection_stability"] is not None and selected["selection_stability"]["folds_selected"] >= 1
    assert 1 <= selected["n_features"] <= out["results"]["baseline"]["n_features"] and out["results"]["parsimony"]["eligibility"] == "applied"
    assert out["results"]["parsimony"]["rule"]["complexity"] == "n_features" and out["results"]["parsimony"]["winner"] in ("baseline", simple["name"], selected["name"])
    assert cp.preset_changes("feature", BASE, ["group:momentum", "stable_select"]) == [{"dataset": {"feature_groups": ["momentum"]}}, {"model": {"kind": "stable_select", "params": {"base": "ridge", "alpha": 10.0}}}]
    assert cp.preset_changes("horizon", BASE, [1, 5, 20]) == [{"dataset": {"target": "fwd_ret_1", "horizon": 1}, "signal": {"holding_period": 1}}, {"dataset": {"target": "fwd_ret_20", "horizon": 20}, "signal": {"holding_period": 20}}]
    with pytest.raises(cp.ComparisonError, match="not a target preset"):
        cp.preset_changes("target", BASE, ["vibes"])
    with pytest.raises(cp.ComparisonError, match="left no candidate"):
        cp.preset_changes("horizon", BASE, [5])


# ------------------------------------------------------------------------------------------------------------------ Release 7: arms that act after the predictions, and saved pairs
BASE1 = {**BASE, "signal": {"method": "top_n_long_short", "n": 3, "holding_period": 1}, "backtest": {"cost_bps": 5.0, "slippage_bps": 2.0, "rebalance": "weekly"}}


def test_a_card_that_changes_only_the_position_rule_re_runs_the_baselines_own_predictions(reg, bundle):
    from src.ml import experiments as xp

    c = card(family="rebalancing", baseline=BASE1, candidates=[{"signal": {"no_trade_margin": 2.0}}], study_id="study-nt", hypothesis="a no-trade region saves more cost than the alpha it gives up",
             one_change_description="the rebalance rule: trade only when the gain exceeds the cost with a margin")
    assert c["candidates"][0]["changed"] == ["signal.no_trade_margin"] and cp._post_model(c["candidates"][0]["changed"]) is True
    cp.freeze(reg, c)
    out = cp.run(reg, c["card_id"], bundle=bundle)
    cand = out["results"]["candidates"][0]
    assert cand["status"] == "compared" and cand["arm"] == "re-run of the baseline's saved predictions"
    ids = out["evidence_artifacts"]["experiments"]
    pb, pc = (xp.load_run(reg, ids[k])[2].predictions for k in ("baseline", cand["name"]))
    pd.testing.assert_series_equal(pb, pc), "the two alternatives are evaluated on the very same forecasts"
    assert cand["manifest"]["fold_manifest_hash"] == out["observed"]["baseline"]["fold_manifest_hash"]
    assert cand["metrics"]["turnover"]["candidate"] < cand["metrics"]["turnover"]["baseline"], "the no-trade region traded less than the plain schedule"
    assert cand["metrics"]["net_sharpe"]["difference"]["eligibility"] in ("applied", "insufficient") and cand["metrics"]["costs"]["candidate"] <= cand["metrics"]["costs"]["baseline"]
    assert reg.report(ids[cand["name"]])["overlays"]["no_trade"]["decisions"] > 10, "the overlay's own log is on the candidate's report"
    retrain = card(family="rebalancing", baseline=BASE1, candidates=[{"signal": {"no_trade_margin": 2.0}}], study_id="study-nt")["candidates"][0]
    assert retrain["name"] != c["baseline"]["name"], "an arm has its own name even when the change leaves the generated name alone"


def test_a_candidate_that_changes_the_model_is_retrained_not_re_run(reg, bundle):
    c = card(candidates=[{"dataset": {"target": "excess_ret_5"}}])
    assert cp._post_model(c["candidates"][0]["changed"]) is False
    assert cp._post_model([]) is False and cp._post_model(["signal.n", "backtest.cost_bps", "execution.mode"]) is True and cp._post_model(["signal.n", "model.kind"]) is False


def test_two_saved_experiments_are_compared_on_their_shared_validation_dates_and_a_mismatch_is_refused(reg, bundle):
    from src.ml import ensembles as en, experiments as xp

    base = xp.run_experiment({**BASE1, "name": "primary"}, bundle=bundle, registry=reg, explain=False, robustness=False)
    child, meta = en.meta_label_experiment(reg, base.id, "logistic", min_rows=200)
    vs = meta["versus_base"]
    assert vs["eligibility"] == "applied" and vs["baseline"] == base.id and vs["candidate"] == child.id and vs["segment"] == "validation"
    assert vs["metrics"]["net_sharpe"]["difference"]["eligibility"] in ("applied", "insufficient") and vs["metrics"]["ic"]["aligned"] > 100
    assert "not a controlled one-change test" in vs["note"]
    assert meta["label_cost"] == pytest.approx(7e-4) and meta["lineage"], "the declared cost and the lineage of each secondary model are on the summary"
    self_pair = cp.compare_saved(reg, base.id, base.id)
    assert self_pair["eligibility"] == "applied"
    d = self_pair["metrics"]["ic"]["difference"]
    assert d["eligibility"] == "insufficient" and d["value"] is None, "a strategy against itself differs by a constant zero: the difference has no variance and is not estimated, not reported as an effect"
    assert self_pair["metrics"]["net_sharpe"]["difference"]["value"] in (0.0, None)
    other = xp.run_experiment({**BASE1, "name": "other-horizon", "dataset": {**BASE1["dataset"], "target": "fwd_ret_10", "horizon": 10}, "signal": {**BASE1["signal"], "holding_period": 1}}, bundle=bundle, registry=reg,
                              explain=False, robustness=False)
    mismatch = cp.compare_saved(reg, base.id, other.id)
    assert mismatch["eligibility"] == "not_comparable" and "horizon and lag differ" in mismatch["reason"]


def test_a_meta_label_must_earn_the_declared_cost_to_count_as_right(reg, bundle):
    from src.ml import ensembles as en, experiments as xp

    base = xp.run_experiment({**BASE1, "name": "primary-cost"}, bundle=bundle, registry=reg, explain=False, robustness=False)
    cfg, ds, train, _ = xp.load_run(reg, base.id)
    free = en.meta_label_scores(ds, train, "logistic", None, 200, 0, cost=0.0)
    dear = en.meta_label_scores(ds, train, "logistic", None, 200, 0, cost=0.01)
    assert free["label_cost"] == 0.0 and dear["label_cost"] == 0.01
    assert dear["primary_hit_rate"] < free["primary_hit_rate"], "a bet has to earn back what it cost to enter and leave it: fewer of them are right"
