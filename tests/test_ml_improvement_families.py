"""Release 7: the experiment families of spec section 10.2 beyond Release 6, combinations and ablations, and the development-selected baseline.

Presets are checked as the changes they must produce, the cards by what they refuse, and the runs end to end on a small market: arms that act after the predictions re-run the baseline's own forecasts, a
risk control is read against a volatility-matched drawdown, a combination only exists for components that were each compared on their own, and its tests join their family.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml import comparison as cp, registry as rg
from tests.ml_helpers import make_bundle

BASE = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}},
        "signal": {"method": "top_n_long_short", "n": 3, "holding_period": 1}, "backtest": {"cost_bps": 5.0, "slippage_bps": 2.0, "rebalance": "weekly"}}


@pytest.fixture(scope="module")
def bundle():
    return make_bundle(10, 1150, seed=43, alpha=1.0)


@pytest.fixture
def reg(tmp_path):
    return rg.Registry(tmp_path / "ml")


def card(family, change, study="study-fam", **kw):
    kw.setdefault("hypothesis", f"the {family} change helps")
    kw.setdefault("one_change_description", f"the {family} change")
    return cp.new_card(family=family, baseline=BASE, candidates=change, study_id=study, **kw)


def run_card(reg, bundle, c):
    cp.freeze(reg, c)
    return cp.run(reg, c["card_id"], bundle=bundle)


# ------------------------------------------------------------------------------------------------------------------ presets
def test_the_presets_are_the_changes_they_say_and_refuse_what_they_cannot_make():
    assert cp.preset_changes("rebalancing", BASE, ["no_trade:1.5", "trade_filter:0.5"]) == [{"signal": {"no_trade_margin": 1.5}}, {"signal": {"trade_filter_k": 0.5}}]
    assert cp.preset_changes("risk_control", BASE, ["regime_exposure:vol_quantile", "regime_exposure:hmm:2"]) == [{"signal": {"regime_exposure": "vol_quantile"}}, {"signal": {"regime_exposure": "hmm", "regime_states": 2}}]
    assert cp.preset_changes("portfolio", BASE, ["calibrated_sizing"]) == [{"signal": {"method": "calibrated_sizing", "holding_period": 1}}]
    assert cp.preset_changes("execution", BASE, ["realistic"], {"capital": 5e5, "max_participation": 0.05, "other": 1}) == [{"execution": {"mode": "realistic_cash_equity", "capital": 5e5, "max_participation": 0.05}}]
    assert cp.preset_changes("ensemble", BASE, ["weighting:equal", "weighting:stacked_regime"], {"members": "ridge,lasso"}) == [
        {"model": {"kind": "model_ensemble", "params": {"members": "ridge,lasso", "weighting": w}}} for w in ("equal", "stacked_regime")]
    for fam, bad, opts, msg in (("rebalancing", "no_trade:abc", {}, "must be a number"), ("rebalancing", "vibes", {}, "not a rebalancing preset"), ("risk_control", "smoothed:hmm", {}, "not a risk_control preset"),
                                ("ensemble", "weighting:magic", {"members": "ridge,lasso"}, "weighting must be one of"), ("ensemble", "weighting:equal", {}, "needs the members"),
                                ("execution", "magic", {}, "not a execution preset")):
        with pytest.raises(cp.ComparisonError, match=msg):
            cp.preset_changes(fam, BASE, [bad], opts)
    assert set(cp.FAMILIES) >= {"rebalancing", "risk_control", "ensemble", "execution", "portfolio", "combination"}


def test_each_family_accepts_its_own_changes_and_refuses_anothers():
    ok = {"rebalancing": [{"signal": {"no_trade_margin": 1.0}}], "risk_control": [{"signal": {"regime_exposure": "vol_quantile"}}], "execution": [{"execution": {"mode": "realistic_cash_equity"}}],
          "ensemble": [{"model": {"kind": "model_ensemble", "params": {"members": "ridge,lasso", "weighting": "equal"}}}]}
    for fam, change in ok.items():
        assert card(fam, change)["family"] == fam
    with pytest.raises(cp.ComparisonError, match="does not allow"):
        card("risk_control", [{"model": {"kind": "lasso", "params": {}}}])
    with pytest.raises(cp.ComparisonError, match="does not allow"):
        card("ensemble", [{"signal": {"no_trade_margin": 1.0}}])
    with pytest.raises(cp.ComparisonError, match="combination card is built from components"):
        card("combination", [{"signal": {"no_trade_margin": 1.0}}])
    derived = card("rebalancing", [{"signal": {"no_trade_margin": 1.0}}])["candidates"][0]
    assert derived["changed"] == ["signal.no_trade_margin"] and "cost_bps" in " ".join(derived["derived"]), "the settings the configuration fills in are listed apart from the one that was declared"


# ------------------------------------------------------------------------------------------------------------------ matched risk
def test_a_drawdown_gain_that_is_only_a_smaller_position_is_called_that():
    rng = np.random.default_rng(3)
    ix = pd.bdate_range("2020-01-01", periods=300)
    base = pd.Series(rng.normal(0.0004, 0.01, 300), index=ix)
    half = cp._risk_matched(base, 0.5 * base)
    assert half["volatility_ratio"] == pytest.approx(0.5) and abs(half["max_drawdown_candidate"]) < abs(half["max_drawdown_baseline"])
    assert half["max_drawdown_candidate_at_baseline_volatility"] == pytest.approx(half["max_drawdown_baseline"], abs=1e-9), "scaled back up it is the baseline's own drawdown: nothing but exposure changed"
    arm = {"name": "half", "status": "compared", "verdict": "supported_on_development", "reason": "x",
           "metrics": {"coverage": {"candidate": 1.0}, "turnover": {"baseline": 1.0, "candidate": 1.0}, "net_sharpe": {"difference": {"eligibility": "applied", "value": 0.1}}, "risk_matched": half}}
    found = cp._negative_findings({"minimum_coverage": 0.8, "primary_metric": "net_sharpe"}, [arm], {})
    assert any(f["what"] == "drawdown_gain_is_lower_exposure" for f in found)
    timing = pd.Series(np.where(base < 0, 0.2 * base, base), index=ix)                                       # a candidate that avoids the bad days is better at the baseline's volatility too
    smart = cp._risk_matched(base, timing)
    assert smart["max_drawdown_candidate_at_baseline_volatility"] > smart["max_drawdown_baseline"]
    assert cp._risk_matched(base.iloc[:30], base.iloc[:30])["eligibility"] == "insufficient"
    assert cp._risk_matched(base, base * 0.0)["eligibility"] == "insufficient"


# ------------------------------------------------------------------------------------------------------------------ cards run end to end
def test_a_risk_control_card_re_runs_the_baselines_forecasts_and_reports_matched_risk(reg, bundle):
    c = card("risk_control", [{"signal": {"regime_exposure": "vol_quantile", "regime_states": 2, "regime_min_train": 100, "regime_min_days": 30}}], study="study-risk", hypothesis="a causal volatility regime lowers drawdown without lowering net Sharpe")
    out = run_card(reg, bundle, c)
    arm = out["results"]["candidates"][0]
    assert arm["status"] == "compared" and arm["arm"] == "re-run of the baseline's saved predictions" and arm["verdict"] in ("supported_on_development", "worse", "inconclusive")
    rm = arm["metrics"]["risk_matched"]
    assert rm["eligibility"] == "applied" and rm["volatility_ratio"] <= 1.0 + 1e-9, "the overlay can only lower the book, so the candidate's volatility is not above the baseline's"
    ids = out["evidence_artifacts"]["experiments"]
    log = reg.report(ids[arm["name"]])["overlays"]["regime_exposure"]
    assert log["n_refits"] >= 1 and log["smoothing"].startswith("none") and out["gate_decision"]["status"] == "not_evaluated"
    assert reg.report(ids["baseline"])["holdout"]["status"] == "development", "the protected segments were not predicted"


def test_a_rebalancing_card_and_an_execution_card_run_on_the_same_forecasts(reg, bundle):
    nt = run_card(reg, bundle, card("rebalancing", [{"signal": {"no_trade_margin": 2.0}}], study="study-reb", hypothesis="a no-trade region saves more cost than the alpha it gives up"))
    a = nt["results"]["candidates"][0]
    assert a["status"] == "compared" and a["metrics"]["turnover"]["candidate"] < a["metrics"]["turnover"]["baseline"]
    ex = run_card(reg, bundle, card("execution", [{"execution": {"mode": "realistic_cash_equity", "capital": 1e6, "max_participation": 0.1}}], study="study-exec", hypothesis="capacity-aware fills change the realisable result"))
    b = ex["results"]["candidates"][0]
    assert b["status"] in ("compared", "failed") and (b["status"] == "compared" and b["arm"].startswith("re-run") or "reason" in b), "a realistic run either compares or says why not"


# ------------------------------------------------------------------------------------------------------------------ combinations and ablations
def test_a_combination_exists_only_for_components_compared_on_their_own_and_its_tests_join_their_family(reg, bundle):
    a = run_card(reg, bundle, card("rebalancing", [{"signal": {"no_trade_margin": 2.0}}], study="study-comb", hypothesis="no-trade region"))
    b = run_card(reg, bundle, card("risk_control", [{"signal": {"regime_exposure": "vol_quantile", "regime_states": 2, "regime_min_train": 100, "regime_min_days": 30}}], study="study-comb", hypothesis="regime exposure"))
    na, nb = a["results"]["candidates"][0]["name"], b["results"]["candidates"][0]["name"]
    parents = [{"card_id": a["card_id"], "candidate": na}, {"card_id": b["card_id"], "candidate": nb}]
    unrun = card("rebalancing", [{"signal": {"no_trade_margin": 3.0}}], study="study-comb", hypothesis="a larger margin")
    cp.freeze(reg, unrun)
    with pytest.raises(cp.ComparisonError, match="has not been run"):
        cp.combination_card(reg, [parents[0], {"card_id": unrun["card_id"], "candidate": unrun["candidates"][0]["name"]}], hypothesis="h", one_change_description="c")
    with pytest.raises(cp.ComparisonError, match="at least two components"):
        cp.combination_card(reg, parents[:1], hypothesis="h", one_change_description="c")
    with pytest.raises(cp.ComparisonError, match="listed twice"):
        cp.combination_card(reg, [parents[0], parents[0]], hypothesis="h", one_change_description="c")
    with pytest.raises(cp.ComparisonError, match="no candidate named"):
        cp.combination_card(reg, [parents[0], {"card_id": b["card_id"], "candidate": "nope"}], hypothesis="h", one_change_description="c")
    combo = cp.combination_card(reg, parents, hypothesis="the two together beat the baseline", one_change_description="no-trade region plus regime exposure")
    assert combo["family"] == "combination" and combo["lineage"]["kind"] == "combination" and [c["candidate"] for c in combo["lineage"]["components"]] == [na, nb]
    assert combo["multiple_testing_scope"]["family_size"] == 3 and "does not reuse the components' findings" in combo["multiple_testing_scope"]["note"]
    assert set(combo["candidates"][0]["changed"]) >= {"signal.no_trade_margin", "signal.regime_exposure"} and combo["baseline_version"] == a["baseline_version"]
    out = run_card(reg, bundle, combo)
    arm = out["results"]["candidates"][0]
    inherited = [x["p_value"] for x in combo["lineage"]["inherited"]]
    own = arm["metrics"]["net_sharpe"]["difference"]["p_value"]
    if own is not None and all(p is not None for p in inherited):
        assert arm["adjusted_p_value"] == pytest.approx(cp.holm([own, *inherited])[0]), "the combination's p-value is adjusted over the family that includes its components' tests"
        assert arm["adjusted_p_value"] >= own - 1e-12
    assert arm["arm"] == "re-run of the baseline's saved predictions"
    ab = cp.ablation_card(reg, parents)
    names = [c["name"] for c in ab["candidates"]]
    assert ab["lineage"]["kind"] == "ablation" and len(set(names)) == 2 and names[0].startswith("without rebalancing:") and names[1].startswith("without risk_control:"), "two components whose cards generated the same name stay distinct"
    assert ab["candidates"][0]["change"]["signal"]["no_trade_margin"] is None and ab["candidates"][1]["change"]["signal"]["regime_exposure"] is None, "a setting the baseline never had is switched off"
    reading = run_card(reg, bundle, ab)["results"]["ablation"]
    assert [r["card_id"] for r in reading] == [a["card_id"], b["card_id"]] and all(r["verdict_of_removing_it"] in ("worse", "supported_on_development", "inconclusive", "not_compared") for r in reading)


def test_components_that_set_the_same_setting_differently_or_against_different_baselines_are_refused(reg, bundle):
    a = run_card(reg, bundle, card("rebalancing", [{"signal": {"no_trade_margin": 1.0}}], study="study-conf", hypothesis="margin 1"))
    b = run_card(reg, bundle, card("rebalancing", [{"signal": {"no_trade_margin": 3.0}}], study="study-conf", hypothesis="margin 3"))
    pa_, pb_ = [{"card_id": x["card_id"], "candidate": x["results"]["candidates"][0]["name"]} for x in (a, b)]
    with pytest.raises(cp.ComparisonError, match="both set signal.no_trade_margin, to different values"):
        cp.combination_card(reg, [pa_, pb_], hypothesis="h", one_change_description="c")
    other = cp.new_card("a different baseline", "risk_control", {**BASE, "backtest": {**BASE["backtest"], "rebalance": "monthly"}}, [{"signal": {"regime_exposure": "vol_quantile"}}], study_id="study-conf2",
                        one_change_description="regime exposure")
    c = run_card(reg, bundle, other)
    with pytest.raises(cp.ComparisonError, match="different baselines"):
        cp.combination_card(reg, [pa_, {"card_id": c["card_id"], "candidate": c["results"]["candidates"][0]["name"]}], hypothesis="h", one_change_description="c")


# ------------------------------------------------------------------------------------------------------------------ the development-selected baseline
def test_the_single_model_development_would_choose_is_the_best_of_the_members_and_is_recorded_as_such(reg, bundle):
    members = [("ridge", {"alpha": 10.0}), ("lasso", {"alpha": 0.05}), ("elasticnet", {"alpha": 0.05})]
    sel = cp.select_baseline(reg, BASE, members, study_id="study-sel", bundle=bundle)
    assert sel["n_candidates"] == 3 and sel["segment"] == "validation" and len(sel["members"]) == 3 and "optimistic" in sel["note"]
    scores = [m["score"] for m in sel["members"]]
    assert all(s is not None for s in scores) and sel["selected"]["score"] == max(scores) and sel["selected"]["kind"] in ("ridge", "lasso", "elasticnet")
    assert all(reg.report(m["experiment_id"])["holdout"]["status"] == "development" for m in sel["members"]), "each member was a development run: the protected segments were not predicted"
    c = cp.new_card("an ensemble beats the development-selected single model", "ensemble", cp.merge(BASE, {"model": {"kind": sel["selected"]["kind"], "params": sel["selected"]["params"]}}),
                    [{"model": {"kind": "model_ensemble", "params": {"members": "ridge,lasso,elasticnet", "weighting": "stacked"}}}], study_id="study-sel", one_change_description="a stacked ensemble of the three",
                    baseline_selection=sel)
    assert c["baseline_selection"]["selected"]["kind"] == sel["selected"]["kind"] and c["card_id"] != cp.new_card(
        "an ensemble beats the development-selected single model", "ensemble", cp.merge(BASE, {"model": {"kind": sel["selected"]["kind"], "params": sel["selected"]["params"]}}),
        [{"model": {"kind": "model_ensemble", "params": {"members": "ridge,lasso,elasticnet", "weighting": "stacked"}}}], study_id="study-sel", one_change_description="a stacked ensemble of the three")["card_id"]
    with pytest.raises(cp.ComparisonError, match="between two and eight"):
        cp.select_baseline(reg, BASE, members[:1], study_id="s", bundle=bundle)
    with pytest.raises(cp.ComparisonError, match="metric must be one of"):
        cp.select_baseline(reg, BASE, members, study_id="s", bundle=bundle, metric="sortino")
