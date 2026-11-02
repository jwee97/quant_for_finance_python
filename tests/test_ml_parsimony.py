"""Release 6, P2-17: the simplicity tie-break, worked by hand and checked by simulation.

Three candidates with known scores and standard errors have a known answer under each rule. The simulation gives five candidates the *same* true score: picking the best observed one is then a coin toss between
them, and a one-standard-error rule should return the simplest far more often. That is a finite Monte Carlo result and is reported with its error.
"""

import pytest

from src.ml import parsimony as pa


def cands(se_diff=None):
    return [{"id": "simple", "score": 0.50, "se": 0.10, "complexity": {"family_rank": 1, "n_features": 3}, "se_diff": se_diff},
            {"id": "medium", "score": 0.58, "se": 0.10, "complexity": {"family_rank": 1, "n_features": 12}},
            {"id": "complex", "score": 0.60, "se": 0.10, "complexity": {"family_rank": 4, "n_features": 40}}]


def test_one_standard_error_rule_by_hand():
    out = pa.choose(cands(), {"rule": "one_se", "k": 1.0, "complexity": "lexicographic"})
    assert out["best_by_score"] == "complex" and out["winner"] == "simple" and out["changed_choice"] is True and out["tie_break_applied"] is True
    row = {r["id"]: r for r in out["table"]}
    assert row["simple"]["se_of_gap"] == pytest.approx(0.1414213, abs=1e-6) and row["simple"]["gap_to_best"] == pytest.approx(0.10) and row["simple"]["eligible"] is True
    assert "independent" in row["simple"]["how"] and row["complex"]["reason"] == "the best score"
    tight = pa.choose(cands(), {"rule": "one_se", "k": 0.5, "complexity": "lexicographic"})
    assert tight["winner"] == "medium", "half a standard error: the gap of 0.10 (0.71 se) is now distinguishable, the gap of 0.02 is not"
    paired = pa.choose(cands(), {"rule": "one_se", "k": 1.0, "complexity": "lexicographic"}) | {}
    cs = cands()
    cs[2]["se_diff"] = {"simple": 0.03, "medium": 0.03}
    assert pa.choose(cs, {"rule": "one_se", "k": 1.0, "complexity": "lexicographic"})["winner"] == "medium", "a paired standard error of 0.03 makes the 0.10 gap three standard errors: not the simple model"
    assert paired["rule"]["k"] == 1.0
    zero = pa.choose(cands(), {"rule": "one_se", "k": 0.0, "complexity": "lexicographic"})
    assert zero["winner"] == "complex" and zero["tie_break_applied"] is False and "best score wins" in zero["note"]


def test_the_complexity_that_orders_candidates_is_the_one_declared():
    cs = [{"id": "few_but_deep", "score": 0.6, "se": 0.1, "complexity": {"family_rank": 6, "n_features": 2}},
          {"id": "many_but_linear", "score": 0.6, "se": 0.1, "complexity": {"family_rank": 1, "n_features": 30}}]
    assert pa.choose(cs, {"rule": "one_se", "k": 1, "complexity": "n_features"})["winner"] == "few_but_deep"
    assert pa.choose(cs, {"rule": "one_se", "k": 1, "complexity": "family_rank"})["winner"] == "many_but_linear"
    assert pa.choose(cs, {"rule": "one_se", "k": 1, "complexity": "lexicographic"})["winner"] == "many_but_linear", "rank first, then features"
    assert pa.complexity_of("ridge", 7.5) == {"family_rank": 1, "n_features": 7.5} and pa.family_rank("lstm") == 6 and pa.family_rank("model_ensemble") == 7 and pa.family_rank("unheard_of") == 5


def test_a_candidate_without_an_uncertainty_estimate_cannot_be_called_indistinguishable_and_a_missing_score_is_listed():
    cs = cands()
    cs[0]["se"] = None
    out = pa.choose(cs, {"rule": "one_se", "k": 2.0, "complexity": "n_features"})
    row = {r["id"]: r for r in out["table"]}
    assert row["simple"]["eligible"] is False and "no standard error for the gap" in row["simple"]["reason"] and out["winner"] == "medium"
    cs.append({"id": "broken", "score": float("nan"), "se": 0.1, "complexity": {"family_rank": 1, "n_features": 1}})
    out = pa.choose(cs, {"rule": "one_se", "k": 2.0, "complexity": "n_features"})
    assert out["skipped"] == [{"id": "broken", "reason": "no finite score"}] and out["winner"] == "medium"
    assert pa.choose([], {"rule": "one_se", "k": 1, "complexity": "n_features"})["eligibility"] == "insufficient"


def test_the_penalty_rule_trades_score_for_complexity_at_a_declared_rate():
    out = pa.choose(cands(), {"rule": "penalty", "lambda": 0.005, "complexity": "n_features"})
    adj = {r["id"]: r["adjusted"] for r in out["table"]}
    assert adj == pytest.approx({"simple": 0.50 - 0.015, "medium": 0.58 - 0.06, "complex": 0.60 - 0.20}) and out["winner"] == "medium"
    assert pa.choose(cands(), {"rule": "penalty", "lambda": 0.0, "complexity": "n_features"})["winner"] == "complex", "no penalty is the best score"
    assert pa.choose(cands(), {"rule": "penalty", "lambda": 1.0, "complexity": "n_features"})["winner"] == "simple"


def test_the_rule_must_be_declared_completely_and_is_checked_against_the_frozen_plan():
    for bad in ({"rule": "lucky"}, {"rule": "one_se", "complexity": "n_features"}, {"rule": "one_se", "k": -1, "complexity": "n_features"}, {"rule": "one_se", "k": 1},
                {"rule": "penalty", "complexity": "n_features"}, {"rule": "penalty", "lambda": 0.1, "complexity": "lexicographic"}):
        with pytest.raises(pa.ParsimonyError):
            pa.choose(cands(), bad)
    rule = {"rule": "one_se", "k": 1.0, "complexity": "n_features"}
    frozen = {"state": "frozen", "content": {"decision_policy": {"parsimony": rule}}}
    assert pa.choose(cands(), rule, frozen)["predeclared"] is True and "in the frozen evaluation plan" in pa.choose(cands(), rule, frozen)["predeclared_note"]
    assert pa.choose(cands(), rule, {"state": "draft", "content": frozen["content"]})["predeclared"] is False
    assert pa.choose(cands(), rule, {"state": "frozen", "content": {"decision_policy": {"parsimony": {**rule, "k": 2.0}}}})["predeclared"] is False, "a different rule from the one frozen is not predeclared"
    unplanned = pa.choose(cands(), rule)
    assert unplanned["predeclared"] is None and "nothing shows the rule was fixed" in unplanned["predeclared_note"]


def test_with_equal_true_scores_the_rule_prefers_the_simplest_far_more_often_than_the_best_observed_score_does():
    rule = {"rule": "one_se", "k": 1.0, "complexity": "n_features"}
    sim = pa.simulate_selection([0.5] * 5, [0.1] * 5, rule, draws=2000, seed=11)
    assert sim["best_score_picked"][0] == pytest.approx(0.2, abs=4 * sim["monte_carlo_se_at_half"]), "the best observed of five equals is each candidate one time in five"
    assert sim["picked"][0] > 0.55, sim["picked"]
    assert sim["picked"][0] > 2.5 * sim["best_score_picked"][0]
    skilled = pa.simulate_selection([0.2, 0.2, 0.2, 0.2, 1.0], [0.1] * 5, rule, draws=1000, seed=3)
    assert skilled["picked"][4] > 0.97, "a complex candidate that is really better (nine standard errors) is not given up"
