"""P2-17: the simplest eligible model that is not meaningfully worse than the highest-ranked one.

Checked against independent references (the bootstrap of ``inference.paired_difference``, a numpy Sharpe), against simulation (the paired bootstrap error of a serially dependent gap, the error rate of the
non-inferiority test), and by property (no fixed penalty: the choice does not depend on the scale of the returns). The two rules are kept apart: the default is a heuristic and says so, the formal test needs a margin.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.ml import inference as inf, simplicity as sm

DAYS = pd.bdate_range("2021-01-04", periods=750)


def cx(kind="ridge", n_features=10, train=1.0, infer=1.0, instab=0.3, params=None):
    return sm.complexity_vector(kind, params, n_features=n_features, train_seconds=train, inference_ms=infer, instability=instab)


def stream(seed, mu=0.0008, sd=0.01, n=750):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n), index=DAYS[:n])


def sharpe(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(252.0))


def shifted(base, shift, seed, sd=0.0015):
    """``base`` minus a known shift plus zero-mean noise: the realised paired gap in the mean is the shift."""
    noise = np.random.default_rng(seed).normal(0, sd, len(base))
    return pd.Series(base.to_numpy() - shift + noise - noise.mean(), index=base.index)


# ------------------------------------------------------------------------------------------------------------------ the paired gap
def test_the_paired_gap_matches_the_inference_module_and_an_independent_sharpe():
    a, b = stream(1, 0.0012), stream(2, 0.0004)
    g = sm.paired_gap(a, b, statistic="sharpe", n_boot=300, seed=4)
    ref = inf.paired_difference(a, b, statistic="sharpe", n_boot=300, seed=4, horizon=1)
    assert g["estimate"] == pytest.approx(sharpe(a) - sharpe(b), abs=1e-10) == pytest.approx(ref.value, abs=1e-10)
    assert g["se"] == pytest.approx(ref.extras["bootstrap_se"], rel=1e-12), "the same resampled blocks give the same error"
    assert g["aligned"] == 750 and g["dropped"] == {"only_best": 0, "only_candidate": 0}
    assert sm.paired_gap(a, b, statistic="ann_return", n_boot=300, seed=4)["estimate"] == pytest.approx((a.mean() - b.mean()) * 252.0, abs=1e-12)


def test_only_the_dates_both_candidates_have_are_compared_and_the_rest_is_counted():
    a, b = stream(1), stream(2)
    g = sm.paired_gap(a.iloc[:700], b.iloc[50:], n_boot=250, seed=1)
    assert g["aligned"] == 650 and g["dropped"] == {"only_best": 50, "only_candidate": 50}
    short = sm.paired_gap(a.iloc[:10], b.iloc[:10])
    assert short["how"] == "unavailable" and "fewer than" in short["reason"]
    with pytest.raises(sm.SimplicityError):
        sm.paired_gap(a, b, statistic="sortino")


def test_the_bootstrap_error_allows_for_serial_dependence_that_an_iid_error_would_miss():
    """The gap between two streams that is itself an AR(1): the sampling spread of its mean is about sqrt((1 + phi) / (1 - phi)) times the iid figure. The paired block bootstrap sees that; a bootstrap that
    resamples days one at a time does not."""
    phi, n, reps = 0.8, 600, 150
    rng = np.random.default_rng(5)

    def pair():
        e = rng.normal(0, 0.004, n)
        d = np.zeros(n)
        for t in range(1, n):
            d[t] = phi * d[t - 1] + e[t]
        z = rng.normal(0.0005, 0.01, n)
        return pd.Series(z + d, index=DAYS[:n]), pd.Series(z, index=DAYS[:n])

    means, block_se, iid_se = [], [], []
    for i in range(reps):
        a, b = pair()
        means.append((a.mean() - b.mean()) * 252.0)
        if i < 12:
            block_se.append(sm.paired_gap(a, b, statistic="ann_return", n_boot=200, seed=i)["se"])
            x = (a - b).to_numpy()
            r = np.random.default_rng(i)
            iid_se.append(np.std([x[r.integers(0, n, n)].mean() * 252.0 for _ in range(200)], ddof=1))
    truth = float(np.std(means, ddof=1))
    assert 0.6 < np.mean(block_se) / truth < 1.5, (np.mean(block_se), truth)
    assert np.mean(iid_se) / truth < 0.6, "ignoring the dependence understates the error of the gap by roughly the factor above"


def test_holm_adjustment_is_the_textbook_step_down():
    assert sm._holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    h = sm._holm([0.5, None, 0.2])
    assert h[1] is None and h[0] == pytest.approx(0.5) and h[2] == pytest.approx(0.4), "a missing p-value stays missing and does not count towards the family"


# ------------------------------------------------------------------------------------------------------------------ the error rates of the formal test
def test_the_non_inferiority_test_rejects_at_about_its_level_on_the_margin_and_has_power_inside_it():
    """The truth is known by construction: independent daily streams whose annual mean differs by exactly the margin (the boundary of H0) or by nothing. At the boundary the test must reject about
    ``level`` of the time; with no gap it must reject most of the time. The Monte Carlo error of each rate is stated in the bound."""
    margin, sims, level = 0.30, 160, 0.05
    rng = np.random.default_rng(8)

    def rate(true_gap):
        hits = 0
        for i in range(sims):
            b = pd.Series(rng.normal(0.0004, 0.01, 750), index=DAYS)
            a = pd.Series(rng.normal(0.0004 + true_gap / 252.0, 0.01, 750), index=DAYS)
            g = sm.paired_gap(a, b, statistic="ann_return", n_boot=200, seed=i)
            hits += sm._one_sided_ps(g, margin)[0] < level
        return hits / sims

    size, power = rate(margin), rate(0.0)
    mc = math.sqrt(level * (1 - level) / sims)
    assert size <= level + 3 * mc + 0.02, f"false non-inferiority rate {size:.3f} at the margin (level {level})"
    assert power >= 0.55, f"power {power:.3f} with no gap"


# ------------------------------------------------------------------------------------------------------------------ the heuristic rule
def candidate(cid, r, kind="ridge", **kw):
    return {"id": cid, "score": sharpe(r), "returns": r, "complexity": cx(kind, **kw)}


def test_the_simplest_candidate_within_the_tolerance_wins_and_a_clearly_worse_simple_one_does_not():
    best = stream(1, 0.0011)
    pool = [candidate("deep", best, "lstm", n_features=40, train=300.0), candidate("ridge", shifted(best, 0.00002, 2)), candidate("tiny_but_bad", shifted(best, 0.0006, 3), n_features=2, train=0.1)]
    out = sm.choose_simplest(pool)
    assert out["highest_ranked"] == "deep" and out["selected"] == "ridge" and out["changed_choice"] is True
    status = {r["id"]: r["statistical_status"] for r in out["table"]}
    assert status == {"deep": "highest_ranked", "ridge": "not_distinguishable_by_heuristic", "tiny_but_bad": "worse_than_the_heuristic_allows"}
    row = next(r for r in out["table"] if r["id"] == "ridge")
    assert row["how"] == "paired_bootstrap" and row["aligned"] == 750 and row["se_of_gap"] > 0 and row["gap_in_se"] < 1.0 and row["mean_block"] >= 1.0
    assert out["claim"]["kind"] == "selection_heuristic" and "not a test of equivalence" in out["claim"]["statement"] and out["equivalence_claimed"] is False


def test_k_sets_how_far_behind_the_best_a_candidate_may_be_and_k_zero_keeps_only_the_best():
    best = stream(1, 0.0011)
    pool = [candidate("deep", best, "lstm", n_features=40), candidate("ridge", shifted(best, 0.00012, 2))]
    gap = next(r for r in sm.choose_simplest(pool)["table"] if r["id"] == "ridge")
    assert 1.0 < gap["gap_in_se"] < 6.0, gap["gap_in_se"]
    assert sm.choose_simplest(pool, sm.SimplicityPolicy(k=1.0))["selected"] == "deep"
    assert sm.choose_simplest(pool, sm.SimplicityPolicy(k=gap["gap_in_se"] + 0.5))["selected"] == "ridge"
    assert sm.choose_simplest(pool, sm.SimplicityPolicy(k=0.0))["selected"] == "deep"


def test_there_is_no_fixed_penalty_and_the_choice_does_not_depend_on_the_scale_of_the_returns():
    assert not any("penal" in f or "lambda" in f for f in sm.SimplicityPolicy.__dataclass_fields__)
    with pytest.raises(sm.SimplicityError, match="simplicity policy"):
        sm.policy_from({"lambda": 0.1})
    best = stream(1, 0.0011)
    mk = lambda c: [candidate("deep", best * c, "lstm", n_features=40), candidate("ridge", shifted(best, 0.00002, 2) * c)]                 # noqa: E731
    one, three = sm.choose_simplest(mk(1.0)), sm.choose_simplest(mk(3.0))
    assert one["selected"] == three["selected"] == "ridge"
    assert next(r for r in one["table"] if r["id"] == "ridge")["gap_in_se"] == pytest.approx(next(r for r in three["table"] if r["id"] == "ridge")["gap_in_se"], rel=1e-9)


def test_the_complexity_order_follows_the_declared_priority_and_treats_unmeasured_as_not_simple():
    pol = sm.SimplicityPolicy()
    small, big = cx("ridge", 5), cx("lstm", 5)
    assert sm.order_key(small, pol) < sm.order_key(big, pol), "the family decides first"
    few, many = cx("ridge", 5), cx("ridge", 50)
    assert sm.order_key(few, pol) < sm.order_key(many, pol)
    unmeasured = sm.complexity_vector("ridge", None, n_features=5)
    assert sm.order_key(cx("ridge", 5, train=500.0), pol) < sm.order_key(unmeasured, pol), "an unmeasured training cost is never read as free: it sorts after a measured slow one"
    flipped = sm.SimplicityPolicy(priority=("n_features", "model_complexity"))
    assert sm.order_key(cx("lstm", 5), flipped) < sm.order_key(cx("ridge", 50), flipped), "with features first, five features beat fifty whatever the family"


def test_costs_within_the_tolerance_tie_and_costs_far_apart_do_not():
    pol = sm.SimplicityPolicy(cost_tolerance=0.25)
    base = sm.order_key(cx(train=10.0), pol)
    assert sm.order_key(cx(train=11.0), pol) == base or sm.order_key(cx(train=11.0), pol)[3] - base[3] <= 1
    assert sm.order_key(cx(train=40.0), pol) > sm.order_key(cx(train=10.0), pol)
    assert sm.order_key(cx(train=10.0), pol) == sm.order_key(cx(train=10.0), pol)
    assert sm._bucket(None, 0.25) == math.inf and sm._bucket(0.0, 0.25) == -math.inf


def test_dependencies_include_those_of_an_ensembles_members_and_none_for_a_linear_model():
    assert sm.dependencies_of("ridge") == [] and cx("ridge")["dependencies"] == 0
    members = sm.dependencies_of("model_ensemble", {"members": "ridge,lightgbm[n_estimators=5],xgboost"})
    assert set(sm.dependencies_of("lightgbm")) | set(sm.dependencies_of("xgboost")) == set(members) and len(members) >= 2
    v = sm.complexity_vector("model_ensemble", {"members": "ridge,lightgbm"}, n_features=9)
    assert v["dependencies"] == len(v["dependency_names"]) >= 1 and v["training_cost"] is None and v["inference_latency"] is None


def test_identical_candidates_tie_on_performance_and_the_simpler_is_chosen_without_error():
    r = stream(1)
    out = sm.choose_simplest([candidate("deep", r, "lstm", n_features=40), candidate("ridge", r.copy())])
    assert out["selected"] == "ridge" and out["changed_choice"] is True
    row = next(x for x in out["table"] if x["id"] == "ridge")
    assert row["gap"] == pytest.approx(0.0, abs=1e-12) and row["qualifies"] is True


def test_without_a_paired_series_the_gap_falls_back_to_independent_errors_and_without_errors_the_best_wins():
    pool = [{"id": "deep", "score": 1.0, "score_se": 0.3, "complexity": cx("lstm", 40)}, {"id": "ridge", "score": 0.9, "score_se": 0.3, "complexity": cx("ridge", 5)}]
    out = sm.choose_simplest(pool)
    row = next(x for x in out["table"] if x["id"] == "ridge")
    assert row["how"] == "unpaired" and row["se_of_gap"] == pytest.approx(math.hypot(0.3, 0.3)) and out["selected"] == "ridge"
    strict = sm.choose_simplest(pool, sm.SimplicityPolicy(require_paired=True))
    assert strict["selected"] == "deep" and next(x for x in strict["table"] if x["id"] == "ridge")["statistical_status"] == "no_uncertainty_estimate"
    bare = sm.choose_simplest([{"id": "deep", "score": 1.0, "complexity": cx("lstm", 40)}, {"id": "ridge", "score": 0.95, "complexity": cx("ridge", 5)}])
    assert bare["selected"] == "deep" and "no uncertainty" in next(x for x in bare["table"] if x["id"] == "ridge")["reason"] or "no paired" in next(x for x in bare["table"] if x["id"] == "ridge")["reason"]


def test_the_daily_ic_gap_uses_a_newey_west_error_and_is_labelled_paired():
    rng = np.random.default_rng(3)
    ic_best = pd.Series(rng.normal(0.04, 0.1, 400), index=DAYS[:400])
    noise = rng.normal(0, 0.02, 400)
    ic_other = pd.Series(ic_best.to_numpy() - 0.0005 + noise - noise.mean(), index=DAYS[:400])                       # behind by exactly 0.0005 on average
    pool = [{"id": "deep", "score": float(ic_best.mean()), "ic": ic_best, "complexity": cx("lstm", 40)}, {"id": "ridge", "score": float(ic_other.mean()), "ic": ic_other, "complexity": cx("ridge", 5)}]
    out = sm.choose_simplest(pool, metric="ic", horizon=5)
    row = next(x for x in out["table"] if x["id"] == "ridge")
    assert row["how"] == "paired_hac" and row["se_of_gap"] > 0 and out["selected"] == "ridge"
    ref = inf.hac_mean(ic_best - ic_other, horizon=5)
    assert row["gap"] == pytest.approx(ref.value) and row["se_of_gap"] == pytest.approx(ref.extras["se"])


# ------------------------------------------------------------------------------------------------------------------ the formal rule is not the heuristic
def test_the_formal_test_needs_a_declared_margin_and_says_what_it_does_and_does_not_show():
    with pytest.raises(sm.SimplicityError, match="margin"):
        sm.SimplicityPolicy(rule="non_inferiority")
    with pytest.raises(sm.SimplicityError, match="margin"):
        sm.SimplicityPolicy(rule="non_inferiority", margin=0.0)
    with pytest.raises(sm.SimplicityError):
        sm.SimplicityPolicy(rule="vibes")
    best = stream(1, 0.0011)
    pool = [candidate("deep", best, "lstm", n_features=40), candidate("ridge", shifted(best, 0.00003, 2)), candidate("meh", shifted(best, 0.0006, 3))]
    wide = sm.choose_simplest(pool, sm.SimplicityPolicy(rule="non_inferiority", margin=1.0, seed=2))
    assert wide["claim"]["kind"] == "formal_non_inferiority" and "H0" in wide["claim"]["statement"] and "declared assumption" in " ".join(wide["claim"]["caveats"])
    rows = {r["id"]: r for r in wide["table"]}
    assert rows["ridge"]["statistical_status"] == "non_inferior_within_margin" and rows["ridge"]["qualifies"] and rows["ridge"]["adjusted_p_non_inferior"] < 0.05
    assert wide["selected"] == "ridge"
    tight = sm.choose_simplest(pool, sm.SimplicityPolicy(rule="non_inferiority", margin=0.05, seed=2))
    t = {r["id"]: r for r in tight["table"]}
    assert t["ridge"]["statistical_status"] == "not_shown_non_inferior" and "not evidence of inferiority" in t["ridge"]["reason"]
    assert t["meh"]["statistical_status"] in ("demonstrably_worse", "not_shown_non_inferior") and tight["selected"] == "deep"


def test_equivalence_is_claimed_only_when_both_one_sided_tests_reject_and_never_by_the_heuristic():
    best = stream(1, 0.0011, n=750)
    near = candidate("ridge", shifted(best, 0.0, 2, sd=0.0008))
    pool = [candidate("deep", best, "lstm", n_features=40), near]
    heuristic = sm.choose_simplest(pool)
    assert heuristic["selected"] == "ridge" and heuristic["equivalence_claimed"] is False and "equivalent_within_margin" not in heuristic["table"][1]
    formal = sm.choose_simplest(pool, sm.SimplicityPolicy(rule="non_inferiority", margin=1.5, seed=1))
    row = next(r for r in formal["table"] if r["id"] == "ridge")
    assert row["equivalent_within_margin"] is True and formal["equivalence_claimed"] is True and row["adjusted_p_not_better_by_margin"] < 0.05
    narrow = sm.choose_simplest(pool, sm.SimplicityPolicy(rule="non_inferiority", margin=0.01, seed=1))
    assert narrow["equivalence_claimed"] is False


def test_the_holm_adjustment_is_over_all_the_comparisons_with_the_best():
    best = stream(1, 0.0011)
    few = sm.choose_simplest([candidate("deep", best, "lstm", n_features=40), candidate("r1", shifted(best, 0.0001, 2))], sm.SimplicityPolicy(rule="non_inferiority", margin=0.8, seed=3))
    many_pool = [candidate("deep", best, "lstm", n_features=40)] + [candidate(f"r{i}", shifted(best, 0.0001, 10 + i)) for i in range(1, 6)]
    many = sm.choose_simplest(many_pool, sm.SimplicityPolicy(rule="non_inferiority", margin=0.8, seed=3))
    p_few = next(r for r in few["table"] if r["id"] == "r1")["adjusted_p_non_inferior"]
    p_many = [r["adjusted_p_non_inferior"] for r in many["table"] if r["id"] != "deep"]
    assert max(p_many) >= p_few - 1e-12 and all(0 <= p <= 1 for p in p_many)


def test_no_finite_score_is_insufficient_and_the_policy_has_a_hash():
    out = sm.choose_simplest([{"id": "a", "score": None, "complexity": cx()}, {"id": "b", "score": float("nan"), "complexity": cx()}])
    assert out["eligibility"] == "insufficient" and out["selected"] is None
    assert sm.SimplicityPolicy().hash() == sm.policy_from({}).hash() != sm.SimplicityPolicy(k=2.0).hash()
