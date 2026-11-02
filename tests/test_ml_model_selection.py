"""P1-15: constrained multi-objective selection on validation data only.

The figures are checked against independent numpy references, the gates one at a time (each must change the outcome of a candidate that passes the rest), the Pareto analysis against a brute-force
definition on random data, the leakage rules by handing the selection protected rows, and the integration on real saved experiments and on a real (small) search.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from src.ml import experiments as xp, model_selection as ms, nested as nst, registry as rg, simplicity as sm, strategy_search as ss
from tests.ml_helpers import make_bundle

DAYS = pd.bdate_range("2022-01-03", periods=300)


def returns(seed=0, mu=0.001, sd=0.01, n=300, idx=None):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=DAYS[:n] if idx is None else idx)


def cand(cid, r=None, kind="ridge", **kw):
    base = {"id": cid, "kind": kind, "returns": returns(1) if r is None else r, "turnover": 30.0, "coverage": 0.95, "n_assets": 8, "exposure": {"gross_max": 1.0, "net_abs_max": 0.1},
            "liquidity": {"status": "measured", "levels": [{"capital": 1e6, "fill_share": 0.9}]}, "complexity": sm.complexity_vector(kind, None, n_features=10, train_seconds=1.0, inference_ms=1.0),
            "lineage": {"checked": True, "n_folds": 4, "folds": [], "violations": []}}
    base.update(kw)
    return base


def sharpe_ref(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(252.0))


# ------------------------------------------------------------------------------------------------------------------ the evidence is one pooled chronological series
def test_the_sharpe_is_computed_on_the_pooled_series_and_is_not_the_mean_of_the_fold_sharpes():
    rng = np.random.default_rng(3)
    quiet = pd.Series(rng.normal(0.002, 0.001, 30), index=DAYS[:30])                      # a short, calm fold with a huge Sharpe ratio
    wild = pd.Series(rng.normal(0.0003, 0.02, 270), index=DAYS[30:300])                    # a long, noisy fold with a small one
    pooled = ms.pool_chronological([wild, quiet])                                          # given out of order: the pooling sorts by date
    assert pooled.index.is_monotonic_increasing and len(pooled) == 300
    m = ms.metrics_of(pooled)
    ref = sharpe_ref(np.concatenate([quiet.to_numpy(), wild.to_numpy()]))
    assert m["net_sharpe"] == pytest.approx(ref, abs=1e-12)
    naive = ms.naive_fold_sharpe([quiet, wild])
    assert abs(naive - ref) > 5.0, "averaging the fold ratios would give a very different (and misleading) number"
    card = ms.evaluate_candidate(cand("a", folds=[wild, quiet], returns=None), ms.SelectionPolicy(eligibility=ms.Eligibility(min_net_sharpe=None)))
    assert card["primary"] == pytest.approx(ref, abs=1e-12) and card["aggregation"]["naive_mean_of_fold_sharpes"] == pytest.approx(naive)
    assert card["aggregation"]["method"] == "pooled chronological validation returns"


def test_folds_that_disagree_about_a_day_are_refused_and_identical_duplicates_collapse():
    a, b = returns(1, n=100), returns(2, n=100)
    with pytest.raises(ms.SelectionError, match="different returns for the same day"):
        ms.pool_chronological([a, b])
    twice = ms.pool_chronological([a, a.copy()])
    assert len(twice) == 100 and np.allclose(twice.to_numpy(), a.to_numpy())
    with pytest.raises(ms.SelectionError, match="no fold returns"):
        ms.pool_chronological([])


def test_the_other_metrics_match_independent_loops():
    r = returns(5, mu=0.0007, sd=0.012)
    m = ms.metrics_of(r)
    x = r.to_numpy()
    eq, peak, dd = 1.0, 1.0, 0.0
    for v in x:                                                                            # drawdown by an explicit loop
        eq *= 1.0 + v
        peak = max(peak, eq)
        dd = max(dd, 1.0 - eq / peak)
    down = math.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    assert m["max_drawdown"] == pytest.approx(dd, abs=1e-12)
    assert m["net_sortino"] == pytest.approx(x.mean() / down * math.sqrt(252.0), abs=1e-12)
    assert m["net_calmar"] == pytest.approx(x.mean() * 252.0 / dd, abs=1e-12)
    assert m["net_ann_return"] == pytest.approx(x.mean() * 252.0) and m["ann_vol"] == pytest.approx(x.std(ddof=1) * math.sqrt(252.0))
    blocks = [sharpe_ref(b) for b in np.array_split(x, 4)]
    assert m["instability"] == pytest.approx(np.std(blocks, ddof=1), abs=1e-12)           # the Overfitting Center's definition of instability
    rng = np.random.default_rng(1)
    ar = np.zeros(2000)
    for t in range(1, 2000):
        ar[t] = 0.6 * ar[t - 1] + rng.normal()
    rho = np.corrcoef(ar[:-1], ar[1:])[0, 1]
    assert ms.effective_n(pd.Series(ar)) == pytest.approx(2000 * (1 - rho) / (1 + rho), rel=1e-9) and ms.effective_n(pd.Series(ar)) < 700


# ------------------------------------------------------------------------------------------------------------------ eligibility: every gate changes the outcome, and the states stay distinct
STRICT = ms.Eligibility(min_validation_days=100, min_coverage=0.8, min_assets=2, max_drawdown=0.5, max_ann_vol=0.5, max_gross_exposure=2.0, max_abs_net_exposure=1.0, max_ann_turnover=100.0,
                        min_capacity_share=0.5, capacity_capital=1e6, min_effective_obs=50.0, min_net_sharpe=0.0, min_sharpe_lower_bound=-5.0)
GOOD = returns(7, mu=0.0015, sd=0.01)


def persistent(phi=0.9, n=300, seed=5):
    """A profitable series whose days are strongly dependent: three hundred days, about sixteen independent ones."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 0.004)
    return pd.Series(x + 0.0012, index=DAYS[:n])


def gates_of(c, pol=STRICT):
    card = ms.evaluate_candidate(c, ms.SelectionPolicy(eligibility=pol))
    return card, {g["gate"]: g["status"] for g in card["gates"]}


def test_a_candidate_that_meets_every_requirement_is_eligible_with_every_gate_enforced():
    card, g = gates_of(cand("ok", GOOD))
    assert card["state"] == "eligible" and set(g.values()) == {"pass"} and len(g) == 13


@pytest.mark.parametrize("gate,change", [
    ("min_coverage", {"coverage": 0.5}), ("min_assets", {"n_assets": 1}), ("max_gross_exposure", {"exposure": {"gross_max": 2.5, "net_abs_max": 0.1}}),
    ("max_abs_net_exposure", {"exposure": {"gross_max": 1.0, "net_abs_max": 1.5}}), ("max_ann_turnover", {"turnover": 150.0}),
    ("min_capacity_share", {"liquidity": {"status": "measured", "levels": [{"capital": 1e6, "fill_share": 0.2}]}}),
    ("max_ann_vol", {"returns": GOOD * 10}), ("min_net_sharpe", {"returns": returns(7, mu=-0.001)}), ("min_effective_obs", {"returns": persistent()}),
    ("max_drawdown", {"returns": pd.concat([GOOD.iloc[:150], pd.Series([-0.7], index=[DAYS[150]]), GOOD.iloc[151:]])}),
    ("min_sharpe_lower_bound", {"returns": returns(9, mu=0.0004)}),
])
def test_each_gate_alone_makes_a_candidate_ineligible(gate, change):
    pol = ms.Eligibility(**{**STRICT.__dict__, "min_sharpe_lower_bound": 1.0 if gate == "min_sharpe_lower_bound" else -5.0})
    card, g = gates_of(cand("c", GOOD, **change), pol)
    assert card["state"] == "ineligible" and g[gate] == "fail", (gate, g)
    assert gate in (card["reason"] or "")
    others = {k: v for k, v in g.items() if k != gate and k not in ("max_ann_vol", "min_net_sharpe", "max_drawdown", "min_sharpe_lower_bound", "min_effective_obs")}
    assert all(v == "pass" for v in others.values()), "only the changed requirement fails"


def folds_of(*pairs):
    return [{"fold": i, "train_start": "2020-01-01", "train_end": "2021-06-30", "n_train": 900, "train_rows_hash": f"h{i}", "fit_cutoff": cut, "label_end_max": end, "test_start": cut, "test_end": "2021-12-31"}
            for i, (cut, end) in enumerate(pairs)]


def test_a_fold_whose_training_labels_reached_past_its_fit_cutoff_is_a_violation_the_gate_refuses():
    honest = ms.fold_lineage(folds_of(("2021-07-01", "2021-06-30"), ("2021-08-02", "2021-07-30")))
    assert honest["checked"] and honest["n_folds"] == 2 and honest["violations"] == [] and honest["folds"][0]["train_rows_hash"] == "h0"
    leaky = ms.fold_lineage(folds_of(("2021-07-01", "2021-06-30"), ("2021-08-02", "2021-08-09")))
    assert [v["fold"] for v in leaky["violations"]] == [1] and leaky["violations"][0]["label_end_max"] == "2021-08-09"
    card, g = gates_of(cand("leak", GOOD, lineage=leaky))
    assert card["state"] == "ineligible" and g["max_label_overlap"] == "fail" and "max_label_overlap" in card["reason"]
    assert gates_of(cand("fine", GOOD, lineage=honest))[1]["max_label_overlap"] == "pass"
    skipped = ms.fold_lineage([{"fold": 0, "skipped": True}, {"fold": 1, "fit_cutoff": "2021-07-01"}])
    assert skipped["checked"] is False and skipped["note"]
    card, g = gates_of(cand("old", GOOD, lineage=skipped))
    assert g["max_label_overlap"] == "unavailable" and card["state"] == "ineligible", "an experiment saved before the lineage was kept cannot show it, and that is not a pass"
    off = ms.Eligibility(**{**STRICT.__dict__, "max_label_overlap": None})
    assert gates_of(cand("old", GOOD, lineage=skipped), off)[1]["max_label_overlap"] == "not_enforced"


def test_row_ranges_round_trip_and_cap_the_number_of_runs():
    from src.ml.training import expand_ranges, row_hash, row_ranges

    rng = np.random.default_rng(4)
    for _ in range(50):
        idx = np.unique(rng.integers(0, 400, int(rng.integers(0, 300))))
        runs = row_ranges(idx)
        assert np.array_equal(expand_ranges(runs), idx) and all(a <= b for a, b in runs) and (len(runs) == 0 or runs[0][0] == idx[0])
        assert row_hash(expand_ranges(runs)) == row_hash(idx)
    assert row_ranges(np.arange(0, 100, 2), cap=10) is None, "too scattered to store as runs: the hash still identifies the rows"
    assert row_ranges(np.array([], dtype=int)) == [] and expand_ranges([]).size == 0


def test_the_audit_rederives_each_folds_facts_from_the_recorded_rows_and_catches_a_tampered_record(saved):
    from src.ml import experiments as xp
    from src.ml.targets import parse_target

    reg, ids = saved
    clean = ms.audit_lineage(reg, ids["ridge"])
    assert clean["audited"] >= 3 and clean["violations"] == [] and all(f["ok"] and f["label_end_recomputed"] <= f["fit_cutoff"] for f in clean["folds"] if f["audited"])
    _, ds, train, _ = xp.load_run(reg, ids["ridge"])
    horizon = int(parse_target(train.target)[1])
    fold = next(f for f in train.folds if not f.get("skipped") and f.get("train_row_ranges"))
    assert fold["train_row_ranges"] and fold["test_row_ranges"] and fold["n_train"] == sum(b - a + 1 for a, b in fold["train_row_ranges"])
    # 1. a leak: the training rows run on into the block the fit was meant to predict
    late = [dict(f) for f in train.folds]
    k = late.index(fold)
    first_test = fold["test_row_ranges"][0][0]
    late[k]["train_row_ranges"] = [[fold["train_row_ranges"][0][0], first_test + 40]]
    bad = ms.audit_folds(ds, late, horizon)
    assert any("reaches position" in v["problem"] or "is a test row" in v["problem"] for v in bad["violations"]) and any("hash" in v["problem"] for v in bad["violations"])
    # 2. rows that are not the rows the hash and the count were taken from
    swapped = [dict(f) for f in train.folds]
    swapped[k]["train_row_ranges"] = [[fold["train_row_ranges"][0][0] + 1, fold["train_row_ranges"][-1][1]]]
    assert any("hash" in v["problem"] or "count" in v["problem"] for v in ms.audit_folds(ds, swapped, horizon)["violations"])
    # 3. a record with no rows is not audited, and is never counted as clean
    bare = [{**f, "train_row_ranges": None} for f in train.folds]
    out = ms.audit_folds(ds, bare, horizon)
    assert out["audited"] == 0 and all(f["audited"] is False for f in out["folds"]) and out["violations"] == []
    # 4. a candidate asked to be audited carries it, and an unaudited fold set cannot satisfy the gate
    cand_ = ms.candidate_from_experiment(reg, ids["ridge"], audit=True)
    assert cand_["lineage"]["audit"]["audited"] >= 3 and cand_["lineage"]["checked"] and cand_["lineage"]["violations"] == []
    selected = ms.select_saved(reg, [ids["ridge"], ids["lasso"]], ms.SelectionPolicy(eligibility=ms.Eligibility(min_net_sharpe=None)), store=False, audit=True)
    assert all(c["lineage"]["audit"]["audited"] >= 3 for c in selected["candidates"]) and all(c["state"] == "eligible" for c in selected["candidates"])


def noise_pool(n_candidates=6, seed=3, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return idx, [cand(f"n{i}", pd.Series(rng.normal(0.0, 0.01, n), index=idx)) for i in range(n_candidates)]


POLICY_NESTED = ms.policy_from({"eligibility": {"min_net_sharpe": None}})


def test_the_nested_estimate_matches_a_hand_computed_chain_and_shows_how_little_picking_among_noise_earns():
    idx, pool = noise_pool()
    out = ms.select(pool, POLICY_NESTED)
    nested = out["nested"]
    assert nested["eligibility"] == "applied" and nested["audit"]["ok"] and nested["n_selected"] == 4
    # an independent statement of the procedure: inner window = the 126 days up to the cutoff, choose the best Sharpe, carry the choice to the days up to the next cutoff
    series = {c["id"]: c["returns"] for c in pool}
    cut = [idx[max(int(len(idx) * f) - 1, 0)] for f in (0.4, 0.55, 0.7, 0.85)]
    chain, choices = [], []
    for k, c in enumerate(cut):
        inner = idx[idx <= c][-126:]
        best = max(series, key=lambda i: (sharpe_ref(series[i].loc[inner]), -ord(i[-1])))               # ties would go to the declared (alphabetical) order
        choices.append(best)
        end = cut[k + 1] if k + 1 < len(cut) else idx[-1]
        chain.append(series[best].loc[(idx > c) & (idx <= end)])
    ref = pd.concat(chain)
    assert nested["choices"] == choices and nested["chain_days"] == len(ref)
    assert nested["chain_sharpe"] == pytest.approx(sharpe_ref(ref), abs=1e-12) and nested["outer_first"] == str(ref.index[0].date()) and nested["outer_last"] == str(ref.index[-1].date())
    assert nested["best_in_hindsight"] - nested["chain_sharpe"] > 0.3, "the best of six noise streams looks good in hindsight and the rule that picks it as it goes earns much less"
    assert nested["best_in_hindsight"] == pytest.approx(max(sharpe_ref(series[i].reindex(ref.index)) for i in series))
    assert nested["median_candidate"] == pytest.approx(float(np.median([sharpe_ref(series[i].reindex(ref.index)) for i in series])))
    assert nested["selected_on_outer_days"] == pytest.approx(sharpe_ref(series[out["selected"]].reindex(ref.index)))


def test_nothing_after_a_cutoff_can_change_the_choice_made_at_it_and_the_rows_are_persisted():
    idx, pool = noise_pool()
    base = ms.select(pool, POLICY_NESTED)["nested"]
    cut = [pd.Timestamp(c) for c in base["cutoffs"]]
    for k in range(4):
        wild = []
        for c in pool:
            r = c["returns"].copy()
            r.loc[r.index > cut[k]] = np.random.default_rng(9).normal(0.5, 3.0, int((r.index > cut[k]).sum()))             # absurd outcomes after cutoff k
            wild.append({**c, "returns": r})
        got = ms.select(wild, ms.policy_from({"eligibility": {"min_net_sharpe": None, "max_drawdown": None}}))["nested"]
        assert got["choices"][: k + 1] == base["choices"][: k + 1], f"a choice made at or before cutoff {k} depended on what came after it"
    step = base["steps"][1]
    assert step["inner_dates"][-1] <= step["cutoff"] < step["outer_dates"][0] and len(step["inner_dates"]) == 126 and step["inner_hash"] and step["outer_hash"] and step["max_inner_label_end"] <= step["cutoff"]


def test_the_nested_audit_fails_a_record_whose_inner_window_runs_past_its_cutoff():
    idx, pool = noise_pool()
    nested = ms.select(pool, POLICY_NESTED)["nested"]
    label_end = pd.Series(idx, index=idx)
    assert nst.audit({"steps": nested["steps"], "until": str(idx[-1].date())}, label_end)["ok"]
    forged = [dict(s) for s in nested["steps"]]
    forged[1]["cutoff"] = forged[0]["cutoff"]                                              # the choice claims to have been made earlier than the days it read
    bad = nst.audit({"steps": forged, "until": str(idx[-1].date())}, label_end)
    assert not bad["ok"] and any("label ends after the cutoff" in v or "starts on or before" in v or "overlap" in v for v in bad["violations"])


def test_the_nested_estimate_says_when_it_does_not_apply_and_refuses_a_bad_policy():
    idx, pool = noise_pool(1)
    assert ms.select(pool, POLICY_NESTED)["nested"]["eligibility"] == "not_applicable"
    idx, pool = noise_pool(3)
    off = ms.policy_from({"eligibility": {"min_net_sharpe": None}, "nested": {"enabled": False}})
    assert ms.select(pool, off)["nested"]["eligibility"] == "not_applicable" and "switched off" in ms.select(pool, off)["nested"]["reason"]
    short = [{**c, "returns": c["returns"].iloc[:70]} for c in pool]
    got = ms.select(short, ms.policy_from({"eligibility": {"min_net_sharpe": None, "min_validation_days": 50}}))["nested"]
    assert got["eligibility"] == "insufficient" and "at least 80" in got["reason"]
    thin = [{**c, "returns": c["returns"].iloc[:90]} for c in pool]                         # long enough to try, but no cutoff has 60 inner days
    got = ms.select(thin, ms.policy_from({"eligibility": {"min_net_sharpe": None, "min_validation_days": 50}, "nested": {"cutoff_fractions": [0.3, 0.5]}}))["nested"]
    assert got["eligibility"] == "insufficient" and all(st["state"] == "insufficient" for st in got["steps"])
    other = ms.policy_from({"eligibility": {"min_net_sharpe": None}, "primary_metric": "net_sortino"})
    assert "net Sharpe ranking" in ms.select(pool, other)["nested"]["reason"]
    for bad in ({"cutoff_fractions": [0.5, 0.4]}, {"cutoff_fractions": []}, {"cutoff_fractions": [0.5, 1.0]}, {"min_inner": 5}, {"inner_days": 30, "min_inner": 60}, {"vibes": 1}):
        with pytest.raises(ms.SelectionError):
            ms.policy_from({"nested": bad})
    assert ms.policy_from({"nested": {"cutoff_fractions": [0.5, 0.8]}}).nested.cutoff_fractions == (0.5, 0.8)


def test_a_custom_aggregate_changes_what_the_inner_window_rewards():
    idx = pd.bdate_range("2022-01-03", periods=200)
    steady = pd.DataFrame({"r": np.full(200, 0.001) + np.random.default_rng(1).normal(0, 0.0005, 200)}, index=idx)          # a small, reliable mean
    lumpy = pd.DataFrame({"r": np.random.default_rng(2).normal(0.002, 0.03, 200)}, index=idx)                                    # a larger mean with more noise
    args = dict(target=pd.DataFrame({"r": np.zeros(200)}, index=idx), label_end=pd.Series(idx, index=idx), cutoffs=[idx[119]], inner_days=100, min_inner=60, until=idx[-1], score=lambda s, t: s.iloc[:, 0])
    by_mean = nst.select_as_of({"steady": steady, "lumpy": lumpy}, **args)
    by_sharpe = nst.select_as_of({"steady": steady, "lumpy": lumpy}, **args, aggregate=lambda v: v.mean() / v.std(ddof=1))
    assert by_mean["choices"] == ["lumpy"] and by_sharpe["choices"] == ["steady"], "a mean rewards the lump, a Sharpe ratio the steadiness"


def test_a_policy_is_predeclared_only_when_a_frozen_plan_carries_it():
    pol = ms.policy_from({"primary_metric": "net_sortino", "simplicity": {"k": 2.0}})
    plan = {"plan_id": "plan-1", "state": "frozen", "content": {"decision_policy": {"selection": {"primary_metric": "net_sortino", "simplicity": {"k": 2.0}}}}}
    assert ms.predeclared(pol, plan)["predeclared"] is True
    assert ms.predeclared(pol, {**plan, "state": "draft"})["predeclared"] is False
    other = {**plan, "content": {"decision_policy": {"selection": {"primary_metric": "net_sharpe"}}}}
    assert ms.predeclared(pol, other)["predeclared"] is False and "supplied with the results" in ms.predeclared(pol, other)["note"]
    assert ms.predeclared(pol, {**plan, "content": {"decision_policy": {}}})["predeclared"] is False
    assert ms.predeclared(pol, None)["predeclared"] is None and "nothing shows" in ms.predeclared(pol, None)["note"]
    assert ms.select([cand("a")], pol, plan)["predeclaration"]["predeclared"] is True


def test_too_little_data_is_insufficient_not_ineligible_and_a_run_that_failed_is_failed():
    short = ms.evaluate_candidate(cand("short", GOOD.iloc[:60]), ms.SelectionPolicy(eligibility=STRICT))
    assert short["state"] == "insufficient" and "60 validation days" in short["reason"]
    dead = ms.evaluate_candidate({"id": "boom", "error": "ExperimentError: too little history"}, ms.SelectionPolicy())
    assert dead["state"] == "failed" and dead["metrics"] is None
    nothing = ms.evaluate_candidate({"id": "empty", "returns": pd.Series(dtype=float)}, ms.SelectionPolicy())
    assert nothing["state"] == "insufficient"
    states = {short["state"], dead["state"], ms.evaluate_candidate(cand("bad", returns(7, mu=-0.001)), ms.SelectionPolicy())["state"], ms.evaluate_candidate(cand("good", GOOD), ms.SelectionPolicy())["state"]}
    assert states == {"insufficient", "failed", "ineligible", "eligible"}


def test_evidence_that_was_not_produced_is_not_a_pass_unless_the_policy_says_to_skip_it():
    c = cand("noliq", GOOD, liquidity={"status": "not_established", "levels": []}, coverage=None)
    card, g = gates_of(c)
    assert g["min_capacity_share"] == "unavailable" and g["min_coverage"] == "unavailable" and card["state"] == "ineligible"
    assert set(card["unevaluated_gates"]) == {"min_capacity_share", "min_coverage"}
    skip = ms.Eligibility(**{**STRICT.__dict__, "missing_evidence": "skip"})
    card2, g2 = gates_of(c, skip)
    assert card2["state"] == "eligible" and g2["min_capacity_share"] == "unavailable", "skipped, and still reported as unavailable rather than passed"
    off = ms.Eligibility(**{**STRICT.__dict__, "min_capacity_share": None, "capacity_capital": None})
    assert gates_of(c, off)[1]["min_capacity_share"] == "not_enforced"
    with pytest.raises(ms.SelectionError, match="capacity_capital"):
        ms.Eligibility(min_capacity_share=0.5)


def test_nothing_is_selected_when_nothing_is_eligible():
    out = ms.select([cand("a", returns(1, mu=-0.002)), cand("b", returns(2, mu=-0.003))])
    assert out["selection_status"] == "no_eligible_candidate" and out["selected"] is None and out["ranking"] == [] and out["counts"]["ineligible"] == 2
    assert "rather than the least bad" in out["reason"]


# ------------------------------------------------------------------------------------------------------------------ ranking, configurable metrics, the policy
def test_the_default_primary_metric_is_the_validation_net_sharpe_and_alternatives_rank_differently():
    rng = np.random.default_rng(11)
    steady = pd.Series(rng.normal(0.0009, 0.006, 300), index=DAYS)                           # a higher Sharpe
    lumpy = pd.Series(rng.normal(0.0016, 0.018, 300), index=DAYS)                            # a higher return
    cs = [cand("steady", steady), cand("lumpy", lumpy)]
    flat = ms.Eligibility(min_net_sharpe=None, max_drawdown=None)
    by = lambda metric: ms.select(cs, ms.SelectionPolicy(primary_metric=metric, eligibility=flat, simplicity=sm.SimplicityPolicy(enabled=False)))["ranking"][0]["id"]       # noqa: E731
    assert ms.SelectionPolicy().primary_metric == "net_sharpe"
    assert by("net_sharpe") == "steady" and by("net_ann_return") == "lumpy"
    assert {by("net_sortino"), by("net_calmar")} <= {"steady", "lumpy"}
    with pytest.raises(ms.SelectionError, match="primary metric"):
        ms.SelectionPolicy(primary_metric="vibes")


def test_ties_in_the_primary_metric_are_broken_by_drawdown_then_turnover():
    r = returns(4)
    out = ms.select([cand("b", r, turnover=50.0), cand("a", r, turnover=10.0)], ms.SelectionPolicy(simplicity=sm.SimplicityPolicy(enabled=False)))
    assert [x["id"] for x in out["ranking"]] == ["a", "b"] and out["selected"] == "a"


def test_policy_from_a_mapping_refuses_unknown_keys_and_keeps_documented_defaults():
    pol = ms.policy_from({"primary_metric": "net_sortino", "eligibility": {"max_drawdown": 0.2}, "simplicity": {"k": 2.0}})
    assert pol.primary_metric == "net_sortino" and pol.eligibility.max_drawdown == 0.2 and pol.eligibility.min_validation_days == 126 and pol.simplicity.k == 2.0
    for bad in ({"vibes": 1}, {"eligibility": {"vibes": 1}}, {"simplicity": {"lambda": 0.1}}, {"secondary": ["net_sharpe"]}, {"pareto": ["nope"]}):
        with pytest.raises(ms.SelectionError):
            ms.policy_from(bad)
    assert ms.policy_from({}).hash() == ms.SelectionPolicy().hash() != pol.hash()


# ------------------------------------------------------------------------------------------------------------------ Pareto against a brute-force definition
def brute_ranks(values, objs, dirs):
    ids = list(values)
    better = lambda a, b: all((values[a][o] >= values[b][o]) if dirs[o] == "max" else (values[a][o] <= values[b][o]) for o in objs) and any(                    # noqa: E731
        (values[a][o] > values[b][o]) if dirs[o] == "max" else (values[a][o] < values[b][o]) for o in objs)
    rank, left, lvl = {}, set(ids), 1
    while left:
        front = {i for i in left if not any(better(j, i) for j in left if j != i)}
        for i in front:
            rank[i] = lvl
        left -= front
        lvl += 1
    return rank


@pytest.mark.parametrize("seed", range(40))
def test_the_pareto_ranks_match_a_brute_force_definition_on_random_data_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 12))
    objs = ["net_sharpe", "max_drawdown", "turnover", "complexity"]
    values = {f"c{i}": {o: float(rng.integers(0, 5)) for o in objs} for i in range(n)}              # small integers: plenty of ties
    out = ms.pareto_analysis(values, objs)
    assert out["ranks"] == brute_ranks(values, objs, ms.DIRECTION)
    assert out["front"] == sorted(i for i, r in out["ranks"].items() if r == 1)
    for i, doms in out["dominated_by"].items():
        assert all(ms.dominates(values[j], values[i], objs) for j in doms) and (not doms) == (out["ranks"][i] == 1)


def test_an_objective_that_one_candidate_lacks_is_dropped_with_the_reason_and_comparisons_are_worded():
    values = {"a": {"net_sharpe": 1.0, "instability": None, "max_drawdown": 0.1}, "b": {"net_sharpe": 0.5, "instability": 0.2, "max_drawdown": 0.3}}
    out = ms.pareto_analysis(values, ["net_sharpe", "instability", "max_drawdown"])
    assert out["objectives"] == ["net_sharpe", "max_drawdown"] and out["dropped"][0]["objective"] == "instability" and "a" in out["dropped"][0]["reason"]
    assert out["front"] == ["a"] and out["dominated_by"]["b"] == ["a"]
    cmp_ = ms.compare("a", "b", {"a": {"net_sharpe": 1.0, "max_drawdown": 0.1}, "b": {"net_sharpe": 0.5, "max_drawdown": 0.3}}, ["net_sharpe", "max_drawdown"])
    assert cmp_["verdict"] == "a dominates b"
    mixed = ms.compare("a", "b", {"a": {"net_sharpe": 1.0, "max_drawdown": 0.4}, "b": {"net_sharpe": 0.5, "max_drawdown": 0.3}}, ["net_sharpe", "max_drawdown"])
    assert mixed["verdict"].startswith("neither dominates") and mixed["better_a"] == ["net_sharpe"] and mixed["better_b"] == ["max_drawdown"]


def test_the_selection_carries_the_pareto_front_and_the_comparison_with_every_other_candidate():
    cs = [cand("fast", returns(1, mu=0.0012), turnover=10.0), cand("slow", returns(2, mu=0.0012), turnover=90.0, kind="hist_gb"), cand("calm", returns(3, mu=0.0009, sd=0.006), turnover=20.0)]
    out = ms.select(cs, ms.SelectionPolicy(simplicity=sm.SimplicityPolicy(enabled=False)))
    assert set(out["pareto"]["objectives"]) >= {"net_sharpe", "max_drawdown", "turnover", "complexity"} and out["pareto"]["front"]
    assert {v["b"] for v in out["against_selected"]} == {c["id"] for c in out["ranking"]} - {out["selected"]}
    assert all(v["a"] == out["selected"] for v in out["against_selected"])


# ------------------------------------------------------------------------------------------------------------------ nothing protected is read
def test_a_candidate_that_carries_test_or_forward_dates_is_refused():
    r = returns(3)
    labels = pd.Series(["validation"] * 250 + ["test"] * 30 + ["forward"] * 20, index=r.index)
    with pytest.raises(ms.SelectionLeakError, match="validation segment only"):
        ms.select([cand("leaky", r, labels=labels)])
    clean = ms.select([cand("fine", r.iloc[:250], labels=labels.iloc[:250])])
    assert clean["scope"]["protected_segments_read"] is False and clean["scope"]["segments"] == ["validation"]


@pytest.fixture(scope="module")
def market():
    return make_bundle(8, 1250, seed=33, alpha=1.0)


BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "signal": {"holding_period": 5}}


@pytest.fixture(scope="module")
def saved(tmp_path_factory, market):
    reg = rg.Registry(tmp_path_factory.mktemp("sel") / "ml")
    ids = {}
    for name, model in (("ridge", {"kind": "ridge", "params": {"alpha": 10.0}}), ("lasso", {"kind": "lasso", "params": {"alpha": 0.001}}), ("forest", {"kind": "random_forest", "params": {"n_estimators": 20, "max_depth": 3}})):
        ids[name] = xp.run_experiment({**BASE, "name": name, "model": model}, bundle=market, registry=reg, explain=False, robustness=False, final_model=False).id
    return reg, ids


def test_selecting_saved_experiments_reads_validation_rows_only_and_ignores_tampered_protected_rows(saved):
    reg, ids = saved
    first = ms.select_saved(reg, list(ids.values()), ms.SelectionPolicy(eligibility=ms.Eligibility(min_net_sharpe=None)), store=False)
    assert first["scope"]["protected_segments_read"] is False and first["origin"]["ignored_rows"], "the finalised experiments carry test and forward rows, which are counted and not read"
    assert set(next(iter(first["origin"]["ignored_rows"].values()))) >= {"test", "forward"}
    path = reg.root / "experiments" / ids["ridge"] / "returns.parquet"
    original = pd.read_parquet(path)
    tampered = original.copy()
    tampered.loc[tampered["segment"].isin(["test", "forward"]), "net"] = 5.0                  # absurd outcomes in the protected segments
    tampered.to_parquet(path)
    try:
        again = ms.select_saved(reg, list(ids.values()), ms.SelectionPolicy(eligibility=ms.Eligibility(min_net_sharpe=None)), store=False)
    finally:
        original.to_parquet(path)
    keep = lambda out: ([(c["id"], c["series_hash"], c["metrics"]["net_sharpe"]) for c in out["candidates"]], out["ranking"], out["selected"])                # noqa: E731
    assert keep(first) == keep(again), "changing the test and forward outcomes changed nothing in the selection"


def test_a_selection_of_saved_experiments_is_recorded_with_its_policy_and_is_repeatable(saved):
    reg, ids = saved
    pol = ms.SelectionPolicy(eligibility=ms.Eligibility(min_net_sharpe=None))
    out = ms.select_saved(reg, list(ids.values()), pol, actor="tester")
    assert out["selection_id"].startswith("msel-") and out["record_seq"] > 0
    stored = ms.stored_selection(reg, out["selection_id"])
    assert stored["policy_hash"] == pol.hash() and stored["selected"] == out["selected"] and stored["scope"]["protected_segments_read"] is False
    assert ms.select_saved(reg, list(ids.values()), pol, store=False)["selection_id"] == out["selection_id"], "the same candidates under the same policy are the same selection"
    assert ms.select_saved(reg, list(ids.values()), ms.SelectionPolicy(eligibility=ms.Eligibility(min_net_sharpe=None, max_drawdown=0.9)), store=False)["selection_id"] != out["selection_id"]
    with pytest.raises(ms.SelectionError):
        ms.select_saved(reg, [], pol)
    gone = ms.select_saved(reg, [ids["ridge"], "no-such-experiment"], pol, store=False)
    assert {c["id"]: c["state"] for c in gone["candidates"]}["no-such-experiment"] == "failed", "an unreadable experiment is a failed candidate, not a failed selection"


def test_the_selection_record_persists_the_evidence_rows_and_the_outer_windows_it_was_made_from(saved):
    reg, ids = saved
    out = ms.select_saved(reg, list(ids.values()), ms.SelectionPolicy(eligibility=ms.Eligibility(min_net_sharpe=None)), store=False)
    assert all(c["state"] == "eligible" for c in out["candidates"]), [c["reason"] for c in out["candidates"]]
    hashes = {c["rows"]["index_hash"] for c in out["candidates"]}
    assert hashes == set(out["evidence_rows"]) and len(hashes) == 1, "the same validation days, stored once"
    dates = out["evidence_rows"][next(iter(hashes))]
    assert len(dates) == out["candidates"][0]["rows"]["n"] > 100 and dates == sorted(dates) and dates[0] == out["candidates"][0]["rows"]["first"][:10]
    lin = out["candidates"][0]["lineage"]
    assert lin["checked"] and lin["n_folds"] >= 3 and lin["violations"] == [] and all(f["train_rows_hash"] and f["fit_cutoff"] and f["label_end_max"] for f in lin["folds"])
    assert all(f["label_end_max"] <= f["fit_cutoff"] for f in lin["folds"]), "independently re-asserted from the recorded facts"
    assert out["predeclaration"]["predeclared"] is None


def test_the_candidates_of_a_saved_experiment_carry_the_evidence_the_gates_need(saved):
    reg, ids = saved
    c = ms.candidate_from_experiment(reg, ids["ridge"])
    assert c["coverage"] and c["coverage"] > 0.8 and c["n_assets"] == 8 and c["exposure"]["gross_max"] > 0 and c["turnover"] > 0
    assert c["liquidity"]["status"] == "not_established" and c["complexity"]["inference_latency"] is not None and c["complexity"]["training_cost"] is not None
    assert (c["labels"] == "validation").all() and len(c["returns"]) == len(c["labels"])
    cut = ms.candidate_from_experiment(reg, ids["ridge"], until=c["returns"].index[100])
    assert len(cut["returns"]) == 101 and cut["returns"].index.max() == c["returns"].index[100]
    assert ms.candidate_from_experiment(reg, ids["forest"])["complexity"]["model_complexity"] > ms.candidate_from_experiment(reg, ids["ridge"])["complexity"]["model_complexity"]


# ------------------------------------------------------------------------------------------------------------------ the search and the tuning
def test_a_search_without_selection_is_unchanged_and_with_it_adds_the_constrained_choice(market, tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    plain = ss.search(BASE, bundle=market, registry=reg, kinds=("ridge", "lasso"), n_random=4, n_bayes=0, generations=0, seed=5, study_name="plain")
    assert "model_selection" not in plain
    out = ss.search(BASE, bundle=market, registry=reg, kinds=("ridge", "lasso"), n_random=4, n_bayes=0, generations=0, seed=5, study_name="chosen",
                    selection={"eligibility": {"min_net_sharpe": None}})
    sel = out["model_selection"]
    assert sel["n_candidates"] == out["n_trials"] == 4 and sel["scope"]["protected_segments_read"] is False and sel["origin"]["n_trials"] == 4
    assert sel["selection_status"] == "selected" and out["best"] is not None and sel["selected_trial"] is not None and sel["highest_ranked_trial"] is not None
    assert all(r["evidence"]["exposure"] and r["evidence"]["coverage"] for r in out["trials"] if not r.get("error"))
    assert ms.stored_selection(reg, sel["selection_id"])["policy_hash"] == sel["policy_hash"]


def test_the_trials_a_selection_sees_are_validation_days_only(market, tmp_path):
    from src.ml import optimization as opt
    from src.ops.hpo import LogFloat

    runner = opt.TrialRunner(BASE, bundle=market, registry=None, study="t")
    runner.evaluate({"model.params.alpha": 5.0})
    net = runner.series[0]
    exp = xp.run_experiment({**BASE, "model": {"kind": "ridge", "params": {"alpha": 5.0}}}, bundle=market, explain=False, robustness=False, final_model=False, stop_after="validation")
    bounds = exp.train.plan.bounds["validation"]
    assert net.index.min() >= exp.train.plan.dates[bounds[0]] and net.index.max() <= exp.train.plan.dates[bounds[1] - 1] + pd.Timedelta(days=10), "the kept series ends with the validation segment"
    out = opt.tune(BASE, {"model.params.alpha": LogFloat(0.1, 100.0)}, n_trials=4, method="random", seed=2, bundle=market, selection={"eligibility": {"min_net_sharpe": None}})
    assert out["model_selection"]["n_candidates"] == 4 and out["best"]["number"] in range(4)
    assert "model_selection" not in opt.tune(BASE, {"model.params.alpha": LogFloat(0.1, 100.0)}, n_trials=3, method="random", seed=2, bundle=market)


def test_finalists_follow_the_selection_and_fall_back_to_the_composite_score_without_one():
    good = [{"number": n, "score": 1.0 - n / 10} for n in range(5)]
    chosen = {"selected_trial": 3, "ranking": [{"id": "trial-1"}, {"id": "trial-3"}, {"id": "trial-0"}]}
    assert [r["number"] for r in ss.finalist_order(good, chosen, 2)] == [3, 1]
    assert ss.finalist_order(good, None, 2) == [] and ss.finalist_order(good, {"selected_trial": None, "ranking": []}, 2) == []
    assert [r["number"] for r in ss.finalist_order(good, chosen, 9)] == [3, 1, 0], "trials the search did not keep as good are not frozen"


# ------------------------------------------------------------------------------------------------------------------ the whole pipeline on controlled candidates
def test_the_pipeline_filters_then_ranks_then_prefers_the_simpler_of_two_that_are_not_distinguishable():
    rng = np.random.default_rng(21)
    x = rng.normal(0.0, 0.01, 300)
    deep = pd.Series(x - x.mean() + 0.0011, index=DAYS)                                      # a pinned sample mean, so that every candidate below is comfortably profitable

    def near(shift, sd=0.0015):                                                              # the same days as ``deep`` minus a known shift: the paired gap is the shift
        noise = rng.normal(0, sd, 300)
        return pd.Series(deep.to_numpy() - shift + noise - noise.mean(), index=DAYS)

    loser = pd.Series(rng.normal(-0.002, 0.01, 300), index=DAYS)
    cs = [cand("deep", deep, kind="lstm", complexity=sm.complexity_vector("lstm", None, n_features=40, train_seconds=300.0, inference_ms=5.0)),
          cand("simple", near(0.00002), kind="ridge"),                                       # a hair behind: not distinguishable from the best
          cand("cheap_but_worse", near(0.0006), kind="ridge", complexity=sm.complexity_vector("ridge", None, n_features=3, train_seconds=0.1, inference_ms=0.1)),            # simplest of all, clearly behind
          cand("loser", loser, kind="ridge")]
    out = ms.select(cs)
    assert out["counts"] == {"eligible": 3, "ineligible": 1, "insufficient": 0, "failed": 0}
    assert out["highest_ranked"] == "deep" and out["selected"] == "simple" and out["changed_by_simplicity"] is True
    status = {r["id"]: r["statistical_status"] for r in out["simplicity"]["table"]}
    assert status["simple"] == "not_distinguishable_by_heuristic" and status["cheap_but_worse"] == "worse_than_the_heuristic_allows", status
    assert out["simplicity"]["claim"]["kind"] == "selection_heuristic" and out["claims"]["statistical_equivalence"] == "not claimed"
    assert [x["id"] for x in out["ranking"]][0] == "deep" and "loser" not in {x["id"] for x in out["ranking"]}
    off = ms.select(cs, ms.SelectionPolicy(simplicity=sm.SimplicityPolicy(enabled=False)))
    assert off["selected"] == "deep" and off["simplicity"] is None


def test_every_threshold_is_a_documented_field_of_the_policy():
    import re
    from pathlib import Path

    page = Path(__file__).resolve().parents[1] / "docs" / "ml_platform.md"
    if not page.exists():
        pytest.skip("NOT_APPLICABLE: the documentation is not shipped in this tree (the Docker image is built without docs)")
    text = page.read_text()
    names = list(ms.Eligibility.__dataclass_fields__) + list(ms.SelectionPolicy.__dataclass_fields__) + list(sm.SimplicityPolicy.__dataclass_fields__) + list(ms.NestedPolicy.__dataclass_fields__)
    missing = [n for n in names if not re.search(rf"\b{re.escape(n)}\b", text)]
    assert not missing, f"undocumented policy fields: {missing}"
    assert len(list(itertools.chain(names))) > 25
