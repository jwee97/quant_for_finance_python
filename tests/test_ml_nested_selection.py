"""Release 5, P1-15: nested chronological selection.

Candidates are synthetic score panels with a known ranking of skill that CHANGES over time, so what an as-of selection should choose at each cutoff is known. The audit is exercised on honest and on tampered records."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.ml import nested as ns

N, A, H = 900, 8, 5
IDX = pd.bdate_range("2016-01-04", periods=N)
COLS = [f"S{i}" for i in range(A)]


def world(seed=0):
    rng = np.random.default_rng(seed)
    fwd = pd.DataFrame(rng.normal(0, 0.01, (N, A)), index=IDX, columns=COLS)
    noise = lambda: pd.DataFrame(rng.normal(0, 1, (N, A)), index=IDX, columns=COLS)           # noqa: E731
    z = fwd / 0.01
    early = z * 0.6 + noise()
    skill_a = np.where(np.arange(N) < 450, 0.0, 1.0)[:, None]                                  # candidate "a" is skilled in the first half, "b" in the second
    a = early * 0 + (z * 1.0 * (1 - skill_a)) + noise() * 0.8
    b = (z * 1.0 * skill_a) + noise() * 0.8
    c = noise()                                                                                 # pure noise
    ends = pd.Series(IDX, index=IDX).shift(-H).fillna(IDX[-1] + pd.Timedelta(days=7))
    return {"a": a, "b": b, "c": c}, fwd, ends


def test_the_choice_at_each_cutoff_follows_what_was_known_then_and_the_chain_is_the_record():
    cands, fwd, ends = world(1)
    cutoffs = [IDX[300], IDX[420], IDX[560], IDX[700]]
    res = ns.select_as_of(cands, fwd, ends, cutoffs, inner_days=150, min_inner=60)
    chosen = {s["cutoff"]: s["chosen"] for s in res["steps"] if s["state"] == "selected"}
    assert chosen[str(IDX[300].date())] == "a" and chosen[str(IDX[420].date())] == "a", "before the switch the history says a"
    assert chosen[str(IDX[700].date())] == "b", "after the switch has been seen for long enough the history says b"
    assert res["eligibility"] == "applied" and res["n_selected"] == 4
    chain = res["chain"]
    assert chain.iloc[:301].isna().all().all() and chain.loc[IDX[301]].notna().all(), "the first outer window starts after the first cutoff; nothing is predicted before it"
    np.testing.assert_allclose(chain.loc[IDX[301]:IDX[420]].to_numpy(), cands["a"].loc[IDX[301]:IDX[420]].to_numpy())
    assert res["choices"] == [chosen[k] for k in sorted(chosen)]


def test_the_audit_recomputes_the_facts_from_the_label_intervals_and_passes_an_honest_record():
    cands, fwd, ends = world(2)
    res = ns.select_as_of(cands, fwd, ends, [IDX[300], IDX[500], IDX[700]], inner_days=150, min_inner=60, until=IDX[800])
    a = ns.audit(res, ends)
    assert a["ok"] and a["steps_checked"] == 3
    for s in res["steps"]:
        assert pd.Timestamp(s["max_inner_label_end"]) <= pd.Timestamp(s["cutoff"]) and pd.Timestamp(s["outer_first"]) > pd.Timestamp(s["cutoff"]) and s["inner_hash"] and s["outer_hash"]
    assert res["steps"][-1]["outer_last"] <= str(IDX[800].date()), "nothing beyond the development window is read"
    fold_ok = {"a": [{"fit_cutoff": "2016-06-01", "test_start": "2016-06-02", "skipped": False}]}
    fold_bad = {"a": [{"fit_cutoff": "2016-06-03", "test_start": "2016-06-02", "skipped": False}]}
    assert ns.audit(res, ends, fold_ok)["ok"] and not ns.audit(res, ends, fold_bad)["ok"]


def test_a_tampered_record_fails_the_audit_in_each_of_the_ways_it_can_be_wrong():
    cands, fwd, ends = world(3)
    res = ns.select_as_of(cands, fwd, ends, [IDX[300], IDX[500]], inner_days=150, min_inner=60)
    leaky = copy.deepcopy(res)
    leaky["steps"][0]["inner_dates"].append(str(IDX[302].date()))                              # a date whose label ends AFTER the cutoff
    bad = ns.audit(leaky, ends)
    assert not bad["ok"] and any("label ends after the cutoff" in v for v in bad["violations"]) and any("hash does not match" in v for v in bad["violations"])
    swapped = copy.deepcopy(res)
    swapped["steps"][1]["outer_dates"] = [str(d.date()) for d in IDX[250:300]]
    assert any("starts on or before the cutoff" in v or "overlaps" in v for v in ns.audit(swapped, ends)["violations"])
    beyond = copy.deepcopy(res); beyond["until"] = str(IDX[400].date())
    assert any("beyond the development window" in v for v in ns.audit(beyond, ends)["violations"])


def test_changing_anything_after_a_cutoff_cannot_change_the_choices_made_up_to_it():
    cands, fwd, ends = world(4)
    cutoffs = [IDX[300], IDX[450], IDX[600], IDX[750]]
    base = ns.select_as_of(cands, fwd, ends, cutoffs, inner_days=150, min_inner=60)
    pert = {k: v.copy() for k, v in cands.items()}
    f2 = fwd.copy()
    for k in pert:
        pert[k].iloc[600:] = np.random.default_rng(9).normal(0, 5, pert[k].iloc[600:].shape)
    f2.iloc[600:] = np.random.default_rng(10).normal(0, 0.05, f2.iloc[600:].shape)
    after = ns.select_as_of(pert, f2, ends, cutoffs, inner_days=150, min_inner=60)
    first3 = lambda r: [(s["cutoff"], s.get("chosen"), s.get("scores")) for s in r["steps"][:2]]            # noqa: E731  cutoffs at 300 and 450: all their inner rows ended before row 600
    assert first3(base) == first3(after)


def test_ties_go_to_the_candidate_declared_first_and_too_little_history_is_insufficient_not_a_guess():
    idx = IDX[:400]
    rng = np.random.default_rng(5)
    fwd = pd.DataFrame(rng.normal(0, 0.01, (400, A)), index=idx, columns=COLS)
    s = fwd * 100
    ends = pd.Series(idx, index=idx).shift(-H).fillna(idx[-1] + pd.Timedelta(days=7))
    res = ns.select_as_of({"simple": s, "complex": s.copy()}, fwd, ends, [idx[300]], inner_days=200, min_inner=60)
    assert res["steps"][0]["chosen"] == "simple", "identical scores: the declared-first (simplest) candidate wins"
    near = ns.select_as_of({"simple": s * 0.999 + rng.normal(0, 1e-3, s.shape), "complex": s}, fwd, ends, [idx[300]], inner_days=200, min_inner=60, tol=0.05)
    assert near["steps"][0]["chosen"] == "simple", "within the declared tolerance of the best, the simpler one is kept"
    early = ns.select_as_of({"x": s}, fwd, ends, [idx[40]], inner_days=200, min_inner=60)
    assert early["eligibility"] == "insufficient" and early["steps"][0]["state"] == "insufficient"
    with pytest.raises(ValueError, match="strictly increasing"):
        ns.select_as_of({"x": s}, fwd, ends, [idx[200], idx[100]])
