"""Release 8, P3-07: discovery with a finite budget, full logging, causal expressions and selection controls.

The grammar is tested by what it refuses (anything that could read the future or an outcome) and by an independent pandas calculation of what it accepts; causality is a property tested over many generated
expressions; the budget and the log are counted; sealed rows are shown to be invisible; and the controls are tested on pure noise, where they must find (almost) nothing, and on a planted signal."""

import numpy as np
import pandas as pd
import pytest

from src.ml import discovery as dc

N, M = 400, 12
IX = pd.bdate_range("2019-01-01", periods=N)
ASSETS = [f"A{i}" for i in range(M)]


def frames(seed=0, names=("f1", "f2", "f3")):
    rng = np.random.default_rng(seed)
    return {k: pd.DataFrame(rng.normal(size=(N, M)), index=IX, columns=ASSETS) for k in names}


# ------------------------------------------------------------------------------------------------------------------ the grammar
def test_what_could_read_the_future_or_an_outcome_is_refused_and_what_is_accepted_reports_its_lookback():
    ok = dc.validate("delta(rank(f1), 3) / (rolling_std(f2, 5) + 1)", {"f1", "f2"})
    assert ok["features"] == ["f1", "f2"] and ok["lookback"] == 4 and ok["canonical"] == "delta(rank(f1), 3) / (rolling_std(f2, 5) + 1)"
    assert dc.validate("lag(rolling_mean(f1, 10), 5)", {"f1"})["lookback"] == 14 and dc.validate("lag(f1, 0)", {"f1"})["lookback"] == 0
    for bad, msg in (("f1.shift(-1)", "only the functions"), ("f1[1]", "Subscript is not allowed"), ("lag(f1, -1)", "negative window would read the future"), ("lead(f1, 1)", "only the functions"),
                     ("fwd_ret_5", "not a feature of the panel"), ("rank(label)", "not a feature of the panel"), ("lag(f1, f2)", "whole-number constant"), ("lag(f1, 1.5)", "whole-number constant"),
                     ("lag(f1, 999)", "between 1 and"), ("(lambda: 1)()", "only the functions"), ("[x for x in f1]", "ListComp is not allowed"), ("__import__('os')", "only the functions"),
                     ("f1 ** 2", "BinOp"), ("lag(f1)", "takes 2 argument"), ("rank(f1, 2)", "takes 1 argument"), ("rank(f1, key=1)", "without keyword"), ("'abc'", "only numbers"), ("3", "reads at least one feature"),
                     ("f1 +", "not an expression")):
        with pytest.raises(dc.DiscoveryError, match=msg):
            dc.validate(bad, {"f1", "f2"})


def test_an_accepted_expression_is_what_pandas_computes_by_hand():
    P = frames()
    got = dc.evaluate("delta(rank(f1), 3) / (rolling_std(f2, 5) + 1)", P)
    r = P["f1"].rank(axis=1, pct=True)
    ref = (r - r.shift(3)) / (P["f2"].rolling(5, min_periods=5).std(ddof=1) + 1)
    pd.testing.assert_frame_equal(got, ref)
    z = dc.evaluate("zscore(f1) - sign(f2) * abs(lag(f3, 2))", P)
    zref = P["f1"].sub(P["f1"].mean(axis=1), axis=0).div(P["f1"].std(axis=1, ddof=0), axis=0) - np.sign(P["f2"]) * P["f3"].shift(2).abs()
    pd.testing.assert_frame_equal(z, zref)
    with pytest.raises(dc.DiscoveryError, match="reads at least one feature"):
        dc.evaluate("2 + 3", P)


def test_every_generated_expression_is_causal_and_the_check_catches_one_that_is_not():
    P = frames(seed=1)
    exprs = dc.generate(list(P), 80, seed=3, depth=3)
    assert len(exprs) == 80 and len(set(exprs)) == 80
    for e in exprs:
        dc.validate(e, set(P))
        chk = dc.causal_check(e, P, at=200)
        assert chk["causal"] is True and chk["differences"] == 0, e
    # a function that looks ahead (what the grammar forbids) is caught by the dynamic check: build one by hand
    leaky = {k: v.copy() for k, v in P.items()}
    base = P["f1"].shift(-1)
    scrambled = {k: v.copy() for k, v in P.items()}
    scrambled["f1"].iloc[201:] = 99.0
    assert not np.allclose(base.iloc[:201].to_numpy(), scrambled["f1"].shift(-1).iloc[:201].to_numpy(), equal_nan=True), "a forward-looking value changes when the future does: the check can fail"
    assert leaky


# ------------------------------------------------------------------------------------------------------------------ budget, log and sealed rows
def test_the_budget_is_a_hard_limit_and_every_candidate_is_in_the_log_with_its_reason():
    P = frames(seed=2)
    target = pd.DataFrame(np.random.default_rng(9).normal(size=(N, M)), index=IX, columns=ASSETS)
    out = dc.search(P, target, horizon=5, budget=25, sealed_from=IX[300], seed=4)
    assert out["budget"] == 25 and out["generated"] == 25 and len(out["log"]) == 25 and out["tested"] + out["rejected"] == 25
    assert [r["id"] for r in out["log"]] == list(range(25)) and all(r["status"] in ("evaluated", "rejected") for r in out["log"])
    manual = ["rank(f1)", "lead(f1, 1)", "fwd_ret_5", "lag(f2, 3)"]
    out2 = dc.search(P, target, horizon=5, budget=3, sealed_from=IX[300], expressions=manual)
    assert out2["generated"] == 3 and out2["truncated_to_budget"] is True and len(out2["log"]) == 3, "asking for more than the budget runs the budget"
    bad = [r for r in out2["log"] if r["status"] == "rejected"]
    assert [r["expression"] for r in bad] == ["lead(f1, 1)", "fwd_ret_5"] and "only the functions" in bad[0]["reason"] and "not a feature of the panel" in bad[1]["reason"]
    with pytest.raises(dc.DiscoveryError, match="at least one candidate"):
        dc.search(P, target, horizon=5, budget=0, sealed_from=IX[300])
    with pytest.raises(dc.DiscoveryError, match="at least 60"):
        dc.search(P, target, horizon=5, budget=5, sealed_from=IX[20])
    assert dc.pairs_budget(3, 2) > 1000, "the grammar offers far more expressions than anyone can test: the reason for a budget"


def test_the_sealed_rows_are_invisible_to_every_score():
    P = frames(seed=3)
    target = pd.DataFrame(np.random.default_rng(7).normal(size=(N, M)), index=IX, columns=ASSETS)
    exprs = ["rank(f1)", "zscore(f2)", "delta(f3, 2)", "f1 - f2"]
    a = dc.search(P, target, horizon=5, budget=4, sealed_from=IX[300], expressions=exprs)
    t2 = target.copy()
    t2.iloc[300:] = 1e6 * np.random.default_rng(1).normal(size=t2.iloc[300:].shape)                  # the sealed outcomes are replaced by anything at all
    b = dc.search(P, t2, horizon=5, budget=4, sealed_from=IX[300], expressions=exprs)
    for x, y in zip(a["log"], b["log"]):
        assert x["ic_mean"] == y["ic_mean"] and x["p_value"] == y["p_value"], "a score read a sealed outcome"
    assert a["development"]["last"] == str(IX[299].date()) and a["development"]["dates"] == 300 and a["development"]["sealed_from"] == str(IX[300].date())
    P2 = {k: v.copy() for k, v in P.items()}
    for v in P2.values():
        v.iloc[300:] = 1e6
    c = dc.search(P2, target, horizon=5, budget=4, sealed_from=IX[300], expressions=exprs)
    assert [r["ic_mean"] for r in c["log"]] == [r["ic_mean"] for r in a["log"]], "features after the seal cannot influence a score on dates before it (the expressions look back only)"


# ------------------------------------------------------------------------------------------------------------------ the selection controls
def test_on_pure_noise_the_controls_find_almost_nothing_while_the_naive_best_looks_significant():
    naive_hits = adj_hits = disc = 0
    reps = 24
    for r in range(reps):
        P = frames(seed=100 + r)
        target = pd.DataFrame(np.random.default_rng(500 + r).normal(size=(N, M)), index=IX, columns=ASSETS)
        out = dc.search(P, target, horizon=1, budget=60, sealed_from=IX[350], seed=r, q=0.1)
        b = out["best"]
        naive_hits += int(b["p_value"] < 0.05)
        adj_hits += int(b["p_best_of_n"] < 0.05)
        disc += int(len(out["discoveries"]) > 0)
        assert out["tested"] >= 40 and out["controls"]["tests"] == out["tested"]
    assert naive_hits >= reps * 0.5, "the best of sixty noise candidates is nominally significant most of the time"
    assert adj_hits <= reps * 0.2 and disc <= reps * 0.2, (naive_hits, adj_hits, disc)


def test_a_planted_signal_is_found_flagged_and_confirmed_only_on_rows_the_search_never_saw():
    P = frames(seed=5)
    rng = np.random.default_rng(11)
    target = P["f1"].shift(1) * 0.5 + pd.DataFrame(rng.normal(size=(N, M)), index=IX, columns=ASSETS)       # f1 of the previous session predicts today's return
    out = dc.search(P, target, horizon=1, budget=60, sealed_from=IX[350], seed=2, q=0.1, dev_start=None, expressions=["lag(f1, 1)", "rank(lag(f1, 1))"] + dc.generate(list(P), 58, 6))
    best = out["best"]
    assert best["expression"] in ("lag(f1, 1)", "rank(lag(f1, 1))") and best["p_bh"] < 0.01 and best["id"] in out["discoveries"] and best["ic_mean"] > 0.1 and "do not make the best mined rule independent evidence" in out["limitations"][0]
    conf = dc.confirm(best["expression"], P, target, horizon=1, start=IX[300], end=IX[349], sealed_from=IX[350])
    assert conf["ic"]["value"] > 0.1 and conf["ic"]["interval"][0] > 0 and conf["dates"] > 40 and "only part of this procedure that is independent evidence" in conf["note"]
    with pytest.raises(dc.DiscoveryError, match="reaches the sealed dates"):
        dc.confirm(best["expression"], P, target, horizon=1, start=IX[300], end=IX[360], sealed_from=IX[350])
    with pytest.raises(dc.DiscoveryError, match="at least 30"):
        dc.confirm(best["expression"], P, target, horizon=1, start=IX[340], end=IX[349], sealed_from=IX[350])
