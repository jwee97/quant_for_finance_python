"""Release 5, P1-19: predeclared regime and stress evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.ml import regime_eval as rg

N = 1500
IDX = pd.bdate_range("2012-01-02", periods=N)


def market(seed=0):
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(N) > 900, 0.02, 0.008)                                           # a high-volatility second half
    drift = np.where((np.arange(N) > 300) & (np.arange(N) < 700), 0.002, 0.0)                  # a long rally in the middle
    return pd.Series(rng.normal(drift, vol), index=IDX)


def test_the_declaration_is_frozen_by_its_hash_and_every_regime_says_whether_it_is_causal():
    a = rg.declare(rg.default_specs({"gfc": ("2013-03-01", "2013-06-30")}))
    b = rg.declare(rg.default_specs({"gfc": ("2013-03-01", "2013-06-30")}))
    c = rg.declare(rg.default_specs({"gfc": ("2013-03-01", "2013-07-31")}))
    assert a["hash"] == b["hash"] != c["hash"], "any change of a definition is a different declaration"
    flags = {s["name"]: s["causal"] for s in a["specs"]}
    assert flags["trend_63"] and flags["volatility_63"] and flags["liquidity_20"] and not flags["crisis:gfc"]
    with pytest.raises(ValueError, match="unique"):
        rg.declare([rg.RegimeSpec("x", "trend"), rg.RegimeSpec("x", "volatility")])
    with pytest.raises(ValueError):
        rg.RegimeSpec("bad", "sentiment")


def test_causal_labels_use_only_the_past_and_the_volatility_terciles_are_those_seen_so_far():
    m = market(1)
    spec = rg.RegimeSpec("trend_63", "trend", {"window": 63, "threshold": 0.05})
    lab = rg.label(spec, m)
    k = 1000
    m2 = m.copy(); m2.iloc[k:] = m2.iloc[k:] * -4
    assert rg.label(spec, m2).iloc[: k + 1].equals(lab.iloc[: k + 1]) or (rg.label(spec, m2).iloc[: k + 1].fillna("-") == lab.iloc[: k + 1].fillna("-")).all()
    assert lab.iloc[:63].isna().all(), "no label until the window exists"
    vspec = rg.RegimeSpec("volatility_63", "volatility", {"window": 63, "min_history": 252})
    v = rg.label(vspec, m)
    before, after = (v.iloc[300:880] == "high").mean(), (v.iloc[1000:1100] == "high").mean()
    assert after > 0.9 and after - before > 0.35, "volatility is high relative to what had been seen so far: far more often just after the shift than before it"
    assert (rg.label(rg.RegimeSpec("l", "liquidity", {"short": 20, "long": 252, "ratio": 0.7}), m).isna()).all(), "no volume: the definition is unavailable, not 'normal'"


def test_evaluation_reads_each_declared_regime_flags_thin_ones_as_insufficient_and_marks_retrospective_slices():
    m = market(2)
    strat = 0.5 * m + pd.Series(np.random.default_rng(3).normal(0.0004, 0.003, N), index=IDX)
    decl = rg.declare(rg.default_specs({"short_crisis": ("2014-01-02", "2014-01-30")}))
    out = rg.evaluate(strat, m, decl, min_days=60)
    assert out["declaration_hash"] == decl["hash"] and out["n_applied"] > 0
    rows = {(r["regime"], r["state"]): r for r in out["rows"]}
    crisis = rows[("crisis:short_crisis", "short_crisis")]
    assert crisis["eligibility"] == "insufficient" and "fewer than the 60" in crisis["reason"] and crisis["causal"] is False and "retrospective" in crisis["note"]
    liq = rows[("liquidity_20", None)]
    assert liq["eligibility"] == "insufficient" and liq["n_days"] == 0, "no volume given: the liquidity regime is listed as insufficient, not dropped"
    up = rows[("trend_63", "up")]
    assert up["eligibility"] == "applied" and up["mean_interval"] and up["sharpe_hac_interval"] and up["sharpe_bootstrap_interval"] and up["sharpe_naive"] is not None
    assert any("retrospective" in n for n in out["notes"]) and out["n_insufficient"] >= 2
    st_in = {r["state"]: r["n_days"] for r in out["rows"] if r["regime"] == "volatility_63"}
    assert set(st_in) <= {"low", "mid", "high"} and sum(st_in.values()) <= N
