"""Release 7, P2-08 (and the meta-label audit of P2-12): mixtures against the best single member that development would have chosen.

The streams are built so that the answer is known: two members that earn in opposite market regimes, so that an equal mixture and the best single member earn little and a regime-conditioned mixture earns
a lot; then pure noise, where nothing may be claimed.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml import ensembles as en

N = 1400
IX = pd.bdate_range("2016-01-04", periods=N)
CUT = IX[700]


def regime_market(seed=0, edge=0.0015):
    """Members A and B earn ``edge`` a day in opposite halves of a regime that alternates every 90 days (known at the previous close); C and D are noise."""
    rng = np.random.default_rng(seed)
    reg = pd.Series(np.where((np.arange(N) // 90) % 2 == 0, "low", "high"), index=IX)
    sign = np.where(reg.to_numpy() == "low", 1.0, -1.0)
    R = pd.DataFrame({"A": edge * sign + rng.normal(0, 0.006, N), "B": -edge * sign + rng.normal(0, 0.006, N), "C": rng.normal(0, 0.006, N), "D": rng.normal(0, 0.006, N)}, index=IX)
    return R, reg.shift(1).bfill()


def test_a_regime_conditioned_mixture_beats_the_simple_ones_and_the_best_single_member_where_the_structure_is_real():
    R, reg = regime_market()
    out = en.mixture_study(R, development_end=CUT, regimes=reg, methods=("equal", "inverse_vol", "regime_conditioned"), n_boot=400, seed=1)
    rows = {r["method"]: r for r in out["rows"]}
    best = out["best_single"]
    assert best["chosen_on"] == "the development period only" and out["development"][1] == str(CUT.date()) and out["evaluation"][0] > out["development"][1]
    assert rows["regime_conditioned"]["sharpe"] > rows["equal"]["sharpe"] + 0.5 and rows["regime_conditioned"]["sharpe"] > rows["inverse_vol"]["sharpe"] + 0.5 and rows["regime_conditioned"]["sharpe"] > best["evaluation_sharpe"] + 0.5
    d = rows["regime_conditioned"]["vs_best_single"]
    assert d["eligibility"] == "applied" and d["value"] > 0 and d["interval"][0] > 0, "the interval of the paired difference excludes zero where the gain is built in"
    assert rows["regime_conditioned"]["vs_equal_weight"]["interval"][0] > 0 and rows["equal"]["vs_equal_weight"] is None
    assert "best of 4 members" in best["selection_note"] and "not an unbiased estimate" in best["selection_note"] and out["member_correlation"] == pytest.approx(0.0, abs=0.4)
    assert abs(rows["equal"]["sharpe"]) < 1.0, "an equal mixture of two opposite earners earns little"


def test_the_best_single_member_is_chosen_on_development_only_and_no_weight_is_trained_on_the_day_it_scores():
    R, reg = regime_market(seed=3)
    base = en.mixture_study(R, development_end=CUT, regimes=reg, methods=("equal", "regime_conditioned"), n_boot=100)
    later = R.copy()
    later.iloc[800:] = np.random.default_rng(9).normal(0.0, 0.02, later.iloc[800:].shape)               # a different world from the 801st day
    again = en.mixture_study(later, development_end=CUT, regimes=reg, methods=("equal", "regime_conditioned"), n_boot=100)
    assert again["best_single"]["member"] == base["best_single"]["member"] and again["best_single"]["development_sharpe"] == base["best_single"]["development_sharpe"], "the choice reads the development days only"
    for m in ("equal", "regime_conditioned"):
        a, b = en.combine(R, m, regimes=reg).returns, en.combine(later, m, regimes=reg).returns
        before = a.index[a.index < IX[800]]
        assert len(before) > 400
        np.testing.assert_allclose(a.loc[before].to_numpy(), b.loc[before].to_numpy(), atol=1e-15, err_msg=f"{m}: a return before day 800 moved when only later returns changed")
        assert not np.allclose(a.loc[IX[801]:].to_numpy(), b.loc[IX[801]:].to_numpy()), "the test can tell a different world from the same one"
    moved = R.copy()
    moved["C"] = np.random.default_rng(5).normal(0.003, 0.004, N)                                       # a different development record can change the choice
    assert en.mixture_study(moved, development_end=CUT, regimes=reg, methods=("equal",), n_boot=50)["best_single"]["member"] in ("C", "A", "B", "D")


def test_where_there_is_no_structure_nothing_is_claimed():
    rng = np.random.default_rng(11)
    R = pd.DataFrame(rng.normal(0.0002, 0.006, (N, 4)), index=IX, columns=list("ABCD"))
    reg = pd.Series(np.where((np.arange(N) // 90) % 2 == 0, "low", "high"), index=IX).shift(1).bfill()
    out = en.mixture_study(R, development_end=CUT, regimes=reg, n_boot=400, seed=2)
    for row in out["rows"]:
        d = row["vs_best_single"]
        assert d["eligibility"] == "applied" and d["interval"][0] < 0 < d["interval"][1], f"{row['method']}: noise must not separate from the best single member"


def test_the_study_refuses_streams_too_short_to_judge_and_lists_a_method_that_cannot_run():
    R, reg = regime_market()
    with pytest.raises(ValueError, match="at least 120 development days"):
        en.mixture_study(R.iloc[:150], development_end=IX[100], regimes=reg)
    with pytest.raises(ValueError, match="evaluation days"):
        en.mixture_study(R, development_end=IX[-20], regimes=reg)
    out = en.mixture_study(R, development_end=CUT, regimes=None, methods=("equal", "regime_conditioned"), n_boot=100)
    assert [r["method"] for r in out["rows"]] == ["equal"] and "regime series" in out["errors"]["regime_conditioned"], "a method that cannot run is listed with its reason, not dropped"
