"""The Stock Universe Research Lab on synthetic worlds whose answers are known by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.universelab import discover as dc
from src.universelab import membership as mb
from src.universelab import robust, screens as sc, study, walk


def world(n=1500, k=14, seed=0, drift=0.0004, common=0.008, idio=0.012):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    mkt = rng.normal(drift, common, n)
    return pd.DataFrame(mkt[:, None] + rng.normal(0, idio, (n, k)), idx, [f"S{i:02d}" for i in range(k)])


# ---------------------------------------------------------------------------------------------------------------- membership
def test_a_listings_table_gives_exact_point_in_time_eligibility_and_says_when_it_is_partial():
    csv = "ticker,start,end,status\nAAA,2016-01-04,,active\nBBB,2016-01-04,2017-06-30,delisted\nCCC,2017-01-02,,active\n"
    m = mb.from_csv(csv)
    el, info = mb.eligibility(pd.bdate_range("2016-01-04", "2018-01-01"), ["AAA", "BBB", "CCC"], m)
    assert el.loc["2016-06-01", ["AAA", "BBB", "CCC"]].tolist() == [True, True, False]       # CCC was not yet listed
    assert el.loc["2017-07-03", ["AAA", "BBB", "CCC"]].tolist() == [True, False, True]       # BBB had left, CCC had joined
    assert info["basis"] == "point_in_time" and info["hindsight"] is False and "survivorship" in info["limitation"]
    assert mb.eligibility(pd.bdate_range("2016-01-04", periods=5), ["AAA"], mb.from_csv(csv, "complete"))[1]["limitation"] is None


def test_without_a_table_the_universe_is_labelled_as_chosen_with_hindsight():
    el, info = mb.eligibility(pd.bdate_range("2020-01-01", periods=5), ["A", "B"], None)
    assert el.all().all() and info["hindsight"] is True and "survivors" in info["limitation"] and info["basis"] == "current_constituents"


def test_an_empty_status_or_end_cell_means_not_said_and_is_not_an_error():
    m = mb.from_csv("ticker,start,end,status\nAAA,2016-01-04,,\nBBB,2016-01-04,2017-06-30,\n")
    assert {lst.ticker: lst.status for lst in m.listings} == {"AAA": "active", "BBB": "delisted"}


@pytest.mark.parametrize("csv, text", [("ticker\nAAA\n", "needs the columns"), ("ticker,start,end\nAAA,2020-01-01,2019-01-01\n", "not after"), ("ticker,start,status\nAAA,2020-01-01,gone\n", "status"), ("", "could not be read")])
def test_a_bad_membership_table_is_refused_with_the_reason(csv, text):
    with pytest.raises(mb.MembershipError, match=text):
        mb.from_csv(csv)


# ---------------------------------------------------------------------------------------------------------------- screens
def test_screens_keep_the_names_they_say_using_only_the_window():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=300)
    vols = np.array([0.005, 0.008, 0.012, 0.02, 0.03, 0.04])
    w = pd.DataFrame(rng.normal(0, 1, (300, 6)) * vols, idx, list("ABCDEF"))
    assert sc.select(w, None, sc.validate({"kind": "low_vol", "top": 0.5}), None) == ["A", "B", "C"]
    dv = pd.DataFrame(np.tile([9e6, 8e6, 1e6, 7e6, 5e5, 3e6], (300, 1)), idx, list("ABCDEF"))
    assert sc.select(w, dv, sc.validate({"kind": "liquidity", "min_dollar_volume": 2e6}), None) == ["A", "B", "D", "F"]
    assert sc.select(w, dv, sc.validate({"kind": "size", "top": 0.5}), None) == ["A", "B", "D"]
    assert sc.select(w, None, sc.validate({"kind": "sector", "groups": ["tech"]}), {"A": "tech", "C": "tech", "E": "energy"}) == ["A", "C"]
    assert sc.select(w, None, sc.validate({"kind": "manual", "names": ["B", "D", "E"]}), None) == ["B", "D", "E"]
    mom = w.copy()
    mom["C"] = 0.004                                                                       # steady gains: the best 12-1 momentum
    assert sc.select(mom, None, sc.validate({"kind": "momentum", "top": 0.34}), None)[0] == "C"


def test_a_volume_screen_without_volume_is_refused_and_bad_specs_are_rejected():
    w = world(400, 5)
    with pytest.raises(sc.ScreenError, match="volume"):
        sc.select(w, None, sc.validate({"kind": "size", "top": 0.5}), None)
    for bad in ({"kind": "nope"}, {"kind": "low_vol", "top": 0.01}, {"kind": "sector"}, {"kind": "manual", "names": ["A"]}, {"kind": "liquidity", "min_dollar_volume": -1}):
        with pytest.raises(sc.ScreenError):
            sc.validate(bad)


# ---------------------------------------------------------------------------------------------------------------- the walk
def test_the_walk_uses_only_data_known_at_each_decision():
    R = world(900, 10)
    a = walk.run(R, {"kind": "all"}, "min_variance", lookback=252, step=21, cost_bps=10)
    cut = 600
    R2 = R.copy()
    R2.iloc[cut + 1:] = np.random.default_rng(9).normal(0, 0.05, R2.iloc[cut + 1:].shape)      # rewrite the whole future
    b = walk.run(R2, {"kind": "all"}, "min_variance", lookback=252, step=21, cost_bps=10)
    upto = R.index[cut]
    assert np.allclose(a.weights.loc[:upto].to_numpy(), b.weights.loc[:upto].to_numpy())
    assert np.allclose(a.net.loc[:R.index[cut]].dropna().to_numpy(), b.net.loc[:R.index[cut]].dropna().to_numpy())


def test_the_decision_does_not_see_the_last_two_days_before_it_takes_effect():
    """Data through day d is traded at the close of d + 1: weights set at rebalance p never read the returns of days p-1 and p."""
    R = world(900, 10)
    a = walk.run(R, {"kind": "all"}, "min_variance", lookback=252, step=21)
    p = a.weights.index[3]
    pos = R.index.get_loc(p)
    R2 = R.copy()
    R2.iloc[pos - 1:pos + 1] = 0.5                                                          # huge returns on the two days just before the rebalance
    b = walk.run(R2, {"kind": "all"}, "min_variance", lookback=252, step=21)
    assert np.allclose(a.weights.loc[p].to_numpy(), b.weights.loc[p].to_numpy())


def test_costs_are_charged_on_the_trade_and_the_first_trade_is_not_free():
    R = world(900, 12)
    free = walk.run(R, {"kind": "all"}, "equal_weight", cost_bps=0.0)
    paid = walk.run(R, {"kind": "all"}, "equal_weight", cost_bps=10.0)
    d = (free.net - paid.net).fillna(0.0)
    assert d.iloc[0] == pytest.approx(0.001, abs=1e-12) and paid.net.index[0] == free.net.index[0]      # building the book (100% in) costs 10 bp once
    assert d.sum() == pytest.approx(2e-3 * paid.turnover.sum(), rel=1e-9)                              # every trade is paid for at 10 bp one way


def test_a_name_without_history_or_outside_the_membership_is_never_held():
    R = world(900, 8)
    R.loc[:R.index[400], "S00"] = np.nan                                                    # S00 has no prices until day 400
    el = pd.DataFrame(True, R.index, R.columns)
    el.loc[R.index[500]:R.index[700], "S01"] = False                                        # S01 is out of the universe for a stretch
    r = walk.run(R, {"kind": "all"}, "equal_weight", eligible=el, lookback=126, step=21)
    assert r.weights.loc[r.weights.index <= R.index[400 + 126], "S00"].abs().sum() == 0
    gone = r.weights.loc[(r.weights.index > R.index[500 + 2]) & (r.weights.index <= R.index[700 + 2])]
    assert (gone["S01"] == 0).all()


def test_the_hindsight_of_a_stock_that_is_not_yet_a_member_does_not_pay():
    """A stock that rises 5% a day while it is NOT a member must add nothing; the same stock with no membership table is the survivorship trap and does pay."""
    R = world(900, 8, seed=3)
    R["S07"] = R["S07"] * 0.2
    R.loc[R.index[300]:R.index[420], "S07"] = 0.05
    el = pd.DataFrame(True, R.index, R.columns)
    el.loc[:R.index[430], "S07"] = False
    honest = walk.run(R, {"kind": "momentum", "top": 0.3}, "equal_weight", eligible=el, lookback=126, step=21)
    trap = walk.run(R, {"kind": "momentum", "top": 0.3}, "equal_weight", lookback=126, step=21)
    assert honest.weights["S07"].loc[:R.index[430]].abs().sum() == 0
    assert trap.net.sum() > honest.net.sum() + 0.05


# ---------------------------------------------------------------------------------------------------------------- discovery
def test_the_candidate_grid_is_limited_and_a_short_history_is_refused():
    with pytest.raises(ValueError, match="narrow the grid"):
        dc.candidates([{"kind": "low_vol", "top": t} for t in (0.2, 0.3, 0.4, 0.5, 0.6)], ["equal_weight", "min_variance", "hrp", "risk_parity"], [126, 252, 504, 63])
    with pytest.raises(ValueError, match="too short"):
        dc.discover(world(700, 10), [{"kind": "all"}], ["equal_weight"], [126], folds=4, min_train_days=504)


def test_each_block_is_judged_on_days_the_choice_never_saw():
    R = world(1500, 12)
    args = ([{"kind": "all"}, {"kind": "low_vol", "top": 0.5}], ["equal_weight", "min_variance"], [126, 252])
    a = dc.discover(R, *args, folds=3, min_train_days=504, embargo=5)
    block = a["folds"][1]
    lo, hi = pd.Timestamp(block["from"]), pd.Timestamp(block["to"])
    R2 = R.copy()
    inside = (R2.index >= lo) & (R2.index <= hi)
    R2.loc[inside] = np.random.default_rng(4).normal(0.002, 0.03, R2.loc[inside].shape)      # rewrite the block itself
    b = dc.discover(R2, *args, folds=3, min_train_days=504, embargo=5)
    assert b["folds"][1]["chosen"] == block["chosen"], "what is chosen for a block cannot depend on that block"
    assert a["folds"][0]["from"] < a["folds"][0]["to"] < a["folds"][1]["from"] < a["folds"][1]["to"] < a["folds"][2]["from"]


def test_with_no_skill_in_the_data_the_selected_configuration_does_worse_out_of_sample_than_it_looked():
    sel, oos = [], []
    for seed in range(6):
        d = dc.discover(world(1500, 12, seed=seed, drift=0.0), [{"kind": "all"}, {"kind": "low_vol", "top": 0.5}, {"kind": "momentum", "top": 0.5}], ["equal_weight", "min_variance", "hrp"], [126, 252],
                        folds=3, min_train_days=504)
        for f in d["folds"]:
            if f.get("chosen"):
                sel.append(f["selection_sharpe"])
                oos.append(f["oos_sharpe"])
    assert np.mean(sel) > np.mean(oos), "selection optimism: the best-looking candidate on past data does not keep its edge on new data"


def test_a_planted_quiet_asset_is_found_by_the_low_volatility_screen_and_pays_out_of_sample():
    """Half the names have a persistent edge in the quiet half of the cross-section (low volatility, higher drift): the procedure should choose a low-vol or min-variance candidate and beat equal weight."""
    rng = np.random.default_rng(5)
    n, k = 1800, 14
    idx = pd.bdate_range("2015-01-05", periods=n)
    vol = np.where(np.arange(k) < 7, 0.006, 0.02)
    drift = np.where(np.arange(k) < 7, 0.0006, 0.0)
    R = pd.DataFrame(0.004 * rng.normal(0, 1, (n, 1)) + rng.normal(0, 1, (n, k)) * vol + drift, idx, [f"S{i:02d}" for i in range(k)])
    s = study.run_study((1 + R).cumprod() * 50, [{"kind": "all"}, {"kind": "low_vol", "top": 0.5}], ["equal_weight", "min_variance"], [252], folds=3, min_train_days=504, bootstrap=100)
    assert s["oos"]["sharpe"] > s["benchmarks"]["equal_weight"]["sharpe"]
    assert any("low_vol" in k.lower() or "Lowest volatility" in k or "min_variance" in k for k in s["picks"])
    assert s["status"] == "selected_on_prior_data_only" and s["n_candidates"] == 4 and s["eligibility"]["hindsight"] is True and s["warnings"]


# ---------------------------------------------------------------------------------------------------------------- diagnostics
def test_stability_concentration_and_the_bootstrap_on_known_inputs():
    w = pd.DataFrame([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]], columns=list("ABCD"))
    st = robust.stability(w)
    assert st["jaccard"] == pytest.approx(0.5) and st["weight_change"] == pytest.approx(0.5)
    c = robust.concentration(w, {"A": "x", "B": "x", "C": "y", "D": "y"})
    assert c["effective_names"] == pytest.approx(2.0) and c["largest_group_share"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.003, 0.005, 800), pd.bdate_range("2020-01-01", periods=800))
    bs = robust.block_bootstrap(good, None, n=200)
    assert bs["p_sharpe_positive"] > 0.99 and bs["sharpe_lo"] > 0
    assert robust.block_bootstrap(good, None, n=50, seed=3) == robust.block_bootstrap(good, None, n=50, seed=3)
    assert robust.block_bootstrap(good.iloc[:30], None)["note"] == "too few days to bootstrap"


def test_the_full_study_reports_every_diagnostic_and_is_json_ready():
    import json

    R = world(1500, 12)
    s = study.run_study((1 + R).cumprod() * 50, [{"kind": "all"}, {"kind": "low_vol", "top": 0.5}], ["equal_weight", "min_variance"], [252], folds=3, bootstrap=100)
    for key in ("folds", "oos", "benchmarks", "fixed_baseline", "deflated_sharpe", "beta", "concentration", "stability", "removal", "bootstrap", "regimes", "in_sample", "equity", "eligibility"):
        assert key in s, key
    assert len(s["removal"]) == 3 and len(s["equity"]["dates"]) == len(s["equity"]["oos"]) == len(s["equity"]["equal_weight"])
    json.dumps(s, allow_nan=False)
