"""Release 6, P2-02, P2-13 and P2-18: alpha diagnostics against independent computations and fixtures with a known answer.

The rank IC is checked against scipy's Spearman on every date (ties included); the decay fixture predicts only the very next day's return, so its IC must be high at no delay and indistinguishable from zero a few
sessions later; the conditional IC fixture has a target built from two known parts, so a copy of the existing signal must add nothing and the other part must add a lot."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src.ml import alpha as al

D = pd.bdate_range("2020-01-01", periods=400)
COLS = [f"a{i:02d}" for i in range(12)]


def frame(seed, shape=(400, 12)):
    return pd.DataFrame(np.random.default_rng(seed).normal(size=shape), index=D[: shape[0]], columns=COLS[: shape[1]])


def test_the_rank_ic_is_spearman_per_date_with_average_ranks_for_ties_and_says_why_a_date_has_none():
    s, t = frame(1), frame(2)
    s.iloc[3] = np.round(s.iloc[3])                                                                 # ties in the signal on one date
    s.iloc[10, :] = 1.0                                                                             # a constant cross-section
    t.iloc[11, :] = np.nan
    t.iloc[12, 4:] = np.nan                                                                         # 4 assets left: fewer than the minimum
    tab = al.ic_table(s, t, min_assets=5)
    for i in (0, 3, 50, 200):
        ref = sps.spearmanr(s.iloc[i], t.iloc[i])[0]
        assert tab["ic"].iloc[i] == pytest.approx(ref, abs=1e-12), i
    assert tab["state"].iloc[10] == "tied_signal" and np.isnan(tab["ic"].iloc[10]), "a constant cross-section has no rank correlation: unavailable, not zero"
    assert tab["state"].iloc[11] == "too_few" and tab["n"].iloc[11] == 0 and tab["state"].iloc[12] == "too_few" and tab["n"].iloc[12] == 4
    assert (tab["state"] == "ok").sum() == 400 - 3
    flat_target = al.ic_table(frame(1), pd.DataFrame(2.0, index=D, columns=COLS))
    assert set(flat_target["state"]) == {"tied_target"}


def test_a_rolling_ic_is_the_mean_of_the_dates_that_have_one_and_says_how_many():
    tab = pd.DataFrame({"ic": [0.1, np.nan, 0.3, 0.5, np.nan, 0.1] * 20, "n": 10, "state": "ok"}, index=pd.bdate_range("2021-01-01", periods=120))
    r = al.rolling_ic(tab, window=12, min_periods=6)
    window = tab["ic"].iloc[20:32]
    assert r["mean"].iloc[31] == pytest.approx(window.mean()) and r["n_ok"].iloc[31] == window.notna().sum() == 8
    assert np.isnan(r["mean"].iloc[3]), "too few observations in the window: no value"


def test_the_ic_interval_widens_with_the_horizon_because_overlapping_labels_are_not_independent():
    rng = np.random.default_rng(5)
    ic = pd.Series(0.03 + pd.Series(rng.normal(0, 0.1, 600)).rolling(5).mean().bfill().to_numpy(), index=pd.bdate_range("2019-01-01", periods=600))   # a smooth IC series: its days are not independent
    tab = pd.DataFrame({"ic": ic, "n": 12, "state": "ok"})
    one, twenty = al.ic_inference(tab, horizon=1), al.ic_inference(tab, horizon=20)
    w = lambda r: r["mean"]["interval"][1] - r["mean"]["interval"][0]                              # noqa: E731
    assert twenty["mean"]["settings"]["lags"] >= 19 and w(twenty) > w(one), "the horizon forces the lag that the overlap implies"
    assert one["mean"]["value"] == pytest.approx(ic.mean()) and one["n_with_ic"] == 600 and one["unavailable_by_reason"] == {} and "descriptive" in one["note"]
    gappy = tab.copy()
    gappy.loc[gappy.index[:30], ["ic", "state"]] = [np.nan, "too_few"]
    assert al.ic_inference(gappy, 5)["unavailable_by_reason"] == {"too_few": 30}


def test_the_decay_grid_shows_a_signal_that_only_knows_tomorrow_going_stale_at_once():
    rng = np.random.default_rng(7)
    n, k = 1500, 20
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, k)), index=pd.bdate_range("2018-01-01", periods=n), columns=[f"s{i}" for i in range(k)])
    prices = (1 + rets).cumprod() * 100
    base_lag = 1
    signal = rets.shift(-(base_lag + 1)) + pd.DataFrame(rng.normal(0, 0.01, (n, k)), index=rets.index, columns=rets.columns)   # knows the return of day t + 2 (the first day the lag-1 label covers)
    grid = al.decay_grid(signal, prices, base_lag=1, extra_lags=(0, 2, 5), horizons=(1, 5), min_assets=5)
    cell = {(c["extra_lag"], c["horizon"]): c for c in grid["cells"]}
    assert cell[(0, 1)]["mean_ic"] > 0.5 / np.sqrt(2) * 0.8, "with equal signal and noise variance the IC of the day it knows is about 0.5/sqrt(2)... well above zero"
    for lag in (2, 5):
        lo, hi = cell[(lag, 1)]["interval"]
        assert lo < 0 < hi, "two sessions late the same signal is indistinguishable from nothing"
        assert abs(cell[(lag, 1)]["share_of_undelayed"]) < 0.3
    assert cell[(0, 5)]["mean_ic"] < cell[(0, 1)]["mean_ic"], "a five-day return dilutes a signal that only knows one of its days"
    zero = al.decay_grid(-signal, prices, base_lag=1, extra_lags=(0, 2), horizons=(1,))
    assert [c["share_of_undelayed"] for c in zero["cells"]] == [None, None], "a share of a non-positive undelayed IC is undefined, not a number"


def test_coverage_counts_the_investable_cells_that_are_empty_and_never_removes_them():
    sig = frame(3, (100, 5))
    sig.iloc[10:25, 1] = np.nan                                                                     # a 15-day hole in one asset
    sig.iloc[:60, 3] = np.nan
    inv = pd.DataFrame(True, index=sig.index, columns=sig.columns)
    inv.iloc[:60, 3] = False                                                                        # but a3 was not investable then: not a hole in the signal
    inv.iloc[90:, 4] = False
    rep = al.coverage_report(sig, inv)
    assert rep["cells"] == 500 - 60 - 10 and rep["missing_cells"] == 15 and rep["coverage"] == pytest.approx(1 - 15 / 430)
    assert rep["longest_missing_run"]["a01"] == 15 and rep["longest_missing_run"]["a00"] == 0 and rep["assets_below_threshold"] == [] and rep["by_asset"]["a01"] == pytest.approx(85 / 100)
    assert rep["assets_per_date"]["min"] == 3 and "not dropped" in rep["note"]
    assert al.coverage_report(sig, pd.DataFrame(False, index=sig.index, columns=sig.columns))["eligibility"] == "insufficient"
    assert al.coverage_report(sig, None, min_asset_coverage=0.9)["assets_below_threshold"] == ["a01", "a03"]


def test_stability_splits_the_dates_into_blocks_and_persistence_tells_a_sticky_signal_from_a_noisy_one():
    alt = pd.DataFrame({"ic": np.r_[np.full(50, 0.08), np.full(50, -0.02), np.full(50, 0.06), np.full(50, 0.05)], "n": 10, "state": "ok"}, index=pd.bdate_range("2020-01-01", periods=200))
    rep = al.stability_report(alt, blocks=4)
    assert rep["eligibility"] == "applied" and [b["mean_ic"] for b in rep["blocks"]] == pytest.approx([0.08, -0.02, 0.06, 0.05]) and rep["same_sign_share"] == pytest.approx(0.75)
    assert rep["spread_of_block_means"] == pytest.approx(np.std([0.08, -0.02, 0.06, 0.05], ddof=1)) and rep["blocks"][0]["n"] == 50
    assert al.stability_report(alt.iloc[:20], blocks=4)["eligibility"] == "insufficient"
    rng = np.random.default_rng(9)
    sticky = pd.DataFrame(np.cumsum(rng.normal(size=(300, 10)) * 0.05, axis=0) + rng.normal(size=(1, 10)) * 3, index=pd.bdate_range("2020-01-01", periods=300), columns=COLS[:10])
    noise = pd.DataFrame(rng.normal(size=(300, 10)), index=sticky.index, columns=sticky.columns)
    p_sticky = al.stability_report(alt, signal=sticky)["rank_persistence"]
    p_noise = al.stability_report(alt, signal=noise)["rank_persistence"]
    assert p_sticky[1] > 0.9 and p_sticky[21] > 0.5 and abs(p_noise[1]) < 0.1 and abs(p_noise[21]) < 0.1


def test_redundancy_finds_the_copy_counts_independent_features_and_names_the_ones_it_cannot_assess():
    rng = np.random.default_rng(11)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=250), COLS[:10]], names=["date", "asset"])
    f1 = rng.normal(size=len(idx))
    X = pd.DataFrame({"f1": f1, "f2": f1 + rng.normal(0, 0.05, len(idx)), "f3": rng.normal(size=len(idx)), "f4": rng.normal(size=len(idx)), "const": 1.0}, index=idx)
    rep = al.redundancy_report(X, threshold=0.9)
    assert rep["eligibility"] == "applied" and [(a, b) for a, b, _ in rep["pairs_above_threshold"]] == [("f1", "f2")] and rep["pairs_above_threshold"][0][2] > 0.95
    rho = rep["pairs_above_threshold"][0][2]
    assert rep["effective_independent_features"] == pytest.approx(16.0 / (4.0 + 2.0 * rho ** 2), abs=0.08), "eigenvalues 1+rho, 1-rho, 1, 1 give a participation ratio of 16/(4+2 rho^2): two copies count as one, not as two"
    assert 2.5 < rep["effective_independent_features"] < 2.8
    assert rep["unavailable"] == {"const": "constant or fewer than 50 ranked rows"} and rep["n_features"] == 4
    thin = X[X.index.get_level_values(1).isin(COLS[:3])]
    assert al.redundancy_report(thin)["eligibility"] == "insufficient", "a cross-section of three assets is below the minimum: nothing is estimated"


def test_a_diagnostic_that_drives_a_choice_of_features_may_only_read_development_data():
    labels = pd.Series(["train"] * 150 + ["validation"] * 100 + ["test"] * 100 + ["forward"] * 50, index=pd.bdate_range("2020-01-01", periods=400))
    sig = frame(3, (400, 4))
    assert len(al.development_only(sig, labels)) == 250 and set(labels.reindex(al.development_only(sig, labels).index)) == {"train", "validation"}
    idx = pd.MultiIndex.from_product([labels.index, list("ab")], names=["date", "asset"])
    assert len(al.development_only(pd.DataFrame({"x": 1.0}, index=idx), labels)) == 500
    for bad in (("test",), ("validation", "forward")):
        with pytest.raises(ValueError, match="development data only"):
            al.development_only(sig, labels, segments=bad)


def test_correlation_reports_align_the_dates_say_what_was_dropped_and_measure_overlap_by_hand():
    idx = pd.bdate_range("2020-01-01", periods=3)
    wa = pd.DataFrame([[0.5, -0.5, 0.0]] * 3, index=idx, columns=list("xyz"))
    wb = pd.DataFrame([[0.25, 0.25, -0.5]] * 3, index=idx, columns=list("xyz"))
    ra, rb = pd.Series([0.01, 0.02, 0.0], index=idx), pd.Series([0.0, 0.01, 0.02], index=idx)
    rep = al.correlation_report({}, {"A": ra, "B": rb}, {"A": wa, "B": wb})
    pair = rep["pairs"][0]
    assert pair["exposure_overlap"] == pytest.approx(0.25) and pair["weight_cosine"] == pytest.approx(0.0, abs=1e-12) and pair["return_correlation"] is None, "three dates are too few for a correlation"
    assert rep["concentration"]["A"] == {"mean_hhi": 0.5, "effective_positions": 2.0, "top3_share": 1.0, "days_with_a_book": 3}
    rng = np.random.default_rng(13)
    n = 1200
    common = rng.normal(0, 0.01, n)
    ix = pd.bdate_range("2018-01-01", periods=n)
    fix = lambda x, ref: (x - ref.mean()) / ref.std(ddof=1) * 0.014 + 0.0008                          # noqa: E731  # the same mean and volatility over the aligned dates for both books: the blend is then algebra, not a draw
    a_raw = pd.Series(common + rng.normal(0, 0.01, n), index=ix)
    b_raw = pd.Series(common + rng.normal(0, 0.01, n), index=ix).iloc[40:]                            # the second book starts later: 40 dates exist for one side only
    a, b = fix(a_raw, a_raw.iloc[40:]), fix(b_raw, b_raw)
    rep = al.correlation_report({"A": frame(1), "B": frame(2)}, {"A": a, "B": b}, {})
    pair = rep["pairs"][0]
    assert pair["return_correlation"] == pytest.approx(0.5, abs=0.06) and pair["aligned_dates"] == n - 40 and pair["dropped_dates"] == {"A": 40, "B": 0}
    assert abs(pair["prediction_rank_correlation"]) < 0.05
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    rho = both["a"].corr(both["b"])
    assert pair["blend"]["diversification_ratio"] == pytest.approx(1.0 / np.sqrt((1.0 + rho) / 2.0), rel=0.01), "equal volatilities: 0.5(2s) / s sqrt((1+rho)/2)"
    bl = pair["blend"]
    assert bl["sharpe_a"] == pytest.approx(bl["sharpe_b"], rel=0.02) and bl["sharpe_blend"] == pytest.approx(bl["sharpe_a"] * np.sqrt(2.0 / (1.0 + rho)), rel=0.01), "equal edge and volatility: the blend has the common mean and sqrt((1+rho)/2) of the volatility"
    assert bl["sharpe_blend"] > max(bl["sharpe_a"], bl["sharpe_b"])


def test_the_conditional_ic_removes_what_the_existing_signals_already_explain():
    rng = np.random.default_rng(17)
    n, k = 300, 30
    ix, cols = pd.bdate_range("2019-01-01", periods=n), [f"s{i:02d}" for i in range(k)]
    a = pd.DataFrame(rng.normal(size=(n, k)), index=ix, columns=cols)
    b = pd.DataFrame(rng.normal(size=(n, k)), index=ix, columns=cols)
    target = a + b + pd.DataFrame(rng.normal(size=(n, k)), index=ix, columns=cols)
    copy = a + pd.DataFrame(rng.normal(0, 0.1, size=(n, k)), index=ix, columns=cols)
    plain_copy = al.ic_table(copy, target)["ic"].mean()
    cond_copy = al.conditional_ic(copy, {"a": a}, target)
    cond_new = al.conditional_ic(b, {"a": a}, target)
    assert plain_copy > 0.4, "unconditionally the copy of the existing signal looks like a good signal"
    assert abs(cond_copy["ic"].mean()) < 0.06 and cond_copy["state"].eq("ok").all(), "given the signal it copies it says nothing more"
    assert cond_new["ic"].mean() > 0.3, "the other part of the target is new information"
    # an independent reference on one date: the partial rank correlation from the inverse of the rank correlation matrix, -P_xy / sqrt(P_xx P_yy)
    d = ix[40]
    ranks = pd.concat([b.loc[d], target.loc[d], a.loc[d]], axis=1).rank()
    prec = np.linalg.inv(np.corrcoef(ranks.to_numpy().T))
    assert cond_new["ic"].loc[d] == pytest.approx(-prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1]), abs=1e-9)
    c = pd.DataFrame(rng.normal(size=(n, k)), index=ix, columns=cols)
    five = al.conditional_ic(b.iloc[:, :5], {"a": a.iloc[:, :5], "c": c.iloc[:, :5]}, target.iloc[:, :5], min_assets=5)
    assert set(five["state"]) == {"too_few"} and (five["n"] == 5).all(), "two regressors need at least six assets (regressors + 4): five cannot support the margin"
    six = al.conditional_ic(b.iloc[:, :6], {"a": a.iloc[:, :6], "c": c.iloc[:, :6]}, target.iloc[:, :6], min_assets=5)
    assert "too_few" not in set(six["state"]) and (six["n"] == 6).all()
    assert six["ic"].notna().eq(six["state"] == "ok").all(), "a date without a residual has an unavailable IC, never a zero"
    own = al.conditional_ic(a, {"a": a}, target)
    assert set(own["state"]) == {"explained_signal"} and own["ic"].isna().all(), "a signal the existing one reproduces exactly has no residual: explained, not zero and not tied"
    flat = al.conditional_ic(pd.DataFrame(1.0, index=ix, columns=cols), {"a": a}, target)
    assert set(flat["state"]) == {"tied_signal"}
    res = al.ic_inference(cond_new.assign(n=cond_new["n"]), horizon=1)
    assert res["mean"]["eligibility"] == "applied" and res["mean"]["interval"][0] > 0


def test_blending_in_a_second_book_is_judged_on_aligned_dates_with_a_paired_difference():
    rng = np.random.default_rng(21)
    n = 1500
    ix = pd.bdate_range("2017-01-01", periods=n)
    base = pd.Series(rng.normal(0.0015, 0.01, n), index=ix)
    new = pd.Series(rng.normal(0.0015, 0.01, n), index=ix).iloc[100:]
    out = al.blend_value(base, new, n_boot=300, seed=1)
    assert out["eligibility"] == "applied" and out["aligned"] == n - 100 and out["dropped"] == {"base": 100, "new": 0} and abs(out["correlation"]) < 0.1
    assert out["difference"]["value"] > 0.5 and out["difference"]["interval"][0] > 0, "an independent book with the same edge raises the Sharpe ratio of the blend"
    assert "not a causal claim" in out["interpretation"] and "not a feature importance" in out["interpretation"]
    copy = al.blend_value(base, base * 2.0, n_boot=300, seed=1)
    assert abs(copy["difference"]["value"]) < 1e-9, "a scaled copy of the base book adds nothing once each is put at the same risk"
    assert al.blend_value(base.iloc[:10], new)["eligibility"] == "insufficient" and al.blend_value(base, base * 0.0)["eligibility"] == "insufficient"
