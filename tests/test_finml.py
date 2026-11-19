"""The financial machine-learning tools, each against a case worked by hand or a property it must have: event sampling and triple-barrier labels (including a bar that touches both barriers), sample uniqueness and the
sequential bootstrap, purged cross-validation (no label of the training set overlaps a test label), feature importance, bet sizing, information-driven bars, structural-break and entropy statistics, and the
pipeline's honesty (a holdout nothing was fitted on, a result that is negative when the filter does not help)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.finml import bars, events as ev, importance as imp, sampling as sp, sizing as sz, structural as st
from src.finml.cv import cpcv_event_splits, leakage_check, purge_train, purged_folds
from src.finml.pipeline import PipelineError, run_pipeline
from src.ops.cv import cpcv_paths, n_cpcv_paths


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# ------------------------------------------------------------------------------------------------ events and triple barriers
def test_cusum_fires_when_the_cumulative_move_exceeds_the_threshold_and_resets():
    y = _series([0, 1, 2, 3, 2, 1, 0, -1, -2, -3.0])
    got = ev.cusum_events(y, 2.5)
    # up: 1, 2, 3 -> s+ = 3 > 2.5 at index 3 (reset); then down moves: s- accumulates -1, -1, -1 (indexes 4, 5, 6) -> -3 < -2.5 at index 6 (reset); then -1, -1, -1 -> index 9
    assert list(got) == [y.index[3], y.index[6], y.index[9]]
    assert len(ev.cusum_events(y, 100)) == 0


def test_a_long_label_is_the_barrier_touched_first_with_the_return_at_the_touch():
    close = _series([100, 100, 101, 102, 104, 103, 90, 90, 90, 90, 90.0])
    tgt = _series([0.03] * 11)                                                           # barriers at 103 and 97 from an entry of 100
    lab = ev.triple_barrier_events(close, [close.index[1]], tgt, vertical_bars=8, side=_series([1] * 11))
    row = lab.iloc[0]
    assert row["barrier"] == "pt" and row["t1"] == close.index[4] and row["pos1"] == 4 and row["exit"] == 104 and row["ret"] == pytest.approx(0.04) and row["bin"] == 1
    # the same path for a SHORT: 104 is above its stop (103), so it is stopped out there
    short = ev.triple_barrier_events(close, [close.index[1]], tgt, vertical_bars=8, side=_series([-1] * 11)).iloc[0]
    assert short["barrier"] == "sl" and short["t1"] == close.index[4] and short["ret"] == pytest.approx(-0.04) and short["bin"] == 0


def test_the_vertical_barrier_and_the_unlabelled_tail():
    close = _series([100, 100.5, 100.2, 100.8, 100.4, 100.6, 100.3, 100.9, 100.1, 100.7, 100.2, 100.5])
    tgt = _series([0.05] * 12)
    lab = ev.triple_barrier_events(close, close.index[[0, 1, 9]], tgt, vertical_bars=5)
    assert list(lab["barrier"]) == ["vertical", "vertical"] and list(lab["pos1"]) == [5, 6]       # the event at row 9 has no full window: no label, whatever happens
    assert (lab["bin"] == 0).all()                                                       # a vertical return far smaller than half a barrier is 0


def test_a_bar_that_touches_both_barriers_is_resolved_by_the_stated_assumption():
    close = _series([100, 100, 100, 100, 100, 100.0])
    high = _series([100, 100, 106, 100, 100, 100.0])
    low = _series([100, 100, 94, 100, 100, 100.0])
    tgt = _series([0.05] * 6)
    side = _series([1] * 6)
    got = {k: ev.triple_barrier_events(close, [close.index[1]], tgt, vertical_bars=3, side=side, high=high, low=low, same_bar=k) for k in ("stop", "profit", "drop")}
    assert got["stop"].iloc[0]["barrier"] == "sl" and got["stop"].iloc[0]["ret"] == pytest.approx(-0.05) and bool(got["stop"].iloc[0]["ambiguous"])
    assert got["profit"].iloc[0]["barrier"] == "pt" and got["profit"].iloc[0]["ret"] == pytest.approx(0.05)
    assert got["drop"].empty
    # and for a short the stop is the other barrier
    short = ev.triple_barrier_events(close, [close.index[1]], tgt, vertical_bars=3, side=_series([-1] * 6), high=high, low=low, same_bar="stop").iloc[0]
    assert short["barrier"] == "sl" and short["ret"] == pytest.approx(-0.05)


def test_entry_lag_costs_and_unsided_labels():
    close = _series([100, 101, 103, 104, 104, 104, 104.0])
    tgt = _series([0.02] * 7)
    lab = ev.triple_barrier_events(close, [close.index[0]], tgt, vertical_bars=3, entry_lag=1, side=_series([1] * 7), cost=0.001).iloc[0]
    assert lab["entry"] == 101 and lab["pos_entry"] == 1 and lab["barrier"] == "pt" and lab["exit"] == 104 and lab["ret"] == pytest.approx(104 / 101 - 1 - 0.001)   # 103 < 101 x 1.02 = 103.02, so the first close beyond it is 104
    plain = ev.triple_barrier_events(close, [close.index[2]], tgt, vertical_bars=3).iloc[0]
    assert plain["side"] == 1 and plain["bin"] in (-1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        ev.triple_barrier_events(close, [close.index[0]], tgt, same_bar="maybe")
    with pytest.raises(ValueError):
        ev.triple_barrier_events(close, [close.index[0]], tgt, high=close)


def test_a_labelled_event_never_depends_on_the_future_beyond_its_window():
    rng = np.random.default_rng(0)
    close = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    vol = ev.daily_vol(close)
    evs = ev.cusum_events(np.log(close), vol.fillna(0.01))
    full = ev.triple_barrier_events(close, evs, vol, vertical_bars=10, side=_series(np.ones(300)), entry_lag=1)
    cut = ev.triple_barrier_events(close.iloc[:200], evs[evs <= close.index[150]], vol.iloc[:200], vertical_bars=10, side=_series(np.ones(300)).iloc[:200], entry_lag=1)
    shared = cut.index.intersection(full.index)
    assert len(shared) > 5
    pd.testing.assert_frame_equal(full.loc[shared], cut.loc[shared])


# ------------------------------------------------------------------------------------------------ sampling
def _labels(spans_):
    """Labels whose returns cover rows start..end (inclusive): pos_entry = start - 1."""
    return pd.DataFrame({"pos0": [a - 1 for a, _ in spans_], "pos_entry": [a - 1 for a, _ in spans_], "pos1": [b for _, b in spans_]}, index=pd.RangeIndex(len(spans_)))


def test_uniqueness_of_a_worked_example():
    # the book's example: three labels covering rows [1..3], [3..4] and [5..5]; concurrency 1,1,2,1,1 on rows 1..5
    lab = _labels([(1, 3), (3, 4), (5, 5)])
    u = sp.uniqueness(lab, 8).to_numpy()
    assert u == pytest.approx([(1 + 1 + 0.5) / 3, (0.5 + 1) / 2, 1.0])
    q = sp.sample_quality(lab, 8)
    assert q["effective_sample_size"] == pytest.approx(u.sum() ** 2 / (u ** 2).sum()) and q["mean_uniqueness"] == pytest.approx(u.mean())
    assert sp.sample_quality(_labels([(1, 2), (3, 4), (5, 6)]), 8)["effective_sample_size"] == pytest.approx(3.0)      # no overlap: three labels are three observations


def test_a_sequential_bootstrap_is_more_unique_than_an_ordinary_one():
    rng = np.random.default_rng(1)
    starts = np.sort(rng.integers(1, 580, 200))
    lab = _labels([(int(s), int(s) + int(rng.integers(3, 25))) for s in starts])
    seq = np.array([sp.bootstrap_uniqueness(lab, 620, sp.sequential_bootstrap(lab, 620, seed=s)) for s in range(10)])
    plain = np.array([sp.bootstrap_uniqueness(lab, 620, np.random.default_rng(s).integers(0, len(lab), len(lab))) for s in range(10)])
    assert seq.mean() > plain.mean() * 1.015 and (seq > plain).mean() >= 0.8
    again = sp.sequential_bootstrap(lab, 620, seed=3)
    assert (again == sp.sequential_bootstrap(lab, 620, seed=3)).all() and len(again) == len(lab) and again.min() >= 0 and again.max() < len(lab)


def test_attribution_weights_and_time_decay():
    lab = _labels([(1, 2), (2, 3), (5, 5)])
    ret = _series([0, 0.10, 0.02, 0.0, 0.0, -0.04, 0])
    w = sp.return_attribution_weights(lab, ret)
    c = [1, 2, 1]                                                                        # concurrency on rows 1, 2, 3
    raw = np.array([abs(0.10 / 1 + 0.02 / 2), abs(0.02 / 2 + 0.0 / 1), abs(-0.04)])
    assert w.to_numpy() == pytest.approx(raw * 3 / raw.sum()) and c
    flat = pd.Series(np.ones(10))
    assert sp.time_decay(flat, 1.0).to_numpy() == pytest.approx(np.ones(10))
    d = sp.time_decay(flat, 0.0).to_numpy()
    assert d[-1] == pytest.approx(1.0) and d[0] < 0.2 and (np.diff(d) > 0).all()
    with pytest.raises(ValueError):
        sp.time_decay(flat, 2.0)


# ------------------------------------------------------------------------------------------------ purged cross-validation
def _overlapping_labels(n=300, hold=12):
    p0 = np.arange(n)
    return pd.DataFrame({"pos0": p0, "pos_entry": p0, "pos1": p0 + hold}, index=pd.RangeIndex(n))


def test_purged_folds_leave_no_training_label_that_overlaps_a_test_label():
    lab = _overlapping_labels()
    for train, test in purged_folds(lab, 5, embargo=5):
        assert leakage_check(lab, train, test) and len(train) > 100
    naive_train = np.setdiff1d(np.arange(len(lab)), np.arange(100, 160))
    assert not leakage_check(lab, naive_train, np.arange(100, 160)), "an unpurged split leaks, which is what the check is for"
    kept = purge_train(lab, naive_train, [np.arange(100, 160)], embargo=5)
    assert (lab["pos1"].to_numpy()[kept] < 100).sum() + (lab["pos0"].to_numpy()[kept] > 171 + 5).sum() == len(kept)


def test_cpcv_splits_count_paths_and_stay_leak_free():
    lab = _overlapping_labels(360)
    splits = cpcv_event_splits(lab, 6, 2, embargo=3)
    assert len(splits) == 15 and len(cpcv_paths(splits, 6)) == n_cpcv_paths(6, 2) == 5
    assert all(leakage_check(lab, s["train"], s["test"]) for s in splits)
    covered = {g for path in cpcv_paths(splits, 6) for _, g in path}
    assert covered == set(range(6))


# ------------------------------------------------------------------------------------------------ feature importance
def _importance_data(n=900, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "twin": signal + rng.normal(0, 0.05, n), "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n)})
    y = pd.Series((signal + rng.normal(0, 0.6, n) > 0).astype(int))
    lab = _overlapping_labels(n, hold=4)
    return X, y, lab


def test_the_informative_feature_ranks_first_by_every_method_and_a_twin_shares_its_cluster():
    X, y, lab = _importance_data()
    m = imp.mdi(X, y, seed=0)
    assert m.index[0] in ("signal", "twin") and m["mean"].sum() == pytest.approx(1.0) and m.loc[["noise_a", "noise_b"], "mean"].max() < m["mean"].iloc[0]
    groups = imp.feature_clusters(X, 0.3)
    assert any({"signal", "twin"} <= set(v) for v in groups.values())
    a = imp.mda(X, y, lab, None, 4, 4, clusters=groups, seed=0)
    top = a.index[0]
    assert {"signal", "twin"} <= set(a.loc[top, "members"].split(", ")) and a.loc[top, "mean"] > 0.05 and a.loc[top, "positive"] == 1.0
    assert a.drop(index=top)["mean"].max() < 0.02, "shuffling pure noise does not hurt the model out of sample"
    s = imp.sfi(X, y, lab, None, 4, 4, seed=0)
    assert s.index[0] in ("signal", "twin") and s.loc[["noise_a", "noise_b"], "gain"].max() < 0.01 < s["gain"].iloc[0]


# ------------------------------------------------------------------------------------------------ sizing
def test_bet_size_from_probability_matches_the_formula_and_is_monotone():
    p = np.array([0.3, 0.5, 0.6, 0.8, 0.95])
    m = sz.size_from_probability(p)
    z = (0.6 - 0.5) / np.sqrt(0.6 * 0.4)
    assert m[2] == pytest.approx(2 * norm.cdf(z) - 1) and m[0] == 0.0 and m[1] == 0.0 and (np.diff(m[1:]) > 0).all() and (m <= 1).all()
    assert sz.size_from_probability(0.8, -1) == pytest.approx(-sz.size_from_probability(0.8, 1))
    assert sz.size_from_probability(1.0) == pytest.approx(1.0, abs=1e-6)
    assert sz.size_from_probability(0.58, 1, floor=0.6) == 0.0
    assert sz.sigmoid_size(0.0) == 0.0 and 0 < sz.sigmoid_size(2.0) < 1
    assert list(sz.discretize([0.33, 0.77, -0.04, 1.4], 0.1)) == pytest.approx([0.3, 0.8, 0.0, 1.0])


def test_average_active_bets_and_turnover_by_hand():
    idx = pd.bdate_range("2021-01-04", periods=8)
    # bet A (+1) earns rows 2..4, bet B (-0.5) earns rows 4..6: row 4 averages (+1 - 0.5) / 2
    pos = sz.average_active(pd.Series([1.0, -0.5]), np.array([1, 3]), np.array([4, 6]), idx)
    assert pos.to_numpy() == pytest.approx([0, 0, 1, 1, 0.25, -0.5, -0.5, 0])
    assert sz.turnover(pos) == pytest.approx(np.abs(np.diff(pos.to_numpy())).sum() / 7)


# ------------------------------------------------------------------------------------------------ bars
def test_tick_volume_and_dollar_bars_close_where_they_should():
    t = bars.simulate_trades(5000, seed=1, regimes=False)
    tb = bars.tick_bars(t, 100)
    assert len(tb) == 50 and (tb["ticks"] == 100).all() and tb["volume"].sum() == pytest.approx(t["volume"].sum())
    size = t["volume"].sum() / 40
    vb = bars.volume_bars(t, size)
    assert (vb["volume"] >= size).all() and (vb["volume"] < size + t["volume"].max()).all() and abs(len(vb) - 40) <= 1
    d = (t["price"] * t["volume"]).sum() / 40
    db = bars.dollar_bars(t, d)
    assert (db["dollar"] >= d).all() and db["high"].ge(db["low"]).all() and db["high"].ge(db["open"]).all() and db["low"].le(db["close"]).all()
    assert list(tb.index[:2]) == list(t.index[[99, 199]]), "a bar is stamped with its last trade"
    with pytest.raises(ValueError):
        bars.tick_bars(t.rename(columns={"price": "p"}), 10)
    with pytest.raises(ValueError):
        bars.tick_bars(t, 0)


def test_imbalance_and_runs_bars_close_sooner_when_one_side_dominates():
    t = bars.simulate_trades(30_000, seed=2, regimes=True)
    n = len(t)
    lo, hi = t.index[n // 2], t.index[n // 2 + n // 10]
    for name, b in (("imbalance", bars.imbalance_bars(t, "volume", 100, span=10)), ("runs", bars.runs_bars(t, "volume", 100, span=10))):
        assert len(b) > 40, name
        hot, cold = ((b["start"] >= lo) & (b["start"] < hi)).sum() / (n // 10), (b["start"] < lo).sum() / (n // 2)
        assert hot > 1.4 * cold, f"{name} bars come more often in the stretch of one-sided heavy trading"
        assert b["ticks"].min() >= 1 and b["ticks"].max() <= 1000
    flat = bars.simulate_trades(30_000, seed=2, regimes=False)
    assert bars.imbalance_bars(flat, "tick", 100)["ticks"].min() >= 5 and len(bars.imbalance_bars(flat, "tick", 100, floor=0.0)) > 3 * len(bars.imbalance_bars(flat, "tick", 100)), "without the floor, balanced flow collapses into minimum-size bars"


# ------------------------------------------------------------------------------------------------ structure and entropy
def test_sadf_flags_an_explosive_stretch_but_not_a_random_walk():
    rng = np.random.default_rng(1)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 260)))
    bubble = pd.Series(np.concatenate([np.cumsum(rng.normal(0, 1, 160)), 10 + np.cumsum(rng.normal(0.9, 1.0, 100))]) * 1.0)
    crit = st.sadf_critical(260, 40, reps=30, seed=0, max_starts=10)
    assert np.nanmax(st.sadf(walk, max_starts=15).to_numpy()) < crit + 1.0
    assert np.nanmax(st.sadf(bubble, max_starts=15).to_numpy()) > max(crit, 2.0) + 1.0


def test_the_cusum_level_test_fires_more_after_a_shift():
    rng = np.random.default_rng(2)
    flat = pd.Series(np.cumsum(rng.normal(0, 1, 300)))
    shifted = pd.Series(np.concatenate([np.cumsum(rng.normal(0, 1, 200)), 30 + np.cumsum(rng.normal(0, 1, 100))]))
    assert st.chu_stinchcombe_white(shifted)["break_"].sum() > 2 * st.chu_stinchcombe_white(flat)["break_"].sum()


def test_entropy_estimators_separate_random_from_repetitive_sequences():
    rng = np.random.default_rng(3)
    rnd, rep = "".join(rng.choice(list("ab"), 600)), "ab" * 300
    assert st.plug_in_entropy(rnd, 2) > 0.95 and st.plug_in_entropy(rep, 2) < 0.55
    assert st.lempel_ziv(rnd) > 1.4 * st.lempel_ziv(rep)
    assert st.kontoyiannis_entropy(rnd) > 3 * st.kontoyiannis_entropy(rep)
    r = pd.Series(rng.normal(size=400))
    code = st.encode_returns(r, 4)
    assert len(code) == 400 and set(code) == set("abcd") and abs(code.count("a") - 100) < 5 and set(st.encode_returns(r, kind="sign")) == {"u", "d"}


# ------------------------------------------------------------------------------------------------ the pipeline
def _regime_prices(n=3500, seed=5):
    """Momentum that works in calm stretches and reverses in turbulent ones: a primary momentum signal whose profit a meta-model can learn from the volatility."""
    rng = np.random.default_rng(seed)
    reg = np.zeros(n, int)
    for i in range(1, n):
        reg[i] = reg[i - 1] if rng.random() > 0.01 else 1 - reg[i - 1]
    sig, phi = np.where(reg == 1, 0.016, 0.007), np.where(reg == 1, -0.45, 0.40)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi[i] * r[i - 1] + rng.normal(0, sig[i])
    return pd.Series(100 * np.exp(np.cumsum(r)), pd.bdate_range("2008-01-02", periods=n))


REGIME_CFG = {"importance": False, "lookback": 5, "vertical_bars": 8, "cusum_k": 0.7, "entry_lag": 0, "cpcv_groups": 5, "cpcv_test": 2}


@pytest.fixture(scope="module")
def regime_run():
    return run_pipeline(_regime_prices(), REGIME_CFG)


def test_the_pipeline_finds_the_regime_dependence_and_proves_nothing_leaks(regime_run):
    r = regime_run
    assert r["validation"]["purging_leak_free"] and r["validation"]["auc"] > 0.53 and r["validation"]["log_loss"] < r["validation"]["base_rate_log_loss"]
    dev, hold = r["development"], r["holdout"]
    assert dev["meta_filter"]["sharpe"] > dev["primary"]["sharpe"] + 0.5 and hold["meta_filter"]["sharpe"] > hold["primary"]["sharpe"]
    assert dev["meta_filter"]["taken"] < dev["primary"]["taken"] and hold["meta_filter"]["sharpe"] > hold["random_filter"]["sharpe"]
    assert r["cpcv"]["n_paths"] == 4 and r["cpcv"]["mean"] > r["cpcv"]["primary_sharpe"] and r["cpcv"]["prob_negative"] <= 0.25
    assert r["split"]["development_events"] + r["split"]["holdout_events"] + r["split"]["dropped_for_overlap"] == r["labels"]["events"]
    assert r["quality"]["effective_sample_size"] < r["quality"]["events"] and r["development"]["evidence"]["meta_filter"]["psr"] > 0.9
    assert {"development", "holdout"} <= set(r["series"]) and {"primary", "meta_filter", "meta_sized", "buy_and_hold"} <= set(r["series"]["holdout"])


def test_the_holdout_comes_strictly_after_every_label_the_model_was_fitted_on():
    close = _regime_prices()
    r = run_pipeline(close, REGIME_CFG)
    start = pd.Timestamp(r["split"]["holdout_starts"])
    assert pd.Timestamp(r["holdout"]["window"]["start"]) >= start
    # changing only what happens inside the holdout must not change a single development number
    altered = close.copy()
    altered.loc[altered.index >= start + pd.Timedelta(days=40)] *= np.exp(np.cumsum(np.random.default_rng(9).normal(0, 0.03, int((altered.index >= start + pd.Timedelta(days=40)).sum()))))
    r2 = run_pipeline(altered, REGIME_CFG)
    assert r2["development"]["meta_filter"]["sharpe"] == pytest.approx(r["development"]["meta_filter"]["sharpe"], abs=1e-9)
    assert r2["validation"]["auc"] == pytest.approx(r["validation"]["auc"], abs=1e-9)


def test_the_pipeline_is_deterministic_and_reports_a_negative_result_when_there_is_nothing_to_find():
    rng = np.random.default_rng(11)
    walk = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2500))), pd.bdate_range("2010-01-04", periods=2500))
    cfg = {"importance": False, "cpcv_groups": 5}
    a, b = run_pipeline(walk, cfg), run_pipeline(walk, cfg)
    assert a["development"]["meta_filter"]["sharpe"] == b["development"]["meta_filter"]["sharpe"] and a["validation"]["auc"] == b["validation"]["auc"]
    assert not a["meta_labelling_demonstrated"] and "not shown" in a["verdict"]
    assert a["validation"]["auc"] < 0.56, "a random walk gives the model nothing to learn"


def test_option_checks_and_the_other_models_and_samplers_run():
    close = _regime_prices(1600, seed=2)
    for bad in ({"primary": "oracle"}, {"holdout": 0.7}, {"threshold": 0.4}, {"model": "gpt"}, {"nonsense": 1}):
        with pytest.raises(PipelineError):
            run_pipeline(close, bad)
    with pytest.raises(PipelineError):
        run_pipeline(close.iloc[:200])
    r = run_pipeline(close, {**REGIME_CFG, "model": "forest", "bagging": "sequential", "n_bags": 3, "weighting": "attribution", "decay": 0.5, "step": 0.1, "importance": True, "cpcv_groups": 4, "primary": "ma_cross", "lookback": 12, "fast": 4})
    assert r["importance"]["mda"] and r["importance"]["mdi"] and r["importance"]["sfi"] and r["development"]["meta_sized"]["turnover_per_day"] >= 0
    hi, lo = close * 1.004, close * 0.996
    h = run_pipeline(close, {**REGIME_CFG, "importance": False, "same_bar": "profit"}, high=hi, low=lo)
    assert h["labels"]["ambiguous_share"] >= 0.0
