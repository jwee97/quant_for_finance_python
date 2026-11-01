"""P0-13, P0-17 and P0-21: label intervals on the real calendar, purging by them, and the dependence of overlapping labels, each against an independent brute-force oracle."""

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as dsm, intervals as iv, training as tr
from tests.ml_helpers import make_bundle


# ------------------------------------------------------------------------------------------------------------ pure functions against brute force
def test_label_intervals_run_from_the_entry_close_to_the_exit_close_and_are_unavailable_at_the_end_of_the_data():
    dates = pd.bdate_range("2024-01-01", periods=12)
    f = iv.label_frame(dates, np.arange(12), lag=1, horizon=3)
    assert list(f["label_start_pos"][:3]) == [1, 2, 3] and list(f["label_end_pos"][:3]) == [4, 5, 6]
    assert f["label_start_at"][0] == dates[1] and f["label_available_at"][0] == dates[4]
    assert f["label_available_at"][7] == dates[11] and f["label_available_at"][8:].isna().all(), "a label that ends after the data does not exist yet"
    assert iv.CONVENTION["entry"] == "close of day t + lag" and iv.CONVENTION["exit"] == "close of day t + lag + horizon"


@pytest.mark.parametrize("seed", range(6))
def test_eligibility_matches_a_brute_force_calendar_oracle_for_irregular_horizons_and_embargoes(seed):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=400)
    positions = np.sort(rng.choice(400, 250, replace=False))                              # a calendar with holes: the list of labelled dates is not the calendar
    lag, horizons = int(rng.integers(0, 3)), rng.integers(1, 21, size=positions.size)       # every row has its own horizon (a multi-horizon label, longest wins)
    decision, embargo = int(rng.integers(150, 350)), int(rng.integers(0, 5))
    got = np.array([iv.eligible_for_fit(np.array([p]), lag, int(h), decision, embargo)[0] for p, h in zip(positions, horizons)])
    want = np.array([days[p + lag + h] <= days[decision] - pd.offsets.BDay(embargo) if p + lag + h < 400 else False for p, h in zip(positions, horizons)])
    # the oracle works in dates: a label is usable when the day it ends is on or before the decision day less the embargo, counted in business days
    assert np.array_equal(got[positions + lag + horizons < 400], want[positions + lag + horizons < 400])


def test_a_row_whose_label_overlaps_the_test_block_is_excluded_and_one_that_ends_before_it_is_kept():
    pos = np.arange(100)
    keep = iv.eligible_for_fit(pos, lag=1, horizon=5, decision_pos=60)
    assert keep.sum() == 55 and keep[54] and not keep[55], "the last row kept is the one whose label ends ON the decision close (position 60)"
    assert not iv.overlap(pos[keep] + 1, iv.label_ends(pos[keep], 1, 5), 61, 80).any(), "no kept label reaches into the test block, which starts the day after the decision"
    late = np.flatnonzero(iv.overlap(pos + 1, iv.label_ends(pos, 1, 5), 61, 80))
    assert late[0] == 55 and late[-1] == 79, "every row from position 55 on has a label that reaches the test block"
    assert iv.eligible_for_fit(pos, 1, 5, 60, embargo=3).sum() == 52


def test_concurrency_and_uniqueness_match_a_hand_calculation_and_a_brute_force_count():
    starts, ends = np.array([0, 1, 2]), np.array([2, 3, 4])
    assert list(iv.concurrency(starts, ends, 6)) == [1, 2, 3, 2, 1, 0]
    u = iv.average_uniqueness(starts, ends, 6)
    assert u == pytest.approx([(1 + 1 / 2 + 1 / 3) / 3, (1 / 2 + 1 / 3 + 1 / 2) / 3, (1 / 3 + 1 / 2 + 1) / 3])
    w = iv.uniqueness_weights(starts, ends, 6)
    assert w.mean() == pytest.approx(1.0) and w[1] < w[0] == pytest.approx(w[2])
    rng = np.random.default_rng(0)
    s = np.sort(rng.integers(0, 150, 80))
    e = s + rng.integers(0, 12, 80)
    c = np.zeros(170)
    for a, b in zip(s, e):
        c[a:b + 1] += 1                                                                    # the slow way
    assert np.array_equal(iv.concurrency(s, e, 170), c)
    slow = np.array([np.mean([1.0 / c[d] for d in range(a, b + 1)]) for a, b in zip(s, e)])
    assert iv.average_uniqueness(s, e, 170) == pytest.approx(slow)


def test_labels_one_day_apart_share_h_minus_one_returns_and_a_non_overlapping_design_has_unit_uniqueness():
    h, pos = 5, np.arange(60)
    s, e = iv.return_days(pos, lag=1, horizon=h)
    assert list(e - s + 1) == [h] * 60, "a label is built from h daily returns"
    u = iv.average_uniqueness(s, e, 80)
    assert u[30] == pytest.approx(1.0 / h), "in the middle of a run of daily labels every return is shared by h labels"
    assert u[0] > u[30] and u[59] > u[30], "labels at the ends of a run overlap fewer neighbours"
    spaced = np.arange(0, 60, h)
    s2, e2 = iv.return_days(spaced, 1, h)
    assert iv.average_uniqueness(s2, e2, 80) == pytest.approx(np.ones(len(spaced)))
    assert iv.uniqueness_weights(s, e, 80).mean() == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------------------ through the walk-forward
@pytest.fixture(scope="module")
def ds():
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 300, "refit_every": 120}}, defaults=mc.DEFAULTS)
    return dsm.build_dataset(make_bundle(6, 900, seed=81, alpha=1.0), cfg["dataset"], "intervals"), cfg


def oracle_rows(data, t_first, embargo, start_pos=0):
    """The training rows by the brute-force definition: the label has a value, and the day its interval ends on is not after the decision day less the embargo."""
    y = data.Y[data.target].notna().to_numpy()
    pos = data.positions()
    ends = np.array([data.dates[min(p + data.lag + data.horizon, len(data.dates) - 1)] for p in pos])
    complete = (pos + data.lag + data.horizon) < len(data.dates)
    ok = y & complete & (pos >= start_pos) & (ends <= data.dates[t_first - embargo])
    return np.flatnonzero(ok)


@pytest.mark.parametrize("embargo", [0, 3])
def test_the_walk_forward_trains_on_exactly_the_rows_the_oracle_allows_and_no_label_reaches_the_first_decision(ds, embargo):
    data, cfg = ds
    res = tr.walk_forward_train(data, {"kind": "ridge", "params": {}}, {**cfg["split"], "embargo": embargo}, 0)
    assert len(res.folds) >= 3
    for f in res.folds:
        if f.get("skipped"):
            continue
        t_first = int(data.dates.get_indexer([pd.Timestamp(f["test_start"])])[0])
        want = oracle_rows(data, t_first, embargo)
        assert f["n_train"] == len(want) and f["train_rows_hash"] == tr.row_hash(want), "the model was fitted to exactly the rows whose labels had ended"
        assert pd.Timestamp(f["label_end_max"]) <= data.dates[t_first - embargo] and pd.Timestamp(f["fit_cutoff"]) == data.dates[t_first]
        assert f["rows_removed_by_interval_rule"] == 0 and f["rows_added_by_interval_rule"] == 0, "on a regular calendar the interval rule and the old position rule agree"


def test_with_dates_that_have_no_label_the_interval_rule_keeps_rows_the_position_rule_would_have_dropped_and_still_leaks_nothing(ds):
    data, cfg = ds
    import dataclasses

    base = tr.walk_forward_train(data, {"kind": "ridge", "params": {}}, cfg["split"], 0)
    dates = data.dates
    near = [int(dates.get_indexer([pd.Timestamp(f["test_start"])])[0]) for f in base.folds]
    gaps = dates[[t - k for t in near for k in (2, 3)]]                                       # no asset has a label on two of the six days before each decision
    Y = data.Y.copy()
    Y.loc[Y.index.get_level_values(0).isin(gaps)] = np.nan
    gapped = dataclasses.replace(data, Y=Y)
    res = tr.walk_forward_train(gapped, {"kind": "ridge", "params": {}}, cfg["split"], 0)
    assert sum(f.get("rows_added_by_interval_rule", 0) for f in res.folds) > 0, "the list of labelled dates is shorter than the calendar, so counting positions on it purged too much"
    for f in res.folds:
        if f.get("skipped"):
            continue
        t_first = int(dates.get_indexer([pd.Timestamp(f["test_start"])])[0])
        want = oracle_rows(gapped, t_first, 0)
        assert f["n_train"] == len(want) and f["train_rows_hash"] == tr.row_hash(want)
        assert pd.Timestamp(f["label_end_max"]) <= dates[t_first]


def test_a_target_whose_name_and_horizon_disagree_is_refused_before_a_dataset_is_built():
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price"], "target": "fwd_ret_20", "horizon": 5}}, defaults=mc.DEFAULTS)
    with pytest.raises(ValueError, match="looks 20 days ahead but dataset.horizon is 5"):
        dsm.build_dataset(make_bundle(4, 400, seed=1), cfg["dataset"], "mismatch")


def test_the_dataset_records_its_label_convention(ds):
    data, _ = ds
    assert data.meta["label_convention"]["entry"] == "close of day t + lag" and data.lag == 1 and data.horizon == 5


# ------------------------------------------------------------------------------------------------------------------ rolling windows, longer targets, K-fold and CPCV on a calendar with missing dates
def test_a_rolling_window_trains_on_the_rows_inside_it_whose_labels_had_ended(ds):
    data, cfg = ds
    split = {**cfg["split"], "window": "rolling", "rolling_days": 250}
    res = tr.walk_forward_train(data, {"kind": "ridge", "params": {}}, split, 0)
    expanding = tr.walk_forward_train(data, {"kind": "ridge", "params": {}}, cfg["split"], 0)
    ok = [f for f in res.folds if not f.get("skipped")]
    assert len(ok) >= 3
    for f in ok:
        t_first = int(data.dates.get_indexer([pd.Timestamp(f["test_start"])])[0])
        start = int(data.dates.get_indexer([pd.Timestamp(f["train_start"])])[0])
        want = oracle_rows(data, t_first, 0, start_pos=start)
        assert f["n_train"] == len(want) and f["train_rows_hash"] == tr.row_hash(want), "the window's rows, minus those whose labels had not ended"
        assert pd.Timestamp(f["label_end_max"]) <= data.dates[t_first]
    assert ok[-1]["n_train"] < [f for f in expanding.folds if not f.get("skipped")][-1]["n_train"], "a rolling window forgets the oldest rows"


def test_a_target_with_a_longer_horizon_than_the_dataset_default_is_purged_by_its_own_interval():
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 300, "refit_every": 120}}, defaults=mc.DEFAULTS)
    data = dsm.build_dataset(make_bundle(6, 900, seed=81, alpha=1.0), cfg["dataset"], "multi", extra_targets=("fwd_ret_20",))
    res = tr.walk_forward_train(data, {"kind": "ridge", "params": {}}, cfg["split"], 0, target="fwd_ret_20")
    for f in res.folds:
        if f.get("skipped"):
            continue
        t_first = int(data.dates.get_indexer([pd.Timestamp(f["test_start"])])[0])
        y = data.Y["fwd_ret_20"].notna().to_numpy()
        pos = data.positions()
        want = np.flatnonzero(y & (pos + data.lag + 20 <= t_first))
        assert f["n_train"] == len(want) and f["train_rows_hash"] == tr.row_hash(want), "20-day labels, 20-day interval"
        assert pd.Timestamp(f["label_end_max"]) <= data.dates[t_first]
    model = tr.fit_final(data, {"kind": "ridge", "params": {}}, 0, target="fwd_ret_20")
    last_labelled = int(data.positions()[data.Y["fwd_ret_20"].notna().to_numpy()].max())
    assert pd.Timestamp(model.fit_info_["label_end_max"]) == data.dates[min(last_labelled + data.lag + 20, len(data.dates) - 1)]


def gapped_positions(seed, n_days=260, keep=0.75, assets=3):
    """Decision positions of rows on a calendar with missing dates: each asset has a row on a date unless the date is missing for the whole market (a holiday of the data, not of the exchange)."""
    rng = np.random.default_rng(seed)
    days = np.flatnonzero(rng.random(n_days) < keep)
    return np.repeat(days, assets), days


def brute_excluded(pos, test_rows, groups, horizon, embargo):
    """Which rows a training set must leave out, by looking at every pair: a row whose label interval (lag cancels) intersects a test row's, and a row in the embargo after a test group's last label."""
    out = set()
    tests = pos[test_rows]
    for i, p in enumerate(pos):
        if i in set(test_rows):
            continue
        if any(abs(int(p) - int(t)) <= horizon for t in tests):
            out.add(i)
        for g in groups:
            if g.max() + horizon < p <= g.max() + horizon + embargo:
                out.add(i)
    return out


@pytest.mark.parametrize("seed", range(4))
def test_purged_kfold_leaves_out_exactly_the_overlapping_and_embargoed_rows_on_a_calendar_with_missing_dates(seed):
    pos, days = gapped_positions(seed)
    lag, h, embargo = 1, 5, 4
    splits = iv.purged_kfold_rows(pos, h, n_splits=5, embargo=embargo)
    assert len(splits) == 5 and not any(s.chronological for s in splits) and {s.kind for s in splits} == {"purged_kfold"}
    blocks = np.array_split(np.unique(pos), 5)
    for s, group in zip(splits, blocks):
        assert set(pos[s.test]) == set(group.tolist()) and not (set(s.train) & set(s.test))
        dropped = brute_excluded(pos, s.test, [group], h, embargo)
        assert set(s.train) == set(range(len(pos))) - set(s.test) - dropped, f"fold {s.fold}"
        starts, ends = iv.label_ends(pos[s.train], lag, 0), iv.label_ends(pos[s.train], lag, h)
        t_starts, t_ends = iv.label_ends(pos[s.test], lag, 0), iv.label_ends(pos[s.test], lag, h)
        assert not any(((starts[i] <= t_ends) & (ends[i] >= t_starts)).any() for i in range(len(starts))), "no training label overlaps a test label"
    assert max(int(pos[s.train].max()) for s in splits) > min(int(pos[s.test].min()) for s in splits[:2]), "they train on later blocks: that is why they are not chronological"


@pytest.mark.parametrize("seed", range(3))
def test_cpcv_leaves_out_exactly_the_overlapping_and_embargoed_rows_for_every_combination(seed):
    pos, days = gapped_positions(seed)
    h, embargo = 5, 3
    splits = iv.cpcv_rows(pos, h, n_groups=6, n_test_groups=2, embargo=embargo)
    assert len(splits) == 15 and sorted({g for s in splits for g in s.test_groups}) == list(range(6))
    blocks = np.array_split(np.unique(pos), 6)
    for s in splits:
        groups = [blocks[g] for g in s.test_groups]
        assert set(pos[s.test]) == set(np.concatenate(groups).tolist()) and not (set(s.train) & set(s.test))
        dropped = brute_excluded(pos, s.test, groups, h, embargo)
        assert set(s.train) == set(range(len(pos))) - set(s.test) - dropped, (s.fold, s.test_groups)
    with pytest.raises(ValueError):
        iv.cpcv_rows(pos, h, n_groups=3, n_test_groups=3)
    with pytest.raises(ValueError, match="at least 10 dates"):
        iv.purged_kfold_rows(pos[:60], h, n_splits=8)


def test_counting_the_horizon_in_rows_of_a_list_with_missing_dates_removes_more_than_the_overlap():
    """What the old rule did on a gapped list: it counted ``horizon`` rows, which are more than ``horizon`` calendar days apart where dates are missing. The interval rule removes exactly the overlap."""
    pos, days = gapped_positions(3, assets=1)
    h = 5
    interval_train = set(iv.purged_kfold_rows(pos, h, 5, 0)[2].train.tolist())
    n = len(pos)
    lo, hi = np.array_split(np.arange(n), 5)[2][[0, -1]]
    rows_rule = set(range(0, max(lo - h, 0))) | set(range(min(hi + h + 1, n), n))
    assert rows_rule < interval_train and len(interval_train) - len(rows_rule) >= 1


# ------------------------------------------------------------------------------------------------------------------ labels of different lengths and the longest of several heads
def brute_pairs_excluded(pos, horizons, lag, test_rows, groups, embargo):
    """Every pair checked: a row is out if its label interval shares a day with a test row's, or if its label starts in the embargo after a group's last label ends."""
    out = set()
    start = pos + lag
    end = pos + lag + horizons
    tests = set(test_rows.tolist())
    for i in range(len(pos)):
        if i in tests:
            continue
        if any(start[i] <= end[t] and end[i] >= start[t] for t in tests):
            out.add(i)
        for g in groups:
            last = max(end[j] for j in range(len(pos)) if pos[j] in g)
            if last < start[i] <= last + embargo:
                out.add(i)
    return out


@pytest.mark.parametrize("seed", range(3))
def test_labels_of_different_lengths_are_purged_by_their_own_intervals_in_kfold_and_cpcv(seed):
    pos, days = gapped_positions(seed, n_days=200, assets=2)
    rng = np.random.default_rng(100 + seed)
    horizons = rng.integers(2, 13, len(pos))                                                       # an irregular label: some end after two days, some after twelve
    lag, embargo = 1, 3
    blocks5, blocks6 = np.array_split(np.unique(pos), 5), np.array_split(np.unique(pos), 6)
    for s, group in zip(iv.purged_kfold_rows(pos, horizons, 5, embargo, lag), blocks5):
        want = set(range(len(pos))) - set(s.test) - brute_pairs_excluded(pos, horizons, lag, s.test, [set(group.tolist())], embargo)
        assert set(s.train) == want, f"purged k-fold {s.fold}"
    for s in iv.cpcv_rows(pos, horizons, 6, 2, embargo, lag)[:6]:
        groups = [set(blocks6[g].tolist()) for g in s.test_groups]
        want = set(range(len(pos))) - set(s.test) - brute_pairs_excluded(pos, horizons, lag, s.test, groups, embargo)
        assert set(s.train) == want, f"cpcv {s.test_groups}"
    uniform = iv.purged_kfold_rows(pos, 5, 5, embargo, lag)
    assert all(a.train.size > 0 for a in uniform) and iv.purged_kfold_rows(pos, np.full(len(pos), 5), 5, embargo, lag)[0].train.tolist() == uniform[0].train.tolist(), "an array of equal horizons is the same as one number"


def test_eligibility_uses_each_rows_own_horizon_and_the_longest_of_several_heads():
    pos = np.arange(100)
    primary, longest = np.full(100, 5), np.full(100, 20)
    a, b = iv.eligible_for_fit(pos, 1, primary, 60), iv.eligible_for_fit(pos, 1, np.maximum(primary, longest), 60)
    assert a.sum() == 55 and b.sum() == 40 and (b <= a).all(), "a head that looks 20 days ahead decides which rows have a complete label at day 60"
    mixed = np.where(pos % 2 == 0, 5, 20)
    got = iv.eligible_for_fit(pos, 1, mixed, 60)
    assert all(got[i] == (i + 1 + mixed[i] <= 60) for i in range(100)) and list(iv.label_ends(pos[:4], 1, mixed[:4])) == [6, 22, 8, 24]


def test_a_panel_built_for_another_target_horizon_lag_or_scaling_is_refused_not_trained_on(ds):
    from src.ml import experiments as xe

    data, cfg = ds
    ok = {"dataset": cfg["dataset"], "split": cfg["split"], "model": {"kind": "ridge", "params": {}}}
    info = {"source": "test", "label": "synthetic", "universe": "synthetic"}
    kw = dict(dataset=data, info=info, registry=None, save=False, explain=False, final_model=False, robustness=False, stop_after="validation")
    assert xe.run_experiment(ok, **kw).report["dataset"]["target"] == data.target
    for change, text in (({"target": "fwd_ret_10", "horizon": 10}, "target: the panel has fwd_ret_5"), ({"lag": 2}, "lag: the panel has 1"), ({"normalize": "rank"}, "normalize: the panel has cross_sectional_z"),
                         ({"feature_groups": ["price"]}, "feature_groups"), ({"horizon": 5, "target": "fwd_ret_5", "min_history": 300}, "min_history")):
        with pytest.raises(xe.ExperimentError, match=f"does not match the configuration.*{text}"):
            xe.run_experiment({**ok, "dataset": {**cfg["dataset"], **change}}, **kw)
