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
