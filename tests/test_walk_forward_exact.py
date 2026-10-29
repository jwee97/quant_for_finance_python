"""Walk-forward folds, purged k-fold and the out-of-sample record, checked against dates and numbers worked out by hand.

A planted-bug audit (``experiments/plant_bugs.py``) found ``validation/walk_forward.py`` weakly pinned: of the places where one operator, comparison or number was changed the tests noticed 18% (13 of
74). The module is where the embargo between training and test lives, so the dates of every fold, the gap between them and what is dropped as too short are stated here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.validation.walk_forward import Fold, WalkForwardResult, WalkForwardSplitter, in_sample_vs_out_of_sample, purged_kfold_indices, run_walk_forward

DAILY = pd.bdate_range("2010-01-04", "2019-12-31")


class Config:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def dates(folds, field):
    return [getattr(f, field).date().isoformat() for f in folds]


# ------------------------------------------------------------------------------------------------------------------ the folds
def test_a_fold_names_itself_and_describes_its_dates():
    fold = Fold(3, pd.Timestamp("2010-01-04"), pd.Timestamp("2014-12-13"), pd.Timestamp("2015-01-04"), pd.Timestamp("2016-01-03"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        fold.index = 4
    assert fold.label == "fold03_2015-01-04_2016-01-03"
    assert fold.describe() == {"fold": 3, "train_start": "2010-01-04", "train_end": "2014-12-13", "test_start": "2015-01-04", "test_end": "2016-01-03", "train_days": None, "test_days": None}


def test_the_default_splitter_trains_on_five_years_tests_a_year_and_leaves_21_days_between():
    plain = WalkForwardSplitter()
    assert (plain.scheme, plain.train_years_min, plain.test_months, plain.rolling_train_years, plain.embargo_days) == ("expanding", 5.0, 12, 8.0, 21)
    assert [f.index for f in plain.iter_folds(DAILY)] == [1, 2, 3, 4, 5] and plain.split(DAILY) == list(plain.iter_folds(DAILY))
    bare = WalkForwardSplitter.from_config(Config())
    assert (bare.scheme, bare.train_years_min, bare.test_months, bare.rolling_train_years, bare.embargo_days) == ("expanding", 5.0, 12, 8.0, 21)
    assert WalkForwardSplitter.from_config(Config(**{"backtest.walk_forward": None})).test_months == 12
    given = WalkForwardSplitter.from_config(Config(**{"backtest.walk_forward": {"scheme": "rolling", "train_years_min": "2.5", "test_months": "6", "rolling_train_years": 3, "embargo_days": "10"}}))
    assert (given.scheme, given.train_years_min, given.test_months, given.rolling_train_years, given.embargo_days) == ("rolling", 2.5, 6, 3.0, 10)


def test_expanding_folds_start_after_the_minimum_history_and_each_test_year_follows_the_last():
    folds = WalkForwardSplitter(scheme="expanding", train_years_min=5, test_months=12, embargo_days=21).split(DAILY)
    assert dates(folds, "test_start") == ["2015-01-04", "2016-01-04", "2017-01-04", "2018-01-04", "2019-01-04"]
    assert dates(folds, "test_end") == ["2016-01-03", "2017-01-03", "2018-01-03", "2019-01-03", "2019-12-31"], "a year less a day, and the last cut at the end of the data"
    assert set(dates(folds, "train_start")) == {"2010-01-04"} and dates(folds, "train_end") == ["2014-12-13", "2015-12-13", "2016-12-13", "2017-12-13", "2018-12-13"]
    assert all((f.test_start - f.train_end).days == 22 for f in folds), "21 days between the last training day and the first test day"
    assert [f.index for f in folds] == [1, 2, 3, 4, 5] and folds[0].label == "fold01_2015-01-04_2016-01-03"


def test_rolling_folds_keep_a_window_of_the_stated_length_and_fractions_of_a_year_count():
    rolling = WalkForwardSplitter(scheme="rolling", train_years_min=5, test_months=12, rolling_train_years=3, embargo_days=21).split(DAILY)
    assert dates(rolling, "train_start") == ["2011-12-13", "2012-12-13", "2013-12-13", "2014-12-13", "2015-12-13"] and dates(rolling, "train_end") == ["2014-12-13", "2015-12-13", "2016-12-13", "2017-12-13", "2018-12-13"]
    half = WalkForwardSplitter(scheme="rolling", train_years_min=5, test_months=12, rolling_train_years=2.5, embargo_days=21).split(DAILY)
    assert half[0].train_start == pd.Timestamp("2012-06-13"), "two and a half years back from 13 December 2014"
    short = WalkForwardSplitter(scheme="rolling", train_years_min=5, test_months=12, rolling_train_years=20, embargo_days=21).split(DAILY)
    assert set(dates(short, "train_start")) == {"2010-01-04"}, "a window longer than the history starts at the first day"
    fractional = WalkForwardSplitter(train_years_min=5.5, test_months=12).split(DAILY)
    assert fractional[0].test_start == pd.Timestamp("2015-07-04"), "five years and six months of history before the first test"
    assert WalkForwardSplitter(train_years_min=5.9, test_months=12).split(DAILY)[0].test_start == pd.Timestamp("2015-11-04"), "0.9 of a year is ten whole months"
    assert WalkForwardSplitter(scheme="rolling", train_years_min=5, rolling_train_years=2.9, embargo_days=21).split(DAILY)[0].train_start == pd.Timestamp("2012-02-13"), "2.9 years back is two years and ten months"
    shorter_tests = WalkForwardSplitter(train_years_min=5, test_months=6, embargo_days=0).split(DAILY)
    assert shorter_tests[0].test_end == pd.Timestamp("2015-07-03") and shorter_tests[1].test_start == pd.Timestamp("2015-07-04") and (shorter_tests[0].test_start - shorter_tests[0].train_end).days == 1


def test_a_first_fold_whose_training_would_end_before_it_starts_is_skipped_and_a_short_last_block_is_dropped():
    no_history = WalkForwardSplitter(train_years_min=0, test_months=12, embargo_days=21).split(DAILY)
    assert no_history[0].test_start == pd.Timestamp("2011-01-04") and no_history[0].train_start == DAILY[0] and no_history[0].train_end == pd.Timestamp("2010-12-13"), "the first year has no training before it"
    assert no_history[0].index == 1
    flush = WalkForwardSplitter(train_years_min=0, test_months=12, embargo_days=364).split(DAILY)          # a year less a day of embargo leaves a first fold whose training ends the day it starts
    assert flush[0].test_start == pd.Timestamp("2012-01-04") and flush[0].train_end > flush[0].train_start, "a fold with no training days at all is skipped"
    twenty = WalkForwardSplitter().split(pd.bdate_range("2010-01-04", "2019-01-31"))           # 20 trading days in the last block: enough
    nineteen = WalkForwardSplitter().split(pd.bdate_range("2010-01-04", "2019-01-30"))         # 19: too few to judge
    assert len(twenty) == 5 and twenty[-1].test_start == pd.Timestamp("2019-01-04") and twenty[-1].test_end == pd.Timestamp("2019-01-31")
    assert len(nineteen) == 4 and nineteen[-1].test_end == pd.Timestamp("2019-01-03")


def test_too_little_data_gives_no_folds_and_the_limit_is_three_hundred_observations():
    weekly = pd.date_range("2010-01-04", periods=300, freq="W-MON")                           # 5.75 years of weekly dates
    assert len(WalkForwardSplitter().split(weekly)) == 1
    assert WalkForwardSplitter().split(weekly[:299]) == [] and WalkForwardSplitter().split(DAILY[:100]) == []
    assert WalkForwardSplitter().split(pd.DatetimeIndex(list(reversed(weekly)))) == WalkForwardSplitter().split(weekly), "the order of the index does not matter"


# ------------------------------------------------------------------------------------------------------------------ running folds
def returns_for(index: pd.DatetimeIndex, seed: int = 1) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(0.0004, 0.01, len(index)), index=index)


def test_a_walk_forward_run_keeps_only_test_days_and_tabulates_each_fold_by_hand():
    index = DAILY
    r = returns_for(index)
    splitter = WalkForwardSplitter()

    def fit(fold):
        gross = r.loc[fold.test_start:fold.test_end]
        return gross, gross - 0.0001, pd.DataFrame({"A": 0.5}, index=gross.index)

    result = run_walk_forward(index, fit, splitter, name="toy")
    folds = splitter.split(index)
    assert isinstance(result, WalkForwardResult) and result.name == "toy" and len(result.fold_table) == 5
    assert result.oos_returns.index[0] == DAILY[DAILY >= folds[0].test_start][0] and result.oos_returns.index[-1] == DAILY[-1], "the record starts with the first test day and nothing before it"
    assert result.oos_returns.index.is_monotonic_increasing and not result.oos_returns.index.has_duplicates
    assert result.oos_net_returns.equals(result.oos_returns - 0.0001) and len(result.weights) == len(result.oos_returns)
    row = result.fold_table.iloc[1]
    net = (r.loc[folds[1].test_start:folds[1].test_end] - 0.0001).to_numpy()
    curve = np.cumprod(1.0 + net)
    assert row["status"] == "ok" and row["fold"] == 2 and row["test_start"] == "2016-01-04" and row["train_end"] == "2015-12-13"
    assert row["test_days"] == len(net) and row["test_return"] == pytest.approx(curve[-1] - 1.0) and row["test_vol"] == pytest.approx(net.std(ddof=1) * np.sqrt(252))
    assert row["test_sharpe"] == pytest.approx(net.mean() * 252 / (net.std(ddof=1) * np.sqrt(252))) and row["test_max_drawdown"] == pytest.approx((curve / np.maximum.accumulate(curve) - 1.0).min())
    assert row["test_hit_rate"] == pytest.approx((net > 0).mean())
    monthly = run_walk_forward(index, fit, splitter, periods_per_year=12).fold_table.iloc[1]
    assert monthly["test_vol"] == pytest.approx(net.std(ddof=1) * np.sqrt(12)) and monthly["test_sharpe"] == pytest.approx(net.mean() * 12 / (net.std(ddof=1) * np.sqrt(12)))
    assert run_walk_forward(index, fit, splitter).name == "strategy", "the default name"


def test_a_failing_empty_or_flat_fold_is_a_row_in_the_table_and_not_a_silent_gap():
    index = DAILY
    r = returns_for(index)

    def fit(fold):
        if fold.index == 2:
            raise RuntimeError("boom")
        if fold.index == 3:
            return pd.Series(dtype=float), None, None
        gross = r.loc[fold.test_start:fold.test_end]
        if fold.index == 4:
            gross = pd.Series(0.0004, index=gross.index)                                    # the same return every day
        return gross, None, None

    result = run_walk_forward(index, fit, WalkForwardSplitter())
    table = result.fold_table
    assert table["status"].tolist() == ["ok", "failed: boom", "empty", "ok", "ok"] and table.loc[1, "fold"] == 2 and pd.isna(table.loc[1, "test_return"])
    assert result.oos_net_returns.equals(result.oos_returns), "without a net series the gross one is used"
    assert len(result.weights) == 0 and isinstance(result.weights, pd.DataFrame)
    assert pd.isna(table.loc[3, "test_sharpe"]) and table.loc[3, "test_vol"] < 1e-9, "a fold that never moves has no Sharpe ratio, not 1e16 from rounding noise"
    summary = result.summary()
    assert result.summary(periods_per_year=252)["sharpe"] == summary["sharpe"] and result.summary(12)["sharpe"] != summary["sharpe"] and result.summary(12)["ann_vol"] == pytest.approx(result.summary()["ann_vol"] * np.sqrt(12 / 252))
    assert summary["n_folds"] == 5 and summary["name"] == "strategy" and summary["folds_positive"] == int((table["test_return"] > 0).sum())
    assert summary["fold_sharpe_mean"] == pytest.approx(table["test_sharpe"].mean()) and summary["fold_sharpe_std"] == pytest.approx(table["test_sharpe"].std(ddof=1))
    assert summary["worst_fold_return"] == pytest.approx(table["test_return"].min()) and "sharpe" in summary and summary["n_obs"] == len(result.oos_net_returns)
    with pytest.raises(RuntimeError, match="produced no out-of-sample returns"):
        run_walk_forward(index, lambda fold: (pd.Series(dtype=float), None, None), WalkForwardSplitter(), name="nothing")
    with pytest.raises(ValueError, match="no walk-forward folds"):
        run_walk_forward(DAILY[:100], fit, WalkForwardSplitter())


def test_a_fold_of_zero_returns_is_not_positive_and_a_zero_day_is_not_a_hit():
    r = returns_for(DAILY)

    def fit(fold):
        gross = r.loc[fold.test_start:fold.test_end].copy()
        if fold.index == 1:
            gross[:] = 0.0                                                                    # a fold that returned exactly nothing
        if fold.index == 2:
            gross.iloc[::2] = 0.0                                                             # half of its days returned exactly nothing
        return gross, None, None

    result = run_walk_forward(DAILY, fit, WalkForwardSplitter())
    table = result.fold_table
    assert table.loc[0, "test_return"] == 0.0 and table.loc[0, "test_hit_rate"] == 0.0 and result.summary()["folds_positive"] == int((table["test_return"] > 0).sum()) and (table["test_return"] > 0).sum() <= 4
    assert table.loc[1, "test_hit_rate"] == pytest.approx(float((r.loc[result.fold_table.loc[1, "test_start"]:result.fold_table.loc[1, "test_end"]].iloc[1::2] > 0).sum()) / table.loc[1, "test_days"])


def test_overlapping_folds_keep_the_first_value_of_a_day():
    index = DAILY
    r = returns_for(index)
    marks = {}

    def fit(fold):
        gross = r.loc[fold.test_start:fold.test_end].copy()
        gross.iloc[-1] = 1000.0 + fold.index                                                 # every fold also reports the same extra day
        extra = pd.Series([2000.0 + fold.index], index=[pd.Timestamp("2019-12-30")])
        marks[fold.index] = 2000.0 + fold.index
        return pd.concat([gross, extra]).sort_index(), None, None

    result = run_walk_forward(index, fit, WalkForwardSplitter())
    assert not result.oos_returns.index.has_duplicates and result.oos_returns[pd.Timestamp("2019-12-30")] == 2001.0, "the first fold to report a day keeps it"


def test_the_gap_between_an_in_sample_and_an_out_of_sample_sharpe_is_the_slippage():
    rng = np.random.default_rng(3)
    inside = pd.Series(rng.normal(0.002, 0.01, 400))
    outside = pd.Series(np.r_[rng.normal(0.0005, 0.01, 300), [np.nan]])
    out = in_sample_vs_out_of_sample(inside, outside)
    is_sharpe = inside.mean() / inside.std(ddof=1) * np.sqrt(252)
    oos_sharpe = outside.dropna().mean() / outside.dropna().std(ddof=1) * np.sqrt(252)
    assert out["is_sharpe"] == pytest.approx(is_sharpe) and out["oos_sharpe"] == pytest.approx(oos_sharpe) and out["sharpe_decay"] == pytest.approx(is_sharpe - oos_sharpe)
    assert out["sharpe_retention"] == pytest.approx(oos_sharpe / is_sharpe) and out["is_ann_return"] == pytest.approx(inside.mean() * 252) and out["oos_ann_return"] == pytest.approx(outside.mean() * 252)
    assert out["is_n_obs"] == 400 and out["oos_n_obs"] == 300, "a missing day is not an observation"
    assert in_sample_vs_out_of_sample(inside, outside, periods_per_year=12)["is_sharpe"] == pytest.approx(inside.mean() / inside.std(ddof=1) * np.sqrt(12))
    barely = pd.Series(np.tile([0.01, -0.01], 50) + 7.56e-13)                                       # an in-sample Sharpe ratio of 1.2e-9: positive, and small enough that retention is still a ratio
    near = in_sample_vs_out_of_sample(barely, outside)
    assert 1.0e-9 < near["is_sharpe"] < 1.5e-9 and np.isfinite(near["sharpe_retention"])
    zero = in_sample_vs_out_of_sample(inside - inside.mean(), outside)
    assert np.isnan(zero["sharpe_retention"]) and zero["sharpe_decay"] == pytest.approx(-oos_sharpe), "no in-sample edge: nothing to retain"


# ------------------------------------------------------------------------------------------------------------------ purged k-fold
def test_purged_folds_remove_the_days_around_each_test_block_from_its_training_set():
    index = pd.bdate_range("2015-01-01", periods=500)
    folds = purged_kfold_indices(index, n_splits=5, embargo_days=10, horizon=4)
    assert len(folds) == 5 and [len(test) for _, test in folds] == [100] * 5
    train, test = folds[2]
    assert test.tolist() == list(range(200, 300)) and all(isinstance(a, np.ndarray) for a in (train, test))
    start, end = index[200], index[299]
    purge_start, purge_end = start - pd.Timedelta(days=int(4 * 1.5) + 10), end + pd.Timedelta(days=10)
    expected = np.where((index < purge_start) | (index > purge_end))[0]
    assert train.tolist() == expected.tolist()
    assert not ((index[train] >= purge_start) & (index[train] <= purge_end)).any() and index[train].max() > purge_end and index[train].min() == index[0]
    assert index[train][index[train] < purge_start].max() < purge_start and (index[train] < purge_start).sum() == int((index < purge_start).sum()), "everything before the purge is kept"
    first_train, first_test = folds[0]
    assert first_test.tolist() == list(range(100)) and first_train.min() > 100 and (index[first_train] > index[99] + pd.Timedelta(days=10)).all(), "nothing before the first block, and the embargo after it"
    assert sorted(np.concatenate([t for _, t in folds]).tolist()) == list(range(500)), "every day is in exactly one test block"
    wider = purged_kfold_indices(index, 5, 10, horizon=20)[2][0]
    assert len(wider) < len(train), "a longer forecast horizon purges more days"


def test_a_block_with_too_little_left_to_train_on_or_nothing_to_test_is_left_out():
    calendar = pd.date_range("2020-01-01", periods=100, freq="D")
    one = purged_kfold_indices(calendar, n_splits=2, embargo_days=0, horizon=1)
    assert len(one) == 1 and len(one[0][0]) == 50 and one[0][1].tolist() == list(range(50)), "exactly fifty training days is enough, and 49 is not"
    assert purged_kfold_indices(calendar, n_splits=2, embargo_days=1, horizon=1) == [], "with a day of embargo each block has 49 or fewer days to train on"
    tiny = purged_kfold_indices(pd.date_range("2020-01-01", periods=3, freq="D"), n_splits=5)
    assert tiny == [], "five blocks of three days leave no block with fifty days to train on"
    full = pd.bdate_range("2015-01-01", periods=1000)
    explicit = purged_kfold_indices(full, 5, 21, 1)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(purged_kfold_indices(full), explicit)), "five blocks, 21 days of embargo and a horizon of one day unless told otherwise"
    default = purged_kfold_indices(pd.bdate_range("2015-01-01", periods=1000))
    assert len(default) == 5 and purged_kfold_indices(pd.bdate_range("2015-01-01", periods=1000), n_splits=4)[0][1].tolist() == list(range(250))
