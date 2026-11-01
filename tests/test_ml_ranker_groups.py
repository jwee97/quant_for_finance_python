"""P0-05: ranker query groups are recomputed AFTER the label filter. A recording stand-in for the LightGBM ranker captures the groups and relevance the model is trained with."""

import numpy as np
import pandas as pd
import pytest

from src.ml import models as md
from src.ml.boosting import MIN_QUERY, query_order, relevance_buckets
from src.utils.optional import unavailable_reason

if unavailable_reason("lightgbm"):
    pytest.skip("LightGBM is not available here", allow_module_level=True)


class Recorder:
    """Takes the place of ``LGBMRanker``: remembers what ``fit`` was given and predicts zeros."""

    def fit(self, X, y, group=None, qid=None):
        self.X, self.y, self.group = X, np.asarray(y), np.asarray(group)
        return self

    def predict(self, X):
        return np.zeros(len(X))


def panel(n_dates=80, n_assets=8, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-06", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"A{i}" for i in range(n_assets)]], names=["date", "asset"])
    X = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["f1", "f2", "f3"])
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    return X, y, dates


def fit_recorded(X, y, **params):
    model = md.make_model("lgbm_ranker", params, "regression", 0)
    rec = Recorder()
    model._estimator = lambda levels: rec
    model.fit(X, y)
    return model, rec


def test_group_sizes_count_only_the_rows_that_survive_the_label_filter():
    X, y, dates = panel()
    day = lambda i: y.index.get_level_values(0) == dates[i]                                       # noqa: E731
    y[day(5)] = np.where(np.arange(8) < 2, y[day(5)], np.nan)                                     # two finite labels of eight: below the minimum group
    y[day(6)] = np.nan                                                                            # no labels at all
    y[day(7)] = 0.01                                                                              # eight tied labels: one relevance level, nothing to rank
    y[day(10)] = np.where(np.arange(8) < 5, y[day(10)], np.nan)                                   # five finite labels: a group of five, not eight
    model, rec = fit_recorded(X, y)
    sizes = list(rec.group)
    assert sizes.count(5) == 1 and sizes.count(8) == 80 - 4 and len(sizes) == 77 and sum(sizes) == len(rec.y)           # 80 dates less the three that cannot teach a ranker anything
    ex = model.diagnostics()["excluded"]
    assert ex["non_finite_label_rows"] == 6 + 8 + 3 and ex["small_group_dates"] == 1 and ex["single_relevance_dates"] == 1 and ex["rows_used"] == sum(sizes) and ex["dates_used"] == 77
    assert MIN_QUERY <= min(sizes) and not np.isnan(rec.y.astype(float)).any() and rec.y.min() >= 0 and rec.y.max() <= 4
    assert ex["dates"] == {"fewer_than_3_finite_labels": [str(dates[5].date())], "no_finite_label": [str(dates[6].date())], "one_relevance_level": [str(dates[7].date())]}, "the excluded dates are named, with the reason"


def test_every_kept_group_has_more_than_one_relevance_level_and_the_groups_are_contiguous_in_date_order():
    X, y, _ = panel(seed=3)
    y.iloc[::7] = np.nan
    model, rec = fit_recorded(X, y)
    start = 0
    for size in rec.group:
        block = rec.y[start:start + size]
        assert len(np.unique(block)) >= 2
        start += size
    assert start == len(rec.y)


def test_shuffling_the_input_rows_changes_neither_the_groups_nor_the_relevance():
    X, y, _ = panel(seed=4)
    y.iloc[::5] = np.nan
    _, base = fit_recorded(X, y)
    order = np.random.default_rng(0).permutation(len(X))
    _, shuffled = fit_recorded(X.iloc[order], y.iloc[order])
    assert list(base.group) == list(shuffled.group) and sorted(base.y) == sorted(shuffled.y)


def test_relevance_buckets_handle_ties_and_singletons_and_the_query_order_is_stable():
    keys = np.array([1, 1, 1, 1, 2, 2, 3])
    rel = relevance_buckets(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 5.0]), keys, 5)
    assert len(set(rel[:4])) == 1 and rel[4] < rel[5] and 0 <= rel.min() and rel.max() <= 4
    order, counts = query_order(np.array([3, 1, 2, 1, 3, 2, 1]))
    assert list(counts) == [3, 2, 2] and list(order) == [1, 3, 6, 2, 5, 0, 4], "a stable sort by date keeps the original order inside a group"
