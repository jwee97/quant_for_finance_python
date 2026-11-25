"""A universe of one or two tickers has no cross-section: a z-score or a rank across assets would leave every cell missing and every row would be dropped. The dataset now scales each asset by its own expanding past
(never by a later observation), says so in its metadata, and, when nothing can be built, names the rule that removed the rows with numbers per asset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as dsm
from tests.ml_helpers import make_bundle

CFG = {"dataset": {"feature_groups": ["price", "volatility"], "min_history": 100}, "split": {"min_train_days": 400}}


def build(n_assets, n=900, **dataset):
    cfg = mc.experiment_config({**CFG, "dataset": {**CFG["dataset"], **dataset}}, defaults=mc.DEFAULTS)
    return dsm.build_dataset(make_bundle(n_assets, n, seed=11, alpha=1.0), cfg["dataset"], "small")


@pytest.mark.parametrize("n_assets", [1, 2])
def test_one_or_two_assets_build_a_dataset_with_time_series_scaling_and_say_so(n_assets):
    ds = build(n_assets)
    assert len(ds.X) > 100 and ds.X.notna().any().mean() > 0.9                  # a market-relative feature (beta) cannot exist for one asset; the rest do
    norm = ds.meta["normalization"]
    assert norm["requested"] == "cross_sectional_z" and norm["applied"] == "time_series_z" and "at least 3 assets" in norm["note"]
    assert ds.meta["assets_in_universe"] == n_assets


def test_three_assets_keep_the_cross_sectional_scaling_and_say_nothing_was_changed():
    ds = build(3)
    assert ds.meta["normalization"] == {"requested": "cross_sectional_z", "applied": "cross_sectional_z", "note": None}
    day = ds.X.xs(ds.X.index.get_level_values(0)[-1], level=0)
    col = day.columns[0]
    assert abs(float(day[col].mean())) < 1e-5


def test_time_series_scaling_uses_only_the_past():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, 2)), columns=["A", "B"], index=pd.bdate_range("2020-01-01", periods=300))
    full = dsm.normalise_time_series(frame)
    changed = frame.copy()
    changed.iloc[200:] += 50.0                                                 # the future moves a lot
    after = dsm.normalise_time_series(changed)
    pd.testing.assert_frame_equal(full.iloc[:200], after.iloc[:200])           # nothing before day 200 may notice
    assert full.iloc[:dsm.TS_MIN_PERIODS - 1].isna().all().all() and full.iloc[dsm.TS_MIN_PERIODS:].notna().all().all()
    ref = (frame["A"].iloc[:150] - frame["A"].iloc[:150].mean()) / frame["A"].iloc[:150].std()
    assert full["A"].iloc[149] == pytest.approx(ref.iloc[-1])                  # the formula, by hand


def test_an_explicit_time_series_choice_works_for_any_universe():
    ds = build(4, normalize="time_series_z")
    assert ds.meta["normalization"]["applied"] == "time_series_z" and ds.meta["normalization"]["note"] is None


def test_an_empty_dataset_says_which_rule_removed_the_rows_and_how_many_days_each_asset_has():
    with pytest.raises(ValueError) as err:
        build(2, n=120, min_history=252)
    text = str(err.value)
    assert "min_history = 252" in text and "priced days" in text and "Per asset" in text and "Lower min_history" in text
