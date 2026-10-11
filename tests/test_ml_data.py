"""The machine-learning data layer: configuration, causal features, targets that start after the features end, time splitters that purge overlapping labels, and the dataset builder."""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as ds, features as F, splitters as sp, targets as tg
from src.framework import bundle_from_prices
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------------------------------------- configuration
def test_configuration_merges_validates_and_fingerprints():
    cfg = mc.experiment_config({"model": {"kind": "lasso", "params": {"alpha": 0.1}}, "signal": {"n": 2}}, defaults=mc.DEFAULTS)
    assert cfg["model"] == {"kind": "lasso", "params": {"alpha": 0.1}} and cfg["signal"]["n"] == 2 and cfg["signal"]["method"] == "top_n_long_short"
    assert cfg["name"] == "lasso_fwd_ret_5_etf" and mc.config_hash(cfg) == mc.config_hash(mc.experiment_config({"model": {"kind": "lasso", "params": {"alpha": 0.1}}, "signal": {"n": 2}, "name": "x"}, defaults=mc.DEFAULTS))
    other = mc.experiment_config({"model": {"kind": "lasso", "params": {"alpha": 0.2}}, "signal": {"n": 2}}, defaults=mc.DEFAULTS)
    assert mc.config_hash(other) != mc.config_hash(cfg) and mc.dataset_key(other) == mc.dataset_key(cfg)             # a different model on the same data
    assert mc.dataset_key(mc.experiment_config({"dataset": {"horizon": 20, "target": "fwd_ret_20"}}, defaults=mc.DEFAULTS)) != mc.dataset_key(cfg)
    for bad, text in [({"dataset": {"horizon": 0}}, "horizon"), ({"dataset": {"normalize": "zscore"}}, "normalize"), ({"split": {"forward_fraction": 0.5, "test_fraction": 0.3}}, "fractions"),
                      ({"backtest": {"rebalance": "hourly"}}, "rebalance"), ({"backtest": {"cost_bps": 400, "slippage_bps": 400}}, "costs"), ({"signal": {"holding_period": 0}}, "holding_period"),
                      ({"nonsense": 1}, "unknown configuration sections"), ({"model": {"kind": "regime_ridge"}}, "regime_detector")]:
        with pytest.raises(mc.MLConfigError, match=text):
            mc.experiment_config(bad, defaults=mc.DEFAULTS)
    assert mc.merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}} and mc.merge({"model": {"params": {"x": 1}}}, {"model": {"params": {"y": 2}}}) == {"model": {"params": {"y": 2}}}


def test_the_yaml_defaults_load_and_agree_with_the_python_defaults():
    loaded = mc.load_ml_config()
    for section in mc.SECTIONS:
        assert set(loaded[section]) >= set(mc.DEFAULTS[section]) - {"regime_detector"} or section == "model"
    mc.validate(mc.experiment_config({}))


# --------------------------------------------------------------------------------------------------------------------- features
@pytest.fixture(scope="module")
def bundle():
    b = make_bundle(10, 1100, seed=3)
    macro = b.macro.copy()
    for a in b.assets[:4]:
        macro[f"CARRY_{a}"] = 0.02 * np.sin(np.arange(len(macro)) / 40.0 + hash(a) % 5)
        macro[f"SLOPE_{a}"] = 0.01 * np.cos(np.arange(len(macro)) / 55.0)
        macro[f"BASISMOM_{a}"] = 0.03 * np.sin(np.arange(len(macro)) / 70.0)
    macro["FUNDING_BTC"] = 0.05 + 0.03 * np.sin(np.arange(len(macro)) / 30.0)
    macro["BASIS_BTC"] = 0.001 * np.cos(np.arange(len(macro)) / 20.0)
    macro["STABLE_SUPPLY"] = 1e11 * np.exp(0.0005 * np.arange(len(macro)))
    b.macro = macro
    return b


def test_every_feature_is_causal_even_when_the_future_is_replaced_by_noise(bundle):
    cutoff = bundle.index[700]
    noisy = bundle.perturbed_after(cutoff, scale=3.0, seed=1)
    a = F.compute_features(F.inputs_from_bundle(bundle), F.GROUPS)
    b = F.compute_features(F.inputs_from_bundle(noisy), F.GROUPS)
    assert set(a.frames) == set(b.frames) and len(a.frames) > 60
    # assets named BTC-style prefixes are absent here, so the crypto frames are empty and skipped; the others must match exactly up to the cutoff
    for name, frame in a.frames.items():
        left, right = frame.loc[:cutoff].to_numpy(dtype=float), b.frames[name].loc[:cutoff].to_numpy(dtype=float)
        assert np.allclose(left, right, equal_nan=True, rtol=1e-4, atol=1e-5), f"{name} changed before the cutoff: it looks ahead"
    changed = [n for n in a.frames if not np.allclose(a.frames[n].loc[cutoff + pd.Timedelta(days=30):].to_numpy(dtype=float), b.frames[n].loc[cutoff + pd.Timedelta(days=30):].to_numpy(dtype=float), equal_nan=True)]
    assert len(changed) > 0.8 * len(a.frames)                                                       # the noise really did change the features after the cutoff, so the test could have failed


def test_groups_without_their_inputs_are_skipped_with_a_reason_not_faked():
    b = make_bundle(8, 600, volume=False, macro=False)
    fs = F.compute_features(F.inputs_from_bundle(b), F.GROUPS)
    for name in ("log_dollar_vol_21", "amihud_21", "carry", "iv_level", "funding_ann", "macro_VIX_z"):
        assert name in fs.skipped and name not in fs.frames
    assert "volume" in fs.skipped["amihud_21"] and "CARRY" in fs.skipped["carry"] and "VIX" in fs.skipped["iv_level"]
    with pytest.raises(ValueError, match="unknown feature groups"):
        F.compute_features(F.inputs_from_bundle(b), ["astrology"])


def test_known_feature_values():
    idx = pd.bdate_range("2020-01-01", periods=400)
    p = pd.DataFrame({"X": 100.0 * 1.001 ** np.arange(400), "Y": 100.0 * 0.999 ** np.arange(400)}, index=idx)
    b = bundle_from_prices(p, min_history=30)
    fs = F.compute_features(F.inputs_from_bundle(b), ["price", "momentum", "volatility"])
    assert fs.frames["ret_21"]["X"].iloc[-1] == pytest.approx(1.001 ** 21 - 1, rel=1e-4) and fs.frames["ret_21"]["Y"].iloc[-1] < 0
    assert fs.frames["mom_12_1"]["X"].iloc[-1] == pytest.approx(1.001 ** (252 - 21) - 1, rel=1e-4)
    assert fs.frames["dist_high_252"]["X"].iloc[-1] == pytest.approx(0.0, abs=1e-6) and fs.frames["dist_low_252"]["Y"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert fs.frames["vol_21"]["X"].iloc[-1] == pytest.approx(0.0, abs=1e-6)                     # constant growth: no volatility
    assert fs.frames["rsi_14"]["X"].iloc[-1] == pytest.approx(0.5, abs=1e-6) and fs.frames["rsi_14"]["Y"].iloc[-1] == pytest.approx(-0.5, abs=1e-6)
    slope = F.rolling_slope(pd.DataFrame({"a": np.arange(100.0) * 2.0 + 1.0}), 20)
    assert slope["a"].iloc[-1] == pytest.approx(2.0) and np.isnan(slope["a"].iloc[10])


def test_a_duplicate_feature_name_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        F.feature("ret_1", "price", "again")(lambda x: x.prices)
    with pytest.raises(ValueError, match="unknown feature group"):
        F.feature("zzz", "nowhere", "bad")


# --------------------------------------------------------------------------------------------------------------------- targets
def _vals(frame: pd.DataFrame) -> np.ndarray:
    """The non-missing values of a frame (``DataFrame.stack`` no longer drops them in recent pandas)."""
    a = frame.to_numpy(dtype=float).ravel()
    return a[~np.isnan(a)]


def _line(n=60, k=3):
    idx = pd.bdate_range("2021-01-01", periods=n)
    prices = pd.DataFrame({f"S{i}": 100.0 + (i + 1) * np.arange(n, dtype=float) for i in range(k)}, index=idx)
    return prices, prices.pct_change()


def test_forward_targets_start_after_the_features_and_leave_the_last_rows_empty():
    prices, returns = _line()
    for lag, h in itertools.product((0, 1, 3), (1, 5, 20)):
        fwd = tg.forward_return(prices, h, lag)
        assert fwd["S0"].iloc[4] == pytest.approx(prices["S0"].iloc[4 + lag + h] / prices["S0"].iloc[4 + lag] - 1.0)
        assert fwd.iloc[-(lag + h):].isna().all().all() and fwd.iloc[: len(fwd) - (lag + h)].notna().all().all()
        assert tg.label_end(h, lag) == lag + h
    # the label at t may not depend on any price before t + lag (the feature date and the lag in between)
    base = tg.forward_return(prices, 5, 2)
    changed = prices.copy()
    changed.iloc[:12] = changed.iloc[:12] * 3.0 + 7.0
    after = tg.forward_return(changed, 5, 2)
    assert np.allclose(base.iloc[12 - 2:].dropna().to_numpy(), after.iloc[12 - 2:].dropna().to_numpy()) and not np.allclose(base.iloc[:5].to_numpy(), after.iloc[:5].to_numpy())


def test_target_kinds_have_the_documented_meaning():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-01", periods=300)
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, (300, 8)), axis=0), index=idx, columns=list("ABCDEFGH"))
    returns = prices.pct_change()
    fwd = tg.make_target("fwd_ret_5", prices, returns, 1)
    ex = tg.make_target("excess_ret_5", prices, returns, 1)
    assert np.allclose(ex.mean(axis=1).dropna(), 0.0, atol=1e-12) and np.allclose((fwd - ex).dropna().std(axis=1), 0.0, atol=1e-12)
    ex_a = tg.make_target("excess_ret_5", prices, returns, 1, benchmark="A")
    assert np.allclose(ex_a["A"].dropna(), 0.0)
    rank = tg.make_target("rank_5", prices, returns, 1)
    assert rank.max().max() == 1.0 and rank.min().min() == pytest.approx(1 / 8)
    tb = tg.make_target("topbottom_5", prices, returns, 1)
    assert set(np.unique(_vals(tb))) == {0.0, 1.0} and tb.notna().sum(axis=1).max() <= 5
    direction = tg.make_target("direction_5", prices, returns, 1)
    assert bool(np.all(_vals(((direction == 1.0) == (fwd > 0)).astype(float).where(fwd.notna())) == 1.0))
    dd = tg.make_target("fwd_dd_20", prices, returns, 1)
    assert (_vals(dd) <= 1e-12).all()
    i = 40
    window = prices.iloc[i + 1: i + 1 + 21]
    assert dd.iloc[i]["A"] == pytest.approx(window["A"].min() / window["A"].iloc[0] - 1.0)
    vol = tg.make_target("fwd_vol_20", prices, returns, 1)
    assert vol.iloc[i]["A"] == pytest.approx(returns["A"].iloc[i + 2: i + 22].std(ddof=1) * np.sqrt(252.0))
    ra = tg.make_target("riskadj_5", prices, returns, 1)
    assert ra.iloc[100]["B"] == pytest.approx(fwd.iloc[100]["B"] / (returns["B"].iloc[:101].tail(63).std(ddof=1) * np.sqrt(5.0)))
    assert tg.is_classification("direction_5") and tg.is_classification("topbottom_20") and not tg.is_classification("fwd_ret_5")
    for bad in ("fwd_ret", "nonsense_5", "fwd_ret_0", "fwd_ret_999"):
        with pytest.raises(ValueError):
            tg.parse_target(bad)
    with pytest.raises(ValueError, match="at least 5 days"):
        tg.make_target("fwd_vol_1", prices, returns, 1)
    with pytest.raises(ValueError, match="not in the universe"):
        tg.make_target("excess_ret_5", prices, returns, 1, benchmark="ZZZ")


# --------------------------------------------------------------------------------------------------------------------- splitters
@pytest.mark.parametrize("n,first,step,purge,embargo,window", [(1000, 400, 21, 6, 0, "expanding"), (1000, 400, 63, 21, 5, "expanding"), (1000, 500, 21, 6, 2, "rolling"), (700, 300, 10, 1, 0, "rolling")])
def test_walk_forward_splits_tile_the_test_period_and_never_overlap(n, first, step, purge, embargo, window):
    splits = sp.walk_forward_splits(n, first, step, purge, embargo, window, rolling_days=200)
    test = np.concatenate([s.test for s in splits])
    assert test[0] == first and test[-1] == n - 1 and np.array_equal(test, np.arange(first, n))        # no gap and no overlap between test blocks
    for s in splits:
        assert sp.check_no_overlap(s, purge, embargo) and s.train[0] >= 0 and np.all(np.diff(s.train) == 1)
        assert s.train.max() + purge + embargo <= s.test.min()                                       # the last training label is complete by the first test day
        if window == "rolling":
            assert len(s.train) <= 200
    assert all(len(b.train) >= len(a.train) for a, b in zip(splits, splits[1:])) or window == "rolling"


def test_walk_forward_refuses_a_start_without_enough_training():
    with pytest.raises(ValueError, match="training positions"):
        sp.walk_forward_splits(500, 20, 10, 6)
    with pytest.raises(ValueError):
        sp.walk_forward_splits(500, 100, 0, 6)
    with pytest.raises(ValueError, match="expanding or rolling"):
        sp.walk_forward_splits(500, 100, 10, 6, window="sliding")


def test_purged_embargoed_and_cpcv_splits_drop_overlapping_rows():
    for s in sp.purged_kfold_splits(500, 5, purge=10, embargo=3):
        assert sp.check_no_overlap(s, 10, 3) and len(s.train) < 500 - len(s.test)
        assert not np.isin(s.train, np.arange(s.test.min() - 10 + 1, s.test.max() + 3 + 1)).any()
    e = sp.embargoed_split(400, 0.25, purge=6, embargo=4)
    assert e.test[0] == 300 and e.train.max() == 300 - 6 - 4 and sp.check_no_overlap(e, 6, 4)
    cp = sp.cpcv_splits(360, 6, 2, purge=8, embargo=2)
    assert len(cp) == 15
    counts = np.zeros(360)
    for s in cp:
        assert sp.check_no_overlap(s, 8, 2) and not np.isin(s.train, s.test).any() and len(s.groups) == 2
        counts[s.test] += 1
    assert np.all(counts == 5)                                                                    # every position is tested in k/N * C(N, k) = 5 splits
    with pytest.raises(ValueError):
        sp.cpcv_splits(360, 5, 5, 1)
    with pytest.raises(ValueError):
        sp.purged_kfold_splits(500, 1, 5)
    broken = sp.Split(np.arange(0, 100), np.arange(95, 120), "walk_forward")
    assert not sp.check_no_overlap(broken, 6)


def test_segment_plan_cuts_from_the_end_and_refuses_a_short_timeline():
    dates = pd.bdate_range("2010-01-01", periods=2000)
    plan = sp.segment_plan(dates, 0.2, 0.15, 0.15, 756)
    b = plan.bounds
    assert b["forward"][1] == 2000 and b["forward"][0] == 2000 - 300 and b["test"][1] == b["forward"][0] and b["validation"][1] == b["test"][0] and b["train"][1] == b["validation"][0]
    assert b["train"][1] - b["train"][0] >= 756 and plan.first_prediction() == b["validation"][0]
    labels = plan.labels()
    assert labels.iloc[0] == "train" and labels.iloc[-1] == "forward" and set(labels) == set(sp.SEGMENTS) and plan.segment_of(b["test"][0]) == "test"
    assert plan.to_dict()["forward"]["n_days"] == 300
    with pytest.raises(ValueError, match="warm-up"):
        sp.segment_plan(dates[:900], 0.2, 0.15, 0.15, 756)
    with pytest.raises(ValueError, match="only"):
        sp.segment_plan(dates[:1000], 0.2, 0.15, 0.02, 100)


# --------------------------------------------------------------------------------------------------------------------- datasets
def test_cross_sectional_normalisation_uses_only_the_date_and_clips_the_tails():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=50)
    frame = pd.DataFrame(rng.normal(size=(50, 12)), index=idx, columns=list("abcdefghijkl"))
    frame.iloc[3, 2] = 50.0                                                  # an outlier
    frame.iloc[7, :] = np.nan
    frame.iloc[9, :10] = np.nan                                              # two assets only: no cross-section
    z = ds.normalise_cross_section(frame, "cross_sectional_z", 0.05)
    ok = z.drop(z.index[[7, 9]])
    assert np.allclose(ok.mean(axis=1), 0.0, atol=1e-9) and np.allclose(ok.std(axis=1, ddof=0), 1.0, atol=1e-9)
    assert z.iloc[7].isna().all() and z.iloc[9].isna().all() and z.iloc[3].abs().max() < 4.0
    r = ds.normalise_cross_section(frame, "rank")
    assert r.drop(r.index[[7, 9]]).abs().max().max() <= 1.0 + 1e-12 and (ds.normalise_cross_section(frame, "none") is frame)
    changed = frame.copy()
    changed.iloc[20:] = changed.iloc[20:] * 100.0
    assert np.allclose(ds.normalise_cross_section(changed, "cross_sectional_z", 0.05).iloc[:20].to_numpy(), z.iloc[:20].to_numpy(), equal_nan=True)       # other dates do not matter
    vals = frame.to_numpy()
    q = ds._row_quantiles(vals, 0.1)
    assert np.allclose(q[:7], [np.nanquantile(vals[i], 0.1) for i in range(7)]) and np.isnan(q[7])


def test_a_dataset_aligns_features_and_labels_and_keeps_the_unlabelled_tail(bundle):
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "momentum", "volatility", "macro"], "target": "fwd_ret_5", "horizon": 5}}, defaults=mc.DEFAULTS)
    d = ds.build_dataset(bundle, cfg["dataset"], "t", ["fwd_ret_1", "direction_5"], {"source": "simulated"})
    assert d.target == "fwd_ret_5" and d.lag == 1 and d.horizon == 5 and d.meta["source"] == "simulated" and "ret_21" in d.features
    assert list(d.Y.columns) == ["fwd_ret_5", "fwd_ret_1", "direction_5"] and d.X.index.equals(d.Y.index) and d.X.index.names == ["date", "asset"]
    date, asset = d.dates[500], d.assets[3]
    expected = bundle.prices.iloc[500 + 1 + 5][asset] / bundle.prices.iloc[500 + 1][asset] - 1.0
    assert d.Y.loc[(date, asset), "fwd_ret_5"] == pytest.approx(expected, rel=1e-5)
    last = d.latest_rows()
    assert len(last) == len(d.assets) and last["ret_21"].notna().all() and d.Y.xs(d.dates[-1], level=0)["fwd_ret_5"].isna().all()           # features exist, the label does not yet
    X, y = d.xy(slice(0, 800))
    assert y.notna().all() and X.index.equals(y.index) and X.index.get_level_values(0).max() <= d.dates[799]
    rows = d.rows(np.array([300, 301, 302, 600]))
    assert set(d.X.index.get_level_values(0)[rows].unique()) == {d.dates[300], d.dates[301], d.dates[302], d.dates[600]}
    assert d.dates.get_indexer(d.X.index.get_level_values(0)).min() >= 251                    # an asset enters the sample only after min_history (252) observations
    macro_cols = [c for c in d.features if c.startswith("macro_")]
    assert macro_cols and all(d.X[c].groupby(level=0).nunique().max() <= 1 for c in macro_cols)           # market-wide features are identical across assets on a date
    assert all(abs(d.X["ret_21"].groupby(level=0).mean().dropna()) < 1e-6)                               # cross-sectionally standardised


def test_dataset_round_trips_through_parquet(bundle, tmp_path):
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}}, defaults=mc.DEFAULTS)
    d = ds.build_dataset(bundle, cfg["dataset"], "roundtrip")
    d.save(tmp_path / "ds")
    back = ds.Dataset.load(tmp_path / "ds")
    assert back.name == "roundtrip" and back.meta["target"] == d.meta["target"] and back.asset_class == d.asset_class
    pd.testing.assert_frame_equal(back.X, d.X, check_freq=False)
    pd.testing.assert_frame_equal(back.Y, d.Y, check_freq=False)
    assert back.summary()["n_rows"] == d.summary()["n_rows"]


def test_dataset_builder_refuses_inconsistent_requests(bundle):
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price"]}}, defaults=mc.DEFAULTS)
    with pytest.raises(ValueError, match="must agree"):
        ds.build_dataset(bundle, {**cfg["dataset"], "horizon": 10}, "x")
    with pytest.raises(ValueError, match="no feature could be computed"):
        ds.build_dataset(make_bundle(6, 500, macro=False), {**cfg["dataset"], "feature_groups": ["futures"]}, "x")
    with pytest.raises(ValueError, match="unknown universe"):
        ds.resolve_bundle(None, "mars")
    with pytest.raises(ValueError, match="ticker store"):
        ds.resolve_bundle(None, ["AAA", "BBB"])
    sliced = ds.build_dataset(bundle, {**cfg["dataset"], "start": str(bundle.index[300].date()), "end": str(bundle.index[600].date())}, "sliced")
    assert sliced.dates[0] >= bundle.index[300] and sliced.dates[-1] <= bundle.index[600]
    reg = ds.build_dataset(bundle, {**cfg["dataset"], "regime_detector": "vol_state"}, "regime")
    assert "regime_state" in reg.features and reg.X["regime_state"].dropna().isin([0.0, 1.0, 2.0]).all()
