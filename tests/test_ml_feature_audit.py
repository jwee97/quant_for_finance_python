"""P0-14: what every feature reads, how far back it reaches, and that nothing after its date can change it.

``audit.feature_dependencies`` perturbs one input panel at a time and recomputes each feature. These tests check the table it produces against windows derived by hand, run it over every registered feature, and
then show that it has teeth by registering features that look ahead, fill backward, centre a window, use the whole sample, read an input they do not declare, or need more history than they declare. The last
group builds whole datasets from a market whose later data are replaced, blanked, frozen or whose later-listed assets differ, and requires every earlier feature row to be identical.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.assets.bundles import futures_bundle

from src.ml import audit as A, config as mc, datasets as dsm, features as F
from tests.ml_helpers import make_bundle, probe_inputs


@pytest.fixture(scope="module")
def x():
    return probe_inputs()


@pytest.fixture(scope="module")
def report(x):
    return A.feature_dependencies(x)


# ------------------------------------------------------------------------------------------------------------------ the whole library
def test_every_registered_feature_is_measured_and_none_looks_ahead_or_misreports_its_inputs(report):
    assert not report["skipped"], report["skipped"]
    assert set(report["features"]) == set(F.FEATURES) and len(report["features"]) >= 80
    assert A.violations(report) == []
    assert A.long_memory(report) == []
    for name, row in report["features"].items():
        assert set(row["forward"]) == set(A.KINDS)
        assert max(row["forward"].values()) <= A.TOLERANCE, f"{name}: {row['forward']}"
        assert set(row["reads"]) <= set(row["declared_reads"])
    assert {n for n, r in report["features"].items() if r["memory"] == "recursive"} == {"rsi_14", "macd_hist", "ewma_vol_30"}


# (reads, trailing rows the last value depends on), derived by hand from each feature's definition
HAND = {
    "ret_1": ({"prices"}, 2), "ret_21": ({"prices"}, 22), "ret_252": ({"prices"}, 253), "ret_lag_3": ({"returns"}, 3), "rev_1": ({"returns"}, 1), "rev_5": ({"prices"}, 6),
    "ma_ratio_50": ({"prices"}, 50), "dist_high_252": ({"prices"}, 251), "mom_6_1": ({"prices"}, 127),                       # shift(21) / shift(126): rows t-21 and t-126, so 127 rows back; a 252-row high/low window excluding nothing
    "mom_12_1": ({"prices"}, 253), "breakout_20": ({"prices"}, 21), "zscore_60": ({"prices"}, 60), "autocorr_63": ({"returns"}, 64),
    "vol_63": ({"returns"}, 63), "vol_ratio_21_63": ({"returns"}, 63), "vol_of_vol_63": ({"returns"}, 83), "drawdown_63": ({"prices"}, 62),
    "beta_63": ({"market", "returns"}, 63), "idio_vol_63": ({"market", "returns"}, 126), "corr_market_63": ({"market", "returns"}, 63),
    "ibs": ({"high", "low", "raw_close"}, 1), "parkinson_21": ({"high", "low"}, 21), "vol_surge_21": ({"volume"}, 21), "log_dollar_vol_21": ({"prices", "volume"}, 21),
    "rank_ret_63": ({"prices"}, 64), "rel_ret_21": ({"prices"}, 22), "carry": ({"macro"}, 1), "carry_to_vol": ({"macro", "returns"}, 63), "iv_chg_5": ({"macro"}, 6),
    "iv_minus_rv_21": ({"macro", "returns"}, 21), "macro_VIX_z": ({"macro"}, 252), "macro_VIX_chg21": ({"macro"}, 252), "funding_z_90": ({"macro"}, 90), "stable_growth_30": ({"macro"}, 31),
}


@pytest.mark.parametrize("name", sorted(HAND))
def test_the_measured_reads_and_window_equal_the_hand_derived_ones(report, name):
    reads, window = HAND[name]
    got = report["features"][name]
    assert set(got["reads"]) == reads and got["window"] == window, (name, got["reads"], got["window"])


def test_a_recursive_feature_depends_on_more_history_than_its_declared_span_and_the_audit_says_so(report):
    for name in ("rsi_14", "macd_hist", "ewma_vol_30"):
        row = report["features"][name]
        assert row["memory"] == "recursive" and row["window"] > row["declared_window"] and max(row["forward"].values()) <= A.TOLERANCE


# ------------------------------------------------------------------------------------------------------------------ the audit has teeth
def register(monkeypatch, name, fn, lookback=20, needs=(), memory="finite"):
    monkeypatch.setitem(F.FEATURES, name, F.FeatureSpec(name, "price", "a deliberately wrong feature", fn, lookback, needs, False, memory))


WRONG = {                                                                                                                              # name: (function, what the audit must say)
    "looks_ahead": (lambda x: x.prices.shift(-1) / x.prices - 1.0, "forward"),
    "centred_window": (lambda x: x.prices.rolling(5, center=True, min_periods=1).mean() / x.prices - 1.0, "forward"),
    "backward_fill": (lambda x: x.prices.bfill() / x.prices.shift(1) - 1.0, "forward"),
    "interpolated": (lambda x: x.prices.interpolate() / x.prices.shift(1) - 1.0, "forward"),
    "whole_sample_z": (lambda x: (x.prices - x.prices.mean()) / x.prices.std(), "forward"),
    "future_extreme": (lambda x: x.prices / x.prices[::-1].cummax()[::-1] - 1.0, "forward"),
}


@pytest.mark.parametrize("name", sorted(WRONG))
def test_a_feature_that_uses_later_data_is_found(x, monkeypatch, name):
    fn, _ = WRONG[name]
    register(monkeypatch, name, fn)
    rep = A.feature_dependencies(x, names=[name])
    assert max(rep["features"][name]["forward"].values()) > 1e-6, rep["features"][name]["forward"]
    assert any(name in v and "looks ahead" in v for v in A.violations(rep))


def test_only_the_straddling_hole_exposes_a_backward_fill(x, monkeypatch):
    """Without a missing value there is nothing to fill, so noise, blanking and freezing the future all pass; the hole that crosses the cutoff is the perturbation that sees the fill reach back."""
    register(monkeypatch, "backward_fill", WRONG["backward_fill"][0])
    row = A.feature_dependencies(x, names=["backward_fill"])["features"]["backward_fill"]["forward"]
    assert row["noise"] <= A.TOLERANCE and row["missing"] <= A.TOLERANCE and row["stale"] <= A.TOLERANCE and row["straddle"] > 1e-6


def test_a_feature_that_reads_an_input_it_does_not_declare_is_found(x, monkeypatch):
    register(monkeypatch, "secret_volume", lambda x: x.volume.rolling(5).mean() / x.volume - 1.0, needs=())
    rep = A.feature_dependencies(x, names=["secret_volume"])
    assert rep["features"]["secret_volume"]["reads"] == ["volume"] and any("reads ['volume']" in v for v in A.violations(rep))


def test_a_feature_that_needs_more_history_than_it_declares_is_found(x, monkeypatch):
    register(monkeypatch, "greedy", lambda x: x.prices / x.prices.rolling(100, min_periods=100).mean() - 1.0, lookback=20)
    rep = A.feature_dependencies(x, names=["greedy"])
    assert rep["features"]["greedy"]["window"] == 100 and any("depends on 100 trailing rows but declares 20" in v for v in A.violations(rep))


def test_an_expanding_statistic_is_not_a_finite_window(x, monkeypatch):
    register(monkeypatch, "expanding_mean", lambda x: x.prices / x.prices.expanding(min_periods=30).mean() - 1.0, lookback=60)
    rep = A.feature_dependencies(x, names=["expanding_mean"])
    assert rep["features"]["expanding_mean"]["window"] == len(x.prices) and max(rep["features"]["expanding_mean"]["forward"].values()) <= A.TOLERANCE
    assert A.long_memory(rep) == ["expanding_mean"]


def test_a_declared_recursive_feature_that_is_really_finite_is_questioned(x, monkeypatch):
    register(monkeypatch, "mislabelled", lambda x: x.prices / x.prices.rolling(10).mean() - 1.0, lookback=30, memory="recursive")
    assert any("declared recursive" in v for v in A.violations(A.feature_dependencies(x, names=["mislabelled"])))


# ------------------------------------------------------------------------------------------------------------------ whole datasets
CFG = mc.experiment_config({"dataset": {"feature_groups": ["price", "momentum", "volatility", "relative", "volume"], "min_history": 130}}, defaults=mc.DEFAULTS)


def build(bundle):
    return dsm.build_dataset(bundle, CFG["dataset"], "audit")


def altered(bundle, cutoff, how, seed=0):
    """The bundle with every market value after ``cutoff`` replaced (``noise``), blanked (``missing``), frozen at the cutoff's value (``stale``) or with the last asset stopping at the cutoff (``delisted``)."""
    rng = np.random.default_rng(seed)
    after = bundle.index > cutoff
    prices, volume, high, low = bundle.prices.copy(), bundle.volume.copy(), bundle.high.copy(), bundle.low.copy()
    if how == "noise":
        prices.loc[after] = prices.loc[after] * rng.uniform(0.6, 1.6, prices.loc[after].shape)
        volume.loc[after] = volume.loc[after] * rng.uniform(0.1, 9.0, volume.loc[after].shape)
    elif how == "missing":
        prices.loc[after] = np.nan
        volume.loc[after] = np.nan
    elif how == "stale":
        prices.loc[after] = prices.loc[~after].iloc[-1].to_numpy()
        volume.loc[after] = volume.loc[~after].iloc[-1].to_numpy()
    elif how == "delisted":
        prices.loc[after, prices.columns[-1]] = np.nan
    high, low = prices * 1.004, prices * 0.996
    new = dataclasses.replace(bundle, prices=prices, returns=prices.pct_change(fill_method=None), volume=volume, high=high, low=low)
    new.investable = prices.notna() & (prices.notna().cumsum() >= 60)
    return new


@pytest.fixture(scope="module")
def market():
    b = make_bundle(7, 700, seed=12, alpha=1.0)
    b.high, b.low = b.prices * 1.004, b.prices * 0.996
    b.returns = b.prices.pct_change(fill_method=None)
    return b


@pytest.mark.parametrize("how", ["noise", "missing", "stale", "delisted"])
def test_every_feature_row_on_or_before_the_cutoff_is_unchanged_whatever_happens_after_it(market, how):
    cutoff = market.index[500]
    base, other = build(market), build(altered(market, cutoff, how))
    before = base.X.index.get_level_values(0) <= cutoff
    left = base.X[before]
    right = other.X.reindex(left.index)
    assert len(left) > 1000 and right.notna().to_numpy().sum() > 0
    assert np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True, rtol=1e-6, atol=1e-6), f"{how}: a feature on or before the cutoff moved"
    assert set(left.index) <= set(other.X.index), "a row that was eligible at the cutoff must still be there"
    later = base.X.index.get_level_values(0) > market.index[560]
    if how != "delisted":
        assert not np.allclose(base.X[later].to_numpy(dtype=float)[:200], other.X.reindex(base.X[later].index).to_numpy(dtype=float)[:200], equal_nan=True), "the alteration must show up later, or the test is vacuous"


def test_labels_before_the_last_settled_interval_are_unchanged_and_later_ones_move(market):
    cutoff_pos = 500
    cutoff = market.index[cutoff_pos]
    base, other = build(market), build(altered(market, cutoff, "noise"))
    lag, h = base.lag, base.horizon
    settled = base.Y.index.get_level_values(0) <= market.index[cutoff_pos - lag - h]                      # a label is complete at the close of decision date + lag + horizon
    straddling = (base.Y.index.get_level_values(0) > market.index[cutoff_pos - lag - h]) & (base.Y.index.get_level_values(0) <= cutoff)
    ya, yb = base.Y[base.target], other.Y[other.target].reindex(base.Y.index)
    assert np.allclose(ya[settled].to_numpy(dtype=float), yb[settled].to_numpy(dtype=float), equal_nan=True)
    assert not np.allclose(ya[straddling].to_numpy(dtype=float), yb[straddling].to_numpy(dtype=float), equal_nan=True), "a label whose interval reaches past the cutoff uses the later price"


def test_an_asset_that_lists_after_the_cutoff_cannot_change_the_cross_section_before_it(market):
    """Cross-sectional availability: the later-listed asset's data differ between the two markets; the other assets' normalised features before its first date must be identical."""
    late = market.assets[-1]
    cut = 520
    a = dataclasses.replace(market, prices=market.prices.copy())
    a.prices.loc[a.index <= market.index[cut], late] = np.nan
    b = dataclasses.replace(a, prices=a.prices.copy())
    rng = np.random.default_rng(3)
    b.prices.loc[b.index > market.index[cut], late] = b.prices.loc[b.index > market.index[cut], late].to_numpy() * rng.uniform(0.5, 2.0, int((b.index > market.index[cut]).sum()))
    for m in (a, b):
        m.returns = m.prices.pct_change(fill_method=None)
        m.high, m.low = m.prices * 1.004, m.prices * 0.996
        m.investable = m.prices.notna() & (m.prices.notna().cumsum() >= 60)
    da, db = build(a), build(b)
    before = da.X.index.get_level_values(0) <= market.index[cut]
    assert late not in set(da.X[before].index.get_level_values(1)), "the asset has no row before it lists"
    assert np.allclose(da.X[before].to_numpy(dtype=float), db.X.reindex(da.X[before].index).to_numpy(dtype=float), equal_nan=True, rtol=1e-6, atol=1e-6)


# ------------------------------------------------------------------------------------------------------------------ the dataset's own audit
def test_the_dataset_audit_accounts_for_every_cell_and_names_what_it_left_out(market):
    ds = build(market)
    a = ds.meta["audit"]
    assert a["cells"] == len(market.index) * len(market.assets) == a["rows_kept"] + sum(a["excluded"].values())
    assert a["rows_kept"] == len(ds.X) and a["excluded"]["short_history"] > 0 and a["excluded"]["no_price"] == 0
    assert a["backward_fill"] == "none" and "median" in a["imputation"]
    lab = a["label"]
    assert lab["lag"] == ds.lag and lab["horizon"] == ds.horizon and lab["unlabelled_tail_dates"] == ds.lag + ds.horizon and lab["labelled_rows"] == int(ds.Y[ds.target].notna().sum())
    assert set(a["features"]) == set(ds.meta["audit"]["features"]) and all(0.0 <= r["missing_share"] <= 1.0 for r in a["features"].values())
    assert a["features"]["vol_63"]["inputs"] == ["market", "prices", "returns"] and a["features"]["vol_63"]["declared_window"] == 63


def test_a_frozen_price_series_is_reported_as_stale_not_as_a_traded_price(market):
    frozen = dataclasses.replace(market, prices=market.prices.copy())
    frozen.prices.loc[frozen.index[300:312], frozen.assets[2]] = float(frozen.prices[frozen.assets[2]].iloc[299])
    frozen.returns = frozen.prices.pct_change(fill_method=None)
    audit = build(frozen).meta["audit"]
    assert audit["stale"]["assets"] == [frozen.assets[2]] and audit["stale"]["longest_run"] >= 12
    clean = build(market).meta["audit"]["stale"]
    assert clean["assets"] == [] and clean["longest_run"] == 0


def test_the_audit_survives_saving_and_loading_the_dataset(market, tmp_path):
    ds = build(market)
    loaded = dsm.Dataset.load(ds.save(tmp_path / "d"))
    assert loaded.meta["audit"]["cells"] == ds.meta["audit"]["cells"] and loaded.meta["audit"]["label"]["target"] == ds.target


# ------------------------------------------------------------------------------------------------------------------ closed-market marks are not tradable prices
def test_a_futures_bundle_flags_exactly_the_cells_it_carried_over_a_holiday():
    from tests.test_futures_bundle_exact import curve

    days = pd.bdate_range("2021-01-04", "2021-04-02")
    holiday = pd.Timestamp("2021-02-15")
    closed = curve(days)
    closed = closed[closed["date"] != holiday]
    held = futures_bundle({"X": curve(days), "Y": closed}, hold_days=1)
    assert held.marks is not None and int(held.marks.to_numpy().sum()) == 1 and bool(held.marks.at[holiday, "Y"]) and not bool(held.marks.at[holiday, "X"])
    assert futures_bundle({"X": curve(days), "Y": closed}).marks is None, "nothing is carried by default, so nothing is a mark"


def test_a_carried_mark_is_not_a_row_not_investable_and_voids_the_labels_that_start_or_end_on_it(market):
    held = dataclasses.replace(market, prices=market.prices.copy())
    asset, shut = held.assets[1], [400, 401, 402]
    original = held.prices[asset].iloc[399]
    held.prices.iloc[shut, 1] = original                                                                  # the market was shut: the last price is carried
    held.returns = held.prices.pct_change(fill_method=None)
    held.marks = pd.DataFrame(False, index=held.index, columns=held.assets)
    held.marks.iloc[shut, 1] = True
    base, marked = build(dataclasses.replace(held, marks=None)), build(held)
    dates = held.index[shut]
    assert all((d, asset) in set(base.X.index) for d in dates) and not any((d, asset) in set(marked.X.index) for d in dates), "the rows on the shut days exist only when the mark is taken for a price"
    assert marked.meta["audit"]["excluded"]["closed_market_mark"] == 3 and marked.meta["audit"]["marks"]["carried_cells"] == 3
    assert not marked.investable.loc[dates, asset].any() and base.investable.loc[dates, asset].all()
    lag, h = marked.lag, marked.horizon
    ends_on_mark = [held.index[i - lag - h] for i in shut]                                                # decision dates whose exit price is a mark
    starts_on_mark = [held.index[i - lag] for i in shut]                                                  # decision dates whose entry price is a mark
    y = marked.Y[marked.target]
    for d in ends_on_mark + starts_on_mark:
        assert np.isnan(y.get((d, asset), np.nan)), f"the label from {d.date()} would have been earned at a price nothing traded at"
    voided = marked.meta["audit"]["label"]["labels_voided_by_marks"]
    assert voided == len(set(ends_on_mark + starts_on_mark)), voided
    other = [a for a in held.assets if a != asset][0]
    assert marked.Y[marked.target].xs(other, level=1).equals(base.Y[base.target].xs(other, level=1)), "no other asset is touched"


# ------------------------------------------------------------------------------------------------------------------ fills that used to look at later data
def old_hedge_weights(R, eta=5.0, vol_window=63):
    """``hedge_weights`` as it was before the fix, kept here as the negative control: the first volatility is back-filled from later days and a missing one is filled from the whole sample."""
    sig = R.rolling(vol_window, min_periods=20).std(ddof=1).shift(1).bfill().replace(0.0, np.nan).fillna(R.std(ddof=1))
    z = (R / sig).clip(-4, 4).fillna(0.0).to_numpy()
    n = R.shape[1]
    w, out = np.full(n, 1.0 / n), np.empty_like(z)
    for t in range(len(z)):
        out[t] = w
        w = w * np.exp(eta * 0.01 * z[t])
        w = np.maximum(w / w.sum(), 0.01)
        w = w / w.sum()
    return pd.DataFrame(out, index=R.index, columns=R.columns)


def test_hedge_weights_do_not_fill_the_first_volatility_from_later_days():
    from src.ml import ensembles as en

    rng = np.random.default_rng(3)
    R = pd.DataFrame(rng.normal(0.0, 0.01, (300, 4)), index=pd.bdate_range("2020-01-01", periods=300), columns=list("ABCD"))
    W = en.hedge_weights(R)
    assert np.allclose(W.iloc[:21], 0.25) and not np.allclose(W.iloc[21], 0.25), "no update until there are 20 days of trailing volatility, and updates after"
    later = R.copy()
    later.iloc[10:20] = rng.normal(0.0, 0.05, (10, 4))                       # days inside the window the old code back-filled the first volatility from
    assert np.array_equal(en.hedge_weights(later).iloc[:21].to_numpy(), W.iloc[:21].to_numpy()), "weights on those days cannot depend on days after them"
    after = R.copy()
    after.iloc[150:] = rng.normal(0.0, 0.05, (150, 4))
    assert np.array_equal(en.hedge_weights(after).iloc[:151].to_numpy(), W.iloc[:151].to_numpy())
    old, old_later = old_hedge_weights(R), old_hedge_weights(later)
    assert not np.allclose(old.iloc[8:21].to_numpy(), old_later.iloc[8:21].to_numpy()), "the old formula let a later day move an earlier weight: the check has teeth"


def test_a_combination_by_online_learning_records_that_its_volatility_is_not_back_filled():
    from src.ml import ensembles as en

    rng = np.random.default_rng(4)
    R = pd.DataFrame(rng.normal(0.0004, 0.01, (500, 3)), index=pd.bdate_range("2020-01-01", periods=500), columns=list("XYZ"))
    assert "causal_v2" in en.combine(R, "online_learning").params["volatility_fill"]
    assert "volatility_fill" not in en.combine(R, "equal").params


def test_the_segment_of_a_date_before_the_plan_starts_is_the_plans_first_and_nothing_is_taken_from_later_values():
    from src.ml.backtest import label_days

    index = pd.bdate_range("2020-01-01", periods=30)
    labels = pd.Series(["validation"] * 8 + ["test"] * 6, index=index[10:24])
    out = label_days(index, labels)
    assert (out.iloc[:10] == "validation").all() and (out.iloc[10:18] == "validation").all() and (out.iloc[18:] == "test").all()
