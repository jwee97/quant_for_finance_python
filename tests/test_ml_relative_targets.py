"""Release 6, P2-01: sector-relative and residual targets, against hand-worked and independently computed values.

The raw target is the compounded return of the label window. A relative target subtracts something that is known only to the same window (the group mean) or that is estimated before it (a loading from the trailing
daily returns ending at the decision date), so it adds no look-ahead; both claims are tested by changing the data the target must not depend on.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as ds, targets as tg
from tests.ml_helpers import make_bundle

IDX = pd.bdate_range("2021-01-04", periods=40)


def panel(seed=0, n=40):
    rng = np.random.default_rng(seed)
    cols = ["A", "B", "C", "D", "E", "F"]
    r = pd.DataFrame(rng.normal(0.0005, 0.01, (n, len(cols))), index=IDX[:n], columns=cols)
    return 100.0 * (1.0 + r).cumprod(), r


GROUPS = {"A": "x", "B": "x", "C": "y", "D": "y", "F": "z"}            # E has no group; F is alone in its


def test_the_sector_relative_target_is_the_forward_return_minus_the_mean_of_the_group_and_missing_without_peers():
    prices, rets = panel(1)
    h, lag = 2, 1
    got = tg.make_target("sector_ret_2", prices, rets, lag=lag, groups=GROUPS)
    for i in (0, 7, 20):                                                                                # the forward return from first principles
        fwd = {a: prices[a].iloc[i + lag + h] / prices[a].iloc[i + lag] - 1.0 for a in prices}
        assert got["A"].iloc[i] == pytest.approx(fwd["A"] - (fwd["A"] + fwd["B"]) / 2.0, abs=1e-14)
        assert got["B"].iloc[i] == pytest.approx(-got["A"].iloc[i], abs=1e-14), "two members: the two relative returns are mirror images"
        assert got["C"].iloc[i] == pytest.approx(fwd["C"] - (fwd["C"] + fwd["D"]) / 2.0, abs=1e-14)
    assert got["E"].isna().all() and got["F"].isna().all(), "an asset with no group, and one alone in its group, has no relative return: not a zero"
    assert got.iloc[-(h + lag):].isna().all().all(), "the last lag + h rows have no label at all"
    holes = prices.copy()
    holes.loc[holes.index[10], "B"] = np.nan                                                          # B's label is missing for the rows whose window touches that price: entry at row i + 1, exit at row i + 3
    part = tg.make_target("sector_ret_2", holes, rets, lag=lag, groups=GROUPS)
    for i in (7, 9):
        assert np.isnan(part["A"].iloc[i]) and np.isnan(part["B"].iloc[i]), "with B missing, A has no peer that day: missing for both, not relative to itself"
        assert part["C"].iloc[i] == pytest.approx(got["C"].iloc[i]), "other groups are untouched"
    assert part["A"].iloc[8] == pytest.approx(got["A"].iloc[8]) and part["B"].iloc[8] == pytest.approx(got["B"].iloc[8]), "row 8 reads rows 9 and 11: the hole at 10 is not in its window"
    with pytest.raises(ValueError, match="needs a group map"):
        tg.make_target("sector_ret_2", prices, rets, lag=lag)


def test_the_residual_target_is_the_forward_return_minus_beta_times_the_benchmark_and_beta_is_the_trailing_slope():
    rng = np.random.default_rng(3)
    n = 420
    ix = pd.bdate_range("2019-01-01", periods=n)
    m = pd.Series(rng.normal(0.0004, 0.01, n), index=ix)
    r = pd.DataFrame({"M": m, "A": 1.5 * m, "B": 0.5 * m, "C": 0.8 * m + rng.normal(0, 0.01, n)}, index=ix)
    r.iloc[0] = 0.0
    prices = 100.0 * (1.0 + r).cumprod()
    out = tg.make_target("resid_ret_1", prices, r, lag=1, benchmark="M")
    assert out["A"].dropna().abs().max() < 1e-12 and out["B"].dropna().abs().max() < 1e-12 and out["M"].dropna().abs().max() < 1e-12, "an asset that is exactly beta times the benchmark has no residual"
    assert out["A"].iloc[:tg.BETA_MIN - 1].isna().all() and out["A"].iloc[tg.BETA_MIN + 5:-2].notna().all(), "no loading before the minimum history: no target"
    i = 300                                                                                              # an independent slope over the 252 sessions ending at i
    win = r.iloc[i - tg.BETA_WINDOW + 1:i + 1]
    beta = np.cov(win["C"], win["M"], ddof=1)[0, 1] / np.var(win["M"], ddof=1)
    fwd_c, fwd_m = prices["C"].iloc[i + 2] / prices["C"].iloc[i + 1] - 1.0, prices["M"].iloc[i + 2] / prices["M"].iloc[i + 1] - 1.0
    assert out["C"].iloc[i] == pytest.approx(fwd_c - beta * fwd_m, abs=1e-12)
    assert beta == pytest.approx(0.8, abs=0.12)


def test_neither_relative_target_reads_anything_after_the_label_window():
    prices, rets = panel(5, 40)
    base_s = tg.make_target("sector_ret_2", prices, rets, lag=1, groups=GROUPS)
    later = prices.copy()
    later.iloc[16:] *= 1.37                                                                              # everything from row 16 on moves; the label of row 10 ends at row 13
    moved_s = tg.make_target("sector_ret_2", later, rets, lag=1, groups=GROUPS)
    assert np.allclose(moved_s.iloc[:13].to_numpy(), base_s.iloc[:13].to_numpy(), equal_nan=True), "a row's label depends only on its own window"
    assert not np.allclose(moved_s.iloc[14:16].to_numpy(), base_s.iloc[14:16].to_numpy(), equal_nan=True), "while rows whose window reaches the move do change"
    rng = np.random.default_rng(7)
    n = 400
    ix = pd.bdate_range("2019-01-01", periods=n)
    r = pd.DataFrame(rng.normal(0, 0.01, (n, 4)), index=ix, columns=list("MABC"))
    bench = tg.benchmark_daily(r, "M")
    beta = tg.rolling_beta(r, bench)
    r2 = r.copy()
    r2.iloc[301:] = rng.normal(0, 0.05, (n - 301, 4))                                                     # a different future
    beta2 = tg.rolling_beta(r2, tg.benchmark_daily(r2, "M"))
    assert np.allclose(beta.iloc[:301].to_numpy(), beta2.iloc[:301].to_numpy(), equal_nan=True), "the loading at date t uses the returns up to and including t, nothing later"
    assert not np.allclose(beta.iloc[301:].to_numpy(), beta2.iloc[301:].to_numpy(), equal_nan=True)


def test_the_target_documentation_states_units_factors_and_that_a_static_sector_map_is_not_point_in_time():
    sec = tg.describe("sector_ret_5", GROUPS)
    assert sec["classification"]["point_in_time"] is False and sec["classification"]["groups"] == ["x", "y", "z"] and "look-ahead" in sec["classification"]["note"]
    res = tg.describe("resid_ret_20", benchmark="SPY", lag=1)
    assert res["factors"] == ["benchmark return (single factor)"] and res["benchmark"] == "SPY" and "252" in res["loading"] and "no point-in-time factor data" in res["other_factors"]
    assert res["horizon"] == 20 and "21 sessions" in res["window"] and res["unit"].startswith("decimal return")
    for kind in ("sector_ret", "resid_ret"):
        assert kind in tg.KINDS and f"{kind}_10" == f"{kind}_10" and tg.parse_target(f"{kind}_10") == (kind, 10)


def test_the_dataset_builder_carries_the_new_targets_beside_the_raw_one_and_records_what_they_mean():
    b = make_bundle(10, 700, seed=21, alpha=1.0)
    b.group = {a: ("tech" if i < 5 else "bank") for i, a in enumerate(b.prices.columns)}
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}}, defaults=mc.DEFAULTS)
    d = ds.build_dataset(b, cfg["dataset"], "rel", extra_targets=("excess_ret_5", "sector_ret_5", "resid_ret_5"))
    assert {"fwd_ret_5", "excess_ret_5", "sector_ret_5", "resid_ret_5"} <= set(d.Y.columns)
    docs = d.meta["target_documentation"]
    assert set(docs) == {"excess_ret_5", "sector_ret_5", "resid_ret_5"} and docs["sector_ret_5"]["classification"]["point_in_time"] is False
    y = d.Y.dropna(subset=["sector_ret_5"])
    by_date = y["sector_ret_5"].groupby(level=0).sum()
    assert y.index.get_level_values(1).isin(b.prices.columns).all() and by_date.abs().max() < 1e-6, "the sector-relative targets of a date sum to zero within each group, hence overall (stored as float32)"
    same_rows = d.Y[["fwd_ret_5", "sector_ret_5"]].dropna()
    assert len(same_rows) > 1000 and d.meta["label_convention"], "the extra targets share the raw target's label window and rows"
    plain = ds.build_dataset(make_bundle(10, 700, seed=21, alpha=1.0), cfg["dataset"], "plain")
    assert list(plain.Y.columns) == ["fwd_ret_5"] and plain.meta["target_documentation"] == {}, "a configuration that does not use them is unchanged"
