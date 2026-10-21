"""Transformations of factor panels: each uses only its own date's cross-section or its own security's past, and a missing value stays missing unless a fill step says otherwise."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import transforms as T
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("transforms"))
    svc.load_simulated(n_companies=60, start_year=2009, end_year=2016, seed=4)
    return ResearchContext(svc.store, UniverseSpec())


def _noise(ctx, seed=0, rows=24):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((rows, ctx.n)), index=ctx.dates[-rows:], columns=ctx.ids)


def test_winsorising_clips_each_dates_tails_at_the_stated_percentiles(ctx):
    x = _noise(ctx)
    x.iloc[:, 0] = 50.0
    for pct in T.WINSOR_LEVELS:
        w = T.winsorize(x, pct)
        hi = x.quantile(1 - pct / 100, axis=1)
        assert (w.max(axis=1) <= hi + 1e-12).all() and (w.min(axis=1) >= x.quantile(pct / 100, axis=1) - 1e-12).all()
    assert (T.winsorize(x, 0) == x).all().all()
    with pytest.raises(FundamentalsError):
        T.winsorize(x, 50)


def test_zscore_rank_and_percentile_work_on_each_date_and_leave_holes_alone(ctx):
    x = _noise(ctx, 1)
    x.iloc[3, 5], x.iloc[7, :10] = np.nan, np.nan
    z, r, p = T.zscore(x), T.rank(x), T.rank(x, pct=True)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-9) and np.allclose(z.std(axis=1), 1) and z.isna().equals(x.isna())
    assert r.min().min() == 1.0 and (r.max(axis=1) == x.notna().sum(axis=1)).all() and (p.max(axis=1) == 1.0).all() and p.isna().equals(x.isna())
    assert np.isnan(z.iloc[3, 5]) and not np.isnan(z.iloc[3, 4])


def test_group_ranks_and_means_are_taken_within_the_sector_and_small_groups_are_left_out(ctx):
    x = _noise(ctx, 2)
    sector = ctx.group("sector")
    r = T.group_rank(x, sector)
    one = x.iloc[-1]
    members = sector.index[sector == sector.iloc[0]]
    assert members.size >= T.MIN_GROUP
    inside = r.iloc[-1][members]
    assert inside.min() == pytest.approx(1 / len(members)) and inside.max() == pytest.approx(1.0) and set(np.round(inside.rank(), 6)) == set(np.round(one[members].rank(), 6))
    d = T.group_demean(x, sector)
    assert np.allclose(d.iloc[-1][members].mean(), 0, atol=1e-12)
    tiny = pd.Series(["a"] * (ctx.n - 3) + ["b"] * 3, index=ctx.ids)
    assert T.group_rank(x, tiny).iloc[-1][tiny.index[tiny == "b"]].isna().all() and T.group_rank(x, tiny).iloc[-1][tiny.index[tiny == "a"]].notna().all()
    zs = T.group_zscore(x, sector)
    assert zs.iloc[-1][members].std() == pytest.approx(1.0)


def test_neutralising_removes_what_the_groups_and_controls_explain(ctx):
    rng = np.random.default_rng(3)
    sector, country = ctx.group("sector"), ctx.group("country")
    size = np.log(ctx.market_cap_usd())
    base = pd.DataFrame(rng.standard_normal((len(ctx.dates), ctx.n)), index=ctx.dates, columns=ctx.ids)
    biased = base + sector.map({s: i for i, s in enumerate(sorted(set(sector)))}).to_numpy() * 0.8 + 0.5 * (size - size.mean().mean()).fillna(0.0)
    window = biased.iloc[40:60]
    by_sector = T.neutralize(window, ctx, ("sector",))
    means = by_sector.iloc[-1].groupby(sector).mean().dropna()
    assert np.allclose(means, 0, atol=1e-9)
    both = T.neutralize(window, ctx, ("sector", "country"))
    last = both.iloc[-1]
    assert np.allclose(last.groupby(sector).mean().dropna(), 0, atol=1e-9) and np.allclose(last.groupby(country).mean().dropna(), 0, atol=1e-9)
    with_size = T.neutralize(window, ctx, ("sector",), ("mcap",))
    row = with_size.iloc[-1]
    ok = row.notna() & size.iloc[59].notna()
    assert ok.sum() > 30 and abs(np.corrcoef(row[ok], size.iloc[59][ok])[0, 1]) < 1e-9 and abs(np.corrcoef(by_sector.iloc[-1][ok], size.iloc[59][ok])[0, 1]) > 0.1
    beta_vol = T.neutralize(window, ctx, (), ("beta", "volatility"))
    assert beta_vol.notna().sum().sum() > 0 and T.neutralize(window, ctx) is window
    with pytest.raises(FundamentalsError, match="cannot neutralise"):
        T.neutralize(window, ctx, ("planet",))
    with pytest.raises(FundamentalsError, match="unknown control"):
        T.neutralize(window, ctx, (), ("luck",))


def test_a_date_needs_the_minimum_number_of_complete_companies_and_five_more_than_the_columns_of_its_regression(ctx):
    window = _noise(ctx, 6, rows=3)

    def kept(frame, min_obs):
        return int(T.neutralize(frame, ctx, ("sector",), ("mcap",), min_obs=min_obs).iloc[-1].notna().sum())

    n = kept(window, 1)
    complete = list(T.neutralize(window, ctx, ("sector",), ("mcap",), min_obs=1).iloc[-1].dropna().index)
    assert n > 30 and kept(window, n) == n and kept(window, n + 1) == 0                                  # exactly the minimum is enough, one short is not
    columns = T._dummies(ctx, ("sector",)).shape[1] + 1                                                  # the sector columns and the size control
    for count, expected in ((columns + 5, columns + 5), (columns + 4, 0)):                                # with a small minimum, five more companies than columns are needed
        thin = window.copy()
        thin.iloc[-1] = np.where(thin.columns.isin(complete[:count]), thin.iloc[-1], np.nan)
        assert kept(thin, 1) == expected, count


def test_changes_percentage_changes_and_historical_z_scores_look_only_backwards(ctx):
    x = pd.DataFrame({1: [1.0, 2.0, 4.0, -1.0, 2.0, 3.0], 2: [np.nan, 1.0, 1.0, 1.0, 2.0, 1.0]}, index=pd.date_range("2020-01-31", periods=6, freq="ME"))
    assert T.delta(x, 2)[1].tolist()[2:] == [3.0, -3.0, -2.0, 4.0]
    pc = T.pct_change(x, 1)
    assert pc[1].iloc[1] == 1.0 and pc[1].iloc[2] == 1.0 and pc[1].iloc[3] == -1.25 and np.isnan(pc[1].iloc[4])       # a change from a loss (-1) is missing
    long = pd.DataFrame({1: np.arange(40.0) + np.random.default_rng(1).standard_normal(40)}, index=pd.date_range("2018-01-31", periods=40, freq="ME"))
    h = T.hist_zscore(long, 24, 12)
    assert h.iloc[:11].isna().all().all() and h.iloc[30, 0] > 0
    future_changed = long.copy()
    future_changed.iloc[35:] = 1000.0
    assert T.hist_zscore(future_changed, 24, 12).iloc[:35].equals(h.iloc[:35])                  # a later value never alters an earlier z-score


def test_a_pipeline_starts_with_the_investable_mask_and_names_its_steps(ctx):
    x = _noise(ctx, 5)
    mask = ctx.investable().reindex_like(x).fillna(False)
    steps = [{"op": "winsorize", "pct": 1}, {"op": "zscore"}, {"op": "clip", "limit": 2.5}, {"op": "sign", "direction": -1}]
    out = T.apply(x, steps, ctx)
    assert out.notna().equals(mask & x.notna()) and out.abs().max().max() <= 2.5 + 1e-12 and np.allclose(T.apply(x, steps[:3], ctx).to_numpy(), -out.to_numpy(), equal_nan=True)
    assert T.describe(steps) == "winsorize(pct=1) -> zscore -> clip(limit=2.5) -> sign(direction=-1)" and T.describe(None) == "raw values"
    ranked = T.apply(x, [{"op": "percentile"}], ctx)
    assert np.allclose(ranked.max(axis=1).dropna(), 1.0)                                         # ranked among the investable only
    with pytest.raises(FundamentalsError, match="unknown transformation"):
        T.apply(x, [{"op": "alchemy"}], ctx)
    assert np.isinf(T.apply(x * np.inf, [], ctx).to_numpy()).sum() == 0                          # infinities never survive


def test_a_missing_value_gets_a_value_only_from_a_named_fill_step(ctx):
    x = _noise(ctx, 6)
    holey = x.mask(np.random.default_rng(0).random(x.shape) < 0.2)
    assert T.apply(holey, [{"op": "zscore"}], ctx).isna().sum().sum() >= holey.where(ctx.investable().reindex_like(holey).fillna(False)).isna().sum().sum()
    neutral = T.apply(holey, [{"op": "zscore"}, {"op": "fill", "method": "neutral"}], ctx)
    assert neutral.isna().sum().sum() == 0 and (neutral == 0).sum().sum() > 0
    grouped = T.apply(holey, [{"op": "fill", "method": "group_median", "by": "sector"}], ctx)
    assert grouped.notna().sum().sum() > holey.where(ctx.investable().reindex_like(holey).fillna(False)).notna().sum().sum()
    with pytest.raises(FundamentalsError):
        T.apply(holey, [{"op": "fill", "method": "mean"}], ctx)
