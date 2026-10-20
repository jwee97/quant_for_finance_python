"""The factor library: every definition is documented, runs on the data, treats a denominator that cannot divide as missing, and means what it says."""

from __future__ import annotations

import numpy as np
import pytest

from src.fundamentals import factors as F
from src.fundamentals import transforms as T
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("factors"))
    svc.load_simulated(n_companies=50, start_year=2009, end_year=2016, seed=8)
    return ResearchContext(svc.store, UniverseSpec())


def test_the_catalogue_is_complete_documented_and_consistent():
    cat = F.catalogue()
    names = [c["name"] for c in cat]
    assert len(names) == len(set(names)) and len(names) > 70
    assert {c["family"] for c in cat} == set(F.FAMILIES)
    assert all(c["text"] and c["label"] and c["direction"] in (1, -1) for c in cat)
    assert F.get("asset_growth").direction == -1 and F.get("fcf_yield").direction == 1 and F.get("accruals_ratio").direction == -1 and F.get("volatility_63").direction == -1
    with pytest.raises(FundamentalsError, match="no factor called"):
        F.get("nothing")
    assert F.names("value")[:3] == ["earnings_yield", "fcf_yield", "cfo_yield"] and set(F.names("price_momentum")) == {"mom_12_1", "mom_6_1", "mom_3_1", "reversal_1m", "high_52w"}


def test_every_factor_runs_on_the_simulated_world_and_most_have_data(ctx):
    ok = F.available(ctx)
    assert all(ok.values())
    covered = {}
    for name in F.FACTORS:
        panel = F.compute(ctx, name)
        assert panel.shape == (len(ctx.dates), ctx.n) and not np.isinf(panel.to_numpy(float)).any(), name
        covered[name] = F.coverage(ctx, name)
    thin = [n for n, c in covered.items() if c < 0.15]
    assert thin == ["rd_intensity"] or not thin, thin
    assert np.mean(list(covered.values())) > 0.6


def test_a_yield_is_the_trailing_amount_over_the_market_capitalisation_at_the_date(ctx):
    ey = F.compute(ctx, "earnings_yield")
    ni, mcap = ctx.to_trading(ctx.panel("net_income")), ctx.market_cap()
    t, sid = 60, ctx.ids[4]
    assert ey.iloc[t][sid] == pytest.approx(ni.iloc[t][sid] / mcap.iloc[t][sid]) or not np.isfinite(ey.iloc[t][sid])
    both = ey.notna() & ni.notna() & (mcap > 0)
    ratio = (ey / (ni / mcap)).to_numpy(float)[both.to_numpy()]
    assert both.to_numpy().sum() > 1000 and np.allclose(ratio[np.isfinite(ratio)], 1.0) and np.isfinite(ratio).mean() > 0.95
    fy = F.compute(ctx, "fcf_yield")
    assert np.isfinite(fy.to_numpy(float)).sum() > 1000


def test_a_denominator_that_is_not_positive_makes_the_ratio_missing(ctx):
    bp = F.compute(ctx, "book_to_price")
    equity = ctx.panel("shareholders_equity")
    assert bp[equity <= 0].isna().all().all()
    ev = ctx.enterprise_value()
    eb = F.compute(ctx, "ebit_to_ev")
    assert eb[ev <= 0].isna().all().all()
    de = F.compute(ctx, "debt_to_equity")
    assert de[equity <= 0].isna().all().all()


def test_price_factors_look_back_and_risk_factors_are_positive(ctx):
    mom = F.compute(ctx, "mom_12_1")
    assert mom.equals(ctx.past(252, 21)) and F.compute(ctx, "reversal_1m").equals(ctx.past(21, 0))
    high = F.compute(ctx, "high_52w")
    assert (high.stack().dropna() <= 1.0 + 1e-9).all() and (high.stack().dropna() > 0).all()
    assert (F.compute(ctx, "volatility_63").stack().dropna() > 0).all() and (F.compute(ctx, "idiosyncratic_volatility").stack().dropna() >= 0).all()
    size = F.compute(ctx, "log_market_cap")
    assert np.allclose(size.to_numpy(float)[ctx.market_cap_usd().to_numpy(float) > 0], np.log(ctx.market_cap_usd()).to_numpy(float)[ctx.market_cap_usd().to_numpy(float) > 0], equal_nan=True)


def test_a_composite_is_the_direction_signed_average_of_standardised_parts_and_a_missing_part_is_not_a_zero(ctx):
    def z(name):
        return T.apply(F.compute(ctx, name), [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}], ctx)

    parts = [("earnings_yield", 1.0), ("asset_growth", 1.0)]                                   # asset growth is bad when high: the composite flips it
    comp = F.composite(ctx, parts)
    t = 70
    both = z("earnings_yield").iloc[t].notna() & z("asset_growth").iloc[t].notna()
    expected = (z("earnings_yield").iloc[t] - z("asset_growth").iloc[t]) / 2.0
    assert both.sum() > 20 and np.allclose(comp.iloc[t][both], expected[both])
    # three-year growth needs three years of history, so early on a company has the yield but not the growth
    gap_parts = [("earnings_yield", 1.0), ("revenue_cagr_3y", 1.0)]
    strict, loose = F.composite(ctx, gap_parts), F.composite(ctx, gap_parts, min_parts=1)
    ey, cagr = z("earnings_yield"), z("revenue_cagr_3y")
    only_one = ey.notna() & cagr.isna()
    row = int(only_one.sum(axis=1).to_numpy().argmax())
    assert only_one.iloc[row].sum() > 5
    cells = only_one.iloc[row]
    assert strict.iloc[row][cells].isna().all() and np.allclose(loose.iloc[row][cells], ey.iloc[row][cells])       # the weights are renormalised over the parts it has: never a zero for the missing one
    assert "value_composite" in F.FACTORS and np.isfinite(F.compute(ctx, "value_composite").to_numpy(float)).sum() > 500


def test_a_factor_whose_data_are_missing_says_so(ctx, tmp_path):
    svc = FundamentalsService(tmp_path)
    svc.load_simulated(n_companies=12, start_year=2012, end_year=2014, seed=3)
    small = ResearchContext(svc.store, UniverseSpec())
    small.tables.ttm.drop(columns=["fcf"], inplace=True)
    assert F.available(small)["fcf_yield"] is False
    with pytest.raises(FundamentalsError, match="needs 'fcf'"):
        F.compute(small, "fcf_yield")
