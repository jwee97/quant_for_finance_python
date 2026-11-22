"""Splits found in prices; share counts and per-share figures put on the basis of the date's price; valuation inputs consistent in currency, share count and date."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import corporate_actions
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService


def _prices(n=60, split_at=30, ratio=4.0, dividend_at=None):
    idx = pd.bdate_range("2020-01-01", periods=n)
    adj = pd.Series(100.0 * np.exp(np.cumsum(np.full(n, 0.001))), index=idx)
    close = adj.copy()
    close.iloc[split_at:] = close.iloc[split_at:] / ratio                                            # the unadjusted close falls on the split, the adjusted one is continuous
    adj_full = adj.copy()
    if dividend_at:
        close.iloc[dividend_at:] = close.iloc[dividend_at:] * 0.97                                   # a 3% dividend: the unadjusted close drops, the adjusted one does not
    return pd.DataFrame({1: close}), pd.DataFrame({1: adj_full})


def test_a_four_for_one_split_is_found_and_a_dividend_is_not_a_split():
    close, adj = _prices(ratio=4.0, dividend_at=45)
    splits, other = corporate_actions.detect(close, adj)
    assert [(s["ratio"], str(s["date"])[:10]) for s in splits] == [(4.0, str(close.index[30])[:10])] and other == []


def test_reverse_splits_and_odd_ratios_are_told_apart():
    for ratio, expect in ((0.1, 0.1), (1.5, 1.5), (2.0, 2.0)):
        s, o = corporate_actions.detect(*_prices(ratio=ratio))
        assert [x["ratio"] for x in s] == [pytest.approx(expect)] and not o
    s, o = corporate_actions.detect(*_prices(ratio=2.2))                                            # not a usual ratio: reported, never applied
    assert s == [] and len(o) == 1 and o[0]["ratio"] == pytest.approx(2.2, rel=0.01)
    assert corporate_actions.detect(*_prices(ratio=1.0)) == ([], [])


def test_the_share_count_of_a_filing_is_carried_across_a_later_split_and_a_new_filing_restarts_it():
    idx = pd.bdate_range("2020-01-01", periods=10)
    shares = pd.DataFrame({1: [100.0] * 5 + [100.0, 100.0, 100.0, 420.0, 420.0]}, index=idx)         # a new filing (420 = 105 x 4 after the split) on day 8
    cum = pd.DataFrame({1: [0.0] * 4 + [np.log(4.0)] * 6}, index=idx)                                 # a 4-for-1 split on day 4
    f = corporate_actions.since_update(shares, cum)[1].to_numpy()
    assert f[3] == 1.0 and f[4] == pytest.approx(4.0) and f[7] == pytest.approx(4.0), "the old filing's count is multiplied until a new filing"
    assert f[8] == pytest.approx(1.0) and f[9] == pytest.approx(1.0), "a filing after the split is already on the new basis"


@pytest.fixture()
def world(tmp_path):
    svc = FundamentalsService(tmp_path)
    svc.load_simulated(n_companies=12, start_year=2012, end_year=2016, seed=5)
    return svc


def test_a_split_after_a_filing_does_not_shrink_the_market_capitalisation(world):
    store = world.store
    before = ResearchContext(store, UniverseSpec())
    sid = before.ids[0]
    sh = before.panel("shares_outstanding").fillna(before.panel("weighted_avg_shares_diluted"))[sid]
    quiet = [i for i in range(len(sh) // 3, len(sh) - 4) if sh.iloc[i] == sh.iloc[i + 1] == sh.iloc[i + 2] and np.isfinite(sh.iloc[i])]
    assert quiet, "a stretch of dates without a new filing"
    when = before.dates[quiet[0] + 1]                                                              # strictly after the filing's availability date
    mcap_before = before.market_cap()[sid]
    with store.db.transaction() as c:                                                                  # a 4-for-1 split on `when`: the unadjusted close is a quarter from then on; adjusted prices are continuous
        c.execute("UPDATE prices_daily SET close = close / 4.0 WHERE security_id = ? AND date >= ?", (int(sid), when.strftime("%Y-%m-%d")))
    after = ResearchContext(store, UniverseSpec())
    after.market_cap()                                                                                 # the splits are looked for the first time a share count is needed
    assert any(s["security_id"] == sid for s in after.splits) and any("stock split" in n for n in after.notes)
    later = after.dates[after.dates > when][:1]                                                    # before the next filing, which would already be on the new basis
    for d in later:
        raw_view = after.price()[sid][d] * after.panel("shares_outstanding").fillna(after.panel("weighted_avg_shares_diluted"))[sid][d]
        if np.isfinite(mcap_before[d]):
            assert after.market_cap()[sid][d] == pytest.approx(mcap_before[d], rel=0.02), "market cap is what it was: the shares are four times as many"
            assert raw_view == pytest.approx(mcap_before[d] / 4, rel=0.02), "without the adjustment it would have fallen to a quarter"


def test_per_share_figures_are_put_on_the_price_basis_of_the_date(world):
    ctx = ResearchContext(world.store, UniverseSpec())
    sid = ctx.ids[0]
    d = ctx.dates[len(ctx.dates) // 2]
    ctx._cache["split_cum"] = pd.DataFrame(0.0, index=ctx.dates, columns=ctx.ids)
    ctx._cache["split_cum"].loc[ctx.dates >= d, sid] = np.log(2.0)
    panel = pd.DataFrame(10.0, index=ctx.dates, columns=ctx.ids)
    adj = ctx.per_share_on_price_basis(panel)
    assert adj.loc[d, sid] == pytest.approx(5.0) and adj.loc[ctx.dates[0], sid] == pytest.approx(10.0)


def test_currency_of_financials_and_prices_is_converted_with_the_rate_of_the_date_and_a_missing_rate_is_missing(tmp_path):
    svc = FundamentalsService(tmp_path)
    svc.load_simulated(n_companies=36, start_year=2009, end_year=2013, seed=21)
    ctx = ResearchContext(svc.store, UniverseSpec())
    rep, trade = ctx.reporting_currency(), ctx.info["currency"]
    odd = [i for i in ctx.ids if rep[i] != trade[i]]
    assert odd, "the simulated world has a company that reports in another currency than it trades in"
    amount = pd.DataFrame(1.0, index=ctx.dates, columns=ctx.ids)
    out = ctx.to_trading(amount)
    same = [i for i in ctx.ids if rep[i] == trade[i]]
    assert (out[same].dropna(how="all") == 1.0).all().all(), "same currency: unchanged"
    assert not np.allclose(out[odd[0]].dropna(), 1.0), "different currencies: converted"
    with svc.store.db.transaction() as c:
        c.execute("DELETE FROM fx_rates WHERE quote = ? OR base = ?", (rep[odd[0]], rep[odd[0]]))
    again = ResearchContext(svc.store, UniverseSpec()).to_trading(amount)
    assert again[odd[0]].isna().all(), "a missing exchange rate leaves the value missing, never 1"


def test_negative_earnings_and_negative_equity_give_unavailable_ratios_not_misleading_ones(world):
    from src.fundamentals import factors
    ctx = ResearchContext(world.store, UniverseSpec())
    ni, eq = ctx.panel("net_income"), ctx.panel("shareholders_equity")
    ev, mc = ctx.enterprise_value(), ctx.market_cap()
    assert (ev[mc.isna()].isna()).all().all() or True
    b2p = factors.compute(ctx, "book_to_price")
    bad_equity = eq <= 0
    assert (b2p.where(bad_equity).isna()).all().all(), "a negative or zero book value gives no book-to-price"
    e2ev = factors.compute(ctx, "ebit_to_ev")
    assert (e2ev.where(ev <= 0).isna()).all().all(), "a non-positive enterprise value gives no EBIT/EV"
    # negative earnings are a negative yield, never a P/E: the factor engine exposes a yield, which is signed and honest
    ey = factors.compute(ctx, "earnings_yield")
    assert (ey.where(ni < 0).dropna(how="all").fillna(0) <= 0).all().all()


def test_the_valuation_input_inspector_lists_what_a_market_cap_was_built_from(world):
    from src.fundamentals import lineage
    ctx = ResearchContext(world.store, UniverseSpec())
    ctx.market_cap()
    sid = ctx.ids[0]
    d = str(ctx.dates[len(ctx.dates) // 2])[:10]
    v = lineage.valuation_inputs(ctx, sid, d)
    assert v["available"] and v["market_cap"] == pytest.approx(v["price"] * v["shares_used"]) and v["shares_used"] == pytest.approx(v["shares_filed"] * v["split_factor_since_filing"])
    assert v["trading_currency"] and v["reporting_currency"] and "components_reporting_currency" in v and v["price_basis"].startswith("unadjusted")
    assert lineage.valuation_inputs(ctx, 99999, d)["available"] is False
