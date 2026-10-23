"""The research context: panels as known on each date, forward returns that keep the losers, currency conversion, size and liquidity filters, and an honest account of survivorship."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamentals.errors import FundamentalsError
from src.fundamentals.research import HORIZONS, ResearchContext, UniverseSpec, horizon_days
from src.fundamentals.service import FundamentalsService


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("research"))
    report = svc.load_simulated(n_companies=36, start_year=2009, end_year=2017, seed=21)
    return svc, report


@pytest.fixture(scope="module")
def ctx(loaded):
    return ResearchContext(loaded[0].store, UniverseSpec())


def test_the_grid_is_the_last_trading_day_of_each_month_and_the_universe_is_everyone_with_prices(ctx, loaded):
    assert ctx.n == 36 and len(ctx.dates) > 90 and ctx.dates.is_monotonic_increasing
    assert all(d.month != (d + pd.offsets.BDay(1)).month for d in ctx.dates[:-1])                    # each date is the last business day of its month
    weekly = ResearchContext(loaded[0].store, UniverseSpec(), freq="W")
    quarterly = ResearchContext(loaded[0].store, UniverseSpec(), freq="Q")
    assert len(weekly.dates) > 3 * len(ctx.dates) and len(quarterly.dates) < len(ctx.dates) / 2
    with pytest.raises(FundamentalsError, match="freq must be"):
        ResearchContext(loaded[0].store, UniverseSpec(), freq="D")


def test_forward_returns_start_a_day_after_the_signal_and_include_the_delisting(ctx):
    fwd = ctx.forward("1m")
    assert horizon_days("1m") == 21 and horizon_days(7) == 7 and set(HORIZONS) >= {"1w", "1m", "3m", "6m", "12m"}
    sid = ctx.ids[0]
    t = 40
    pos = ctx.pos[t]
    adj = ctx.adj[sid]
    assert fwd.iloc[t][sid] == pytest.approx(adj.iloc[pos + 1 + 21] / adj.iloc[pos + 1] - 1.0)
    immediate = ResearchContext(ctx.store, UniverseSpec(), entry_lag=0).forward("1m")
    assert immediate.iloc[t][sid] == pytest.approx(adj.iloc[pos + 21] / adj.iloc[pos] - 1.0) and immediate.iloc[t][sid] != fwd.iloc[t][sid]
    assert fwd.iloc[-1].isna().all()                                                             # nothing is known about the month after the data end
    gone = [i for i in ctx.ids if not bool(ctx.info.loc[i, "active"])]
    assert gone, "the simulated world delists some companies"
    long = ctx.forward("3m")
    last_pos = ctx.adj.index.get_loc(ctx.adj[gone[0]].last_valid_index())
    outliving = [k for k, p in enumerate(ctx.pos) if p + 1 <= last_pos < p + 1 + 63 and not np.isnan(ctx.adj[gone[0]].iloc[p + 1])]
    assert len(outliving) >= 2                                                                    # the last two month-ends before the delisting have a three-month horizon that outlives the company
    k = outliving[0]
    assert np.isfinite(long.iloc[k][gone[0]])                                                     # carried to the last price, not dropped
    assert long.iloc[k][gone[0]] == pytest.approx(ctx.adj[gone[0]].iloc[last_pos] / ctx.adj[gone[0]].iloc[ctx.pos[k] + 1] - 1.0)


def test_backward_looking_panels_use_only_what_was_known_by_the_date(ctx):
    mom = ctx.past(252, 21)
    sid, t = ctx.ids[3], 50
    pos = ctx.pos[t]
    assert mom.iloc[t][sid] == pytest.approx(ctx.adj[sid].iloc[pos - 21] / ctx.adj[sid].iloc[pos - 252] - 1.0)
    assert mom.iloc[:8].isna().all().all()                                                       # a year of history is needed first
    vol, beta = ctx.volatility(63), ctx.beta(252)
    assert 0.05 < vol.stack().median() < 1.5 and 0.5 < beta.stack().median() < 1.6
    assert beta.iloc[:5].isna().all().all()                                                      # half a year of returns is needed first


def test_statement_figures_are_visible_only_from_the_first_date_on_or_after_their_availability(ctx):
    revenue = ctx.panel("revenue", "ttm")
    tables = ctx.tables.ttm
    inside = tables[tables["revenue"].notna() & ~tables["pit_estimated"] & (tables["available_date"] > ctx.dates[12]) & (tables["available_date"] < ctx.dates[-12])]
    row = inside.iloc[len(inside) // 2]
    sid = int(row["security_id"])
    series = revenue[sid]
    appears = series[series == row["revenue"]]
    assert not appears.empty and appears.index[0] >= row["available_date"]
    assert appears.index[0] - row["available_date"] < pd.Timedelta(days=45)                      # and no later than the next month end


def test_a_company_reporting_in_a_foreign_currency_is_converted_at_the_rate_of_the_date(ctx):
    before = ctx.info["currency"].copy()
    rep, trade = ctx.reporting_currency(), ctx.info["currency"]
    assert trade.equals(before)                                                                   # looking up the reporting currencies must not touch the trading currencies
    odd = [i for i in ctx.ids if rep[i] != trade[i]]
    assert odd, "the simulated world has a company that reports in another currency than it trades in"
    usd = ctx.panel("revenue")
    converted = ctx.to_trading(usd)
    sid = odd[0]
    t = usd[sid].last_valid_index()
    fx = ctx._fx_per_usd()
    expected = usd.loc[t, sid] * fx.loc[t, trade[sid]] / fx.loc[t, rep[sid]]
    assert converted.loc[t, sid] == pytest.approx(expected) and converted.loc[t, sid] != usd.loc[t, sid]
    same = [i for i in ctx.ids if rep[i] == trade[i]][0]
    assert np.allclose(converted[same].dropna(), usd[same].dropna())                              # nothing to convert
    assert ctx.to_trading(usd).shape == usd.shape


def test_market_cap_is_the_close_times_the_shares_known_on_the_date_and_enterprise_value_adds_net_debt(ctx, loaded):
    world = loaded[0].provider("simulated").world
    mcap, ev = ctx.market_cap(), ctx.enterprise_value()
    tickers = {i: str(ctx.info.loc[i, "ticker"]) for i in ctx.ids}
    col = {t: k for k, t in enumerate(world.ticker)}
    ratios = []
    for sid in ctx.ids[:12]:
        for t in (60, 80):
            truth = world.mcap[min(t, len(world.months) - 1), col[tickers[sid]]]
            got = mcap.iloc[t][sid]
            if np.isfinite(got) and np.isfinite(truth) and truth > 0:
                ratios.append(got / truth)
    assert len(ratios) > 10 and 0.7 < np.median(ratios) < 1.4                                     # shares are those of the last filing, the truth moves monthly
    both = (mcap.notna() & ev.notna()).to_numpy()
    assert both.sum() > 500 and (ev.to_numpy()[both] != mcap.to_numpy()[both]).mean() > 0.9
    usd = ctx.market_cap_usd()
    assert (usd.stack().dropna() > 0).all()


def test_enterprise_value_adds_debt_and_the_claims_ahead_of_the_shares_takes_off_cash_and_is_missing_where_nothing_is_known():
    """Market value + debt - cash + minority interest + preferred stock + redeemable equity (the last is in the data only where a filing has it). Cash falls back from 'cash and equivalents' to
    'cash and short-term investments'; a company with neither debt nor cash on file has no enterprise value."""
    index = pd.to_datetime(["2020-01-31", "2020-02-28"])

    def frame(first, second):
        return pd.DataFrame({1: first, 2: second}, index=index)
    nan = np.nan
    data = {"total_debt": frame([100.0, 100.0], [50.0, nan]), "cash_and_equivalents": frame([10.0, 10.0], [nan, nan]), "cash": frame([12.0, 12.0], [5.0, nan]),
            "minority_interest": frame([20.0, 20.0], [nan, nan]), "preferred_equity": frame([30.0, 30.0], [0.0, 0.0]), "temporary_equity": frame([40.0, 40.0], [nan, 7.0])}
    market = frame([1000.0, 1100.0], [500.0, 550.0])

    def context(columns):
        ctx = object.__new__(ResearchContext)
        ctx._cache = {}
        ctx.panel = lambda name, basis="ttm": data[name]
        ctx.has = lambda name, basis="ttm": name in columns
        ctx.market_cap = lambda: market
        ctx.to_trading = lambda amount: amount
        return ctx
    ev = context(set(data)).enterprise_value()
    assert ev.loc[index[0], 1] == pytest.approx(1000.0 + 100.0 - 10.0 + 20.0 + 30.0 + 40.0) and ev.loc[index[1], 1] == pytest.approx(1100.0 + 180.0)
    assert ev.loc[index[0], 2] == pytest.approx(500.0 + 50.0 - 5.0) and np.isnan(ev.loc[index[1], 2])
    without = context(set(data) - {"temporary_equity"}).enterprise_value()
    assert without.loc[index[0], 1] == pytest.approx(1000.0 + 100.0 - 10.0 + 20.0 + 30.0)                            # databases that never had the figure (OpenFilings, the simulated world) are as before


def test_the_investable_mask_applies_the_size_and_liquidity_limits(loaded):
    store = loaded[0].store
    base = ResearchContext(store, UniverseSpec())
    big = ResearchContext(store, UniverseSpec(min_market_cap_usd=float(base.market_cap_usd().stack().quantile(0.6))))
    assert base.investable().to_numpy().sum() > big.investable().to_numpy().sum() > 0
    cap = big.market_cap_usd()
    assert (cap.where(big.investable()).stack().dropna() >= big.spec.min_market_cap_usd).all()
    liquid = ResearchContext(store, UniverseSpec(min_dollar_volume_usd=1e18))
    assert liquid.investable().to_numpy().sum() == 0


def test_a_universe_is_cut_by_country_sector_and_development(loaded):
    store = loaded[0].store
    frame = store.securities()
    country = frame["country"].mode().iloc[0]
    only = ResearchContext(store, UniverseSpec(countries=(country,)))
    assert set(only.info["country"]) == {country} and 0 < only.n < 36
    assert list(only.info.index) == only.ids and "index" not in only.info.columns                              # the old row numbers of the securities table are not kept as a column
    sector = frame["sector"].mode().iloc[0]
    assert set(ResearchContext(store, UniverseSpec(sectors=(sector,))).info["sector"]) == {sector}
    assert set(ResearchContext(store, UniverseSpec(developed_only=True)).info["country"]) <= {"US", "GB", "JP", "SG", "AU", "DE"}
    with pytest.raises(FundamentalsError, match="no companies"):
        ResearchContext(store, UniverseSpec(countries=("ZZ",)))
    assert UniverseSpec.from_dict({"countries": ["US"], "min_market_cap_usd": 5}).countries == ("US",) and UniverseSpec().key() == UniverseSpec().key() != UniverseSpec(countries=("US",)).key()


def test_the_survivorship_report_does_not_claim_more_than_the_data_support(loaded):
    svc = loaded[0]
    keeping = ResearchContext(svc.store, UniverseSpec(include_inactive=True))
    s = keeping.survivorship()
    assert s["inactive"] > 0 and s["unbiased"] is False and "not a complete history" in s["verdict"]
    assert s["declared"] == "not declared" and s["companies"] == keeping.n and s["inactive_share"] == s["inactive"] / keeping.n
    dropping = ResearchContext(svc.store, UniverseSpec(include_inactive=False))
    d = dropping.survivorship()
    assert d["inactive"] == 0 and "survivorship bias" in d["verdict"] and d["unbiased"] is False and dropping.n < keeping.n
    svc.store.set_meta("universe_kind", "point_in_time")
    try:
        declared = ResearchContext(svc.store, UniverseSpec()).survivorship()
        assert declared["unbiased"] is True and declared["declared"] == "point_in_time" and "historical constituents" in declared["verdict"]
    finally:
        svc.store.set_meta("universe_kind", "")


def test_coverage_and_the_description_say_what_is_in_the_sample(ctx):
    cov = ctx.coverage()
    assert cov.loc["revenue", "coverage"] > 0.8 and cov.loc["total_assets", "coverage"] > 0.8
    d = ctx.describe()
    assert d["companies"] == 36 and d["freq"] == "M" and d["entry_lag_days"] == 1 and d["survivorship"]["inactive"] > 0 and 0.0 <= d["pit_estimated_share"] < 0.2


def test_the_horizons_are_the_documented_trading_days_and_the_settled_lag_is_the_longest_wait_for_a_window_to_end(ctx):
    assert HORIZONS == {"1d": 1, "1w": 5, "2w": 10, "1m": 21, "2m": 42, "3m": 63, "6m": 126, "9m": 189, "12m": 252}
    pos = list(ctx.pos)
    for horizon, h in (("1m", 21), ("3m", 63), ("12m", 252)):
        waits = []
        for i in range(len(pos)):
            end = pos[i] + ctx.entry_lag + h                                                                    # the last day of the window that starts the day after the date
            later = [j for j in range(i, len(pos)) if pos[j] >= end]
            if later:
                waits.append(later[0] - i)
        assert ctx.settled_lag(horizon) == max(max(waits), 1), horizon
    assert ctx.settled_lag("12m") > ctx.settled_lag("3m") >= 3 > 1


def test_a_volatility_needs_half_its_window_of_returns_and_never_fewer_than_twenty():
    returns = pd.DataFrame({"a": np.random.default_rng(3).normal(0.0, 0.01, 120)}, index=pd.bdate_range("2020-01-01", periods=120))

    class Stub:                                                                                                  # the method reads only these four things from its context
        _cache: dict
        pos = np.arange(120)
        dates = returns.index

        def __init__(self):
            self._cache = {}

        def daily_returns(self):
            return returns

    for window, needed in ((63, 31), (40, 20), (30, 20), (22, 20)):                                              # half the window, but twenty at the least
        vol = ResearchContext.volatility(Stub(), window)["a"]
        assert returns.index.get_loc(vol.first_valid_index()) == needed - 1, window
        assert vol.iloc[-1] == pytest.approx(returns["a"].iloc[-window:].std() * np.sqrt(252.0))                 # a sample deviation of the last window's returns, annualised


def test_the_coverage_the_volatility_and_the_traded_value_are_what_their_definitions_say(ctx):
    mask, panel = ctx.investable(), ctx.panel("revenue", "ttm")
    assert ctx.coverage(["revenue"]).loc["revenue", "coverage"] == pytest.approx(float(panel.where(mask).notna().to_numpy().sum() / mask.to_numpy().sum()))
    assert np.array_equal(mask.to_numpy(), ctx.price().notna().to_numpy())                                      # with no limit set, everyone with a price on the date is investable
    vol = ctx.daily_returns().rolling(63, min_periods=31).std() * np.sqrt(252.0)
    assert np.allclose(ctx.volatility(63).to_numpy(float), vol.iloc[ctx.pos].to_numpy(float), equal_nan=True)  # a quarter of trading days, at least half of them present, annualised
    traded = (ctx.close * ctx.volume).rolling(63, min_periods=31).mean().iloc[ctx.pos].set_axis(ctx.dates) / ctx.trading_per_usd()
    assert np.allclose(ctx.dollar_volume_usd().to_numpy(float), traded.to_numpy(float), equal_nan=True)


def test_a_database_without_prices_says_what_to_do(tmp_path):
    svc = FundamentalsService(tmp_path)
    from src.fundamentals.store import CompanyProfile
    svc.store.upsert_security(CompanyProfile("simulated", "AAA", "AAA", "A", exchange="NYSE"))
    with pytest.raises(FundamentalsError, match="no prices"):
        ResearchContext(svc.store, UniverseSpec())
