"""BACKTEST THIS ALPHA, against known truth: a book is what its rules say it is (the names, the weights, the limits, the cash), it is decided from what was known on each date, the platform's engine
earns exactly what the factor tests say the same holdings earn, costs and borrow are charged, and the report does not flatter a signal that has nothing in it."""

from __future__ import annotations

import json
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import bridge as B
from src.fundamentals import expr as X
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.evaluate import EvalSpec, build_signal, holding_returns
from src.fundamentals.library import AlphaLibrary
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import FundamentalsStore


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("bridge"))
    svc.load_simulated(n_companies=100, start_year=2009, end_year=2018, seed=13, planted={"delta_roic": 0.011, "fcf_yield": 0.0068, "asset_growth": 0.0045})
    with svc.store.db.transaction() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    ctx = ResearchContext(svc.store, UniverseSpec())
    signal, info = B.resolve(ctx, {"parts": [{"kind": "factor", "name": "fcf_yield"}, {"kind": "factor", "name": "d_roic"}]})
    return svc, ctx, signal, info


def noise(ctx: ResearchContext, seed: int) -> pd.DataFrame:
    return pd.DataFrame(np.random.default_rng(seed).normal(size=(len(ctx.dates), ctx.n)), index=ctx.dates, columns=ctx.ids)


def spec(**kw) -> B.PortfolioSpec:
    return B.PortfolioSpec(**{"side": "long_only", "n": 20, **kw})


# ------------------------------------------------------------------------------------------------------------------------ the portfolio rules
def test_the_portfolio_settings_are_checked_and_whole_numbers_stay_whole():
    ok = B.PortfolioSpec.from_dict({"n": 30.0, "min_names": 4.0, "max_sector": 0.3})
    assert ok.n == 30 and isinstance(ok.n, int) and isinstance(ok.min_names, int) and ok.describe()["max_sector"] == 0.3
    for bad, why in (({"side": "both"}, "side must be"), ({"selection": "best"}, "selection must be"), ({"weighting": "kelly"}, "weighting must be"), ({"n": 0}, "whole number"),
                     ({"n": 2.5}, "whole number"), ({"percentile": 0}, "percentile must be above 0"), ({"side": "long_short", "percentile": 60}, "at most 50"), ({"max_position": 1.5}, "max_position"),
                     ({"max_sector": 0}, "max_sector"), ({"max_beta": -1}, "max_beta"), ({"cost_bps": -5}, "cost_bps"), ({"capital_usd": 0}, "capital_usd"), ({"max_turnover": "a lot"}, "must be a number"),
                     ({"leg_exposure": 5}, "leg_exposure")):
        with pytest.raises(FundamentalsError, match=why):
            B.PortfolioSpec.from_dict(bad)
    with pytest.raises(FundamentalsError, match="unknown portfolio setting"):
        B.PortfolioSpec.from_dict({"leverage": 3})
    assert B.PortfolioSpec(n=3, min_names=10).min_leg == 3 and B.PortfolioSpec(selection="percentile", min_names=7).min_leg == 7


def test_a_signal_is_described_from_factors_technicals_formulas_alphas_rules_and_models(world):
    svc, ctx, signal, info = world
    own = build_signal(ctx, EvalSpec(factor="fcf_yield"))
    flipped, _ = B.resolve(ctx, {"kind": "factor", "name": "fcf_yield", "direction": -1})
    straight, one = B.resolve(ctx, {"kind": "factor", "name": "fcf_yield"})
    pd.testing.assert_frame_equal(straight.dropna(how="all"), own.reindex_like(straight).dropna(how="all"))
    assert np.allclose(flipped.fillna(0.0), -straight.fillna(0.0)) and one["label"] == "fcf_yield" and one["n_trials"] == 1
    with pytest.raises(FundamentalsError, match="choosing the sign from the data"):
        B.resolve(ctx, {"kind": "factor", "name": "fcf_yield", "direction": "auto"})
    assert "mom_12_1" in B.technical_names() and "fcf_yield" not in B.technical_names()
    mom, _ = B.resolve(ctx, {"kind": "technical", "name": "mom_12_1"})
    assert mom.notna().any().any()
    with pytest.raises(FundamentalsError, match="technical signal is one of"):
        B.resolve(ctx, {"kind": "technical", "name": "fcf_yield"})
    both, both_info = B.resolve(ctx, [{"kind": "factor", "name": "fcf_yield"}, {"kind": "factor", "name": "d_roic", "weight": 2.0}])
    assert both_info["kind"] == "composite" and [p["weight"] for p in both_info["parts"]] == [1.0, 2.0] and "2 x d_roic" in both_info["label"]
    d_roic, _ = B.resolve(ctx, {"kind": "factor", "name": "d_roic"})
    top_corr = lambda a, b: float(np.nanmean([a.iloc[i].corr(b.iloc[i], method="spearman") for i in range(30, len(a), 6)]))     # noqa: E731
    even, _ = B.resolve(ctx, [{"kind": "factor", "name": "fcf_yield"}, {"kind": "factor", "name": "d_roic"}])
    assert top_corr(both, d_roic) > top_corr(even, d_roic) > 0.4 and top_corr(both, straight) < top_corr(even, straight)             # a larger weight moves the composite towards that part
    assert top_corr(both, straight) > 0.0 and top_corr(both, d_roic) > top_corr(both, straight)
    negative, _ = B.resolve(ctx, [{"kind": "factor", "name": "fcf_yield", "weight": -1.0}])
    assert np.allclose(negative.fillna(0.0), -straight.fillna(0.0))
    with pytest.raises(FundamentalsError, match="other than zero"):
        B.resolve(ctx, [{"kind": "factor", "name": "fcf_yield", "weight": 0}, {"kind": "factor", "name": "d_roic", "weight": 0}])
    formula, finfo = B.resolve(ctx, {"kind": "expression", "expression": X.parse("div(fcf,market_cap)")})
    assert finfo["label"] == "div(fcf,market_cap)" and formula.notna().any().any()
    rule, rinfo = B.resolve(ctx, {"kind": "combination", "method": "icir", "of": [{"kind": "factor", "name": "fcf_yield"}, {"kind": "factor", "name": "d_roic"}, {"kind": "technical", "name": "mom_12_1"}]})
    assert rinfo["label"] == "icir combination of 3" and rule.shape == straight.shape
    model, minfo = B.resolve(ctx, {"kind": "ml", "model": "ridge", "of": [{"kind": "factor", "name": "fcf_yield"}, {"kind": "factor", "name": "d_roic"}, {"kind": "technical", "name": "mom_12_1"}]})
    first = pd.Timestamp(minfo["ml"]["first_prediction"])
    assert model.loc[model.index < first].isna().all().all() and model.loc[first:].notna().any().any() and minfo["ml"]["model"] == "ridge"
    for bad, why in (({"kind": "magic"}, "kind must be one of"), ({"kind": "alpha", "alpha_id": "x"}, "none was given"), ({"kind": "expression"}, "needs an 'expression'"),
                     ({"kind": "ml", "of": [{"kind": "factor", "name": "roic"}]}, "between 2 and"), ("fcf_yield", "dictionary"), ({"parts": []}, "between 1 and")):
        with pytest.raises(FundamentalsError, match=why):
            B.resolve(ctx, bad)
    deep = {"kind": "factor", "name": "roic"}
    for _ in range(5):
        deep = {"parts": [deep]}
    with pytest.raises(FundamentalsError, match="levels deep"):
        B.resolve(ctx, deep)


def test_a_library_alpha_brings_its_status_its_trial_count_and_its_dates(world):
    svc, ctx, _, _ = world
    lib = AlphaLibrary(FundamentalsStore(_copy(svc, ctx, "alpha_lib.db")))
    blocks = {"train": ["2010-02-26", "2014-06-30"], "validation": ["2014-08-29", "2016-03-31"], "oos": ["2016-05-31", "2018-12-31"]}
    alpha_id = lib.add_candidate(X.parse("div(fcf,market_cap)"), {"universe": {}, "freq": "M", "horizon": "1m", "steps": [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}], "direction": 1,
                                                                 "blocks": blocks}, {"ic": 0.05, "n_tested": 240})
    signal, info = B.resolve(ctx, {"kind": "alpha", "alpha_id": alpha_id}, lib)
    assert info["kind"] == "alpha" and info["n_trials"] == 240 and info["alphas"][0]["status"] == "Candidate" and "not yet validated" in info["warnings"][0]
    assert B.alpha_blocks(lib, alpha_id) == {k: tuple(v) for k, v in blocks.items()} and signal.notna().any().any()
    result = B.tear(ctx, {"kind": "alpha", "alpha_id": alpha_id}, {"side": "long_only", "n": 20}, lib)
    assert set(result["blocks"]) == {"train", "validation", "oos"} and result["deflated_sharpe"]["n_trials"] == 240
    assert sum(b["n_obs"] for b in result["blocks"].values()) <= result["summary"]["n_obs"] and any("not yet validated" in w for w in result["warnings"])
    one_of_two = B.tear(ctx, [{"kind": "alpha", "alpha_id": alpha_id}, {"kind": "factor", "name": "d_roic"}], {"side": "long_only", "n": 20}, lib)
    assert one_of_two["blocks"] == {}                                               # the dates belong to the alpha, not to a blend that has something else in it
    single = B.tear(ctx, [{"kind": "alpha", "alpha_id": alpha_id}], {"side": "long_only", "n": 20}, lib)
    assert set(single["blocks"]) == {"train", "validation", "oos"}


def _copy(svc, ctx, name: str):
    target = svc.store.db.path.parent / name
    shutil.copy(svc.store.db.path, target)
    return target


# ------------------------------------------------------------------------------------------------------------------------ the book
def test_the_book_holds_the_top_names_by_the_signal_among_those_that_may_be_bought(world):
    _, ctx, signal, _ = world
    s = spec(n=15)
    book = B.build_book(ctx, signal, s)
    eligible = B.eligibility(ctx, s)
    w = book.weights
    assert list(w.index) == list(ctx.dates) and np.allclose(w.sum(axis=1)[w.sum(axis=1) > 0], 1.0)
    for date in w.index[[20, 60, 100]]:
        ok = eligible.loc[date] & signal.loc[date].notna()
        wanted = set(signal.loc[date][ok].sort_values(ascending=False).index[:15])
        held = set(w.columns[w.loc[date] > 0])
        assert held == wanted and np.allclose(w.loc[date][list(held)], 1.0 / 15)
    assert (w.where(~eligible, 0.0) == 0.0).all().all()                                       # never a name that could not be bought that day
    assert (book.diagnostics["long_names"].iloc[30:] == 15).all() and (book.diagnostics["short_names"] == 0).all()


def test_a_long_short_book_has_two_separate_legs_of_equal_size(world):
    _, ctx, signal, _ = world
    book = B.build_book(ctx, signal, spec(side="long_short", n=12, leg_exposure=1.0))
    w = book.weights
    live = w.abs().sum(axis=1) > 0
    assert np.allclose(w[w > 0].sum(axis=1)[live], 1.0) and np.allclose(w[w < 0].sum(axis=1)[live], -1.0) and np.allclose(w.sum(axis=1)[live], 0.0)
    date = w.index[80]
    longs, shorts = w.columns[w.loc[date] > 0], w.columns[w.loc[date] < 0]
    assert len(longs) == len(shorts) == 12 and not set(longs) & set(shorts) and signal.loc[date, longs].min() > signal.loc[date, shorts].max()
    half = B.build_book(ctx, signal, spec(side="long_short", n=12, leg_exposure=0.5)).weights
    assert np.allclose(half, w * 0.5)


def test_selecting_by_percentile_takes_that_share_of_the_eligible_names(world):
    _, ctx, signal, _ = world
    s = spec(selection="percentile", percentile=10.0)
    book = B.build_book(ctx, signal, s)
    for date in book.weights.index[[40, 90]]:
        m = int((B.eligibility(ctx, s).loc[date] & signal.loc[date].notna()).sum())
        assert int((book.weights.loc[date] > 0).sum()) == int(np.ceil(m * 0.10))
    two = B.build_book(ctx, signal, spec(side="long_short", selection="percentile", percentile=50.0)).weights
    date = two.index[80]
    assert (two.loc[date] > 0).sum() + (two.loc[date] < 0).sum() >= 2 and not ((two.loc[date] > 0) & (two.loc[date] < 0)).any()


def test_signal_weights_give_more_to_the_stronger_and_inverse_volatility_gives_less_to_the_wilder(world):
    _, ctx, signal, _ = world
    by_signal = B.build_book(ctx, signal, spec(weighting="signal", max_position=1.0)).weights
    date = by_signal.index[80]
    held = by_signal.loc[date][by_signal.loc[date] > 0].sort_index()
    order = signal.loc[date, held.index].sort_values(ascending=False).index
    assert np.all(np.diff(held[order].to_numpy()) <= 1e-12) and held.sum() == pytest.approx(1.0) and held.max() > 1.5 * held.min() and held.min() > 0
    inverse = B.build_book(ctx, signal, spec(weighting="inverse_vol", max_position=1.0)).weights
    vol = ctx.volatility(63)
    row = inverse.loc[date][inverse.loc[date] > 0]
    floor = np.nanpercentile(vol.loc[date, row.index], 5)
    unfloored = row.index[vol.loc[date, row.index] > floor]
    product = (row[unfloored] * vol.loc[date, unfloored])
    assert len(unfloored) > 10 and np.allclose(product, product.iloc[0]) and row.sum() == pytest.approx(1.0)


def test_with_too_few_eligible_names_the_date_is_cash_and_the_report_says_why(world):
    _, ctx, signal, _ = world
    thin = signal.copy()
    thin.iloc[50] = np.nan
    thin.iloc[50, :3] = [1.0, 2.0, 3.0]
    book = B.build_book(ctx, thin, spec(n=10, min_names=5))
    assert (book.weights.iloc[50] == 0).all() and "held in cash" in book.diagnostics.iloc[50]["reason"] and book.weights.iloc[51].sum() == pytest.approx(1.0)
    empty = B.build_book(ctx, signal * np.nan, spec())
    assert (empty.weights == 0).all().all() and (empty.diagnostics["reason"] != "").all()
    with pytest.raises(FundamentalsError, match="never held anything: only 0 eligible"):
        B.backtest(ctx, signal * np.nan, spec())


def test_the_projection_finds_the_nearest_book_that_respects_the_limits():
    project = B._project
    up = np.array([1.0, 1.0, 1.0, 1.0])
    w, t, how = project(np.array([0.5, 0.3, 0.1, 0.1]), up, np.full(4, 0.4), [], [], 1.0, False)
    assert how == "projected" and t == 1.0 and np.allclose(w, [0.4, 1 / 3, 2 / 15, 2 / 15], atol=1e-6)
    w, t, _ = project(np.full(4, 0.25), up, np.full(4, 0.2), [], [], 1.0, False)                # four names at 20% cannot make a whole account: 80% invested
    assert t == pytest.approx(0.8, abs=1e-5) and w.sum() == pytest.approx(0.8, abs=1e-5) and np.allclose(w, 0.2, atol=1e-5)
    w, t, _ = project(np.array([0.5, 0.5]), np.ones(2), np.ones(2), [np.array([1.5, 0.5])], [0.9], 1.0, False)       # net beta at most 0.9
    assert np.allclose(w, [0.4, 0.6], atol=1e-6) and t == 1.0
    signs = np.array([1.0, 1.0, -1.0, -1.0, -1.0])                                              # two names can carry a 0.4-capped long leg only to 0.8: the short leg is scaled with it
    w, t, _ = project(np.array([0.5, 0.5, -1 / 3, -1 / 3, -1 / 3]), signs, np.full(5, 0.4), [], [], 1.0, True)
    assert t == pytest.approx(0.8, abs=1e-5) and w[:2].sum() == pytest.approx(0.8, abs=1e-5) and w[2:].sum() == pytest.approx(-0.8, abs=1e-5) and w.sum() == pytest.approx(0.0, abs=1e-6)
    w, t, how = project(np.array([0.5, 0.5]), np.ones(2), np.ones(2), [np.array([1.0, 1.0])], [0.0], 1.0, False)    # a group that may hold nothing: nothing can be invested
    assert t == 0.0 and not w.any() and "no room" in how


def test_no_limit_is_ever_broken_and_what_cannot_be_invested_stays_in_cash(world):
    _, ctx, signal, _ = world
    book = B.build_book(ctx, signal, spec(n=20, max_position=0.04))                             # 20 x 4% is only 80% of the account
    held = book.weights.sum(axis=1)
    assert held.iloc[30:].max() <= 0.8 + 1e-6 and held.iloc[30:].min() >= 0.8 - 1e-4 and book.weights.max().max() <= 0.04 + 1e-9
    assert (book.diagnostics["fraction"].iloc[30:] < 1.0).all() and "rest in cash" in book.diagnostics["reason"].iloc[60]
    sector = ctx.group("sector")
    book = B.build_book(ctx, signal, spec(n=20, max_sector=0.2, max_position=0.1))
    net = book.weights.T.groupby(sector.reindex(book.weights.columns).to_numpy()).sum().T
    assert net.abs().max().max() <= 0.2 + 1e-6 and np.allclose(book.weights.sum(axis=1).iloc[30:], 1.0, atol=1e-6) and book.diagnostics["projected"].sum() > 0
    country = ctx.group("country")
    book = B.build_book(ctx, signal, spec(side="long_short", n=20, max_country=0.1, max_industry=0.12))
    for labels, limit in ((country, 0.1), (ctx.group("industry"), 0.12)):
        net = book.weights.T.groupby(labels.reindex(book.weights.columns).to_numpy()).sum().T
        assert net.abs().max().max() <= limit + 1e-6
    assert book.weights.abs().max().max() <= 0.1 + 1e-9


def test_the_net_beta_stays_in_its_band_and_a_position_never_outgrows_the_name_liquidity(world):
    _, ctx, signal, _ = world
    book = B.build_book(ctx, signal, spec(side="long_short", n=20, max_beta=0.05))
    open_book = B.build_book(ctx, signal, spec(side="long_short", n=20))
    beta = ctx.beta(252).reindex_like(book.weights)
    clear = (beta.index >= beta.index[30]) & ~((book.weights != 0) & beta.isna()).any(axis=1) & ~((open_book.weights != 0) & beta.isna()).any(axis=1)      # every held beta is measured
    assert clear.sum() > 20
    assert (book.weights * beta.fillna(0.0)).sum(axis=1)[clear].abs().max() <= 0.05 + 1e-5 and book.diagnostics["beta"].abs().max() <= 0.05 + 1e-5
    assert (open_book.weights * beta.fillna(0.0)).sum(axis=1)[clear].abs().max() > 0.05                                   # without the band the book's beta wanders outside it
    s = spec(n=20, max_adv_share=0.1, capital_usd=20_000_000.0)
    limited = B.build_book(ctx, signal, s)
    adv = ctx.dollar_volume_usd().reindex_like(limited.weights)
    assert (limited.weights * s.capital_usd <= s.max_adv_share * adv.fillna(0.0) + 1e-6).all().all() and (limited.weights.sum(axis=1).iloc[30:] <= 1.0 + 1e-9).all()
    assert limited.weights.sum(axis=1).iloc[30:].min() < 0.99 and limited.diagnostics["projected"].sum() > 10                  # the small names could not carry a full weight


def _shares_traded(ctx, book, target) -> list:
    """For each date after the first: the most bought or sold, as a share of the book wanted, measured against the last book as prices have moved it (worked out here, apart from the code under test)."""
    carried = ctx.adj.ffill()
    out = []
    for i in range(1, len(ctx.dates)):
        a, b = ctx.pos[i - 1] + ctx.entry_lag, ctx.pos[i]
        grown = book.iloc[i - 1] * (carried.iloc[b] / carried.iloc[a]).fillna(1.0)
        gross = book.iloc[i - 1].abs().sum()
        drifted = grown * (gross / grown.abs().sum()) if grown.abs().sum() > 0 else grown
        change = book.iloc[i] - drifted
        out.append(max(change.clip(lower=0).sum(), (-change).clip(lower=0).sum()) / max(target.iloc[i].abs().sum(), drifted.abs().sum()))
    return out


def test_a_turnover_limit_trades_part_of_the_way_and_builds_the_first_book_over_several_dates(world):
    _, ctx, signal, _ = world
    free = B.build_book(ctx, signal, spec(n=20))
    held = B.build_book(ctx, signal, spec(n=20, max_turnover=0.15))
    assert held.weights.sum(axis=1).iloc[0] == pytest.approx(0.15, abs=1e-6) and held.weights.sum(axis=1).iloc[:6].is_monotonic_increasing and held.weights.sum(axis=1).iloc[8] > 0.9
    limited, unlimited = _shares_traded(ctx, held.weights, free.weights), _shares_traded(ctx, free.weights, free.weights)
    assert max(limited) <= 0.15 + 0.02 and max(unlimited) > 0.25 and np.mean(limited[10:]) < np.mean(unlimited[10:])        # without the limit the book is turned over by more than the limit allows
    assert held.diagnostics["limited_turnover"].astype(float).sum() > 5 and held.diagnostics["wanted_turnover"].iloc[5:].mean() > 0.0
    assert B.backtest(ctx, signal, spec(n=20, max_turnover=0.15))["costs"]["annual_turnover"] < B.backtest(ctx, signal, spec(n=20))["costs"]["annual_turnover"]


def test_a_date_is_decided_from_what_was_known_on_it_not_from_prices_that_came_later(world, tmp_path):
    svc, ctx, signal, _ = world
    cut = 70
    cutoff = str(ctx.dates[cut].date())
    path = _copy(svc, ctx, "later_prices.db")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE prices_daily SET adj_close = adj_close * (1.0 + ((security_id * 7 + CAST(strftime('%j', date) AS INTEGER) * 13) % 50) / 100.0), "
                     "close = close * (1.0 + ((security_id * 7 + CAST(strftime('%j', date) AS INTEGER) * 13) % 50) / 100.0), volume = volume * 3 WHERE date > ?", (cutoff,))
    other = ResearchContext(FundamentalsStore(path), UniverseSpec())
    assert not np.allclose(other.adj.iloc[-1].to_numpy(), ctx.adj.iloc[-1].to_numpy())
    kw = dict(side="long_short", n=15, weighting="inverse_vol", max_sector=0.3, max_beta=0.2, max_turnover=0.3, max_adv_share=0.05, min_market_cap_usd=1e6, min_dollar_volume_usd=1e3)
    a = B.build_book(ctx, signal, spec(**kw)).weights
    b = B.build_book(other, signal, spec(**kw)).weights
    pd.testing.assert_frame_equal(a.iloc[:cut + 1], b.iloc[:cut + 1])
    assert not np.allclose(a.iloc[cut + 6:].to_numpy(), b.iloc[cut + 6:].to_numpy())


# ------------------------------------------------------------------------------------------------------------------------ the engine
def test_the_engine_earns_exactly_what_the_factor_tests_say_the_same_holdings_earn(world):
    _, ctx, signal, info = world
    s = spec(n=20, max_position=0.5, cost_bps=0.0, borrow_bps=0.0)
    result = B.backtest(ctx, signal, s, "check", info)
    book = B.build_book(ctx, signal, s)
    expected = (book.weights * holding_returns(ctx).reindex_like(book.weights).fillna(0.0)).sum(axis=1)
    equity = pd.Series({pd.Timestamp(d): v for d, v in result["series"]["equity"]})
    entry = ctx.pos + ctx.entry_lag
    checked = 0
    for p in range(1, len(ctx.dates) - 2):
        a, b = ctx.adj.index[entry[p]], ctx.adj.index[entry[p + 1]]
        if a in equity.index and b in equity.index:
            assert equity[b] / equity[a] - 1.0 == pytest.approx(expected.iloc[p], abs=5e-6)         # the same holdings, entered the same day, earn the same
            checked += 1
    assert checked > 90 and result["first_trade"] > str(ctx.dates[0].date())


def test_costs_and_borrow_are_charged_including_the_cost_of_building_the_first_book(world):
    _, ctx, signal, info = world
    free = B.backtest(ctx, signal, spec(side="long_short", n=20, cost_bps=0.0, borrow_bps=0.0), "free", info)
    paid = B.backtest(ctx, signal, spec(side="long_short", n=20, cost_bps=25.0, borrow_bps=0.0), "paid", info)
    borrowing = B.backtest(ctx, signal, spec(side="long_short", n=20, cost_bps=0.0, borrow_bps=300.0), "borrow", info)
    assert paid["summary"]["cagr"] < free["summary"]["cagr"] - 0.01 and borrowing["summary"]["cagr"] < free["summary"]["cagr"] - 0.02
    assert paid["costs"]["trading_bps_a_year"] == pytest.approx(25.0 * paid["costs"]["annual_turnover"], rel=1e-6)
    assert borrowing["costs"]["borrow_bps_a_year"] == pytest.approx(300.0 * borrowing["exposure"]["gross_average"] / 2.0, rel=0.08) and free["costs"]["borrow_bps_a_year"] == 0.0
    sens = {row["cost_bps"]: row["net_sharpe"] for row in paid["costs"]["sensitivity"]}
    assert sens[0.0] > sens[25.0] > sens[100.0] and paid["costs"]["breakeven_bps"] > 25.0
    only = B.backtest(ctx, signal, spec(side="long_only", n=20, borrow_bps=500.0), "long", info)
    assert only["costs"]["borrow_bps_a_year"] == 0.0
    # building the first book is a purchase: with a 100 bp cost and a single rebalance in the window, the whole book is paid for on the first day of returns
    day = str(ctx.dates[100].date())
    one = B.backtest(ctx, signal, spec(n=20, cost_bps=100.0, start=day, end=day), "once", info)
    assert one["series"]["gross_equity"][0][1] - one["series"]["equity"][0][1] == pytest.approx(0.01, abs=3e-6) and one["limits"]["dates"] == 1


def test_a_planted_signal_is_found_and_a_signal_of_noise_is_not(world):
    _, ctx, signal, info = world
    real = B.backtest(ctx, signal, spec(side="long_short", n=20), "real", info)
    assert real["summary"]["sharpe"] > 1.0 and real["summary"]["alpha_ann"] > 0.08 and real["deflated_sharpe"]["probability"] > 0.99
    for seed in range(3):
        fake = B.backtest(ctx, noise(ctx, seed), spec(side="long_short", n=20), f"noise{seed}", {"label": "noise", "warnings": [], "alphas": [], "n_trials": 1})
        assert fake["summary"]["sharpe"] < 0.6 and fake["summary"]["sharpe"] < real["summary"]["sharpe"] - 0.7
    only = B.backtest(ctx, noise(ctx, 0), spec(side="long_only", n=20), "noise", info)
    assert abs(only["summary"]["alpha_ann"]) < 0.06 and only["summary"]["information_ratio"] < 0.5          # a long-only book of nothing earns the market, not more


def test_the_deflated_sharpe_falls_with_the_number_of_ideas_tried(world):
    _, ctx, signal, info = world
    one = B.backtest(ctx, signal, spec(n=20), "one", info, n_trials=1)["deflated_sharpe"]["probability"]
    many = B.backtest(ctx, signal, spec(n=20), "many", info, n_trials=100_000)["deflated_sharpe"]
    assert many["n_trials"] == 100_000 and many["probability"] < one


def test_the_report_has_everything_a_page_needs_and_is_plain_json(world):
    svc, ctx, signal, info = world
    result = B.backtest(ctx, signal, spec(side="long_short", n=15, max_sector=0.3, max_beta=0.1), "report", info)
    json.dumps(result, allow_nan=False)
    assert set(result) >= {"summary", "gross_summary", "benchmark", "costs", "exposure", "limits", "series", "holdings", "warnings", "reading", "experiment_id", "universe", "portfolio", "signal"}
    assert {"equity", "gross_equity", "benchmark", "drawdown", "exposure", "turnover"} <= set(result["series"]) and result["series"]["equity"][0][0] > result["first_trade"]
    assert result["exposure"]["net_average"] == pytest.approx(0.0, abs=0.05) and result["exposure"]["gross_average"] == pytest.approx(2.0, abs=0.05)
    top = result["holdings"][0]
    assert set(top) == {"security_id", "ticker", "name", "weight", "sector", "country"} and len(result["holdings"]) <= 40 and abs(top["weight"]) <= 0.1 + 1e-9
    assert any("survivorship" in w for w in result["warnings"]) and "not that it will keep doing so" in result["reading"] and result["limits"]["dates"] == len(ctx.dates)
    assert result["limits"]["max_net_beta"] <= 0.1 + 1e-5 and result["limits"]["average_names_short"] == pytest.approx(15.0, abs=0.5)
    with svc.store.db.transaction() as conn:
        row = conn.execute("SELECT kind, status, summary FROM experiments WHERE experiment_id = ?", (result["experiment_id"],)).fetchone()
    assert row["kind"] == "backtest" and row["status"] == "finished" and json.loads(row["summary"])["sharpe"] == pytest.approx(result["summary"]["sharpe"])
    again = B.backtest(ctx, signal, spec(side="long_short", n=15, max_sector=0.3, max_beta=0.1), "report", info)
    other = B.backtest(ctx, signal, spec(side="long_short", n=16, max_sector=0.3, max_beta=0.1), "report", info)
    assert again["experiment_id"] == result["experiment_id"] != other["experiment_id"]


def test_a_signal_built_from_a_model_trades_only_from_its_first_prediction(world):
    _, ctx, _, _ = world
    parts = [{"kind": "factor", "name": "fcf_yield"}, {"kind": "factor", "name": "d_roic"}, {"kind": "technical", "name": "mom_12_1"}]
    plain = B.tear(ctx, parts, {"side": "long_short", "n": 15})
    result = B.tear(ctx, {"kind": "ml", "model": "ridge", "of": parts}, {"side": "long_short", "n": 15})
    assert result["signal"]["ml"]["first_prediction"] <= result["first_trade"] and result["limits"]["dates_in_cash"] >= 36
    assert any("held in cash" in w for w in result["warnings"]) and result["summary"]["n_obs"] < plain["summary"]["n_obs"] - 600
    mixed = B.tear(ctx, [{"kind": "ml", "model": "ridge", "of": parts, "weight": 1.0}, {"kind": "factor", "name": "fcf_yield", "weight": 1.0}], {"side": "long_only", "n": 20})
    assert mixed["signal"]["kind"] == "composite" and mixed["limits"]["dates_in_cash"] < result["limits"]["dates_in_cash"]      # before the model speaks, the factor alone is the view
