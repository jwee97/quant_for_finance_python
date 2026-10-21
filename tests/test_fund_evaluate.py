"""Factor evaluation against known truth: planted alphas are found, noise is not, a signal that knows the future is caught, and the arithmetic of IC, buckets, portfolios and costs is exact."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import evaluate as E
from src.fundamentals import factors as F
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.pit import PitSettings
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("evaluate"))
    # the hidden alphas are made two and a half times their default strength so that a world of a hundred companies is conclusive about them
    svc.load_simulated(n_companies=100, start_year=2009, end_year=2018, seed=13, planted={"delta_roic": 0.011, "fcf_yield": 0.0068, "asset_growth": 0.0045})
    return ResearchContext(svc.store, UniverseSpec())


# ------------------------------------------------------------------------------------------------------------------------ the arithmetic
def test_the_newey_west_error_matches_a_reference_and_widens_with_autocorrelation():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    e = rng.standard_normal(300)
    x = np.empty(300)
    x[0] = e[0]
    for i in range(1, 300):
        x[i] = 0.6 * x[i - 1] + e[i]
    x = x + 0.3
    ours = E.newey_west(x, lags=6)
    ref = sm.OLS(x, np.ones(300)).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": False})
    assert ours["se"] == pytest.approx(float(ref.bse[0]), rel=1e-6) and ours["t"] == pytest.approx(float(ref.tvalues[0]), rel=1e-6)
    naive_se = x.std(ddof=1) / np.sqrt(300)
    assert ours["se"] > 1.5 * naive_se                                                                  # positive autocorrelation: the naive error is far too small
    white = E.newey_west(rng.standard_normal(500), lags=0)
    assert white["se"] == pytest.approx(1 / np.sqrt(500), rel=0.1) and E.newey_west([1.0, 2.0])["t"] != E.newey_west([1.0, 2.0])["t"]
    assert E.overlap_lags(63, 21) == 2 and E.overlap_lags(21, 21) == 0 and E.overlap_lags(252, 21) == 11


def test_buckets_split_by_rank_and_a_perfect_signal_has_an_ic_of_one_and_rising_buckets():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.standard_normal((30, 100)), index=pd.date_range("2020-01-31", periods=30, freq="ME"), columns=range(100))
    b = E.buckets(returns, 5)
    assert set(b.stack().unique()) == {1.0, 2.0, 3.0, 4.0, 5.0} and (b.apply(lambda r: r.value_counts().min(), axis=1) == 20).all()
    assert E.rank_ic(returns, returns, 20).round(10).eq(1.0).all()
    q = E.quantile_returns(returns, returns, 5).mean()
    assert list(q.index) == [1, 2, 3, 4, 5] and q.is_monotonic_increasing and E.monotonicity(q)["spearman"] == pytest.approx(1.0) and E.monotonicity(q)["rising_pairs"] == 1.0
    noise = pd.DataFrame(rng.standard_normal(returns.shape), index=returns.index, columns=returns.columns)
    assert abs(E.rank_ic(noise, returns, 20).mean()) < 0.03
    assert E.monotonicity(pd.Series([3, 1, 2, 5, 4.0]))["rising_pairs"] == 0.5 and np.isnan(E.monotonicity(pd.Series([1.0, 2.0]))["spearman"])
    ties = pd.DataFrame(np.tile([0.0] * 50 + [1.0] * 50, (3, 1)), columns=range(100), index=returns.index[:3])
    assert set(E.buckets(ties, 5).stack().unique()) == {2.0, 4.0}                                      # two distinct values cannot fill five buckets


def test_the_stats_of_an_ic_series_include_the_error_that_allows_for_overlapping_horizons():
    ic = pd.Series(np.random.default_rng(2).normal(0.03, 0.08, 120), index=pd.date_range("2010-01-31", periods=120, freq="ME"))
    one, three = E.ic_stats(ic, 21, 21, 12), E.ic_stats(ic, 63, 21, 12)
    assert one["mean_ic"] == pytest.approx(ic.mean()) and one["icir"] == pytest.approx(ic.mean() / ic.std()) and one["icir_annualised"] == pytest.approx(one["icir"] * np.sqrt(12))
    assert one["positive_share"] == pytest.approx((ic > 0).mean()) and three["hac_lags"] >= 2 and E.ic_stats(pd.Series(dtype=float), 21, 21, 12) == {"n_dates": 0}


def test_holding_returns_are_contiguous_and_compound_to_the_price_ratio(ctx):
    hold = E.holding_returns(ctx)
    alive = [i for i in ctx.ids if bool(ctx.info.loc[i, "active"])][0]
    first, last = 5, len(ctx.dates) - 2
    compounded = (1.0 + hold[alive].iloc[first:last]).prod()
    entry = ctx.pos + ctx.entry_lag
    assert compounded == pytest.approx(ctx.adj[alive].iloc[entry[last]] / ctx.adj[alive].iloc[entry[first]])
    assert hold.iloc[-1].isna().all()
    gone = [i for i in ctx.ids if not bool(ctx.info.loc[i, "active"])][0]
    last_pos = ctx.adj.index.get_loc(ctx.adj[gone].last_valid_index())
    k = int(np.searchsorted(entry, last_pos, side="left")) - 1
    assert np.isfinite(hold[gone].iloc[k])                                                              # the period that contained the delisting is carried to the last price


def test_the_legs_of_a_long_short_portfolio_are_chosen_and_weighted_as_described(ctx):
    sig = pd.DataFrame(np.random.default_rng(3).standard_normal((20, ctx.n)), index=ctx.dates[40:60], columns=ctx.ids)
    for weighting in ("equal", "signal", "inv_vol"):
        wl, ws = E.leg_weights(sig, 0.2, weighting, ctx)
        assert np.allclose(wl.sum(axis=1), 1.0) and np.allclose(ws.sum(axis=1), 1.0) and (wl * ws).sum().sum() == 0
        top, bottom = sig.rank(axis=1, pct=True) > 0.8, sig.rank(axis=1, pct=True) <= 0.2
        if weighting == "inv_vol":                                                                      # a company whose volatility is unknown cannot be sized, so it is left out of its leg
            assert (((wl > 0) & ~top).sum().sum() == 0) and (((ws > 0) & ~bottom).sum().sum() == 0)
        else:
            assert ((wl > 0) == top).all().all() and ((ws > 0) == bottom).all().all()
    equal, _ = E.leg_weights(sig, 0.2, "equal", ctx)
    assert np.allclose(equal.iloc[0][equal.iloc[0] > 0], 1 / (equal.iloc[0] > 0).sum())
    heavy, _ = E.leg_weights(sig, 0.2, "signal", ctx)
    top0 = heavy.iloc[0][heavy.iloc[0] > 0]
    assert top0.idxmax() == sig.iloc[0][top0.index].idxmax()                                           # the strongest signal gets the largest weight
    empty, _ = E.leg_weights(sig.iloc[:, :8], 0.1, "equal", ctx)
    assert (empty.sum(axis=1) == 0).all()                                                               # a leg of fewer than three companies is not traded
    with pytest.raises(FundamentalsError):
        E.leg_weights(sig, 0.2, "magic", ctx)


def test_costs_come_off_in_proportion_to_what_was_traded_and_a_constant_signal_costs_almost_nothing(ctx):
    cheap = E.EvalSpec(cost_bps=0.0, borrow_bps=0.0)
    dear = E.EvalSpec(cost_bps=50.0, borrow_bps=100.0)
    rng = np.random.default_rng(4)
    random_signal = pd.DataFrame(rng.standard_normal((len(ctx.dates), ctx.n)), index=ctx.dates, columns=ctx.ids)
    a, b = E.long_short(random_signal, ctx, cheap), E.long_short(random_signal, ctx, dear)
    assert a["gross"].equals(b["gross"]) and (a["net"] - a["gross"]).abs().max() == 0
    drag = (b["gross"] - b["net"]).dropna()
    traded = b["turnover"].dropna() * 2
    borrow = 100.0 / 1e4 * 21 / 252.0
    assert np.allclose(drag.loc[traded.index], 50.0 / 1e4 * traded + borrow, atol=1e-12)
    assert b["turnover"].dropna().mean() > 0.5                                                          # a signal with no memory rebuilds most of the book every month
    sticky = pd.DataFrame(np.tile(rng.standard_normal(ctx.n), (len(ctx.dates), 1)), index=ctx.dates, columns=ctx.ids)
    s = E.long_short(sticky, ctx, dear)["turnover"].dropna()
    assert s.iloc[2:].mean() < 0.1 and s.iloc[2:].max() < 0.2                                           # the same signal every month only trades the drift


# ------------------------------------------------------------------------------------------------------------------------ known truth
def test_planted_alphas_are_found_and_noise_is_not(ctx):
    found = {n: E.evaluate_factor(ctx, E.EvalSpec(factor=n, breakdowns=())) for n in ("d_roic", "fcf_yield", "current_ratio", "debt_to_equity", "operating_margin")}
    assert found["d_roic"]["ic"]["t_stat"] > 3.0 and found["d_roic"]["ic"]["mean_ic"] > 0.015 and found["d_roic"]["portfolio"]["net"]["sharpe"] > 0.5
    assert found["fcf_yield"]["ic"]["t_stat"] > 4.0 and found["fcf_yield"]["quantiles"]["monotonicity"]["spearman"] >= 0.9
    for noise in ("current_ratio", "debt_to_equity", "operating_margin"):
        assert abs(found[noise]["ic"]["t_stat"]) < 2.5, noise
    assert "not validated" in found["d_roic"]["verdict"] or "validated" not in found["d_roic"]["verdict"].replace("not validated", "")
    assert found["fcf_yield"]["evidence_level"] == "good" and all(found[n]["evidence_level"] == "bad" for n in ("current_ratio", "debt_to_equity", "operating_margin"))
    assert found["fcf_yield"]["verdict"].startswith("Strong in-sample evidence") and "validated" in found["fcf_yield"]["verdict"]


def test_the_direction_can_be_given_flipped_or_chosen_after_the_fact_and_the_report_says_which(ctx):
    plain = E.evaluate_factor(ctx, E.EvalSpec(factor="asset_growth", breakdowns=()))
    flipped = E.evaluate_factor(ctx, E.EvalSpec(factor="asset_growth", direction=1, breakdowns=()))
    assert plain["ic"]["mean_ic"] == pytest.approx(-flipped["ic"]["mean_ic"]) and plain["factor"]["direction"] == -1
    auto = E.evaluate_factor(ctx, E.EvalSpec(factor="asset_growth", direction="auto", breakdowns=()))
    assert any("choice made on the data" in w for w in auto["warnings"]) and auto["ic"]["mean_ic"] >= 0
    with pytest.raises(FundamentalsError, match="no factor called"):
        E.evaluate_factor(ctx, E.EvalSpec(factor="vibes"))
    with pytest.raises(FundamentalsError, match="name a factor"):
        E.build_signal(ctx, E.EvalSpec())


def test_the_ic_decay_curve_rises_with_the_horizon_for_a_slow_signal(ctx):
    r = E.evaluate_factor(ctx, E.EvalSpec(factor="d_roic", breakdowns=()))
    ic = {d["horizon"]: d["mean_ic"] for d in r["decay"]}
    assert list(ic) == ["1w", "1m", "3m", "6m", "12m"] and ic["1w"] < ic["1m"] < ic["3m"] and ic["6m"] > ic["1m"]


def test_the_breakdowns_report_countries_sectors_regimes_and_halves(ctx):
    r = E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield"))
    countries = r["breakdown"]["country"]
    assert set(countries) <= set(ctx.info["country"]) and countries and all(v["companies"] >= 8 and v["n_dates"] >= 12 for v in countries.values())
    assert r["breakdown"]["sector"] and set(r["breakdown"]["regime"]) == {"trend", "volatility"}
    trend = r["breakdown"]["regime"]["trend"]
    assert set(trend) <= {"bull", "bear"} and sum(v["n_dates"] for v in trend.values()) <= r["ic"]["n_dates"]
    assert r["stability"]["halves_agree"] is True and r["stability"]["positive_years"] >= 0.7 and len(r["stability"]["by_year"]) >= 8
    labels = E.regimes(ctx)
    assert labels["trend"].iloc[:10].isna().all() and set(labels["trend"].dropna()) <= {"bull", "bear"} and set(labels["vol"].dropna()) <= {"high", "low"}


def test_the_report_is_plain_json_with_the_charts_the_tear_sheet_needs_and_honest_warnings(ctx):
    r = E.evaluate_factor(ctx, E.EvalSpec(factor="piotroski_f", breakdowns=("country",)))
    text = json.dumps(r)
    assert "NaN" not in text and "Infinity" not in text
    charts = r["charts"]
    assert {"ic", "ic_rolling", "ic_hist", "quantile_mean", "quantile_cumulative", "ls_cumulative_gross", "ls_cumulative_net", "ls_drawdown", "turnover", "decay", "coverage"} <= set(charts)
    assert len(charts["ic_hist"]["counts"]) == 20 and len(charts["quantile_cumulative"]) == 5 and charts["ic"][0][0].count("-") == 2 and min(v for _, v in charts["ls_drawdown"]) <= 0
    assert any("survivorship" in w for w in r["warnings"]) and any("distinct values" in w for w in r["warnings"])        # an F-score has ten values: uneven buckets
    assert r["universe"]["companies"] == 100 and r["spec"]["factor"] == "piotroski_f"


def test_a_window_restricts_the_study_to_the_dates_asked_for(ctx):
    full = E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield", breakdowns=()))
    early = E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield", end="2014-12-31", breakdowns=()))
    assert early["ic"]["n_dates"] < full["ic"]["n_dates"] and early["charts"]["ic"][-1][0] <= "2014-12-31"


def test_nothing_before_a_date_depends_on_prices_after_it(ctx):
    twin = ResearchContext(ctx.store, UniverseSpec())
    cut = ctx.adj.index[len(ctx.adj) // 2]
    twin.adj = twin.adj.copy()
    twin.adj.loc[twin.adj.index > cut] *= 3.0
    twin._cache.clear()
    before = ctx.dates < cut - pd.Timedelta(days=40)
    for panel in ("past", "vol", "beta"):
        a, b = {"past": ctx.past, "vol": ctx.volatility, "beta": ctx.beta}[panel](), {"past": twin.past, "vol": twin.volatility, "beta": twin.beta}[panel]()
        assert np.allclose(a[before].to_numpy(float), b[before].to_numpy(float), equal_nan=True), panel
    assert np.allclose(F.compute(ctx, "mom_12_1")[before].to_numpy(float), F.compute(twin, "mom_12_1")[before].to_numpy(float), equal_nan=True)


# ------------------------------------------------------------------------------------------------------------------------ the verdict and the look-ahead audit
def test_the_verdict_has_five_levels_and_never_calls_anything_validated():
    def fake(t, mean, sharpe, mono, n=60):
        return {"ic": {"n_dates": n, "t_stat": t, "mean_ic": mean}, "portfolio": {"net": {"sharpe": sharpe}}, "quantiles": {"monotonicity": {"spearman": mono}}}

    assert E.verdict(fake(0.5, 0.01, 0.1, 0.5)).startswith("No clear evidence")
    assert "weak" in E.verdict(fake(2.2, 0.02, 0.5, 0.9)) and "not validated" in E.verdict(fake(2.2, 0.02, 0.5, 0.9))
    strong = E.verdict(fake(4.0, 0.04, 1.2, 0.95))
    assert strong.startswith("Strong in-sample evidence") and "Interesting is not validated" in strong and "buckets rise in order" in strong
    assert "loses money after costs" in E.verdict(fake(4.0, 0.04, -0.3, 0.95)) and "do not rise cleanly" in E.verdict(fake(4.0, 0.04, 1.0, 0.2))
    assert "AGAINST" in E.verdict(fake(-3.5, -0.03, -1.0, -0.9)) and E.verdict(fake(4.0, 0.04, 1.0, 0.9, n=10)).startswith("Too little data")


def test_the_headline_the_sentence_under_it_and_the_terminal_agree_and_change_at_a_t_of_2_and_3_and_a_net_sharpe_of_one_half():
    from src.fundamentals import report

    def fake(t, sharpe, n=60):
        return {"ic": {"n_dates": n, "t_stat": t, "mean_ic": 0.03 if (t or 0) >= 0 else -0.03}, "portfolio": {"net": {"sharpe": sharpe}}, "quantiles": {"monotonicity": {"spearman": 0.9}}}

    lead = {"good": "Strong in-sample evidence", "mixed": "Some evidence", "bad": "No clear evidence"}
    for t in (None, float("nan"), 0.0, 1.64, 1.66, 1.99, 2.0, 2.99, 3.0, 3.01, -2.5, -3.5, 5.0):
        for sharpe in (None, float("nan"), -0.3, 0.0, 0.49, 0.5, 0.51, 2.0):
            result = fake(t, sharpe)
            level = E.evidence_level(t, sharpe)
            assert report.headline(result) == E.EVIDENCE_WORDS[level], (t, sharpe)
            if t is not None and np.isfinite(t):
                assert E.verdict(result).startswith(lead[level]), (t, sharpe)
    assert [E.evidence_level(t, 1.0) for t in (1.99, 2.0, 2.99, 3.0, -3.0)] == ["bad", "mixed", "mixed", "good", "good"]
    assert [E.evidence_level(3.0, s) for s in (0.5, 0.51, None, float("nan"), -1.0)] == ["mixed", "good", "mixed", "mixed", "mixed"]          # a clear signal that does not clearly pay is not "strong"
    assert E.evidence_level(None, 2.0) == "bad" and E.evidence_level(float("nan"), 2.0) == "bad"
    clear_but_unpaid = E.verdict(fake(3.4, 0.43))
    assert "short of the 0.5" in clear_but_unpaid and "net Sharpe of 0.43" in clear_but_unpaid and "not validated" in clear_but_unpaid
    assert "could not be measured" in E.verdict(fake(3.4, None)) and "could not be measured" in E.verdict(fake(3.4, float("nan")))


def _anticipation_world(tmp_path, n=150, seed=5):
    """A world where the market learns each year's figure in the three months before the company files it: a signal built from the figure predicts those months only if it is used before filing."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2011-01-03", "2019-12-31")
    store = FundamentalsStore(tmp_path / "anticipation.db")
    ids, figures = [], {}
    for i in range(n):
        ids.append(store.upsert_security(CompanyProfile("simulated", f"S{i}", f"S{i}", f"S{i}", exchange="NYSE", market="NYSE", country="US", currency="USD", sector="X")))
    drift = np.zeros((len(days), n))
    year_of = days.year.to_numpy()
    for year in range(2010, 2019):
        z = rng.standard_normal(n)
        figures[year] = z
        window = (year_of == year + 1) & (days.month.to_numpy() <= 3)
        drift[window] += 0.0025 * z                                                                 # the market prices the news before the filing, which is on 31 March
    returns = rng.normal(0, 0.012, size=(len(days), n)) + drift
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    frame = pd.DataFrame({"security_id": np.tile(ids, len(days)), "date": np.repeat(days.to_numpy(), n), "adj_close": prices.ravel(), "close": prices.ravel(), "volume": 1e6})
    store.store_prices(frame)
    for k, sid in enumerate(ids):
        store.store_financials(sid, "simulated", [PeriodStatement("FY", f"{y}-12-31", {"net_income": float(figures[y][k]), "revenue": 100.0}, fiscal_year=y, filing_date=f"{y + 1}-03-31", reporting_currency="USD")
                                                  for y in range(2010, 2019)])
    return store


def test_a_figure_used_before_it_was_public_is_caught_by_the_look_ahead_audit(tmp_path):
    store = _anticipation_world(tmp_path)
    spec = E.EvalSpec(factor=None, steps=[{"op": "zscore"}], horizon="1m")
    ctx = ResearchContext(store, UniverseSpec(), settings=PitSettings())
    naive = ResearchContext(store, UniverseSpec(), settings=PitSettings(naive_availability=True))
    honest_ic = E.rank_ic(E.transforms.apply(ctx.panel("net_income", "fy"), spec.steps, ctx), ctx.forward("1m"), 20)
    naive_ic = E.rank_ic(E.transforms.apply(naive.panel("net_income", "fy"), spec.steps, naive), naive.forward("1m"), 20)
    assert naive_ic.mean() > 0.04 and honest_ic.mean() < 0.01 and naive_ic.mean() > 5 * max(honest_ic.mean(), 0.002)         # known early, the figure "predicts"; known on time, it does not
    sig = E.transforms.apply(ctx.panel("net_income", "fy"), spec.steps, ctx)
    assert E.evaluate(ctx, sig, spec)["verdict"].startswith(("No clear evidence", "Some evidence"))


def test_the_audit_function_runs_the_study_both_ways_and_says_which_to_believe(ctx):
    audit = E.look_ahead_audit(ctx.store, E.EvalSpec(factor="fcf_yield", breakdowns=()), UniverseSpec())
    assert {"honest", "naive", "naive_minus_honest_ic", "flattered", "verdict"} <= set(audit) and audit["honest"]["n_dates"] > 100
    assert audit["flattered"] is False and "changes little" in audit["verdict"]


def test_the_trend_regime_is_bull_when_the_market_rose_over_the_last_252_days_and_unknown_before_a_year_of_history(ctx):
    trend = E.regimes(ctx)["trend"]
    m = ctx.market_return().fillna(0.0).to_numpy()
    for pos, label in zip(ctx.pos, trend):
        if pos < 252:
            assert pd.isna(label)
        else:
            assert label == ("bull" if float(np.prod(1.0 + m[pos - 251: pos + 1])) - 1.0 > 0 else "bear")
    assert (trend == "bull").sum() > 10 and (trend == "bear").sum() > 10 and trend.isna().sum() == int((ctx.pos < 252).sum())
