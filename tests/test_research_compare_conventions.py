"""The Research Comparison's arithmetic and sampling, checked against numbers worked out independently of the code under test.

The conventions: CAGR over elapsed calendar time; volatility and Sharpe scaled by the periods per year of the calendar the series was sampled on (252 business days, 365 for a continuous calendar, 52 weeks, 12 months) or, for irregular
spacing, by elapsed time; the risk-free rate accrued by elapsed time; results measured only on the dates they all observed, so that a date one lacks is never a zero return in another.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.research import compare as rc
from src.research import conventions as cv
from src.research import store as rs


@pytest.fixture()
def store(tmp_path):
    return rs.ResearchStore(tmp_path / "research")


def put(store, title, index, levels, *, meta=None, assets=("A",), **extra):
    s = store.create_session("c")["id"] if not hasattr(store, "_s") else store._s
    store._s = s
    series = {"dates": [str(d.date()) for d in index], "equity": [None if v is None else float(v) for v in levels]}
    if meta:
        series["meta"] = meta
    return store.add_artifact(s, {"kind": "backtest", "title": title, "evidence": "historical_simulation", "data_kind": "platform",
                                  "context": {"assets": list(assets), "start": str(index[0].date()), "end": str(index[-1].date()), "dataset_version": "v1"}, "metrics": {"sharpe": 1.0}, "series": series, **extra})


def levels_from(returns, start=1.0):
    return start * np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns))])


def alternating(n, mu, amp):
    return np.array([mu + amp if i % 2 == 0 else mu - amp for i in range(n)])


# ------------------------------------------------------------------------------------------------------ CAGR over elapsed time
def test_cagr_uses_the_elapsed_calendar_time_not_a_fixed_252_periods(store):
    idx = pd.date_range("2022-01-01", periods=366, freq="D")                                  # a 24/7 series: 365 days elapsed
    lv = levels_from(np.full(365, 0.001))
    a, b = put(store, "a", idx, lv), put(store, "b", idx, levels_from(np.full(365, 0.0005)))
    out = rc.compare([a, b])
    want = (1.001 ** 365) ** (365.25 / 365) - 1                                               # worked by hand: growth 1.001^365 over 365 days
    assert out["metrics"][a["id"]]["cagr"] == pytest.approx(want, rel=1e-9)
    old = (1.001 ** 365) ** (252 / 365) - 1                                                    # what the 252-per-period shortcut gave: a quite different number
    assert abs(out["metrics"][a["id"]]["cagr"] - old) > 0.15, "44% against 29%: the shortcut understated a 24/7 series by a third"


def test_cagr_of_an_irregular_series_is_the_elapsed_time_between_its_first_and_last_date():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-03-15", "2021-01-01", "2022-01-01"])
    lv = pd.Series([1.0, 1.1, 1.3, 1.69], index=idx)
    m = cv.measure_levels(lv, {"basis": "elapsed", "periods_per_year": None, "calendar": "custom", "frequency": "irregular", "source": "test"})
    years = (idx[-1] - idx[0]).days / 365.25
    assert m["cagr"] == pytest.approx(1.69 ** (1 / years) - 1, rel=1e-12) and m["elapsed_years"] == pytest.approx(years)


# ------------------------------------------------------------------------------------------------------ volatility, Sharpe, Sortino per calendar
@pytest.mark.parametrize("freq,ppy,n", [("B", 252, 400), ("D", 365, 500), ("W-FRI", 52, 160), ("ME", 12, 60)])
def test_volatility_sharpe_and_sortino_are_scaled_by_the_periods_per_year_of_the_calendar(store, freq, ppy, n):
    idx = pd.date_range("2018-01-01", periods=n + 1, freq=freq)
    r1, r2 = alternating(n, 0.0008, 0.01), alternating(n, 0.0004, 0.006)
    a, b = put(store, "a", idx, levels_from(r1)), put(store, "b", idx, levels_from(r2))
    out = rc.compare([a, b])
    assert out["conventions"]["common_dates"]["periods_per_year"] == ppy and out["conventions"]["common_dates"]["basis"] == "periods"
    for art, r in ((a, r1), (b, r2)):
        got = out["metrics"][art["id"]]
        sd = np.std(r, ddof=1)
        assert got["ann_vol"] == pytest.approx(sd * math.sqrt(ppy), rel=1e-9)
        assert got["sharpe"] == pytest.approx(math.sqrt(ppy) * r.mean() / sd, rel=1e-9)
        down = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
        assert got["sortino"] == pytest.approx(math.sqrt(ppy) * r.mean() / down, rel=1e-9)


def test_the_risk_free_rate_is_accrued_by_elapsed_time_so_a_weekend_return_carries_three_days_of_it(store):
    idx = pd.bdate_range("2021-01-04", periods=301)
    r = alternating(300, 0.0006, 0.008)
    a, b = put(store, "a", idx, levels_from(r)), put(store, "b", idx, levels_from(alternating(300, 0.0002, 0.005)))
    out = rc.compare([a, b], risk_free=0.04)
    dt = np.diff(idx.values).astype("timedelta64[D]").astype(float) / 365.25
    excess = r - ((1.04 ** dt) - 1.0)
    assert out["metrics"][a["id"]]["sharpe"] == pytest.approx(math.sqrt(252) * excess.mean() / excess.std(ddof=1), rel=1e-9)
    flat = r - (1.04 ** (1 / 252) - 1.0)
    assert abs(out["metrics"][a["id"]]["sharpe"] - math.sqrt(252) * flat.mean() / flat.std(ddof=1)) > 1e-6, "not the same as a flat per-period rate"


def test_irregular_spacing_is_annualised_by_elapsed_time_and_recovers_the_generating_volatility(store):
    rng = np.random.default_rng(3)
    gaps = rng.integers(1, 30, size=600)                                                       # gaps of 1 to 29 days
    idx = pd.DatetimeIndex(pd.Timestamp("2000-01-01") + pd.to_timedelta(np.concatenate([[0], np.cumsum(gaps)]), unit="D"))
    sigma = 0.20
    dt = gaps / 365.25
    z = np.where(np.arange(600) % 2 == 0, 1.0, -1.0)
    r = sigma * np.sqrt(dt) * z                                                                # volatility per root-year is sigma by construction
    a, b = put(store, "a", idx, levels_from(r)), put(store, "b", idx, levels_from(r * 0.5))
    out = rc.compare([a, b])
    assert out["conventions"]["common_dates"]["basis"] == "elapsed" and "irregular_spacing" in {i["code"] for i in out["issues"]}
    assert out["metrics"][a["id"]]["ann_vol"] == pytest.approx(sigma, rel=0.03)
    assert out["metrics"][b["id"]]["ann_vol"] == pytest.approx(sigma / 2, rel=0.03)
    naive = np.std(r, ddof=1) * math.sqrt(252)
    assert abs(naive - sigma) / sigma > 0.5, "multiplying irregular returns by sqrt(252) would have missed it badly"


# ------------------------------------------------------------------------------------------------------ equity and crypto calendars
def test_an_equity_and_a_crypto_series_are_measured_on_the_business_days_both_have_and_the_weekend_moves_are_kept(store):
    seven = pd.date_range("2022-01-03", periods=400, freq="D")                                # Monday start
    crypto_r = alternating(399, 0.001, 0.02)
    crypto = levels_from(crypto_r)
    bdays = pd.bdate_range("2022-01-03", seven[-1])
    eq_r = alternating(len(bdays) - 1, 0.0005, 0.008)
    a, b = put(store, "equity", bdays, levels_from(eq_r)), put(store, "crypto", seven, crypto)
    out = rc.compare([a, b])
    dates = pd.to_datetime(out["series"]["dates"])
    assert (dates.dayofweek < 5).all() and len(dates) == len(bdays)
    cry = pd.Series(crypto, index=seven)
    mon = [d for d in dates if d.dayofweek == 0][3]
    fri = dates[list(dates).index(mon) - 1]
    got_eq = out["series"]["equity"][b["id"]]
    ratio = got_eq[list(dates).index(mon)] / got_eq[list(dates).index(fri)]
    assert ratio == pytest.approx(cry[mon] / cry[fri], rel=1e-3), "Friday to Monday is the three-day return of the crypto series, not a dropped weekend"
    assert out["conventions"]["common_dates"]["calendar"] == "business_days" and out["conventions"]["common_dates"]["periods_per_year"] == 252.0
    assert "different_calendars" in {i["code"] for i in out["issues"]}
    cr = pd.Series(crypto, index=seven).loc[dates].pct_change().dropna()
    assert out["metrics"][b["id"]]["ann_vol"] == pytest.approx(cr.std(ddof=1) * math.sqrt(252), rel=1e-6)
    own = out["individual"][b["id"]]
    assert own["annualisation"]["periods_per_year"] == 365.0 and own["ann_vol"] == pytest.approx(np.std(crypto_r, ddof=1) * math.sqrt(365), rel=1e-6)


# ------------------------------------------------------------------------------------------------------ missing returns are not zeros
def test_a_result_with_a_hole_in_its_history_is_not_given_zero_returns_there_and_the_loss_is_reported(store):
    idx = pd.bdate_range("2021-01-04", periods=260)
    ra, rb = alternating(259, 0.0006, 0.01), alternating(259, 0.0004, 0.007)
    la, lb = levels_from(ra), levels_from(rb)
    hole = list(range(100, 112))                                                                # 12 business days missing from b only
    keep = [i for i in range(260) if i not in hole]
    a, b = put(store, "a", idx, la), put(store, "b", idx[keep], lb[keep])
    out = rc.compare([a, b])
    grid = idx[keep]
    assert out["series"]["dates"] == [str(d.date()) for d in grid]
    assert out["metrics"][a["id"]]["n_days"] == out["metrics"][b["id"]]["n_days"] == len(grid) - 1
    ref = pd.Series(la, index=idx).loc[grid].pct_change().dropna()
    assert out["metrics"][a["id"]]["ann_vol"] == pytest.approx(ref.std(ddof=1) * math.sqrt(252), rel=1e-9)
    assert out["metrics"][a["id"]]["total_return"] == pytest.approx(la[-1] / la[0] - 1, rel=1e-9)
    eq_b = out["series"]["equity"][b["id"]]
    assert eq_b[-1] == pytest.approx(lb[-1] / lb[0], rel=1e-4) and None not in eq_b
    cov = out["coverage"]
    assert cov[a["id"]]["dropped_in_window"] == 12 and cov[b["id"]]["dropped_in_window"] == 0 and cov[b["id"]]["longest_gap_days"] >= 15
    issue = next(i for i in out["issues"] if i["code"] == "sample_reduced")
    assert "12 of 260" in issue["message"] and "never counted" in issue["message"]


def test_a_genuine_flat_stretch_stays_in_the_sample_as_zero_returns_and_a_missing_one_does_not(store):
    idx = pd.bdate_range("2021-01-04", periods=201)
    r = alternating(200, 0.0005, 0.01)
    r_flat = r.copy()
    r_flat[80:120] = 0.0                                                                       # a strategy that held cash: observed, zero return
    la, lf = levels_from(r), levels_from(r_flat)
    a, b = put(store, "a", idx, la), put(store, "flat", idx, lf)
    out = rc.compare([a, b])
    assert out["metrics"][a["id"]]["n_days"] == out["metrics"][b["id"]]["n_days"] == 200
    assert out["coverage"][b["id"]]["dropped_in_window"] == 0
    assert out["metrics"][b["id"]]["ann_vol"] == pytest.approx(np.std(r_flat, ddof=1) * math.sqrt(252), rel=1e-9)
    assert out["metrics"][b["id"]]["ann_vol"] < out["metrics"][a["id"]]["ann_vol"], "the observed zeros lower the volatility, as they should"
    assert "sample_reduced" not in {i["code"] for i in out["issues"]}


def test_results_with_different_inception_dates_are_measured_from_the_later_one_and_show_their_own_history_separately(store):
    idx_a = pd.bdate_range("2019-01-01", periods=700)
    idx_b = idx_a[300:]
    ra, rb = alternating(699, 0.0007, 0.01), alternating(399, 0.0003, 0.006)
    la, lb = levels_from(ra), levels_from(rb)
    a, b = put(store, "old", idx_a, la), put(store, "new", idx_b, lb)
    out = rc.compare([a, b])
    assert out["window"]["start"] == str(idx_b[0].date())
    want = la[-1] / la[300] - 1
    assert out["metrics"][a["id"]]["total_return"] == pytest.approx(want, rel=1e-9), "measured from the day the later result began, not from its own start"
    assert out["individual"][a["id"]]["first"] == str(idx_a[0].date()) and out["individual"][a["id"]]["total_return"] == pytest.approx(la[-1] / la[0] - 1, rel=1e-9)
    assert out["individual"][a["id"]]["n_days"] > out["metrics"][a["id"]]["n_days"]
    assert "window_cut" in {i["code"] for i in out["issues"]}


def test_partial_coverage_that_leaves_too_few_common_dates_is_refused_not_padded(store):
    idx_a = pd.bdate_range("2021-01-04", periods=200)
    idx_b = pd.bdate_range("2021-09-01", periods=200)                                          # only about 40 days overlap, then fewer after the window edge
    a, b = put(store, "a", idx_a, levels_from(alternating(199, 0.0005, 0.01))), put(store, "b", idx_b, levels_from(alternating(199, 0.0005, 0.01)))
    out = rc.compare([a, b], start="2021-09-01", end="2021-10-01")
    assert out["metrics"] == {} and out["ranking"] is None and "short_overlap" in {i["code"] for i in out["issues"]}


def test_every_date_on_the_chart_was_observed_by_every_result(store):
    idx = pd.bdate_range("2021-01-04", periods=300)
    rng = np.random.default_rng(1)
    pieces = []
    for k in range(3):
        keep = sorted(rng.choice(300, size=270, replace=False))
        pieces.append(put(store, f"s{k}", idx[keep], levels_from(rng.normal(0.0004, 0.01, 269))))
    out = rc.compare(pieces)
    common = set.intersection(*[set(a["series"]["dates"]) for a in pieces])
    assert set(out["series"]["dates"]) <= common
    assert all(None not in v for v in out["series"]["equity"].values())


# ------------------------------------------------------------------------------------------------------ incompatible frequencies
def test_a_daily_and_a_monthly_series_are_compared_only_on_the_monthly_dates_and_the_page_is_told(store):
    daily = pd.bdate_range("2015-01-01", "2020-12-31")
    month_end = pd.DatetimeIndex(sorted({d for d in daily if (d + pd.offsets.BDay(1)).month != d.month}))
    rng = np.random.default_rng(5)
    ld = pd.Series(levels_from(rng.normal(0.0004, 0.01, len(daily) - 1)), index=daily)
    lm = pd.Series(levels_from(rng.normal(0.006, 0.04, len(month_end) - 1)), index=month_end)
    a, b = put(store, "daily", daily, ld.to_numpy()), put(store, "monthly", month_end, lm.to_numpy())
    out = rc.compare([a, b])
    assert out["conventions"]["common_dates"]["frequency"] == "monthly" and out["conventions"]["common_dates"]["periods_per_year"] == 12.0
    assert "mixed_frequency" in {i["code"] for i in out["issues"]}
    r = ld.loc[month_end].pct_change().dropna()
    assert out["metrics"][a["id"]]["ann_vol"] == pytest.approx(r.std(ddof=1) * math.sqrt(12), rel=1e-9)
    assert out["individual"][a["id"]]["annualisation"]["periods_per_year"] == 252.0


def test_declared_metadata_overrides_inference_and_is_kept_with_the_saved_series(store):
    idx = pd.bdate_range("2021-01-04", periods=300)
    meta = {"frequency": "daily", "calendar": "business_days", "periods_per_year": 260, "return_type": "simple", "currency": "USD"}
    r = alternating(299, 0.0005, 0.01)
    a, b = put(store, "a", idx, levels_from(r), meta=meta), put(store, "b", idx, levels_from(r * 0.5), meta=meta)
    assert store.get_artifact(a["id"])["series"]["meta"]["periods_per_year"] == 260
    out = rc.compare([a, b])
    assert out["conventions"]["common_dates"]["periods_per_year"] == 260.0 and out["conventions"]["common_dates"]["source"] == "declared by every series"
    assert out["metrics"][a["id"]]["ann_vol"] == pytest.approx(np.std(r, ddof=1) * math.sqrt(260), rel=1e-9)
    assert "annualisation_inferred" not in {i["code"] for i in out["issues"]}


def test_different_declared_currencies_or_return_types_stop_the_ranking(store):
    idx = pd.bdate_range("2021-01-04", periods=120)
    base = {"frequency": "daily", "calendar": "business_days", "periods_per_year": 252}
    r = alternating(119, 0.0005, 0.01)
    a = put(store, "usd", idx, levels_from(r), meta={**base, "currency": "USD", "return_type": "simple"})
    b = put(store, "eur", idx, levels_from(r), meta={**base, "currency": "EUR", "return_type": "simple"})
    c = put(store, "log", idx, levels_from(r), meta={**base, "currency": "USD", "return_type": "log"})
    assert rc.compare([a, b])["ranking"] is None and "different_currency" in {i["code"] for i in rc.compare([a, b])["issues"]}
    out = rc.compare([a, c])
    assert out["ranking"] is None and "different_return_type" in {i["code"] for i in out["issues"]}


# ------------------------------------------------------------------------------------------------------ rolling statistics, drawdown, benchmark-relative
def test_rolling_volatility_rolling_sharpe_and_drawdown_agree_with_an_independent_computation(store):
    idx = pd.date_range("2022-01-01", periods=301, freq="D")
    r = np.random.default_rng(2).normal(0.001, 0.02, 300)
    r2 = np.random.default_rng(3).normal(0.0005, 0.015, 300)
    a, b = put(store, "a", idx, levels_from(r)), put(store, "b", idx, levels_from(r2))
    out = rc.compare([a, b], roll=60, risk_free=0.03)
    s = pd.Series(r)
    rf = (1.03 ** (1 / 365.25)) - 1
    rv = s.rolling(60).std(ddof=1) * math.sqrt(365)
    rs_ = math.sqrt(365) * (s - rf).rolling(60).mean() / (s - rf).rolling(60).std(ddof=1)
    got_v, got_s = out["series"]["rolling_vol"][a["id"]], out["series"]["rolling_sharpe"][a["id"]]
    assert got_v[0] is None and got_v[59] is None and got_v[60] == pytest.approx(rv.iloc[59], abs=1e-4) and got_v[-1] == pytest.approx(rv.iloc[-1], abs=1e-4)
    assert got_s[-1] == pytest.approx(rs_.iloc[-1], abs=2e-3)
    eq = np.concatenate([[1.0], np.cumprod(1 + r)])
    dd = eq / np.maximum.accumulate(eq) - 1
    assert out["series"]["drawdown"][a["id"]][-1] == pytest.approx(dd[-1], abs=1e-5)
    assert out["metrics"][a["id"]]["max_drawdown"] == pytest.approx(dd.min(), rel=1e-9)
    assert out["metrics"][a["id"]]["calmar"] == pytest.approx(out["metrics"][a["id"]]["cagr"] / -dd.min(), rel=1e-9)


def test_benchmark_relative_numbers_use_the_calendar_of_the_common_dates(store):
    idx = pd.date_range("2022-01-01", periods=301, freq="D")
    rng = np.random.default_rng(8)
    rb = rng.normal(0.0005, 0.015, 300)
    r = 0.7 * rb + rng.normal(0.0003, 0.008, 300)
    a, b = put(store, "strategy", idx, levels_from(r)), put(store, "reference", idx, levels_from(rb))
    out = rc.compare([a, b], benchmark=b["id"])
    rel = out["versus_reference"][a["id"]]
    act = r - rb
    assert rel["tracking_error"] == pytest.approx(np.std(act, ddof=1) * math.sqrt(365), rel=1e-9)
    beta = np.cov(r, rb, ddof=1)[0, 1] / np.var(rb, ddof=1)
    assert rel["beta"] == pytest.approx(beta, rel=1e-9) and rel["alpha_annual"] == pytest.approx((r.mean() - beta * rb.mean()) * 365, rel=1e-9)
    assert rel["information_ratio"] == pytest.approx(act.mean() * 365 / (np.std(act, ddof=1) * math.sqrt(365)), rel=1e-9)


# ------------------------------------------------------------------------------------------------------ the old behaviour, written down so it cannot come back
def test_measuring_a_gappy_series_by_reindexing_and_filling_zero_would_have_changed_the_answer():
    idx = pd.bdate_range("2021-01-04", periods=100)
    lv = pd.Series(levels_from(np.full(99, 0.002)), index=idx)
    gappy = lv.drop(lv.index[40:50])
    zero_filled = gappy.pct_change().reindex(idx).fillna(0.0)
    assert (1 + zero_filled).prod() - 1 == pytest.approx(gappy.iloc[-1] / gappy.iloc[0] - 1, rel=1e-9), "the product survives only because the gap return lands on one date"
    assert zero_filled.std(ddof=1) < gappy.pct_change().dropna().std(ddof=1) or zero_filled.std(ddof=1) != gappy.pct_change().dropna().std(ddof=1)
    assert (zero_filled == 0).sum() >= 10, "ten dates with no information became ten observed zero returns"
