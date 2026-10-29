"""The book rules of ``strategies.systematic`` against independent calculations: loops written from the formulas of the sources, hand-worked paths, and properties the rule must have
(bounded, causal, sign of the trend), plus the parameter checks."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.strategies import systematic as sy

load_library()
IDX = pd.bdate_range("2010-01-04", periods=1800)


def bundle(paths: dict, n: int = 1800, seed: int = 11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    cols = {}
    for name, spec in paths.items():
        if isinstance(spec, tuple):
            drift, noise = spec
            cols[name] = 100 * np.cumprod(1 + drift + rng.normal(0, noise, n))
        else:
            cols[name] = np.asarray(spec, dtype=float)
    prices = pd.DataFrame(cols, index=idx)
    return bundle_from_prices(prices, asset_class={c: "equity" for c in prices.columns}, name="sys")


@pytest.fixture(scope="module")
def trending():
    return bundle({"UP": (0.0008, 0.008), "DOWN": (-0.0006, 0.008), "FLAT": (0.0, 0.010)})


def model(name, **params):
    return MODELS.create(name, **params)


# ------------------------------------------------------------------------------------------------------------------ Carver
def ema(x, span):
    a, out = 2.0 / (span + 1.0), []
    for i, v in enumerate(x):
        out.append(v if i == 0 else (1 - a) * out[-1] + a * v)
    return np.array(out)


def test_ewmac_raw_is_the_gap_of_two_averages_over_the_volatility_of_price_changes(trending):
    p = trending.prices["UP"]
    raw = sy.ewmac_raw(p.to_frame(), 8, 32, 35)["UP"]
    gap = ema(p.to_numpy(), 8) - ema(p.to_numpy(), 32)
    sigma = p.diff().ewm(span=35, adjust=False, min_periods=35).std()
    expected = pd.Series(gap, index=p.index) / sigma
    expected[:32 - 1] = np.nan                                                      # the slow average needs its span before it is used
    assert raw.iloc[:31].isna().all()
    assert np.allclose(raw.iloc[40:], expected.iloc[40:], rtol=1e-10, atol=1e-12)


def test_ewmac_on_a_straight_line_is_half_the_gap_in_spans_per_unit_slope():
    p = pd.DataFrame({"X": 100.0 + 0.5 * np.arange(600) + 0.01 * np.sin(np.arange(600))}, index=pd.bdate_range("2015-01-02", periods=600))
    gap = (p["X"].ewm(span=16, adjust=False).mean() - p["X"].ewm(span=64, adjust=False).mean()).iloc[-1]
    assert gap == pytest.approx(0.5 * (64 - 16) / 2.0, abs=0.01)                    # an average of span s lags a ramp of slope b by b (s - 1) / 2 days


@pytest.mark.parametrize("corr, expected", [
    (np.eye(6), math.sqrt(6.0)),                                                    # six uncorrelated forecasts: 1 / sqrt(6 / 36)
    (np.ones((6, 6)), 1.0),                                                         # perfectly correlated: averaging shrinks nothing
    (np.array([[1.0, 0.5], [0.5, 1.0]]), 1.0 / math.sqrt(0.75)),
    (np.array([[1.0, -0.5], [-0.5, 1.0]]), math.sqrt(2.0)),                          # a negative correlation counts as zero
])
def test_the_forecast_diversification_multiplier_is_one_over_the_volatility_of_the_equal_weighted_average(corr, expected):
    assert sy.forecast_diversification_multiplier(corr) == pytest.approx(expected, rel=1e-12)


def test_the_multiplier_is_capped():
    assert sy.forecast_diversification_multiplier(np.eye(10), cap=2.5) == 2.5 and sy.forecast_diversification_multiplier(np.eye(10), cap=4.0) == pytest.approx(math.sqrt(10))


def test_the_scalar_on_a_date_is_the_target_over_the_pooled_average_absolute_raw_forecast_through_that_date(trending):
    m = model("carver_ewmac", fast_spans=(8, 16), target_abs=10.0, cap=1e9, diversification=False, min_history=300)
    raw = m.raw_forecasts(trending)
    capped, ready = m.capped_forecasts(trending)
    for r, c in zip(raw, capped):
        for t in (700, 1200, 1799):
            pooled = r.iloc[:t + 1].abs().stack().mean()                              # every asset, every date up to t that has a value
            assert np.allclose(c.iloc[t], r.iloc[t] * 10.0 / pooled, rtol=1e-12)
    assert int(ready.iloc[-1]) == 1800 - 64 + 1 and int(ready.iloc[0]) == 0           # the slowest average (span 64, for fast 16) is the last to be ready: position 63 is the first date


def test_the_expanding_scalar_at_a_date_uses_data_through_that_date_only(trending):
    m = model("carver_ewmac", fast_spans=(8,), cap=1e9, diversification=False)
    full = m.capped_forecasts(trending)[0][0]
    cut = trending.index[900]
    part = m.capped_forecasts(bundle_from_prices(trending.prices.loc[:cut], asset_class={c: "equity" for c in trending.prices.columns}, name="cut"))[0][0]
    assert np.allclose(full.loc[:cut], part.loc[:cut], equal_nan=True, rtol=1e-10)


def test_carver_score_is_bounded_follows_the_trend_and_is_causal(trending):
    m = model("carver_ewmac")
    s = m.score(trending).dropna()
    assert s.abs().max().max() <= 20.0 / 10.0 + 1e-12
    late = s.iloc[-400:]
    assert late["UP"].mean() > 0.3 and late["DOWN"].mean() < -0.3                    # the drifting assets are held in the direction of their drift (seed 11: +0.82 and -1.45)
    assert check_causality(m, trending)["ok"]


def test_carver_score_is_empty_until_enough_history_for_the_scalars(trending):
    first = model("carver_ewmac", min_history=900).score(trending).dropna(how="all").index[0]
    assert trending.index.get_loc(first) == 255 + 900 - 1                            # the span-256 average is ready at position 255; then 900 dates with every speed ready


def test_the_carver_multiplier_raises_the_forecast_of_uncorrelated_speeds(trending):
    base = model("carver_ewmac", diversification=False).score(trending)
    boosted = model("carver_ewmac", diversification=True).score(trending)
    both = base.notna() & boosted.notna()
    ratio = (boosted.where(both).abs().stack().sum()) / (base.where(both).abs().stack().sum())
    assert 1.0 < ratio <= 2.5 + 1e-9


@pytest.mark.parametrize("kwargs", [dict(fast_spans=()), dict(slow_multiple=1.0), dict(vol_span=2), dict(target_abs=0.0), dict(cap=5.0, target_abs=10.0), dict(min_history=5), dict(fast_spans=(0,))])
def test_carver_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        model("carver_ewmac", **kwargs)


# ------------------------------------------------------------------------------------------------------------------ Clenow
def test_exponential_regression_matches_numpy_polyfit_on_a_window(trending):
    annual, r2 = sy.exponential_regression(trending.prices, 90, 250)
    t = 1500
    y = np.log(trending.prices["UP"].iloc[t - 89:t + 1].to_numpy())
    x = np.arange(90.0)
    slope, intercept = np.polyfit(x, y, 1)
    fit = intercept + slope * x
    expected_r2 = 1.0 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert annual["UP"].iloc[t] == pytest.approx(math.exp(slope * 250) - 1.0, rel=1e-8)
    assert r2["UP"].iloc[t] == pytest.approx(expected_r2, rel=1e-8)
    assert annual["UP"].iloc[:89].isna().all() and annual["UP"].iloc[89:].notna().all()


def test_a_perfect_exponential_trend_has_r2_one_and_the_annual_rate_it_was_built_with():
    idx = pd.bdate_range("2015-01-02", periods=300)
    p = pd.DataFrame({"X": 50.0 * np.exp(0.001 * np.arange(300))}, index=idx)
    annual, r2 = sy.exponential_regression(p, 90, 250)
    assert r2["X"].iloc[-1] == pytest.approx(1.0, abs=1e-12) and annual["X"].iloc[-1] == pytest.approx(math.exp(0.25) - 1.0, rel=1e-9)


def test_clenow_score_is_slope_times_r2_and_excludes_names_below_trend_or_with_a_gap():
    n = 400
    idx = pd.bdate_range("2015-01-02", periods=n)
    rng = np.random.default_rng(3)
    smooth = 100 * np.exp(0.0015 * np.arange(n) + rng.normal(0, 0.002, n))
    jumpy = smooth.copy()
    jumpy[350:] *= 1.25                                                              # a 25% one-day jump inside the 90-day window
    falling = 100 * np.exp(-0.001 * np.arange(n) + rng.normal(0, 0.002, n))
    b = bundle_from_prices(pd.DataFrame({"SMOOTH": smooth, "JUMPY": jumpy, "FALL": falling}, index=idx), asset_class={"SMOOTH": "equity", "JUMPY": "equity", "FALL": "equity"}, name="cl")
    m = model("clenow_momentum")
    s = m.score(b)
    annual, r2 = sy.exponential_regression(b.prices, 90, 250)
    assert s["SMOOTH"].iloc[-1] == pytest.approx(annual["SMOOTH"].iloc[-1] * r2["SMOOTH"].iloc[-1], rel=1e-12)
    assert np.isnan(s["JUMPY"].iloc[-1]) and np.isnan(s["FALL"].iloc[-1])           # the gap is inside the window; the falling name is below its 100-day average
    assert not np.isnan(s["JUMPY"].iloc[300])                                        # before the jump it was eligible
    plain = model("clenow_momentum", use_r2=False).score(b)
    assert plain["SMOOTH"].iloc[-1] == pytest.approx(annual["SMOOTH"].iloc[-1], rel=1e-12)
    assert check_causality(model("clenow_momentum"), bundle({"A": (0.0006, 0.01), "B": (0.0002, 0.01), "C": (-0.0004, 0.012)}))["ok"]


@pytest.mark.parametrize("kwargs", [dict(window=5), dict(trend_window=2), dict(gap=0.0), dict(annual_days=0)])
def test_clenow_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        model("clenow_momentum", **kwargs)


# ------------------------------------------------------------------------------------------------------------------ t-statistic trend
def test_tstat_momentum_is_the_capped_t_statistic_of_the_daily_log_returns(trending):
    m = model("tstat_momentum", lookback=252)
    s = m.score(trending)
    r = np.log(trending.prices["UP"]).diff().iloc[1000 - 251:1001].to_numpy()
    t = r.mean() / (r.std(ddof=1) / math.sqrt(252))
    assert s["UP"].iloc[1000] == pytest.approx(max(-1.0, min(1.0, t)), rel=1e-10)
    assert s.abs().max().max() <= 1.0 and s["UP"].iloc[:251].isna().all()


def test_tstat_momentum_scale_stretches_the_cap_and_is_causal(trending):
    wide = model("tstat_momentum", scale=3.0).score(trending)
    narrow = model("tstat_momentum", scale=1.0).score(trending)
    inside = narrow.abs() < 1.0                                                      # where the cap does not bind, tripling the scale divides the score by three
    assert inside.to_numpy().sum() > 1000 and np.allclose((narrow / 3.0)[inside], wide[inside], equal_nan=True)
    assert (wide.abs().max() <= 1.0).all() and (wide.abs().max() < narrow.abs().max() + 1e-12).all()
    assert check_causality(model("tstat_momentum"), trending)["ok"]
    with pytest.raises(ValueError):
        model("tstat_momentum", lookback=5)


# ------------------------------------------------------------------------------------------------------------------ Lo-MacKinlay
def lm_reference(r, q):
    """The statistics straight from Lo and MacKinlay (1988): loops over the formulas, no vectorisation."""
    n = len(r)
    mu = sum(r) / n
    s_a = sum((x - mu) ** 2 for x in r) / (n - 1)
    m = q * (n - q + 1) * (1 - q / n)
    s_c = sum((sum(r[t - q + 1:t + 1]) - q * mu) ** 2 for t in range(q - 1, n)) / m
    vr = s_c / s_a
    den = sum((x - mu) ** 2 for x in r) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = n * sum((r[t] - mu) ** 2 * (r[t - j] - mu) ** 2 for t in range(j, n)) / den
        theta += 4.0 * (1.0 - j / q) ** 2 * delta
    return vr, math.sqrt(n) * (vr - 1.0) / math.sqrt(theta)


@pytest.mark.parametrize("q", [2, 3, 5, 10])
def test_variance_ratio_statistics_match_the_formulas_of_the_paper(q):
    rng = np.random.default_rng(q)
    w = rng.standard_t(5, (7, 120)) * 0.01 + 0.0003
    vr, z = sy.variance_ratio_stats(w, q)
    for i in range(len(w)):
        ref_vr, ref_z = lm_reference(w[i].tolist(), q)
        assert vr[i] == pytest.approx(ref_vr, rel=1e-10) and z[i] == pytest.approx(ref_z, rel=1e-10)


def test_an_alternating_return_series_has_a_zero_variance_ratio_and_a_known_statistic():
    n = 20
    w = np.array([[0.01 * (-1) ** i for i in range(n)]])
    vr, z = sy.variance_ratio_stats(w, 2)
    assert vr[0] == pytest.approx(0.0, abs=1e-12)
    assert z[0] == pytest.approx(-n / math.sqrt(n - 1), rel=1e-12)                    # delta_1 = (n - 1) / n and theta = delta_1 for q = 2


def test_the_robust_statistic_is_about_standard_normal_for_a_random_walk_and_has_the_right_sign_for_autocorrelation():
    rng = np.random.default_rng(7)
    iid = rng.normal(0, 0.01, (4000, 252))
    _, z = sy.variance_ratio_stats(iid, 5)
    assert abs(z.mean()) < 0.1 and 0.9 < z.std() < 1.15 and 0.03 < (np.abs(z) > 1.96).mean() < 0.08
    e = rng.normal(0, 0.01, (400, 252))
    for phi, bound in ((0.4, 5.0), (-0.4, -2.5)):                                  # VR(5) - 1 is about 0.95 and -0.49 for these AR(1) series: z* about +6.9 and -3.5
        x = np.zeros_like(e)
        for t in range(1, 252):
            x[:, t] = phi * x[:, t - 1] + e[:, t]
        _, zz = sy.variance_ratio_stats(x, 5)
        assert (np.median(zz) > bound) if phi > 0 else (np.median(zz) < bound)


def test_the_rolling_statistic_uses_the_window_ending_on_each_date():
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(0, 0.01, 400), index=pd.bdate_range("2018-01-01", periods=400))
    s.iloc[0] = np.nan
    vr, z = sy.rolling_variance_ratio(s, 100, 4)
    t = 250
    ref = lm_reference(s.iloc[t - 99:t + 1].tolist(), 4)
    assert (vr.iloc[t], z.iloc[t]) == pytest.approx(ref, rel=1e-10)
    assert vr.iloc[:100].isna().all() and vr.iloc[100:].notna().all()                # the window ending on date 99 holds the missing first return


def test_the_switch_follows_persistent_moves_and_fades_reversing_ones():
    n = 1500
    rng = np.random.default_rng(5)
    e = rng.normal(0, 0.01, (2, n))
    ret = np.zeros((2, n))
    for t in range(1, n):
        ret[0, t] = 0.35 * ret[0, t - 1] + e[0, t]
        ret[1, t] = -0.35 * ret[1, t - 1] + e[1, t]
    b = bundle_from_prices(pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=1)).T, index=pd.bdate_range("2012-01-02", periods=n), columns=["TREND", "REVERT"]), asset_class={"TREND": "equity", "REVERT": "equity"}, name="vr")
    m = model("variance_ratio_switch", window=252, q=5)
    s = m.score(b)
    _, z = m.statistics(b)
    last_move = np.sign(b.prices / b.prices.shift(5) - 1.0)
    for asset in ("TREND", "REVERT"):
        sc, zz, lm = s[asset], z[asset], last_move[asset]
        live = sc.notna() & zz.notna() & lm.notna()
        persist, reverse, unsure = live & (zz >= 1.645), live & (zz <= -1.645), live & (zz.abs() < 1.645)
        assert (sc[persist] == lm[persist]).all() and (sc[reverse] == -lm[reverse]).all() and (sc[unsure] == 0).all()   # follow, fade, or stay out
    assert z["TREND"].median() > 1.645 > -1.645 > z["REVERT"].median()
    assert check_causality(model("variance_ratio_switch", window=120, q=4), b)["ok"]


def test_the_smooth_switch_scales_with_the_statistic_and_the_parameters_are_checked():
    b = bundle({"A": (0.0003, 0.01)})
    hard, smooth = model("variance_ratio_switch").score(b), model("variance_ratio_switch", smooth=True).score(b)
    live = smooth.notna() & hard.notna()
    assert smooth.abs().max().max() <= 1.0 and ((smooth.abs() <= hard.abs() + 1e-12) | ~live).all().all() and ((smooth == 0) | (hard != 0) | ~live).all().all()
    for kwargs in (dict(q=1), dict(window=10, q=5), dict(z_crit=-1.0)):
        with pytest.raises(ValueError):
            model("variance_ratio_switch", **kwargs)


# ------------------------------------------------------------------------------------------------------------------ Connors
def rsi_reference(x, n):
    a, ag, al, out = 1.0 / n, None, None, [math.nan]
    for i in range(1, len(x)):
        d = x[i] - x[i - 1]
        g, l_ = max(d, 0.0), max(-d, 0.0)
        ag, al = (g, l_) if ag is None else ((1 - a) * ag + a * g, (1 - a) * al + a * l_)
        if i < n:
            out.append(math.nan)
        elif al > 0:
            out.append(100.0 - 100.0 / (1.0 + ag / al))
        else:
            out.append(100.0 if ag > 0 else 50.0)
    return np.array(out)


def test_rsi_is_wilders_recursion():
    rng = np.random.default_rng(1)
    x = 100 + np.cumsum(rng.normal(0, 1, 120))
    got = sy.rsi(pd.DataFrame({"X": x}), 3)["X"].to_numpy()
    assert np.allclose(got, rsi_reference(x, 3), equal_nan=True, rtol=1e-10)
    assert np.isnan(got[:3]).all() and not np.isnan(got[3:]).any()


def test_rsi_of_a_series_that_only_rises_is_100_and_of_a_flat_one_is_50():
    up = sy.rsi(pd.DataFrame({"X": np.arange(1.0, 30.0)}), 3)["X"]
    flat = sy.rsi(pd.DataFrame({"X": np.full(30, 7.0)}), 3)["X"]
    assert (up.dropna() == 100.0).all() and (flat.dropna() == 50.0).all()


def test_the_streak_counts_consecutive_closes_in_one_direction_and_resets_on_a_tie():
    close = pd.DataFrame({"X": [10.0, 11.0, 12.0, 12.0, 11.0, 10.0, 9.0, 10.0, 10.5, 10.5]})
    got = sy.streak(close)["X"].tolist()
    assert math.isnan(got[0]) and got[1:] == [1.0, 2.0, 0.0, -1.0, -2.0, -3.0, 1.0, 2.0, 0.0]


def test_the_percent_rank_is_the_share_of_the_previous_window_below_todays_value():
    x = pd.DataFrame({"X": [3.0, 1.0, 2.0, 5.0, 4.0, 0.0, 6.0]})
    got = sy.percent_rank(x, 3)["X"].to_numpy()
    expected = [math.nan, math.nan, math.nan, 100.0, 200.0 / 3.0, 0.0, 100.0]       # 5 beats {3,1,2}; 4 beats two of {1,2,5}; 0 beats none of {2,5,4}; 6 beats all of {5,4,0}
    assert np.allclose(got, expected, equal_nan=True)


def test_connors_rsi_is_the_average_of_its_three_parts():
    rng = np.random.default_rng(4)
    close = pd.DataFrame({"X": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))})
    c = sy.connors_rsi(close, 3, 2, 100)
    parts = sy.rsi(close, 3) + sy.rsi(sy.streak(close), 2) + sy.percent_rank(close / close.shift(1) - 1.0, 100)
    assert np.allclose(c.dropna(), (parts / 3.0).dropna()) and c.dropna().min().min() >= 0 and c.dropna().max().max() <= 100


class _Scripted(sy.ConnorsRSI):
    """The same rule with the indicator replaced by a hand-written path, to test only the entry and exit logic."""

    def __init__(self, values, **kw):
        super().__init__(**kw)
        self.values = values

    def indicator(self, data):
        return pd.DataFrame({c: self.values for c in data.prices.columns}, index=data.index)


def test_connors_rule_enters_below_the_threshold_in_an_uptrend_and_leaves_above_the_exit_level():
    n = 260
    idx = pd.bdate_range("2015-01-02", periods=n)
    b = bundle_from_prices(pd.DataFrame({"X": 100 * np.exp(0.002 * np.arange(n))}, index=idx), asset_class={"X": "equity"}, name="cr")
    crsi = np.full(n, 50.0)
    crsi[230], crsi[231], crsi[232], crsi[233], crsi[240] = 8.0, 30.0, 72.0, 5.0, 75.0
    s = _Scripted(pd.Series(crsi, index=idx)).score(b)["X"]
    assert s.iloc[229] == 0 and s.iloc[230] == 1 and s.iloc[231] == 1                  # entered at 8, held at 30
    assert s.iloc[232] == 0 and s.iloc[233] == 1 and s.iloc[239] == 1 and s.iloc[240] == 0
    assert s.iloc[150:199].notna().all() and (s.iloc[150:199] == 0).all()              # no 200-day average yet, so no up-trend and no entry


def test_connors_rule_does_not_buy_below_its_trend_average_and_can_sell_short_when_asked():
    n = 260
    idx = pd.bdate_range("2015-01-02", periods=n)
    down = bundle_from_prices(pd.DataFrame({"X": 100 * np.exp(-0.002 * np.arange(n))}, index=idx), asset_class={"X": "equity"}, name="cd")
    crsi = np.full(n, 50.0)
    crsi[240], crsi[245] = 5.0, 95.0
    series = pd.Series(crsi, index=idx)
    assert (_Scripted(series).score(down)["X"].iloc[210:] == 0).all()
    short = _Scripted(series, allow_short=True).score(down)["X"]
    assert short.iloc[244] == 0 and short.iloc[245] == -1 and short.iloc[255] == -1
    assert check_causality(model("connors_rsi"), bundle({"A": (0.0004, 0.012), "B": (0.0, 0.01)}))["ok"]


@pytest.mark.parametrize("kwargs", [dict(rsi_len=1), dict(buy_below=80.0), dict(exit_above=5.0), dict(cover_below=95.0, sell_above=90.0), dict(trend_len=2)])
def test_connors_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        model("connors_rsi", **kwargs)


# ------------------------------------------------------------------------------------------------------------------ Double 7s
def test_double_seven_matches_a_plain_loop():
    rng = np.random.default_rng(9)
    n = 600
    idx = pd.bdate_range("2014-01-02", periods=n)
    price = 100 * np.cumprod(1 + 0.0004 + rng.normal(0, 0.012, n))
    b = bundle_from_prices(pd.DataFrame({"X": price}, index=idx), asset_class={"X": "equity"}, name="d7")
    got = model("double_seven").score(b)["X"].to_numpy()
    held, ref = 0, np.full(n, np.nan)
    for t in range(199, n):
        ma = price[t - 199:t + 1].mean()
        lo, hi = price[t - 6:t + 1].min(), price[t - 6:t + 1].max()
        if held == 1 and price[t] >= hi:
            held = 0
        if held == 0 and price[t] <= lo and price[t] > ma:
            held = 1
        ref[t] = held
    assert np.allclose(got[199:], ref[199:]) and np.isnan(got[:199]).all()
    assert check_causality(model("double_seven"), bundle({"A": (0.0004, 0.012)}))["ok"]
    with pytest.raises(ValueError):
        model("double_seven", window=1)


# ------------------------------------------------------------------------------------------------------------------ skewness
def test_skew_premium_is_minus_the_rolling_sample_skewness(trending):
    s = model("skew_premium", window=252).score(trending)
    r = trending.returns["UP"].iloc[1200 - 251:1201]
    mean = r.mean()
    n = 252
    m2, m3 = ((r - mean) ** 2).mean(), ((r - mean) ** 3).mean()
    g1 = m3 / m2 ** 1.5
    adjusted = g1 * math.sqrt(n * (n - 1)) / (n - 2)                                  # the bias-adjusted sample skewness
    assert s["UP"].iloc[1200] == pytest.approx(-adjusted, rel=1e-8)
    assert s.iloc[:250].isna().all().all()


def test_skew_premium_prefers_the_negatively_skewed_asset_and_is_causal():
    n = 800
    rng = np.random.default_rng(12)
    crash = rng.normal(0.0004, 0.006, n)
    crash[rng.random(n) < 0.01] -= 0.08
    lottery = rng.normal(0.0, 0.006, n)
    lottery[rng.random(n) < 0.01] += 0.08
    idx = pd.bdate_range("2014-01-02", periods=n)
    b = bundle_from_prices(pd.DataFrame({"CRASH": 100 * np.cumprod(1 + crash), "LOTTERY": 100 * np.cumprod(1 + lottery)}, index=idx), asset_class={"CRASH": "equity", "LOTTERY": "equity"}, name="sk")
    s = model("skew_premium", window=252).score(b).dropna()
    assert (s["CRASH"] > s["LOTTERY"]).mean() > 0.95
    assert check_causality(model("skew_premium", window=120), b)["ok"]
    with pytest.raises(ValueError):
        model("skew_premium", window=5)


# ------------------------------------------------------------------------------------------------------------------ CPPI / TIPP
def test_cppi_follows_the_hand_worked_account():
    r = np.array([np.nan, 0.10, -0.05, 0.02, -0.30, 0.10])
    e = sy.cppi_exposure(r, multiplier=4.0, floor=0.8, max_exposure=1.0, cash_daily=0.0, ratchet=False)
    # day 1 (the first return): value 1, floor 0.8, cushion 0.2, exposure 4 x 0.2 / 1 = 0.8
    # day 2: value 1 + 0.8 x -0.05 = 0.96, cushion 0.16, exposure 4 x 0.16 / 0.96
    # day 3: value 0.96 x (1 + 0.666667 x 0.02) = 0.9728, cushion 0.1728, exposure 4 x 0.1728 / 0.9728
    # day 4: value 0.9728 x (1 + 0.710526 x -0.30) = 0.76544, below the floor: nothing at risk, and nothing on day 5 either
    assert math.isnan(e[0])
    assert e[1:] == pytest.approx([0.8, 0.64 / 0.96, 0.6912 / 0.9728, 0.0, 0.0], rel=1e-12)


def test_cppi_account_never_falls_through_the_floor_unless_the_asset_gaps_by_more_than_one_over_the_multiplier():
    rng = np.random.default_rng(6)
    r = np.clip(rng.normal(0.0003, 0.012, 3000), -0.2, 0.2)
    m, floor = 4.0, 0.85
    e = sy.cppi_exposure(r, m, floor, 1.0, 0.0, False)
    value = 1.0
    for t in range(1, len(r)):
        value *= 1.0 + e[t - 1] * r[t]
        assert value >= floor - 1e-12 or r[t] < -1.0 / m                              # only a daily loss beyond 1/m can breach the floor
    assert 0.0 <= np.nanmin(e) and np.nanmax(e) <= 1.0


def test_cppi_is_flat_after_the_floor_is_breached_and_tipp_ratchets_the_floor_up():
    crash = np.array([0.0] + [-0.30] * 3 + [0.05] * 5)
    assert sy.cppi_exposure(crash, 4.0, 0.8, 1.0, 0.0, False)[-1] == 0.0
    up_then_down = np.array([0.0] + [0.05] * 20 + [-0.04] * 12)
    plain = sy.cppi_exposure(up_then_down, 4.0, 0.8, 1.0, 0.0, False)
    ratchet = sy.cppi_exposure(up_then_down, 4.0, 0.8, 1.0, 0.0, True)
    assert plain[20] == pytest.approx(1.0) and ratchet[20] <= plain[20]
    assert ratchet[-1] < plain[-1]                                                     # the higher floor leaves a smaller cushion after the fall


def test_cppi_cash_rate_grows_the_floor_and_the_score_wraps_the_account(trending):
    flat = np.zeros(300)
    flat[0] = np.nan
    with_cash = sy.cppi_exposure(flat, 4.0, 0.9, 1.0, 0.0001, False)
    assert with_cash[1] == pytest.approx(0.4)                                          # 4 x (1 - 0.9)
    assert (np.diff(with_cash[1:]) <= 1e-15).all() and with_cash[-1] < with_cash[1]    # the floor grows at the cash rate, the account at (1 - exposure) of it: the cushion shrinks
    m = model("cppi", multiplier=3.0, floor=0.9, cash_rate=0.02)
    s = m.score(trending)
    expected = sy.cppi_exposure(trending.returns["UP"].to_numpy(dtype=float), 3.0, 0.9, 1.0, 1.02 ** (1 / 252) - 1.0, False)
    live = s["UP"].notna().to_numpy()
    assert live.sum() > 1500 and np.allclose(s["UP"].to_numpy()[live], expected[live])
    assert check_causality(model("cppi"), trending)["ok"]
    for kwargs in (dict(multiplier=0.0), dict(floor=1.0), dict(floor=0.0), dict(max_exposure=0.0), dict(cash_rate=-0.01)):
        with pytest.raises(ValueError):
            model("cppi", **kwargs)


def test_every_new_rule_is_registered_with_a_family_and_a_description():
    for name in ("carver_ewmac", "clenow_momentum", "tstat_momentum", "variance_ratio_switch", "connors_rsi", "double_seven", "skew_premium", "cppi"):
        entry = next(e for e in MODELS.entries() if e.name == name)
        assert entry.family and len(entry.description) > 40
