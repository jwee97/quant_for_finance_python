"""The second strategy batch: each rule does what its source says on constructed data, uses only the past, and rejects bad parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, bundle_from_prices, load_library
from src.strategies._common import month_end_flags
from src.strategies.trend import _kama

load_library()
N = 1500
IDX = pd.bdate_range("2012-01-02", periods=N)


def bundle(paths: dict, classes: dict | None = None, volume: bool = False, high_low: bool = False, macro: pd.DataFrame | None = None, n: int = N):
    """Prices from per-asset daily drift/noise specs: value = (drift, noise) or a ready-made array."""
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2012-01-02", periods=n)
    cols = {}
    for name, spec in paths.items():
        if isinstance(spec, tuple):
            drift, noise = spec
            cols[name] = 100 * np.cumprod(1 + drift + rng.normal(0, noise, n))
        else:
            cols[name] = np.asarray(spec, dtype=float)
    prices = pd.DataFrame(cols, index=idx)
    kwargs = {}
    if high_low:
        kwargs.update(high=prices * 1.004, low=prices * 0.996)
    if volume:
        kwargs["volume"] = pd.DataFrame(2e6, index=idx, columns=prices.columns)
    return bundle_from_prices(prices, asset_class=classes or {c: "equity" for c in prices.columns}, macro=macro, name="v2", **kwargs)


def model(name, **params):
    return MODELS.create(name, **params)


@pytest.fixture(scope="module")
def trending():
    return bundle({"UP": (0.0010, 0.004), "DOWN": (-0.0008, 0.004), "FLAT": (0.0, 0.010)}, high_low=True)


# ----------------------------------------------------------------------------------------------------------- trend
def test_tsmom_follows_the_sign_of_the_year_and_sizes_by_volatility(trending):
    s = model("tsmom").score(trending).dropna()
    last = s.iloc[-1]
    assert last["UP"] > 0 > last["DOWN"] and (s.abs() <= 4.0 + 1e-9).all().all()
    ret = trending.prices / trending.prices.shift(252) - 1.0
    assert (np.sign(s) == np.sign(ret.reindex(s.index))).all().all()
    calm, wild = model("tsmom").score(bundle({"A": (0.001, 0.003), "B": (0.001, 0.02)})).dropna().iloc[-1]
    assert calm > wild > 0                                              # the same trend in a calmer asset gets the larger position


def test_tsmom_multi_is_the_average_of_the_horizon_signs(trending):
    s = model("tsmom_multi", horizons=(21, 63, 252)).score(trending).dropna()
    expected = sum(np.sign(trending.prices / trending.prices.shift(h) - 1.0) for h in (21, 63, 252)).div(3).reindex(s.index)
    assert np.allclose(s, expected) and s.min().min() >= -1 and s.max().max() <= 1


def test_ma_ensemble_is_bounded_and_agrees_with_the_trend(trending):
    s = model("ma_ensemble").score(trending).dropna()
    assert s["UP"].iloc[-300:].mean() > 0.05 and s["DOWN"].iloc[-300:].mean() < -0.05 and s.abs().max().max() < 1.5
    with pytest.raises(ValueError):
        model("ma_ensemble", pairs=((30, 10),))


def test_breakout_ensemble_sits_in_minus_one_to_one_and_is_high_in_a_trend(trending):
    s = model("breakout_ensemble").score(trending).dropna()
    assert s.abs().max().max() <= 1.0 + 1e-9 and s["UP"].iloc[-200:].mean() > 0.3 and s["DOWN"].iloc[-200:].mean() < -0.3


def test_kama_matches_a_plain_loop_and_tracks_a_clean_trend_closely():
    rng = np.random.default_rng(1)
    p = 100 + np.cumsum(rng.normal(0, 1, (300, 1)), axis=0)
    out = _kama(p, 10, 2, 30)[:, 0]
    k, ref = np.nan, np.full(300, np.nan)
    for t in range(10, 300):
        er = abs(p[t, 0] - p[t - 10, 0]) / np.abs(np.diff(p[t - 10:t + 1, 0])).sum()
        sc = (er * (2 / 3 - 2 / 31) + 2 / 31) ** 2
        k = p[t, 0] if not np.isfinite(k) else k
        k += sc * (p[t, 0] - k)
        ref[t] = k
    assert np.allclose(out[10:], ref[10:])
    line = np.linspace(100, 200, 200)[:, None]
    follow = _kama(line, 10, 2, 30)[:, 0]
    assert abs(follow[-1] - line[-1, 0]) < abs(line[-1, 0] - np.mean(line[-31:, 0]))        # in a perfect trend it is faster than a 30-day average


def test_adx_trend_trades_trends_and_stands_aside_in_noise(trending):
    m = model("adx_trend")
    s = m.score(trending).dropna()
    assert s["UP"].iloc[-300:].mean() > 0.3 and s["DOWN"].iloc[-300:].mean() < -0.15
    assert s["FLAT"].abs().mean() < s["UP"].abs().mean()
    plus, minus, adx = m.indicators(trending)
    assert ((adx.dropna() >= 0) & (adx.dropna() <= 100)).all().all()


def test_supertrend_flips_only_when_the_close_crosses_its_band():
    up = np.linspace(100, 160, 120)
    path = np.concatenate([up, np.linspace(160, 90, 60), np.linspace(90, 150, 80)])
    b = bundle({"X": path, "Y": path * 1.1}, high_low=True, n=len(path))
    s = model("supertrend").score(b)["X"]
    assert s.iloc[100] == 1.0 and s.iloc[170] == -1.0 and s.iloc[-1] == 1.0
    assert set(np.unique(s.dropna())) <= {-1.0, 1.0} and (s.dropna().diff().abs() > 0).sum() <= 6               # a handful of flips, not daily noise


def test_keltner_breakout_enters_on_the_break_and_holds_until_the_middle_line():
    path = np.concatenate([np.full(60, 100.0) + np.sin(np.arange(60)), np.linspace(100, 130, 40), np.linspace(130, 110, 40)])
    b = bundle({"X": path, "Y": path}, high_low=True, n=len(path))
    s = model("keltner_breakout").score(b)["X"]
    assert s.iloc[75] == 1.0 and s.iloc[-1] in (0.0, -1.0) and 1.0 in set(s.dropna())


def test_macd_ichimoku_and_the_52_week_high_in_a_trend(trending):
    macd = model("macd_trend").score(trending).dropna()
    assert macd["UP"].iloc[-200:].mean() > 0.1 > -0.1 > macd["DOWN"].iloc[-200:].mean()
    hist = model("macd_trend", use="histogram").score(trending).dropna()
    assert hist.abs().max().max() <= 1.0 and not np.allclose(hist, macd.reindex(hist.index))
    cloud = model("ichimoku_trend").score(trending).dropna()
    assert cloud["UP"].iloc[-100:].mean() > 0.5 and cloud["DOWN"].iloc[-100:].mean() < -0.5 and set(np.unique(cloud)) <= {-1.0, 0.0, 1.0}
    high = model("high_52w").score(trending).dropna()
    assert (high <= 1e-12).all().all() and high["UP"].iloc[-200:].mean() > high["DOWN"].iloc[-200:].mean()
    with pytest.raises(ValueError):
        model("ichimoku_trend", tenkan=30, kijun=26)


def test_13612w_momentum_matches_the_published_formula(trending):
    s = model("momentum_13612w").score(trending)
    p, d = trending.prices, trending.index[-1]
    r = lambda k: p.loc[d] / p.shift(21 * k).loc[d] - 1.0                  # noqa: E731
    assert np.allclose(s.loc[d], (12 * r(1) + 4 * r(3) + 2 * r(6) + r(12)) / 19.0)


def test_smooth_momentum_prefers_the_gain_that_came_in_many_small_steps():
    n = 600
    smooth = 100 * np.cumprod(np.full(n, 1.0008))
    jumpy_returns = np.zeros(n)
    jumpy_returns[::40] = 0.0008 * 40 * 0.98
    jumpy = 100 * np.cumprod(1 + jumpy_returns)
    s = model("smooth_momentum").score(bundle({"SMOOTH": smooth, "JUMPY": jumpy}, n=n)).dropna()
    final_returns = (smooth[-22] / smooth[-253], jumpy[-22] / jumpy[-253])
    assert abs(final_returns[0] - final_returns[1]) < 0.06 and s["SMOOTH"].iloc[-1] > s["JUMPY"].iloc[-1]


def test_accelerating_dual_momentum_holds_only_assets_that_beat_cash():
    b = bundle({"SHY": (0.0001, 0.001), "A": (0.0012, 0.005), "B": (0.0006, 0.005), "C": (-0.001, 0.005), "D": (0.0002, 0.005)}, classes={"SHY": "rates", "A": "equity", "B": "equity", "C": "equity", "D": "equity"})
    s = model("accelerating_dual_momentum", top_k=2).score(b).dropna()
    assert (s.sum(axis=1) <= 2).all() and s["C"].iloc[-1] == 0.0 and s["SHY"].iloc[-1] == 0.0 and s["A"].iloc[-1] == 1.0


def test_residual_momentum_ranks_the_asset_with_its_own_trend_above_one_that_only_rides_the_market():
    rng = np.random.default_rng(2)
    n = 1500
    market = rng.normal(0.0004, 0.01, n)
    rets = {f"M{i}": 1.0 * market + rng.normal(0, 0.003, n) for i in range(4)}
    rets["ALPHA"] = market + rng.normal(0, 0.003, n) + 0.0012
    prices = {k: 100 * np.cumprod(1 + v) for k, v in rets.items()}
    s = model("residual_momentum").score(bundle(prices, n=n)).dropna()
    assert s.iloc[-1].idxmax() == "ALPHA" and s["ALPHA"].iloc[-200:].mean() > 1.0


# ------------------------------------------------------------------------------------------------------- reversion
def _pullback_path():
    """A rising market with a three-day sharp drop in the middle, then recovery."""
    up = np.linspace(100, 220, 330)
    drop = up[-1] * np.array([0.97, 0.94, 0.91])
    recover = np.linspace(drop[-1], 250, 60)
    return np.concatenate([up, drop, recover])


def test_rsi2_buys_the_deep_pullback_in_an_up_trend_and_exits_on_the_bounce():
    path = _pullback_path()
    s = model("rsi2").score(bundle({"X": path, "Y": path}, n=len(path)))["X"]
    drop_day = 330 + 2
    assert s.iloc[drop_day] == 1.0 and s.iloc[-1] == 0.0 and s.iloc[300] == 0.0
    assert set(np.unique(s.dropna())) <= {0.0, 1.0}                                  # long-only by default
    down = np.linspace(220, 100, 330)
    mirrored = np.concatenate([down, down[-1] * np.array([1.03, 1.06, 1.09]), np.linspace(down[-1] * 1.09, 60, 60)])
    both = model("rsi2", shorts=True).score(bundle({"X": mirrored, "Y": mirrored}, n=len(mirrored)))["X"]
    assert both.iloc[332] == -1.0 and model("rsi2").score(bundle({"X": mirrored, "Y": mirrored}, n=len(mirrored)))["X"].iloc[332] == 0.0


def test_ibs_bollinger_stochastic_and_streak_rules_buy_the_stretch_in_an_up_trend():
    path = _pullback_path()
    b = bundle({"X": path, "Y": path}, n=len(path), high_low=True)
    for name in ("bollinger_reversion", "stochastic_reversion", "consecutive_down"):
        s = model(name).score(b)["X"].dropna()
        assert s.loc[b.index[332]:b.index[334]].max() == 1.0 and s.iloc[-1] == 0.0, name
    ibs = model("ibs_reversion").score(bundle({"X": path, "Y": path}, n=len(path)))["X"].dropna()
    assert set(np.unique(ibs)) <= {0.0, 1.0}


def test_short_term_reversal_is_the_negative_of_recent_performance(trending):
    s = model("short_term_reversal", lookback=5).score(trending).dropna()
    ret = (trending.prices / trending.prices.shift(5) - 1.0).reindex(s.index)
    assert (np.sign(s) == -np.sign(ret)).all().all()


def test_ou_reversion_acts_on_a_mean_reverting_asset_not_on_a_random_walk():
    rng = np.random.default_rng(4)
    n = 1500
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.85 * x[t - 1] + rng.normal(0, 0.02)
    ou = 100 * np.exp(x)
    walks = {f"RW{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for i in range(6)}
    b = bundle({"OU": ou, **walks}, n=n)
    m = model("ou_reversion")
    half_life, t_stat = m.fit(b)
    assert half_life["OU"].iloc[-500:].median() < 15 and t_stat["OU"].iloc[-500:].median() < -3.2
    s = m.score(b).dropna()
    active = (s != 0).mean()
    assert active["OU"] > 0.7 and active[list(walks)].mean() < 0.15                       # the unit-root test keeps it off random walks
    with pytest.raises(ValueError):
        model("ou_reversion", df_critical=1.0)


def test_range_reversion_switches_off_in_a_strong_trend(trending):
    s = model("range_reversion").score(trending).dropna()
    assert s["UP"].abs().mean() < s["FLAT"].abs().mean()


# --------------------------------------------------------------------------------------------------------- seasonal
def test_turn_of_month_flags_the_last_day_and_the_first_three_of_each_month():
    m = model("turn_of_month")
    flags = m.flags(IDX)
    by_month = flags.groupby(IDX.to_period("M")).sum()
    assert (by_month.iloc[1:-1] == 4).all()
    last_days = IDX[month_end_flags(IDX).to_numpy()]
    assert flags.loc[last_days].eq(1.0).all()
    first = pd.Series(IDX, index=IDX).groupby(IDX.to_period("M")).first()
    assert flags.loc[first.to_numpy()].eq(1.0).all()
    classes = {"E1": "equity", "R1": "rates"}
    s = m.score(bundle({"E1": (0.0003, 0.01), "R1": (0.0001, 0.003)}, classes=classes)).dropna()
    assert s["R1"].abs().sum() == 0 and s["E1"].sum() > 0


def test_sell_in_may_holds_november_to_april():
    s = model("sell_in_may").score(bundle({"A": (0.0003, 0.01), "B": (0.0003, 0.01)}))["A"].dropna()
    months = pd.Series(s.index.month, index=s.index)
    assert (s[months.isin([11, 12, 1, 2, 3, 4])] == 1).all() and (s[months.isin([5, 6, 7, 8, 9, 10])] == 0).all()
    assert model("sell_in_may", start_month=5, end_month=10).score(bundle({"A": (0.0003, 0.01)}))["A"].dropna().iloc[0] in (0.0, 1.0)


def test_seasonal_rank_learns_a_recurring_month_and_never_reads_the_current_month():
    n = 252 * 9
    idx = pd.bdate_range("2012-01-02", periods=n)
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.004, (n, 3))
    january = idx.month == 1
    r[january, 0] += 0.003                                           # asset 0 has a strong January every year
    prices = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=["JAN", "B", "C"])
    b = bundle_from_prices(prices, name="season")
    s = model("seasonal_rank").score(b)
    late_january = s.loc[(s.index.year == 2019) & (s.index.month == 1)]
    assert (late_january["JAN"] > late_january[["B", "C"]].max(axis=1)).all()
    shocked = prices.copy()
    this_month = (idx.year == 2019) & (idx.month == 1)
    shocked.loc[this_month, "B"] *= np.linspace(1.0, 3.0, this_month.sum())      # the current month's own return must not move its own score
    s2 = model("seasonal_rank").score(bundle_from_prices(shocked, name="season2"))
    assert np.allclose(s.loc[this_month].to_numpy(), s2.loc[this_month].to_numpy(), equal_nan=True)


# ---------------------------------------------------------------------------------------------------------- factors
def test_bab_is_beta_neutral_levered_long_low_beta_and_short_high_beta():
    rng = np.random.default_rng(6)
    n = 800
    market = rng.normal(0.0004, 0.01, n)
    betas = {"L1": 0.4, "L2": 0.6, "M1": 1.0, "H1": 1.5, "H2": 1.8}
    prices = {k: 100 * np.cumprod(1 + b * market + rng.normal(0, 0.003, n)) for k, b in betas.items()}
    bnd = bundle(prices, n=n)
    m = model("bab")
    w = m.weights(bnd).dropna(how="all")
    last = w.iloc[-1]
    assert last["L1"] > 0 and last["H2"] < 0 and last.sum() != 0
    from src.strategies.factors import _beta
    decided = bnd.index[month_end_flags(bnd.index).to_numpy()][-1]            # the weights held at the end were decided on the last month-end
    last = w.loc[decided]
    beta = _beta(bnd, m.window, m.shrink).loc[decided]
    assert abs((last * beta).sum()) < 1e-9                              # long beta 1 minus short beta 1
    long_beta, short_beta = (last.clip(lower=0) * beta).sum(), (last.clip(upper=0) * beta).sum()
    assert np.isclose(long_beta, 1.0) and np.isclose(short_beta, -1.0)


def test_low_idio_vol_max_effect_and_illiquidity_orderings():
    rng = np.random.default_rng(7)
    n = 800
    market = rng.normal(0.0004, 0.01, n)
    prices = {"CALM": 100 * np.cumprod(1 + market + rng.normal(0, 0.001, n)), "WILD": 100 * np.cumprod(1 + market + rng.normal(0, 0.02, n)),
              "MID": 100 * np.cumprod(1 + market + rng.normal(0, 0.006, n)), "MID2": 100 * np.cumprod(1 + market + rng.normal(0, 0.004, n))}
    b = bundle(prices, n=n)
    assert model("low_idio_vol").score(b).dropna().iloc[-1].idxmax() == "CALM"
    spike = np.zeros(n)
    spike[-5] = 0.15
    p2 = {"SPIKE": 100 * np.cumprod(1 + market + spike), "CALM": prices["CALM"]}
    mx = model("max_effect").score(bundle(p2, n=n)).dropna().iloc[-1]
    assert mx["SPIKE"] < mx["CALM"]
    with pytest.raises(KeyError, match="volume"):
        model("amihud_illiquidity").score(b)
    thin = bundle({"THIN": prices["WILD"], "THICK": prices["CALM"]}, n=n, volume=True)
    thin.volume["THIN"] = 1e3
    assert model("amihud_illiquidity").score(thin).dropna().iloc[-1].idxmax() == "THIN"


def test_value_momentum_blends_the_two_ranks():
    b = bundle({"A": (0.001, 0.01), "B": (-0.0005, 0.01), "C": (0.0003, 0.01), "D": (0.0, 0.01)}, n=1600)
    both, mom_only, val_only = (model("value_momentum", value_weight=w).score(b).dropna().iloc[-1] for w in (0.5, 0.0, 1.0))
    assert np.allclose(both, 0.5 * mom_only + 0.5 * val_only) and both.abs().max() <= 0.5 + 1e-9
    with pytest.raises(ValueError):
        model("value_momentum", value_lookback=100)


# ------------------------------------------------------------------------------------------------------ risk timing
def test_vol_managed_long_shrinks_exposure_when_variance_rises():
    rng = np.random.default_rng(8)
    n = 800
    calm_then_wild = np.concatenate([rng.normal(0.0004, 0.005, 500), rng.normal(0.0004, 0.025, 300)])
    b = bundle({"X": 100 * np.cumprod(1 + calm_then_wild), "Y": 100 * np.cumprod(1 + calm_then_wild)}, n=n)
    s = model("vol_managed_long").score(b)["X"]
    assert s.iloc[400:480].mean() > 1.0 > s.iloc[-50:].mean() and s.max() <= 3.0 + 1e-9 and (s.dropna() >= 0).all()


def test_vix_spike_reversion_enters_on_a_spike_and_exits_when_it_fades():
    n = 400
    idx = pd.bdate_range("2012-01-02", periods=n)
    vix = pd.Series(15.0, index=idx) + np.sin(np.arange(n)) * 0.3
    vix.iloc[250:256] = [30, 35, 33, 28, 22, 16.5]
    macro = pd.DataFrame({"VIX": vix})
    b = bundle({"E": (0.0003, 0.01), "B": (0.0001, 0.003)}, classes={"E": "equity", "B": "rates"}, macro=macro, n=n)
    s = model("vix_spike_reversion").score(b)
    assert s["E"].iloc[250] == 1.0 and s["E"].iloc[-1] == 0.0 and s["B"].abs().sum() == 0
    with pytest.raises(KeyError, match="VIX"):
        model("vix_spike_reversion").score(bundle({"E": (0.0, 0.01), "F": (0.0, 0.01)}))


def test_credit_spread_timing_goes_risk_on_when_credit_beats_rates_and_needs_both_classes():
    n = 1200
    credit = 100 * np.cumprod(np.full(n, 1.0006))
    safe = 100 * np.cumprod(np.full(n, 1.0001))
    classes = {"CR": "credit", "SF": "rates", "EQ": "equity"}
    b = bundle({"CR": credit, "SF": safe, "EQ": (0.0003, 0.01)}, classes=classes, n=n)
    s = model("credit_spread_timing").score(b).dropna()
    assert s["EQ"].iloc[-1] == s["CR"].iloc[-1] and s["SF"].iloc[-1] == pytest.approx(-0.5 * s["EQ"].iloc[-1])
    with pytest.raises(KeyError, match="credit"):
        model("credit_spread_timing").score(bundle({"A": (0.0, 0.01), "B": (0.0, 0.01)}))


# ------------------------------------------------------------------------------------------------------------ TAA
CLASSES = {"E1": "equity", "E2": "equity", "E3": "equity", "E4": "equity", "E5": "equity", "E6": "equity", "B1": "rates", "B2": "fixed_income"}


def _taa_bundle(up: int, n: int = 600):
    """Six risky assets, ``up`` of which trend up; two safe assets with a gentle uptrend."""
    paths = {f"E{i + 1}": (0.0010 if i < up else -0.0010, 0.0015) for i in range(6)}
    paths.update({"B1": (0.0003, 0.0008), "B2": (0.0002, 0.0008)})
    return bundle(paths, classes=CLASSES, n=n)


def _last_row(model_name, b, **params):
    w = model(model_name, **params).weights(b)
    return w.dropna(how="all").iloc[-1]


def test_faber_holds_equal_weights_in_assets_above_their_10_month_average_and_cash_otherwise():
    w = _last_row("faber_gtaa", _taa_bundle(up=4))
    held = w[w > 0]
    assert len(held) == 6 and np.allclose(held, 1 / 8) and w.sum() == pytest.approx(6 / 8) and (w[["E5", "E6"]] == 0).all()


@pytest.mark.parametrize("up,protection", [(6, 1), (4, 1), (3, 1), (2, 2), (0, 0)])
def test_paa_bond_fraction_follows_the_published_formula(up, protection):
    b = _taa_bundle(up=up)
    risky, safe = tuple(f"E{i}" for i in range(1, 7)), ("B1", "B2")
    w = _last_row("paa", b, protection=protection, top=6, risky=risky, safe=safe)
    n_total, n1 = 6, protection * 6 / 4
    bond_fraction = min(1.0, (n_total - up) / (n_total - n1))
    assert w[list(safe)].sum() == pytest.approx(bond_fraction) and w[list(risky)].sum() == pytest.approx(1 - bond_fraction) and w.sum() == pytest.approx(1.0)
    assert w[safe[0]] >= w[safe[1]]                                        # the safe fraction goes to the stronger safe asset
    with pytest.raises(ValueError):
        model("paa", protection=3)


def test_vaa_goes_all_in_on_one_asset_and_switches_to_defence_when_any_offensive_asset_turns():
    risky, safe = ("E1", "E2", "E3", "E4"), ("B1", "B2")
    on = _last_row("vaa", _taa_bundle(up=6), offensive=risky, defensive=safe)
    assert on.max() == 1.0 and on[list(risky)].sum() == 1.0
    paths = {"E1": (0.001, 0.0015), "E2": (0.001, 0.0015), "E3": (0.001, 0.0015), "E4": (-0.001, 0.0015), "B1": (0.0003, 0.0008), "B2": (0.0001, 0.0008)}
    off = _last_row("vaa", bundle(paths, classes={"E1": "equity", "E2": "equity", "E3": "equity", "E4": "equity", "B1": "rates", "B2": "rates"}, n=600), offensive=risky, defensive=safe)
    assert off["B1"] == 1.0 and off.sum() == 1.0


@pytest.mark.parametrize("bad,expected_cash", [(0, 0.0), (1, 0.5), (2, 1.0)])
def test_daa_moves_a_share_to_safety_for_each_canary_with_negative_momentum(bad, expected_cash):
    paths = {f"E{i}": (0.001, 0.0015) for i in range(3, 7)}
    paths["E1"] = (-0.001 if bad >= 1 else 0.001, 0.0015)
    paths["E2"] = (-0.001 if bad >= 2 else 0.001, 0.0015)
    paths.update({"B1": (0.0003, 0.0008), "B2": (0.0002, 0.0008)})
    b = bundle(paths, classes=CLASSES, n=600)
    w = _last_row("daa", b, canary=("E1", "E2"), offensive=tuple(f"E{i}" for i in range(1, 7)), defensive=("B1", "B2"), top=3, breadth=2)
    assert w[["B1", "B2"]].sum() == pytest.approx(expected_cash) and w.sum() == pytest.approx(1.0)


def test_adaptive_asset_allocation_weights_the_top_assets_by_inverse_volatility():
    paths = {"A": (0.0012, 0.002), "B": (0.0010, 0.008), "C": (0.0008, 0.004), "D": (-0.0005, 0.004), "E": (-0.001, 0.004)}
    b = bundle(paths, classes={k: "equity" for k in paths}, n=600)
    w = _last_row("adaptive_asset_allocation", b, top_k=3)
    assert w.sum() == pytest.approx(1.0) and (w > 0).sum() == 3 and w[["D", "E"]].sum() == 0
    assert w["A"] > w["B"] and np.isclose(w["A"] / w["B"], b.returns["B"].rolling(63).std().iloc[-1] / b.returns["A"].rolling(63).std().iloc[-1], rtol=0.5)


@pytest.mark.parametrize("preset,expected", [("60_40", {"E1": 0.6, "B1": 0.4}), ("permanent", {"E1": 0.25, "B2": 0.25, "C1": 0.25, "B1": 0.25}),
                                              ("bogleheads", {"E1": 0.4, "E2": 0.2, "B3": 0.4})])
def test_model_portfolios_hold_their_published_weights(preset, expected):
    names = {"60_40": {"SPY": "E1", "IEF": "B1"}, "permanent": {"SPY": "E1", "TLT": "B2", "GLD": "C1", "SHY": "B1"}, "bogleheads": {"SPY": "E1", "EFA": "E2", "AGG": "B3"}}[preset]
    cols = list(names)
    b = bundle({c: (0.0003, 0.005) for c in cols}, classes={"SPY": "equity", "EFA": "equity", "IEF": "rates", "TLT": "rates", "SHY": "rates", "GLD": "commodity", "AGG": "fixed_income"}, n=300)
    w = _last_row("model_portfolio", b, preset=preset)
    target = {t: expected[v] for t, v in names.items()}
    assert {k: round(v, 6) for k, v in w[w > 0].items()} == target and w.sum() == pytest.approx(1.0)


def test_model_portfolio_falls_back_to_classes_and_renormalises_when_an_asset_is_not_yet_investable():
    classes = {"E1": "equity", "E2": "equity", "B1": "rates"}
    b = bundle({"E1": (0.0003, 0.005), "E2": (0.0003, 0.005), "B1": (0.0001, 0.002)}, classes=classes, n=400)
    w = _last_row("model_portfolio", b, preset="60_40")
    assert w["E1"] == pytest.approx(0.3) and w["E2"] == pytest.approx(0.3) and w["B1"] == pytest.approx(0.4)
    late = b.prices.copy()
    late.loc[late.index[:300], "B1"] = np.nan
    b2 = bundle_from_prices(late, asset_class=classes, name="late", min_history=20)
    early = model("model_portfolio", preset="60_40").weights(b2).dropna(how="all")
    first = early.iloc[0]
    assert first["B1"] == 0 and first[["E1", "E2"]].sum() == pytest.approx(1.0)                  # bonds not yet investable: the bucket weights renormalise
    eq = _last_row("model_portfolio", b, preset="equal_class")
    assert eq["B1"] == pytest.approx(0.5) and eq[["E1", "E2"]].sum() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        model("model_portfolio", preset="nonsense")


# -------------------------------------------------------------------------------------------- invalid parameters
@pytest.mark.parametrize("name,params", [
    ("tsmom", {"lookback": 2}), ("tsmom", {"target_vol": 0}), ("tsmom_multi", {"horizons": ()}), ("breakout_ensemble", {"lookbacks": ()}), ("kama_trend", {"fast": 30, "slow": 2}),
    ("adx_trend", {"window": 1}), ("supertrend", {"multiplier": 0}), ("keltner_breakout", {"k": 0}), ("macd_trend", {"fast": 30, "slow": 12}), ("high_52w", {"window": 3}),
    ("smooth_momentum", {"lookback": 30, "skip": 21}), ("momentum_13612w", {"month": 1}), ("accelerating_dual_momentum", {"top_k": 0}), ("residual_momentum", {"beta_window": 10}),
    ("rsi2", {"entry": 70}), ("ibs_reversion", {"entry": 0.9, "exit": 0.5}), ("bollinger_reversion", {"k": 0}), ("stochastic_reversion", {"entry": 60, "exit": 50}),
    ("consecutive_down", {"days": 1}), ("short_term_reversal", {"lookback": 0}), ("ou_reversion", {"window": 10}), ("range_reversion", {"adx_max": 0}),
    ("turn_of_month", {"before": 0, "after": 0}), ("sell_in_may", {"start_month": 13}), ("seasonal_rank", {"min_years": 1}), ("bab", {"shrink": 2}), ("low_idio_vol", {"window": 5}),
    ("max_effect", {"window": 2}), ("amihud_illiquidity", {"window": 2}), ("vol_managed_long", {"max_scale": 0}), ("vix_spike_reversion", {"entry_z": 0}),
    ("credit_spread_timing", {"window": 2}), ("faber_gtaa", {"months": 1}), ("daa", {"breadth": 0}), ("adaptive_asset_allocation", {"top_k": 0}),
])
def test_invalid_parameters_are_rejected(name, params):
    with pytest.raises(ValueError):
        model(name, **params)


# ------------------------------------------------------------------------------------------------ rebalance hints
def test_fast_rules_declare_their_rebalance_frequency_and_the_spec_can_override_it():
    from src.framework import PipelineSpec
    from src.framework.pipeline import _engine
    from src.utils.config import load_config

    config = load_config()
    spec = PipelineSpec.from_dict({"models": [{"name": "rsi2"}]})
    assert _engine(config, spec, [model("rsi2")]).rebalance == "daily"
    assert _engine(config, spec, [model("rsi2"), model("tsmom")]).rebalance == "daily"                  # the fastest hint wins
    assert _engine(config, spec, [model("faber_gtaa")]).rebalance == config.get("backtest.engine.rebalance")
    assert _engine(config, PipelineSpec.from_dict({"models": [], "execution": {"rebalance": "monthly"}}), [model("rsi2")]).rebalance == "monthly"


def test_a_calendar_rule_actually_trades_when_run_through_the_pipeline():
    from src.framework import Pipeline
    from src.utils.config import load_config

    b = bundle({"E1": (0.0004, 0.01), "E2": (0.0004, 0.01), "E3": (0.0004, 0.01), "E4": (0.0004, 0.01), "E5": (0.0004, 0.01), "B1": (0.0001, 0.003)},
               classes={"E1": "equity", "E2": "equity", "E3": "equity", "E4": "equity", "E5": "equity", "B1": "rates"}, n=1200)
    spec = {"models": [{"name": "turn_of_month"}], "allocation": {"allocator": "score_stack"}, "evaluation": {"causality": False, "benchmarks": []}}
    out = Pipeline(spec, load_config(), b).run(validate=False)
    assert out.metrics["ann_vol"] > 0.01 and out.metrics["ann_turnover"] > 2
