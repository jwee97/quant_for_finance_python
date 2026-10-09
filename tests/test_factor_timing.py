"""Factor timing: the premium forecast against hand arithmetic and against the past only, the calendar and macroeconomic states against constructed data, and each model against a world in which the
timing it should find was planted."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.equity import alpha_model as am
from src.equity.synthetic import simulate_earnings_world, simulate_timing_world
from src.framework import MODELS, bundle_from_prices, load_library
from src.strategies import alpha_styles
from src.strategies import factor_timing as ft
from src.strategies.factor_models import _grid
from src.utils.dates import rebalance_dates

load_library()


# ------------------------------------------------------------------------------------------------------------------ the premium forecast
def test_factor_slopes_are_the_cross_sectional_regression_slopes():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(6, 30))
    y = 0.02 * z + 0.05 * rng.normal(size=(6, 30))
    z[2, :25] = np.nan                                                                               # too few assets in one month
    y[3, 4] = np.nan                                                                                 # a missing return is dropped from that month
    out = ft.factor_slopes(z, y, min_assets=8)
    assert np.isnan(out[2])
    for s in (0, 1, 3, 4, 5):
        ok = np.isfinite(z[s]) & np.isfinite(y[s])
        assert out[s] == pytest.approx(np.polyfit(z[s, ok], y[s, ok], 1)[0], abs=1e-12)
    assert ft.factor_slopes(np.zeros((2, 20)), np.ones((2, 20))).tolist() == [np.nan, np.nan] or np.isnan(ft.factor_slopes(np.zeros((2, 20)), np.ones((2, 20)))).all()      # an exposure that does not vary prices nothing


def test_the_conditional_premium_is_the_shrunk_state_mean_of_earlier_months():
    slopes = np.array([0.01, 0.03, 0.01, 0.03, 0.01, 0.03, 0.01, 0.03, 0.01, 0.03])
    state = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    out = ft.conditional_premium(slopes, state, shrink=4.0, min_obs=4)
    assert np.isnan(out[:4]).all()
    # t = 6: earlier months 0..5: overall mean 0.02; state 0 (months 0, 2, 4): n = 3 and mean 0.01; weight 3 / (3 + 4)
    assert out[6] == pytest.approx(0.02 + 3 / 7 * (0.01 - 0.02))
    # t = 7: earlier months 0..6 have mean 0.13 / 7; state 1 (months 1, 3, 5): n = 3 and mean 0.03
    overall = slopes[:7].mean()
    assert out[7] == pytest.approx(overall + 3 / 7 * (0.03 - overall))
    assert ft.conditional_premium(slopes, state, shrink=1e9, min_obs=4)[6] == pytest.approx(0.02, abs=1e-6)          # a huge shrink is the overall mean
    assert ft.conditional_premium(slopes, state, shrink=0.0, min_obs=4)[6] == pytest.approx(0.01)                   # none is the state's own mean
    unseen = np.array([0, 0, 0, 0, 0, 0, 2])
    assert ft.conditional_premium(slopes[:7], unseen, shrink=4.0, min_obs=4)[6] == pytest.approx(slopes[:6].mean())  # a state never seen before gets the overall premium
    windowed = ft.conditional_premium(slopes, state, shrink=0.0, min_obs=4, window=4)
    assert windowed[6] == pytest.approx(slopes[2:6][state[2:6] == 0].mean())                                          # only the last four months count


def test_the_premium_forecast_of_a_month_uses_only_earlier_months():
    rng = np.random.default_rng(1)
    slopes = rng.normal(0.005, 0.02, 80)
    state = rng.integers(0, 3, 80)
    base = ft.conditional_premium(slopes, state, 6.0, 12)
    for t in (30, 50, 79):
        changed = slopes.copy()
        changed[t:] += 10.0                                                                              # month t's own return and everything after it is unknown at t
        assert ft.conditional_premium(changed, state, 6.0, 12)[:t + 1].tolist() == pytest.approx(base[:t + 1].tolist(), nan_ok=True)
    state2 = state.copy()
    state2[0] = -1
    assert np.isnan(ft.conditional_premium(slopes, np.full(80, -1), 6.0, 12)).all()                       # unknown state: no forecast


# ------------------------------------------------------------------------------------------------------------------ the states
def test_calendar_states_name_the_month_the_return_is_earned_in():
    grid = pd.DatetimeIndex(["2020-12-31", "2021-01-29", "2021-02-26", "2021-03-31", "2021-04-30", "2021-05-28", "2021-10-29"])      # month-ends: the return after each is for the next month
    assert ft.calendar_state(grid, "january").tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert ft.calendar_state(grid, "month").tolist() == [0, 1, 2, 3, 4, 5, 10]
    assert ft.calendar_state(grid, "quarter").tolist() == [0, 1, 2, 0, 1, 2, 1]                           # January, February, March, April, May, June, November
    assert ft.calendar_state(grid, "halloween").tolist() == [1, 1, 1, 1, 0, 0, 1]                         # November to April against May to October
    with pytest.raises(ValueError):
        ft.calendar_state(grid, "friday")


def macro_frame(index, **series):
    return pd.DataFrame(series, index=index)


def test_the_fed_state_reads_the_direction_of_policy_and_a_flat_band():
    idx = pd.bdate_range("2010-01-01", periods=1500)
    months = np.arange(len(idx)) / 21.0
    dff = np.where(months < 24, 2.0, np.where(months < 36, 2.0 + 0.1 * (months - 24), np.where(months < 48, 3.2 - 0.1 * (months - 36), 2.0)))       # flat, hiking 0.1 a month, cutting 0.1 a month, flat
    bundle = bundle_from_prices(pd.DataFrame(100 * np.ones((len(idx), 3)), index=idx, columns=list("abc")), macro=macro_frame(idx, DFF=dff), name="fed")
    grid = pd.DatetimeIndex([idx[21 * 20], idx[21 * 34], idx[21 * 44], idx[21 * 55], idx[21 * 3]])
    st = ft.macro_state(bundle, "fed", grid)
    assert st[0] == 1 and st[1] == 2 and st[2] == 0 and st[3] == 1                                       # flat, tightening, easing, flat again
    assert st[4] == -1 or st[4] == 1                                                                      # the first months: no six-month-old rate yet, or still flat
    assert ft.macro_state(bundle, "fed", grid, flat_band=100.0).tolist()[1] == 1                          # a wide flat band calls everything flat


@pytest.mark.parametrize("kind,series", [("m1", "M1SL"), ("gdp", "GDPC1"), ("inflation", "CPIAUCNS"), ("ppi", "PPIACO")])
def test_growth_states_compare_the_year_on_year_growth_with_its_own_median_so_far(kind, series):
    idx = pd.bdate_range("2005-01-03", periods=21 * 12 * 12)
    months = np.arange(len(idx)) / 21.0
    growth = np.where(months < 72, 0.02, 0.08)                                                           # slow for six years, then fast
    level = 100 * np.exp(np.cumsum(growth / 252.0))
    bundle = bundle_from_prices(pd.DataFrame(100 * np.ones((len(idx), 3)), index=idx, columns=list("abc")), macro=macro_frame(idx, **{series: level}), name="growth")
    grid = pd.DatetimeIndex([idx[21 * 8], idx[21 * 40], idx[21 * 100], idx[21 * 130]])
    st = ft.macro_state(bundle, kind, grid, min_history=24)
    assert st[0] == -1                                                                                    # less than a year of history, no growth rate yet
    assert st[2] == 1 and st[3] == 1                                                                      # after the step up, growth is above the median of everything so far; the median is over every month-end, not just the dates asked for
    other = macro_frame(idx, **{series: np.where(np.arange(len(idx)) < len(idx) // 2, level, level * 5)})   # a future jump cannot change an earlier state
    b2 = bundle_from_prices(pd.DataFrame(100 * np.ones((len(idx), 3)), index=idx, columns=list("abc")), macro=other, name="g2")
    assert ft.macro_state(b2, kind, grid[:3], min_history=24).tolist()[:2] == st[:2].tolist()


def test_the_market_state_is_the_sign_of_the_universes_return_over_the_last_months():
    idx = pd.bdate_range("2010-01-01", periods=21 * 60)
    up_then_down = np.where(np.arange(len(idx)) < 21 * 30, 0.0008, -0.0008)
    prices = 100 * np.cumprod(1 + np.tile(up_then_down[:, None], (1, 4)) + 0.0001 * np.random.default_rng(0).normal(size=(len(idx), 4)), axis=0)
    bundle = bundle_from_prices(pd.DataFrame(prices, index=idx, columns=list("abcd")), name="mkt")
    grid = pd.DatetimeIndex([idx[21 * 5], idx[21 * 20], idx[21 * 29], idx[21 * 50], idx[21 * 59]])
    st = ft.macro_state(bundle, "market", grid, market_months=12)
    assert st.tolist() == [-1, 1, 1, 0, 0]                                                                # not enough history, up, up, down, down
    for bad in ("weather",):
        with pytest.raises(ValueError):
            ft.macro_state(bundle, bad, grid)


# ------------------------------------------------------------------------------------------------------------------ worlds with planted timing
def make_bundle(idx, prices, **kw):
    return bundle_from_prices(pd.DataFrame(prices, index=idx, columns=[f"S{i:02d}" for i in range(prices.shape[1])]), min_history=60, name="timing", **kw)


def monthly_ic(bundle, score, skip=48, months=None):
    grid = rebalance_dates(bundle.index, "monthly")
    px = bundle.prices.loc[grid]
    fwd = px.shift(-1) / px - 1.0
    ic = am.information_coefficients({"x": score.loc[grid]}, fwd, "spearman", 15)["x"].iloc[skip:]
    if months is not None:
        ic = ic[pd.Series((ic.index.month % 12) + 1, index=ic.index).isin(months)]                       # keep the months the return is earned in
    return float(ic.mean()), float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 2 else float("nan")


def forecast_skill(model, bundle, skip=48):
    """The mean of (forecast premium x premium actually earned) over the months, and its t-statistic: positive when the forecast has the right sign and size. The rank correlation of a score with returns
    depends only on the sign of the premium forecast, never its size; this measure depends on both."""
    grid, forward = _grid(bundle)
    f = model.premia(bundle)
    gains = [f[n].to_numpy() * ft.factor_slopes(z, forward.to_numpy()) for n, z in model.exposures(bundle, grid).items()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                                                  # months before any factor has a forecast are all NaN
        product = np.nanmean(np.vstack(gains), axis=0)[skip:]
    product = product[np.isfinite(product)]
    return float(product.mean()), float(product.mean() / product.std() * np.sqrt(len(product)))


@pytest.fixture(scope="module")
def january_world():
    idx, prices, _ = simulate_timing_world(0, premium=lambda month, st: -0.02 if month == 1 else 0.004)          # momentum loses 2% a month per sd in January and earns 0.4% in the other months
    return make_bundle(idx, prices)


def test_calendar_timing_finds_that_momentum_loses_in_january_and_pays_otherwise(january_world):
    model = MODELS.create("calendar_factor_timing", factor="mom", state="january")
    plain_model = MODELS.create("calendar_factor_timing", factor="mom", state="january", shrink=1e9)         # a huge shrink never uses the state: the plain factor
    premia = model.premia(january_world)["mom"].dropna()
    target_jan = pd.Series(((premia.index.month % 12) + 1) == 1, index=premia.index)
    assert premia[target_jan].mean() < -0.001 and premia[target_jan].iloc[-5:].mean() < -0.004          # the premium forecast before a January is negative, and more so as Januaries are seen
    assert premia[~target_jan].mean() > 0.003 and premia[target_jan].mean() < premia[~target_jan].mean() - 0.005
    timed, plain = monthly_ic(january_world, model.score(january_world)), monthly_ic(january_world, plain_model.score(january_world))
    assert timed[0] > plain[0] + 0.03 and timed[1] > plain[1] + 2                                       # the sign of the forecast is right far more often
    jan_timed = monthly_ic(january_world, model.score(january_world), months=[1])
    jan_plain = monthly_ic(january_world, plain_model.score(january_world), months=[1])
    assert jan_timed[0] > 0.1 and jan_plain[0] < -0.1                                                   # in the Januaries it turns a loss into a gain
    skill, plain_skill = forecast_skill(model, january_world), forecast_skill(plain_model, january_world)
    assert skill[0] > 1.8 * plain_skill[0] and skill[1] > plain_skill[1] + 1.5                          # and, counting the size of the forecast, more than doubles what acting on it earns


def test_calendar_timing_does_no_harm_where_there_is_nothing_to_time():
    idx, prices, _ = simulate_timing_world(1, premium=lambda month, st: 0.004)                                   # the same premium every month
    bundle = make_bundle(idx, prices)
    timed = MODELS.create("calendar_factor_timing", factor="mom", state="january")
    plain = MODELS.create("calendar_factor_timing", factor="mom", state="january", shrink=1e9)
    a, b = forecast_skill(timed, bundle), forecast_skill(plain, bundle)
    assert a[0] > 0 and a[0] > 0.8 * b[0]                                                                # the shrinkage keeps a state seen a dozen times from creating a pattern
    assert monthly_ic(bundle, timed.score(bundle))[0] == pytest.approx(monthly_ic(bundle, plain.score(bundle))[0], abs=0.01)


def test_the_twelve_month_state_finds_january_too_with_a_little_more_noise(january_world):
    jan = MODELS.create("calendar_factor_timing", factor="mom", state="january").premia(january_world)["mom"]
    month = MODELS.create("calendar_factor_timing", factor="mom", state="month").premia(january_world)["mom"]
    assert jan.notna().sum() == month.notna().sum() > 100
    target = pd.Series(((month.index.month % 12) + 1), index=month.index)
    late = month.index > month.index[100]
    assert month[late & (target == 1)].mean() < month[late & (target != 1)].mean() - 0.005
    assert MODELS.create("calendar_factor_timing", factor="mom", state="quarter").premia(january_world)["mom"].dropna().shape[0] > 100


def test_a_list_of_factors_is_the_sum_of_the_timed_factors(january_world):
    both = MODELS.create("calendar_factor_timing", factor="mom,rev", state="january")
    p = both.premia(january_world)
    assert list(p.columns) == ["mom", "rev"]
    single = MODELS.create("calendar_factor_timing", factor="mom", state="january").score(january_world)
    mixed = both.score(january_world)
    assert not np.allclose(single.fillna(0).to_numpy(), mixed.fillna(0).to_numpy())
    assert list(MODELS.create("calendar_factor_timing", factor="all").names) == list(("mom", "rev", "lowvol", "lowbeta", "nomax", "high"))


def test_timing_models_are_causal_and_say_what_they_need(january_world):
    cut = january_world.index[2800]
    noisy = january_world.perturbed_after(cut)
    for name, kw in (("calendar_factor_timing", {"state": "halloween"}), ("macro_factor_timing", {"state": "market"})):
        a, b = MODELS.create(name, **kw).score(january_world), MODELS.create(name, **kw).score(noisy)
        assert np.allclose(a.loc[:cut].to_numpy(), b.loc[:cut].to_numpy(), equal_nan=True), name
    with pytest.raises(KeyError, match="macro series"):
        MODELS.create("macro_factor_timing", state="fed").score(january_world)                          # a state that reads DFF says so when the bundle has no DFF
    for name, bad in (("calendar_factor_timing", {"state": "friday"}), ("calendar_factor_timing", {"factor": "beauty"}), ("calendar_factor_timing", {"shrink": -1.0}), ("calendar_factor_timing", {"min_obs": 2}),
                      ("macro_factor_timing", {"state": "weather"}), ("macro_factor_timing", {"market_months": 0}), ("macro_factor_timing", {"flat_band": -1.0}), ("macro_factor_timing", {"min_history": 3})):
        with pytest.raises(ValueError):
            MODELS.create(name, **bad)


@pytest.fixture(scope="module")
def market_world():
    up = lambda s, ret: int(np.prod(1 + ret[s - 251:s + 1].mean(axis=1)) - 1 > 0)                         # the equal-weight 12-month return, exactly as the model measures it
    idx, prices, states = simulate_timing_world(2, years=18, premium=lambda month, st: 0.012 if st == 1 else -0.012, state_of=up, market_regimes=True)
    return make_bundle(idx, prices)


def test_macro_timing_by_the_markets_own_state_finds_momentum_after_up_and_down_markets(market_world):
    model = MODELS.create("macro_factor_timing", factor="mom", state="market")
    premia = model.premia(market_world)["mom"].dropna()
    state = pd.Series(ft.macro_state(market_world, "market", premia.index), index=premia.index)
    assert state.isin([0, 1]).all() and 0.2 < state.mean() < 0.8                                         # both kinds of market occur
    assert premia[state == 1].mean() > 0.004 and premia[state == 0].mean() < -0.004                      # a premium forecast of each sign, as planted
    timed = monthly_ic(market_world, model.score(market_world), skip=60)
    plain = monthly_ic(market_world, MODELS.create("macro_factor_timing", factor="mom", state="market", shrink=1e9).score(market_world), skip=60)
    assert timed[0] > plain[0] + 0.03 and timed[1] > 4                                                   # the plain factor averages a premium that changes sign and finds little


class Shell:
    """The attributes ``macro_state`` reads from a bundle, for building a world from a macro series before there is a bundle."""

    def __init__(self, index, macro):
        self.macro, self.index = macro, index
        self.returns = pd.DataFrame(0.0, index=index, columns=["a"])
        self.investable = pd.DataFrame(True, index=index, columns=["a"])


def test_macro_timing_by_fed_policy_reads_the_funds_rate_with_its_lag():
    idx = pd.bdate_range("2004-01-05", periods=16 * 252)
    cycle = np.sin(2 * np.pi * (np.arange(len(idx)) / 21.0) / 28.0)                                       # policy rates rise for fourteen months and fall for fourteen
    macro = macro_frame(idx, DFF=3.0 + 1.5 * cycle)
    shell = Shell(idx, macro)
    premium = lambda month, st: {0: -0.012, 1: 0.0, 2: 0.012}.get(st, 0.0)                                # momentum pays when policy tightens and loses when it eases
    _, prices, states = simulate_timing_world(3, years=16, premium=premium, state_of=lambda s, ret: int(ft.macro_state(shell, "fed", idx[[s]])[0]))
    bundle = make_bundle(idx, prices, macro=macro)
    assert {0, 1, 2} <= set(states.values())
    model = MODELS.create("macro_factor_timing", factor="mom", state="fed")
    premia = model.premia(bundle)["mom"].dropna()
    st = pd.Series(ft.macro_state(bundle, "fed", premia.index), index=premia.index)
    assert premia[st == 2].mean() > 0.003 and premia[st == 0].mean() < -0.003
    timed = monthly_ic(bundle, model.score(bundle), skip=60)
    plain = monthly_ic(bundle, MODELS.create("macro_factor_timing", factor="mom", state="fed", shrink=1e9).score(bundle), skip=60)
    assert timed[0] > plain[0] + 0.03 and timed[1] > 3
    model.require(bundle)


# ------------------------------------------------------------------------------------------------------------------ the earnings announcement premium
def earnings_world(seed=0, years=12, n=60, premium=0.02):
    prices, volume, events = simulate_earnings_world(seed, years, n, premium)
    return bundle_from_prices(prices, volume=volume, min_history=60, name="earnings"), events


def test_the_volume_proxy_finds_the_announcement_days():
    bundle, planted = earnings_world(0, premium=0.0)
    found = ft.announcement_events(bundle, path="/nonexistent/earnings.csv", source="volume")
    hits = float((found.to_numpy() * planted.to_numpy()).sum())
    assert hits / planted.to_numpy().sum() > 0.9 and hits / found.to_numpy().sum() > 0.9                # recall and precision of the spikes: a spike of five times the usual volume is occasionally hidden by a quiet day's noise


def test_a_file_of_dates_is_read_as_the_announcements(tmp_path):
    bundle, planted = earnings_world(1, years=6, n=12, premium=0.0)
    rows = [(planted.index[i].strftime("%Y-%m-%d"), planted.columns[j]) for i, j in zip(*np.nonzero(planted.to_numpy()))]
    saturday = pd.Timestamp(rows[0][0]) + pd.offsets.Week(weekday=5)
    rows += [(saturday.strftime("%Y-%m-%d"), rows[0][1]), (rows[0][0], "NOT_IN_THE_UNIVERSE")]
    path = tmp_path / "earnings.csv"
    pd.DataFrame(rows, columns=["date", "ticker"]).to_csv(path, index=False)
    events = ft.announcement_events(bundle, path=str(path), source="file")
    assert (events.to_numpy() >= planted.to_numpy()).all()                                               # everything planted is there
    extra = events.to_numpy() - planted.to_numpy()
    assert extra.sum() == 1 and events.shape == planted.shape                                              # plus the weekend date, moved to the next trading day; the unknown ticker is ignored
    broken = tmp_path / "broken.csv"
    pd.DataFrame({"day": ["2010-01-04"], "name": ["x"]}).to_csv(broken, index=False)
    with pytest.raises(ValueError, match="date and ticker"):
        ft.announcement_events(bundle, path=str(broken), source="file")


def test_the_earnings_season_premium_finds_the_month_of_reporting_and_nothing_where_there_is_none():
    bundle, _ = earnings_world(2, premium=0.02)                                                          # reporters earn 2% more in their months: a binary flag on a third of the stocks needs a big effect to show in a rank correlation
    model = MODELS.create("earnings_season_premium", path="/nonexistent/earnings.csv", source="volume")
    ic, t = monthly_ic(bundle, model.score(bundle), skip=36)
    assert ic > 0.08 and t > 5
    null, _ = earnings_world(3, premium=0.0)
    ic0, t0 = monthly_ic(null, MODELS.create("earnings_season_premium", path="/nonexistent/earnings.csv", source="volume").score(null), skip=36)
    assert abs(t0) < 3
    cut = bundle.index[2000]
    a, b = model.score(bundle), model.score(bundle.perturbed_after(cut))
    assert np.allclose(a.loc[:cut].to_numpy(), b.loc[:cut].to_numpy(), equal_nan=True)


def test_dates_from_a_file_do_at_least_as_well_as_the_proxy(tmp_path):
    bundle, planted = earnings_world(4, premium=0.02)
    rows = [(planted.index[i].strftime("%Y-%m-%d"), planted.columns[j]) for i, j in zip(*np.nonzero(planted.to_numpy()))]
    path = tmp_path / "dates.csv"
    pd.DataFrame(rows, columns=["date", "ticker"]).to_csv(path, index=False)
    from_file = monthly_ic(bundle, MODELS.create("earnings_season_premium", path=str(path), source="file").score(bundle), skip=36)
    proxy = monthly_ic(bundle, MODELS.create("earnings_season_premium", path="/nonexistent/earnings.csv", source="volume").score(bundle), skip=36)
    assert from_file[0] >= proxy[0] - 0.005 and from_file[0] > 0.08
    auto = MODELS.create("earnings_season_premium", path=str(path), source="auto").score(bundle)          # auto prefers the file when there is one
    assert np.allclose(auto.fillna(0).to_numpy(), MODELS.create("earnings_season_premium", path=str(path), source="file").score(bundle).fillna(0).to_numpy())


def test_the_earnings_model_says_what_it_needs(tmp_path):
    bundle, _ = earnings_world(5, years=4, n=12)
    with pytest.raises(KeyError, match="needs the file"):
        MODELS.create("earnings_season_premium", path=str(tmp_path / "nope.csv"), source="file").score(bundle)
    no_volume = bundle_from_prices(bundle.prices, min_history=60, name="no volume")
    with pytest.raises(KeyError, match="needs the file"):
        MODELS.create("earnings_season_premium", path=str(tmp_path / "nope.csv"), source="auto").score(no_volume)
    for bad in ({"source": "rumour"}, {"min_obs": 2}, {"window": -1}, {"volume_multiple": 1.0}):
        with pytest.raises(ValueError):
            MODELS.create("earnings_season_premium", **bad)
    assert alpha_styles.USER_DATA.name == "user"
