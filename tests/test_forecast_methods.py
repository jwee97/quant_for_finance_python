"""Every forecasting method on its own: it fits and forecasts a finite, reproducible distribution; it refuses a history it cannot learn from; and, tested on a market with a known truth, it finds what was planted and
nothing where nothing was."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.forecast import baselines, gp, markov, montecarlo, others
from src.forecast.base import ForecastError, History
from src.forecast.evaluate import _plain, combine
from src.forecast.registry import BASELINE, BLEND, BLEND_WEIGHTED, DEFAULT, METHODS, catalog, create
from tests import forecast_worlds as W

HS = (1, 5, 21)


@pytest.fixture(scope="module")
def world():
    """A volatile market with clustering, as a history with highs and lows."""
    return W.to_history(W.garch(1600, seed=11), high_low=True)


def forecast(method, history, n=400, seed=0, horizons=HS):
    state = method.fit(history.tail(method.window), horizons)
    return method.predict(state, history.tail(method.lookback), horizons, n, np.random.default_rng(seed))


# ---------------------------------------------------------------------------------------------------------------------------------- the contract
def test_the_registry_lists_every_method_once_with_what_the_form_shows():
    ids = [m["id"] for m in catalog()]
    assert ids == list(METHODS) + [BLEND, BLEND_WEIGHTED] and len(set(ids)) == len(ids) and ids[0] == BASELINE
    for m in catalog():
        assert m["label"] and m["short"] and m["text"] and m["assumes"] and m["fails"] and m["family"], m["id"]
    assert set(DEFAULT) <= set(METHODS) and BASELINE in DEFAULT and create("garch") is METHODS["garch"]
    with pytest.raises(KeyError):
        create("astrology")


@pytest.mark.parametrize("name", list(METHODS))
def test_a_method_forecasts_a_finite_widening_reproducible_distribution_with_plain_numbers_in_its_report(name, world):
    m = METHODS[name]
    a, b = forecast(m, world, seed=3), forecast(m, world, seed=3)
    assert a.samples.shape == (400, 21) and np.isfinite(a.samples).all()
    assert np.allclose(a.samples, b.samples, atol=1e-7, rtol=0), "the same seed must give the same forecast (up to the last digits of a numerical fit)"
    assert not np.array_equal(a.samples, forecast(m, world, seed=4).samples)
    sd = a.samples.std(axis=0)
    assert sd[20] > 1.5 * sd[0], "uncertainty must grow with the horizon"
    json.dumps(_plain(a.info))                                                     # the report is JSON
    if not m.mean_signal:
        assert abs(a.at(21).mean()) < 5 * a.at(21).std() / np.sqrt(400), "a method that claims no view on the average must have none"


@pytest.mark.parametrize("name", list(METHODS))
def test_a_method_refuses_a_history_it_cannot_learn_from_and_never_fails_in_another_way(name):
    m = METHODS[name]
    for h in (W.to_history(W.iid(120, seed=1)), History(np.full(1500, 100.0))):
        try:
            forecast(m, h, n=50)
        except ForecastError:
            pass                                                                    # the refusal the evaluation knows how to count


# --------------------------------------------------------------------------------------------------------------------------------- the baselines
def test_the_random_walk_takes_the_spread_of_the_last_year_and_has_no_drift():
    h = W.to_history(np.concatenate([W.iid(500, 0.03, seed=1), W.iid(300, 0.005, seed=2)]))
    f = forecast(baselines.RandomWalk(), h, n=20_000)
    year = h.r[-252:]
    assert f.info["daily_volatility"] == pytest.approx(np.sqrt(np.mean(year ** 2)))
    assert f.at(21).std() == pytest.approx(f.info["daily_volatility"] * np.sqrt(21), rel=0.03) and abs(f.at(21).mean()) < 0.003


def test_riskmetrics_follows_the_exponentially_weighted_variance_and_reacts_to_a_storm():
    calm_then_storm = W.to_history(np.concatenate([W.iid(600, 0.004, seed=1), W.iid(15, 0.03, seed=2)]))
    calm = W.to_history(W.iid(615, 0.004, seed=1))
    storm_sd = forecast(baselines.RiskMetrics(), calm_then_storm).info["daily_volatility"]
    assert storm_sd > 3 * forecast(baselines.RiskMetrics(), calm).info["daily_volatility"]
    r = np.array([0.01, -0.02, 0.015, 0.0, 0.005] * 40)
    v = 0.0
    for i, x in enumerate(r):                                                      # the textbook recursion, seeded as the module seeds it
        v = (np.mean(r[:30] ** 2) if i == 0 else v)
        v = 0.94 * v + 0.06 * x * x
    assert baselines.RiskMetrics().predict(None, W.to_history(r), HS, 10, np.random.default_rng(0)).info["daily_volatility"] == pytest.approx(np.sqrt(v), rel=1e-9)


def test_historical_simulation_carries_the_last_five_years_average_return_and_gbm_the_last_years():
    r = np.concatenate([W.iid(1000, 0.01, 0.0, seed=1), W.iid(252, 0.01, 0.002, seed=2)])
    h = W.to_history(r)
    hist = forecast(baselines.Historical(), h, n=20_000)
    gbm = forecast(baselines.GBM(), h, n=20_000)
    assert hist.at(21).mean() == pytest.approx(21 * r[-1260:].mean(), abs=0.004)
    assert gbm.at(21).mean() == pytest.approx(21 * r[-252:].mean(), abs=0.004) and gbm.at(21).mean() > hist.at(21).mean()
    assert gbm.info["annual_drift"] == pytest.approx(r[-252:].mean() * 252)


def test_the_volatility_scaled_simulation_takes_todays_spread_with_the_shape_of_the_past():
    storm = W.to_history(np.concatenate([W.garch(1200, seed=5), W.iid(20, 0.04, seed=6)]))
    calm = W.to_history(np.concatenate([W.garch(1200, seed=5), W.iid(60, 0.002, seed=6)]))
    f_storm, f_calm = forecast(baselines.HullWhite(), storm, n=3000), forecast(baselines.HullWhite(), calm, n=3000)
    assert f_storm.at(5).std() > 3 * f_calm.at(5).std()
    assert f_storm.info["kurtosis_of_shocks"] > 3.0, "the shape of the past, fat tails included, is kept"


def test_a_pool_draws_the_same_number_from_every_member_and_a_small_member_is_drawn_with_replacement():
    rng = np.random.default_rng(0)
    members = [np.full((50, 3), 1.0), np.full((600, 3), 2.0), np.full((600, 3), 3.0)]
    pooled = combine(members, rng, 300)
    assert pooled.shape == (300, 3) and [int((pooled[:, 0] == v).sum()) for v in (1.0, 2.0, 3.0)] == [100, 100, 100]


# ------------------------------------------------------------------------------------------------------------------------------ Monte Carlo
def test_the_garch_forecast_learns_the_persistence_and_the_leverage_that_were_planted_and_starts_from_todays_volatility():
    r = W.garch(2500, seed=3, alpha=0.05, gamma=0.12, beta=0.85)
    f = forecast(montecarlo.GarchMC(), W.to_history(r))
    assert 0.9 < f.info["persistence"] < 1.0 and f.info["gamma"] > 0.03, f.info
    quiet = W.to_history(np.concatenate([r, np.full(40, 1e-4)]))
    wild = W.to_history(np.concatenate([r, np.random.default_rng(1).normal(0, 0.04, 40)]))
    assert forecast(montecarlo.GarchMC(), wild).at(1).std() > 2 * forecast(montecarlo.GarchMC(), quiet).at(1).std()


def test_filtered_historical_simulation_uses_the_fits_own_residuals():
    h = W.to_history(W.garch(1500, seed=4))
    fhs, parametric = forecast(montecarlo.FilteredHistorical(), h, n=8000), forecast(montecarlo.GarchMC(), h, n=8000)
    assert fhs.at(1).std() == pytest.approx(parametric.at(1).std(), rel=0.25)
    assert montecarlo.FilteredHistorical().dist == "normal" and montecarlo.GarchMC().dist == "t"


def test_the_jump_diffusion_finds_the_rate_the_size_and_the_calm_volatility_of_planted_jumps():
    r = W.jumps(3000, seed=2, sigma=0.005, rate=0.03, mean=-0.03, sd=0.05)
    m = montecarlo.JumpDiffusion()
    state = m.fit(W.to_history(r).tail(1260), HS)
    assert 0.4 * 0.03 <= state["lam"] <= 1.1 * 0.03, state                          # most jumps are large enough to be seen; the smallest are lost among the ordinary days
    assert state["mu_j"] < 0 and state["sigma"] == pytest.approx(0.005, rel=0.35)
    f = m.predict(state, W.to_history(r), HS, 20_000, np.random.default_rng(0))
    assert abs(f.at(21).mean()) < 0.004, "the average return is set to zero"
    plain = baselines.RandomWalk().predict(None, W.to_history(r), HS, 20_000, np.random.default_rng(0))
    assert np.quantile(f.at(5), 0.01) < np.quantile(plain.at(5), 0.01), "the jumps fatten the left tail"
    assert m.fit(W.to_history(W.iid(1500, 0.01, seed=1)).tail(1260), HS)["lam"] < 0.01, "no jumps are found where none were planted"


# ---------------------------------------------------------------------------------------------------------------------------------- Markov
def test_the_markov_chain_on_volatility_finds_the_persistent_states_and_where_the_market_is():
    r, state = W.regimes(2400, seed=3)
    m = markov.MarkovVolatility()
    h = W.to_history(r)
    f = forecast(m, h, n=500)
    P = np.array(f.info["transition_matrix"])
    assert P.shape == (3, 3) and np.allclose(P.sum(axis=1), 1.0) and P[0, 0] > 0.8 and P[2, 2] > 0.8, "calm stays calm and stormy stays stormy for weeks"
    vols = f.info["state_annual_volatility"]
    assert vols[0] < vols[1] < vols[2] and sum(f.info["stationary"]) == pytest.approx(1.0)
    assert min(f.info["expected_days_in_state"]) > 3
    stormy_end = W.to_history(np.concatenate([r[:2000], np.random.default_rng(0).normal(0, 0.025, 30)]))
    calm_end = W.to_history(np.concatenate([r[:2000], np.random.default_rng(0).normal(0, 0.003, 30)]))
    assert forecast(m, stormy_end).info["state_today"] == 2 and forecast(m, calm_end).info["state_today"] == 0
    assert forecast(m, stormy_end).at(5).std() > 1.5 * forecast(m, calm_end).at(5).std()


def test_the_markov_chain_on_the_days_move_finds_no_memory_in_independent_days_and_some_in_clustered_ones():
    m = markov.MarkovReturns()
    iid_state = m.fit(W.to_history(W.iid(1500, 0.01, seed=5)).tail(1260), HS)
    clustered = m.fit(W.to_history(W.garch(1500, seed=5)).tail(1260), HS)
    assert iid_state["order_p"] > 0.01 and clustered["order_p"] < 0.001, (iid_state["order_p"], clustered["order_p"])
    P = iid_state["P"]
    assert P.shape == (5, 5) and np.allclose(P.sum(axis=1), 1.0) and np.abs(P - 0.2).max() < 0.12
    assert [len(p) for p in iid_state["pools"]] == [pytest.approx(252, abs=2)] * 5
    f = m.predict(clustered, W.to_history(W.garch(1500, seed=5)), HS, 300, np.random.default_rng(0))
    assert f.info["state_today"] in range(5) and len(f.info["states"]) == 5


def test_the_hidden_markov_model_recovers_the_two_regimes_and_knows_which_one_it_is_in():
    r, state = W.regimes(2600, seed=3)
    m = markov.HiddenMarkov()
    h_calm = W.to_history(r[: _last(state, 0, 2400)])
    h_storm = W.to_history(r[: _last(state, 1, 2400)])
    fc, fs = forecast(m, h_calm, n=400), forecast(m, h_storm, n=400)
    vols = np.array(fc.info["regime_annual_volatility"])
    assert vols[0] == pytest.approx(0.006 * np.sqrt(252), rel=0.3) and vols[1] == pytest.approx(0.02 * np.sqrt(252), rel=0.3), vols
    A = np.array(fc.info["transition_matrix"])
    assert A[0, 0] > 0.95 and A[1, 1] > 0.9 and np.allclose(A.sum(axis=1), 1.0)
    assert fc.info["probability_of_each_regime_today"][0] > 0.9 and fs.info["probability_of_each_regime_today"][1] > 0.9
    assert fs.at(5).std() > 2 * fc.at(5).std(), "the forecast spread follows the regime"
    assert min(fc.info["expected_days_in_regime"]) > 20


def _last(state, regime, upto, run=40):
    """The last position before ``upto`` that ends a run of ``run`` days in ``regime``."""
    for t in range(upto, run, -1):
        if (state[t - run:t] == regime).all():
            return t
    raise AssertionError("no such stretch")


# ------------------------------------------------------------------------------------------------------------------------ Gaussian processes
def test_the_regression_process_finds_nothing_in_noise_and_something_where_the_past_month_predicts_the_next_days():
    m = gp.GPReturns()
    noise = forecast(m, W.to_history(W.iid(1500, 0.01, seed=1)), horizons=(1, 5, 21))
    assert min(noise.info["unexplained_share"].values()) > 0.97, "noise: the process explains none of it"
    signal = W.reverting(2200, 0.5, seed=1)
    f = forecast(m, W.to_history(signal), horizons=(1, 5, 21))
    assert f.info["unexplained_share"]["21"] < 0.99 and f.info["feature_relevance"]["move over 21 days"] > 0 and set(f.info["standardised_forecast"]) == {"1", "5", "21"}
    up = W.to_history(np.concatenate([signal[:2000], np.full(21, 0.004)]))         # a steady month-long rise
    down = W.to_history(np.concatenate([signal[:2000], np.full(21, -0.004)]))
    assert forecast(m, up, horizons=(21,)).at(21).mean() < forecast(m, down, horizons=(21,)).at(21).mean(), "after a rise the next month is forecast lower: the planted reversal"


def test_the_drift_process_follows_a_slowly_changing_average_and_stays_at_zero_in_noise():
    m = gp.GPDrift()
    r = W.slow_drift(2400, seed=2)
    means, truth = [], []
    for i in range(900, 2300, 70):
        f = forecast(m, W.to_history(r[:i]), n=200)
        means.append(f.info["drift_in_a_month_in_volatilities"])
        truth.append(r[i:i + 21].mean())
    assert np.corrcoef(means, truth)[0, 1] > 0.5, "its forecast of the drift tracks the planted drift"
    flat = [forecast(m, W.to_history(W.iid(900 + 80 * k, 0.01, seed=k)), n=200).info for k in range(4)]
    assert all(abs(info["drift_in_a_month_in_volatilities"]) < 0.5 and info["drift_share_of_variance"] < 0.3 for info in flat), "no drift is found in independent days"


def test_the_price_path_process_extrapolates_a_noiseless_line_and_is_the_control_that_does_badly_on_a_real_looking_price():
    line = W.to_history(np.full(600, 0.002))
    f = forecast(gp.GPPrice(), line, n=200)
    assert f.at(21).mean() == pytest.approx(21 * 0.002, rel=0.35), "a straight line is continued"
    noisy = W.to_history(W.iid(1500, 0.01, seed=7))
    state = gp.GPPrice().fit(noisy.tail(500), HS)
    errs_gp, errs_rw = [], []
    for i in range(700, 1400, 35):
        past = noisy.upto(i)
        y = np.log(noisy.close[i + 21] / noisy.close[i])
        errs_gp.append((gp.GPPrice().predict(state, past.tail(504), HS, 100, np.random.default_rng(i)).at(21).mean() - y) ** 2)
        errs_rw.append(y ** 2)
    assert np.mean(errs_gp) > np.mean(errs_rw), "on a random walk the extrapolated curve is worse than staying put"


# ---------------------------------------------------------------------------------------------------------------------- state space, time series, volatility, analogues
def test_the_kalman_filter_finds_the_slope_of_a_steady_trend_and_a_change_of_direction():
    slopes = [forecast(others.KalmanTrend(), W.to_history(W.trend(1500, 0.001, 0.003, seed=k)), n=200).info["slope_per_day_annual"] for k in range(1, 6)]
    assert all(0.05 < v < 0.45 for v in slopes) and np.mean(slopes) == pytest.approx(0.001 * 252, abs=0.1), slopes
    swing = W.slow_drift(1500, amplitude=0.003, period=250, sigma=0.002, seed=1)       # the average swings up for 125 days, then down for 125
    rising, falling = swing[:1312], swing[:1437]                                        # cut where the average is rising fastest and where it is falling fastest
    assert forecast(others.KalmanTrend(), W.to_history(rising), n=100).info["slope_per_day_annual"] > 0 > forecast(others.KalmanTrend(), W.to_history(falling), n=100).info["slope_per_day_annual"]


def test_the_arma_model_finds_the_serial_dependence_that_was_planted_and_none_in_independent_returns():
    f = forecast(others.ARMAReturns(), W.to_history(W.ar1(2200, 0.3, seed=2)), n=400)
    psi1 = f.info["ar"][0] + f.info["ma"][0]
    assert 0.12 < psi1 < 0.45, f.info
    g = forecast(others.ARMAReturns(), W.to_history(W.iid(2200, 0.01, seed=2)), n=400)
    assert abs(g.info["ar"][0] + g.info["ma"][0]) < 0.12
    up_day = W.to_history(np.concatenate([W.ar1(2200, 0.3, seed=2), [0.03]]))
    assert forecast(others.ARMAReturns(), up_day, n=100).info["next_day_mean"] > 0.002, "after a big up day the next day's mean is up: the planted persistence"


def test_the_har_model_forecasts_more_volatility_after_a_stormy_month_and_uses_the_range_when_there_is_one():
    r = W.garch(2200, seed=6)
    storm = W.to_history(np.concatenate([r, np.random.default_rng(1).normal(0, 0.03, 25)]), high_low=True)
    calm = W.to_history(np.concatenate([r, np.random.default_rng(1).normal(0, 0.002, 25)]), high_low=True)
    fs, fc = forecast(others.HAR(), storm, n=400), forecast(others.HAR(), calm, n=400)
    assert fs.info["proxy"] == "range" and fs.info["volatility_tomorrow_annual"] > 2 * fc.info["volatility_tomorrow_annual"]
    assert fs.info["volatility_next_month_annual"] < fs.info["volatility_tomorrow_annual"] * 1.2, "a storm fades back"
    no_range = forecast(others.HAR(), W.to_history(r), n=400)
    assert no_range.info["proxy"] == "returns" and 5.0 <= no_range.info["tail_degrees_of_freedom"] <= 100.0
    with pytest.raises(ForecastError):
        others.HAR().predict(others.HAR().fit(storm.tail(1260), HS), W.to_history(r).tail(504), HS, 10, np.random.default_rng(0))     # the fit used highs and lows that the later data do not have


def test_the_analogues_use_what_followed_days_like_today_and_scale_it_to_todays_volatility():
    signal = W.reverting(2200, 0.5, seed=3)
    m = others.Analogues()
    rise = W.to_history(np.concatenate([signal[:2000], np.full(21, 0.004)]))
    fall = W.to_history(np.concatenate([signal[:2000], np.full(21, -0.004)]))
    r_up, r_down = forecast(m, rise, horizons=(5, 21)), forecast(m, fall, horizons=(5, 21))
    assert r_up.info["mean_standardised_outcome"]["21"] < r_down.info["mean_standardised_outcome"]["21"]
    assert r_up.info["analogues_used"] == 60 and len(r_up.info["features_today"]) == 5
    stormy = W.to_history(np.concatenate([signal[:2000], np.random.default_rng(2).normal(0, 0.04, 25)]))
    assert forecast(m, stormy, horizons=(5, 21)).at(5).std() > forecast(m, W.to_history(signal[:2025]), horizons=(5, 21)).at(5).std()
