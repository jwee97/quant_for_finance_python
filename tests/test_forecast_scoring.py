"""The scoring rules of ``forecast.scoring`` against closed forms and against what they must do on forecasts of known quality."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.forecast import scoring as S
from src.models.probabilistic import crps_gaussian


def test_the_crps_of_draws_matches_the_closed_form_for_a_gaussian_and_is_the_absolute_error_for_a_point():
    rng = np.random.default_rng(0)
    y = np.array([0.0, 1.0, -2.0, 0.5])
    draws = rng.normal(0.3, 1.2, size=(4, 40_000))
    assert np.allclose(S.crps(draws, y), crps_gaussian(y, 0.3, 1.2), atol=0.01)
    point = np.full((3, 50), 2.0)                                  # a forecast with no spread is scored by its absolute error
    assert np.allclose(S.crps(point, np.array([2.0, 3.5, -1.0])), [0.0, 1.5, 3.0])
    assert S.crps(rng.normal(size=100), 0.2).shape == (1,)          # one forecast as a plain vector
    with pytest.raises(ValueError):
        S.crps(rng.normal(size=(3, 10)), np.zeros(2))


def test_the_fair_crps_has_no_small_sample_bias_and_the_plain_one_has():
    rng = np.random.default_rng(1)
    draws, y = rng.normal(size=(30_000, 8)), rng.normal(size=30_000)       # the truth is N(0, 1) and so is the outcome: the expected CRPS is 1/sqrt(pi)
    target = 1.0 / np.sqrt(np.pi)
    assert S.crps(draws, y).mean() == pytest.approx(target, abs=0.01)
    assert S.crps(draws, y, fair=False).mean() > target + 0.03


def test_crps_is_scale_equivariant_and_prefers_the_sharp_and_right_forecast():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 500)
    right = rng.normal(0, 1, (500, 300))
    assert np.allclose(S.crps(3.0 * right, 3.0 * y), 3.0 * S.crps(right, y))
    too_wide, shifted = rng.normal(0, 3, (500, 300)), rng.normal(1.5, 1, (500, 300))
    assert S.crps(right, y).mean() < S.crps(too_wide, y).mean() and S.crps(right, y).mean() < S.crps(shifted, y).mean()


def test_twice_the_average_pinball_loss_over_all_levels_is_the_crps():
    rng = np.random.default_rng(3)
    draws, y = rng.normal(size=(200, 4000)), rng.normal(size=200)
    levels = np.linspace(0.01, 0.99, 99)
    assert 2.0 * S.pinball(draws, y, levels).mean(axis=1).mean() == pytest.approx(S.crps(draws, y).mean(), rel=0.03)


def test_the_pinball_loss_of_one_quantile_is_tau_times_the_miss_above_and_one_minus_tau_below():
    x = np.arange(1.0, 101.0)                                      # the 25% quantile of these draws is 25.75
    q = np.quantile(x, 0.25)
    above, below = S.pinball(x, 60.0, levels=(0.25,))[0, 0], S.pinball(x, 10.0, levels=(0.25,))[0, 0]
    assert above == pytest.approx(0.25 * (60.0 - q)) and below == pytest.approx(0.75 * (q - 10.0))


def test_intervals_and_exceedances_have_their_nominal_rate_for_a_calibrated_forecast_and_not_for_a_biased_one():
    rng = np.random.default_rng(4)
    draws, y = rng.normal(size=(6000, 400)), rng.normal(size=6000)
    assert S.interval_hits(draws, y, 0.9).mean() == pytest.approx(0.9, abs=0.015)
    assert S.interval_hits(draws, y, 0.5).mean() == pytest.approx(0.5, abs=0.02)
    assert S.below_quantile(draws, y, 0.05).mean() == pytest.approx(0.05, abs=0.01)
    assert S.below_quantile(draws, y - 1.0, 0.05).mean() > 0.2                       # outcomes a standard deviation lower than forecast fall below the 5% quantile far too often
    assert S.below_quantile(draws, np.array([-10.0] + [0.0] * 5999), 0.05)[0]


def test_the_pit_is_uniform_when_calibrated_humped_when_the_forecast_is_too_wide_and_u_shaped_when_too_narrow():
    rng = np.random.default_rng(5)
    y = rng.normal(size=5000)
    flat = S.pit_histogram(S.pit(rng.normal(size=(5000, 300)), y))
    wide = S.pit_histogram(S.pit(rng.normal(0, 2.5, (5000, 300)), y))
    narrow = S.pit_histogram(S.pit(rng.normal(0, 0.4, (5000, 300)), y))
    assert np.allclose(flat, 0.1, atol=0.02) and flat.sum() == pytest.approx(1.0)
    assert wide[4:6].sum() > 0.4 and wide[0] < 0.04                                   # too wide: the outcome sits in the middle of the forecast
    assert narrow[0] + narrow[-1] > 0.5                                                # too narrow: it falls in the tails
    assert S.pit(np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([2.5]))[0] == pytest.approx(0.5)
    assert S.pit(np.array([[1.0, 2.0, 2.0, 4.0]]), np.array([2.0]))[0] == pytest.approx(0.5)        # a tie counts half
    assert np.isnan(S.pit_histogram(np.array([np.nan]))).all()


def test_the_pit_and_coverage_tests_flag_a_forecast_that_is_off_and_pass_one_that_is_not():
    rng = np.random.default_rng(6)
    y = rng.normal(size=1200)
    good, bad = S.pit(rng.normal(size=(1200, 200)), y), S.pit(rng.normal(0, 0.5, (1200, 200)), y)
    assert S.pit_p(good) > 0.01 and S.pit_p(bad) < 1e-6
    hits_good = S.interval_hits(rng.normal(size=(1200, 200)), y, 0.9)
    hits_bad = S.interval_hits(rng.normal(0, 0.5, (1200, 200)), y, 0.9)
    assert S.binomial_p(hits_good, 0.9) > 0.01 and S.binomial_p(hits_bad, 0.9) < 1e-6
    assert np.isnan(S.binomial_p(hits_good[:5], 0.9)) and np.isnan(S.pit_p(good[:5]))


def test_skill_is_the_share_below_the_baseline_and_the_test_sees_a_steady_gain_but_not_noise():
    rng = np.random.default_rng(7)
    base = rng.gamma(4, 1.0, 400)
    better = base * 0.9 * rng.normal(1.0, 0.05, 400)
    noise = base * rng.normal(1.0, 0.3, 400)
    assert S.skill(better, base) == pytest.approx(0.1, abs=0.02) and S.skill(base, base) == 0.0 and S.skill(2 * base, base) == pytest.approx(-1.0)
    assert S.better_than(better, base)["p"] < 1e-6 and S.better_than(noise, base)["p"] > 0.02
    assert np.isnan(S.skill([np.nan], [np.nan])) and np.isnan(S.better_than(better[:5], base[:5])["p"])


def test_the_comparison_test_allows_for_overlapping_forecasts_by_widening_its_variance():
    rng = np.random.default_rng(8)
    steady = np.convolve(rng.normal(size=500), np.ones(8) / 8, mode="valid")           # a loss difference that is autocorrelated over eight lags, as the losses of overlapping forecasts are
    base = np.full(len(steady), 5.0)
    plain, allowed = S.better_than(base + steady - 0.08, base, lag=0), S.better_than(base + steady - 0.08, base, lag=7)
    assert allowed["p"] > plain["p"] and abs(allowed["statistic"]) < abs(plain["statistic"])
    assert S.stride_for(21, 10) == 3 and S.stride_for(21, 21) == 1 and S.stride_for(5, 10) == 1 and S.stride_for(63, 5) == 13


def test_the_average_return_is_compared_with_a_benchmark_and_a_mean_that_is_not_there_scores_below_it():
    rng = np.random.default_rng(9)
    signal = rng.normal(0, 0.01, 600)
    y = signal + rng.normal(0, 0.01, 600)
    good = S.mean_skill(y, signal)                                        # forecasting the true signal beats forecasting zero
    assert good["r2"] > 0.35 and good["p"] < 0.001
    noise_mean = rng.normal(0, 0.01, 600)
    bad = S.mean_skill(y, noise_mean)
    assert bad["r2"] < 0 and bad["p"] > 0.05
    shifted = S.mean_skill(y + 0.004, np.full(600, 0.004), benchmark=np.zeros(600))      # a constant drift that is really there beats zero ...
    assert shifted["r2"] > 0
    assert S.mean_skill(y + 0.004, np.full(600, 0.004), benchmark=np.full(600, 0.004))["r2"] == pytest.approx(0.0, abs=1e-9)      # ... and is no better than the historical average that already has it
    assert np.isnan(S.mean_skill(y[:5], y[:5])["r2"])


def test_the_direction_is_scored_against_always_saying_up_and_a_caller_with_no_skill_does_not_beat_it():
    rng = np.random.default_rng(10)
    y = rng.normal(0.0008, 0.01, 2000)                                       # the market rises a little more often than it falls
    p_random = rng.random(2000)
    r = S.direction_skill(y, p_random)
    assert r["always_up"] == pytest.approx((y > 0).mean()) and r["edge"] < 0.02 and r["p"] > 0.01
    oracle = S.direction_skill(y, (y > 0).astype(float) * 0.9 + 0.05)
    assert oracle["hit"] == 1.0 and oracle["edge"] > 0.4 and oracle["p"] < 1e-6
    always_up = S.direction_skill(y, np.full(2000, 0.7))
    assert always_up["edge"] == 0.0
    assert np.isnan(S.direction_skill(y[:5], p_random[:5])["hit"])


def test_the_holm_adjustment_matches_the_reference_implementation_and_ignores_missing_values():
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.012, np.nan])
    got = S.holm_adjust(p)
    ref = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    sm = pytest.importorskip("statsmodels.stats.multitest")
    ref[ok] = sm.multipletests(p[ok], method="holm")[1]
    assert np.allclose(got[ok], ref[ok]) and np.isnan(got[-1])
    assert S.holm_adjust(np.array([0.2, 0.2, 0.2])) == pytest.approx([0.6, 0.6, 0.6])
    assert (got[ok] >= p[ok]).all()


def test_the_normal_scoring_of_a_gaussian_forecast_agrees_with_the_pit_of_its_draws():
    rng = np.random.default_rng(11)
    y = np.array([-1.5, 0.1, 2.2])
    draws = rng.normal(0.0, 1.0, size=(3, 100_000))
    assert np.allclose(S.pit(draws, y), stats.norm.cdf(y), atol=0.005)
