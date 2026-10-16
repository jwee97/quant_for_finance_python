"""The shared parts of the forecasting package: the history and forecast types, the interpolation of a fan between horizons, the volatility-scaled features, and the conditional simulation of the GARCH models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.econometrics.garch import fit_garch
from src.forecast import base, features
from src.forecast.base import Forecast, ForecastError, History, fill_between, standard_t
from tests.forecast_worlds import garch, iid, to_history


def test_a_history_must_be_a_run_of_positive_finite_prices_and_slices_are_the_days_up_to_a_point():
    with pytest.raises(ForecastError):
        History(np.array([1.0, 2.0]))
    with pytest.raises(ForecastError):
        History(np.array([1.0, np.nan, 2.0, 3.0]))
    with pytest.raises(ForecastError):
        History(np.array([1.0, -2.0, 2.0, 3.0]))
    with pytest.raises(ForecastError):
        History(np.arange(1.0, 11.0), high=np.arange(1.0, 6.0))
    with pytest.raises(ForecastError):
        History(np.arange(1.0, 11.0), index=pd.bdate_range("2020-01-01", periods=5))
    h = to_history(iid(300, seed=1), high_low=True)
    cut = h.upto(99)
    assert cut.n == 100 and np.array_equal(cut.close, h.close[:100]) and np.array_equal(cut.high, h.high[:100]) and cut.index[-1] == h.index[99]
    assert h.tail(50).n == 50 and h.tail(5000).n == h.n and np.array_equal(h.tail(50).close, h.close[-50:])
    assert len(h.r) == h.n - 1 and np.allclose(np.exp(np.cumsum(h.r)) * h.close[0], h.close[1:])


def test_a_frame_with_inconsistent_highs_and_lows_is_used_without_them():
    h = to_history(iid(200, seed=2), high_low=True)
    close = pd.Series(h.close, h.index)
    good = History.from_frame(close, pd.Series(h.high, h.index), pd.Series(h.low, h.index))
    assert good.high is not None
    crossed = History.from_frame(close, pd.Series(h.low, h.index), pd.Series(h.high, h.index))           # the "high" below the "low"
    assert crossed.high is None and crossed.low is None
    holes = pd.Series(h.high, h.index)
    holes.iloc[10] = np.nan
    assert History.from_frame(close, holes, pd.Series(h.low, h.index)).high is None
    assert History.from_frame(close).high is None
    gappy = close.copy()
    gappy.iloc[5] = np.nan
    assert History.from_frame(gappy).n == len(close) - 1


def test_a_forecast_is_a_finite_paths_by_steps_array():
    f = Forecast(np.zeros((5, 3)))
    assert f.steps == 3 and f.at(2).shape == (5,)
    with pytest.raises(ForecastError):
        Forecast(np.array([1.0, 2.0]))
    with pytest.raises(ForecastError):
        Forecast(np.array([[1.0, np.inf]]))


def test_a_fan_between_horizons_keeps_each_anchors_quantiles_and_interpolates_linearly_from_zero():
    rng = np.random.default_rng(0)
    a1, a5, a21 = rng.normal(0, 1, 800), rng.normal(0.5, 2, 800), rng.normal(1, 4, 800)
    out = fill_between({1: a1, 5: a5, 21: a21}, 21)
    assert out.shape == (800, 21)
    for hz, anchor in ((1, a1), (5, a5), (21, a21)):
        assert np.allclose(np.sort(out[:, hz - 1]), np.sort(anchor))
    q = np.sort(out[:, 0]), np.sort(out[:, 4]), np.sort(out[:, 2])
    assert np.allclose(q[2], q[0] + 2 / 4 * (q[1] - q[0]))                       # day 3 is halfway between day 1 and day 5
    zero_start = fill_between({3: a1}, 3)
    assert np.allclose(zero_start[:, 0], np.sort(a1) / 3) and np.allclose(zero_start[:, 1], 2 * np.sort(a1) / 3)       # before the first anchor the quantile function grows from zero in a straight line
    assert np.array_equal(np.argsort(np.argsort(out[:, 20])), np.arange(800)) and (np.diff(out[:, 10]) >= 0).all()     # rows are quantiles, in order
    with pytest.raises(ForecastError):
        fill_between({5: a5}, 21)
    with pytest.raises(ForecastError):
        fill_between({}, 3)


def test_student_t_shocks_have_unit_variance_and_fat_tails_and_the_normal_is_used_for_large_degrees_of_freedom():
    rng = np.random.default_rng(1)
    z = standard_t(rng, (400_000,), 6.0)
    assert z.var() == pytest.approx(1.0, abs=0.05) and abs(z.mean()) < 0.01 and ((z - z.mean()) ** 4).mean() / z.var() ** 2 > 4.0
    big = standard_t(np.random.default_rng(2), (100_000,), 1e6)
    assert np.array_equal(big, np.random.default_rng(2).standard_normal(100_000))
    assert base.cumulate(np.ones((2, 4))).tolist() == [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert base.trailing_rms(np.array([0.01, -0.01] * 50), 100) == pytest.approx(0.01)
    with pytest.raises(ForecastError):
        base.trailing_rms(np.ones(5), 100)


# ---------------------------------------------------------------------------------------------------------------------------------------- features
def test_the_features_use_only_past_prices_and_do_not_depend_on_the_price_level():
    close = to_history(garch(700, seed=3)).close
    X, sigma = features.feature_table(close)
    assert X.shape == (len(close), len(features.FEATURES)) and np.isnan(X[:features.WARMUP]).all() and np.isfinite(X[features.WARMUP:]).all()
    changed = close.copy()
    changed[500:] *= np.exp(np.random.default_rng(0).normal(0, 0.1, len(close) - 500))
    X2, sigma2 = features.feature_table(changed)
    assert np.allclose(X[:500], X2[:500], equal_nan=True) and np.allclose(sigma[:500], sigma2[:500], equal_nan=True), "a later price changed an earlier feature"
    X3, _ = features.feature_table(close * 37.0)
    assert np.allclose(X, X3, equal_nan=True), "the features are in units of volatility and so do not care what the price is in"
    short = features.feature_table(close[:60])
    assert np.isnan(short[0]).all()


def test_a_move_is_described_in_units_of_the_volatility_of_its_time():
    calm, storm = iid(400, 0.004, seed=1), iid(400, 0.02, seed=2)
    r = np.concatenate([calm, storm])
    X, sigma = features.feature_table(100 * np.exp(np.concatenate([[0], np.cumsum(r)])))
    assert sigma[300] < 0.006 and sigma[790] > 0.015                                    # the EWMA volatility follows the regime
    assert abs(X[300:350, 1]).mean() < 3.0 and abs(X[600:700, 1]).mean() < 3.0           # moves of both regimes land on one scale
    assert features.ewma_variance(np.array([0.01] * 40))[-1] == pytest.approx(1e-4)
    assert np.isnan(features.rolling_rms(np.ones(20), 30)).all() and features.rolling_rms(np.ones(20) * 0.02, 5)[-1] == pytest.approx(0.02)
    with pytest.raises(ForecastError):
        features.ewma_variance(np.ones(10))


# ------------------------------------------------------------------------------------------------------------------------------ GARCH simulation
@pytest.mark.parametrize("model, dist", [("garch", "normal"), ("gjr", "t"), ("egarch", "t"), ("tgarch", "normal")])
def test_simulated_paths_have_the_variance_the_model_forecasts_and_are_reproducible(model, dist):
    r = pd.Series(garch(1500, seed=5))
    g = fit_garch(r, model, dist, "zero")
    sims = g.simulate(10, 30_000, seed=1)
    assert sims.shape == (30_000, 10) and np.isfinite(sims).all()
    ratio = (sims ** 2).mean(axis=0) / g.forecast_variance(10, n_sim=30_000, seed=2)
    assert np.allclose(ratio, 1.0, atol=0.06), ratio
    assert np.array_equal(g.simulate(4, 100, seed=7), g.simulate(4, 100, seed=7)) and not np.array_equal(g.simulate(4, 100, seed=7), g.simulate(4, 100, seed=8))


def test_a_model_fitted_earlier_forecasts_from_the_volatility_of_later_days_without_being_refitted():
    r = pd.Series(garch(1600, seed=6))
    g = fit_garch(r.iloc[:1200], "gjr", "t", "zero")
    now = g.simulate(3, 20_000, seed=3, returns=r.iloc[:1200].to_numpy())
    later = g.simulate(3, 20_000, seed=3, returns=r.iloc[:1500].to_numpy())
    calm_stretch = np.concatenate([r.iloc[:1200].to_numpy(), np.full(60, 1e-5)])
    quiet = g.simulate(3, 20_000, seed=3, returns=calm_stretch)
    assert quiet[:, 0].std() < now[:, 0].std() and later[:, 0].std() != now[:, 0].std()
    assert quiet[:, 0].std() < 0.75 * now[:, 0].std()                                     # sixty dead-quiet days cut tomorrow's volatility by a third


def test_resampled_shocks_follow_the_residuals_and_a_bad_pool_or_a_bad_request_is_refused():
    r = pd.Series(garch(1500, seed=7))
    g = fit_garch(r, "gjr", "normal", "zero")
    boot = g.simulate(1, 60_000, seed=2, innovations=g.std_resid.to_numpy())
    z = boot[:, 0] / boot[:, 0].std()
    assert abs(((z - z.mean()) ** 4).mean() / z.var() ** 2 - ((g.std_resid - g.std_resid.mean()) ** 4).mean() / g.std_resid.var() ** 2) < 0.4      # the kurtosis of the residuals carries over
    with pytest.raises(ValueError):
        g.simulate(1, 10, innovations=np.ones(50))
    with pytest.raises(ValueError):
        g.simulate(0, 10)
    with pytest.raises(ValueError):
        g.simulate(5, 0)
