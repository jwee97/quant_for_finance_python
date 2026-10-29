"""The forecasters of ``forecast.more`` against what their sources define: the smoothing recursion and the Theta formula (checked against the two-line definition), the block structure of the stationary
bootstrap, the generalised Pareto tails, and the model each GARCH variant fits."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.forecast import more
from src.forecast.base import ForecastError
from src.forecast.registry import METHODS
from tests import forecast_worlds as W

HS = (1, 5, 21)


def forecast(method, history, n=400, seed=0):
    state = method.fit(history.tail(method.window), HS)
    return state, method.predict(state, history.tail(method.lookback), HS, n, np.random.default_rng(seed))


# ------------------------------------------------------------------------------------------------------------------ Theta
def test_simple_exponential_smoothing_follows_the_recursion_by_hand():
    level, errors = more.ses(np.array([1.0, 2.0, 4.0]), 0.5)
    assert level == pytest.approx(2.75) and errors.tolist() == pytest.approx([1.0, 2.5])        # level 1; error 1, level 1.5; error 2.5, level 2.75


@pytest.mark.parametrize("n, alpha, seed", [(8, 0.4, 2), (12, 0.2, 5), (60, 0.3, 1)])
def test_the_theta_formula_equals_the_average_of_the_two_theta_lines(n, alpha, seed):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0.3, 1.0, n))
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    line0 = intercept + slope * t                                                         # theta = 0: the linear trend
    line2 = 2.0 * y - line0                                                               # theta = 2: twice the curvature
    level2, _ = more.ses(line2, alpha)
    level, _ = more.ses(y, alpha)
    for h in (1, 2, 5, 21):
        two_lines = 0.5 * (intercept + slope * (n - 1 + h)) + 0.5 * level2                # the Assimakopoulos-Nikolopoulos definition
        closed_form = level + more.theta_mean(level, float(y[-1]), slope, alpha, n, np.array([float(h)]))[0] - (level - float(y[-1]))
        assert closed_form == pytest.approx(two_lines, rel=1e-12, abs=1e-12)               # Hyndman and Billah (2003): the same thing as SES with drift


def test_the_theta_forecast_has_the_formulas_mean_and_the_spread_of_its_one_step_errors():
    h = W.to_history(W.iid(900, 0.012, 0.0004, seed=3))
    method = more.Theta()
    state, f = forecast(method, h, n=20_000)
    y = np.log(h.close[-method.lookback:])
    level, errors = more.ses(y, state["alpha"])
    slope = np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]
    expected = more.theta_mean(level, float(y[-1]), slope, state["alpha"], len(y), np.arange(1.0, 22.0))
    sigma = float(np.sqrt(np.mean(errors ** 2)))
    tolerance = 4.0 * sigma * np.sqrt(np.arange(1, 22)) / np.sqrt(20_000)                  # four standard errors of the mean of 20,000 paths at each step
    assert (np.abs(f.samples.mean(axis=0) - expected) < tolerance).all()
    assert f.samples[:, 0].std() == pytest.approx(sigma, rel=0.03) and f.samples[:, 20].std() == pytest.approx(sigma * np.sqrt(21), rel=0.03)


def test_theta_chooses_a_low_smoothing_weight_for_a_noisy_level_and_a_high_one_for_a_random_walk():
    rng = np.random.default_rng(4)
    walk = np.cumsum(rng.normal(0, 0.01, 800))
    noisy = np.cumsum(rng.normal(0, 0.002, 800)) + rng.normal(0, 0.03, 800)                   # a slowly moving level seen through large noise
    alpha = lambda path: more.Theta().fit(W.to_history(np.diff(path)), HS)["alpha"]           # noqa: E731
    assert alpha(walk) > 0.9 > 0.3 > alpha(noisy)


def test_theta_refuses_a_constant_price():
    with pytest.raises(ForecastError):
        more.Theta().fit(W.to_history(np.zeros(700)), HS)


# ------------------------------------------------------------------------------------------------------------------ stationary bootstrap
def test_the_bootstrap_draws_only_from_the_demeaned_returns_and_keeps_the_blocks_together():
    h = W.to_history(W.garch(1500, seed=2))
    method = more.StationaryBootstrap()
    state, f = forecast(method, h, n=3000)
    r, block = state["r"], state["block"]
    assert block == round(len(h.tail(method.window).r) ** (1 / 3)) and abs(r.mean()) < 1e-15
    daily = np.diff(np.concatenate([np.zeros((3000, 1)), f.samples], axis=1), axis=1)            # back to the daily returns
    position = {round(v, 14): i for i, v in enumerate(r)}
    index = np.array([[position[round(v, 14)] for v in row] for row in daily[:300]])
    assert set(map(float, np.unique(daily[:300].round(14)))) <= {round(v, 14) for v in r}         # every draw is one of the observed (demeaned) returns
    follows = (index[:, 1:] == (index[:, :-1] + 1) % len(r)).mean()
    assert follows == pytest.approx(1.0 - 1.0 / block, abs=0.02)                                   # a block continues with probability 1 - 1/block: geometric lengths of mean `block`


def test_the_bootstrap_has_no_drift_and_the_samples_spread_of_the_sample():
    h = W.to_history(W.iid(1400, 0.011, 0.0007, seed=8))
    state, f = forecast(more.StationaryBootstrap(), h, n=20_000)
    assert abs(f.at(21).mean()) < 5 * f.at(21).std() / np.sqrt(20_000)
    assert f.samples[:, 0].std() == pytest.approx(state["r"].std(), rel=0.03)
    with pytest.raises(ForecastError):
        more.StationaryBootstrap().fit(W.to_history(np.zeros(900)), HS)


# ------------------------------------------------------------------------------------------------------------------ the GARCH family
@pytest.fixture(scope="module")
def stock():
    return W.to_history(W.garch(1700, seed=21), high_low=True)


def test_each_garch_variant_fits_the_model_it_is_named_for(stock):
    plain, _ = forecast(more.Garch11MC(), stock)
    gjr, _ = forecast(METHODS["garch"], stock)
    expo, f = forecast(more.EGarchMC(), stock)
    assert (plain.model, gjr.model, expo.model) == ("garch", "gjr", "egarch")
    assert plain.params.get("gamma", 0.0) == 0.0 and gjr.params["gamma"] > 0.0               # the world has a leverage effect: the GJR model finds it, the plain model has no term for it
    assert expo.params["gamma"] < 0.0                                                         # in EGARCH a negative gamma is the same leverage effect
    assert f.info["persistence"] == pytest.approx(expo.params["beta"])
    assert gjr.loglik > plain.loglik                                                          # and it fits better (the nested model's likelihood is never lower)


def test_the_symmetric_model_cannot_tell_a_fall_from_a_rise_and_the_leverage_models_can(stock):
    from src.econometrics.garch import fit_garch

    r = pd.Series(stock.r)
    gjr, plain = fit_garch(r, "gjr", "t", "zero"), fit_garch(r, "garch", "t", "zero")
    shock = 0.03 * r.std() / 0.01
    up = gjr.forecast_variance(1, returns=np.append(stock.r, +shock))[0]
    down = gjr.forecast_variance(1, returns=np.append(stock.r, -shock))[0]
    assert down > up                                                                          # a fall raises tomorrow's variance more than a rise of the same size
    sym_up = plain.forecast_variance(1, returns=np.append(stock.r, +shock))[0]
    sym_down = plain.forecast_variance(1, returns=np.append(stock.r, -shock))[0]
    assert sym_up == pytest.approx(sym_down, rel=1e-9)


# ------------------------------------------------------------------------------------------------------------------ extreme-value tails
def test_the_tail_pool_keeps_the_body_and_draws_each_tail_from_the_fitted_generalised_pareto():
    from scipy.stats import genpareto, kstest

    rng = np.random.default_rng(6)
    z = rng.standard_t(4, 3000) * np.sqrt(2.0 / 4.0)                                           # fat-tailed standardised residuals
    method = more.EVTFilteredHistorical()
    pool = method._innovations(SimpleNamespace(std_resid=pd.Series(z)))
    lo, hi = np.quantile(z, [0.10, 0.90])
    assert len(pool) == pytest.approx(method.DRAWS, rel=0.01)
    body = pool[(pool >= lo) & (pool <= hi)]
    assert len(body) / len(pool) == pytest.approx(0.80, abs=0.01) and set(np.round(body, 12)) <= set(np.round(z[(z >= lo) & (z <= hi)], 12))
    assert (pool > hi).mean() == pytest.approx(0.10, abs=0.01) and (pool < lo).mean() == pytest.approx(0.10, abs=0.01)
    for side, exceed in (("upper", pool[pool > hi] - hi), ("lower", lo - pool[pool < lo])):
        sample_exceed = z[z > hi] - hi if side == "upper" else lo - z[z < lo]
        shape, _, scale = genpareto.fit(sample_exceed, floc=0.0)
        fitted = genpareto(float(np.clip(shape, -0.25, 0.5)), scale=scale)
        assert kstest(exceed, fitted.cdf).pvalue > 0.01, side                                   # the pool's tail IS the fitted distribution, drawn 2,000 times
    assert pool.max() > hi + 3 * (np.quantile(z, 0.99) - hi)                                     # and it runs far out into the tail


def test_the_tail_pool_needs_enough_extreme_shocks():
    with pytest.raises(ForecastError):
        more.EVTFilteredHistorical()._innovations(SimpleNamespace(std_resid=pd.Series(np.random.default_rng(1).normal(size=100))))


def test_the_new_methods_are_listed_where_the_form_expects_them():
    ids = list(METHODS)
    assert ids.index("garch11") < ids.index("garch") < ids.index("egarch") < ids.index("fhs") < ids.index("evt") and ids.index("bootstrap") < ids.index("garch11")
    for name in ("garch11", "egarch", "evt", "bootstrap", "theta"):
        m = METHODS[name]
        assert m.label and m.text and m.assumes and m.fails and m.family in ("Monte Carlo", "Time series")
    assert METHODS["theta"].mean_signal and not METHODS["bootstrap"].mean_signal
