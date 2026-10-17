"""Forecasters built on the option-model machinery: heavy-tailed shocks from a fitted Levy law, a stochastic-volatility state-space model, and the Heston model.

``garch_vg``, ``garch_nig``  the volatility model of ``garch`` with shocks from a variance-gamma or a normal-inverse-Gaussian law fitted to the model's own standardised residuals
                             (``derivatives.levy``). Both laws have the skew and the kurtosis as free parameters (the normal has neither, the Student-t has a kurtosis but no skew), and unlike
                             the resampling of ``fhs`` they have smooth tails that go beyond the largest shock in the sample.
``sv_kalman``                 a stochastic-volatility model whose log variance is a latent AR(1) (``econometrics.trackers.sv_tracker``, the quasi-maximum-likelihood method of Harvey, Ruiz and
                             Shephard 1994): the Kalman filter tracks today's log variance and says how uncertain it is, and the forecast simulates the volatility forward from that uncertain
                             state. Volatility has its own noise here, which GARCH (where volatility is a function of past returns) does not have.
``heston``                   the Heston (1993) model under the real-world measure, simulated with Andersen's quadratic-exponential scheme (``probability.sde``). Its parameters come from the method of
                             moments (``heston_moments``): the autocorrelation of squared returns decays at the speed of mean reversion and its level gives the variance of the variance, and the covariance of a return
                             with later squared returns gives the correlation. (Translating a fitted GARCH through its diffusion limit was tried first and rejected: the GARCH persistence of Heston data is
                             too close to one, and the speed it implies is several times too slow.) A continuous-time stochastic-volatility forecast with square-root noise and a return-variance correlation.

As in ``montecarlo``, none of them forecasts a direction: the average return is set to zero, so they are tested on the spread of what might happen.
"""

from __future__ import annotations

import numpy as np

from ..derivatives import levy
from ..econometrics import trackers
from ..probability import sde
from .base import Forecast, ForecastError, Forecaster, History, cumulate
from .montecarlo import GarchMC

TRADING_DAYS = 252
SCALE = 0.01                                              # the standardised residuals are fitted as one-day returns of a one-per-cent volatility


# ---------------------------------------------------------------------------------------------------------------------------------- GARCH with Levy shocks
class GarchLevy(GarchMC):
    """The GJR-GARCH volatility model of ``GarchMC`` (fitted by normal quasi-maximum likelihood) with shocks drawn from a Levy law fitted to its standardised residuals."""

    dist = "normal"
    law = "vg"
    pool_size = 200_000
    fit_bounds = {"theta": (-12.0, 12.0)}                                 # the skew of a daily shock needs a far larger drift parameter than the volatilities of an option market do
    heavy = True                                                          # the volatility fit and the fit of the shock law together take a second or more

    def fit(self, h: History, horizons):
        g = super().fit(h, horizons)
        z = g.std_resid.to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        z = (z - z.mean()) / z.std()
        try:
            fit = levy.fit_returns(SCALE * z, self.law, dt=1.0 / TRADING_DAYS, bounds=self.fit_bounds)
        except (ValueError, FloatingPointError) as error:
            raise ForecastError(f"the {self.law} law could not be fitted to the standardised shocks: {error}") from None
        if not fit.converged and not np.isfinite(fit.loglik):
            raise ForecastError(f"the {self.law} fit did not converge")
        rng = np.random.default_rng(12345)                 # the pool is a property of the fitted law: the draws that use it differ with the forecast's own generator
        draws = fit.model.sample(1.0 / TRADING_DAYS, self.pool_size, rng)
        draws = (draws - draws.mean()) / draws.std()
        moments = fit.model.moments()                                  # per unit time: the skewness of an increment over dt is these over sqrt(dt), the excess kurtosis over dt
        g.levy_pool, g.levy_fit = draws, fit
        g.shock_skew, g.shock_excess_kurtosis = float(moments["skewness"] * np.sqrt(TRADING_DAYS)), float(moments["excess_kurtosis"] * TRADING_DAYS)
        return g

    def _innovations(self, g):
        return g.levy_pool

    def predict(self, state, h: History, horizons, n, rng):
        forecast = super().predict(state, h, horizons, n, rng)
        g = state
        forecast.info.update({"shock_skewness": g.shock_skew, "shock_excess_kurtosis": g.shock_excess_kurtosis, "shock_law": self.law, "shock_law_parameters": dict(g.levy_fit.model.params)})
        return forecast


class GarchVG(GarchLevy):
    name = "garch_vg"
    short = "GARCH, variance gamma"
    label = "GJR-GARCH with variance-gamma shocks (Monte Carlo)"
    text = "The GARCH volatility model with shocks from a variance-gamma law fitted to its own residuals: a skew and a kurtosis the data choose, and tails that go past the worst day in the sample."
    assumes = "The volatility model is right and the standardised shocks are independent draws from one skewed, fat-tailed law of the variance-gamma kind."
    fails = "The variance-gamma tails are lighter than a power law's: a crash far outside the sample is still too unlikely; all the errors of the volatility model are inherited."
    law = "vg"


class GarchNIG(GarchLevy):
    name = "garch_nig"
    short = "GARCH, NIG"
    label = "GJR-GARCH with normal-inverse-Gaussian shocks (Monte Carlo)"
    text = "The same volatility model with shocks from a normal-inverse-Gaussian law: a little heavier in the tails than the variance gamma for the same variance, with its own skew."
    assumes = "The volatility model is right and the standardised shocks are independent draws from one skewed, fat-tailed law of the NIG kind."
    fails = "Its tails are exponential, not power-law: an extreme crash is still treated as rarer than it can be; the volatility model's errors are inherited."
    law = "nig"


# ---------------------------------------------------------------------------------------------------------------------------------- stochastic volatility by Kalman filter
class SVKalman(Forecaster):
    name = "sv_kalman"
    short = "Stochastic volatility"
    label = "Stochastic volatility, Kalman-filtered log variance (Monte Carlo)"
    family = "State space"
    text = "The log variance is a hidden AR(1): a Kalman filter tracks it from the log squared returns, and the forecast simulates the volatility forward from the filter's uncertain state."
    assumes = "Volatility has a mind of its own: a persistent process with its own noise that returns only reveal imperfectly."
    fails = "The quasi-likelihood treats the noise of a log squared return as normal when it is skewed (so the fit is a little biased); there is no leverage, so a fall raises volatility no more than a rise."
    min_history = 500
    window = 1260

    def fit(self, h: History, horizons):
        r = h.r[-self.window:]
        if len(r) < 400 or float(np.std(r)) <= 0:
            raise ForecastError("too few, or constant, returns for a volatility model")
        try:
            track = trackers.sv_tracker(r)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            raise ForecastError(f"the stochastic-volatility model could not be fitted: {error}") from None
        if not np.isfinite(track.loglik) or not 0.0 < track.phi < 1.0:
            raise ForecastError("the stochastic-volatility fit did not give a stationary, persistent log variance")
        return {"mu": track.mu, "phi": track.phi, "sigma_eta": track.sigma_eta, "offset": track.offset}

    def predict(self, state, h: History, horizons, n, rng):
        r = h.r[-self.lookback:]
        y = np.log(r ** 2 + state["offset"])
        hf, pf, _ = trackers._sv_filter(y, state["mu"], state["phi"], state["sigma_eta"])
        steps = max(horizons)
        mu, phi, s_eta = state["mu"], state["phi"], state["sigma_eta"]
        logv = rng.normal(hf[-1], np.sqrt(max(pf[-1], 0.0)), n)
        daily = np.empty((n, steps))
        for k in range(steps):
            logv = mu + phi * (logv - mu) + s_eta * rng.standard_normal(n)
            daily[:, k] = np.exp(0.5 * logv) * rng.standard_normal(n)
        stationary = s_eta ** 2 / (1.0 - phi ** 2)
        return Forecast(cumulate(daily), {"persistence": phi, "volatility_of_log_variance": s_eta, "volatility_today_annual": float(np.exp(0.5 * hf[-1] + 0.125 * pf[-1]) * np.sqrt(TRADING_DAYS)),
                                          "long_run_volatility_annual": float(np.exp(0.5 * mu + 0.125 * stationary) * np.sqrt(TRADING_DAYS)), "log_variance_uncertainty": float(np.sqrt(max(pf[-1], 0.0))),
                                          "half_life_days": float(np.log(0.5) / np.log(phi))})


# ---------------------------------------------------------------------------------------------------------------------------------- Heston by the method of moments
def _acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = x - x.mean()
    d = float(x @ x)
    return np.array([float(x[:-k] @ x[k:]) / d for k in range(1, lags + 1)])


def heston_moments(r: np.ndarray, dt: float = 1.0 / TRADING_DAYS, lags: int = 60, leverage_lags: int = 20) -> dict:
    """The parameters of a Heston model by the method of moments on daily returns ``r`` (mean zero): the long-run variance is the mean square return per year; the autocorrelations of the
    squared returns decay as ``A exp(-kappa k dt)`` with ``A = Var(v) / (2 theta^2 + 3 Var(v))``, which gives the speed ``kappa`` and the variance of the variance, hence the volatility of variance
    ``xi^2 = 2 kappa Var(v) / theta``; and the covariance of a return with the squared return ``k`` days later is ``rho xi theta dt^2 exp(-kappa (k - 1) dt)``, which gives the correlation."""
    from scipy import optimize

    r = np.asarray(r, dtype=float)
    theta = float(np.mean(r ** 2) / dt)
    a = _acf(r ** 2, lags)
    ks = np.arange(1, lags + 1)
    fit = optimize.least_squares(lambda p: p[0] * np.exp(-p[1] * ks * dt) - a, [0.15, 5.0], bounds=([0.0, 0.5], [1.0, 80.0]))
    if fit.x[0] < 1.5 / np.sqrt(len(r)):                                      # the autocorrelations of the squares are within sampling noise: no clustering to model, a constant variance
        return {"kappa": 5.0, "theta": theta, "xi": 1e-3, "rho": 0.0, "amplitude": float(fit.x[0])}
    amp, kappa = float(np.clip(fit.x[0], 0.02, 0.30)), float(fit.x[1])
    var_v = 2.0 * theta ** 2 * amp / (1.0 - 3.0 * amp)
    xi = float(np.sqrt(2.0 * kappa * var_v / theta))
    lk = np.arange(1, leverage_lags + 1)
    cov = np.array([float(np.mean(r[:-k] * r[k:] ** 2)) for k in lk])
    shape = xi * theta * dt ** 2 * np.exp(-kappa * (lk - 1) * dt)
    rho = float(np.clip((cov @ shape) / (shape @ shape), -0.95, 0.3)) if xi > 0 else 0.0
    return {"kappa": kappa, "theta": theta, "xi": max(xi, 1e-3), "rho": rho, "amplitude": amp}


class HestonMC(GarchMC):
    """The Heston model with parameters from the method of moments (``heston_moments``), simulated by the quadratic-exponential scheme from today's variance (the fitted GARCH's, for which the
    parent class is kept)."""

    name = "heston"
    short = "Heston"
    label = "Heston stochastic volatility, by the method of moments (Monte Carlo)"
    text = "Variance follows a square-root process that reverts to a long-run level and moves against the return; its parameters come from the autocorrelation of squared returns and their leverage, and it is simulated with Andersen's quadratic-exponential scheme."
    assumes = "Variance is a mean-reverting square-root diffusion whose shocks are correlated (negatively) with the returns', and the squared returns' autocorrelations say how."
    fails = "The moments of squared returns are noisy (heavy tails), so the speed and the volatility of variance are rough; the square-root noise gives a thinner far tail than the heavy-tailed shocks of real days."
    dist = "t"

    def fit(self, h: History, horizons):
        g = super().fit(h, horizons)                                          # the GARCH fit supplies today's variance at the forecast origin
        g.heston = heston_moments(h.r[-self.window:])
        return g

    def predict(self, state, h: History, horizons, n, rng):
        g = state
        pr = g.heston
        steps = max(horizons)
        v0 = float(g.forecast_variance(1, returns=h.r[-self.lookback:])[0]) * TRADING_DAYS                 # tomorrow's variance from the model brought up to the origin, annualised
        v0 = float(np.clip(v0, 0.05 * pr["theta"], 20.0 * pr["theta"]))
        try:
            S, _ = sde.simulate_heston(1.0, v0, 0.0, 0.0, pr["kappa"], pr["theta"], pr["xi"], pr["rho"], steps / TRADING_DAYS, steps, n, "qe", rng)
        except ValueError as error:
            raise ForecastError(f"the Heston simulation failed: {error}") from None
        x = np.log(S[:, 1:])                                                # cumulative log returns of a martingale price: centre them (the average log return is set to zero)
        x = x - x.mean(axis=0, keepdims=True)
        return Forecast(x, {"kappa": pr["kappa"], "long_run_volatility_annual": float(np.sqrt(pr["theta"])), "volatility_of_variance": pr["xi"], "leverage_correlation": pr["rho"],
                            "volatility_today_annual": float(np.sqrt(v0)), "feller_ratio": float(2.0 * pr["kappa"] * pr["theta"] / pr["xi"] ** 2), "half_life_days": float(np.log(2.0) / pr["kappa"] * TRADING_DAYS)})

