"""More ways of forecasting a price distribution, each from a standard text or paper.

``garch11``  the plain symmetric GARCH(1,1) with Student-t shocks (Bollerslev 1986): the benchmark of Hansen and Lunde's (2005) comparison of 330 volatility models, which found nothing that beat it in exchange
             rates but found that a model with a leverage effect beats it for a stock. Having both beside ``garch`` (GJR) lets that finding be checked on your stock.
``egarch``   Nelson's (1991) exponential GARCH with Student-t shocks: the log of the variance follows the recursion, so no positivity limits are needed and a fall can raise volatility by more than a rise lowers it.
``evt``      conditional extreme-value theory (McNeil and Frey 2000): the volatility model of ``fhs``, but the 10% of standardised residuals in each tail are replaced by a generalised Pareto
             distribution fitted to them, so a forecast can reach beyond the worst shock in the sample, which resampling cannot.
``bootstrap`` the stationary bootstrap (Politis and Romano 1994): resample the last years' returns in blocks of random length (geometric, the mean about the cube root of the sample size) so that runs of calm
             and stormy days stay together; the mean is removed first, so it forecasts a spread and no direction.
``theta``    the Theta method (Assimakopoulos and Nikolopoulos 2000), the winner of the M3 forecasting competition and a benchmark of Hyndman and Athanasopoulos' *Forecasting: Principles and Practice*:
             simple exponential smoothing of the log price plus half the slope of its linear trend (Hyndman and Billah 2003 show this is exactly SES with drift), with the spread of the one-step errors.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize
from scipy.stats import genpareto

from .base import Forecast, ForecastError, Forecaster, History, cumulate
from .montecarlo import TRADING_DAYS, FilteredHistorical, GarchMC


class Garch11MC(GarchMC):
    name = "garch11"
    short = "GARCH(1,1)"
    label = "GARCH(1,1) with Student-t shocks (Monte Carlo)"
    family = "Monte Carlo"
    text = "The textbook volatility model: volatility clusters and reverts, a fall and a rise of the same size have the same effect, shocks have fat tails."
    assumes = "Volatility is a persistent process that reverts to a long-run level, and up and down moves matter equally."
    fails = "For a stock a fall raises volatility more than a rise does (the leverage effect), which this model cannot say; a regime change is a surprise."
    model = "garch"


class EGarchMC(GarchMC):
    name = "egarch"
    short = "EGARCH"
    label = "EGARCH with Student-t shocks (Monte Carlo)"
    family = "Monte Carlo"
    text = "The log of the variance follows a recursion, so a fall can raise volatility more than a rise lowers it, with no limits on the parameters; shocks have fat tails."
    assumes = "Volatility reacts to the size and sign of yesterday's shock in the way a log-linear recursion allows, and reverts at a fixed pace."
    fails = "The log form makes the forecast variance sensitive to the largest shocks in the fit, and the long-run level is hard to pin down; a regime change is a surprise."
    model = "egarch"


class EVTFilteredHistorical(FilteredHistorical):
    name = "evt"
    short = "Filtered EVT"
    label = "GARCH volatility with extreme-value tails (conditional EVT)"
    family = "Monte Carlo"
    text = "The same volatility model as filtered historical simulation, but the worst tenth of shocks on each side is replaced by a generalised Pareto tail fitted to it, so it can reach beyond the worst day it has seen."
    assumes = "The volatility model is right and the largest shocks, once scaled by it, follow a power-law tail, which extreme-value theory says is the shape of any tail."
    fails = "The tail has few points to fit (about fifty on each side), so its shape is uncertain; a shape that is too heavy gives forecasts with an infinite variance."
    TAIL = 0.10
    DRAWS = 20000
    MIN_EXCEEDANCES = 30

    def _innovations(self, g):
        z = g.std_resid.to_numpy()
        z = z[np.isfinite(z)]
        lo, hi = np.quantile(z, [self.TAIL, 1.0 - self.TAIL])
        upper, lower = z[z > hi] - hi, lo - z[z < lo]
        if len(upper) < self.MIN_EXCEEDANCES or len(lower) < self.MIN_EXCEEDANCES:
            raise ForecastError("too few extreme shocks to fit the tails")
        rng = np.random.default_rng(len(z))
        body = rng.choice(z[(z >= lo) & (z <= hi)], int(self.DRAWS * (1.0 - 2.0 * self.TAIL)))
        draws = [body]
        for exceed, sign, anchor in ((upper, 1.0, hi), (lower, -1.0, lo)):
            shape, _, scale = genpareto.fit(exceed, floc=0.0)
            shape = float(np.clip(shape, -0.25, 0.5))                                 # a shape above one half has no variance; the data rarely say more than 0.3
            draws.append(anchor + sign * genpareto.rvs(shape, scale=scale, size=int(self.DRAWS * self.TAIL), random_state=rng))
        return np.concatenate(draws)


class StationaryBootstrap(Forecaster):
    name = "bootstrap"
    short = "Stationary bootstrap"
    label = "Stationary bootstrap of daily returns (Monte Carlo)"
    family = "Monte Carlo"
    text = "Resample the last five years of daily returns in blocks of random length, so that runs of calm and of stormy days stay together; the average return is removed first."
    assumes = "The next days will look like the last five years in pieces: clustering within a block is kept and anything longer than a block is not."
    fails = "It cannot produce a shock bigger than the largest in the sample, a volatility level that has not been seen, or a trend longer than a block."
    min_history = 500

    def fit(self, h: History, horizons):
        r = h.r[-self.window:]
        if len(r) < 400 or float(r.std()) <= 0:
            raise ForecastError("too few, or constant, returns to resample")
        block = max(2, int(round(len(r) ** (1.0 / 3.0))))
        return {"r": r - r.mean(), "block": block}

    def predict(self, state, h: History, horizons, n, rng):
        r, block = state["r"], state["block"]
        steps, length = max(horizons), len(r)
        starts = rng.integers(0, length, (n, steps))
        restart = rng.random((n, steps)) < 1.0 / block
        restart[:, 0] = True
        index = np.empty((n, steps), dtype=int)
        for k in range(steps):
            index[:, k] = np.where(restart[:, k], starts[:, k], (index[:, k - 1] + 1) % length) if k else starts[:, 0]
        return Forecast(cumulate(r[index]), {"mean_block_days": block, "days_resampled": length, "annual_volatility": float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))})


def ses(y: np.ndarray, alpha: float, start: float | None = None) -> tuple[float, np.ndarray]:
    """Simple exponential smoothing of ``y``: the final level and the one-step forecast errors (the first value starts the level, as in the Theta method's textbook form)."""
    level = float(y[0]) if start is None else float(start)
    err = np.empty(len(y) - 1)
    for t in range(1, len(y)):
        err[t - 1] = y[t] - level
        level += alpha * err[t - 1]
    return level, err


def theta_mean(level: float, last: float, slope: float, alpha: float, n_obs: int, steps: np.ndarray) -> np.ndarray:
    """The Theta forecast of the log return from the last price: ``level - last + (b / 2)(h - 1 + 1/alpha - (1 - alpha)^n / alpha)`` at each horizon ``h`` (Hyndman and Billah 2003, eq. 5), where
    ``level`` is the smoothed level, ``b`` the slope of the linear trend of the series and ``n`` its length."""
    return level - last + 0.5 * slope * (steps - 1.0 + 1.0 / alpha - (1.0 - alpha) ** n_obs / alpha)


class Theta(Forecaster):
    name = "theta"
    short = "Theta"
    label = "Theta method (exponential smoothing with half the trend)"
    family = "Time series"
    text = "Simple exponential smoothing of the log price plus half the slope of its straight-line trend: the method that won the M3 competition and the benchmark of the forecasting textbooks."
    assumes = "A series has a level that moves in a smoothed way and a trend that should be followed, but only halfway."
    fails = "A stock's trend is too small to estimate from two years of daily prices, so what the method adds to a random walk is mostly noise, and its spread is constant."
    min_history = 400
    mean_signal = True
    window = 500

    def fit(self, h: History, horizons):
        y = np.log(h.close[-self.window:])
        if len(y) < 250 or float(np.std(np.diff(y))) <= 0:
            raise ForecastError("too few, or constant, prices for exponential smoothing")
        res = optimize.minimize_scalar(lambda a: float(np.sum(ses(y, a)[1] ** 2)), bounds=(0.01, 0.99), method="bounded")
        return {"alpha": float(res.x)}

    def predict(self, state, h: History, horizons, n, rng):
        alpha = state["alpha"]
        y = np.log(h.close[-self.lookback:])
        level, errors = ses(y, alpha)
        t = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(t, y, 1)[0])
        steps = np.arange(1, max(horizons) + 1, dtype=float)
        mean = theta_mean(level, float(y[-1]), slope, alpha, len(y), steps)
        sigma = float(np.sqrt(np.mean(errors ** 2)))
        daily = np.diff(np.concatenate([[0.0], mean]))[None, :] + sigma * rng.standard_normal((n, len(steps)))
        return Forecast(cumulate(daily), {"smoothing_alpha": alpha, "trend_per_day": slope, "level_gap": float(level - y[-1]), "annual_volatility": sigma * math.sqrt(TRADING_DAYS)})
