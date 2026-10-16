"""The simplest forecasters. They are the yardsticks: a method is worth running only if it beats the first of them.

``random_walk``   no view on direction, the spread of the last year. The benchmark.
``riskmetrics``   the same, with the spread from an exponentially weighted average of squared returns (JP Morgan's RiskMetrics, decay 0.94): it reacts within weeks to a change in volatility.
``historical``    historical simulation: resample past days in blocks, so the shape (fat tails, skew) and the clustering of the past five years carry over, with their average return.
``hull_white``    volatility-scaled historical simulation (Hull and White 1998): past returns divided by the volatility of their day and multiplied by today's, then resampled in blocks. It is the control for
                  the methods that learn from returns in units of volatility (the Gaussian process, the analogues): what they add beyond this is what they learn about the future.
``gbm``           geometric Brownian motion with the drift and volatility of the last year, the Monte Carlo of the textbooks: it takes the drift of a year to be a forecast of the next.
"""

from __future__ import annotations

import numpy as np

from .base import Forecast, ForecastError, Forecaster, History, cumulate, trailing_rms
from .features import LAMBDA, ewma_variance

TRADING_DAYS = 252


class RandomWalk(Forecaster):
    name = "random_walk"
    short = "Random walk"
    label = "Random walk, constant volatility"
    family = "baseline"
    text = "No view on direction and a spread taken from the last year of returns. The benchmark every other method has to beat."
    assumes = "Returns are independent and equally volatile, and the last year is typical of the next weeks."
    fails = "Too narrow when a calm spell ends in a storm; too wide for months after one."
    min_history = 260

    def __init__(self, days: int = TRADING_DAYS):
        self.days = int(days)

    def fit(self, h: History, horizons):
        return None

    def predict(self, state, h: History, horizons, n, rng):
        sigma = trailing_rms(h.r, self.days)
        steps = max(horizons)
        return Forecast(cumulate(sigma * rng.standard_normal((n, steps))), {"daily_volatility": sigma, "annual_volatility": sigma * np.sqrt(TRADING_DAYS)})


class RiskMetrics(Forecaster):
    name = "riskmetrics"
    short = "RiskMetrics"
    label = "Exponentially weighted volatility (RiskMetrics)"
    family = "Volatility"
    text = "No view on direction; the spread follows an exponentially weighted average of recent squared returns, so it reacts within weeks."
    assumes = "Volatility is persistent and follows a random walk (no pull back to a long-run level), and shocks are normal."
    fails = "Never returns to normal on its own, so it stays too high long after a storm; the normal shocks make the tails too thin."
    min_history = 260

    def __init__(self, lam: float = LAMBDA):
        self.lam = float(lam)

    def fit(self, h: History, horizons):
        return None

    def predict(self, state, h: History, horizons, n, rng):
        r = h.r[-self.lookback:]
        if len(r) < 60:
            raise ForecastError("too few days for an exponentially weighted volatility")
        sigma = float(np.sqrt(ewma_variance(r, self.lam)[-1]))
        return Forecast(cumulate(sigma * rng.standard_normal((n, max(horizons)))), {"daily_volatility": sigma, "annual_volatility": sigma * np.sqrt(TRADING_DAYS)})


class Historical(Forecaster):
    name = "historical"
    short = "Historical sim."
    label = "Historical simulation (resampled blocks of past days)"
    family = "Monte Carlo"
    text = "Draws blocks of consecutive days from the last five years, so fat tails, skew and some clustering come along, and so does the average return of those years."
    mean_signal = True
    assumes = "The next weeks look like a random reshuffling of the last five years, in blocks of about ten days."
    fails = "It cannot know the market has changed: after a calm five years it has no storms to draw on, and it carries the old average return forward."
    min_history = 500

    def __init__(self, block: float = 10.0):
        self.block = float(block)

    def fit(self, h: History, horizons):
        r = h.r[-self.window:]
        if len(r) < 250:
            raise ForecastError("too few days to resample")
        return {"r": r}

    def predict(self, state, h: History, horizons, n, rng):
        r = state["r"]
        m, steps = len(r), max(horizons)
        idx = np.empty((n, steps), dtype=int)
        idx[:, 0] = rng.integers(0, m, n)
        jump = rng.random((n, steps)) < 1.0 / self.block
        fresh = rng.integers(0, m, (n, steps))
        for k in range(1, steps):
            idx[:, k] = np.where(jump[:, k], fresh[:, k], (idx[:, k - 1] + 1) % m)
        return Forecast(cumulate(r[idx]), {"days_resampled": int(m), "block_days": self.block, "annual_drift": float(r.mean() * TRADING_DAYS), "annual_volatility": float(r.std() * np.sqrt(TRADING_DAYS))})


class GBM(Forecaster):
    name = "gbm"
    short = "Brownian motion"
    label = "Geometric Brownian motion (Monte Carlo)"
    family = "Monte Carlo"
    text = "The textbook model: normal returns with the drift and volatility of the last year, simulated forward."
    assumes = "A year's average return is a fair forecast of the next weeks' average return."
    fails = "A year of returns says almost nothing about the next month's average (the noise in it is as big as the equity premium), so it adds a random shift to every forecast."
    min_history = 260
    mean_signal = True

    def __init__(self, days: int = TRADING_DAYS):
        self.days = int(days)

    def fit(self, h: History, horizons):
        return None

    def predict(self, state, h: History, horizons, n, rng):
        r = h.r[-self.days:]
        if len(r) < 60:
            raise ForecastError("too few days for a drift and a volatility")
        mu, sigma = float(r.mean()), float(r.std(ddof=1))
        return Forecast(cumulate(mu + sigma * rng.standard_normal((n, max(horizons)))), {"annual_drift": mu * TRADING_DAYS, "annual_volatility": sigma * np.sqrt(TRADING_DAYS)})


class HullWhite(Forecaster):
    name = "hull_white"
    short = "Hull-White"
    label = "Volatility-scaled historical simulation (Hull-White)"
    family = "Monte Carlo"
    text = "Past returns rescaled to today's volatility (each divided by the volatility of its day, multiplied by today's) and resampled in blocks: the shape of the history with the current spread."
    assumes = "Returns are a stable shape times a volatility that an exponentially weighted average tracks, and today's volatility holds for the next weeks."
    fails = "Like the plain historical simulation it carries the average return of the last five years; and the volatility never reverts, so a storm's spread is held for the whole horizon."
    min_history = 500
    mean_signal = True

    def __init__(self, block: float = 10.0, lam: float = LAMBDA):
        self.block, self.lam = float(block), float(lam)

    def fit(self, h: History, horizons):
        r = h.r[-self.window:]
        if len(r) < 250:
            raise ForecastError("too few days to resample")
        v = ewma_variance(r, self.lam)
        z = r[31:] / np.sqrt(v[30:-1])                                     # a day's return over the volatility known before it (the first month only seeds the average)
        return {"z": z}

    def predict(self, state, h: History, horizons, n, rng):
        z = state["z"]
        r = h.r[-self.lookback:]
        if len(r) < 60:
            raise ForecastError("too few days for an exponentially weighted volatility")
        sigma = float(np.sqrt(ewma_variance(r, self.lam)[-1]))
        m, steps = len(z), max(horizons)
        idx = np.empty((n, steps), dtype=int)
        idx[:, 0] = rng.integers(0, m, n)
        jump = rng.random((n, steps)) < 1.0 / self.block
        fresh = rng.integers(0, m, (n, steps))
        for k in range(1, steps):
            idx[:, k] = np.where(jump[:, k], fresh[:, k], (idx[:, k - 1] + 1) % m)
        return Forecast(cumulate(sigma * z[idx]), {"daily_volatility": sigma, "annual_volatility": sigma * np.sqrt(TRADING_DAYS), "mean_standardised_return": float(z.mean()), "kurtosis_of_shocks": float(np.mean(((z - z.mean()) / z.std()) ** 4))})
