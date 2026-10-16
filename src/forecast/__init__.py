"""Forecasting a stock's price as a distribution, and finding out honestly how good each way of doing it is.

    base        History, Forecast and the Forecaster interface; the helpers the methods share
    scoring     CRPS, pinball, coverage and PIT scores, and the tests that compare forecasters without pretending to more evidence than there is
    baselines   random walk, RiskMetrics, historical simulation, geometric Brownian motion
    montecarlo  GJR-GARCH, filtered historical simulation, jump diffusion
    markov      a Markov chain on the day's move, on the volatility level, and a hidden Markov model of regimes
    gp          Gaussian processes: regression on recent moves, and extrapolation of the price trend
    others      Kalman filter trend, ARMA, HAR volatility, nearest-neighbour analogues
    evaluate    the walk-forward evaluation of the methods over rolling origins, and the forecast from the latest day
    verdict     what the evaluation says, in words
"""

from .base import Forecast, ForecastError, Forecaster, History
from .registry import BASELINE, BLEND, DEFAULT, METHODS, catalog, create

__all__ = ["Forecast", "ForecastError", "Forecaster", "History", "METHODS", "BASELINE", "BLEND", "DEFAULT", "catalog", "create"]
