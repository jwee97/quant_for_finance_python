---
title: "Gaussian processes for forecasting: what they find and what they invent"
slug: gaussian-process-forecasting
difficulty: 3
chapter: Platform
prerequisites: [hierarchical-bayes-decisions-and-gaussian-processes, probabilistic-forecasting]
stages: []
files: [src/forecast/gp.py, src/forecast/features.py, src/models/gaussian_process.py]
figures: []
tests: [tests/test_forecast_methods.py, tests/test_gaussian_process.py, tests/test_forecast_base.py]
models: []
---

# Gaussian processes for forecasting: what they find and what they invent

## In one sentence

A Gaussian process is a distribution over smooth functions that gives, at any new input, a prediction and an honest uncertainty that widens where the data are thin; applied to stock returns it mostly finds that there is little to find, and applied to the price path it extrapolates noise as if it were a trend.

## The idea

A Gaussian process (Rasmussen and Williams 2006) says any handful of values of an unknown function is jointly normal, with a covariance `k(x, x')` that says how alike the outputs at two inputs should be. After seeing `y = f(x) + noise` the prediction at a new input is normal again, in closed form: the mean is a weighted average of the data (`k* (K + s2 I)^-1 y`) and the variance is the prior variance less what the data explain, which grows with distance from the data. The kernel is the whole model: a Matern kernel allows a function as rough as you like, one *length scale* per input (automatic relevance determination) lets an input that does not matter drop out by getting a long length scale, and the few numbers (length scales, signal variance, noise) are chosen by maximising the marginal likelihood, which balances fit against complexity without a validation set.

There are three ways to point one at a stock, and they are very different.

- **Regression on features** (`gp_returns`). Learn how the return over the next `h` days, in units of the volatility of the day, depends on five things about where the market is now: the moves of the last 5, 21 and 63 days in units of their own volatility, whether volatility over the last 10 days is above or below its 126-day level, and how far the price sits below its 126-day high. The prediction is a mean and an uncertainty; the noise around it is resampled from the fit's own residuals so the tails are the data's.
- **A smooth drift in time** (`gp_drift`). Treat the daily return in units of volatility as a slowly changing average plus noise: a Gaussian process with a Matern 3/2 kernel in time. Its length scale says how fast the drift may change (from ten days to three hundred), and the forecast carries the average forward and fades it back to the year's average.
- **Extrapolating the price** (`gp_price`). Fit a smooth curve (a Matern kernel plus a linear trend) through the last 120 days of log price and continue it, with posterior draws that give price paths whose spread widens with distance. It is the picture people have in mind for a Gaussian-process forecast, and it is in the repository as the control.

## Why it matters

A Gaussian process is the standard tool for regression with an uncertainty that can be trusted when data are few, and the repository already uses one to search for good settings in the Tune tab and to forecast factor returns. It is also the method most easily misread: a smooth curve with a shaded band looks like knowledge. The honest question for a financial series is how much of what the process finds is signal, and the process answers it itself, through the share of the variance it leaves to noise.

## How this repo uses it

`src/models/gaussian_process.py` is an exact Gaussian process (kernels `rbf`, `matern32`, `matern52`, `linear` and sums, one length scale per input if wanted, analytic gradients, checked against scikit-learn). Forecasting needed two guards that the module now offers, bounds on the noise and on the length scales: for returns the noise is kept at or above 75% of the variance, so at most a quarter of what the process sees may be called signal, and a length scale cannot fall below 0.7 of the spread of a feature, because a shorter one is what lets a process interpolate noise. `src/forecast/gp.py` has the three methods, and `src/forecast/features.py` the five features (each computed from prices up to the day and checked for that). The regression is *heavy*: it trains on at most 220 evenly spaced days, one process per horizon, and the walk-forward test refits it half as often as the others.

## What we found

On markets with a known truth the processes behave: on independent returns the regression leaves more than 97% of the variance unexplained at every horizon, and the drift process finds none (a drift share of its variance below 30% and a month-ahead drift within half a volatility of zero); where the past month's move predicts the next days' return the other way, the regression explains part of the variance and forecasts a lower return after a steady rise than after a steady fall; where the average swings slowly the drift process tracks it (a correlation above 0.5 between its forecast of the next month's drift and the realised one), and the evaluation finds an out-of-sample R-squared above 10% against the five-year average with p below 1% for the regression, the drift process and the Kalman filter; the price-path process continues a straight line and, on a random walk, forecasts worse than staying put. In development, without the two bounds the regression fitted 80% to 93% of the in-sample variance of independent noise and gave ranges that were too narrow, which is why they exist.

On real prices the processes find very little. At its last refit on SPY the regression explained 0.01% of the variance of the next month's return in units of volatility and the drift process 0.01%, with a length scale at the upper bound of 300 days: the forecast was the average of the last year. The regression was 6.6% better than the random walk on SPY and 8.8% worse on TLT; its in-sample fit on TLT at 21 days at its last refit explained 42% of the variance of its training outcomes (overlapping outcomes are easy to fit; a fit made a few months later explained 4%), and over eight years its forecasts scored worse than the random walk. Against its control, Hull-White (the same units, no learning), it scored between −1.7% and −7.2% on the five ETFs (p-values of 0.74 to 0.97 for the claim that it was better): what it shows is what the control already has. The drift process was between +1.3% and −10.7%. The price-path process was between 63% and 90% *worse* than the random walk on every ticker: at its last refit on SPY it had a length scale of 14 days (a curve that follows every wiggle), a noise share of 1.5% and a trend of −28% a year, which it extrapolated.

## Pitfalls

- A band is not evidence. The uncertainty a process reports is the uncertainty *under its kernel and noise*; with the noise left free it will call a noisy series a smooth one with little noise.
- A price is the sum of its own shocks. A curve fitted to it takes the last shocks for a trend, and the further it extrapolates the worse it does; a Gaussian process on the *level* of a price is the control, not a candidate.
- Overlapping outcomes inflate the apparent fit. The return over the next 21 days of two neighbouring days share 20 days, so 220 training points are worth about ten independent ones.
- Far from the data a process returns to its prior mean, which is where the average of the last year comes from; do not read that as a forecast.
- The cost is a cube of the number of points, which is why the regression trains on a subsample.
- Length scales learned from a short, noisy series are poorly determined; a feature with a short length scale is *possibly* relevant, and the share of unexplained variance says whether it matters.

## Try it

```bash
quant forecast SPY --methods gp_returns gp_drift gp_price hull_white --effort quick
```

```python
import numpy as np
from src.forecast import History, create
from src.framework import load_default_bundle
from src.utils.config import load_config

history = History.from_frame(load_default_bundle(load_config(), with_macro=False).prices["TLT"])
gp = create("gp_returns")
state = gp.fit(history.tail(gp.window), (21,))
info = gp.predict(state, history.tail(gp.lookback), (21,), 1000, np.random.default_rng(0)).info
print(round(info["unexplained_share"]["21"], 2), {k: round(v, 2) for k, v in info["feature_relevance"].items()})
```
