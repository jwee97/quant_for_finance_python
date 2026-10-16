---
title: "Monte Carlo forecasting: simulating many futures"
slug: monte-carlo-forecasting
difficulty: 2
chapter: Platform
prerequisites: [volatility-forecasting, resampling-and-monte-carlo]
stages: []
files: [src/forecast/baselines.py, src/forecast/montecarlo.py, src/econometrics/garch.py, src/forecast/base.py]
figures: []
tests: [tests/test_forecast_methods.py, tests/test_forecast_base.py, tests/test_forecast_evaluate.py]
models: []
---

# Monte Carlo forecasting: simulating many futures

## In one sentence

Monte Carlo forecasting draws thousands of random futures from a model of how returns behave and reads the range of outcomes off the draws, so the answer is only as good as the model that makes the draws, and what matters most in that model is how the volatility moves.

## The idea

A forecast here is a table of simulated futures: `samples[j, k-1]` is the cumulative log return after `k` days in draw `j`. A price forecast of one month is 1,000 paths of 21 daily returns added up; the 5% and 95% quantiles of the sums are the edges of the 90% range, the share of sums above zero is the chance of a rise, and the share below `log(0.95)` is the chance of losing 5% or more. Every Monte Carlo method differs only in how one day's return is drawn.

- **Geometric Brownian motion**: a normal return with the drift and volatility of the last year. The model of the textbooks, and the reason the next method exists.
- **Historical simulation**: draw blocks of about ten consecutive days from the last five years, so the fat tails, the skew and some of the clustering of volatility come along, and so does the average return of those years.
- **Volatility-scaled historical simulation** (Hull and White 1998): divide each past return by the volatility of its own day (an exponentially weighted average with decay 0.94, RiskMetrics) and multiply by today's, then resample in blocks. The shape of the history with the current spread.
- **GJR-GARCH with Student-t shocks** (Glosten, Jagannathan and Runkle 1993): `sigma2[t] = omega + (alpha + gamma * 1[r[t-1] < 0]) * r[t-1]^2 + beta * sigma2[t-1]`. Volatility clusters (`alpha + gamma/2 + beta` close to one), a fall raises it more than a rise (`gamma`), and the shocks have fat tails (`nu` degrees of freedom). The simulation starts from the variance the model has *today* and lets it fall back toward its long-run level, with a half-life that the fit gives.
- **Filtered historical simulation** (Barone-Adesi, Giannopoulos and Vosper 1999): the same volatility recursion, but the shocks are the model's own past standardised residuals drawn at random, so the tails are the ones the stock really had.
- **Jump diffusion** (Merton 1976): small ordinary days plus rare large ones, found as the days more than 3.5 robust standard deviations from the median; a Poisson number of jumps a day, each normal with a fitted mean and size, and the drift set so that the average return is zero.

## Why it matters

The spread of the next returns decides how much to hold, where a stop belongs, what an option is worth and how much capital a book needs. Few of the questions have a closed form once volatility changes, tails are fat and the horizon is more than a day, and a simulation answers all of them from the same draws: the chance of a 5% loss, the expected shortfall, the probability of touching a level. The cost is that a simulation is only as honest as its model, and many draws make an answer smooth, not right.

## How this repo uses it

The methods are in `src/forecast/baselines.py` and `src/forecast/montecarlo.py`; every one has `fit` (every 63 trading days on the last five years) and `predict` (brings its state up to the forecast date with the fitted numbers held fixed). The GARCH methods use `GARCHResult.simulate` in `src/econometrics/garch.py`, which simulates conditional paths for the GARCH, GJR, EGARCH and threshold models from the variance at the last day, optionally resampling a given set of standardised residuals; its variance after `k` days agrees with the closed-form `forecast_variance`. The Forecast tab and `quant forecast` score every method walk forward against a random walk of constant volatility ([forecasting a stock's price](../forecasting.md)); [scoring forecasts honestly](forecast-evaluation.md) explains the scores.

## What we found

On markets with a known truth the methods find what was planted: GJR-GARCH recovers a persistence between 0.9 and 1 and a positive leverage from simulated returns with both, and its next-day spread after a stormy month is more than twice that after a quiet one; filtered historical simulation matches the parametric spread within 25% and uses the model's own residuals; the jump diffusion finds the rate, the (negative) size and the calm volatility of planted jumps and finds none where there were none; the volatility-scaled simulation has a spread more than three times larger after a storm than after a calm spell and keeps the fat tails of the past; in independent returns nothing beats the random walk.

On real prices (SPY, QQQ, EEM, TLT and GLD at 21 days, eight years of forecasts), the zero-average volatility models gained 2.6% (RiskMetrics) to 6.7% (filtered historical simulation) on SPY, where GJR-GARCH and the filtered simulation held up after the correction for the number of methods, between 0.6% and 2.7% on QQQ, and about nothing on TLT and less than nothing on GLD. The fit on SPY at the last forecast origin (18 September 2026) had volatility persistence 0.981, a half-life of 36 days, a leverage `gamma` of 0.20, Student-t shocks with 8.8 degrees of freedom, a volatility tomorrow of 10.7% a year against a long-run 19.4%. The filtered simulation added nothing to GARCH (between −0.0% and +0.3%, p of 0.24 to 0.54); Brownian motion gained 1.5% at most and lost 4.0% on TLT, because the drift of one year is noise; the jump diffusion fitted 4.8 jumps a year of mean size −1.3% and gained 1.1% on SPY at most.

## Pitfalls

- A simulated forecast is the model's answer. A GJR-GARCH fitted on a calm five years has no storm in it, and a crisis unlike the ones in the fit is a surprise to it.
- The drift is the weakest number in the model. A year of daily returns pins the average down to within several times its own size, so a method that carries it forward adds a random shift to every forecast; a zero-average forecast is the better default for weeks.
- A historical simulation after a calm stretch has no storms to draw on, and carries the old average return forward.
- The columns of `samples` are the same paths at different days, not independent forecasts: do not average a score over the 21 steps as if each were another test.
- The 5% quantile of 1,000 draws is itself noisy. Use more draws for tails, fix the seed to compare two runs, and do not read a difference in the second decimal as a difference between methods.

## Try it

```bash
quant forecast SPY --methods garch fhs hull_white jump gbm historical --effort quick
```

```python
import numpy as np
from src.forecast import History, create
from src.framework import load_default_bundle
from src.utils.config import load_config

history = History.from_frame(load_default_bundle(load_config(), with_macro=False).prices["SPY"])
garch = create("garch")
state = garch.fit(history.tail(garch.window), (21,))
forecast = garch.predict(state, history.tail(garch.lookback), (21,), 2000, np.random.default_rng(0))
print(np.quantile(forecast.at(21), [0.05, 0.5, 0.95]).round(3), round(forecast.info["half_life_days"]), round(forecast.info["persistence"], 3))
```
