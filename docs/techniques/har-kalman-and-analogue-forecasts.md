---
title: "HAR volatility, the Kalman trend, ARMA and analogues"
slug: har-kalman-and-analogue-forecasts
difficulty: 3
chapter: Platform
prerequisites: [volatility-forecasting, time-series-econometrics, probabilistic-forecasting]
stages: []
files: [src/forecast/others.py, src/forecast/features.py, src/econometrics/arima.py]
figures: []
tests: [tests/test_forecast_methods.py, tests/test_forecast_evaluate.py, tests/test_forecast_base.py]
models: []
---

# HAR volatility, the Kalman trend, ARMA and analogues

## In one sentence

Four tools from the quant toolbox answer four different questions: the HAR model asks how volatility remembers over a day, a week and a month, the Kalman filter whether there is a drift to follow, ARMA whether yesterday predicts today, and analogues whether similar days have similar sequels; of the four only the volatility model found anything on real prices, and only on one ETF.

## The idea

**HAR** (heterogeneous autoregression, Corsi 2009) regresses the volatility of the next days on the volatility of the last day, the last week and the last month, because traders act on different horizons and so volatility has memory at all three. Here it is a regression of the log of the variance over the next `h` days on the logs of the average variance over the last 1, 5 and 22 days. The day's variance is read from its high and low where the data have them (Parkinson 1980: `ln(high/low)^2 / (4 ln 2)`, rescaled so that it averages to the close-to-close variance), else from the squared return. The shocks are Student-t with a tail weight read from the fit's standardised residuals, and the variance forecast is itself uncertain, with a lognormal factor of mean one shared along a path, which fattens the tails of longer horizons.

The **Kalman filter** on a *local linear trend* (Harvey 1989) says the log price is a level that moves with a slope, and the slope itself drifts: `level[t+1] = level[t] + slope[t] + noise`, `slope[t+1] = slope[t] + noise`, observed with a little noise. The three variances are fitted by maximum likelihood, the filter tracks the level and slope day by day, and the forecast carries the slope forward with the uncertainty of the state added. It is an adaptive estimate of the drift.

**ARMA(1,1)** (Box and Jenkins 1970) with no constant models the day's return as `phi * yesterday's return + theta * yesterday's surprise + a new surprise`, with normal shocks of constant variance. The forecast of a multi-day return sums the model's impulse-response weights.

**Analogues** (nearest neighbours) find the 60 days in the last five years that looked most like today by five numbers (the moves of 5, 21 and 63 days and the 10-day against the 126-day volatility, each in units of volatility, and the distance from the 126-day high) and use what followed them, scaled to today's volatility, as the forecast.

## Why it matters

Each is a family of methods that is used constantly and judged rarely. Putting them on the same walk-forward test, with the same yardstick, turns an opinion into a number, and it shows how little a flexible method adds when the thing it hopes to find (a drift, a memory in returns, a pattern) is small compared with the noise.

## How this repo uses it

`src/forecast/others.py` has the four. The ARMA fit is `src/econometrics/arima.py::fit_arima`; the Kalman filter is a scalar-arithmetic recursion that is fast enough to refit every few weeks on 500 days. The analogues and the Gaussian process regression use the same five features (`src/forecast/features.py`), and Hull-White is their control: what the analogues add to it is what they learn about the future. The results appear in the Forecast tab under *Volatility*, *State space*, *Time series* and *Pattern matching*.

## What we found

On markets with a known truth each finds what was planted. The HAR model forecasts more than twice the volatility after a stormy month than after a calm one, uses the range when the data have it, forecasts that a storm fades back (the month's volatility is no more than a fifth above the next day's), and refuses data that have lost the highs and lows its fit used. The Kalman filter finds the slope of a steady trend (the mean of five markets within 0.1 of the true yearly slope) and flips its sign when the average turns, and in a market with a slowly swinging average it has an out-of-sample R-squared above 10% against the five-year average with p below 1%. The ARMA model finds a planted serial dependence and none in independent returns, and forecasts a positive mean after a big up day in the dependent one. The analogues forecast a lower return after a steady rise than after a fall where the past month reverses, and beat Hull-White's rescaled history there by more than luck allows.

On real prices (five ETFs, 21 days, eight years of forecasts) the volatility model was between 3.4% better than the random walk on SPY (the gain held up after the correction for the number of methods) and 1.3% worse on GLD, ahead of RiskMetrics on four of the five. The Kalman filter was between −3.0% and +0.4%, ARMA within 1.5% of the random walk either way, and the analogues +5.9% on SPY, +3.6% on QQQ and between −7.6% and −9.1% on EEM, TLT and GLD; against Hull-White they scored −0.9% to −8.0% on all five. On SPY at the last forecast origin the Kalman filter found a slope of +17% a year with a standard error of 11% and a slope that barely changes, ARMA a pair of coefficients that nearly cancel (AR 0.73, MA −0.76: a model of no memory), and the HAR model a volatility of 10% tomorrow rising to 12% over the next month. None of the three that forecast an average beat the average of the last five years, and none called the direction better than always saying up.

## Pitfalls

- A volatility model forecasts the *size* of moves and never their direction. A good one does not make money by itself.
- Without highs and lows the day's variance is one squared return, a very noisy proxy; the page says when that is the case.
- A trend filter on a stock finds a slope that is a few times smaller than its own standard error. Read the standard error before the slope.
- ARMA with coefficients that cancel is a white-noise model with extra parameters. An AR and an MA term that are nearly equal and opposite say there is nothing to model.
- The closest of 1,200 days in five dimensions are close partly by luck, and neighbouring days share most of their sequel, so 60 analogues are worth far fewer than 60 independent examples.
- Standardising by volatility puts different periods on one scale; it also assumes that volatility measured by an average of the last few weeks is the volatility of the day before, which a gap violates.

## Try it

```bash
quant forecast SPY --methods har kalman arma analog hull_white riskmetrics --effort quick
```

```python
import numpy as np
from src.forecast import History, create
from src.framework import load_default_bundle
from src.utils.config import load_config

bundle = load_default_bundle(load_config(), with_macro=False)
history = History.from_frame(bundle.prices["SPY"], bundle.high["SPY"], bundle.low["SPY"])
har = create("har")
state = har.fit(history.tail(har.window), (1, 21))
info = har.predict(state, history.tail(har.lookback), (1, 21), 2000, np.random.default_rng(0)).info
print(info["proxy"], round(info["volatility_tomorrow_annual"], 3), round(info["volatility_next_month_annual"], 3), round(info["tail_degrees_of_freedom"], 1))
```
