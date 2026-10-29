---
title: "GARCH variants, extreme-value tails, the bootstrap, Theta and weighted blends"
slug: garch-evt-bootstrap-theta-and-weighted-blends
difficulty: 3
chapter: Platform
prerequisites: [monte-carlo-forecasting, forecast-evaluation, stochastic-volatility-and-levy-forecasting]
stages: []
files: [src/forecast/more.py, src/forecast/evaluate.py, src/forecast/registry.py, src/econometrics/garch.py, src/references.py, experiments/forecast_study.py]
figures: []
tests: [tests/test_forecast_more_exact.py, tests/test_forecast_combine_exact.py, tests/test_forecast_methods.py]
models: []
---

# GARCH variants, extreme-value tails, the bootstrap, Theta and weighted blends

## In one sentence

Five more ways to forecast the distribution of a stock's next returns (the plain GARCH(1,1), EGARCH, conditional extreme-value theory, the stationary bootstrap and the Theta method) and a second blend, weighted by past accuracy, were added to the Forecast tab, each from a standard source and each tested on the same walk-forward footing as the rest: a leverage term paid for the stocks at a month's horizon, the other volatility variants matched GJR-GARCH without beating it, and weighting the blend by past accuracy added a few tenths of a point at most to the equal-weight blend.

## The idea

**`garch11`: the textbook model.** Today's variance is a constant plus a share of yesterday's squared shock plus a share of yesterday's variance, `sigma2[t] = omega + alpha e[t-1]^2 + beta sigma2[t-1]` (Bollerslev 1986), with Student-t shocks. Hansen and Lunde (2005) compared 330 volatility models and found that nothing beat this one on exchange rates, while for a stock (IBM) models with a *leverage effect* (a fall raises volatility more than a rise of the same size) did better. Having it beside the GJR model that `garch` already was lets that finding be checked on your own ticker.

**`egarch`: the log of the variance.** Nelson (1991) writes `ln sigma2[t] = omega + beta ln sigma2[t-1] + alpha (|z[t-1]| - E|z|) + gamma z[t-1]`, so the variance cannot go negative whatever the parameters, and a negative `gamma` is the leverage effect.

**`evt`: the tails of the shock.** The filtered historical simulation (`fhs`) resamples the standardised residuals of the GARCH fit, so it can never produce a shock larger than the largest in the sample. McNeil and Frey (2000) keep the body of the residuals as it is and fit a **generalised Pareto distribution** to the worst tenth on each side: by the Pickands-Balkema-de Haan theorem the excesses over a high threshold follow that family whatever the distribution of the shock, so a forecast can reach beyond the worst day in the data. The shape parameter is held between -0.25 and 0.5 (above a half the variance does not exist).

**`bootstrap`: blocks of days.** Resampling single days destroys the clustering of calm and stormy days; the stationary bootstrap (Politis and Romano 1994) draws blocks whose lengths are geometric, so that a block continues with probability `1 - 1/b` and otherwise restarts on a random day, with a mean block length of `b`, here the cube root of the number of days (about 11 for five years). The average return is removed first, so it forecasts a spread and no direction.

**`theta`: the M3 winner.** Assimakopoulos and Nikolopoulos (2000) split a series into two "theta lines", the straight-line trend (theta 0) and a line with twice the curvature (theta 2), forecast the first by its extrapolation and the second by simple exponential smoothing, and average. Hyndman and Billah (2003) showed that this is exactly simple exponential smoothing with a drift equal to half the slope of the trend. Here it is applied to the log price, and the spread comes from the one-step errors.

**`blend_weighted`: weights from the past.** Bates and Granger (1969) weight each forecaster by the inverse of its mean squared error. For distributions the same idea uses the CRPS: at every forecast origin each member's weight is proportional to the inverse of its mean CRPS over the forecasts whose outcomes were already known at that origin (at least ten of them, otherwise equal weights), and whole simulated paths are drawn from each member in that proportion, at least one from each.

## Why it matters

Each answers a question that the earlier list could not. Does a leverage term matter for the ticker in hand? Does the shape of the tail of a shock matter for a one-day or a one-month forecast? Does keeping clusters of days together beat treating days as independent? How does the benchmark of the forecasting textbooks do on prices? And does estimating combination weights from past accuracy pay, which the forecasting literature has debated since 1969?

## How this repo uses it

`src/forecast/more.py` has `Garch11MC` and `EGarchMC` (subclasses of the GARCH simulator that choose the model fitted by `src/econometrics/garch.py`), `EVTFilteredHistorical` (a subclass of `FilteredHistorical` that replaces the pool of shocks), `StationaryBootstrap`, and `ses` and `theta_mean` with the `Theta` class. `src/forecast/evaluate.py` holds the weighted blend (`inverse_loss_weights`, `combine_weighted`), the pairwise matrix and the cumulative gain. They are registered in `src/forecast/registry.py` and offered by the Forecast tab and `quant forecast`; every method's sources are listed under it (`src/references.py`).

## What we found

The tests check each against its definition: the smoothing recursion by hand, the Theta formula against the average of the two theta lines to twelve digits, the block structure of the bootstrap (a block continues with probability `1 - 1/b`), the fitted tails against the generalised Pareto they were drawn from (a Kolmogorov-Smirnov test), the model each GARCH variant fits (leverage found in a world that has it, a symmetric model unable to tell a fall from a rise), the weights against the inverse of the mean CRPS, and that the weighted blend never uses a score that was not known at its origin.

On real prices (SPY, QQQ, EEM, TLT and GLD, 2018 to 2026, 202 forecasts a run; the tables are in the [forecasting page](../forecasting.md)):

- **A leverage term pays for a stock at a month's horizon.** GJR-GARCH beat GARCH(1,1) by 3.6% on SPY at 21 days (one-sided Diebold-Mariano p = 0.001), 2.0% on QQQ, 1.0% on EEM, 0.7% on TLT, and was 0.2% behind on gold: Hansen and Lunde's finding, on five ETFs. EGARCH was within 0.6 points of GJR-GARCH everywhere.
- **Extreme-value tails barely change the month and change the day a little.** At 21 days the extreme-value version was within 0.4 points of the filtered historical simulation on every ticker; at one day it was 0.6 to 1.0 points ahead on SPY, QQQ and EEM.
- **The stationary bootstrap and Theta added little to the random walk** (+0.3% and +0.4% averaged over the five), though Theta was the best method on gold and, on TLT, the best at the average return and the direction, short of the threshold for many methods.
- **Weighting the blend by past accuracy added a few tenths of a point at most:** +2.4% averaged against +2.6% for equal weights, +7.1% against +6.8% on SPY (p = 0.18).

## Going deeper

```
GJR variance:       sigma2[t] = omega + (alpha + gamma 1{e[t-1] < 0}) e[t-1]^2 + beta sigma2[t-1]
EGARCH:             ln sigma2[t] = omega + beta ln sigma2[t-1] + alpha (|z| - E|z|) + gamma z,     E|z| = mean absolute unit-variance shock (sqrt(2/pi) normal; a Gamma-function formula for the t)
GPD tail:           P(Z - u > y | Z > u) = (1 + xi y / s)^(-1/xi);   a draw is  u + (s / xi)((1 - U)^(-xi) - 1)
bootstrap:          next index = previous + 1 (mod T) with probability 1 - 1/b, otherwise uniform on 0..T-1
theta forecast:     y_hat[T+h] = l[T] + (b / 2)(h - 1 + 1/alpha - (1 - alpha)^T / alpha)        l = SES level, b = slope of the trend of y
weights:            w[i] = (1 / CRPS_i) / sum_j (1 / CRPS_j),  CRPS_i = mean CRPS of member i over the origins whose outcomes are known
cumulative gain:    G[k] = sum over origins up to k of (CRPS_random_walk - CRPS_method) / mean(CRPS_random_walk)
```

## Pitfalls

- A forecast that scores best at 21 days can score nothing at one day and the reverse; the extreme-value tails are the example. Look at the horizon you will use.
- Plain GARCH(1,1) looks fine on gold and bonds and loses on stocks. The leverage effect is a property of the asset, not of the model; check the pair on your ticker with the method-against-method matrix.
- Maximum likelihood for the shape of a tail rests on about fifty points on each side; the estimate is noisy, and a shape above a half would give a forecast with an infinite variance, so it is held there.
- Theta's spread is a constant (the standard deviation of one-step errors over two years); on an asset whose volatility moves it is a worse spread than any volatility model's, and its good result on gold is that, not a forecast of the trend.
- Combination weights estimated from the past are themselves estimates. With a few hundred overlapping forecasts the noise in them costs about what the information in them is worth (the forecast combination puzzle; Smith and Wallis 2009), which is why equal weights are so hard to beat.

## Try it

```bash
quant forecast SPY --methods garch11 garch egarch fhs evt --horizon 21 --effort quick      # the volatility variants side by side
quant forecast SPY --methods bootstrap theta garch blend blend_weighted --horizon 21       # the rest, and both blends
python -m experiments.forecast_study --tickers SPY QQQ --methods garch11 garch egarch evt   # the skill table for these (add --cache runs/ to keep each evaluation)
```

Sources: [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1), [Nelson (1991)](https://doi.org/10.2307/2938260), [Hansen and Lunde (2005)](https://doi.org/10.1002/jae.800), [McNeil and Frey (2000)](https://doi.org/10.1016/s0927-5398(00)00012-8), [Politis and Romano (1994)](https://doi.org/10.1080/01621459.1994.10476870), [Assimakopoulos and Nikolopoulos (2000)](https://doi.org/10.1016/s0169-2070(00)00066-2), [Hyndman and Billah (2003)](https://doi.org/10.1016/s0169-2070(01)00143-1), [Bates and Granger (1969)](https://doi.org/10.1057/jors.1969.103), [Smith and Wallis (2009)](https://doi.org/10.1111/j.1468-0084.2008.00541.x), [Goyal and Welch (2008)](https://doi.org/10.1093/rfs/hhm014).
