---
title: "Unit roots, variance ratios and cointegration tests"
slug: unit-roots-and-cointegration
difficulty: 2
chapter: Ch. 12
prerequisites: [pairs-and-statistical-arbitrage]
stages: []
files: [src/stats/timeseries.py, src/econometrics/var.py]
figures: []
tests: [tests/test_stats.py, tests/test_econometrics.py]
models: []
---

# Unit roots, variance ratios and cointegration tests

## In one sentence

Before you model a price, a spread or a rate you need to know whether it wanders like a random walk or is pulled back to a mean; `src.stats.timeseries` provides the tests that decide, and the Granger and Johansen tools say whether series move together.

## The idea

A series is **stationary** if its mean, variance and autocovariances do not drift over time; a **unit-root** series (a random walk) accumulates shocks forever. Four tests look at it from different sides:

- **Augmented Dickey-Fuller** (`adf_test`) regresses the change on the lagged level and lagged changes; the null is a unit root, so a very negative statistic rejects it. The critical values are the non-standard MacKinnon ones, not the normal ones.
- **KPSS** (`kpss_test`) reverses the null: stationarity. Using both guards against low power. `stationarity_summary` combines ADF, PP and KPSS into a verdict.
- **Phillips-Perron** (`phillips_perron`) corrects the ADF statistic non-parametrically for serial correlation instead of adding lags.
- **Variance ratio** (`variance_ratio`, Lo-MacKinlay) compares the variance of `q`-period returns with `q` times the one-period variance; a ratio above one means momentum, below one mean reversion, with a heteroskedasticity-robust z-statistic.

`hurst_exponent` and `half_life` (from an AR(1) fit) put a number on how fast a mean-reverting spread decays. **Engle-Granger** (`engle_granger`) tests whether a regression residual of one price on another is stationary, which is the definition of cointegration for two series. **Johansen** (`src.econometrics.var.johansen`) handles several series at once and reports the cointegration rank through the trace and max-eigenvalue statistics. `granger_causality` tests whether lags of `x` improve a forecast of `y`.

## Why it matters

Regressing one trending price on another gives a high R-squared and a significant slope even when they are unrelated (Granger and Newbold's spurious regression). Every pairs-trading or stat-arb claim depends on the spread being stationary out of sample, and a half-life of two years is not a tradable mean reversion for a book rebalanced weekly.

## How this repo uses it

The pairs and stat-arb strategies select pairs with these tests; the econometrics module uses them to choose the differencing order for ARIMA and the rank for the VECM; the findings script reports a stationarity table for the 15 ETFs. All are written out in numpy and compared with `statsmodels` and `arch` in the tests.

## What we found

The tests reproduce the reference implementations: ADF (with and without automatic lag selection), KPSS, Engle-Granger and Granger causality match `statsmodels`, and Phillips-Perron and the variance ratio match `arch`. On simulated data they separate a random walk from an AR(1), flag a planted cointegrated pair, and the Granger test is directional. On the real ETF prices every price series fails to reject a unit root and the returns are stationary, as expected; see [the findings](../institutional_findings.md).

## Pitfalls

- ADF has low power against a root near one: failing to reject does not prove a unit root. Read it together with KPSS.
- The deterministic terms (constant, trend) matter; a trend regression on a series with no trend lowers power.
- Cointegration found in-sample often breaks; re-test on rolling windows and trade only what stays stable.
- Structural breaks look like unit roots to all of these tests.

## Try it

```python
import numpy as np, pandas as pd
from src.stats.timeseries import adf_test, engle_granger, half_life

rng = np.random.default_rng(0)
walk = pd.Series(rng.standard_normal(1000).cumsum())
spread = pd.Series(np.zeros(1000))
for t in range(1, 1000):
    spread[t] = 0.9 * spread[t - 1] + rng.standard_normal()
print(adf_test(walk).pvalue > 0.05, adf_test(spread).pvalue < 0.05, round(half_life(spread), 1))
print(engle_granger(walk + spread, walk)["pvalue"] < 0.05)
```
