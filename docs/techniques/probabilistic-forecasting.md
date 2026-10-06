---
title: "Forecasts as distributions: CRPS and the PIT"
slug: probabilistic-forecasting
difficulty: 3
chapter: Ch. 20
prerequisites: [volatility-forecasting]
stages: [19, 33]
files: [src/models/probabilistic.py, src/framework/types.py, experiments/stage19_probabilistic.py]
figures: [38, 39]
tests: [tests/test_probabilistic.py, tests/test_framework.py]
models: []
---

# Forecasts as distributions: CRPS and the PIT

## In one sentence

A forecast should say how unsure it is, and the continuous ranked probability score rewards forecasts that are both accurate and honestly spread.

## The idea

A point forecast of +1% is useless without its spread. A Gaussian forecast N(mean, std) can be scored by the CRPS, a proper scoring rule (it is minimised in expectation only by the true
distribution) that generalises absolute error. The probability integral transform checks calibration: if forecasts are right, the realised outcomes' percentiles under the forecast are
uniform. In this repo a forecast carries a mean, a standard deviation and a confidence, |2 Phi(mean/std) - 1|, which downstream sizing uses.

## Why it matters

Sizing by confidence only helps if confidence carries information, and that is an empirical question that a proper score can test.

## How this repo uses it

`src/models/probabilistic.py` has the scores, the Platt calibrator and the walk-forward fit; `src/framework/types.py` defines the `Forecast` object every plugin returns. Every forecast stage
uses the same EWMA volatility as the spread, so differences in CRPS isolate the mean.

## What we found

The price-only Gaussian forecast did not beat the asset's own historical mean on CRPS (EXP-047), and adding macro and regime features did not help (EXP-048). Fractional-Kelly sizing of the Gaussian
forecast beat direction-only sizing net of costs (EXP-050), while sizing by calibrated edge did not (EXP-049).

## Going deeper

```
Gaussian CRPS: CRPS(y; mu, s) = s * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],     z = (y - mu)/s
confidence    = | 2 Phi(mu / s) - 1 |           (0 when mu = 0, near 1 when the mean is many sigmas from zero)
PIT_t         = Phi( (y_t - mu_t) / s_t )       uniform on [0, 1] if the forecast is calibrated
```
CRPS is in the units of the outcome, so a CRPS of 0.0224 against 0.0230 is a 2.6% improvement. Diebold-Mariano tests the mean of the CRPS differences with a heteroskedasticity-robust standard error.

## Pitfalls

- A forecast that is better than a poor benchmark may still be worse than a good one (the historical mean).
- Overlapping 21-day outcomes need HAC or monthly origins.
- A calibrated probability that is no better than the base rate adds nothing.

## Try it

```bash
python -m experiments.stage19_probabilistic
quant explain crps
```
