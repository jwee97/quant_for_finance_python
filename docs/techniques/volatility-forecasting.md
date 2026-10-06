---
title: "Forecasting volatility: EWMA and GARCH"
slug: volatility-forecasting
difficulty: 2
chapter: Ch. 20.2
prerequisites: [data-integrity]
stages: [2, 8]
files: [src/features/volatility.py, src/portfolio/covariance.py, experiments/stage08_covariance.py]
figures: [4, 18]
tests: [tests/test_returns.py, tests/test_portfolio.py]
models: []
---

# Forecasting volatility: EWMA and GARCH

## In one sentence

Volatility clusters, so tomorrow's risk is best estimated by weighting recent squared returns more than old ones.

## The idea

An exponentially weighted moving average (EWMA) sets variance to a decaying average of squared returns, controlled by a half-life. GARCH(1,1) adds mean reversion toward a long-run level.
Both turn "how volatile has it been" into "how volatile will it be", which is what position sizing needs. They are judged out of sample with a loss that is robust to noisy proxies of
volatility, such as QLIKE, not with in-sample fit.

## Why it matters

Volatility is the one thing in finance that is genuinely forecastable. Sizing positions inversely to forecast volatility is the basis of risk parity and volatility targeting, and the
volatility forecast is the spread of every probabilistic forecast in the later stages.

## How this repo uses it

Stage 8 compares rolling-window, EWMA, GARCH and shrunk covariance estimators out of sample. The Generation 2 and 5 forecasts all use an EWMA volatility with a 40-day half-life as the
spread, so that the only thing that varies between models is the conditional mean.

## What we found

GARCH wins on QLIKE but a short half-life EWMA wins on correlation with realised volatility at a fraction of the complexity, so EWMA is the production estimator and GARCH a documented
extension (EXP-004, recorded as `investigate`).

## Going deeper

```
EWMA:  s2_t = lambda * s2_{t-1} + (1 - lambda) * r_t^2,         lambda = 0.5^(1/half_life)
GARCH: s2_t = omega + alpha * r_{t-1}^2 + beta * s2_{t-1},       long run s2 = omega / (1 - alpha - beta)
QLIKE: loss = s2_forecast_true_proxy / s2_forecast - ln(s2_proxy / s2_forecast) - 1
```
With a 40-day half-life, a return from 40 days ago has half the weight of today's. EWMA is GARCH with `omega = 0` and `alpha + beta = 1`: no pull toward a long-run level.

## Pitfalls

- Short half-lives react fast but are noisy; long ones are stable but late. There is no free lunch.
- Volatility forecasts fail exactly when it jumps (a crash day); do not size to leverage you cannot survive.
- A better volatility forecast does not mean a better portfolio.

## Try it

```bash
python -m experiments.stage08_covariance
```
Change the half-life in `config/forecasting.yaml` and see how the CRPS of every forecast stage responds.
