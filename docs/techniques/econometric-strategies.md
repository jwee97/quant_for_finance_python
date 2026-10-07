---
title: "Econometric strategies: Kalman trend, GARCH and EVT sizing, and a shrunk VAR"
slug: econometric-strategies
difficulty: 3
chapter: Ch. 13
prerequisites: [time-series-econometrics, risk-managed-exposure]
stages: []
files: [src/strategies/econometric.py, src/econometrics/garch.py, src/probability/evt.py, src/econometrics/factor.py]
figures: []
tests: [tests/test_econometric_strategies.py]
models: [kalman_trend, garch_vol_managed, evt_risk_managed, bvar_lead_lag]
---

# Econometric strategies: Kalman trend, GARCH and EVT sizing, and a shrunk VAR

## In one sentence

Four strategies that put the econometrics and tail-risk layers to work: a trend filter with a signal-to-noise ratio instead of a window, volatility management with a better volatility forecast, sizing on the expected tail loss, and a cross-asset lead-lag model with heavy shrinkage.

## The idea

**`kalman_trend`** models log price as a local linear trend: a level that moves with a hidden drift. The Kalman filter estimates that drift recursively; the steady-state gains depend only on the ratio of process noise to observation noise, so the trend's memory is a signal-to-noise assumption rather than a moving-average window. The filtered slope, divided by the asset's own volatility and squashed by `tanh`, is the position. **`garch_vol_managed`** is the Moreira-Muir volatility-managed long (hold `c / sigma^2` or `c / sigma`) with a walk-forward GARCH forecast of `sigma` replacing last month's realised variance; the model refits only on data up to the origin. **`evt_risk_managed`** scales exposure by the inverse of a conditional-EVT expected-shortfall forecast (an EWMA volatility filter, then a generalised Pareto fit to the standardised losses), because drawdown limits bind on losses, not on variance. **`bvar_lead_lag`** forecasts each asset's next month from every asset's last month with a Minnesota-prior VAR(1): the prior shrinks every coefficient toward a random walk, so only strong, stable lead-lag relations survive.

## Why it matters

Each is a test of whether a more careful statistical model adds value over its simple cousin: the moving-average trend rule, trailing-variance management, variance-based sizing and a plain regression. The answer matters because the careful model costs complexity and parameters.

## How this repo uses it

All four are plugins in the model registry, so they run through the same pipeline, costs, validation and deflated-Sharpe accounting as the other 80 models. `garch_vol_managed` and `evt_risk_managed` use the sleeve book (one sleeve per asset); `bvar_lead_lag` is a monthly cross-sectional long-short.

```bash
quant backtest --model kalman_trend --tearsheet
quant backtest --model evt_risk_managed
```

## What we found

All numbers come from one run of [the strategy survey](../strategy_survey.md): the platform's 15 ETFs, default parameters, net of 10 bps costs, 79 strategies counted as trials in the deflated Sharpe ratio. Each rule starts on its own first day, so the comparison is with equal weight over the same dates. The numbers are exploratory. `evt_risk_managed` earned a net Sharpe of +0.68 (deflated probability 0.56) against +0.76 for equal weight over its dates (from January 2012), with a 20% drawdown, the same as equal weight, at 2x turnover a year: no better than holding everything. `garch_vol_managed` earned +0.56 (13x turnover, drawdown 26%, equal weight +0.76), below `vol_managed_long` (+0.81 against +0.94 over its own dates), so a better volatility forecast did not help here; the extra turnover costs what the forecast gains. `kalman_trend` earned +0.07 (equal weight +0.66) at 2.6% volatility, because it holds small positions by design, and `bvar_lead_lag` +0.10 (equal weight +0.76) at 19x turnover: the lead-lag structure is too weak to survive costs. None approaches a deflated probability of 0.95. These are results on 15 ETFs; the tests establish that the machinery is causal and correct, which is a separate claim from profitability.

## Pitfalls

- Walk-forward refits are expensive: the GARCH and EVT models refit every 504 and 63 days by default, a choice between freshness and runtime.
- The filter's signal-to-noise ratio is a tuning parameter in disguise; do not select it by backtest without counting the trials.
- Volatility management levers calm assets (bonds) up to the cap, which creates its own risk.
- A VAR on monthly returns has few observations per parameter; the shrinkage tightness decides whether it learns anything.

## Try it

```bash
quant backtest --model garch_vol_managed --param model=gjr --param dist=t
quant backtest --model bvar_lead_lag --param tightness=0.05
```
