# kalman_trend

*Family: time-series*

## What it bets on

Adaptive trend by Kalman filter: the filtered slope of a local linear trend in log price, in units of the asset's own daily volatility

A trend is a persistent drift hidden in noisy prices; the Kalman filter estimates it recursively, and its signal-to-noise ratio sets the memory.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `level_noise` = 0.02, `slope_noise` = 1e-05, `vol_halflife` = 60.0, `clip` = 3.0, `scale` = 0.15

## Run it

```bash
quant backtest --model kalman_trend --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.07, CAGR 0.1%, volatility 2.6%, max drawdown -6.8%, turnover 1.9 times a year, deflated Sharpe probability 0.02 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Econometric strategies: Kalman trend, GARCH and EVT sizing, and a shrunk VAR](../techniques/econometric-strategies.md)
