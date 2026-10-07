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

