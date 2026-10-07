# garch_vol_managed

*Family: volatility*

## What it bets on

Volatility-managed long using a walk-forward GARCH(1,1) volatility forecast in place of trailing realised variance

If a better volatility forecast is available, scaling exposure by it should capture more of the volatility-timing benefit than a one-month lookback does.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `model` = 'garch', `dist` = 'normal', `min_train` = 750, `refit_every` = 504, `max_scale` = 3.0

## Run it

```bash
quant backtest --model garch_vol_managed --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.56, CAGR 7.0%, volatility 13.7%, max drawdown -26.4%, turnover 13.1 times a year, deflated Sharpe probability 0.39 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Uses VIX-type indices as the implied-volatility input; no option-level data.

## Learn more

[Econometric strategies: Kalman trend, GARCH and EVT sizing, and a shrunk VAR](../techniques/econometric-strategies.md)
