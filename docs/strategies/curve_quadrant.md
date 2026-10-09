# curve_quadrant

*Family: macro*

## What it bets on

Yield-curve strategy: the change in the level and slope of the curve gives four states (bull/bear steepener/flattener); each asset is held in the direction it has paid in the current state

Rates fall or rise (bull or bear) and the curve steepens or flattens: the four combinations mean different things for growth, inflation and policy (a bull steepener is typically an easing cycle,
a bear flattener a tightening one) and so for bonds, equities, credit and gold. Instead of fixing the table, the strategy measures the average next-month return of each asset in each state from
earlier data only and trades the combinations that are significant. Needs the 2- and 10-year Treasury yields (``DGS2``, ``DGS10``) in the bundle's macro frame.

## Inputs

- Macro or alternative series required: DGS2, DGS10
- Parameters: `window` = 63, `horizon` = 21, `min_obs` = 500, `tstat` = 1.5, `scale` = 2.0, `assets` = ()

## Run it

```bash
quant backtest --model curve_quadrant --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.70, CAGR 2.1%, volatility 3.0%, max drawdown -8.5%, turnover 3.2 times a year, deflated Sharpe probability 0.45 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Economic-outlook strategies: the yield curve and the credit cycle, with learned consequences](../techniques/economic-outlook-strategies.md)
