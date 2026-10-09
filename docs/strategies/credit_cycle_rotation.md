# credit_cycle_rotation

*Family: macro*

## What it bets on

Credit strategy: the credit spread's level (high or low) and direction (widening or tightening) give four phases of the credit cycle; each asset is held in the direction it has paid in the current phase

Credit spreads lead the cycle: they tighten in recovery and expansion, begin to widen late in the cycle and blow out in contractions, and risky assets, Treasuries and gold earn very different
returns in each phase. The spread is the BAA-Treasury spread (``BAA10Y``) when the bundle has it, otherwise the relative performance of the credit assets against the Treasury assets (spreads
widen when credit lags). High or low is against the spread's own trailing three-year median; widening or tightening is its change over ``window`` days. As in ``curve_quadrant`` the
consequences are measured from earlier data and only the significant ones are traded.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 63, `horizon` = 21, `min_obs` = 500, `tstat` = 1.5, `scale` = 2.0, `assets` = ()

## Run it

```bash
quant backtest --model credit_cycle_rotation --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.43, CAGR 0.8%, volatility 1.9%, max drawdown -6.5%, turnover 1.9 times a year, deflated Sharpe probability 0.13 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Economic-outlook strategies: the yield curve and the credit cycle, with learned consequences](../techniques/economic-outlook-strategies.md)
