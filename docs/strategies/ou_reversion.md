# ou_reversion

*Family: time-series*

## What it bets on

Ornstein-Uhlenbeck reversion: fade the deviation from the 120-day mean, but only where a unit-root test rejects a random walk and the half-life is short

Fitting an AR(1) to log prices says whether an asset pulls back to its mean at all, and how fast. On a short window a random walk looks mean-reverting by chance
(the estimate of the AR coefficient is biased down), so a deviation only counts if the Dickey-Fuller t-statistic is below its critical value and the half-life is short.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 120, `max_half_life` = 30.0, `df_critical` = -2.86

## Run it

```bash
quant backtest --model ou_reversion --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.56, CAGR -0.3%, volatility 0.5%, max drawdown -4.9%, turnover 3.6 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Short-term mean reversion: pullbacks, bands and reversals](../techniques/short-term-mean-reversion.md)
