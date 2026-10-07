# range_reversion

*Family: time-series*

## What it bets on

Range-bound reversion: fade the 20-day z-score, but only while the ADX shows no trend (below 20)

Mean reversion works inside ranges and fails in trends; the ADX decides which one the market is in.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 20, `adx_window` = 14, `adx_max` = 20.0

## Run it

```bash
quant backtest --model range_reversion --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -1.36, CAGR -3.3%, volatility 2.5%, max drawdown -46.8%, turnover 28.8 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Short-term mean reversion: pullbacks, bands and reversals](../techniques/short-term-mean-reversion.md)
