# bollinger_reversion

*Family: time-series*

## What it bets on

Bollinger band reversion: buy a close below the lower band (20 days, 2 deviations), hold until price returns to the middle band

A close two standard deviations below its own mean is a stretch that usually partly closes; the middle band is the natural exit.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 20, `k` = 2.0, `shorts` = False

## Run it

```bash
quant backtest --model bollinger_reversion --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.27, CAGR 1.7%, volatility 6.9%, max drawdown -22.9%, turnover 8.7 times a year, deflated Sharpe probability 0.11 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Short-term mean reversion: pullbacks, bands and reversals](../techniques/short-term-mean-reversion.md)
