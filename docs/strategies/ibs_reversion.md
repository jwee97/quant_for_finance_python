# ibs_reversion

*Family: time-series*

## What it bets on

Internal bar strength reversion: buy when the close sits in the bottom fifth of the day's range, sell when it recovers to the top third

A close near the day's low means sellers pushed to the end; the next session tends to open higher. Without highs and lows a 10-day range of closes stands in.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `entry` = 0.2, `exit` = 0.7, `trend_window` = 200, `shorts` = False

## Run it

```bash
quant backtest --model ibs_reversion --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.76, CAGR 6.9%, volatility 9.3%, max drawdown -18.4%, turnover 6.3 times a year, deflated Sharpe probability 0.78 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Short-term mean reversion: pullbacks, bands and reversals](../techniques/short-term-mean-reversion.md)
