# macd_trend

*Family: time-series*

## What it bets on

MACD trend: the 12/26-day EMA difference (the MACD line) scaled by the price's own volatility; optionally its histogram against the 9-day signal line

The gap between a fast and a slow average is positive in up-trends and negative in down-trends; its histogram (the gap minus its own average) measures whether the trend is strengthening.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `fast` = 12, `slow` = 26, `signal` = 9, `vol_window` = 20, `use` = 'line'

## Run it

```bash
quant backtest --model macd_trend --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.10, CAGR -0.3%, volatility 2.5%, max drawdown -10.3%, turnover 4.2 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
