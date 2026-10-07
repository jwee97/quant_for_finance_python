# kama_trend

*Family: time-series*

## What it bets on

Kaufman adaptive moving average trend: long above the adaptive average, short below, sized by distance in volatility units

An average that speeds up in clean trends and slows in noise gives fewer whipsaws than a fixed-length average.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `er_window` = 10, `fast` = 2, `slow` = 30, `vol_window` = 20

## Run it

```bash
quant backtest --model kama_trend --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.32, CAGR -1.9%, volatility 5.4%, max drawdown -32.9%, turnover 19.9 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
