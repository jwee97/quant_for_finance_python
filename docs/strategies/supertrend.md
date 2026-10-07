# supertrend

*Family: time-series*

## What it bets on

SuperTrend: an ATR band that trails price and flips the position when the close crosses it (10-day ATR, multiplier 3)

A stop that only ever tightens in the direction of the trend turns 'is the trend intact?' into a yes or a no.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `atr_window` = 10, `multiplier` = 3.0

## Run it

```bash
quant backtest --model supertrend --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.31, CAGR -2.8%, volatility 8.2%, max drawdown -54.0%, turnover 4.8 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
