# ichimoku_trend

*Family: time-series*

## What it bets on

Ichimoku cloud: long above the cloud with the conversion line over the base line, short below the cloud with it under, flat inside

The cloud is a support and resistance zone built from three ranges; trading only when price is clear of it avoids the middle of a range.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `tenkan` = 9, `kijun` = 26, `senkou` = 52

## Run it

```bash
quant backtest --model ichimoku_trend --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.34, CAGR -3.2%, volatility 8.5%, max drawdown -60.1%, turnover 11.5 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
