# adx_trend

*Family: time-series*

## What it bets on

Wilder's directional movement: follow the stronger of +DI and -DI, but only while the ADX says a trend exists (default above 25)

Trend-following loses in trendless markets; the ADX measures how directional the recent range has been and switches the strategy off in chop.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 14, `threshold` = 25.0

## Run it

```bash
quant backtest --model adx_trend --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.24, CAGR -1.9%, volatility 6.8%, max drawdown -36.7%, turnover 9.2 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
