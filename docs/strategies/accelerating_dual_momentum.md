# accelerating_dual_momentum

*Family: time-series*

## What it bets on

Accelerating dual momentum: hold the top-k assets by the average 1-, 3- and 6-month return, but only those that beat cash

A shorter, averaged lookback reacts to a turn sooner than 12-month dual momentum, at the price of more switching.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `top_k` = 2, `cash` = 'SHY', `month` = 21

## Run it

```bash
quant backtest --model accelerating_dual_momentum --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.46, CAGR 4.2%, volatility 10.2%, max drawdown -25.0%, turnover 16.7 times a year, deflated Sharpe probability 0.32 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
