# tsmom_multi

*Family: time-series*

## What it bets on

Multi-horizon time-series momentum: the average of the signs of the 1-, 3- and 12-month returns (a century of evidence)

Averaging trend signals over several horizons keeps most of the premium and halves the turnover of any single lookback.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `horizons` = (21, 63, 252)

## Run it

```bash
quant backtest --model tsmom_multi --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.01, CAGR -0.2%, volatility 6.5%, max drawdown -26.2%, turnover 12.5 times a year, deflated Sharpe probability 0.01 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
