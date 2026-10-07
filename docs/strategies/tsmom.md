# tsmom

*Family: time-series*

## What it bets on

Time-series momentum (Moskowitz-Ooi-Pedersen): the sign of the 12-month return, sized by 40% over each asset's own volatility

An asset that rose over the past year tends to keep rising, and one that fell tends to keep falling (under-reaction, hedging demand, herding).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 252, `target_vol` = 0.4, `max_scale` = 4.0, `halflife` = 60.0

## Run it

```bash
quant backtest --model tsmom --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.41, CAGR 5.3%, volatility 15.5%, max drawdown -34.9%, turnover 13.4 times a year, deflated Sharpe probability 0.24 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
