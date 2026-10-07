# keltner_breakout

*Family: time-series*

## What it bets on

Keltner channel breakout: enter beyond the 20-day EMA plus or minus two ATRs, hold until the close crosses the EMA back

A close outside a volatility-scaled envelope is a move bigger than noise; the middle line is the exit.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 20, `k` = 2.0, `atr_window` = 14

## Run it

```bash
quant backtest --model keltner_breakout --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.08, CAGR -0.2%, volatility 2.7%, max drawdown -11.2%, turnover 2.4 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
