# turn_of_month

*Family: time-series*

## What it bets on

Turn-of-the-month effect: hold equities from the last trading day of a month through the third trading day of the next, otherwise stay out

Pension and payroll inflows, and the settlement of month-end flows, concentrate buying around the month boundary; most of the equity premium has historically come in these few days.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `before` = 1, `after` = 3, `classes` = ('equity',)

## Run it

```bash
quant backtest --model turn_of_month --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.01, CAGR -0.0%, volatility 3.1%, max drawdown -12.2%, turnover 8.1 times a year, deflated Sharpe probability 0.01 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Seasonality and calendar effects](../techniques/seasonality-and-calendar-effects.md)
