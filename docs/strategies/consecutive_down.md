# consecutive_down

*Family: time-series*

## What it bets on

Down-streak pullback (Connors): buy after three straight lower closes in an up-trend, sell on the first higher close

Three or more down days in a row inside an up-trend is a pause that tends to be bought back quickly.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `days` = 3, `trend_window` = 200

## Run it

```bash
quant backtest --model consecutive_down --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.26, CAGR -0.4%, volatility 1.5%, max drawdown -12.4%, turnover 15.2 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Short-term mean reversion: pullbacks, bands and reversals](../techniques/short-term-mean-reversion.md)
