# rsi2

*Family: time-series*

## What it bets on

Connors RSI(2) pullback: buy when the 2-day RSI is below 10 in an up-trend (price above the 200-day average), sell when price closes above its 5-day average

A deep two-day oversold reading inside a rising market is forced selling that is usually reversed within days (liquidity provision).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `rsi_window` = 2, `entry` = 10.0, `trend_window` = 200, `exit_window` = 5, `shorts` = False

## Run it

```bash
quant backtest --model rsi2 --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.35, CAGR 0.9%, volatility 2.7%, max drawdown -6.1%, turnover 13.9 times a year, deflated Sharpe probability 0.17 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Short-term mean reversion: pullbacks, bands and reversals](../techniques/short-term-mean-reversion.md)
