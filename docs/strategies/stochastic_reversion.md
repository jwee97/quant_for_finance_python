# stochastic_reversion

*Family: time-series*

## What it bets on

Stochastic oscillator reversion: buy when the smoothed %K falls below 20 in an up-trend, sell when it recovers above 50

Where the close sits in its 14-day range, smoothed over three days: a low reading in a rising market marks a pullback rather than a reversal of trend.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 14, `smooth` = 3, `entry` = 20.0, `exit` = 50.0, `trend_window` = 200

## Run it

```bash
quant backtest --model stochastic_reversion --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.76, CAGR 6.6%, volatility 9.0%, max drawdown -19.8%, turnover 2.9 times a year, deflated Sharpe probability 0.77 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Short-term mean reversion: pullbacks, bands and reversals](../techniques/short-term-mean-reversion.md)
