# short_term_reversal

*Family: cross-sectional*

## What it bets on

Short-term reversal (Jegadeesh, Lehmann): buy last week's laggards and sell its leaders, in volatility units

Last week's biggest movers partly reverse as liquidity providers are paid for absorbing order-flow imbalance.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 5, `vol_window` = 63

## Run it

```bash
quant backtest --model short_term_reversal --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.31, CAGR -2.5%, volatility 7.2%, max drawdown -46.6%, turnover 98.7 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Short-term mean reversion: pullbacks, bands and reversals](../techniques/short-term-mean-reversion.md)
