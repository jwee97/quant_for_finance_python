# faber_gtaa

*Family: allocation*

## What it bets on

Faber's GTAA / Ivy portfolio: hold each asset class with an equal weight only while its price is above its 10-month average, otherwise hold cash

A simple trend filter on each asset class keeps most of the return of buy-and-hold with far shallower drawdowns (Faber 2007).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `months` = 10

## Run it

```bash
quant backtest --model faber_gtaa --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.75, CAGR 6.9%, volatility 9.6%, max drawdown -27.8%, turnover 3.1 times a year, deflated Sharpe probability 0.75 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Tactical asset allocation: Faber, Keller and the classic model portfolios](../techniques/tactical-asset-allocation.md)
