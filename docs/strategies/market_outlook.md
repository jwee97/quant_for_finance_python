# market_outlook

*Family: allocation*

## What it bets on

Top-down market outlook: trend, twelve-month momentum, breadth and calm of the risky assets add up to an outlook in [-1, 1] that tilts the book between risky and defensive assets

A manager's market view, made mechanical. Four readings of the equal-weight risky basket are each squashed to [-1, 1] and averaged: how far its price is from its ``trend``-day average (in units of
its own monthly volatility), its ``momentum``-day return, the share of risky assets above their own averages (breadth: a rally carried by few assets is fragile) and the calm of its volatility against
its own history. The outlook times ``tilt`` is the score of every risky asset (long when bullish, short when bearish) and, when it is negative, the score of every safe asset (bearish views move
into safety; a bullish view does not short bonds, which would pay their carry for nothing). ``bias`` adds a constant opinion of your own (positive is bullish) to the computed outlook.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `trend` = 200, `momentum` = 252, `tilt` = 1.0, `bias` = 0.0, `min_periods` = 252

## Run it

```bash
quant backtest --model market_outlook --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.14, CAGR 0.5%, volatility 4.2%, max drawdown -12.8%, turnover 2.5 times a year, deflated Sharpe probability 0.03 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Learn more

[Portfolio rebalancing styles: policy bands, month-end flows, flight to quality, market outlook](../techniques/portfolio-rebalancing-styles.md)
