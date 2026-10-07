# butterfly

*Family: fixed income*

## What it bets on

Butterfly: long the 10-year belly against the 2- and 20-year wings, DV01-neutral, when the belly looks cheap

The belly yield relative to the average of the wings mean-reverts: 2 x 10y - 2y - 20y is a stationary curvature measure.

## Inputs

- Macro or alternative series required: DGS2, DGS5, DGS10, DGS20
- Parameters: none

## Run it

```bash
quant backtest --model butterfly --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.03, CAGR -0.5%, volatility 7.7%, max drawdown -24.5%, turnover 10.3 times a year, deflated Sharpe probability 0.01 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.12 (-0.03 over its own, longer live window), CAGR -1.1%, volatility 7.1%, max drawdown -24.5%, turnover 9.4 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Bond ETFs only; durations are estimated from rolling data, not taken from bond analytics.

## Learn more

[Fixed income and volatility strategy families](../techniques/fixed-income-and-volatility-strategies.md)
