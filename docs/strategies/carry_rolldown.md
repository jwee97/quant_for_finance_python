# carry_rolldown

*Family: fixed income*

## What it bets on

Carry plus roll-down: favour the bond ETFs with the highest yield over cash plus the roll down the curve

A bond earns its yield and, if the curve is unchanged, ages down it: price gains of duration times the local slope per year.

## Inputs

- Macro or alternative series required: DGS2, DGS5, DGS10, DGS20, DFF
- Parameters: none

## Run it

```bash
quant backtest --model carry_rolldown --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.17 (0.09 over its own, longer live window), CAGR 0.9%, volatility 6.4%, max drawdown -15.0%, turnover 4.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Bond ETFs only; durations are estimated from rolling data, not taken from bond analytics.

## Learn more

[Fixed income and volatility strategy families](../techniques/fixed-income-and-volatility-strategies.md)
