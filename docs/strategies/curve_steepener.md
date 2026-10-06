# curve_steepener

*Family: fixed income*

## What it bets on

Curve steepener: long the 2-year (SHY), short the 20-year (TLT), DV01-neutral, sized by the momentum of the slope

Curve moves persist: a slope that has been steepening keeps steepening while the Fed or term premia adjust.

## Inputs

- Macro or alternative series required: DGS2, DGS5, DGS10, DGS20
- Parameters: `change_window` = 63

## Run it

```bash
quant backtest --model curve_steepener --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.36 (-0.17 over its own, longer live window), CAGR -4.1%, volatility 10.3%, max drawdown -59.2%, turnover 14.9 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Bond ETFs only; durations are estimated from rolling data, not taken from bond analytics.

## Learn more

[Fixed income and volatility strategy families](../techniques/fixed-income-and-volatility-strategies.md)
