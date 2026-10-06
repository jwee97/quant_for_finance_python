# yield_curve_regime

*Family: macro*

## What it bets on

Yield-curve regimes: defensive (bonds, gold, no equity) when the 10y-3m curve is inverted, risk-on otherwise

An inverted curve has preceded most recessions; the market's own forecast of weaker growth and lower rates.

## Inputs

- Macro or alternative series required: T10Y3M
- Parameters: none

## Run it

```bash
quant backtest --model yield_curve_regime --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.50 (0.52 over its own, longer live window), CAGR 3.1%, volatility 6.6%, max drawdown -23.4%, turnover 3.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Macro and alternative data without look-ahead](../techniques/macro-and-alternative-data.md)
