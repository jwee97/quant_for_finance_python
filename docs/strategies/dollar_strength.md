# dollar_strength

*Family: macro*

## What it bets on

Dollar strength: short emerging-market equities, commodities and gold when the broad dollar is rising

A stronger dollar tightens global financial conditions and weighs on dollar-priced commodities and emerging markets.

## Inputs

- Macro or alternative series required: DTWEXBGS
- Parameters: `window` = 126, `exposed` = ('EEM', 'DBC', 'GLD', 'SLV')

## Run it

```bash
quant backtest --model dollar_strength --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.26 (0.26 over its own, longer live window), CAGR 2.3%, volatility 11.7%, max drawdown -41.0%, turnover 4.4 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Macro and alternative data without look-ahead](../techniques/macro-and-alternative-data.md)
