# value_proxy

*Family: cross-sectional*

## What it bets on

A price-based stand-in for value: long-horizon reversal, favouring assets that have fallen most over five years

Assets that are cheap relative to their own long history mean-revert (De Bondt and Thaler). Not fundamental value: no earnings, no book value.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 1260

## Run it

```bash
quant backtest --model value_proxy --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.16 (0.16 over its own, longer live window), CAGR 0.1%, volatility 0.8%, max drawdown -2.8%, turnover 0.5 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Cross-sectional factors: low volatility, value, quality, carry, defensive beta](../techniques/cross-sectional-factors.md)
