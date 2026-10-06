# defensive_beta

*Family: cross-sectional*

## What it bets on

Defensive / betting against beta: favour assets with the lowest beta to the market proxy

High-beta assets are bid up by constrained investors, so low-beta assets have higher risk-adjusted returns.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252

## Run it

```bash
quant backtest --model defensive_beta --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe nan (0.10 over its own, longer live window), CAGR 0.0%, volatility 0.0%, max drawdown 0.0%, turnover 0.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Cross-sectional factors: low volatility, value, quality, carry, defensive beta](../techniques/cross-sectional-factors.md)
