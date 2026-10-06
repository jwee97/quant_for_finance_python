# quality_proxy

*Family: cross-sectional*

## What it bets on

A price-based stand-in for quality: smoothness of the one-year price path (R-squared of log price on time)

Steady, smooth gainers are held by patient owners and drift further (the 'frog in the pan' effect). Not accounting quality: no profitability or leverage.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252

## Run it

```bash
quant backtest --model quality_proxy --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.28 (-0.50 over its own, longer live window), CAGR -1.9%, volatility 6.0%, max drawdown -35.1%, turnover 5.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Cross-sectional factors: low volatility, value, quality, carry, defensive beta](../techniques/cross-sectional-factors.md)
