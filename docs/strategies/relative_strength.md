# relative_strength

*Family: cross-sectional*

## What it bets on

Relative strength: six-month change in each asset's price relative to the market proxy

An asset whose price ratio to the market is rising is gaining leadership.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 126

## Run it

```bash
quant backtest --model relative_strength --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe nan (0.14 over its own, longer live window), CAGR 0.0%, volatility 0.0%, max drawdown 0.0%, turnover 0.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Momentum and trend following](../techniques/momentum.md)
