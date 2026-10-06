# pca_residual

*Family: stat-arb*

## What it bets on

PCA residual trading: fade each asset's deviation from what three principal components explain

After removing the common factors, what is left in an asset's return is noise that reverts.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `components` = 3, `window` = 252, `zscore_window` = 21

## Run it

```bash
quant backtest --model pca_residual --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.33 (-0.45 over its own, longer live window), CAGR -1.5%, volatility 4.4%, max drawdown -27.2%, turnover 21.6 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Needs pairs that are genuinely related; with 15 ETFs there are few. Trades both legs, so costs are doubled.

## Learn more

[Pairs, cointegration and statistical arbitrage](../techniques/pairs-and-statistical-arbitrage.md)
