# sparse_basket

*Family: stat-arb*

## What it bets on

Sparse basket mean reversion: regress each asset on a LASSO-selected basket of the others and fade the cumulative residual

The few other assets that actually explain an ETF define its fair value; the residual from that sparse basket reverts.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252, `refit_every` = 21, `alpha` = 0.05, `z_window` = 21

## Run it

```bash
quant backtest --model sparse_basket --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.18 (-0.16 over its own, longer live window), CAGR -0.2%, volatility 1.0%, max drawdown -5.2%, turnover 1.7 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Needs pairs that are genuinely related; with 15 ETFs there are few. Trades both legs, so costs are doubled.

## Learn more

[Pairs, cointegration and statistical arbitrage](../techniques/pairs-and-statistical-arbitrage.md)
