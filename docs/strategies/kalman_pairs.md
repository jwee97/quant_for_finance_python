# kalman_pairs

*Family: stat-arb*

## What it bets on

Pairs trading with a Kalman-filter hedge ratio: fade the standardised innovation of each pair

A pair's hedge ratio drifts; filtering it avoids the stale-regression problem of a fixed-window hedge.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `pairs` = (('EEM', 'EFA'), ('GLD', 'SLV'), ('IEF', 'TLT'), ('LQD', 'HYG'), ('SPY', 'QQQ'), ('IWM', 'SPY')), `delta` = 1e-05, `obs_var` = 0.001, `clip` = 3.0

## Run it

```bash
quant backtest --model kalman_pairs --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.06 (0.01 over its own, longer live window), CAGR -0.3%, volatility 3.6%, max drawdown -8.4%, turnover 24.5 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Needs pairs that are genuinely related; with 15 ETFs there are few. Trades both legs, so costs are doubled.

## Learn more

[Pairs, cointegration and statistical arbitrage](../techniques/pairs-and-statistical-arbitrage.md)
