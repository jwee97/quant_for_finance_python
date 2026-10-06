# cointegration_pairs

*Family: stat-arb*

## What it bets on

Rolling-cointegration pairs: trade a pair only while an Engle-Granger test on the last year says it is cointegrated

Trade only relationships that currently look stationary; pairs that fail the test are left alone.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `pairs` = (('EEM', 'EFA'), ('GLD', 'SLV'), ('IEF', 'TLT'), ('LQD', 'HYG'), ('SPY', 'QQQ'), ('IWM', 'SPY')), `window` = 252, `refit_every` = 21, `p_value` = 0.1, `z_window` = 63, `clip` = 3.0

## Run it

```bash
quant backtest --model cointegration_pairs --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe nan, CAGR 0.0%, volatility 0.0%, max drawdown 0.0%, turnover 0.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Needs pairs that are genuinely related; with 15 ETFs there are few. Trades both legs, so costs are doubled.

## Learn more

[Pairs, cointegration and statistical arbitrage](../techniques/pairs-and-statistical-arbitrage.md)
