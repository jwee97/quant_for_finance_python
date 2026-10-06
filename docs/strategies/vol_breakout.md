# vol_breakout

*Family: time-series*

## What it bets on

Volatility breakout: go long above the moving average plus k ATRs, short below it minus k ATRs, exit at the average

A move beyond what recent volatility explains marks a new regime of trend (Keltner-channel logic).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 20, `k` = 2.0, `atr_window` = 14

## Run it

```bash
quant backtest --model vol_breakout --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.06 (0.05 over its own, longer live window), CAGR 0.1%, volatility 2.9%, max drawdown -6.1%, turnover 0.6 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Learn more

[Momentum and trend following](../techniques/momentum.md)
