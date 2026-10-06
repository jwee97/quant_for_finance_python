# ma_crossover

*Family: time-series*

## What it bets on

Moving-average crossover: the 50-day average against the 200-day average, scaled by volatility

Prices above their long average are in an up-trend; a smooth version of the golden/death cross.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `fast` = 50, `slow` = 200

## Run it

```bash
quant backtest --model ma_crossover --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.50 (0.61 over its own, longer live window), CAGR 3.0%, volatility 6.4%, max drawdown -15.4%, turnover 5.3 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Learn more

[Momentum and trend following](../techniques/momentum.md)
