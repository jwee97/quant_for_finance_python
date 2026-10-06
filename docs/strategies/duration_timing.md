# duration_timing

*Family: fixed income*

## What it bets on

Duration timing: extend duration when yields have been falling, shorten when they have been rising

Yield changes trend (policy is gradual), so recent yield direction forecasts bond returns.

## Inputs

- Macro or alternative series required: DGS10
- Parameters: `window` = 126, `assets` = ('IEF', 'TLT', 'AGG')

## Run it

```bash
quant backtest --model duration_timing --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.22 (-0.19 over its own, longer live window), CAGR -0.7%, volatility 3.1%, max drawdown -13.3%, turnover 0.6 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Bond ETFs only; durations are estimated from rolling data, not taken from bond analytics.

## Learn more

[Fixed income and volatility strategy families](../techniques/fixed-income-and-volatility-strategies.md)
