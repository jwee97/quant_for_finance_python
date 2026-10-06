# ml_ridge

*Family: machine learning*

## What it bets on

Walk-forward ridge regression of the next 21-day return on price features, optionally plus macro features

Features that describe an asset's recent history (momentum, z-scores, volatility, drawdown, RSI) carry a little forecasting information.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `features` = 'price', `min_train` = 1260, `refit_every` = 252, `embargo` = 21, `alpha` = 100.0

## Run it

```bash
quant backtest --model ml_ridge --tearsheet
```

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.42 (-0.40 over its own, longer live window), CAGR -2.3%, volatility 5.4%, max drawdown -28.9%, turnover 14.3 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.

## Learn more

[Machine learning on returns: ridge, forests and honesty](../techniques/machine-learning-for-returns.md)
