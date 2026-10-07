# dual_momentum

*Family: time-series*

## What it bets on

Antonacci dual momentum: hold the top-k assets by 12-month return, but only those that beat cash

Absolute momentum (beat cash) keeps you out of bear markets; relative momentum (rank) picks the leaders.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 252, `skip` = 21, `top_k` = 4, `cash` = 'SHY'

## Run it

```bash
quant backtest --model dual_momentum --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.88, CAGR 9.8%, volatility 11.3%, max drawdown -20.1%, turnover 5.3 times a year, deflated Sharpe probability 0.89 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.90 (0.86 over its own, longer live window), CAGR 10.2%, volatility 11.6%, max drawdown -20.1%, turnover 5.1 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Learn more

[Momentum and trend following](../techniques/momentum.md)
