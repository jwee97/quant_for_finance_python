# momentum

*Family: time-series/cross-sectional*

## What it bets on

Generation 1 momentum: 126-day return, skipping the last day, divided by volatility

Assets that have risen keep rising for a while (under-reaction, herding, slow-moving capital).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 126, `skip` = 1, `vol_lookback` = 63

## Run it

```bash
quant backtest --model momentum --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.35, CAGR 2.1%, volatility 6.7%, max drawdown -14.8%, turnover 9.3 times a year, deflated Sharpe probability 0.17 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.15 (-0.16 over its own, longer live window), CAGR -1.1%, volatility 5.9%, max drawdown -30.6%, turnover 7.7 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Learn more

[Momentum and trend following](../techniques/momentum.md)
