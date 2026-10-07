# ma_ensemble

*Family: time-series*

## What it bets on

Exponential moving-average crossover ensemble (Baz et al.): three speeds, volatility-normalised, passed through a response that fades extreme trends

Crossovers at several speeds, normalised by price volatility and the typical size of the signal, capture trends of different lengths without choosing one.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `pairs` = ((8, 24), (16, 48), (32, 96)), `price_window` = 63, `signal_window` = 252

## Run it

```bash
quant backtest --model ma_ensemble --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.26, CAGR 1.1%, volatility 4.8%, max drawdown -13.6%, turnover 4.0 times a year, deflated Sharpe probability 0.09 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
