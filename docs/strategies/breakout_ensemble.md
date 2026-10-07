# breakout_ensemble

*Family: time-series*

## What it bets on

Breakout ensemble (Carver): where the price sits in its 10- to 320-day range, smoothed and averaged across horizons

Price near the top of a long range is a trend in progress; averaging many range lengths gives a smooth, low-turnover signal.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookbacks` = (10, 20, 40, 80, 160, 320), `min_horizons` = 3

## Run it

```bash
quant backtest --model breakout_ensemble --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.16, CAGR 0.7%, volatility 5.2%, max drawdown -10.4%, turnover 9.5 times a year, deflated Sharpe probability 0.05 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
