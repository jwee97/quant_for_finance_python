# tstat_momentum

*Family: time-series*

## What it bets on

t-statistic trend (Baltas-Kosowski): the t-statistic of a year of daily log returns, capped at plus or minus one, in place of the sign of the year's return

The sign of a return throws away how reliable the trend was. The t-statistic of the mean daily return weighs a steady drift above a lucky jump and moves in and out of positions gradually.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 252, `scale` = 1.0

## Run it

```bash
quant backtest --model tstat_momentum --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.22, CAGR 1.3%, volatility 6.9%, max drawdown -23.4%, turnover 4.8 times a year, deflated Sharpe probability 0.05 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Baltas and Kosowski (2020), Demystifying time-series momentum strategies: volatility estimators, trading rules and pairwise correlations, in Market Momentum, Wiley](https://doi.org/10.1002/9781119599364.ch3)
- [Baltas and Kosowski (2013), Momentum strategies in futures markets and trend-following funds, SSRN 1968996](https://doi.org/10.2139/ssrn.1968996)

## Learn more

[Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance](../techniques/book-rules-systematic-trading.md)
