# tsmom_multi

*Family: time-series*

## What it bets on

Multi-horizon time-series momentum: the average of the signs of the 1-, 3- and 12-month returns (a century of evidence)

Averaging trend signals over several horizons keeps most of the premium and halves the turnover of any single lookback.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `horizons` = (21, 63, 252)

## Run it

```bash
quant backtest --model tsmom_multi --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.01, CAGR -0.2%, volatility 6.5%, max drawdown -26.2%, turnover 12.5 times a year, deflated Sharpe probability 0.01 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Hurst, Ooi and Pedersen (2017), A century of evidence on trend-following investing, Journal of Portfolio Management 44(1)](https://doi.org/10.3905/jpm.2017.44.1.015)
- [Moskowitz, Ooi and Pedersen (2012), Time series momentum, Journal of Financial Economics 104(2)](https://doi.org/10.1016/j.jfineco.2011.11.003)

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
