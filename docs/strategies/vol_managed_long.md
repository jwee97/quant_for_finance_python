# vol_managed_long

*Family: time-series*

## What it bets on

Volatility-managed long (Moreira-Muir): hold each asset in inverse proportion to its last month's realised variance, long only

Volatility is persistent but the return earned per unit of risk is not higher when it is high, so shrinking exposure in turbulence raises the Sharpe ratio.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 21, `max_scale` = 3.0, `min_history` = 252

## Run it

```bash
quant backtest --model vol_managed_long --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.81, CAGR 13.5%, volatility 17.7%, max drawdown -30.6%, turnover 10.2 times a year, deflated Sharpe probability 0.83 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Risk-managed exposure: volatility scaling, fear and credit](../techniques/risk-managed-exposure.md)
