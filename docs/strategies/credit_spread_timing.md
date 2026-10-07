# credit_spread_timing

*Family: time-series*

## What it bets on

Credit risk appetite: when credit has been beating rates over three months (relative to its own history) hold risk assets, otherwise lean to safety

Credit markets price default risk faster than equities; credit underperforming government bonds warns that risk appetite is fading.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 63, `history` = 504

## Run it

```bash
quant backtest --model credit_spread_timing --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.17, CAGR -0.6%, volatility 3.5%, max drawdown -16.8%, turnover 3.0 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Risk-managed exposure: volatility scaling, fear and credit](../techniques/risk-managed-exposure.md)
