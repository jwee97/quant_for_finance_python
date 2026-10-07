# vix_spike_reversion

*Family: time-series*

## What it bets on

Buy fear: hold equities after the VIX jumps more than two standard deviations above its 63-day mean, exit when it falls back to the mean

Implied-volatility spikes overshoot because investors pay up for protection, and equities have tended to recover once the panic fades.

## Inputs

- Macro or alternative series required: VIX
- Parameters: `window` = 63, `entry_z` = 2.0, `classes` = ('equity',)

## Run it

```bash
quant backtest --model vix_spike_reversion --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.24, CAGR 1.1%, volatility 5.1%, max drawdown -16.5%, turnover 2.2 times a year, deflated Sharpe probability 0.08 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Learn more

[Risk-managed exposure: volatility scaling, fear and credit](../techniques/risk-managed-exposure.md)
