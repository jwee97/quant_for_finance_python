# evt_risk_managed

*Family: volatility*

## What it bets on

Tail-risk-managed long: exposure inversely proportional to a conditional-EVT expected-shortfall forecast

Losses, not variance, are what drawdown limits bind on: size positions by the fitted tail so exposure falls when the tail is heavy or the scale is high.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `p` = 0.99, `min_obs` = 750, `refit_every` = 63, `halflife` = 40.0, `max_scale` = 3.0

## Run it

```bash
quant backtest --model evt_risk_managed --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## Caveats

Uses VIX-type indices as the implied-volatility input; no option-level data.
