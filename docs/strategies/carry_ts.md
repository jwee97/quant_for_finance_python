# carry_ts

*Family: carry*

## What it bets on

Time-series carry: long an instrument while its carry is positive and short while negative, sized by carry over volatility

Each instrument is its own bet: an upward-sloping reward-for-waiting signal, scaled so a unit of carry-to-risk has the same weight everywhere.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `vol_halflife` = 60.0, `scale` = 0.5, `smooth` = 21

## Run it

```bash
quant backtest --model carry_ts --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## Learn more

[Cross-asset carry, basis momentum and long-term reversal](../techniques/cross-asset-carry.md)
