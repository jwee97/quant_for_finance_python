# carry_xs

*Family: carry*

## What it bets on

Cross-asset carry: rank instruments by carry per unit of volatility within their asset class and hold the high-carry ones against the low-carry ones

Carry is the return an instrument earns if prices do not move (roll yield, interest differential, yield over cash): a premium for bearing risk that has appeared in every liquid asset class.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `vol_halflife` = 60.0, `clip` = 3.0, `per_class` = True

## Run it

```bash
quant backtest --model carry_xs --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.

