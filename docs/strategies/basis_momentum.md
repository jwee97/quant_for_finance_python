# basis_momentum

*Family: carry*

## What it bets on

Basis momentum: within commodities, favour instruments whose front contract has outperformed the second contract over the last year

When the nearby contract beats the deferred one for months, the market is tightening; the effect has predicted returns beyond ordinary momentum and carry.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `classes` = ('commodity',)

## Run it

```bash
quant backtest --model basis_momentum --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.

