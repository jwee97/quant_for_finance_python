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


## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Boons and Prado (2019), Basis-momentum, Journal of Finance 74(1)](https://doi.org/10.1111/jofi.12738)

## Learn more

[Cross-asset carry, basis momentum and long-term reversal](../techniques/cross-asset-carry.md)
