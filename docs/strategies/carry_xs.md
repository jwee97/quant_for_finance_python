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


## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Koijen, Moskowitz, Pedersen and Vrugt (2018), Carry, Journal of Financial Economics 127(2)](https://doi.org/10.1016/j.jfineco.2017.11.002)

## Learn more

[Cross-asset carry, basis momentum and long-term reversal](../techniques/cross-asset-carry.md)
