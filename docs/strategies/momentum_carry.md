# momentum_carry

*Family: carry*

## What it bets on

Trend and carry together: the weighted average of a volatility-scaled trend signal and a carry signal for each instrument

Trend pays after a move has started and carry pays while the curve stays as it is, so a book of both is smoother than either (Carver 2015 combines the two as separate forecasts; Fuertes, Miffre and
Rallis 2010 sort commodities on momentum and on the slope of the curve together). Each instrument's signal is ``trend_weight * tanh(z / trend_scale) + (1 - trend_weight) * tanh(carry / (volatility *
carry_scale))``, where ``z`` is the log return from ``lookback`` days ago to ``skip`` days ago over the volatility it should have had (``volatility * sqrt((lookback - skip) / 252)``): both parts lie in
(-1, 1), so the signal does too, and a day on which either part is missing has no signal.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 252, `skip` = 21, `trend_weight` = 0.5, `vol_halflife` = 60.0, `trend_scale` = 1.0, `carry_scale` = 0.5, `smooth` = 21

## Run it

```bash
quant backtest --model momentum_carry --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- Carver (2015), Systematic Trading, Harriman House
- [Fuertes, Miffre and Rallis (2010), Tactical allocation in commodity futures markets: combining momentum and term structure signals, Journal of Banking & Finance 34(10)](https://doi.org/10.1016/j.jbankfin.2010.04.009)
- [Koijen, Moskowitz, Pedersen and Vrugt (2018), Carry, Journal of Financial Economics 127(2)](https://doi.org/10.1016/j.jfineco.2017.11.002)
- [Moskowitz, Ooi and Pedersen (2012), Time series momentum, Journal of Financial Economics 104(2)](https://doi.org/10.1016/j.jfineco.2011.11.003)

## Learn more

[Cross-asset carry, basis momentum and long-term reversal](../techniques/cross-asset-carry.md)
