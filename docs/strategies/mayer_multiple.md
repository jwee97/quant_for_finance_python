# mayer_multiple

*Family: time-series*

## What it bets on

Mayer multiple: hold while the price is under 2.4 times its 200-day average (an exit from euphoria); with a floor of 1 it is a trend filter with an exit

The price over its 200-day average, the measure Trace Mayer popularised for Bitcoin: a multiple above about 2.4 has been called overheated and one below 1 cheap. The rule holds the asset (score 1)
while the multiple is at least ``floor`` and under ``cap`` and sits out (score 0) otherwise: ``floor = 0`` (the default) is buy and hold with an exit from euphoria, ``floor = 1`` also requires the
price to be above its average, a trend filter with an exit. The thresholds are a practitioner's, not tested in a paper; moving-average rules themselves have been (Brock, Lakonishok and LeBaron 1992
for stocks, Gerritsen et al. 2020 and Detzel et al. 2021 for Bitcoin).

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 200, `cap` = 2.4, `floor` = 0.0

## Run it

```bash
quant backtest --model mayer_multiple --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Brock, Lakonishok and LeBaron (1992), Simple technical trading rules and the stochastic properties of stock returns, Journal of Finance 47(5)](https://doi.org/10.1111/j.1540-6261.1992.tb04681.x)
- [Gerritsen, Bouri, Ramezanifar and Roubaud (2020), The profitability of technical trading rules in the Bitcoin market, Finance Research Letters 34](https://doi.org/10.1016/j.frl.2019.08.011)
- [Detzel, Liu, Strauss, Zhou and Zhu (2021), Learning and predictability via technical analysis: evidence from bitcoin and stocks with hard-to-value fundamentals, Financial Management 50(1)](https://doi.org/10.1111/fima.12310)

## Learn more

[Crypto: trend rules, the Mayer multiple, the cone of volatility and what every rule says](../techniques/crypto-trend-volatility-and-signals.md)
