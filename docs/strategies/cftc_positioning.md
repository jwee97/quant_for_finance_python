# cftc_positioning

*Family: macro*

## What it bets on

CFTC positioning: fade crowded speculative net positioning in the futures matching each ETF (sign configurable)

When leveraged funds or managed money are extremely long, the marginal buyer may already be in. The Generation 3 test found no
significant predictive power for the next month, so treat this as a documented null result and a template for non-price data, not an edge.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `sign` = -1, `min_periods` = 756, `mapping` = None

## Run it

```bash
quant backtest --model cftc_positioning --tearsheet
```

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Macro and alternative data without look-ahead](../techniques/macro-and-alternative-data.md)
