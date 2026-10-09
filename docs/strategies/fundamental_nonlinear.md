# fundamental_nonlinear

*Family: fundamental*

## What it bets on

Nonlinear effects: factors plus their squares, products and conditional versions, forecast with the trailing average cross-sectional (Fama-MacBeth) slopes

Returns are not always a straight line in a factor (capital expenditure is the textbook case: both starved and gorged firms do worse than the middle). Each month the next month's returns are
regressed on the linear factors and the terms given (``quadratic:f``, ``interaction:f:g``, ``conditional:f:g:high``) across stocks, and a stock's forecast is the trailing mean of those slopes (those
whose month has ended) times its terms. A term that earns nothing gets a slope near zero and so little weight. Needs ``data/user/fundamentals.csv``.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `factors` = 'b2p,rnoa,icapx,ret9', `terms` = 'quadratic:icapx', `window` = 60, `min_obs` = 24, `path` = 'fundamentals.csv', `lag_days` = 60, `expiry` = 400

## Run it

```bash
quant backtest --model fundamental_nonlinear --tearsheet
```

## Caveats

Reads company statements from a file you supply (data/user/fundamentals.csv); without it the model refuses to run, except the price-only momentum factors. Needs a wide cross-section of stocks, not 15 ETFs, and statements that are point-in-time.

## Learn more

[Fundamental factors: value, quality, momentum, DCF, and nonlinear and contextual effects](../techniques/fundamental-factors.md)
