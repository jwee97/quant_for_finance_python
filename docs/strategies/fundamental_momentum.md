# fundamental_momentum

*Family: fundamental*

## What it bets on

Momentum and revisions: the one-month reversal, nine-month price momentum (raw and per unit of risk) and, with analyst columns, earnings revisions and diffusion

Stocks keep moving the way they have for about nine months (skipping the latest month, which reverses), and analysts' earnings estimates drift the same way for months because they are revised a
little at a time. The three price factors need no file; the three estimate factors need ``eps_fy1``, ``n_up``, ``n_down``, ``n_estimates`` and ``ltg`` in ``data/user/fundamentals.csv``. The composite
uses the price factors, plus the estimate factors when the file has their columns.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `factor` = 'composite', `path` = 'fundamentals.csv', `lag_days` = 60, `expiry` = 400, `oriented` = True

## Run it

```bash
quant backtest --model fundamental_momentum --tearsheet
```

## Caveats

Reads company statements from a file you supply (data/user/fundamentals.csv); without it the model refuses to run, except the price-only momentum factors. Needs a wide cross-section of stocks, not 15 ETFs, and statements that are point-in-time.

## Learn more

[Fundamental factors: value, quality, momentum, DCF, and nonlinear and contextual effects](../techniques/fundamental-factors.md)
