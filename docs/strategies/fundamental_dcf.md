# fundamental_dcf

*Family: fundamental*

## What it bets on

Discounted cash flow value per share against the price, as a point estimate or as the median of a Monte Carlo of the uncertain inputs (multipath DCF)

A stock is worth the cash it will pay out, discounted. Free cash flow grows from a first-stage rate (the analysts' long-term growth, or the last three years' sales growth) fading to a terminal rate,
and is discounted at a rate built from the stock's own beta. The multipath version draws growth, starting cash flow, discount rate and terminal rate from distributions centred on those inputs, values
every draw and uses the median value, or the share of draws above the price, so a stock whose value is high only if everything goes right ranks below one that is cheap across the range.
``factor`` is ``dcf_upside``, ``mdcf_upside`` or ``mdcf_prob``. Needs ``data/user/fundamentals.csv``; values are refreshed monthly.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `factor` = 'mdcf_upside', `path` = 'fundamentals.csv', `lag_days` = 60, `expiry` = 400, `risk_free` = 0.03, `equity_premium` = 0.05, `terminal_growth` = 0.025, `years` = 10, `paths` = 400

## Run it

```bash
quant backtest --model fundamental_dcf --tearsheet
```

## Caveats

Reads company statements from a file you supply (data/user/fundamentals.csv); without it the model refuses to run, except the price-only momentum factors. Needs a wide cross-section of stocks, not 15 ETFs, and statements that are point-in-time.

## Learn more

[Fundamental factors: value, quality, momentum, DCF, and nonlinear and contextual effects](../techniques/fundamental-factors.md)
