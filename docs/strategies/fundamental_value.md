# fundamental_value

*Family: fundamental*

## What it bets on

Value from company statements: cash flow, EBITDA, earnings, payout, net financing, book value or sales against what the market asks (one factor or the composite)

Cheap stocks, measured by what a dollar of market value buys of the company's cash flow, EBITDA, earnings, payout, book value or sales, have earned more than expensive ones, a premium that is either
compensation for distress or the market extrapolating bad news too far. Enterprise value (equity plus debt and preferred less cash) is the base for the operating measures, market value for the
shareholder measures. A factor is the ratio as it stood on the date, from the latest filing public by then. ``factor`` picks one of the eight or the equal-weighted composite. Needs
``data/user/fundamentals.csv``.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `factor` = 'composite', `path` = 'fundamentals.csv', `lag_days` = 60, `expiry` = 400, `oriented` = True

## Run it

```bash
quant backtest --model fundamental_value --tearsheet
```

## Caveats

Reads company statements from a file you supply (data/user/fundamentals.csv); without it the model refuses to run, except the price-only momentum factors. Needs a wide cross-section of stocks, not 15 ETFs, and statements that are point-in-time.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Fama and French (1993), Common risk factors in the returns on stocks and bonds, Journal of Financial Economics 33(1)](https://doi.org/10.1016/0304-405x(93)90023-5)
- [Novy-Marx (2013), The other side of value: the gross profitability premium, Journal of Financial Economics 108(1)](https://doi.org/10.1016/j.jfineco.2013.01.003)

## Learn more

[Fundamental factors: value, quality, momentum, DCF, and nonlinear and contextual effects](../techniques/fundamental-factors.md)
