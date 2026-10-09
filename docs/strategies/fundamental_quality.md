# fundamental_quality

*Family: fundamental*

## What it bets on

Quality from company statements: return on operating assets, CFROI, operating leverage, accruals, capital expenditure, external financing and share issuance

Companies that earn high returns on the capital they use, and fund their growth from inside, have earned more than those that grow by absorbing capital: accruals and capital expenditure
above the firm's own history, new debt and new shares all signal a firm that is spending more than it earns, which the market is slow to price. ``factor`` picks one of ten or the composite, each
turned round where the literature expects the high end to be bad. Needs ``data/user/fundamentals.csv``.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `factor` = 'composite', `path` = 'fundamentals.csv', `lag_days` = 60, `expiry` = 400, `oriented` = True

## Run it

```bash
quant backtest --model fundamental_quality --tearsheet
```

## Caveats

Reads company statements from a file you supply (data/user/fundamentals.csv); without it the model refuses to run, except the price-only momentum factors. Needs a wide cross-section of stocks, not 15 ETFs, and statements that are point-in-time.

## Learn more

[Fundamental factors: value, quality, momentum, DCF, and nonlinear and contextual effects](../techniques/fundamental-factors.md)
