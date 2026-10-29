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

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Novy-Marx (2013), The other side of value: the gross profitability premium, Journal of Financial Economics 108(1)](https://doi.org/10.1016/j.jfineco.2013.01.003)
- [Piotroski (2000), Value investing: the use of historical financial statement information to separate winners from losers, Journal of Accounting Research 38(S)](https://doi.org/10.2307/2672906)
- [Sloan (1996), Do stock prices fully reflect information in accruals and cash flows about future earnings?, The Accounting Review 71(3)](https://doi.org/10.2308/tar-9608042309)
- [Asness, Frazzini and Pedersen (2019), Quality minus junk, Review of Accounting Studies 24(1)](https://doi.org/10.1007/s11142-018-9470-2)

## Learn more

[Fundamental factors: value, quality, momentum, DCF, and nonlinear and contextual effects](../techniques/fundamental-factors.md)
