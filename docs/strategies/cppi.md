# cppi

*Family: risk-managed*

## What it bets on

Constant-proportion portfolio insurance (Black-Jones, Perold): hold a multiple of the cushion above a floor in the asset and the rest in cash; the ratchet option raises the floor with the account's high (TIPP)

The multiple times the cushion is the amount at risk, so the account cannot fall through the floor unless the asset gaps by more than one over the multiple in a day. Each asset runs its own account.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `multiplier` = 4.0, `floor` = 0.85, `max_exposure` = 1.0, `cash_rate` = 0.0, `ratchet` = False

## Run it

```bash
quant backtest --model cppi --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.70, CAGR 2.8%, volatility 4.2%, max drawdown -14.8%, turnover 1.1 times a year, deflated Sharpe probability 0.67 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Black and Jones (1987), Simplifying portfolio insurance, Journal of Portfolio Management 14(1)](https://doi.org/10.3905/jpm.1987.409131)
- [Perold and Sharpe (1988), Dynamic strategies for asset allocation, Financial Analysts Journal 44(1)](https://doi.org/10.2469/faj.v44.n1.16)
- [Black and Perold (1992), Theory of constant proportion portfolio insurance, Journal of Economic Dynamics and Control 16(3-4)](https://doi.org/10.1016/0165-1889(92)90043-e)
- [Estep and Kritzman (1988), TIPP: insurance without complexity, Journal of Portfolio Management 14(4)](https://doi.org/10.3905/jpm.1988.409172)

## Learn more

[Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance](../techniques/book-rules-systematic-trading.md)
