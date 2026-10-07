# risk_on_off

*Family: macro*

## What it bets on

Risk-on / risk-off: a composite of VIX, the credit spread, the curve and MOVE sets equity and credit exposure against bonds and gold

Stress indicators move together; when they rise, risk assets keep falling for a while and safe assets keep rising.

## Inputs

- Macro or alternative series required: VIX, BAA10Y, T10Y3M, MOVE
- Parameters: none

## Run it

```bash
quant backtest --model risk_on_off --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.07, CAGR 0.2%, volatility 9.0%, max drawdown -32.2%, turnover 6.0 times a year, deflated Sharpe probability 0.02 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.08 (0.02 over its own, longer live window), CAGR -0.6%, volatility 5.8%, max drawdown -23.5%, turnover 5.8 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Macro and alternative data without look-ahead](../techniques/macro-and-alternative-data.md)
