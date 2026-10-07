# variance_carry

*Family: volatility*

## What it bets on

Variance carry: be long equities when the volatility term structure is in contango (VIX3M above VIX)

Contango means the market expects calm to continue and pays to hold protection; harvesting it is selling insurance.

## Inputs

- Macro or alternative series required: VIX, VIX3M
- Parameters: none

## Run it

```bash
quant backtest --model variance_carry --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.10, CAGR -1.9%, volatility 12.2%, max drawdown -46.1%, turnover 6.7 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.19 (-0.09 over its own, longer live window), CAGR -3.0%, volatility 12.0%, max drawdown -46.1%, turnover 7.7 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Uses VIX-type indices as the implied-volatility input; no option-level data.

## Learn more

[Fixed income and volatility strategy families](../techniques/fixed-income-and-volatility-strategies.md)
