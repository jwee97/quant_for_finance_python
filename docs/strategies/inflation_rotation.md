# inflation_rotation

*Family: macro*

## What it bets on

Inflation rotation: overweight commodities and gold and underweight long bonds when breakeven inflation is rising

Real assets hedge rising inflation; nominal long bonds are hurt by it.

## Inputs

- Macro or alternative series required: T10YIE
- Parameters: `window` = 126

## Run it

```bash
quant backtest --model inflation_rotation --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.38, CAGR -4.6%, volatility 10.7%, max drawdown -64.1%, turnover 8.2 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.47 (-0.23 over its own, longer live window), CAGR -4.6%, volatility 9.2%, max drawdown -49.2%, turnover 8.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Macro and alternative data without look-ahead](../techniques/macro-and-alternative-data.md)
