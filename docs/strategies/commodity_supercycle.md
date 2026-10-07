# commodity_supercycle

*Family: macro*

## What it bets on

Commodity supercycle: follow the five-year trend of the commodity assets

Commodity cycles last a decade because supply responds slowly; a five-year rise signals the upswing is under way.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 1260

## Run it

```bash
quant backtest --model commodity_supercycle --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.34, CAGR 3.3%, volatility 11.5%, max drawdown -34.9%, turnover 1.8 times a year, deflated Sharpe probability 0.12 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.33 (0.33 over its own, longer live window), CAGR 3.2%, volatility 11.5%, max drawdown -34.9%, turnover 1.8 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Macro and alternative data without look-ahead](../techniques/macro-and-alternative-data.md)
