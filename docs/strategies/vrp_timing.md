# vrp_timing

*Family: volatility*

## What it bets on

Volatility risk premium timing: be long equities when implied variance (VIX squared) exceeds realised variance

Investors pay more for protection than realised volatility justifies; that premium also forecasts equity returns (Bollerslev, Tauchen and Zhou).

## Inputs

- Macro or alternative series required: VIX
- Parameters: `realised_window` = 21, `proxy` = 'SPY'

## Run it

```bash
quant backtest --model vrp_timing --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.18, CAGR 1.4%, volatility 12.0%, max drawdown -49.4%, turnover 8.5 times a year, deflated Sharpe probability 0.05 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.06 (0.22 over its own, longer live window), CAGR 0.0%, volatility 12.0%, max drawdown -49.7%, turnover 9.8 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Uses VIX-type indices as the implied-volatility input; no option-level data.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Bollerslev, Tauchen and Zhou (2009), Expected stock returns and variance risk premia, Review of Financial Studies 22(11)](https://doi.org/10.1093/rfs/hhp008)
- [Carr and Wu (2009), Variance risk premiums, Review of Financial Studies 22(3)](https://doi.org/10.1093/rfs/hhn038)

## Learn more

[Fixed income and volatility strategy families](../techniques/fixed-income-and-volatility-strategies.md)
