# low_volatility

*Family: cross-sectional*

## What it bets on

Low volatility: favour the assets with the lowest trailing volatility

Investors overpay for risky assets (leverage constraints, lottery preferences), so the calm ones earn more per unit of risk.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252

## Run it

```bash
quant backtest --model low_volatility --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.06, CAGR 0.2%, volatility 5.9%, max drawdown -23.8%, turnover 1.3 times a year, deflated Sharpe probability 0.02 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe nan (-0.38 over its own, longer live window), CAGR 0.0%, volatility 0.0%, max drawdown 0.0%, turnover 0.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Cross-sectional factors: low volatility, value, quality, carry, defensive beta](../techniques/cross-sectional-factors.md)
