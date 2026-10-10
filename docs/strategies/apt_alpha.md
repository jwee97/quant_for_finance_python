# apt_alpha

*Family: cross-sectional*

## What it bets on

Statistical APT: remove the first K principal components from the last two years of returns and rank assets by the appraisal ratio of what they earned beyond them

The arbitrage pricing theory says only the exposures to a few common factors are paid; anything an asset earned beyond them is a mispricing (or luck). With no named factors the common ones are
taken to be the first ``factors`` principal components of the trailing ``window`` days of returns, refitted every ``refit`` days, and the score is the intercept of each asset's regression on them over
its residual volatility (an appraisal ratio), skipping the last ``skip`` days so the short-term reversal does not leak into a long-term signal. The generalisation of ``jensen_alpha`` from one factor
(the market) to several.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `factors` = 3, `window` = 504, `skip` = 21, `refit` = 21

## Run it

```bash
quant backtest --model apt_alpha --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.44, CAGR 1.8%, volatility 4.2%, max drawdown -18.9%, turnover 4.4 times a year, deflated Sharpe probability 0.22 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Factor models and machine learning: statistical APT, macroeconomic factors, cross-sectional characteristics, LASSO to neural networks](../techniques/factor-models-and-machine-learning.md)
