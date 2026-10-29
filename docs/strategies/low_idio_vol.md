# low_idio_vol

*Family: cross-sectional*

## What it bets on

Low idiosyncratic volatility (Ang-Hodrick-Xing-Zhang): favour assets whose returns are least explained by noise the market does not share

Assets with high residual volatility have been persistently over-priced by investors who treat them as lottery tickets.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252

## Run it

```bash
quant backtest --model low_idio_vol --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.02, CAGR 0.0%, volatility 4.6%, max drawdown -19.0%, turnover 1.3 times a year, deflated Sharpe probability 0.01 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Ang, Hodrick, Xing and Zhang (2006), The cross-section of volatility and expected returns, Journal of Finance 61(1)](https://doi.org/10.1111/j.1540-6261.2006.00836.x)

## Learn more

[Low risk, lottery demand, illiquidity and value with momentum](../techniques/low-risk-and-multi-factor.md)
