# xs_momentum

*Family: cross-sectional*

## What it bets on

Classic 12-1 month cross-sectional momentum: rank by the return from 12 months ago to 1 month ago

Winners over the past year, excluding the last month (which tends to reverse), keep outperforming.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 252, `skip` = 21

## Run it

```bash
quant backtest --model xs_momentum --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.32, CAGR 2.0%, volatility 7.2%, max drawdown -19.3%, turnover 6.5 times a year, deflated Sharpe probability 0.13 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe 0.21 (0.07 over its own, longer live window), CAGR 1.0%, volatility 5.7%, max drawdown -14.8%, turnover 5.6 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Jegadeesh and Titman (1993), Returns to buying winners and selling losers: implications for stock market efficiency, Journal of Finance 48(1)](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)

## Learn more

[Momentum and trend following](../techniques/momentum.md)
