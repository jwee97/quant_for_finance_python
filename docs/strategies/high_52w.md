# high_52w

*Family: cross-sectional*

## What it bets on

52-week-high proximity (George-Hwang): favour assets trading closest to their trailing 252-day high

Investors anchor on the 52-week high and under-react to news that pushes a price toward it, so assets near their highs keep outperforming.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252

## Run it

```bash
quant backtest --model high_52w --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.34, CAGR 1.7%, volatility 5.5%, max drawdown -12.1%, turnover 5.6 times a year, deflated Sharpe probability 0.16 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
