# smooth_momentum

*Family: cross-sectional*

## What it bets on

Frog-in-the-pan momentum (Da-Gurun-Warachka): 12-1 month return, boosted when the gain came in many small steps rather than a few jumps

Investors under-react to information that arrives gradually, so momentum built from many small moves lasts longer than momentum from a few big ones.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lookback` = 252, `skip` = 21

## Run it

```bash
quant backtest --model smooth_momentum --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.32, CAGR 2.0%, volatility 7.2%, max drawdown -19.2%, turnover 6.4 times a year, deflated Sharpe probability 0.13 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Da, Gurun and Warachka (2014), Frog in the pan: continuous information and momentum, Review of Financial Studies 27(7)](https://doi.org/10.1093/rfs/hhu003)

## Learn more

[Trend following and momentum: the indicator toolkit](../techniques/trend-following-toolkit.md)
