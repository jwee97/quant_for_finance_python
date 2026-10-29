# variance_ratio_switch

*Family: time-series*

## What it bets on

Variance-ratio switch (Lo-MacKinlay): follow the last q-day move where the robust variance ratio says moves persist, fade it where it says they reverse, stay out where it cannot tell

Whether a market trends or reverts is itself measurable. A variance ratio above one (autocorrelation of returns positive) favours momentum, below one favours mean reversion, and
a statistic inside +-``z_crit`` is no evidence either way.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252, `q` = 5, `z_crit` = 1.645, `smooth` = False

## Run it

```bash
quant backtest --model variance_ratio_switch --allocator sleeves --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.37, CAGR -0.7%, volatility 1.9%, max drawdown -12.2%, turnover 4.1 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Lo and MacKinlay (1988), Stock market prices do not follow random walks: evidence from a simple specification test, Review of Financial Studies 1(1)](https://doi.org/10.1093/rfs/1.1.41)
- [Lo and MacKinlay (1989), The size and power of the variance ratio test in finite samples, Journal of Econometrics 40(2)](https://doi.org/10.1016/0304-4076(89)90083-3)
- Campbell, Lo and MacKinlay (1997), The Econometrics of Financial Markets, Princeton University Press
- Chan (2013), Algorithmic Trading: Winning Strategies and Their Rationale, Wiley

## Learn more

[Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance](../techniques/book-rules-systematic-trading.md)
