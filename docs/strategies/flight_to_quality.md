# flight_to_quality

*Family: macro*

## What it bets on

Flight to quality: a price-only stress gauge (drawdown of the risky assets, their volatility, an unusually strong bid for bonds) moves the book from risky assets to rates and gold when stress is high

Investors run from risky to safe assets together: Baele, Bekaert, Inghelbrecht and Wei (2020) find flights to safety on the rare days when bonds beat stocks by a wide margin, with money moving from equity funds into government bond and money market funds. Three symptoms measured
from prices alone are averaged into a stress level in [0, 1]: the drawdown of the equal-weight risky basket from its half-year high, its volatility against its own history, and how unusually
negative the stock-bond correlation is. In calm markets the book is long the risky assets; as stress passes ``threshold`` it moves to rates, fixed income and gold (gold ETFs count as havens), and
it shorts the risky assets unless ``short_risky`` is off. It is the version of ``risk_on_off`` that needs only the prices you already have.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `threshold` = 0.5, `high_window` = 126, `vol_window` = 21, `corr_window` = 63, `short_risky` = True

## Run it

```bash
quant backtest --model flight_to_quality --allocator sleeves --tearsheet
```
This rule declares a daily rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.60, CAGR 3.3%, volatility 5.8%, max drawdown -12.0%, turnover 5.4 times a year, deflated Sharpe probability 0.50 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Portfolio rebalancing styles: policy bands, month-end flows, flight to quality, market outlook](../techniques/portfolio-rebalancing-styles.md)
