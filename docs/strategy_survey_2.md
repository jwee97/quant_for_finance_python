# Strategy survey, part two: investment, portfolio and economic-outlook strategies

13 strategies added after the first survey ([strategy_survey.md](strategy_survey.md): 79 strategies), run the same way: the platform's 15 ETFs, 2006-01-03 to 2026-09-21 (20.7 years), net of costs, default parameters,
nothing tuned. **The deflated Sharpe probability here counts all 92 strategies as trials** (79 before, 13 now). The first survey's probabilities counted 79, so they are slightly flattering next to these. This is a survey, not a study.

How to read it: several of these rules are made for single stocks, company news or data you supply, and ETFs are not their natural test bed; a negative row on 15 ETFs says the rule does not pay on liquid funds at these costs, not that it fails on the assets it was written for. Rules that learn from earlier data (`curve_quadrant`, `credit_cycle_rotation`, `event_study_drift`) start late and stay flat while they have no evidence.

| Strategy | Family | Net Sharpe | Equal weight, same dates | CAGR | Volatility | Max drawdown | Turnover (x/yr) | Deflated Sharpe | First day |
|---|---|---|---|---|---|---|---|---|---|
| [policy_portfolio](strategies/policy_portfolio.md) | allocation | +0.83 | +0.65 | 9.1% | 11.4% | -26.6% | 1.4 | 0.85 | 2008-03-03 |
| [curve_quadrant](strategies/curve_quadrant.md) | macro | +0.70 | +0.78 | 2.1% | 3.0% | -8.5% | 3.2 | 0.45 | 2015-02-26 |
| [flight_to_quality](strategies/flight_to_quality.md) | macro | +0.60 | +0.95 | 3.3% | 5.8% | -12.0% | 5.4 | 0.50 | 2009-03-03 |
| [credit_cycle_rotation](strategies/credit_cycle_rotation.md) | macro | +0.43 | +0.91 | 0.8% | 1.9% | -6.5% | 1.9 | 0.13 | 2016-03-01 |
| [market_outlook](strategies/market_outlook.md) | allocation | +0.14 | +0.93 | 0.5% | 4.2% | -12.8% | 2.5 | 0.03 | 2009-03-17 |
| [jensen_alpha](strategies/jensen_alpha.md) | cross-sectional | +0.13 | +0.84 | 0.6% | 6.6% | -19.4% | 3.0 | 0.02 | 2009-09-02 |
| [rebalancing_flow](strategies/rebalancing_flow.md) | seasonal | +0.02 | +0.87 | 0.0% | 1.7% | -5.6% | 7.7 | 0.01 | 2009-01-29 |
| [squeeze_breakout](strategies/squeeze_breakout.md) | time-series | -0.41 | +0.66 | -1.8% | 4.2% | -32.5% | 12.5 | 0.00 | 2008-05-14 |
| [abnormal_volume_drift](strategies/abnormal_volume_drift.md) | time-series | -0.55 | +0.68 | -1.9% | 3.4% | -31.4% | 15.3 | 0.00 | 2008-01-17 |
| [event_study_drift](strategies/event_study_drift.md) | event-driven | -0.78 | +0.68 | -0.2% | 0.3% | -4.4% | 3.9 | 0.00 | 2008-01-17 |
| [adaptive_autocorrelation](strategies/adaptive_autocorrelation.md) | time-series | -0.84 | +0.68 | -6.7% | 7.9% | -78.7% | 116.2 | 0.00 | 2008-07-11 |

Could not run on this bundle:

- `news_sentiment`: KeyError: 'news_sentiment needs the file /home/user/quant_for_finance_python/data/user/headlines.csv. Columns: date, tic
- `panel_signal`: KeyError: 'panel_signal needs the file /home/user/quant_for_finance_python/data/user/signals.csv. Wide: date plus one co

Regenerate with `python -m experiments.library_survey --new`; the next full run (`python -m experiments.library_survey`) folds these rows into the first survey.
