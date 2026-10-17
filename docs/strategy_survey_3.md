# Strategy survey, part three: factor timing, factor models, equity factors and an Ornstein-Uhlenbeck pairs rule

13 strategies added after parts one and two ([strategy_survey.md](strategy_survey.md), [strategy_survey_2.md](strategy_survey_2.md): 92 strategies), run the same way: the platform's 15 ETFs, 2006-01-03 to 2026-09-21 (20.7 years), net of costs, default parameters,
nothing tuned. **The deflated Sharpe probability here counts all 105 strategies as trials** (92 before, 13 now). The first survey's probabilities counted 92, so they are slightly flattering next to these. This is a survey, not a study.

How to read it: these are the factor-timing, characteristic-regression, APT, macro-factor and equity-factor rules of the two books. Rules that read a file the repository does not have (`fundamental_*`, `earnings_season_premium`) are listed under *could not run*, not scored. Rules that learn from earlier data start late and stay flat while they have no evidence. A negative or flat row on 15 ETFs says the rule does not pay on liquid funds at these costs, not that it fails on the single stocks it was written for. `ou_threshold_pairs` (the Ornstein-Uhlenbeck pairs rule) trades a pair only while its spread passes a Dickey-Fuller test with the Engle-Granger critical value, which the six default ETF pairs do on 4 to 10% of days: its row is mostly cash, and it says the pairs here are not cointegrated well enough to trade, which is what the test is for.

| Strategy | Family | Net Sharpe | Equal weight, same dates | CAGR | Volatility | Max drawdown | Turnover (x/yr) | Deflated Sharpe | First day |
|---|---|---|---|---|---|---|---|---|---|
| [earnings_season_premium](strategies/earnings_season_premium.md) | event-driven | +0.55 | +0.72 | 2.9% | 5.5% | -17.3% | 13.9 | 0.35 | 2011-05-27 |
| [apt_alpha](strategies/apt_alpha.md) | cross-sectional | +0.44 | +0.81 | 1.8% | 4.2% | -18.9% | 4.4 | 0.22 | 2010-03-05 |
| [macro_factor_timing](strategies/macro_factor_timing.md) | macro | +0.26 | +0.73 | 1.5% | 6.5% | -18.9% | 8.2 | 0.06 | 2011-03-01 |
| [calendar_factor_timing](strategies/calendar_factor_timing.md) | seasonal | +0.21 | +0.82 | 1.2% | 6.5% | -22.0% | 11.0 | 0.05 | 2010-03-02 |
| [fundamental_momentum](strategies/fundamental_momentum.md) | fundamental | +0.08 | +0.65 | 0.3% | 7.3% | -26.1% | 13.2 | 0.01 | 2008-03-05 |
| [macro_factor_model](strategies/macro_factor_model.md) | macro | -0.06 | +0.73 | -0.6% | 6.0% | -28.5% | 4.8 | 0.00 | 2011-03-01 |
| [ou_threshold_pairs](strategies/ou_threshold_pairs.md) | stat-arb | -0.16 | +0.65 | -0.1% | 0.7% | -3.0% | 1.5 | 0.00 | 2008-02-04 |
| [characteristic_regression](strategies/characteristic_regression.md) | cross-sectional | -0.32 | +0.73 | -2.3% | 6.7% | -37.9% | 15.3 | 0.00 | 2011-03-01 |

Could not run on this bundle:

- `fundamental_alpha`: KeyError: 'this fundamental model needs the file /home/user/quant_for_finance_python/data/user/fundamentals.csv. Columns
- `fundamental_dcf`: KeyError: 'this fundamental model needs the file /home/user/quant_for_finance_python/data/user/fundamentals.csv. Columns
- `fundamental_nonlinear`: KeyError: 'this fundamental model needs the file /home/user/quant_for_finance_python/data/user/fundamentals.csv. Columns
- `fundamental_quality`: KeyError: 'this fundamental model needs the file /home/user/quant_for_finance_python/data/user/fundamentals.csv. Columns
- `fundamental_value`: KeyError: 'this fundamental model needs the file /home/user/quant_for_finance_python/data/user/fundamentals.csv. Columns

Regenerate with `python -m experiments.library_survey --part3`; the next full run (`python -m experiments.library_survey`) folds these rows into the first survey.
