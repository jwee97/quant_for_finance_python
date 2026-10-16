---
title: "Multiple testing, false discovery and the deflated Sharpe ratio"
slug: multiple-testing
difficulty: 3
chapter: Ch. 22
prerequisites: [performance-metrics]
stages: [3, 11, 26]
files: [src/validation/multiple_testing.py, src/validation/forecast_tests.py, src/validation/robustness.py, src/ops/hpo.py, src/webapp/tuning.py]
figures: [52, 53]
tests: [tests/test_multiple_testing.py, tests/test_robustness.py, tests/test_tuning.py, tests/test_hpo_search.py]
models: []
---

# Multiple testing, false discovery and the deflated Sharpe ratio

## In one sentence

If you try enough ideas, the best one will look good even if none works; the remedy is to count the ideas and demand more of the winner.

## The idea

Test 72 signals at the 5% level and about 3.6 will 'work' by chance. Benjamini-Hochberg controls the expected share of false discoveries among those declared. The deflated Sharpe ratio asks
whether the best Sharpe among N trials exceeds what N random strategies would show, given the sample length, skew and kurtosis. The probabilistic Sharpe ratio is the single-trial version.

## Why it matters

This is the single biggest source of fake alpha in published backtests, and the easiest to commit by accident: every parameter tweak is a trial.

## How this repo uses it

Declared decision families use Benjamini-Hochberg throughout. The experiment manager counts distinct specifications in a group and feeds that count to the deflated Sharpe ratio in every
tear sheet. Stage 26 runs White's Reality Check, Hansen's SPA and probability of backtest overfitting on a grid of 1,584 rules.

The dashboard's Tune tab and the machine-learning tuner apply the same ideas to a search over a strategy's settings (`ops.hpo.Study.selection_report`): every try is a trial; the deflated Sharpe
ratio is asked to beat the best of the *effective* number of independent tries (`ops.hpo.effective_trials`), because settings that sit next to each other trade almost the same days; and the
Romano-Wolf p-value of the winner is reported, which compares it with the best that a bootstrap of the same days would show if no try had any edge, whatever the tries have in common. The verdict
asks more of a gain than that it held on the untouched days: the paired t statistic of the gain (`webapp.tuning.paired_t`) and that p-value must both pass. The thresholds come from a simulation on
strategies with a known truth, `experiments/tune_selection_study.py`: the hold-out gain alone said "good" one time in five for a strategy with no edge.

## What we found

In Stage 3, 15 of 72 momentum tests cleared the naive 5% threshold against 3.6 expected by chance, and 10 survived Benjamini-Hochberg control (EXP-005). Across the 1,584-rule grid, neither
the Reality Check nor SPA found a rule with positive expected net return once the search was accounted for (EXP-071).

## Going deeper

```
Benjamini-Hochberg: sort p_(1) <= ... <= p_(m); find the largest k with p_(k) <= (k/m) * q; reject hypotheses 1..k
Deflated Sharpe:    SR0 = sqrt(V[SR_trials]) * ( (1-g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)) )     g = Euler-Mascheroni
                    DSR = Phi( (SR - SR0) sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )
```
`SR0` is the Sharpe you would expect from the best of N random strategies. If the observed Sharpe is below it, the probability that the strategy has skill is below one half, however good the Sharpe looks in isolation.

```
Effective number of tries:   N_eff = rho + (1 - rho) N      rho = the average correlation between two tries' returns
Romano-Wolf (stepdown):      bootstrap the days, recentre every try at zero, compare the best of them with the observed best; then drop it and repeat on the rest
Paired t of a gain:          t = mean(r_tuned - r_start) / ( std(r_tuned - r_start) / sqrt(days) )
```
With `rho` near 1 the tries are one try repeated, and `N_eff` is near 1. The spread of the tries' own Sharpe ratios is a poor stand-in for the noise level when the settings really do differ (it counts the real difference as noise), so the version here uses the sampling variance of one Sharpe ratio.

## Pitfalls

- Do not forget the trials you did not report.
- Deflation assumes trials are exchangeable; correlated trials need fewer 'effective' trials.
- A search that learns from its own scores finds higher noise peaks than a fixed list of the same size would, so every test that treats its tries as a fixed list under-corrects a little: the Romano-Wolf p-value of the winner of a three-setting search was under 5% in about one case in five when no setting had any edge. That is why the verdict also asks for a gain on days no try saw.
- A multiple-testing correction cannot rescue a biased design.

## Try it

```bash
quant backtest --model momentum --group my_ideas
quant backtest --model donchian --group my_ideas       # the group's trial count rises to 2
```
