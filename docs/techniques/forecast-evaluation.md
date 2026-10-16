---
title: "Scoring forecasts honestly: CRPS, calibration and skill tests"
slug: forecast-evaluation
difficulty: 3
chapter: Platform
prerequisites: [probabilistic-forecasting, multiple-testing, walk-forward-and-leakage]
stages: []
files: [src/forecast/scoring.py, src/forecast/evaluate.py, src/forecast/verdict.py, src/validation/forecast_tests.py, experiments/forecast_study.py]
figures: []
tests: [tests/test_forecast_scoring.py, tests/test_forecast_evaluate.py, tests/test_forecast_verdict.py]
models: []
---

# Scoring forecasts honestly: CRPS, calibration and skill tests

## In one sentence

A forecast of a whole distribution is judged by scoring rules that an honest forecaster cannot game (the CRPS and the pinball loss), by whether its ranges held the outcomes as often as they claimed (coverage and the probability integral transform), and by whether it beat a plain yardstick by more than luck allows once you count how many methods were tried and how much the forecasts overlap.

## The idea

**Walk forward.** Forecast from many past dates, giving each method only the prices up to the date, and score each forecast against what then happened. This is the only way to find out how a method would have done, because a method fitted to all the data has seen the answer.

**A proper score.** The *continuous ranked probability score* of a forecast with draws `X` and an outcome `y` is `E|X - y| - 0.5 E|X - X'|`: the average distance of the draws from the outcome, less half their average distance from each other. It is in the units of the outcome, it equals the absolute error when the forecast is one number, and it is *proper*: the forecaster's best answer is the one it believes. The plain formula with `n` draws favours a method run with more draws, so the *fair* form (Ferro 2014) divides the spread term by `n(n-1)`. The *pinball loss* scores one quantile `q` at level `tau` as `tau (y - q)` above it and `(1 - tau)(q - y)` below, and twice its average over all levels is the CRPS.

**Calibration.** Of the central 90% ranges, about 90% should hold the outcome; the outcome should fall below the 5% quantile one time in twenty (the Kupiec test of value at risk); and the *probability integral transform* (the share of the draws below the outcome) should be uniform on 0 to 1, a hump in the middle meaning ranges too wide, a U too narrow, a slope a bias. A forecast can be calibrated and useless (the unconditional distribution is) and sharp and wrong; the CRPS rewards being sharp *subject to* being calibrated.

**Skill.** `skill = 1 - CRPS_method / CRPS_baseline`: 0 is no better than the baseline, +5% is a loss 5% smaller. The baseline here is the random walk with the spread of the last year.

**Is the gain luck?** A one-sided Diebold-Mariano test of the two methods' losses origin by origin, with a Newey-West variance because forecasts made ten days apart for a 21-day horizon share most of their outcome (the lag is the number of origins one horizon spans, less one). **Many methods:** the Holm step-down adjustment of the p-values controls the chance of *any* false win among all the methods; the verdict also asks for a gain of at least 1%. **The average and the direction** are different questions from the spread: the out-of-sample R-squared against the average of the last five years (Campbell and Thompson 2008; Welch and Goyal 2008) with the Clark-West test (a model that estimates a mean the market lacks is mechanically worse than the benchmark that does not, so the usual test is biased against it), and the hit rate of up-or-down calls against *always saying up*, which is right as often as the market rose.

## Why it matters

A forecaster that is scored by the squared error of its mean ignores half of what it said, and one that is scored on the data it was fitted to always looks good. Compared with no yardstick, any method looks clever. Compared with a yardstick, any method that wins once looks real, and with eighteen methods on one history one of them wins by luck. The scores and tests here exist so that the first reading of a result is the cautious one.

## How this repo uses it

`src/forecast/scoring.py` has the scores and tests (`crps`, `pinball`, `interval_hits`, `pit`, `skill`, `better_than`, `holm_adjust`, `mean_skill`, `direction_skill`), built on `src/validation/forecast_tests.py` (`diebold_mariano`, `clark_west`). `src/forecast/evaluate.py` runs the walk-forward evaluation (the origins, the refit schedule, the controls and the equal-weight blend) and `src/forecast/verdict.py` says what the numbers mean in words, built from the numbers of the evaluation and nothing else. The Forecast tab and `quant forecast` show them; `experiments/forecast_study.py` reproduces the real-data tables. The older [guide to probabilistic forecasting](probabilistic-forecasting.md) covers the same ideas for the Gaussian forecasts of the plugin framework.

## What we found

The scores match what they should: the CRPS of draws from a Gaussian matches the closed form and is the absolute error for a single value; the fair form has no small-sample bias where the plain one has; twice the average pinball loss over all levels is the CRPS; the PIT is uniform for a calibrated forecast, humped for one too wide and U-shaped for one too narrow; the Holm adjustment matches the reference implementation. The evaluation finds skill where it was planted (volatility models beat the random walk by more than 2% at 1 and 5 days on GJR-GARCH returns, with adjusted p below 5%) and none where there was none (independent returns: every skill within 6% of zero, nothing called better, the yardstick calibrated), and a pool of forecasts scores no worse than the average of its members, as the convexity of the CRPS says.

The adjustment matters on real prices. On SPY at 21 days 11 of 18 methods had a skill above zero with an unadjusted p-value below 5%, and 5 after the Holm correction; on QQQ 3 had, and none after the correction; on EEM 2 and none. The yardstick itself is calibrated in coverage and not in shape: its central 90% range held the outcome 88% to 93% of the time on the five ETFs and its 5% quantile was breached 1.5% to 5.5% of the time, yet the test of the whole PIT histogram rejected it on SPY, QQQ and GLD (p of 0.003, 0.007 and 0.005), because a constant spread is too wide in calm periods and too narrow in storms. A bigger average gain is not always a more reliable one: at 63 days Hull-White was 14.9% better than the random walk on SPY and did not clear the correction (p = 0.18), while GJR-GARCH at 7.7% did (p = 0.001), which is what a gain from carrying a positive average into a rising market looks like: large when the window rose, negative when it fell, so large on average and less reliable. The tables are in [forecasting a stock's price](../forecasting.md#what-it-found-on-real-prices).

## Pitfalls

- A test on overlapping forecasts that ignores the overlap is far too confident. 200 forecasts of 21-day returns made ten days apart are worth about 67 independent ones.
- A yardstick with a zero average beats nothing in a falling market and loses to every method that carries a positive drift in a rising one; compare the average with the average of the last five years, as the page does, before calling a gain a forecast of the average.
- Skill against the random walk says a method is better than a constant spread, not that it is good. A method can be 5% better and still be uncalibrated.
- The p-value is for the claim "better than the yardstick", not for the claim "better than the others": the ranking among the best few is within what luck could change.
- A p-value after a search is only as honest as the count of what was tried: add a method to the list and the correction gets stricter.
- A score on one stock over one history is one history. The tables on the page are evidence about the method on that history and not a law.

## Try it

```bash
python -m experiments.forecast_study --tickers SPY QQQ                    # the skill tables of the forecasting page (about three minutes)
quant forecast SPY --all --effort quick                                    # the verdict and the scoreboard in the terminal
```

```python
import numpy as np
from src.forecast.scoring import crps, holm_adjust, pit, skill

rng = np.random.default_rng(0)
y = rng.normal(0, 0.04, 400)                                               # what happened: 400 returns with a 4% spread
wide = rng.normal(0, 0.06, (400, 1000))                                    # a forecast that is too wide
right = rng.normal(0, 0.04, (400, 1000))
print(round(skill(crps(right, y), crps(wide, y)), 3), np.histogram(pit(wide, y), bins=5, range=(0, 1))[0])   # 0.053, and a humped PIT: [43 111 110 93 43]
print(holm_adjust([0.004, 0.02, 0.2, 0.6]).round(3))                                  # [0.016 0.06 0.4 0.6]: the second no longer clears 5%
```
