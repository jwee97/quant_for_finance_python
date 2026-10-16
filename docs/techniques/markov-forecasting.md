---
title: "Markov chains and hidden Markov models for forecasting"
slug: markov-forecasting
difficulty: 3
chapter: Platform
prerequisites: [bayesian-inference-and-markov-models, regime-detection]
stages: []
files: [src/forecast/markov.py, src/probability/markov.py, src/models/regimes.py, src/forecast/features.py]
figures: []
tests: [tests/test_forecast_methods.py, tests/test_probability.py]
models: []
---

# Markov chains and hidden Markov models for forecasting

## In one sentence

A Markov model says the market is in one of a few states, that the state changes with fixed probabilities that depend only on the state it is in now, and that each state has its own kind of day; the states worth modelling are the ones that last, and volatility lasts for weeks while the direction of a day does not.

## The idea

A **Markov chain** is a table of probabilities `P[i, j]` of moving from state `i` to state `j` in one day, each row adding to one. From it come the share of time the market spends in each state in the long run (the stationary distribution, `pi P = pi`), the expected number of days a state lasts (`1 / (1 - P[i, i])`) and the distribution of the state `k` days ahead (`P` to the power `k`). To forecast a return, give every state a bag of days that were in it and draw the day's return from the bag of the state the path is in.

Two chains are used. The **chain on the day's move** has five states, the quintiles of the day's return from a big fall to a big rise, with `P` counted from the last five years (with one pseudo-count in every cell so that no probability is zero). The **chain on volatility** has three states, the terciles of the root mean square of the last 21 days (calm, ordinary, stormy), and the bag of a state holds the returns of the day *after* a day in that state.

A **hidden Markov model** (Hamilton 1989; Rabiner 1989) does not observe the state. Each of two regimes has its own average return and volatility, the regime follows a Markov chain, and Baum-Welch (expectation-maximisation) fits all of it to the last five years of returns. The forecast starts from the *filtered* probability of each regime today, computed with the forward algorithm from today's and earlier days only, then moves the regime forward with `P` and draws each day from the normal of its regime.

## Why it matters

Regimes decide how risk should be sized, and persistence is the property that makes a regime usable: a state that lasts three weeks can be recognised and acted on, one that lasts a day cannot. The difference between filtered and smoothed probabilities is the difference between a tradable signal and a look-ahead bug: a smoothed probability uses days after the one it labels, and is flattering for that reason alone. A chain also gives something no regression does, a plain statement of how long a storm tends to last.

## How this repo uses it

`src/forecast/markov.py` has the three methods. They reuse the platform's own Markov code: `estimate_transition_matrix`, `MarkovChain` (stationary distribution, expected duration) and `test_markov_order` (a test of whether more than one day of memory is needed) from `src/probability/markov.py`, and `fit_hmm`, `forward_filter` and `log_emission` from `src/models/regimes.py`. The hidden Markov model is a *heavy* method: its fit takes seconds, so the walk-forward test refits it half as often as the others (every 126 trading days at the Standard setting, and not every 63). The forecast tab shows, for each method, the transition matrix, the expected days in each state, the state the market is in today and the volatility of every state under *What the methods learned*.

## What we found

On markets with a known truth: on a calm-and-stormy market the chain on volatility finds states that persist (calm and stormy each stay put with probability above 0.8), orders their volatilities correctly, and puts today in the right state after thirty stormy or calm days, with a five-day spread at least 1.5 times larger in the stormy one; the hidden Markov model recovers each regime's volatility to within 30%, the persistence of each regime and the regime the market is in on the last day (with probability above 0.9); the chain on the day's move finds no memory in independent days and clear memory in clustered ones (the order test's p-value is above 0.01 in the first and below 0.001 in the second).

On real prices the chains and the model are decent descriptions and not better forecasts than a GARCH model. On SPY at the last forecast origin the chain on volatility had calm, ordinary and stormy states of 10%, 14% and 24% a year that last 13, 8 and 20 days on average; the two regimes of the hidden Markov model were a calm one (+27% a year, 11% volatility, 99 days) and a turbulent one (−15%, 25%, 59 days), with a probability of 98% of being in the calm one. Memory in the sign and size of the day's move is real (the order test's p-value on SPY is 0.0003) and is almost all volatility clustering. At 21 days the chain on volatility was 6.1% better than the random walk on SPY (Holm-adjusted p = 0.08, short of the bar), between −2.1% and +3.3% on the others; the hidden Markov model was 4.9% on SPY (p = 0.41) and between −1.9% and +2.0% on the others. Against its control, GJR-GARCH, the hidden Markov model scored −1.7%, −0.7%, −3.6%, −1.5% and +2.4% on the five ETFs (p between 0.14 and 1.0): it did not add to what the volatility model already says. None forecast the average return better than the average of the last five years, and none called the direction better than always saying up.

## Pitfalls

- A regime is recognised after it has been under way for a while. The filtered probability of the turbulent regime rises *after* the first big falls, so a Markov forecast is late by construction.
- The labels are arbitrary: regime 0 is not "bull" because it was listed first. Check the fitted means and volatilities before reading them.
- Two states and three states are choices. More states fit the past better and the future worse; every extra state needs its own bag of days.
- The average return of a regime is poorly estimated (a few years of days in a state, with a large noise), and the forecast carries it forward. Read the regime's volatility and its duration with more confidence than its mean.
- A transition matrix counted from five years is a sample: one pseudo-count keeps it positive, but a row built from a few dozen transitions is a rough number.
- Smoothed probabilities in a chart look like a clean signal and are not available on the day.

## Try it

```bash
quant forecast SPY --methods markov_returns markov_volatility hmm garch --effort quick
```

```python
import numpy as np
from src.forecast import History, create
from src.framework import load_default_bundle
from src.utils.config import load_config

history = History.from_frame(load_default_bundle(load_config(), with_macro=False).prices["SPY"])
hmm = create("hmm")
state = hmm.fit(history.tail(hmm.window), (21,))
info = hmm.predict(state, history.tail(hmm.lookback), (21,), 1000, np.random.default_rng(0)).info
print([round(v, 2) for v in info["regime_annual_volatility"]], [round(v) for v in info["expected_days_in_regime"]], [round(v, 2) for v in info["probability_of_each_regime_today"]])
```
