---
title: "The machine-learning research pipeline: datasets, experiments, search and forward tests"
slug: machine-learning-research-pipeline
difficulty: 3
chapter: Ch. 23
prerequisites: [walk-forward-and-leakage, information-coefficient, multiple-testing, machine-learning-for-returns]
stages: []
files: [src/ml/datasets.py, src/ml/training.py, src/ml/experiments.py, src/ml/registry.py, src/ml/strategy_search.py, src/ml/ensembles.py, src/ml/monitoring.py, src/webapp/ml_api.py, src/webapp/static/ml.js, config/ml_research.yaml]
figures: []
tests: [tests/test_ml_data.py, tests/test_ml_pipeline.py, tests/test_ml_search.py, tests/test_ml_cli.py, tests/test_webapp_assets_ml.py]
models: []
---

# The machine-learning research pipeline: datasets, experiments, search and forward tests

## In one sentence

A model is never trained on its own here: it is trained inside a pipeline that builds features without look-ahead, refits it walk-forward, turns its scores into positions, backtests them with costs, splits the result into train, validation, test and forward segments, and counts every look at the data so that luck can be priced in.

## The idea

A machine-learning backtest has three ways to lie. The *features* can see the future (a z-score computed over the whole sample, a label that overlaps the test window). The *evaluation* can reward the fit instead of the forecast (a Sharpe ratio measured on the days the model was trained on). And the *search* can pick a winner from many tries and report it as if it were the only one. The pipeline answers each in turn. Features at the close of day `t` use data up to `t`; the target is the return from the close of `t + lag` to the close of `t + lag + horizon`; training rows whose label window overlaps a test day are purged. Predictions are made only by models fitted before the day they predict. The result is cut from the end into **train** (the warm-up, in-sample), **validation** (where settings are chosen), **test** and **forward** (reserved, reported once). And every experiment and search trial on the same sample is **counted**, because the best of `N` tries has a higher Sharpe ratio than any one of them by luck alone.

## Why it matters

A gradient-boosted model on fifty features can reach a Sharpe ratio of 3 in sample on pure noise. The only evidence that counts is what it does where nothing was fitted, and the only honest way to report a search is with the number of trials beside the winner. The deflated Sharpe ratio, the probability of backtest overfitting and the generalisation score exist so that a number like 3 arrives with the question it should provoke.

## How this repo uses it

The pipeline lives in `src/ml/` and has three front doors that share it: the command line (`python -m src.cli ml-build-dataset | ml-train | ml-backtest | ml-report`), the dashboard's **Machine learning** tab and plain Python. Settings are in `config/ml_research.yaml`; experiments, datasets and models are saved under `artifacts/ml/` with SQLite (`registry.sqlite`) for the records.

1. **Dataset.** Ten groups of causal features (price, momentum, mean reversion, volatility, volume, relative, macro, futures, options, crypto), each tested for causality, built only where the data has what they need and skipped with a reason otherwise. Targets are forward return, excess return, rank, direction, top-versus-bottom, forward volatility, forward drawdown and risk-adjusted return, at horizons of 1 to 252 days. The dashboard lists each feature's coverage and its rank IC with the target on the warm-up segment only.
2. **Train.** Walk-forward refits (linear, ridge, lasso, elastic net, logistic, random forest, histogram gradient boosting, XGBoost and LightGBM if installed, autoregression, Kalman, online ridge and logistic, regime ridge), with a refit every few weeks by model kind.
3. **Positions and backtest.** Eight position rules (top-n long-short, long-only quantile, proportional, volatility-scaled, rank-weighted, threshold, regime-filtered, confidence-weighted), a rebalance schedule, commission and slippage, a volatility target and a leverage cap, through the platform's backtest engine.
4. **Evaluation.** Rank IC (mean, information ratio, t-statistic, by segment, by horizon, by refit), the return by score bucket, Sharpe by segment, the generalisation score (the worst out-of-sample Sharpe over the larger of the train and validation Sharpe), the probabilistic Sharpe ratio on test and forward, and the deflated Sharpe ratio that charges for every trial.
5. **Explain.** Permutation importance (the drop in out-of-sample rank IC when a feature is shuffled), the model's own importance, and Shapley values for one prediction on one date, an exact decomposition of the score.
6. **Stress.** Ten tests and a control: random start dates, universe subsets, higher costs, delayed execution, noisy features, parameter perturbation, block bootstrap, volatility regimes, crisis windows, bull and bear markets, and, as the control, what random scores earn through the same rules.
7. **Search.** Random trials, then Bayesian (TPE), then genetic, over model kind, settings, horizon, position rule and rebalancing. Trials see only the warm-up and validation segments; the best few are finalised on every segment once. Selection statistics (deflated Sharpe, probability of backtest overfitting, Romano-Wolf p-value) are reported with the winner.
8. **Combine.** Ten ways to weight several saved strategies, each estimated from the past only: equal weight, inverse volatility, risk parity, mean-variance, Bayes-Stein, regime-conditioned, online learning, bandit, best trailing and best in the current regime. Meta-labelling sizes each bet of a model by a second classifier's confidence without ever flipping a side.
9. **Forward test.** Signals are recorded as paper orders as the days arrive, settled when their window ends, and compared with the backtest: the live rank IC against the backtest's, and the population stability index of each feature and of the scores against their training distribution.

The tab has nine views for this: Dataset, Train, Experiments, Compare, Explain, Overfitting, Search, Combine and Forward test. Anything slow runs as a job and shows its progress. The ETF and crypto universes are platform data; the futures universe is simulated and every result says so.

## What we found

The tests pin the discipline, not an edge. Every feature passes the causality test (its value on or before a cutoff is identical when everything after the cutoff is replaced by noise). The purge drops the rows whose label window reaches a test day, and each refit fits only labels that were complete when it ran. A model fitted on data with a planted signal finds it (a rank-IC t-statistic above 3 and a positive test Sharpe ratio); on pure noise it does not (an insignificant IC and a generalisation score of at most 0.5). The trial counter rises with every experiment and search trial on the same sample, and a search never predicts the reserved segments before its finalists. A re-backtest reuses the saved predictions without refitting and counts as another look; meta-labelling keeps the side of the primary model's bet and only sizes it; the forward loop uses only the data of the day it runs for, and a signal recorded after its date is marked as replayed history, not a live forecast. These show that the machinery cannot flatter a result by construction. They do not show that any model has an edge on real prices, which is why the dashboard draws the in-sample bar in grey and puts the number of looks beside every winner.

## Going deeper

```
decision at close t            features X_t use data <= t
target (horizon h, lag L)      y_t = P[t + L + h] / P[t + L] - 1
purge                          drop training rows with t' + L + h >= first test day;   embargo adds e more days
segments (from the end)        forward 15%, test 15%, validation 20%, the rest train; the first prediction needs min_train_days
generalisation score           min(SR_val, SR_test, SR_fwd) / max(SR_train, SR_val, 0.25), clipped to [0, 1], 0 if any is <= 0
deflated Sharpe                P( SR_val > E[max of N trials' Sharpe] )                  N = looks at this sample
meta-label size                clip(2 p - 1, 0, 1) * sign(primary bet)
```
Rank IC is the cross-sectional Spearman correlation between the day's scores and the realised forward returns, averaged over days. An IC of 0.02 to 0.05 with a positive information ratio is a good signal; a high IC measured on the training segment is a symptom, not a result.

## Pitfalls

- Choosing features by their IC over the whole sample leaks the test segment into the choice; the dataset view measures it on the warm-up only for that reason.
- A good validation Sharpe after a search is partly selection. Judge a finalist by test and forward, and by how many trials it was picked from.
- The forward segment is reserved but it is not the future: once it has been looked at repeatedly it is another validation set. The forward test (live signals) is the only evidence that cannot be re-tuned.
- A re-backtest or a meta-labelled run counts as another look at the data and is recorded as one.
- Permutation importance and Shapley values describe the model, not the market: a feature that drives a score has not been shown to drive returns.
- Simulated universes (futures here) exercise the code; they say nothing about whether a model works on real prices.

## Try it

```bash
python -m src.cli ml-build-dataset --universe etf --groups price momentum volatility --target fwd_ret_5
python -m src.cli ml-train --model ridge --param alpha=10 --signal top_n_long_short --n 3 --rebalance weekly --robustness
python -m src.cli ml-report                       # the leaderboard
quant serve                                       # the Machine learning tab: Train, then Overfitting, then Forward test
python -m pytest tests/test_ml_pipeline.py tests/test_ml_search.py -q
```
