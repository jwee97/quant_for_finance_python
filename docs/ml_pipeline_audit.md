# Shared ML pipeline audit: why models lose to the equal-weight benchmark

This is the Phase 1 audit of the pipeline every model shares:

```
market data -> features -> labels -> walk-forward training -> predictions (scores)
            -> signal rule (scores -> target weights) -> backtest engine (lag, schedule, drift, costs)
            -> net returns -> comparison with the equal-weight benchmark
```

Every claim below was checked by running the code, and each has a test that fails if the claim stops being true (`tests/test_ml_pipeline_reference.py`, `tests/test_ml_decisions.py`). Nothing here
tunes a model: the point is to separate what is wrong with the *plumbing* from what is true of the *models*.

## Method

1. **Reference worlds.** Synthetic returns with a known answer: a perfect-foresight score as a diagnostic control (it must make a fortune at the engine's own delay and nothing at any other delay),
   constant weights with a known cost, a world where every asset drifts up with a common factor. A defect in the plumbing shows up here, because no model is involved.
2. **Cross-model diagnostics.** The same dataset (15 ETFs, 2006-2026), the same walk-forward splits, the same predictions per model, re-run through different position rules *without refitting*
   (`evaluate_run` on a trained result). Differences between rows are therefore caused by the position rule and nothing else.
3. **Measured on the scored period only.** The first three years have no model (the training window); exposure and beta are measured from the first prediction.

## What was verified correct

| Stage | Check | Result |
| --- | --- | --- |
| Timing | A position decided at the close of `t` with lag 1 earns the return of day `t + 2` (trade at `t + 1`, earn from then). Perfect foresight at that delay: Sharpe ~49. At delays 0, 1, 3: about zero. | Correct |
| Labels | `fwd_ret_h` is `P[t+lag+h] / P[t+lag] - 1`: the same window the position earns. The last `lag + h` rows have no label. | Correct, matches the engine |
| Cash | A book with 50% in one asset earns 50% of that asset's return; the rest is cash earning zero. | Correct |
| Costs | Linear on traded notional; buying 100% at 10 bp costs 10 bp, once; a constant target trades nothing afterwards. | Correct, **except the entry cost (below)** |
| Compounding | The equity curve is the product of `1 + net`. | Correct |
| Benchmark | The equal-weight benchmark is the daily cross-sectional mean of the investable assets' returns (an asset that is not yet investable does not enter). | Correct |
| Leakage | Causal views, point-in-time checks, null controls and the holdout guard already exist and were not touched. | Unchanged |

## Defects found and fixed

### 1. The cost of the first trade was never charged (shared by every strategy on the platform)

The first trade of a book is made on a day with no book held the day before, so its gross return is `NaN` and `net = gross - cost` was `NaN`: the cost of building the first position vanished from
every net series. It is small for a long history (one one-way cost, about 7 bp) and larger the shorter the sample, but it made every backtest slightly cheaper than it should be. The engine now treats
that day as earning nothing and paying the cost (`src/backtest/engine.py`). The cost-sensitivity grid, the summary and the ledger all see the same day.

### 2. Exposure was averaged over the warm-up (a measurement error, not a pipeline error)

An earlier look reported "average gross exposure 0.45", and the obvious reading was that the strategy sat in cash half the time. That number averaged the weights over the whole timeline including the
three-year training window where no model has predicted anything. Over the scored period the gross exposure of the default rule is 0.95-0.99 (net 0), and a long-only rule is fully invested (1.00).
**The platform is not systematically under-invested.** The diagnostics now measure from the first signal.

## Root causes of the shared under-performance (verified)

Cross-model table (ridge and lasso, 2016-2026 scored period, the same predictions in every row; benchmark: equal-weight, Sharpe 0.88, CAGR 8.8%):

| Model | Position rule | Net Sharpe | CAGR | Beta to benchmark | Gross / net exposure | Turnover (x/yr) | Costs (bp/yr) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| ridge | default: top-3 long / bottom-3 short, 10% vol target, weekly | -0.58 | -6.1% | 0.41 | 0.95 / 0.00 | 55 | 386 |
| ridge | the same without the vol target | -0.58 | -5.0% | 0.35 | 0.95 / 0.00 | 41 | 285 |
| ridge | long-only top 30%, weekly | +0.28 | 3.1% | 1.35 | 1.00 / 1.00 | 33 | 231 |
| ridge | long-only top 30%, monthly | +0.58 | 7.8% | 1.32 | 1.00 / 1.00 | 11 | 78 |
| lasso | default | -0.37 | -4.3% | 0.20 | 0.99 / 0.00 | 23 | 158 |
| lasso | the same without the vol target | -0.27 | -3.0% | 0.23 | 0.99 / 0.00 | 17 | 116 |
| lasso | long-only top 30%, weekly | +0.56 | 8.4% | 1.46 | 1.00 / 1.00 | 12 | 84 |
| lasso | long-only top 30%, monthly | +0.58 | 9.0% | 1.51 | 1.00 / 1.00 | 7 | 50 |

Rank IC of the two models: 0.017 (ridge) and 0.011 (lasso).

1. **The default rule and the default benchmark are different bets.** The default position rule is dollar-neutral (net exposure 0, beta 0.2-0.4 to the benchmark) and the benchmark is the equal-weight
   long-only universe (beta 1). In a market that rose over the period, a book that holds no market exposure cannot match a book that does, whatever its skill. A reference test
   (`test_a_dollar_neutral_book_has_zero_beta...`) shows this on a world where the predictions are perfect. **The comparison the platform shows by default is not like-for-like.** Beta-matched,
   net-exposure-matched and risk-matched benchmarks already exist (`src/ml/benchmarks.py`); the strategy builder below compares against them by default.
2. **Costs and turnover are the largest controllable drag.** Weekly rebalancing of a three-name book turns the portfolio over 33-55 times a year, costing 230-390 bp a year, which is more than the
   benchmark's whole excess return over cash. The same predictions rebalanced monthly cost 50-78 bp and earn 0.3-0.5 more Sharpe. Nothing in the model changed.
3. **The predictions are weak.** A rank IC of 0.01-0.02 over 5-day horizons is about what a cross-section of 15 correlated ETFs can support. It is a real, small signal that does not survive
   realistic costs at a high turnover, and it does not beat holding the market at 0.88 Sharpe in a rising period. This is a property of the models and the universe, not a defect.
4. **Concentration raises beta, not return.** The long-only top-30% book has a beta of 1.3-1.5 (it holds the higher-beta ETFs), so its risk-adjusted result is below the benchmark's even when
   its return is similar.
5. **The vol target adds turnover** (55 vs 41 x/yr for ridge) without improving the result when the signal is weak.

What is *not* a cause: look-ahead or label misalignment (verified correct), cash accounting, compounding, the benchmark calculation, under-investment (measurement artifact, above).

## What changes as a result

* **Explicit decisions.** The signal rule `decision_policy` (`src/ml/decisions.py`) turns scores into BUY / SELL / SHORT / COVER / INCREASE / REDUCE / HOLD and the forced exits STOP_LOSS /
  TAKE_PROFIT / TIME_EXIT, with hysteresis (a name is not sold for a one-place change of rank), a cooldown after a forced exit, a minimum rebalance interval and a resize band. Every decision is logged
  with the information that was available. Selling a long is a different action from opening a short.
* **An event-driven engine mode** (`backtest.rebalance = on_change`): the book trades only on days the policy changed a target and drifts with prices otherwise, so a stop can fire on any day and
  the engine does not charge for putting a drifted book back to target every day.
* **A ledger that reconciles.** `src/ml/trade_log.py` builds the executed trades, the round trips and the analytics from the engine's own book and trades. The sum of the round trips' net
  contributions equals the engine's net return and the trips' costs equal the engine's costs (tested to 1e-10).
* **Like-for-like comparison** (next): a long-only decision policy is compared with the equal-weight benchmark, a dollar-neutral one with a beta-matched benchmark, and both with a random-signal
  control at the same turnover.

## Limits

* Two linear models and one universe (15 ETFs). The models' rank IC is too small to say anything about the nonlinear models; the same table for those is produced by the comparison workspace
  rather than asserted here. Random forest and boosted trees were too slow in walk-forward to include in this pass.
* The entry-cost fix changes every backtest by about one one-way cost. No stored experiment is rewritten; a re-run reproduces the new number.
* A perfect-foresight signal is used only as a diagnostic control in tests; it is never offered as a strategy.
