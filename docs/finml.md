# Financial machine learning: the research pipeline

This page covers the **Financial ML Lab** tab, the `quant finml` command and the `src/finml` package. They follow the methods of Marcos López de Prado's *Advances in Financial Machine Learning* (2018) as research baselines, and they are built to make a result **harder to fool yourself with**, not to add another model.

## What was already in the platform, and what was added

Before writing anything, each idea was checked against the code. Most already existed in some form, so they are reused, not repeated.

| Idea (chapter) | Already in the platform | Added here |
|---|---|---|
| Triple-barrier labels (3) | `src/ml/targets.py` `triple_barrier`: a binary target on every day of a panel, closes only, symmetric barriers | An **event-level engine** (`src/finml/events.py`): long and short sides, separate profit and stop multiples, a vertical barrier, an entry lag, costs, the **timestamp each label was decided**, and an explicit rule for a bar that touches both barriers |
| Event sampling (2) | none | **CUSUM filter** |
| Meta-labelling (3) | `src/ml/ensembles.py` `meta_label_scores`: a second model inside the machine-learning lab | A meta-model on **event labels** with a primary side, compared with the primary strategy, with random filters and with buy and hold |
| Sample uniqueness (4) | `src/ml/intervals.py`, `src/ml/training.py`: concurrency, average uniqueness and weights on the lab's panel labels | **Sequential bootstrap**, return-attribution weights, time decay and an effective-sample-size report for event labels (`src/finml/sampling.py`) |
| Purged cross-validation and CPCV (7, 12) | `src/ops/cv.py`, `src/validation/cpcv.py`: purging by one fixed horizon | Purging **by each label's own span** and an embargo, and CPCV on event labels with the path assembly of `ops.cv` (`src/finml/cv.py`) |
| Feature importance (8) | permutation importance and stable selection in `src/ml/explainability.py`, `src/ml/featsel.py` | **MDI, MDA by cluster of correlated features, and SFI**, with the share of folds in which a feature helped (`src/finml/importance.py`) |
| Information-driven bars (2) | none | Tick, volume, dollar, **imbalance and runs bars** (`src/finml/bars.py`), from trades |
| Fractional differentiation (5) | `src/ml/features.py` `fracdiff` | nothing: reused |
| Bet sizing (10) | none (the lab sizes by confidence in `meta_label_scores`) | Size from a probability, a sigmoid size, discretisation, **averaging of active bets** and turnover (`src/finml/sizing.py`) |
| Backtest overfitting (11–15) | PSR, DSR, PBO and the haircut Sharpe in `src/stats/inference.py`, `src/validation/multiple_testing.py`, `src/backtest/metrics.py` | nothing: reused by the pipeline's evidence |
| Hierarchical risk parity (16) | `src/portfolio/hierarchical.py`: HRP, HERC, stability | nothing: reused |
| Structural breaks and entropy (17–18) | none | **SADF**, a **CUSUM test on levels** (Chu–Stinchcombe–White), Shannon, **Lempel–Ziv** and **Kontoyiannis** entropy (`src/finml/structural.py`) |
| Microstructure features (19) | `src/microstructure/order_flow.py`: Roll, Corwin–Schultz, Amihud, Kyle, VPIN, order-flow imbalance | nothing: reused |
| Research factory (1, 20) | the research workspace, jobs, experiments and the verification registry | the pipeline below, one reproducible run |

## The pipeline

```
CUSUM events -> triple-barrier labels (primary side) -> uniqueness weights -> purged CV
   -> meta-model -> probability-based sizing -> cost-aware backtest -> evidence
```

`run_pipeline` (`src/finml/pipeline.py`) runs it on one ticker's daily prices. What it does to avoid a flattering answer:

* **An untouched holdout.** The last `holdout` share of the history (by date, so nothing inside it can move the line) is never used to fit, tune or choose. A development label that reaches into it is dropped. It is scored once, last. A test changes only the prices inside the holdout and checks that no development number moves.
* **Purging by each label's span, an embargo, and a check.** Each fold is verified: no training label overlaps a test label. The check is part of the report.
* **Features and fitted choices stay inside the training rows.** Missing values are filled with the training median, the logistic model scales inside its pipeline, and the sequential bootstrap draws only from the training labels.
* **A comparison that can come out negative.** The filtered strategy is set against the primary strategy it filters (same signals, same costs), against **40 random filters that skip as many signals**, and against buy and hold. The report says the filter "helped" only when it beat the primary on the development folds *and* the holdout, with a block-bootstrap probability of at least 0.9 on the holdout, *and* beat 95% of the random filters. Otherwise it says so in words, and on a random walk that is what you get.
* **Costs on every change of position**, and the cost of a round trip inside each label, so a label is "profitable" only after costs.
* **The Sharpe ratio is deflated** for the number of configurations you say you tried (`n_trials`, default 1, which is rarely true; the page says what was assumed).
* **Position = the average of the bets open that day**, with skipped signals counting as zero, so the primary and filtered strategies are scaled the same way.
* **CPCV gives a distribution of out-of-sample paths**, not one path. It is a research evaluation on the development events and is labelled as **not a prospective test**: only forward, chronological testing supplies that.

The page lists the checks that are not done as "Not completed" (a prospective test, independent replication) and the ones that need a decision as "Review" (entry at the signal's close, closes only, one configuration assumed).

## Things to know

* **Triple barriers.** An event at row *i* is entered at the close of row *i* + `entry_lag`; barriers are watched from the next row. With closes only, a barrier is touched when a close is beyond it. With the daily high and low (the *Use each day's high and low* option, adjusted for splits and dividends with the price series' own ratio) a barrier is touched when the range reaches it, and a bar that reaches **both** cannot be ordered from OHLC data: the `same_bar` setting decides (`stop` by default, the conservative choice) and the report counts how many labels that touched. Intraday data would settle it.
* **Labels need a whole window.** An event is labelled only if its whole vertical window exists, whether or not a barrier was touched early, so which events are labelled never depends on what happened.
* **Sample uniqueness.** `sample_quality` reports mean concurrency, average uniqueness and the effective sample size `(Σu)² / Σu²`. Thousands of overlapping events can be worth a few hundred independent ones.
* **Bars need trades.** `src/finml/bars.py` takes a frame of individual trades with `price` and `volume`. A day's open, high, low, close and volume cannot be cut into them, so no tab offers them and nothing should be described as evidence about a daily strategy. Bars are not independent observations. The imbalance-bar rule has one stabilisation that is not in the book (`floor`): without it, balanced flow drives the threshold to zero and every bar closes at the minimum size.
* **SADF critical values** are simulated under a random walk of the same length, so they are a guide and vary with the seed. A reading above it says the price was explosive over some window ending that day, not that it will keep going.
* **Feature importance.** MDI explains the training sample and is biased toward features with many values; MDA shuffles a feature, or a cluster of correlated ones, out of sample; SFI scores each alone. A feature that helps in some folds and hurts in others is not reliable.
* **The older scikit-learn workarounds in the book were not copied.** Weights go through `sample_weight` of the current estimators, and the forest uses `class_weight="balanced_subsample"`.
* **Not done:** a saved Financial ML result in a research session (it is not one of the saveable kinds yet), multi-asset pipelines, intraday data, fractional differentiation inside the pipeline's features, and bars in the interface.

## Using it

```bash
quant finml SPY --primary momentum --holdout 0.2 --entry-lag 1 --cost-bps 5 --n-trials 1
```

```python
from src.finml.pipeline import run_pipeline
report = run_pipeline(close, {"primary": "momentum", "cusum_k": 1.0, "holdout": 0.2})
```

The pieces can be used alone: `events.cusum_events` and `events.triple_barrier_events`, `sampling.sample_quality` and `sampling.sequential_bootstrap`, `cv.purged_folds` and `cv.cpcv_event_splits`, `importance.mdi`, `mda` and `sfi`, `sizing.size_from_probability` and `average_active`, `bars.dollar_bars`, `imbalance_bars` and `runs_bars`, and `structural.sadf`, `chu_stinchcombe_white` and the entropy estimators. The tests (`tests/test_finml.py`) check each against a case worked by hand or a property it must have: a bar that touches both barriers, the uniqueness of the book's own example, that purging leaves no overlap, that an informative feature ranks first and noise does not, and that nothing inside the holdout can change a development number.
