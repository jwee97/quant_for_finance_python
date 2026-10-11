# The machine-learning research platform

`src/ml/` trains models **only inside a research pipeline**: a leak-free dataset, walk-forward fits, positions, a backtest through the platform's own engine, results split into train, validation, test and forward segments, and a count of every look at the same data. There is no way to train a model in isolation: `experiments.run_experiment` is the single door, and the command line, the dashboard and Python all go through it. [The technique guide](techniques/machine-learning-research-pipeline.md) explains the ideas; [the dashboard guide](dashboard.md#machine-learning) walks through the tab. This page is the map of the code.

## Modules

| module | what it does |
|---|---|
| `config.py` | defaults, `config/ml_research.yaml` merged over them, validation, `config_hash`, `dataset_key` (the sample, to count looks) and `build_key` (the dataset section, to reuse a built panel) |
| `features.py` | the feature library: ten groups of named, documented, causal functions, each tested by replacing everything after a cutoff with noise |
| `targets.py` | `fwd_ret_h`, `excess_ret_h`, `rank_h`, `direction_h`, `topbottom_h`, `fwd_vol_h`, `fwd_dd_h`, `riskadj_h`: the label runs from the close of `t + lag` to `t + lag + h`, after the features end |
| `datasets.py` | a market bundle in, a `Dataset` indexed by `(date, asset)` out; universes `etf`, `crypto`, `futures` (simulated); skipped features are listed with their reason |
| `splitters.py` | walk-forward and purged k-fold splits on the date axis, purge (`lag + horizon`) and embargo, and the segment plan (train, validation, test, forward) |
| `models.py` | the model zoo behind one `Model` interface (`fit`, `predict`, `importance`, `to_bytes`); optional packages are reported as unavailable with a reason |
| `training.py` | walk-forward training: each refit uses only rows whose labels were complete, predicts the next block, and the blocks laid end to end are the forecasts |
| `signals.py` | eight ways to turn scores into target weights, with gross, net, position and sector caps |
| `backtest.py` | hands the weights to `backtest.engine.BacktestEngine` (lag, rebalance schedule, drift, costs, volatility target) and cuts the returns into segments by the day the profit was realised |
| `evaluation.py` | rank IC and its decay, return by score bucket, Sharpe by segment, the generalisation score, the probabilistic and deflated Sharpe ratios |
| `explainability.py` | model importance across refits, permutation importance, per-feature IC, Monte Carlo Shapley values for one prediction |
| `robustness.py` | ten stress tests and a random-signal control |
| `experiments.py` | `run_experiment`, `rebacktest` (saved predictions under new position rules, no refit), `load_run` |
| `registry.py` | SQLite records and file artifacts, trial counts, model stages |
| `optimization.py`, `strategy_search.py` | validation-only tuning and the staged random, Bayesian and genetic search, with the selection statistics |
| `ensembles.py` | the strategy combination engine, bandit allocation, selectors, comparison and meta-labelling |
| `monitoring.py` | the forward tracker: paper signals, settlement, live rank IC and population-stability drift |
| `visualization.py` | JSON payloads for the dashboard and the self-contained HTML report |
| `cli.py` | `ml-build-dataset`, `ml-train`, `ml-backtest`, `ml-report` (registered into `python -m src.cli`) |

## Where things are saved

```
artifacts/ml/registry.sqlite            experiments, trials, metrics, artifacts, models, feature_sets, predictions, forward_signals
artifacts/ml/experiments/<id>/          config.yaml, metrics.json, predictions.parquet, positions.parquet, returns.parquet, model.pkl, report.json, report.html
artifacts/ml/features/<key>/            a built dataset, reused when the dataset section is unchanged
```
The directory is git-ignored. Deleting an experiment from the dashboard (or `Registry.delete_experiment`) removes its rows and its folder.

## Settings

`config/ml_research.yaml` holds the defaults for the dataset, split, model, signal, backtest, robustness, search and forward sections. Anything can be overridden per experiment by the command line (`--param alpha=10`, `--cost-bps 10`), by a YAML file (`ml-train --config my.yaml`) or by the dashboard form; the merged configuration is validated and saved beside the result, and its hash identifies the experiment. The file is read by the ML package only: it is not part of the platform's fingerprinted configuration namespaces.

## If LightGBM or XGBoost shows as "not available"

The two boosting libraries are optional and need the OpenMP runtime of the operating system (`libgomp.so.1` on Linux). Where the library is missing, importing them fails with `OSError: libgomp.so.1: cannot open shared object file`. The Machine learning tab does not fail with it: the model list shows LightGBM and XGBoost as not available, with the reason, and every other model (linear, ridge, lasso, elastic net, logistic, random forest, scikit-learn's histogram gradient boosting, autoregression, Kalman, online and regime models) works as before. Scikit-learn's histogram gradient boosting is the same family of model as LightGBM and needs nothing extra.

To switch the two on, install the runtime and restart the app: on Debian, Ubuntu or a Docker image, `apt-get install -y libgomp1`; on a Mac, `brew install libomp`. On Replit, which provides system libraries through Nix, add a package that provides `libgomp` (the GCC runtime library) in the Dependencies tool and press Stop, then Run; if that does not work, leave the two models off, since nothing else depends on them.

## The HTTP API

Served by `src/webapp/ml_api.py`; JSON in, JSON out. Anything slow returns `{"job": id}` and is polled through `/api/job/<id>`.

| route | what |
|---|---|
| `GET /api/ml/catalog` | models and their settings, feature groups, targets, position rules, combination methods, defaults, limits |
| `POST /api/ml/dataset/build` (job) | build and describe a dataset |
| `POST /api/ml/train` (job) | the whole experiment; returns the id and the report |
| `POST /api/ml/rebacktest` (job) | saved predictions under other signal, backtest or robustness settings |
| `POST /api/ml/meta-label` (job) | size an experiment's bets by a second classifier |
| `POST /api/ml/search`, `/api/ml/tune` (jobs) | strategy search; tuning one model |
| `GET /api/ml/experiments`, `/api/ml/experiment`, `/api/ml/models`, `/api/ml/trials`, `/api/ml/feature-sets` | the registry |
| `GET /api/ml/compare`, `/api/ml/explain`, `POST /api/ml/combine` | comparison, one prediction explained, strategy combination |
| `GET /api/ml/forward`, `POST /api/ml/forward/step` (job) | forward-test signals and the daily step |
| `POST /api/ml/promote`, `/api/ml/delete` | model stages; delete an experiment (needs `confirm: true`) |

The browser may send only the settings of `SPEC` in `ml_api.py`, each checked for type and range; the merged configuration is then validated by `experiment_config`. A request cannot name a file path.

## Extending it

- **A model:** add a class with the `Model` interface to `models.py` and an entry to its catalog (`kind`, `family`, `tasks`, default `params`); it appears in the Train form, the search space needs a line in `strategy_search.kind_space`.
- **A feature:** write a function of `Inputs` returning a date by asset frame, register it in a group of `features.py` with a description; the causality test in `tests/test_ml_data.py` picks it up and fails if it looks ahead.
- **A position rule:** add it to `signals.build_weights` and `METHODS`; give it a line in `webapp/ml_api.SIGNAL_TEXT` and `static/ml.js`.
- **A universe:** teach `datasets.resolve_bundle` to return a `MarketBundle` for it.

## What it does not do

It does not send an order, and the forward tracker's signals are paper. It does not make a model good: on data without an edge the validation, test and forward Sharpe ratios fall towards zero and the page says so. A search that picks the best of many trials reports the number of trials beside the winner and deflates its Sharpe ratio, and the reserved segments are seen only once a finalist is run on them: they are still not the future, and the forward test is the only evidence that cannot be re-tuned.
