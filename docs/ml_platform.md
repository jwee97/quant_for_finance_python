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
| `GET /api/ml/catalog` | models and their settings, feature groups, targets, position rules, combination methods, defaults, limits, and what tuning each model varies (`tune_spaces`, `tune_efforts`) |
| `POST /api/ml/dataset/build` (job) | build and describe a dataset |
| `POST /api/ml/train` (job) | the whole experiment; returns the id and the report |
| `POST /api/ml/rebacktest` (job) | saved predictions under other signal, backtest or robustness settings |
| `POST /api/ml/meta-label` (job) | size an experiment's bets by a second classifier |
| `POST /api/ml/search`, `/api/ml/tune` (jobs), `POST /api/ml/tune/stop` | strategy search; tuning one model (with `effort`: from the form's own settings, with a verdict); end a tuning early |
| `GET /api/ml/experiments`, `/api/ml/experiment`, `/api/ml/models`, `/api/ml/trials`, `/api/ml/feature-sets` | the registry |
| `GET /api/ml/compare`, `/api/ml/explain`, `POST /api/ml/combine` | comparison, one prediction explained, strategy combination |
| `GET /api/ml/forward`, `POST /api/ml/forward/step` (job) | forward-test signals and the daily step |
| `POST /api/ml/promote`, `/api/ml/delete` | model stages; delete an experiment (needs `confirm: true`) |
| `GET /api/jobs`, `GET /api/job/<id>` | every job the app has run, newest first (without results), and one job with its result: a page that was reloaded, or an app that was restarted, finds a run again (see below) |

The browser may send only the settings of `SPEC` in `ml_api.py`, each checked for type and range; the merged configuration is then validated by `experiment_config`. A request cannot name a file path.

## Tuning a model for beginners

*Search*, then *Tune one model*, finds good settings for one model on the tab's dataset without choosing any numbers of trials. Pick the model, and the form lists what will be tried: the model's own settings (a ridge's `alpha`, the Kalman regression's `q` and number of features, a tree's depth and learning rate) and the ones that decide how its predictions become trades (how many days ahead it predicts, how scores become positions, how many assets on each side, how long a position is held, how often positions change), each with its range and a line on what it does. Then choose how long to look: **Quick** (8 tries), **Standard** (16) or **Thorough** (30). A try is a whole walk-forward experiment on the validation segment, so it takes seconds to a minute, and a Thorough search of a slow model can take half an hour; it runs as a background job, and **Stop** on its row in the runs card ends it after the try it is on.

What it does, in order:

1. **The first try is the settings in the Train form** (the model's own defaults where a box is blank), so the answer comes with a comparison. The search space is stretched to hold them.
2. **A Tree-structured Parzen Estimator** proposes the rest, after an evenly spread start of a third of the tries (a scrambled Sobol sequence: with 8 to 30 tries in a space of several settings, the gaps and clumps of random draws are most of what the search has to learn from), never running a combination twice. Every try is scored on the validation segment only, with the penalties of `search.objective_weights` for turnover, drawdown, instability and the gap between train and validation; every try is recorded and counted against the dataset's looks at the same data. The test and forward segments are not predicted during the search.
3. **The winner and the starting settings are each run once on every segment**, and saved as experiments called *`<model>` tuned* and *`<model>` starting settings* (one run only when the starting settings won), with the number of tries counted against them.
4. **A verdict in words**, by the same rules as the Tune tab ([Tuning a strategy's settings](tuning.md)): the Sharpe ratio on the validation segment, where the search chose, and on the test and forward segments (weighted by days), where it did not, for the starting settings and the winner. *The improvement held up*, *only part of it held up*, *better, but not by enough to rule out luck*, *no real difference*, *probably overfitted* or *keep what you have*, with the chance that the winner's Sharpe ratio is more than luck after so many tries. A *held up* asks for more than a gain on the test and forward days: a paired t statistic of the winner's daily returns against the starting settings' over those days of at least 1, and a Romano-Wolf p-value of the winner under 30% over every try's validation returns ([why these numbers](tuning.md#how-the-verdict-was-calibrated)).
5. **Use these settings in the Train form** copies the winner's settings (the model's own, the horizon, the position rule, the holding period and the rebalance) into the Train view; the Tune tab link on the card is for tuning a library strategy such as the Kalman trend filter on the Backtest tab's tickers.

The plain search of `method` and `n_trials` (random, TPE or a Gaussian process with `bayes`; no comparison with the form's settings) is still what `POST /api/ml/tune` does when no `effort` is sent. The efforts use TPE: a Gaussian process is the better search for a few numeric settings, and this search mixes a model's own settings with five choices about how predictions become positions, which is where TPE holds up. `src/ml/optimization.py::tune` takes `start` (the first trial, which counts as one of the `n_trials`) and `stop`; `current_settings` reads a configuration's value for each searched setting and `include` stretches a space to hold them.

A real example, one run, for scale and nothing more: a ridge regression on the 15 ETFs from the Train form's defaults (alpha 30, a five-day target, long the top three and short the bottom three, held five days), *Standard* (16 tries, seed 0). The winner (alpha 0.135, a 20-day target, long only the top three, held 20 days) took the Sharpe ratio from −1.05 to 0.42 on the validation segment and from −0.39 to 0.56 on the test and forward segments, and the verdict was **mixed**: *better on the test and forward segments, but not by enough to rule out luck*. A winner this good among 16 tries would turn up by luck about 45% of the time (Romano-Wolf; under 30% is asked for), about 4 of the 16 tries were independent (the deflated Sharpe ratio is 42%), and holding every ticker in equal amounts scored 0.94 on the same segments against 0.56 for the tuned settings. The search found settings that fit better, the starting settings lose money on all three later segments (−1.05, −0.75 and −0.04), and the page says it has not found an edge: the right answer, because the default setup has none on these ETFs.

## Runs in the background

A training, a search, a re-backtest or a forward step is a *job*: it runs on the app's one worker thread, one at a time (models set process-wide flags and share the data caches), and the page polls `/api/job/<id>`. The thread belongs to the app, not to the page, so closing the tab or reloading the page does not stop it, and `run_experiment` saves the experiment to the registry when it ends, whether or not anything is watching (`tests/test_jobs.py` trains a model and never asks after the job).

What used to be lost was the knowledge of the job: the table of jobs lived in memory, so a reloaded page could not find its run and a restarted app forgot it. `src/webapp/jobs.py` keeps one small record per job under `data/user/jobs/` (label, kind, status, the last progress message at most every two seconds, the times it started and ended, and the process that ran it) and the result of a finished one. `GET /api/jobs` lists them; the Machine learning tab shows a card with what is running, waiting or ended in the last six hours, follows it, and opens the experiment a finished run made. A job whose app is gone can never finish: it is marked `interrupted`, with the last thing it said it was doing, when the app starts again and whenever the list is read. A process id alone cannot say whether the app is still there (the numbers are reused, and an app in a container is process 1 every time), so a running app writes a sign of life on each job's record every ten seconds, and another process's job that has shown none for 45 seconds counts as gone: after a crash the sign flips to *Stopped* within about a minute, and a page that is waiting for the run is told, instead of waiting for ever. Each row has a **Clear** button, and **Clear all ended runs** takes every finished, failed or stopped run off the list (an experiment stays in the registry); on a row that still says running, **Not running? Clear** removes a crashed run at once. A run that really is going in this app cannot be cleared, because it would come back and a run cannot be stopped from the page: stop the app (on Replit: Stop, then Run) to end it. **Nothing is resumed**: a training is not checkpointed, so an interrupted run starts again from the beginning. The log keeps the last 60 jobs.

Whether the run outlives *you* depends on whether the app does. Your own computer or a server: yes, as long as the process lives. Replit: the app lives as long as the workspace does, which Replit does not promise after you close the tab; the section *Training in the background on Replit* of `replit.md`, in the repository's top folder, says what Replit documents, what keeping a run alive costs there, and the command-line route (`python -m src ml-train ...`) for a run you start from the Shell or from a Scheduled deployment.

## Extending it

- **A model:** add a class with the `Model` interface to `models.py` and an entry to its catalog (`kind`, `family`, `tasks`, default `params`); it appears in the Train form, the search space needs a line in `strategy_search.kind_space`.
- **A feature:** write a function of `Inputs` returning a date by asset frame, register it in a group of `features.py` with a description; the causality test in `tests/test_ml_data.py` picks it up and fails if it looks ahead.
- **A position rule:** add it to `signals.build_weights` and `METHODS`; give it a line in `webapp/ml_api.SIGNAL_TEXT` and `static/ml.js`.
- **A universe:** teach `datasets.resolve_bundle` to return a `MarketBundle` for it.

## What it does not do

It does not send an order, and the forward tracker's signals are paper. It does not make a model good: on data without an edge the validation, test and forward Sharpe ratios fall towards zero and the page says so. A search that picks the best of many trials reports the number of trials beside the winner and deflates its Sharpe ratio, and the reserved segments are seen only once a finalist is run on them: they are still not the future, and the forward test is the only evidence that cannot be re-tuned.
