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
| `catalog_extra.py`, `boosting.py`, `regimes.py`, `graph.py`, `ensemble_models.py` | the 21 kinds added to the zoo ([below](#the-model-families)): catalogue entries, default settings, form specs and search spaces; rankers and probabilistic boosters; regime models; graph networks; the member ensemble. Each is built lazily, so importing the platform loads none of their packages |
| `deep/` | the reusable PyTorch infrastructure: windows without look-ahead (`data`), nine networks (`nets`), one training loop (`trainer`), the `Model` wrappers with serialization (`model`) |
| `rl/` | `PortfolioEnv`, a Gymnasium environment with the backtest engine's timing and costs (`env`), and the PPO, SAC and recurrent-PPO agents (`agent`) |
| `model_report.py` | the `model_extras` section of a report: training curves, ranking quality, prediction intervals, regime probabilities, ensemble weights, the asset graph, an agent's rewards, and the cost and provenance of the run |
| `training.py` | walk-forward training: each refit uses only rows whose labels were complete, predicts the next block, and the blocks laid end to end are the forecasts |
| `signals.py` | eleven ways to turn scores into target weights, with gross, net, position and sector caps (three of them read a model's uncertainty or an agent's own weights) |
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
                                        + extra_<name>.parquet (what a model reported beside its scores) and diagnostics.json (the final fit's facts), where it has them
artifacts/ml/features/<key>/            a built dataset, reused when the dataset section is unchanged
```
The directory is git-ignored. Deleting an experiment from the dashboard (or `Registry.delete_experiment`) removes its rows and its folder.

## Settings

`config/ml_research.yaml` holds the defaults for the dataset, split, model, signal, backtest, robustness, search and forward sections. Anything can be overridden per experiment by the command line (`--param alpha=10`, `--cost-bps 10`), by a YAML file (`ml-train --config my.yaml`) or by the dashboard form; the merged configuration is validated and saved beside the result, and its hash identifies the experiment. The file is read by the ML package only: it is not part of the platform's fingerprinted configuration namespaces.

## The model families

Twenty-one kinds sit beside the base zoo. All of them are selectable in the Train form with their own settings (a menu, a switch or text where a number is the wrong control), tunable in the Search and Tune forms, comparable on the leaderboard, and trained only through `run_experiment`, so every one is walk-forward, purged, costed and counted like a ridge regression.

| phase | kinds | package | what the report adds |
|---|---|---|---|
| 1 ranking | `lgbm_ranker` (LambdaMART), `xgb_ranker` (`rank:ndcg`, `rank:pairwise`, `rank:map`) | LightGBM, XGBoost | **Ranking quality**: NDCG@k of the daily ordering against a random one, by segment |
| 1 distributions | `lgbm_quantile` (low, median and high quantile regressions), `ngboost` (a Normal distribution by natural-gradient boosting) | LightGBM, NGBoost | **Prediction intervals** and their coverage, and **whether the spread predicts the error** |
| 1 regimes | `hmm_regime` (Gaussian HMM), `markov_switching` | hmmlearn, statsmodels | **Regime probabilities** through time and the time spent in each regime |
| 2 sequences | `lstm`, `gru`, `tcn` | PyTorch | **Training curves** (loss by epoch) and rank IC by forecast horizon (`horizons`) |
| 3 forecasting, representation | `patchtst`, `nbeats`, `nhits`, `tft` (quantile outputs and attention), `autoencoder`, `vae` | PyTorch | the same, plus attention over the window, intervals (`tft`), and the **rank IC of each latent factor** |
| 4 graph, ensemble | `gcn`, `gat`, `model_ensemble` | PyTorch | **The asset graph** and a portfolio-level forecast; **ensemble weights** through time and a mixture distribution |
| 5 reinforcement learning | `ppo`, `sac`, `recurrent_ppo` | Stable-Baselines3, sb3-contrib | **Agent reward while training** and the agent on its own training days |

Every report also carries **Compute and provenance**: training time, device, parameter count, peak memory, the configuration hash, the data fingerprint, the git commit, the versions of the libraries the run imported, and the size and SHA-256 of the saved model. Each chart has a table behind it (the Chart / Table switch on the card).

### What a model can do beyond scoring a row

The `Model` interface gained three optional hooks, so the pipeline did not need a second path for the new families:

- `bind(ds)` gives a model read access to the dataset's history. A model that needs it (`needs_dataset`: the sequence networks, the graph and regime models, the ensemble, the agents) is bound by the trainer before every fit, and by `fit_final` and `predict_latest`; a model restored from the registry says so if it is asked to predict unbound. It may read **only dates up to the date it predicts**.
- `predict_extras(X)` returns named frames beside the scores: `distribution` (mean, standard deviation, lower, upper), `regimes`, `heads` (one score per forecast horizon), `weights`, `latent` and `portfolio`. The trainer lays them end to end like the predictions; they are saved as `extra_<name>.parquet` and restored by `load_run`, so a re-backtest reports them again.
- `diagnostics()` returns the facts of one fit (loss curves, device, parameters, memory, the graph, rewards).

The catalogue says, per kind, which `outputs` it produces, which `needs` packages it imports, and the `specs` the form draws its controls from. A kind whose package is missing is listed as unavailable with the reason and the `pip` command, the request to train it is refused before any job is queued, and every other kind works as before (`tests/test_ml_integration.py` simulates the absence of each optional package).

### No look-ahead, family by family

- **Rankers, quantile models, NGBoost.** A prediction is a function of the row alone. A ranker's scores are standardised with the mean and spread of its *training* scores, not of the day's scores, so a row's score does not depend on which other rows are scored with it.
- **Regime models.** The probabilities used on a day are *filtered*: P(regime on day t | market data to day t), a forward recursion with the parameters fitted on the training window. They are never smoothed (the forward-backward `predict_proba` lets tomorrow change today's probability, which is right for describing history and wrong for trading). The market state is built from the dataset's returns, because the asset features are standardised across assets and so carry no market-wide level.
- **Networks.** A row's window holds the asset's features on that day and the days before it. The scaler and the target statistics come from the training rows. Early stopping uses the *last* `val_fraction` of the training dates, separated from the rest by `lag + horizon` days (the walk-forward's own purge). Extra forecast horizons must be shorter than the dataset's, so their labels are complete whenever the primary label is.
- **Graph networks.** The edges (the most correlated assets over the *training* window, and assets of one sector or class) are fixed per fit; a layer reads neighbours' features **on the same day**, all known by that close.
- **Ensembles.** Members are fitted on the first part of the training window and scored on the held-out tail; each is calibrated to the target's units there. A dynamic weight on a day uses only outcomes realised by that day (`position + lag + horizon <= day`), so the days still in flight never vote.
- **Agents.** An agent trains in the environment on the training window only and then plays the next block one day at a time from a flat book, with its own previous book in the observation. A fold boundary therefore costs one round of turnover.
- **Tests.** `tests/test_ml_phase1.py`, `test_ml_deep.py`, `test_ml_phase4.py` and `test_ml_rl.py` replace everything after a cutoff with noise and assert that nothing at or before it changes.

### Position rules that read a model's uncertainty

| rule | what it does | needs |
|---|---|---|
| `uncertainty_weighted` | weight proportional to the (demeaned) score, moved towards zero by `kappa` times the asset's *excess* uncertainty over the day's median, then divided by `std ** uncertainty_power`: conviction per unit of the model's own uncertainty | a model with a `distribution` (quantile LightGBM, NGBoost, `tft`, `model_ensemble`) |
| `uncertainty_mean_variance` | the same views through the inverse of a Ledoit-Wolf covariance of the trailing `mv_lookback` days of returns (up to the day), re-solved every rebalance and held between; dollar neutral, scaled to `gross` | the same |
| `target_weights` | the scores are read as weights and traded as sized; the agents' default | an agent (any model works) |

`kappa` defaults to 0 because a model's own spread is usually many times its scores: shrinking by the whole standard deviation removes every view. When the rule is `top_n_long_short` and the model is an agent, the report warns that the rule re-ranks the agent's weights.

### Reinforcement learning

`PortfolioEnv` is a Gymnasium environment on the dataset's panel. The observation is the day's scaled features, the agent's current weights and its drawdown. The action is a vector per asset: long-short books are centred (dollar neutral) and scaled to `gross`; `long_only` uses a softmax with a cash slot. Both are capped at `max_weight` and, with `max_turnover`, the trade is shortened to that one-way turnover (`project_weights` is the one place these constraints live, shared by training and the out-of-sample rollout). A book decided on day *t* is held from *t + lag* and earns the return of day *t + lag + 1*, like the backtest engine.

Costs are one-way turnover times `cost_bps` (the form fills in the backtest's commission plus slippage) and an optional quadratic `impact` term. Rewards, in percent per day: `return`; `risk_adjusted` (the net return less `risk_aversion / 2` times its squared deviation from a running mean, less `drawdown_penalty` and `turnover_penalty` terms); or `dsr`, the differential Sharpe ratio of Moody and Saffell. The algorithms are Stable-Baselines3's PPO and SAC and sb3-contrib's recurrent PPO.

### Searching and tuning

The search spaces of the new kinds are in the catalogue (`tune_spaces`). The Search and Tune forms have an **engine** menu: the platform's own random, TPE and Bayesian searches, or **Optuna** (a multivariate TPE sampler, fed the trials already run); both count every trial against the dataset's looks and score on the validation segment only.

### Optional packages

`requirements.txt` already has PyTorch, LightGBM, XGBoost, hmmlearn and statsmodels. The rest is optional:

| install | adds |
|---|---|
| `pip install -r requirements-ml-extras.txt` (or `pip install ".[ml-extras]"`) | everything below |
| `pip install ".[ml-probabilistic]"` | NGBoost |
| `pip install ".[ml-tuning]"` | the Optuna engine |
| `pip install ".[ml-rl]"` | Gymnasium, Stable-Baselines3, sb3-contrib (and PyTorch) |

The CI `ml-base` job runs the tab on the base install alone; the others install the extras and run every test.

### Limits worth knowing

- The agents are trained for the number of steps in the form (a few thousand by default). With so little experience they usually learn little; the reward curve says whether they did, and the equity curve (out of sample) is the evidence, not the in-sample panel.
- The networks, the graph models and the agents are compact implementations written for this platform (PatchTST, N-HiTS and the Temporal Fusion Transformer follow the papers' main ideas; they are not ports of the reference code), and a graph is dense, static per fit, and meant for a few dozen assets.
- The regime models fit the *market's* regimes (from the equal-weight return, its volatility and the cross-sectional dispersion) and then one ridge per regime; a Markov-switching fit can fail to converge on a short window, and that refit is then skipped, with the reason, like any other fit that raises an error (the report lists the skipped fits).
- A re-backtest restores the intervals, regime probabilities and weights (saved with the predictions) but not the training curves, which belong to the fits.
- All of this runs on CPU unless a GPU is visible; `device: auto` picks CUDA when there is one and says which it used.

## If LightGBM or XGBoost shows as "not available"

The two boosting libraries are optional and need the OpenMP runtime of the operating system (`libgomp.so.1` on Linux). Where the library is missing, importing them fails with `OSError: libgomp.so.1: cannot open shared object file`. The Machine learning tab does not fail with it: the model list shows LightGBM and XGBoost as not available, with the reason, and every other model (linear, ridge, lasso, elastic net, logistic, random forest, scikit-learn's histogram gradient boosting, autoregression, Kalman, online and regime models) works as before. Scikit-learn's histogram gradient boosting is the same family of model as LightGBM and needs nothing extra.

To switch the two on, install the runtime and restart the app: on Debian, Ubuntu or a Docker image, `apt-get install -y libgomp1`; on a Mac, `brew install libomp`. On Replit, which provides system libraries through Nix, add a package that provides `libgomp` (the GCC runtime library) in the Dependencies tool and press Stop, then Run; if that does not work, leave the two models off, since nothing else depends on them.

## The HTTP API

Served by `src/webapp/ml_api.py`; JSON in, JSON out. Anything slow returns `{"job": id}` and is polled through `/api/job/<id>`.

| route | what |
|---|---|
| `GET /api/ml/catalog` | models and their settings (`family`, `specs`, `needs`, `outputs`, `available` with the reason), feature groups, targets, position rules, combination methods, search `engines`, defaults, limits, and what tuning each model varies (`tune_spaces`, `tune_efforts`) |
| `POST /api/ml/dataset/build` (job) | build and describe a dataset |
| `POST /api/ml/train` (job) | the whole experiment; returns the id and the report |
| `POST /api/ml/rebacktest` (job) | saved predictions under other signal, backtest or robustness settings |
| `POST /api/ml/meta-label` (job) | size an experiment's bets by a second classifier |
| `POST /api/ml/search`, `/api/ml/tune` (jobs), `POST /api/ml/tune/stop` | strategy search; tuning one model (with `effort`: from the form's own settings, with a verdict; with `engine: optuna`: Optuna's sampler); end a tuning early |
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
- **A model from a package that is not in `requirements.txt`:** put its entry in `catalog_extra.py` with `needs=("package",)` and a module in `EXTRA_MODULE` whose `build(kind, task, params, seed)` imports the package lazily; the catalogue lists it as unavailable, with the reason, where the package is missing. Give it `specs` for any setting that is not a number, and `outputs` for what its report shows.
- **A feature:** write a function of `Inputs` returning a date by asset frame, register it in a group of `features.py` with a description; the causality test in `tests/test_ml_data.py` picks it up and fails if it looks ahead.
- **A position rule:** add it to `signals.build_weights` and `METHODS`; give it a line in `webapp/ml_api.SIGNAL_TEXT` and `static/ml.js`.
- **A universe:** teach `datasets.resolve_bundle` to return a `MarketBundle` for it.

## What it does not do

It does not send an order, and the forward tracker's signals are paper. It does not make a model good: on data without an edge the validation, test and forward Sharpe ratios fall towards zero and the page says so. A search that picks the best of many trials reports the number of trials beside the winner and deflates its Sharpe ratio, and the reserved segments are seen only once a finalist is run on them: they are still not the future, and the forward test is the only evidence that cannot be re-tuned.
