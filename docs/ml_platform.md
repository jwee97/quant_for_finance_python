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
| `param_schema.py` | the settings of each model as a schema (type, range or choices, default): what the web API validates a request's `model.params` against, and how what a request sends is laid over the model's defaults ([below](#a-models-settings)) |
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

### A model's settings

Every model has a schema for its settings (`src/ml/param_schema.py`): each setting is a whole number, a number, a switch, a menu, text or a list of feature names, with the range or the choices the model accepts and the default the catalogue gives it. `GET /api/ml/catalog` carries it as `schema` beside `params` and `specs`, and the Train form is built from it, so **every setting of the chosen model is on the form**, however many there are (twenty-three for PatchTST).

A request's `model.params` is checked against the schema; the settings are not counted.

- **The form sends only what was changed.** A setting left at its default is not sent, and one put back to its default stops being sent. The server lays what it receives over the model's defaults, so the experiment's saved configuration states every setting the model ran with, and the same settings are one configuration hash whether a client sent all of them, some or none (a browser's `10` and a script's `10.0` are stored as `10`; a switch is stored as 1 or 0). An agent's `cost_bps` that is not sent is the Backtest section's costs.
- **Each value is checked for what it is.** A whole number must be whole (`2.5` layers is refused, not truncated). A number must be a finite JSON number inside its range (`dropout` is at least 0 and below 1, `learning_rate` is above 0); text, `true`, `NaN` and a number too large to use are refused. A switch is `true`, `false`, 1 or 0. A menu takes one of its listed values. Text has no control characters and at most 400 characters; `horizons`, `pool_sizes` and an ensemble's `members` are read by their grammar, and each member's own settings are checked against that member's schema. A null or a blank is "not set": the default stands.
- **Settings that must agree are checked together**, as the networks themselves require when they are built: `d_model` a multiple of `n_heads` and `lookback` at least `patch_len` for PatchTST, `hidden` a multiple of `n_heads` for the TFT and of `heads` for the GAT.
- **Everything else is refused before a job is queued**, with a message that names the setting and what is allowed (`model.params.dropout must be a number, at least 0 and below 1 (got 1.5)`): an unknown setting, a setting of another model, a `model` or `params` that is not an object, and a model kind that is unknown or not available here.

The ranges are those the model's constructor or library enforces (a lower bound that mirrors a check is a fact about the model), the natural bounds of fractions and probabilities, and, for the counts that set the cost of a fit (trees, epochs, widths, timesteps), wide sanity limits that bind no real use. The tests build every available model with each setting at the edges of its range, and show that where a constructor checks a limit itself the schema refuses what the constructor refuses and accepts what it accepts, so the two cannot drift apart unnoticed. The two settings every model accepts although they are not among its own, `columns` (the features it may use) and `halflife_days` (the half-life of the weights on training rows), are in the schema too.

Nothing re-validates a saved configuration. An experiment made earlier, with all of its settings, some of them or a script's own, is read as it is, and a model built from a partial configuration merges the defaults itself, as it always did.

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
- **A restricted view of history.** Every family that reads the dataset's history (regime models, graph networks, networks, agents, ensembles) is bound to a `CausalView`: the dataset physically truncated at a cutoff, with the realised labels removed (a model that asks for `Y` is refused). While it is fitted, the cutoff is the first decision day of the block it will predict; while it predicts the block, it is the block's last day, so a model that looks one day ahead inside the block still sees nothing past it.
- **Tests.** `tests/test_ml_phase1.py`, `test_ml_deep.py`, `test_ml_phase4.py` and `test_ml_rl.py` replace everything after a cutoff with noise and assert that nothing at or before it changes. `tests/test_ml_causal_views.py` and `tests/test_ml_training_isolation.py` do it for the whole walk-forward (ridge, HMM, LSTM, GCN, ensemble, PPO) with a deliberately look-ahead model and a label-reading model as controls.

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

### Choosing the winner: constrained selection and the simplest adequate model

A search leaves many candidates and a leaderboard rewards the lucky ones. `quant ml-select`, `POST /api/ml/select`, the **Select** view and the `selection` option of the search and tuning forms choose among them in four declared steps, and store the result (append-only, with the policy hash) so it can be audited later (`GET /api/ml/selection`).

1. **Evidence on validation data only.** Each candidate is reduced to its **pooled chronological validation net returns**: the validation days of all its walk-forward folds laid end to end in date order. The Sharpe ratio, drawdown and the other figures are computed on that one series. They are never the average of per-fold Sharpe ratios, which weights a 20-day fold like a 200-day one and lets one noisy fold dominate; the report prints the naive average beside the pooled figure so the difference is visible, and two folds that disagree about the same day are refused. A candidate that carries test or forward dates is refused, the test and forward rows of a finalised experiment are counted and never read, and the finalists a search then freezes into an evaluation plan are the selected trial first and the next-ranked eligible ones.
2. **Eligibility before ranking.** A candidate must pass the declared requirements for data, risk, exposure, liquidity and statistical evidence. Each gate is reported with its threshold, the observed value and a status: `pass`, `fail`, `unavailable` (the evidence was not produced; for an enforced gate that is not a pass) or `not_enforced`. A candidate that could not be run is `failed`, one with too little data is `insufficient`, one that broke a gate is `ineligible`. If nothing is eligible nothing is selected, rather than the least bad.
3. **Ranking, the nested estimate of the rule, and Pareto.** Eligible candidates are ranked by the primary metric (`primary_metric`, default the validation net Sharpe; also `net_sortino`, `net_calmar`, `net_ann_return`, `ic`), ties broken by drawdown, turnover and complexity. The Pareto analysis over the primary metric and the secondary criteria (`secondary`: `max_drawdown`, `turnover`, `instability` across validation sub-periods, `complexity`) names the front, who dominates whom, and how the selected candidate compares with each other one. An objective that some candidate lacks is dropped and listed, not guessed. Beside the ranking, the **nested estimate** (`nested`) runs the ranking *rule* through the nested chronological procedure: at each of several cutoffs (`cutoff_fractions`) it chooses by the net Sharpe of the `inner_days` days known at that cutoff (at least `min_inner`), carries the choice unchanged over the days up to the next cutoff, and chains those days. The chained Sharpe is what choosing by this rule earned on days after each choice; the best candidate in hindsight, and the finally selected one, were chosen with those days in view, so the distance between them is the optimism of selecting among these candidates. Every step records its cutoff, the dates of its inner and outer windows and their hashes, and an audit recomputes from the dates that no inner day was after its cutoff and no window overlaps another.
4. **The simplest candidate that is not meaningfully worse.** The complexity vector holds the family's rank, the number of features, the optional packages the model needs, the cost of training it, how long it takes to predict and how stable its validation performance was; an unmeasured dimension is counted as not simple, never as free. There is **no fixed Sharpe penalty**. "Not meaningfully worse" than the highest-ranked candidate comes from the **paired** difference on the days both have, with a dependence-aware joint stationary bootstrap, in one of two rules that are kept apart: `one_se` (the default) is a **selection heuristic** (the gap is at most `k` standard errors of the paired gap) and never claims equivalence; `non_inferiority` is a **formal one-sided test** against a `margin` the policy declares before the results, Holm-adjusted over the comparisons, which also reports where the candidate is equivalent within the margin. The margin is an assumption, the highest-ranked candidate is itself the best of N on these data (so every gap to it is overstated), and the claim is about validation data only.

Every threshold is a field of the policy; `GET /api/ml/selection-policy` returns the defaults with what each does, and unknown fields are refused.

| field | default | what it does |
|---|---|---|
| `primary_metric` | `net_sharpe` | the ranking metric |
| `secondary` | `max_drawdown`, `turnover`, `instability`, `complexity` | tie-breakers and Pareto criteria |
| `pareto` | the primary metric and `secondary` | the objectives of the Pareto analysis |
| `blocks`, `min_block_days` | 4, 20 | the sub-periods the stability (spread of the block Sharpe ratios) is measured over |
| `nested`, `enabled` | on | the nested estimate of the ranking rule (net Sharpe ranking only) |
| `cutoff_fractions`, `inner_days`, `min_inner` | 0.40, 0.55, 0.70, 0.85; 126; 60 | where the cutoffs fall (shares of the pooled validation days), the days the rule looks at, and the fewest it needs |
| `eligibility` | (the rows below) | the requirements |
| `simplicity` | (the rows below) | the simplicity step |
| `min_validation_days` | 126 | data: pooled validation days (fewer is `insufficient`) |
| `min_coverage` | 0.80 | data: the share of validation days with at least two assets scored |
| `min_assets` | 1 | data: assets scored |
| `max_drawdown` | 0.40 | risk: largest peak-to-trough fall of the validation equity |
| `max_ann_vol` | off | risk: annualised volatility |
| `max_gross_exposure`, `max_abs_net_exposure` | 2.0, 1.0 | exposure: the largest sum of absolute weights, and the largest absolute sum of weights, on any validation day |
| `max_ann_turnover` | off | liquidity: annualised one-way turnover |
| `min_capacity_share`, `capacity_capital` | off | liquidity: the share of the book one session's participation limit lets through at a capital level; measured by the realistic execution mode with volume, otherwise unavailable and never assumed |
| `min_effective_obs` | 60 | statistical: `n(1 - r)/(1 + r)` for the lag-1 autocorrelation `r` of the returns |
| `min_net_sharpe` | 0.0 | statistical: the validation net Sharpe must be above this |
| `min_sharpe_lower_bound`, `sharpe_level` | off, 0.95 | statistical: the lower end of the HAC interval of the Sharpe ratio |
| `max_label_overlap` | 0 | statistical: folds whose training labels reached past the date their fit may inform, asserted from the recorded fold lineage |
| `missing_evidence` | `ineligible` | what an enforced gate does with evidence that was not produced: `ineligible`, or `skip` (still reported as unavailable) |
| `enabled` | true | switch the simplicity step off to keep the highest-ranked candidate |
| `rule` | `one_se` | `one_se` (heuristic) or `non_inferiority` (formal) |
| `k` | 1.0 | `one_se`: standard errors of the paired gap that still count as not distinguishable |
| `margin`, `level` | none, 0.05 | `non_inferiority`: the largest loss accepted (in the metric's units, required) and the one-sided Holm-adjusted level |
| `priority` | `model_complexity`, `n_features`, `dependencies`, `training_cost`, `inference_latency`, `stability` | the order in which the complexity dimensions decide |
| `cost_tolerance` | 0.25 | a relative difference under which two costs, latencies or instabilities are a tie |
| `n_boot`, `seed` | 1000, 0 | the bootstrap of the paired gap |
| `require_paired` | false | refuse an unpaired standard error for the gap |

`quant ml-select EXP [EXP ...] [--policy policy.json] [--until DATE] [--json]` prints the same result. A search or tuning asked for `selection` returns it next to the old answer (`best`, the highest composite score, is unchanged); the proposals of the search still follow `objective_weights`, the selection decides which trial is chosen. A selection can be declared before the results by putting its policy in a frozen plan's `decision_policy.selection`; the result then says whether it was predeclared or only supplied with the data.

### Optional packages

`requirements.txt` already has PyTorch, LightGBM, XGBoost, hmmlearn and statsmodels. The rest is optional:

| install | adds |
|---|---|
| `pip install -r requirements-ml-extras.txt` (or `pip install ".[ml-extras]"`) | everything below |
| `pip install -r requirements-ml-no-ngboost.txt` (or `pip install ".[ml-rl,ml-tuning]"`) | the Optuna engine and the reinforcement-learning agents (Gymnasium, Stable-Baselines3, sb3-contrib), without NGBoost |
| `pip install ".[ml-probabilistic]"` | NGBoost alone |

**NGBoost and pandas 3.** NGBoost depends on `lifelines`, which requires pandas below 3.0. On Python 3.11 or later, installing NGBoost therefore replaces pandas 3 with the last pandas 2 (the platform runs on both, and on Python 3.10 pandas 2 is all there is). If you want pandas 3, install `requirements-ml-no-ngboost.txt` and leave the `ngboost` model unavailable. CI follows the same rule: the Python 3.11 jobs and the Docker image install the file without NGBoost, so that they keep testing pandas 3; the Python 3.10 job installs everything and is the only one that runs the NGBoost tests; and the `ml-base` job installs none of them and checks that the tab, the pipeline and the endpoints work as before.

### Limits worth knowing

- The agents are trained for the number of steps in the form (a few thousand by default). With so little experience they usually learn little; the reward curve says whether they did, and the equity curve (out of sample) is the evidence, not the in-sample panel.
- The networks, the graph models and the agents are compact implementations written for this platform (PatchTST, N-HiTS and the Temporal Fusion Transformer follow the papers' main ideas; they are not ports of the reference code; the model form says which networks are reduced and what each leaves out, from the architecture manifest in `src/ml/deep/manifest.py`, which `GET /api/ml/catalog` also serves), and a graph is dense, static per fit, and meant for a few dozen assets.
- The regime models fit the *market's* regimes (from the equal-weight return, its volatility and the cross-sectional dispersion) and then one ridge per regime; a Markov-switching fit can fail to converge on a short window, and that refit is then skipped, with the reason, like any other fit that raises an error (the report lists the skipped fits).
- A re-backtest restores the intervals, regime probabilities and weights (saved with the predictions) but not the training curves, which belong to the fits.
- All of this runs on CPU unless a GPU is visible; `device: auto` picks CUDA when there is one and says which it used.

## Correctness checks and what a number means

The correctness programme of spec v3 (progress, requirement-by-requirement traceability, architecture and limitations in [ml_correctness_v3](ml_correctness_v3/PROGRESS.md)) adds three things you will meet in the tab and the API:

- **Output contracts.** Each model's primary score has a kind (`expected_return`, `ranking_score`, `class_probability`, `quantile_forecast`, `portfolio_weights`, `volatility_forecast`, `drawdown_forecast`), and a position rule that would misread it is refused before training starts, with
  the rule to use instead: an agent's weights go through `target_weights` and are not ranked again, a ranker's score is not a return forecast, a volatility forecast is not how attractive an asset is. `signal.reinterpret_outputs` is the explicit override and is
  written to the report. Every report carries `output_contract` (schema, kind, units, horizon, asset order, feature-schema hash, state schema).
- **Checked, not assumed.** `python -m src.cli ml-verify --run` (or `POST /api/ml/verification/run`) runs the tests behind each model-correctness requirement and records the result in the registry; `GET /api/ml/verification` shows the latest status of each,
  and marks it `stale` when the source it checked has changed. A status is written only by a run. `blocked` means tests were skipped (a package is missing), which is not a pass.
- **Safer artifacts.** A saved model carries a checksum; a damaged file is refused with a message, a missing feature is named, and an agent trained under an older state schema says it must be retrained.

### Causal data and training integrity (Release 2)

- **Labels have intervals.** The label of the row on day `t` starts at the close of `t + lag` and ends at the close of `t + lag + horizon`; it is *available* at that last close. A model fitted at the close of day `D` trains only on rows whose label ended by `D` (less the embargo), counted on the real calendar, not on a list of the dates that happen to have labels. The same-close order is: prices print, labels settle, the model refits if due, features are computed, the decision is made. Each fold of a report says when it was fitted (`fit_cutoff`), the last day its labels reach (`label_end_max`) and a hash of its training rows; `ml-report` prints this as *fold lineage*. `intervals.purged_kfold_rows` and `cpcv_rows` give purged K-fold and CPCV splits on the same intervals; they train on later blocks, so they are marked `chronological=False` and are diagnostics, not simulations of trading.
- **Features are audited, not trusted.** `src/ml/audit.py` perturbs one input panel at a time and recomputes every registered feature: it reports which panels each feature reads, how many trailing rows it needs and whether replacing, blanking, freezing or leaving a hole across a cutoff changes anything before it. The dataset audit (`report['audit']`, `meta['audit']`) lists the label cutoffs, every excluded row by reason, stale price runs, closed-market marks, the point-in-time quality of the macro inputs and the session policy. Values the data lack stay missing; each model learns its fills from its own eligible training rows. Nothing is back-filled.
- **A carried price is not a trade.** A bundle that carries a closed market's last price over a holiday marks those cells (`MarketBundle.marks`). Such a row is excluded, the asset is not investable that day, and a label whose entry or exit price is a mark is void.
- **Point in time.** `src/ml/pit.py` joins a published series on when each value became public (the fundamentals engine's rule: the latest period that was public, with its latest public vintage), session-aware. The platform keeps no macro vintage history, so its FRED series are reported as `estimated` (a fixed release lag on the latest vintage) and validation claims that rest on them are marked restricted; market prints (VIX, Treasury yields) and series computed from prices are `observed`.
- **Sessions.** Markets close at different instants, and a daily bar is final only then. `dataset.sessions` (optional) names the session of a decision; a bar that closes after the decision (a crypto day against a US close, any other market against a Tokyo close) is read as of the decision, a lag of zero is refused where an entry price would precede the decision, and the report lists each asset's session and where it was only assumed. `policy: date_labels` keeps the old join on the date label and says so.
- **Stacking and meta-labelling.** The ensemble weights and calibrates its members only on predictions they made for rows none of them was fitted on, and records the two stages; meta-labelling trains on labels settled by its own fit cutoff and records which folds produced the primary scores. `TrainResult.producers()` and `oos_violations()` answer *which model made this prediction and what had it seen*.
- **Overlapping labels.** Labels one day apart share most of their returns. `split.sample_weighting: uniqueness` weights each training row by its average uniqueness (López de Prado); the report says how overlapped the labels were and whether the model used the weights (the linear, boosted-tree and regime models do; the others say `requested, not applied`).
- **One artifact to answer from.** Each saved experiment also writes `model.qfb`, an inference bundle: the model, the exact columns it was fitted on (names, order, dtype kinds), a description of its preprocessing, its cutoffs and label convention. The forward tracker checks new data against it and refuses drift; a declared, versioned adapter can bring a renamed or reordered input into line and is recorded when used. A signal the tracker produces for a date the saved model had already seen is flagged as a replay. The experiment's HTML report and `ml-report` show the dataset audit, the treatment of overlapping labels and the fold lineage; a run saved before they were kept says `not recorded`.

### Research controls and protected holdouts (Release 3)

Every way of trying something on the data is recorded in a *study* before it runs: experiments, search and tuning trials, model choices, signal and portfolio variations, ensembles, meta-labelling, feature searches. Failed
and cancelled tries stay in the count; deleting an experiment leaves a tombstone and lowers nothing. `GET /api/ml/lineage`, `ml-lineage` and the *Research controls* view show it.

The test and forward segments are protected by **evaluation plans**: `POST /api/ml/evaluation-plan` (or `ml-plan create`) stores the candidates and rules, reserves the outcomes they would reveal and reports any overlap with an
earlier look; freezing is an explicit step; `POST /api/ml/evaluate` (or `ml-evaluate`) is the one evaluation event, idempotent by key. A search or a tuning stops at the end of the validation segment; `finalize` is off by default and,
when asked for, freezes a plan and evaluates it once. A history that a plan reserves cannot be read by a direct run (train with `mode: development`, which stops at validation); history no plan covers is recorded as an
exposure and the run is labelled exploratory. A result computed outside a plan, or from a re-used history, is shown as not untouched and cannot count toward Strategy Validated.

`ml-selection-tests` and `POST /api/ml/selection-tests/run` apply the deflated Sharpe ratio, PBO, the Reality Check, SPA and FDR control to a study's trials *where they apply* and keep each result or the reason it was not applied
(`not_applicable`, `unsupported`, `insufficient`, `failed`); `ml-null-controls` runs the whole search pipeline on placebo data and reports the false-promotion rate with an exact interval, labelling a small run as a check of the plumbing.
`ml-promotion` and `GET /api/ml/promotion-status` show the three gates, each computed on the server from persisted evidence; `ml-promote` and `POST /api/ml/promote` move a model only as far as that evidence allows, and production
also needs a recorded human approval. Nothing here places an order or deploys anything.

### Executable trading and reconciled accounting (Release 4)

A result is evaluated in one declared mode. With no `execution` section it is **vectorized_legacy** (the platform's backtest engine: a cost per unit of turnover; no orders, capacity, cash or borrow). With `execution.mode: realistic_cash_equity` the same weights go through orders, fills and the portfolio ledger.
A realistic result is always a child run of the experiment it re-runs:

```
quant ml-realistic <experiment> --capital 1000000 --participation 0.05 [--prices prices.csv] [--shorts refuse] [--min-adv-dollars 5e6 --min-listing-days 60]
POST /api/ml/rebacktest {"id": "...", "execution": {"mode": "realistic_cash_equity", "capital": 1000000, "max_participation": 0.05, "impact_coefficient": 0.5}}
GET  /api/ml/trading-diagnostics?id=...
```

The report (`report["execution"]`) carries the timeline from decision to fill (the lag is applied once), the price convention, gross and net, the itemised costs, turnover, leverage, equity before and after costs, capacity at capital levels declared before the run, the accounting reconciliation, a comparison with the vectorized engine on the same days, the
capability report (what the data and adapter support, the blockers and the limitations) and a manifest of the order, fill and journal files with their hashes. Costs default to the backtest section's; the impact coefficient is an assumption unless calibrated, and says so. Gate B reads this evidence; a vectorized result cannot pass it.

### Statistical and prospective validation (Release 5)

**A policy first.** The platform implies no Sharpe, deflated-Sharpe, PBO, p-value or paper-trading-duration threshold. A named person adopts a versioned *validation policy* before the evaluation (`quant ml-policy --file policy.json --actor NAME --note REASON`, or `POST /api/ml/validation-policy`); an illustrative example exists (`--adopt-example`) and is not a recommendation. Until one is adopted, Gates B and C are blocked.

```
quant ml-diagnostics <experiment> [--neighborhood '{"signal.n": [2, 3, 4]}'] [--delays 0,1,2,5] [--capitals 1e5,1e6,1e7] [--prices prices.csv]
POST /api/ml/statistical-diagnostics/run {"id": "...", "n_boot": 500, "delays": [0, 1], "neighborhood": {"signal.n": [2, 3, 4]}}
GET  /api/ml/statistical-diagnostics?id=...        # the stored record, read against the policy in force (stale under another policy)
```

The diagnostics give the annualised Sharpe ratio with **three statements of its uncertainty around one estimate**: the naive interval (which says it treats days as independent), a HAC interval (Newey-West, delta method) and a stationary-bootstrap interval; the active return against *matched* benchmarks (beta, net-exposure, sector and risk matched, estimated only from information known before each day) with the unmatched
equal-weight market shown as descriptive; regimes declared before the evaluation (a regime with too few days is insufficient evidence, never dropped); and the neighbourhood, delay and capacity analyses, which re-run the saved predictions, report the whole grid and never a best cell, and count as looks. Diagnostics refuse the test and forward segments of an experiment that no frozen plan evaluated.

**Forward testing is a ledger.** `quant ml-forward <experiment>` (or the Forward test view, `POST /api/ml/forward/step`) records the model's signal, writes an immutable *commitment* (keyed by strategy version, decision time, universe version and horizon; the server's creation time; the data and fit cutoffs and artifact hashes) and settles the commitments whose window has passed. The server classifies each commitment as **prospective** or a
**replay**: a replay of history is listed and never counts toward prospective evidence. A revision of realised data is a new settlement version with a reason. The strategy's equity is built from the committed holdings and the daily prices; a multi-day forecast is not turned into a daily return.

**A strategy version** is the model with its signal, execution and risk settings (`quant ml-governance <experiment> --register --actor NAME --limits '{"max_gross": 1.5, "max_drawdown": 0.2}'`). The platform refuses to commit a book that breaches a documented limit and a version that is halted or rolled back; it keeps a written recovery procedure, monitoring checks and the separate human deployment approval, each with a name and a time. Gate C reads them.

### Alpha research foundation (Release 6)

**Targets.** Besides `fwd_ret_h` and `excess_ret_h` there are `sector_ret_h` (the return less the mean of the asset's group on that date) and `resid_ret_h` (the return less a loading times the benchmark's, the loading estimated from the 252 trailing days up to the decision). The dataset's record says what each one means, how its loading was estimated and that a static sector map is not point in time.

**Diagnostics of a saved experiment** (development data; the sealed segments stay sealed):

```
quant ml-alpha <experiment> [--baseline <other experiment>]
quant ml-portfolio <experiment> [--horizon 21] [--dates 24]
GET  /api/ml/alpha-diagnostics?id=...&baseline=...      # rank IC with a dependence-aware interval, decay by delay and horizon, coverage, stability, redundancy, conditional IC, blend, correlation
GET  /api/ml/portfolio-diagnostics?id=...                # the book held, covariance estimators on realised risk, risk contributions, allocation sensitivity (return forecasts only)
```

**Controlled comparisons.** One change at a time against a baseline, on the same data, folds and costs, on the validation segment. A *card* states the hypothesis and the one change before any number exists, is frozen, and runs once.

```
quant ml-improve --spec card.json --prices prices.csv        # family: target | feature | horizon | portfolio
POST /api/ml/improvement-experiment {"family": "target", "from_experiment": "...", "presets": ["excess_ret", "resid_ret"], "hypothesis": "...", "one_change": "..."}
GET  /api/ml/improvement-experiment?id=cmp-...               # the card with its results, negative findings and gate decision (always not evaluated)
```

The verdicts are `supported_on_development`, `worse` or `inconclusive`, each with its reason and the paired difference with its interval; losing and inconclusive candidates stay listed. The feature family compares single economically motivated feature groups before the stability-selected set (model kind `stable_select`, which chooses its features inside each fold's training rows and records the chosen and rejected ones with their frequencies). The horizon family moves the target, the horizon and the holding period together and compares the daily net returns of the book each implies; it never divides a multi-day return by its horizon. The portfolio family compares the existing position rule with `cost_aware_mean_variance`, which weighs a return forecast against risk and trading cost from the book actually held and counts every solve (optimal, infeasible, failed, no trade); it is refused for a rank score.

### Strategy-improvement experiments (Release 7)

New rules are overlays on the book a position rule builds; each is off unless its setting is present, so an existing configuration is unchanged.

```
signal: {no_trade_margin: 1.0}            # trade only if the gain exceeds the cost times (1 + margin); no_trade_hold_margin adds hysteresis
signal: {trade_filter_k: 1.0}             # trade only if the net edge is at least k forecast-error deviations
signal: {method: calibrated_sizing}       # size from a walk-forward calibrated forecast and its uncertainty; flat with no demonstrated edge
signal: {regime_exposure: vol_quantile}   # also hmm, markov_switching: filtered states, budgets from the book's own past, never above 1
model:  {kind: model_ensemble, params: {members: "ridge,lasso", weighting: stacked, shrink: 0.5}}   # also stacked_regime
```

```
quant ml-delay <experiment> [--delays 0,1,2,5]     # IC, expected edge, net result and capacity when the book is executed later
GET  /api/ml/delay-report?id=...&delays=0,1,2,5&participation=0.1
POST /api/ml/improvement-experiment {"family": "rebalancing", "presets": ["no_trade:1.0", "trade_filter:1.0"], ...}
POST /api/ml/improvement-experiment {"family": "risk_control", "candidates": [{"signal": {"regime_exposure": "vol_quantile"}}], ...}
POST /api/ml/improvement-experiment {"family": "ensemble", "members": "ridge,lasso,elasticnet", "select_baseline": true, "presets": ["weighting:stacked"], ...}
POST /api/ml/improvement-experiment {"family": "combination", "combination_of": [{"card_id": "cmp-...", "candidate": "..."}, ...], "ablation": true}
POST /api/ml/combine {"ids": [...], "development_end": "2021-12-31"}   # adds mixture_study: the best single member chosen up to the date against each mixture after it
```

The families are `target`, `feature`, `horizon`, `portfolio`, `rebalancing`, `risk_control`, `ensemble`, `execution` and `combination`. A combination exists only for components that were each compared on their own, and its tests join their multiple-testing family; an ablation removes one component at a time. A risk-control card reports the candidate's drawdown at the baseline's volatility, because a smaller position lowers drawdown without any timing skill. The verdicts are still development-segment verdicts, and the gate decision of every card is `not_evaluated`.

### Optional advanced modules (Release 8)

Twelve modules are available on request and are off otherwise: nothing in a training, search, tuning or forward-test workflow imports them, and they need only the base install. `quant ml-advanced catalog` (or `GET /api/ml/advanced`) lists them with how each is reached and whether it can run here.

```
quant ml-advanced drift <experiment>                 # PSI and an adjusted KS test of every feature against the window the model was fitted on; drift of the settled rank IC
quant ml-advanced monitor <experiment>               # calibration, forecast distribution, stated uncertainty and member disagreement, each on its own
quant ml-advanced retrain-trigger <experiment> [--kind manual] [--policy policy.json]    # a trigger is recorded, and is a proposal or a suppressed trigger with the reason
quant ml-advanced retrain-approve <experiment> --proposal N --actor NAME --note TEXT      # also retrain-reject, retrain-recover, retrain-status
quant ml-advanced champion-register --spec pair.json --actor NAME     # also champion-commit PAIR --spec decision.json --actor NAME, champion-show PAIR and champion-look PAIR --actor NAME
quant ml-advanced report-build <experiment>          # also report-show REPORT [--markdown] and report-verify REPORT
```

A drift alert, a monitor alert and a retraining proposal are things for a person to read: none retrains, halts, rolls back or promotes anything, and a stored report does not change when models or data are updated later. The numerical modules (`bma`, `dro`, `horizon_combo`, `transfer`, `discovery`, `offpolicy`, `tax_lots`) take arrays and frames you supply and have no route; each states its assumptions and refuses, with a reason, what it cannot support: an off-policy estimate without logged behaviour probabilities, an after-tax figure without a supplied policy, a transfer between assets that share a security.

### Adaptive retraining (P3-02)

The retraining procedure above (a trigger becomes a proposal or is suppressed with its reason, a named person approves, a failure suspends, a recovery lifts it, a rollback goes through governance) now has the three things it was missing: what triggers a retraining, what a retraining trains, and how the result is judged. Nothing here is automatic: autonomous retraining cannot be switched on from a route, the command line or the dashboard.

```
quant ml-advanced retrain-evaluate <experiment> [--refresh] [--as-of DATE]   # every trigger now: the schedule from the calendar, the diagnostics from the stored looks
quant ml-advanced retrain-trigger <experiment> --kind schedule|drift_alert|performance_alert|calibration_alert|manual [--policy policy.json]
quant ml-advanced retrain-approve <experiment> --proposal N --actor NAME --note TEXT
quant ml-advanced retrain-start <experiment> --proposal N --actor NAME [--policy policy.json]   # the constrained challenger search of an approved proposal
quant ml-advanced retrain-evidence <experiment>      # also retrain-status (with the versions), retrain-rollback --from-version V --to-version W --actor NAME --note TEXT
```

* **Triggers.** A *schedule*, monthly by default (a daily strategy refits every 21 sessions; weekly, quarterly and a session count are options), is due when a new calendar period has begun since the last fit. A *drift alert* and a *performance alert* come from the latest drift report (P3-01), a *calibration alert* from the calibration check of the uncertainty monitor (P3-11, `calibration_min_consecutive` looks in a row). Each is verified against the calendar or the stored record: a diagnostic never run is `not_computed`, one that cannot exist for the model is `not_applicable`, and neither is read as health.
* **A constrained challenger search.** The incumbent's configuration is the baseline of a frozen comparison card (family `retraining`). Challengers change only how the same model is trained: the **training window** (`windows`: expanding, or rolling over a number of days), the **refit interval** (`refit_every`, 21 sessions is monthly) and the hyper-parameter values listed in `param_grid`, one at a time. Anything else is refused, the number of challengers is capped (`max_challengers`) and the Holm adjustment counts every one.
* **Identical out-of-sample observations.** The incumbent is re-run beside its challengers on the same data; the card refuses an arm that scored different validation rows, every difference is paired on the days both have, and the evidence records that no day was dropped. All of it is development data: the protected segments are never read (the holdout access ledger does not move).
* **Judgement.** The arms go through the constrained selection above. A challenger is *recommended* only if it is the arm the selection chose **and** it is distinguishably better than the incumbent (Holm-adjusted, interval above zero, at least `min_effect`). A drift alert is why the search ran, not evidence for the challenger, and a better historical score is not evidence either: in the tests a challenger with a clearly higher validation Sharpe whose interval includes zero leaves the incumbent in force.
* **Nothing is promoted.** The outcome is a recommendation with its evidence (stored append-only as `retrain_evidence`, with the policy hash). The incumbent, its re-run and every challenger get a strategy version in governance, so a rollback names two versions and keeps both; a promotion still needs the gates and, for production, a recorded human approval.

| field | default | what it does |
|---|---|---|
| `name`, `triggers` | `adaptive-monthly`; schedule, drift, performance, calibration, manual | the policy's name and the trigger kinds it enables |
| `schedule` | `monthly` (`frequency`; `every_sessions` 21 for the session schedule) | when a scheduled retraining is due |
| `cooldown_days`, `min_new_rows`, `min_train_days`, `max_failures` | 21, 10, 504, 3 | days between proposals; rows since the last fit; history needed to retrain; failures before a suspension |
| `calibration_min_consecutive` | 1 | looks in a row at which the calibration check must alert |
| `windows` | expanding, rolling over 1260 days | the training windows tried (the incumbent's own is left out) |
| `refit_every` | 21 | sessions between refits tried (monthly for a daily strategy) |
| `param_grid` | none | hyper-parameter values to try, one parameter at a time |
| `max_challengers` | 6 | a hard cap (at most 12) |
| `min_effect`, `level` | 0.0, 0.05 | the smallest Sharpe improvement that counts; the Holm-adjusted level |
| `n_boot`, `seed` | 1000, 0 | the bootstrap of the paired difference |
| `selection` | the selection defaults | a selection policy mapping for the choice among the incumbent and its challengers |

Limits: the calibration trigger reads interval coverage (a model that states no interval has none); probability calibration of a classifier is reported by the research record but has no alerting look. A challenger is fitted by the same walk-forward machinery, so a retraining that only refits on more data reproduces the incumbent: what the search can tell apart is a different window, schedule or nearby hyper-parameters, which the policy lists in advance.

## If LightGBM or XGBoost shows as "not available"

The two boosting libraries are optional and need the OpenMP runtime of the operating system (`libgomp.so.1` on Linux). Where the library is missing, importing them fails with `OSError: libgomp.so.1: cannot open shared object file`. The Machine learning tab does not fail with it: the model list shows LightGBM and XGBoost as not available, with the reason, and every other model (linear, ridge, lasso, elastic net, logistic, random forest, scikit-learn's histogram gradient boosting, autoregression, Kalman, online and regime models) works as before. Scikit-learn's histogram gradient boosting is the same family of model as LightGBM and needs nothing extra.

To switch the two on, install the runtime and restart the app: on Debian, Ubuntu or a Docker image, `apt-get install -y libgomp1`; on a Mac, `brew install libomp`. On Replit, which provides system libraries through Nix, add a package that provides `libgomp` (the GCC runtime library) in the Dependencies tool and press Stop, then Run; if that does not work, leave the two models off, since nothing else depends on them.

## The HTTP API

Served by `src/webapp/ml_api.py`; JSON in, JSON out. Anything slow returns `{"job": id}` and is polled through `/api/job/<id>`.

| route | what |
|---|---|
| `GET /api/ml/catalog` | models and their settings (`family`, `specs`, `schema`, `needs`, `outputs`, `available` with the reason), feature groups, targets, position rules, combination methods, search `engines`, defaults, limits, and what tuning each model varies (`tune_spaces`, `tune_efforts`) |
| `POST /api/ml/dataset/build` (job) | build and describe a dataset |
| `POST /api/ml/train` (job) | the whole experiment; returns the id and the report. `model.params` names only the settings to change; each is validated against the model's schema and laid over its defaults |
| `POST /api/ml/rebacktest` (job) | saved predictions under other signal, backtest or robustness settings |
| `POST /api/ml/meta-label` (job) | size an experiment's bets by a second classifier |
| `POST /api/ml/search`, `/api/ml/tune` (jobs), `POST /api/ml/tune/stop` | strategy search; tuning one model (with `effort`: from the form's own settings, with a verdict; with `engine: optuna`: Optuna's sampler); end a tuning early |
| `GET /api/ml/experiments`, `/api/ml/experiment`, `/api/ml/models`, `/api/ml/trials`, `/api/ml/feature-sets` | the registry |
| `GET /api/ml/compare`, `/api/ml/explain`, `POST /api/ml/combine` | comparison, one prediction explained, strategy combination |
| `GET /api/ml/forward`, `POST /api/ml/forward/step` (job) | forward-test signals and the daily step |
| `GET /api/ml/alpha-diagnostics`, `GET /api/ml/portfolio-diagnostics`, `GET /api/ml/improvement-experiment`, `POST /api/ml/improvement-experiment` (job) | Release 6: alpha and portfolio diagnostics of a saved experiment; frozen comparison cards (Release 7 adds the families `rebalancing`, `risk_control`, `ensemble`, `execution` and `combination`) |
| `GET /api/ml/delay-report` | Release 7: IC, expected edge, net result and capacity at declared delays |
| `GET /api/ml/advanced` | Release 8: the twelve optional advanced modules, off by default, with their door, availability and latest verification |
| `GET /api/ml/drift`, `GET /api/ml/uncertainty-monitor` | Release 8: drift of the features and the settled IC; the four uncertainty checks (`stored=1` reads the latest record and computes nothing) |
| `GET /api/ml/retraining`, `POST /api/ml/retraining/action` | Release 8: the retraining history of an experiment with the adaptive policy's defaults, the latest challenger-search evidence and the versions; evaluate, trigger, approve, reject, start (a job: the constrained challenger search of an approved proposal; development data only, never a promotion), recover, rollback (decisions name a person) |
| `POST /api/ml/select`, `GET /api/ml/selection`, `GET /api/ml/selection-policy` | the constrained selection over saved experiments (validation rows only), a stored selection, and the documented defaults of every threshold; `selection` is also an option of `POST /api/ml/search` and `/api/ml/tune` |
| `GET /api/ml/champion`, `POST /api/ml/champion/register`, `/commit`, `/look` | Release 8: a frozen champion/challenger pair, both forecasts committed together, the predeclared looks |
| `GET /api/ml/reports`, `GET /api/ml/report`, `POST /api/ml/report/build` | Release 8: reproducible reports built from a stored snapshot |
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
- **A model from a package that is not in `requirements.txt`:** put its entry in `catalog_extra.py` with `needs=("package",)` and a module in `EXTRA_MODULE` whose `build(kind, task, params, seed)` imports the package lazily; the catalogue lists it as unavailable, with the reason, where the package is missing. Give it `specs` for any setting that is not a number, `outputs` for what its report shows, and, in `param_schema.RANGES`, a range for each numeric setting (a test fails for a setting that has none, and for a range that belongs to no setting).
- **A feature:** write a function of `Inputs` returning a date by asset frame, register it in a group of `features.py` with a description; the causality test in `tests/test_ml_data.py` picks it up and fails if it looks ahead.
- **A position rule:** add it to `signals.build_weights` and `METHODS`; give it a line in `webapp/ml_api.SIGNAL_TEXT` and `static/ml.js`.
- **A universe:** teach `datasets.resolve_bundle` to return a `MarketBundle` for it.

## What it does not do

It does not send an order, and the forward tracker's signals are paper. It does not make a model good: on data without an edge the validation, test and forward Sharpe ratios fall towards zero and the page says so. A search that picks the best of many trials reports the number of trials beside the winner and deflates its Sharpe ratio, and the reserved segments are seen only once a finalist is run on them: they are still not the future, and the forward test is the only evidence that cannot be re-tuned.
