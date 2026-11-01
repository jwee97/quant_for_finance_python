# Architecture: how Releases 1 and 2 fit the existing platform

## What was reused

The existing `src/ml` pipeline (`datasets` -> `training.walk_forward_train` -> `experiments.evaluate_run` -> `backtest`), the `Model` interface, the model catalogue (`models.py`, `catalog_extra.py`), the SQLite registry (`registry.py`),
the API (`webapp/ml_api.py`) and its job queue, and the `quant` command line (`src/ml/cli.py`). No second model zoo, backtester, registry or job queue was added.

## Time and state contract of the RL portfolio

```
close of day t:   observe (features[t], holdings, pending order, drawdown)  ->  choose target weights
                  queue the order for the close of day t + lag
session t -> t+1: marks move by returns[t+1]; financing and borrow accrue
close of day t+1: orders that are due execute against the DRIFTED holdings (mandatory limits first, then the turnover
                  budget, then the hard position/exposure limits); the cost is charged; equity, peak and drawdown update
reward:           the realised net return of the session, in percent (return | risk_adjusted | differential Sharpe)
```

A book decided on day `t` is held from `t + lag` and earns the return of day `t + lag + 1`, the platform's signal-lag convention. No observation contains a return that has not happened by its close. The same `PortfolioEnv.step` is used
by training and by the policy rollout that produces the strategy's books, so the two cannot diverge. A fitted agent has two replay chains, both starting flat (the training window, and everything after it), which makes its books independent
of how the rows are chunked in calls.

The ledger (`ledger.py`) derives equity, weights and costs from cash, quantities and marks. A trade is solved for cash feasibility: target weights are weights of the equity left after the trading cost.

## Output contract

`contracts.output_kind(kind, task, target)` names what a model's primary score is; `check_signal_use` decides whether a position rule may read it; the validators enforce each kind's promise where it is produced. The versioned contract
(`schema_version`, `output_kind`, `units`, `horizon`, `asset_order`, `feature_schema_hash`, `decision_at`, `state_schema_version`, ...) is written to every report. Fields that cannot be known are `null`.

## Graph masking

`GraphNet.forward(x, valid)` removes absent nodes from the graph of each batch row: GCN recomputes `D^-1/2 (A+I) D^-1/2` over the present nodes; GAT masks both the query and the key and renormalises; embeddings are multiplied by the
presence mask after the input layer and after every residual layer. The reference in the tests is a second network built on the induced sub-adjacency with the same weights.

## Persistence

`registry.MIGRATIONS` is an ordered list of additive migrations, each applied once inside one transaction and recorded in `schema_migrations`. `verification_checks` is append-only. `verification.REQUIREMENTS` maps a requirement ID to the test
files that check it and the source files it hashes; `run_verification` runs them in a subprocess, reads the JUnit report and records one row per requirement. `current()` marks a row `stale` when the hashed source has changed since.

## Version changes

| Item | Version |
| --- | --- |
| RL state schema | 2 (observation = features, holdings, pending order, drawdown) |
| Model artifact envelope | `QFM1` (SHA-256 of the payload) |
| Output contract | schema 1 |
| Architecture manifest | 1 |
| Registry schema | migration 1 (`verification_checks`) |
| Verification check | version 1 |


# Release 2: causal data and training integrity

## What was reused

`datasets.build_dataset`, `training.walk_forward_train` and `fit_final`, `splitters`, the model classes' existing `bind` hook, the fundamentals point-in-time engine (`src/fundamentals/pit.step_panel`), the registry's report JSON, `experiments.run_experiment`
and the forward tracker. No second trainer, splitter, registry or job queue was added. New modules: `intervals.py`, `causal.py`, `audit.py`, `sessions.py`, `pit.py`, `bundle.py`.

## The order of events at one closing timestamp

```
close of day D:   prices print
                  labels whose last day is D settle  (label_end <= D, less the embargo, are eligible)
                  a model due for refit is fitted on the eligible rows, bound to a CausalView that ends at D
                  features of day D are computed from data to D (session-aware: bars that closed after the decision are read as of it)
                  the decision for D is made; the order executes at the close of D + lag; it earns the return of D + lag + 1
```

## The causal view

`CausalView(ds, until)` copies nothing it should not have: `X` is the prefix of the dataset's rows up to `until`, `prices`, `returns` and `investable` are cut at `until`, `dates` ends there, and `Y` is not reachable (`CausalAccessError`).
`walk_forward_train` binds a fit view that ends at the first decision of a block and a predict view that ends at the block's last day; `predict_latest` binds a view that ends at the decision date and refuses (`ReplayError`)
to decide a date before the last day the model's labels reach. A raw dataset handed to `bind` is wrapped in a view whose cutoff is its last date.

## Label intervals

`intervals.py` holds the label convention (`CONVENTION`: close of `t + lag` to close of `t + lag + horizon`, adjusted close, date resolution), `label_frame` (start, end, availability per decision date, persisted as `label_intervals.parquet`), `eligible_for_fit`,
`concurrency`, `average_uniqueness`, `uniqueness_weights`, and the interval forms of purged K-fold and CPCV. The trainer takes the horizon of the target it trains.

## Audit, point in time and sessions

`audit.feature_dependencies` is a measurement, run by the tests over `FEATURES`; `audit.dataset_audit` is stored with every dataset (`meta['audit']`) and copied into the report. `pit.vintage_asof` reuses the fundamentals engine's value selection
with availability resolved to the decision instant by `sessions.decision_instants`. `sessions.asof_panel` gives each asset's latest final bar and its age; `datasets._session_inputs` applies it to the features of assets that close after the
decision session and leaves the labels on the real prices.

## Lineage and the bundle

`fit_info_` on every model (cutoff, last label day, row hash, feature schema); `TrainResult.producers()`; `MemberEnsemble.lineage_`; `meta_label_scores(...)['lineage']`; the fold list of the report. `bundle.InferenceBundle` (`QFB1`: SHA-256, a JSON header, the
model's own `QFM1` bytes) checks schema drift strictly and applies declared adapters.

## Version changes (Release 2)

| Item | Version |
| --- | --- |
| Label convention | `close_to_close` (recorded in `meta['label_convention']`) |
| Inference bundle | `QFB1`, version 1 |
| Verification registry | 23 requirements (P0-01 to P0-22 and P0-30) |
| Changed semantics | training rows are chosen by label interval (the position rule it replaces was at least as strict on regular calendars: results are identical there); the strategy combination `online_learning` no longer back-fills early volatility (`params.volatility_fill` says `causal_v2`); `idio_vol_63` declares a window of 126; the regime model's diagnostics describe the fit window, not the block it predicts |
| New optional settings (absent from the defaults, so no existing configuration hash changes) | `dataset.sessions`, `split.sample_weighting` |
