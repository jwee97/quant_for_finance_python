# Architecture: how Release 1 fits the existing platform

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
