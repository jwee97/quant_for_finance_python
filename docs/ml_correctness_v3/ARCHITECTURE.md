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

# Release 3: research controls, reproducibility and protected holdouts

## What was reused

The experiment registry (`src/ml/registry.py`, SQLite with additive versioned migrations) is the only authority; no second database, job queue or registry exists. The statistical procedures are the platform's own
(`stats/inference.py`, `validation/multiple_testing.py`, `ops/hpo.py`); the new layer decides whether each one applies and records the answer. The job mechanism, the response envelopes and the `MlApi` route tables carry the
new endpoints; the dashboard gets one new view in the Machine Learning workspace (Research controls), not a new top-level tab.

## Where the pieces live

| Concern | Module | Persisted in |
| --- | --- | --- |
| Study lineage (every try, before it runs) | `research_controls.py` (`begin`, `experiment_op`, `lineage`, `raw_looks`) | `research_studies`, `lineage_events` (append-only), `experiment_tombstones` |
| Provenance identity | `provenance.py` | `provenance` (append-only) and `experiments/<id>/provenance.json` |
| Evaluation plans, reservations, the access ledger | `research_controls.py` (`create_plan`, `reserve`, `freeze_plan`, `begin_evaluation`, `open_outcome_access`), `frozen_eval.py` | `evaluation_plans` (immutable), `plan_events`, `holdout_reservations`, `holdout_access_log` (append-only) |
| Selection-test eligibility | `selection.py` | `selection_evidence` (append-only), trial return streams in `trials/<study>/` |
| Null-control studies | `null_controls.py` | `null_studies`, `null_replications` |
| Gates and promotion | `gates.py`, `Registry.promote` | `gate_decisions` (append-only), `stage_changes` |

## Lineage

An *operation* is registered (`registered`, then `running`) before the work starts and ends as `done`, `failed` or `cancelled`; each state is an appended event. `counts=0` marks an operation that re-runs a counted try
(the finalisation of a trial, the evaluation of a frozen plan); it has a `parent_key` and is never counted twice. An enclosing operation that *adopts* its experiments (a search trial, a finalisation) is the experiment: the
inner `run_experiment` records the experiment id on it instead of registering another. A study is by default the universe the data comes from, not a cache key: changing the feature set, target, dates or the name of an
experiment keeps the history; `study` in the configuration names one, and `link_study` records that one continues another. Deleting an experiment leaves a tombstone and the legacy `trial_count` includes tombstoned
experiments, so deletion lowers no count. Raw look counts are counts of recorded tries; they are not estimates of independent trials, which the selection tests report (`effective_trials`).

## Holdout lifecycle

1. *Development*: `stop_after="validation"` runs predict nothing after the validation segment (`guard_development` asserts it). Search, tuning and the dataset diagnostics are development: `feature_ic` refuses to read every date without
   an explicit horizon or an authorised context.
2. *Plan*: `create_plan` stores the full merged configurations of the baseline and candidates, the signal/execution/retraining rules, metrics, uncertainty method, selection adjustment and decision policy. A plan with several
   candidates must declare its selection treatment. Plans are immutable (a trigger refuses updates).
3. *Reserve*: `reserve` binds the plan to the assets, dates and a hash of the realised returns of its test and forward segments, and records every overlap with an earlier reservation, evaluation, exposure or identical
   outcome content (renamed assets). `untouched` is false when any overlapping look was consumed.
4. *Freeze*: the explicit transition; refused without a reservation.
5. *Evaluate*: `begin_evaluation` runs in one write transaction: an idempotency key returns the stored result on a retry, `in_progress` while claimed, and is takeable with `resume`; a different key on a consumed plan or a
   history with consumed overlaps is refused unless `allow_reuse`, which is recorded as `evaluate_reuse` and reported as not untouched. A failed evaluation stays consumed.
6. *Exposure*: a direct run that predicts the test and forward segments is refused when a reservation covers that history; otherwise it is recorded as an `exposure` and its report says `exposed_exploratory`. Derived runs
   (re-backtest, meta-labelling) inherit what their parents read; combinations record an `ensemble` operation with their parents.

The legacy `finalize: true` request on a tuning or a search is routed through the same machinery (`frozen_eval.compat_finalize`): it freezes a plan from the chosen candidates, evaluates it once as a recorded re-use where the
history was already read, and says so in its response (`deprecated`, `evaluation`). `finalize` is now off by default.

## Provenance

`provenance.input_manifest` hashes the whole history of every input of a market bundle (prices, returns, mask, high, low, volume, every macro series, marks, and the descriptions of how values became public); the dataset
manifest hashes the dataset's own frames and policies. Values are hashed as bytes and dates as UTC nanoseconds, so the identity does not depend on the datetime unit pandas chose. The identity covers data, source tree,
configuration, seed, features, folds and output contract; the environment and hardware are recorded beside it, not in it. `datasets.bundle_fingerprint` stays a cache key and says so.

## Gates

`gates.evaluate` computes Gates A, B and C from the persisted evidence; `decide` appends a decision when status, staleness, policy or evidence changes. A is the family's `ml-verify` evidence (current for the source it
hashed), provenance for the checked-out code, lineage, output contract and a loadable bundle (inherited from a parent for a re-backtest). B adds the frozen plan with an untouched evaluation, applicable selection
controls, a sufficient null study, the realistic-execution evidence of Release 4 (`gates._execution_item`) and, from Release 5, the statistical diagnostics read against the adopted validation policy (`r5_gates.statistical_items`). C adds the prospective evidence of the current strategy version
(`r5_gates.prospective_items`) and its operating controls (`r5_gates.operational_items`). `Registry.promote` refuses `staging` without A and B, and `production` without A, B, C and a recorded human approval. Providers may return one item or several.

## Version changes (Release 3)

* Registry migrations 2 to 6 (additive; nothing existing is rewritten). A registry from before them opens, keeps every row and gains the tables.
* `run_experiment(..., evaluation=, study=, entry_point=, kind=)`; `evaluate_run(..., holdout=, lineage=)`; every saved report gains `holdout`, `lineage`, `reproducibility`, `trials.study`.
* `search(finalize_top=False)` and the API `finalize` default to off. `opt.finalize` with a registry needs an evaluation context. `explainability.feature_ic` needs `until`.
* `Registry.promote(model_id, stage, actor=, note=, approval=)` returns the decision and can raise `PromotionBlocked`.


# Release 4: executable trading and reconciled accounting

## What was reused

The vectorized engine (`src/backtest/engine.py`) is kept as the declared mode `vectorized_legacy`. The new mode drives components that already existed: the portfolio ledger (`src/ledger`: every fill, accrual and corporate action is a journal entry that obeys the journal identity), the instrument registry and
the futures, crypto and FX instruments, the event-driven stack's cost models (`src/engine/costs.py`, whose `ImpactModel` is the square-root law used here), and the engine's own lag, rebalance and volatility-target helpers (`src/backtest/execution.py`). No third engine, ledger or cost model was written;
`realism.py` holds the data structures and policies that did not exist (membership, corporate-action events, orders with states, borrow, marks, FX as-of lookups, contract specifications) and `execution.py` the adapter.

## Two engines, one declared mode

```
config: no `execution` section   -> vectorized_legacy   (BacktestEngine: a cost per unit of turnover, constant-gross drift between rebalances, no orders)
config: execution.mode = realistic_cash_equity -> orders -> fills -> ledger (same weights, same lag, same schedule, same volatility target)
```

The mode is part of the configuration (hashed only when present, so no existing hash changed) and of the report (`report["execution"]`). A realistic result is a re-run: `rebacktest(execution=...)` makes a child experiment with a new id and the parent's id, a counted look at the same study that
inherits the parent's holdout state. An experiment saved before the modes existed is read as `vectorized_legacy`.

## The adapter

```
close of day d:   the target weights, already masked to the investable set, scaled to the volatility target and thinned to the rebalance dates (the engine's own helpers), are the DECISION
                  an eligible asset's target is converted to shares at the decision close; an ineligible asset's target is zero (an addition is rejected on the record)
session d+lag:    the order (a target POSITION) becomes active; its size is re-set to the gap to that target; it is capped at the participation limit of the PRIOR average volume, rounded
                  to lots, filled at the close with the modelled spread, slippage and impact, and booked in the ledger: principal at the mid, each concession and the fee as separate entries
every session:    corporate actions at the ex-date; marks; interest and borrow accrue on what was held overnight; recalls force covers; unfilled remainders expire after `expiry_sessions`
```

The lag is `dataset.lag + backtest.extra_lag` and is applied once, here: `execution.delay` is refused as a setting, and a test shows the adapter and the vectorized engine produce the same return stream at lags 1 and 2 when costs and limits are removed (`sizing: fill_time` is that agreement
mode). Decisions that overlap in time (a lag of two sessions with a decision every day) queue per asset; a newer decision replaces what an older one left unfilled when it becomes active.

## Reconciliation

Gross P&L is computed independently from the share holdings and the marks, not from the ledger; gross plus the non-price flows (interest, corporate-action cash) less the itemised costs must equal the ledger's net change. The ledger's own `reconcile` (cash conservation, the identity of every entry) is run as well. In the tests a
shadow account books only cash and shares from the reported fills, and the corporate-action, borrow, multi-currency, futures-roll and funding fixtures are compared with hand-worked or closed-form arithmetic.

## Artifacts and the gate

`save_experiment` writes `execution_orders`, `execution_fills`, `execution_journal` and `execution_equity` (parquet) and `execution_manifest.json`: the row count and sha256 of each file, the hashes of the weights, prices and volume the run used, the specification and the reconciliation. The report cites the manifest's
own hash. Gate B's `realistic_net_oos` item reads it: a vectorized result is `blocked` (it cannot pass what it does not model), a realistic one is `failed` if a file no longer matches its hash or the accounting does not reconcile, `missing`/`stale` if the verification of the execution machinery (P1-01 to P1-11 and
P1-14, plus P1-08 when shorts were held) is not current, `blocked` by the capability blockers (no volume, shorts without borrow records, simulated or unknown data) and `ok` otherwise.

## Version changes (Release 4)

* New modules `src/ml/realism.py` and `src/ml/execution.py`; `config` gains the optional `execution` section; `rebacktest(..., bundle=, config=, builder=, market=)` and `run_experiment(..., market=)`; `evaluate_run(..., market=)`; `Experiment.execution`.
* Every new report gains `execution`. `GET /api/ml/trading-diagnostics`, `POST /api/ml/rebacktest` and `POST /api/ml/train` take the `execution` section; `quant ml-realistic`.
* Gate policy `gates-v2`: Gate B reads the execution evidence. Verification requirements P1-01 to P1-14.


# Release 5: statistical and prospective validation

## What was reused

The platform's own bootstrap and block-length code (`stats/inference.py`), its OLS with HAC errors and the Newey-West lag rule (`stats/regression.py`), the Release 3 selection layer, `rebacktest` (every re-run of the saved predictions is a child experiment and a counted look), the Release 4 adapter and ledger
(the equity of committed holdings is built by `execution.execute_weights`, not by a new accounting), the registry and its migrations, and the gate evaluator. New modules: `inference`, `benchmarks`, `regime_eval`, `calibration`, `sensitivity`, `nested`, `multiseed`, `policy`, `diagnostics`, `prospective`, `governance`, `r5_gates`, `r5_schema`.

## One output contract

Every statistical result is a `StatResult` (`inference.py`): name and version, hypothesis, sample identity (segment, count, first and last date, a hash of the values), frequency and units, risk-free convention, estimator, the dependence assumed, its settings (lags, block length), level, seed and replications,
the selection-family scope, and value, interval and p-value **only when the procedure was applied**; an eligibility (`applied`, `insufficient`, `unsupported`, `not_applicable`, `failed`) with a reason and the limitations. The three Sharpe statements share ONE point estimate: the naive iid standard error (and
what it ignores), the delta method on the moment conditions (x, x^2) with a Newey-West long-run covariance, and a stationary-bootstrap percentile interval (200 replications at least). A HAC standard error of a mean is never presented as a standard error of a Sharpe ratio. A horizon of h forces at least h-1 lags.

## The validation policy is a declaration

`policy.ValidationPolicy` is a name, a version and a set of optional criteria (the segment judged, the minimum days, the interval method, the lower bound the Sharpe interval must exceed, the deflated-probability, PBO and selection-p thresholds, the benchmark kind and the active-return bound, regime, neighbourhood,
delay, calibration and capacity requirements, the prospective minimums and the monitoring age). A criterion left unset is not required and is shown as not required; there are no hidden defaults, and the platform implies no threshold. The policy has a content hash; changing any threshold is a new policy. `example_policy()` is
labelled illustrative. `adopt` needs a named person and a note and is an append-only record whose latest row is the policy in force. Algorithm acceptance (Gate A) never reads it: a correct implementation passes Gate A whatever the policy says about profit.

## Diagnostics and Gate B

`diagnostics.run` computes, for one experiment and the policy's segment, the three Sharpe statements, the HAC mean, the incremental return against each matched benchmark (beta, net-exposure, sector and risk matched, each estimated from information known before the day, and the equal-weight market labelled descriptive and
unmatched), the regimes (causal rules and retrospective slices declared before the evaluation, with insufficient evidence stated per regime), calibration where the model produced class probabilities, the study's selection evidence and, when asked, the neighbourhood, delay and capacity analyses. It refuses a protected
segment of an experiment that no frozen plan evaluated. The record is appended to `analysis_records` with the policy hash. `evaluate_policy` turns a record and a policy into items (`ok`, `failed`, `missing`, `blocked`, `not_applicable`); `r5_gates.statistical_items` returns them to Gate B with the verification state of the
procedures themselves (P1-16 to P1-19, and P1-20 to P1-22 when the policy asks for them). Under another policy the record is `stale`. With no policy the item is `blocked`. A reinforcement-learning result additionally needs a multi-seed record.

## The prospective ledger

`prospective.commit` writes an immutable commitment under the key `strategy version | decision time | universe version | horizon`. An identical repeat is a no-op that is audited; different content under the same key is rejected, and the rejection is audited (the audit row is written before the error is raised). The creation
time is the server's. The ledger itself classifies the commitment: **prospective** only if it was written before the first session whose price the label needs, the data cutoff and the fit cutoff are at or before the decision, and the model artifact existed before the commitment; otherwise **replay**, with the reasons. Settlements are
appended as versions (a revision needs a reason; the original is kept) and counted separately for the two classes; `counts` never adds a replay to prospective evidence. The strategy's equity is built from the committed holdings by the Release 4 adapter on the daily prices (`equity`); forecast skill (a rank IC per decision) is a
different stream. `ForwardTracker.step` writes the commitment before it records anything as held; `settle` appends the outcomes; the old return/horizon compounding is removed.

## Governance

A *strategy version* (`governance.register_version`) is the identity of the model artifact together with the dataset, model, signal, execution and risk settings; the platform's code hash is recorded with each commitment, not in the identity. `commit_forecast` is the service function every entry point uses: it refuses a halted or
rolled-back version and a book that breaches a documented limit (gross, net, position, turnover, drawdown, daily loss, staleness) and records each refusal. `monitor` reads drift (PSI), the live rank IC and the limits on the committed holdings' returns; a hard breach halts the version. Halt, resume, rollback, the written recovery
procedure and the deployment approval are appended to `governance_events` with a name; the approval is tied to the version it covered and `Registry.promote` records it. Gate C reads all of it.

## Migrations 7 to 9

7 `analysis_records`, `validation_policies`, `policy_adoptions`; 8 `commitments`, `commitment_audit`, `settlements`; 9 `strategy_versions`, `risk_limits`, `governance_events`. Additive, with triggers that refuse updates and deletes of the append-only tables (`r5_schema.py`).

## Version changes (Release 5)

* Gate policy `gates-v3`. `GET/POST /api/ml/validation-policy`, `POST /api/ml/statistical-diagnostics/run`, `POST /api/ml/multiseed/run`, `GET /api/ml/governance`, `POST /api/ml/governance/version` and `/action`; `GET /api/ml/statistical-diagnostics` also returns the stored record read against the policy; `GET /api/ml/forward` returns the ledger; `POST /api/ml/forward/step` refuses unknown fields.
* `quant ml-policy`, `ml-diagnostics`, `ml-multiseed`, `ml-governance`, `ml-forward`.
* `selection.run_procedures` excludes candidates with a constant return stream and records the exclusion (a statistic built on them was a number without meaning); the DSR item judges the more conservative of the two trial conventions. The bootstrap needs 200 replications.
* `ForwardTracker.summary` no longer reports a cumulative return or Sharpe computed from horizon returns. Comparisons name each result's holdout status and execution mode.
* Verification requirements P1-15 to P1-25.

# Release 6: alpha research foundation

## What was reused

Targets go through the existing catalogue (`targets.make_target`, `dataset.target`), models through the model catalogue (`stable_select` is one more kind with a builder module), the position rule through `signals.build_weights`, the covariance estimators of `src/portfolio/covariance.py`, the impact law of the event-driven stack, `run_experiment` for every arm of a comparison and the append-only `analysis_records` table of Release 5 for every stored card, result and diagnostic. No new configuration subsystem was added: the new settings are keys of the existing `dataset`, `model` and `signal` sections, validated by `config.validate`, and the ones that are derived (the optimiser's cost and window) are set there from the `backtest` and `dataset` sections so that one place decides what a trade costs.

## Where the pieces live

| Concern | Module |
| --- | --- |
| relative and residual targets; target documentation | `targets.py` (`sector_ret_h`, `resid_ret_h`, `describe`) |
| rank IC, decay, coverage, stability, redundancy, correlation, conditional IC, blend | `alpha.py` |
| stability selection inside a fold | `featsel.py` (`StableSelect`, model kind `stable_select`) |
| covariance estimators compared on realised risk | `risk_models.py` |
| cost-aware allocation, its perturbation grid, partial fills carried forward | `allocation.py`; the pipeline method is `signals.cost_aware_mean_variance` |
| horizons and tranche accounting | `horizons.py`, `signals.holding_period` |
| the simplicity tie-break | `parsimony.py` |
| frozen cards, matched arms, paired verdicts | `comparison.py` |
| diagnostics of a saved experiment | `r6_reports.py` (`alpha_diagnostics`, `portfolio_diagnostics`) |

## The frozen comparison

`comparison.new_card` takes a baseline and *changes* laid over it, and refuses a change that alters anything the family does not allow (`diff_paths`) or alters nothing. `freeze` stores the card append-only; `run` runs the baseline and each candidate once through `run_experiment` with `stop_after="validation"` (the test and forward outcomes are not predicted), checks that the data fingerprint, the assets and, for the families that need it, the fold boundaries are the baseline's, and compares each candidate on the dates both have. The primary metric is either the paired mean difference of the daily rank IC against the **one raw forward return of the baseline's horizon** (a Newey-West interval whose lag is at least the horizon less one) or the paired Sharpe difference of the daily net returns (a joint stationary bootstrap). The verdict needs the Holm-adjusted p-value over the card's candidates below the level, an interval that excludes zero and an effect at least the declared minimum; `inconclusive` and `worse` stay in the record with every other negative finding. The gate decision on a card is always `not_evaluated`: a development comparison is not a frozen final evaluation.

## Allocation in the pipeline

`cost_aware_mean_variance` reads the model's score as an expected return over the dataset's horizon, fits a covariance as of each decision date (only returns up to and including it), scales it to the same window, and solves `allocation.solve` every `opt_every` days from the book actually held. It is allowed only for an `expected_return` output on a return-unit target (absolute targets may carry a net exposure; relative ones are held dollar-neutral), holds the book between solves and counts every solve in `report['allocation']`: optimal, infeasible, failed, no-trade, repaired covariances and fallbacks. A solve that gives no book holds the previous book and is counted, never presented as an optimum.

## Version changes (Release 6)

* New target kinds `sector_ret_h` and `resid_ret_h`; `HOLDING_PERIODS` is 1, 5, 10, 20; model kind `stable_select`; signal method `cost_aware_mean_variance` with optional `risk_aversion`, `cov_method`, `cov_window`, `opt_every`, `turnover_penalty` and `alloc_fallback` (absent from the defaults, so no existing configuration hash changes) and derived `cost_bps` and `window_days`.
* `GET /api/ml/alpha-diagnostics`, `GET /api/ml/portfolio-diagnostics`, `GET` and `POST /api/ml/improvement-experiment`; `quant ml-alpha`, `ml-portfolio`, `ml-improve`.
* No registry migration: cards, results and diagnostics are `analysis_records` of new kinds.
* Verification requirements P2-01, P2-02, P2-03, P2-04, P2-07, P2-10, P2-13, P2-14, P2-16, P2-17, P2-18.


# Release 7: controlled strategy-improvement experiments

## What was reused

The position rules are still built by `signals.build_weights`; the new rules are overlays that run on the book it returns, with the same covariance estimators (`risk_models`), the same costs and the same engine. Every arm of a comparison is a `run_experiment`, or a `rebacktest` of the baseline's saved predictions when the change acts only after the forecasts exist (the position rule, the schedule, the costs, the execution), so that the two alternatives are evaluated on the same forecasts, the same folds and the same risk model. Comparison cards, results and reports are `analysis_records` of new kinds; no migration was added.

## Where the pieces live

| Concern | Module |
| --- | --- |
| no-trade regions: utility of the held book against the proposed one, a safety margin, hysteresis, hard controls that override with reasons | `notrade.py` |
| the net-edge trade filter: a forecast-error scale from settled decisions, coverage and rejection log | `tradefilter.py` |
| sizing from a calibrated forecast and its uncertainty | `sizing.py` (signal method `calibrated_sizing`) |
| causal regime exposure: filtered states numbered by volatility, a budget per state from the unscaled book's own past | `regime_exposure.py` |
| regularised stacked ensemble weights, member coverage | `ensemble_models.py` (`stacked`, `stacked_regime`) |
| mixtures of strategy streams against the best single member chosen on development | `ensembles.py` (`mixture_study`) |
| meta-labels with the declared cost, the base-plus-filter strategy against its base | `ensembles.py`, `comparison.compare_saved` |
| IC, edge, net result and capacity at declared delays | `delay_report.py` |
| the families, presets, combinations, ablations and the development-selected baseline | `comparison.py` |

## Overlays and their order

`build_weights` applies, after the book is built and averaged over the holding period: the trade filter (`trade_filter_k`), then the no-trade region (`no_trade_margin`), then the regime exposure (`regime_exposure`). The first two weigh an expected return against a cost and therefore need a return forecast (`contracts.check_signal_use` refuses a rank or probability output). The regime exposure is last because it is a risk control on the whole book: a risk cut must not be vetoed by a rule that weighs return against cost. It multiplies the book by a number between its floor and its cap (at most 1), changes only when the filtered state changes, and so cannot lever a book past the limits it was built within. The in-sample book of an experiment is a display and runs no overlay, and a forward test refuses every rule that needs state (`monitoring.ForwardTracker`).

## Combinations and ablations

`comparison.combination_card` builds a card from components that were each run on their own: the candidate is their changes laid together (two components that set the same setting differently are refused), its allowed paths are the union of the components' families, and its Holm family includes the components' own tests, so a combination is another logged candidate and not a free reuse of the best findings. `ablation_card` takes the full combination as the baseline and removes one component at a time (its settings return to the original baseline's; a setting the baseline never had is switched off); a candidate that is `worse` says the removed component helps. Both record their components in `lineage`.

## Version changes (Release 7)

* Signal settings (absent from the defaults, so no existing configuration hash changes): `no_trade_margin`, `no_trade_hold_margin`, `trade_filter_k`, `trade_filter_cost_margin`, `trade_filter_window`, `calib_window`, `calib_min_t`, `stability_window`, `regime_exposure`, `regime_states`, `regime_floor`, `regime_cap`, `regime_refit_every`, `regime_min_days`, `regime_min_train`; the signal method `calibrated_sizing`; the derived settings `cost_bps`, `window_days`, `label_offset`, `rebalance` and `lag_total` are filled in by the configuration only when a rule needs them.
* Ensemble weightings `stacked` and `stacked_regime` with `shrink` and `min_coverage`; a member below the coverage floor is left out of every weighting.
* Comparison families `rebalancing`, `risk_control`, `ensemble`, `execution` and `combination`; `meta_label_scores` takes a `cost`.
* `GET /api/ml/delay-report`, `development_end` on `POST /api/ml/combine`, the families and `members`, `select_baseline`, `combination_of`, `ablation` and `options` on `POST /api/ml/improvement-experiment`; `quant ml-delay`.
* Verification requirements P2-05, P2-06, P2-08, P2-09, P2-11, P2-12, P2-15.

