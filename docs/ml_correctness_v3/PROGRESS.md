# ML v3 implementation progress

Current release: **3 — research controls, reproducibility and protected holdouts** (delivered in part; the traceability table says, row by row, what is and is not done). Release 1 (numerical and model correctness) and Release 2 (causal data and training
integrity) were delivered earlier, also in part. Releases 4 to 8 are `not_started` at this commit.
Checkout: branch `claude/multi-asset-quant-platform-uwqect`. Release 1 is commit `ae5a041` (CI run 100: every job green). Release 2 is `dad795e` and the closing `31e5cd7` (CI run 103: every job green). Release 3 is the commit that follows `d7fc4aa`, which
recorded the Release 2 CI result; its own CI result is recorded in the next release's progress note once the run completes. Source tree hash of the code each check hashed: the `subject_hash` of the check in the registry (`ml-verify --json`).
Updated: 2026-11-01.

## What exists now

### Release 1 (numerical and model correctness)

- **RL portfolio ledger and environment** (`src/ml/rl/ledger.py`, `env.py`, `agent.py`): a quantity-and-cash ledger, one state transition for training and for playing a policy forward, an event timeline with pending orders, a corrected drawdown
  penalty, a stated differential-Sharpe convention, rejected/clipped/forced actions with reasons, mandatory eligibility liquidation and hard position/exposure limits that outrank the optional turnover budget, ruin handling, and a state schema
  version (2) so that agents trained under the old state are refused with a retrain message.
- **Rankers** (`boosting.py`): query groups and relevance recomputed after the label filter; excluded dates and reasons recorded.
- **Graph networks** (`graph.py`): absent nodes are removed from both ends of every edge at every layer (GCN degrees, GAT keys and queries, biases, residuals); the graph's cutoff, window, asset order and relationship version are recorded.
- **Output contracts** (`contracts.py`): typed output kinds, a position-rule/output-kind compatibility check that runs before training and again on re-backtest, `signal.reinterpret_outputs` as the one explicit override, validators for
  probabilities, quantiles (with the raw crossing rate), predictive distributions and pinball loss, and `report["output_contract"]`.
- **Serialization** (`models.py`): a checksummed `QFM1` envelope (legacy artifacts still load), `ModelArtifactError` for damaged files, `ModelSchemaError` for missing features.
- **Architecture manifest** (`deep/manifest.py`) with property tests; reduced variants are labelled as reduced.
- **Registry migrations and verification evidence** (`registry.py`, `verification.py`, `webapp/ml_verify.py`): additive versioned migrations, an append-only `verification_checks` table, `ml-verify` on the command line and
  `GET /api/ml/verification` / `POST /api/ml/verification/run` in the API. Only a run of the tests writes a status.

### Release 2 (causal data and training integrity)

- **Label intervals** (`intervals.py`): the label of a row starts at the close of `t + lag` and ends at the close of `t + lag + horizon`; a model fitted at the close of `D` trains on the rows whose label ended by `D` (less the embargo), counted on the real
  calendar. Per-row intervals are persisted with the dataset (`label_intervals.parquet`); the report's fold list carries `fit_cutoff`, `label_end_max` and a hash of the training rows. Purged K-fold and CPCV on the same intervals exist as research diagnostics.
- **Restricted views** (`causal.py`): every family that reads history is bound to a `CausalView` (truncated data, sealed labels); a fit view ends at the first decision of a block, a predict view at the block's last day; `predict_latest` refuses to replay.
- **Feature and dataset audit** (`audit.py`): measured dependencies of all 85 features, the dataset audit stored in `meta['audit']` and the report (label cutoffs, exclusions by reason, stale runs, closed-market marks, point-in-time quality, session policy).
  Closed-market marks are not tradable prices. Two back-fills that used later data (strategy combination by online learning, segment labels) were removed.
- **Point in time and sessions** (`pit.py`, `sessions.py`): a published series joins on when each value became public (the fundamentals engine's rule), macro inputs carry a quality (`observed`, `estimated`, `unavailable`) and claims on the weaker ones are
  marked restricted; a daily bar is final at its market's close on the real clock, a decision sees only bars that were final, and a lag that would put an entry price before the decision is refused.
- **Lineage** (`training.py`, `ensemble_models.py`, `ensembles.py`): `TrainResult.producers()` and `oos_violations()`, the ensemble's two-stage cross-fitting record, meta-labelling's settled-label and producer-fold record.
- **Overlapping labels** (`training.dependence_weights`, optional `split.sample_weighting`): concurrency, average uniqueness and weights; the report records the treatment and whether the model used the weights.
- **Inference bundle** (`bundle.py`, `model.qfb` beside each experiment's model): schema (names, order, dtype kinds), preprocessing description, cutoffs, label convention; strict drift refusal, declared versioned adapters; the forward tracker uses it and
  flags signals for dates the saved model had already seen.
- **Surfaces**: `ml-report` prints fold lineage, the dependence treatment and the dataset audit; `ml-build-dataset` prints the audit; `/api/ml/experiment` and `/api/ml/dataset/build` return it.
- **Verification registry**: 23 requirements (P0-01 to P0-22 and P0-30) with their test files and the source files each hashes; the new test files are in the `ml-base` and `replit-python` CI lists and run in the blocking `ml-correctness` lane.

### Release 3 (research controls, reproducibility and protected holdouts)

- **Study lineage** (`research_controls.py`, `registry.py` migration 2): every entry point that looks at outcomes (a run, a tuning trial, a strategy-search candidate, an ensemble, a meta-label, a re-backtest, a feature search, an evaluation) is registered as an operation of a
  study **before it runs** (`registered` -> `running` -> `done | failed | cancelled`). Failed and cancelled operations are looks; an enclosing operation adopts the experiment it starts instead of counting it twice; the default study is the universe (`u-<universe>`), not a cache key,
  so a renamed experiment, a different feature set or a changed cache key does not start a new count; a deleted experiment leaves a tombstone that `trial_count` still counts. `lineage()` reports the raw looks next to the count of operations that `count`; no estimate of
  *independent* trials is made or implied.
- **Provenance** (`provenance.py`, migration 3): content hashes of the prices, every input table (macro, fundamentals, universe), the feature panel and the label intervals, a source-tree hash, the configuration, the seed, the fold list and the output contract, combined into an
  identity. The environment and hardware are recorded but are not part of the identity; hashing is independent of the datetime unit; when no commit is available the record says so. `report["reproducibility"]`, `experiments/<id>/provenance.json` and an append-only `provenance` table
  carry it, and `compare` / `reproducible` check a repeat run against the first within the family tolerance.
- **Protected holdouts** (`frozen_eval.py`, `research_controls.py`, migration 4): `stop_after="validation"` development runs that never read the test or forward outcomes (`guard_development`); an immutable evaluation plan; a reservation by assets, dates and the hash of the outcome content (a
  renamed asset with the same outcomes is caught); a freeze; and **one** evaluation event per plan (`begin_evaluation`: single transaction, idempotency key, `in_progress`, `resume`, an explicit `allow_reuse`). Every access is written to an append-only ledger. A direct run on an
  unconsumed reservation is refused; a run after consumption records an *exposure* and its report says `exposed_exploratory`. Re-backtests, meta-labels and combinations inherit the protection of their parents (`inherit_holdout`). The legacy `finalize=true` is routed through the same
  path with a deprecation text, and `finalize` is now off by default in the API and the search.
- **Selection-test eligibility** (`selection.py`, migration 5): the deflated Sharpe ratio, the CSCV probability of backtest overfitting, White's Reality Check, Hansen's SPA and Benjamini–Yekutieli false-discovery control, each with a state (`applied`, `not_applicable`, `unsupported`,
  `insufficient`, `failed`) and a reason; a procedure that was not applied is never reported as passed. The feature screen's adjustment is Holm's. Evidence rows are append-only.
- **Null-control pipeline study** (`null_controls.py`, migration 5): the whole search pipeline is run again on placebo data (returns circularly shifted so the dependence on the features is destroyed, targets recomputed), with a frozen design hash, exact Clopper–Pearson and Wilson
  intervals for the false-promotion rate, the verdicts `sufficient` and `controlled`, resumable replications, and a plumbing-only label for small runs. `.github/workflows/ml-heavy.yml` runs it weekly with 200 replications (minimum 100) and keeps the evidence.
- **Gate evaluator** (`gates.py`, migration 6): Gate A (algorithm verified), Gate B (strategy validated) and Gate C (operationally ready) are computed from persisted evidence only. Gate A reads current `ml-verify` rows for the source hash, the provenance record, the lineage, the
  output contract, the bundle (inherited from the parent for re-backtests), a frozen-plan evaluation of an untouched holdout, the selection evidence and the null study. The evidence that later releases supply (execution realism, statistical validation, prospective operation) has a
  placeholder provider whose answer is `blocked`. Decisions are appended to `gate_decisions`, `Registry.promote` enforces them (staging needs A and B; production needs A, B and C and a recorded human approval), and a request cannot supply a trusted status.
- **Surfaces**: `GET /api/ml/lineage`, `/api/ml/holdout`, `/api/ml/promotion-status`, `/api/ml/statistical-diagnostics`, `/api/ml/null-controls`; `POST /api/ml/evaluation-plan`, `/api/ml/evaluation-plan/freeze`, `/api/ml/evaluate`, `/api/ml/selection-tests/run`,
  `/api/ml/null-controls/run`; `POST /api/ml/train` takes `mode` (`full`, the default, records an exposure, or `development`); `quant ml-lineage`, `ml-plan`, `ml-evaluate`, `ml-holdout`, `ml-promotion`, `ml-promote`, `ml-selection-tests`, `ml-null-controls`; the experiment page shows the
  holdout state, the provenance identity and the lineage count.
- **Verification registry**: 31 requirements (P0-01 to P0-30 and P1-26) with their test files and source groups; the Release 3 test files are in the `ml-base` and `replit-python` CI lists and run in the blocking `ml-correctness` lane.

## Actual test runs

| Command | Environment | Exit code | Passed/failed/skipped | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| `python -m src.cli ml-verify --run --require-pass --registry <tmp>` (Release 3, final tree) | Python 3.11.17, pandas 3.1.0, numpy 2.4.6, torch 2.14.1 (CPU), scikit-learn 1.9.1, statsmodels 0.15.0, hmmlearn 0.3.3, Gymnasium 1.4.0, SB3 2.9.2, LightGBM 4.7.0, XGBoost 3.2.0 (a copy of the Python 3.11 CI environment) | 0 | 31 of 31 requirements `pass`; 456 tests in 27 files, 0 skipped, 0 failed, 239 s (sum of the per-file times); none stale | registry `verification/*.xml` (JUnit), `verification_checks` rows | per requirement: P0-23 35, P0-24 9, P0-25 35, P0-26 7, P0-27 12, P0-28 34, P0-29 29, P1-26 22 (the earlier ones as before) |
| ML regression group (the files listed in the Release 2 row below), `-n 3` | same | 0 | 1,025 passed, 0 failed, 2 skipped (counted from the progress output; the summary line is not printed under `-q` with workers) | pytest output | the skips are the Chronos and TimesFM foundation models (weights not available offline) |
| `python -m src.cli ml-verify --run --require-pass --registry <tmp>` (Release 2, final tree) | Python 3.11.17, pandas 3.1.0, numpy 2.4.6, torch 2.14.1 (CPU), scikit-learn 1.9.1, statsmodels 0.15.0, hmmlearn 0.3.3, Gymnasium 1.4.0, SB3 2.9.2, LightGBM 4.7.0, XGBoost 3.2.0 (a copy of the Python 3.11 CI environment) | 0 | 23 of 23 requirements `pass`; 364 tests in 18 files, 0 skipped, 0 failed, 144 s; none stale | registry `verification/*.xml` (JUnit), `verification_checks` rows | per requirement: P0-13 53, P0-14 61, P0-15 and P0-16 55, P0-17 30, P0-18 33, P0-19 7, P0-20 20, P0-21 46, P0-22 27 (the Release 1 ones as before) |
| ML regression group (Release 2): `tests/test_ml_*.py`, `test_deep_models.py`, `test_webapp_assets_ml.py`, `test_futures_bundle_exact.py`, `test_futures_real_backtest.py`, `test_assets.py`, `test_models_api.py`, `test_online_model.py`, `test_optional_packages.py`, `test_hpo_search.py`, `test_webapp_ml_models_browser.py`, `test_docs.py`, `test_tuning.py`, `test_tune_api.py`, `test_tune_study.py`, `-n 3` | same | 0 | 922 passed, 0 failed, 2 skipped | pytest output | the two skips are the Chronos and TimesFM foundation models (weights not available offline), which this programme does not touch |
| `ruff check src tests experiments`; `mkdocs build --strict` | same | 0 | clean | | |
| the same two commands on pandas 2.3.3 (the default interpreter of the development container, not what CI uses) | Python 3.11.17, pandas 2.3.3 | 0 | 23 of 23 `pass`, and 728 passed with the same 2 skips, on earlier trees of this release | | kept for the record; the pandas 3 runs above are the ones that count, and the pandas 2 runs are the ones that missed the datetime-unit defect |

CI results are recorded when a run completes. Run 100 (Release 1): every job green. Run 101 (Release 2 part 1, `dad795e`): the `test` job reported 7 failed, 5,277 passed, 7 skipped (coverage 93.66%, floor 88%) and the `docker` job failed in the same tests; all
seven are `tests/test_ml_pit_sessions.py`, below. Run 102 (`9658b52`) was cancelled by the next push (the workflow cancels a superseded run). **Run 103 (`31e5cd7`, the closing commit of Release 2): all 10 jobs succeeded** — `lint`, `docs`,
`ml-correctness`, `ml-base`, `replit-python`, `browser`, `compat (3.10)`, `compat (3.12)`, `docker` and `test`. The `test` job: 5,371 passed, 7 skipped, 0 failed in 42 min 32 s, coverage 93.71% (floor 88%). This is a CI result for the whole suite;
what it shows about Release 2 is that the new checks pass on the CI environment (Python 3.11.17, pandas 3.1) as they did locally, not that the requirements hold beyond those tests.

### What running the checks found (and fixed)

- **pandas 3 datetime units.** `DatetimeIndex.asi8` counts in the index's own unit, and pandas 3 picks that unit by what it parsed (microseconds here). The session and point-in-time code treated the integers as nanoseconds, so a bar's age came out 1,000
  times too small and the availability join saw nothing public. CI found it (the seven failures above); my first local runs used pandas 2.3.3 and missed it, and the brute-force oracle I had written for the join shared the same flaw and passed vacuously.
  Now every comparison goes through one nanosecond conversion (`sessions.ns`), the oracle is independent of it, and every local gate runs on a pandas 3.1 copy of the CI environment.
- **A defect in the regime model's diagnostics**, found by the training-isolation test: the regime statistics it reported as "training" included the block it was then asked to predict. They now stop at the fit window.
- **Strategy combination by online learning** back-filled the first volatility from later days and filled gaps from the whole sample (found by the fill audit). It now makes no update until 20 days of trailing volatility exist; results of earlier runs are not re-evaluated.
- **`idio_vol_63` declared a 63-day lookback and needs 126** (found by the dependency audit).
- **Saved reports carried too few fold fields**: the first regression run failed on `KeyError: 'fit_cutoff'` when meta-labelling a run saved without lineage. Runs saved before this release now say `not recorded`.
- **A feature dtype check that numpy cannot make**: pandas 3's text dtype has no numpy equivalent, so the first version of the model's dtype safety net skipped it. It reads the dtype kind from pandas.

## Gate evidence

| Model / strategy version | Algorithm Verified | Strategy Validated | Operationally Ready | Reasons |
| --- | --- | --- | --- | --- |
| every model kind | computed per experiment by `gates.decide` from current evidence: it can pass only with current `ml-verify` rows for the source it ran on, a provenance record, lineage, the output contract, the inference bundle, an evaluation of an untouched holdout under a frozen plan, applied selection evidence and a null-control study | `blocked` | `blocked` | Gate B needs the execution-realism evidence of Release 4 and the statistical validation of Release 5; Gate C needs the prospective evidence of Release 5, risk limits and the monitoring of Release 8. None of those providers exists yet, so the evaluator answers `blocked` rather than guessing. No real-market experiment has been evaluated: the tests exercise the evaluator on simulated data, and the 200-replication null study has not yet run. |

## Compatibility and migration

- Existing experiments, registry rows and model files are untouched. Experiments saved before this release have no fold lineage, no dataset audit and no inference bundle, and say `not recorded`; they still load, re-backtest and track forward.
- `split.sample_weighting` and `dataset.sessions` are absent from the defaults, so no existing configuration hash changes; setting either makes a different experiment.
- Training rows are chosen by label interval. On a calendar without missing dates this selects exactly the rows the old position rule did (tested); where dates are missing it keeps rows the old rule dropped and still leaks nothing.
- A panel built for another target, horizon, lag, scaling or feature set than the configuration asks for is refused instead of trained on.
- For a universe whose assets close at different times (a crypto day against a US close), the default is now to read the later-closing assets as of the decision; `dataset.sessions.policy: date_labels` reproduces the earlier join.
- Strategy combination by online learning no longer back-fills (see above); `params.volatility_fill` records `causal_v2`.
- Release 3 adds migrations 2 to 6 (lineage, provenance, holdout, selection evidence and null studies, gate decisions). They only add tables and triggers; a registry written before them opens in place, keeps every row (tested), applies nothing twice, and a migration that fails halfway leaves
  no trace. Experiments saved earlier have no lineage, provenance or holdout record and say `not recorded`; the lineage of a study that holds only such experiments shows zero looks (none is invented), and `trial_count` is unchanged for them.
- **Defaults that changed** (documented in the limitations note): `finalize` is off in the API, the search and the CLI (the old behaviour is `finalize=true`, which now freezes a plan and consumes it with a deprecation text); `optimization.finalize` and `explainability.feature_ic` raise when a
  registry is in use and no horizon or evaluation context is given; `Registry.promote` blocks staging and production without gate evidence (taking a model out of production needs none); `POST /api/ml/train` defaults to `mode: full`, which records a holdout *exposure*, not to `development` — a deliberate
  deviation from "development by default", so that existing callers keep their results, with the exposure visible in the report.
- Release 1's items (RL state schema 2, output contracts, the `QFM1` envelope, the registry migrations) are as described in the Release 1 section of the architecture note.

## Data and optional dependency limitations

See `ASSUMPTIONS_AND_LIMITATIONS.md`.

## Next executable actions

1. Release 4: executable trading and reconciled accounting (P1-01 to P1-14): dated membership, corporate actions, point-in-time liquidity, the decision-to-fill timeline, itemised costs, impact, order states, borrow, participation and capacity, mark policies, the shadow ledger,
   currencies, futures and funding, and the capability report; then the provider that lets Gate B read execution evidence.
2. Release 5: nested selection, dependence-robust inference, benchmarks, regimes, robustness, calibration, multi-seed RL, prospective commitments and the Gate C policy (P1-15 to P1-25); finish the cross-study holdout tracking (P1-26) for exports and comparison paths.
3. Run the 200-replication null study in Actions (`ml-heavy`) and record its verdict; until then no error-control claim is made.
4. Finish the Release 2 gaps listed against P0-13, P0-15 and P0-16 in `TRACEABILITY.md`: intraday and non-close label conventions, a macro vintage store, macro release instants in the loader, and the per-ticker session table beyond its defaults.
5. Give returns at a missing price one explicit definition (`pct_change()` pads on pandas 2 and drops the move on pandas 3; see the limitations note), in `data/loader.py` and `framework/data.py`, and run the whole suite on both.
6. Finish the Release 1 gaps listed against P0-01, P0-02, P0-04, P0-06, P0-08, P0-09, P0-10, P0-11, P0-12 and P0-30.
7. A Model Verification view and a dataset-audit / fold-lineage view in the dashboard (the API and CLI exist; the pages do not show them yet), and a promotion-gate view beyond the status line on the experiment page.
