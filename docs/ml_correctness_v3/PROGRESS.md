# ML v3 implementation progress

Current release: **2 — causal data and training integrity** (delivered in part; the traceability table says, row by row, what is and is not done). Release 1 (numerical and model correctness) was delivered earlier, also in part. Releases 3 to 8 are `not_started`.
Checkout: branch `claude/multi-asset-quant-platform-uwqect`. Release 1 is commit `ae5a041` (CI run 100: every job green). Release 2 part 1 is `dad795e` (CI run 101: the `test` and `docker` jobs failed on seven failures in the new session and point-in-time tests, all one
cause, described below and fixed in the closing commit of this release). Source tree hash of the code each check hashed: the `subject_hash` of the check in the registry (`ml-verify --json`).
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

## Actual test runs

| Command | Environment | Exit code | Passed/failed/skipped | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| `python -m src.cli ml-verify --run --require-pass --registry <tmp>` (Release 2, final tree) | Python 3.11.17, pandas 3.1.0, numpy 2.4.6, torch 2.14.1 (CPU), scikit-learn 1.9.1, statsmodels 0.15.0, hmmlearn 0.3.3, Gymnasium 1.4.0, SB3 2.9.2, LightGBM 4.7.0, XGBoost 3.2.0 (a copy of the Python 3.11 CI environment) | 0 | 23 of 23 requirements `pass`; 364 tests in 18 files, 0 skipped, 0 failed, 144 s; none stale | registry `verification/*.xml` (JUnit), `verification_checks` rows | per requirement: P0-13 53, P0-14 61, P0-15 and P0-16 55, P0-17 30, P0-18 33, P0-19 7, P0-20 20, P0-21 46, P0-22 27 (the Release 1 ones as before) |
| ML regression group: `tests/test_ml_*.py`, `test_deep_models.py`, `test_webapp_assets_ml.py`, `test_futures_bundle_exact.py`, `test_futures_real_backtest.py`, `test_assets.py`, `test_models_api.py`, `test_online_model.py`, `test_optional_packages.py`, `test_hpo_search.py`, `test_webapp_ml_models_browser.py`, `test_docs.py`, `test_tuning.py`, `test_tune_api.py`, `test_tune_study.py`, `-n 3` | same | 0 | 922 passed, 0 failed, 2 skipped | pytest output | the two skips are the Chronos and TimesFM foundation models (weights not available offline), which this programme does not touch |
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
| every model kind | pending | not started | not started | Gate A needs the Release 3 research-control evidence (study lineage, holdout reservation, provenance hashing) as well as the Release 1 and 2 checks; those checks are inputs to it, not the gate. No gate evaluator exists yet. |

## Compatibility and migration

- Existing experiments, registry rows and model files are untouched. Experiments saved before this release have no fold lineage, no dataset audit and no inference bundle, and say `not recorded`; they still load, re-backtest and track forward.
- `split.sample_weighting` and `dataset.sessions` are absent from the defaults, so no existing configuration hash changes; setting either makes a different experiment.
- Training rows are chosen by label interval. On a calendar without missing dates this selects exactly the rows the old position rule did (tested); where dates are missing it keeps rows the old rule dropped and still leaks nothing.
- A panel built for another target, horizon, lag, scaling or feature set than the configuration asks for is refused instead of trained on.
- For a universe whose assets close at different times (a crypto day against a US close), the default is now to read the later-closing assets as of the decision; `dataset.sessions.policy: date_labels` reproduces the earlier join.
- Strategy combination by online learning no longer back-fills (see above); `params.volatility_fill` records `causal_v2`.
- Release 1's items (RL state schema 2, output contracts, the `QFM1` envelope, the registry migrations) are as described in the Release 1 section of the architecture note.

## Data and optional dependency limitations

See `ASSUMPTIONS_AND_LIMITATIONS.md`.

## Next executable actions

1. Release 3: study lineage for every entry point, full provenance hashing, the holdout access ledger, the null-control runner, the selection-test eligibility layer and the gate evaluator.
2. Finish the Release 2 gaps listed against P0-13, P0-15 and P0-16 in `TRACEABILITY.md`: intraday and non-close label conventions, a macro vintage store, macro release instants in the loader, and the per-ticker session table beyond its defaults.
3. Give returns at a missing price one explicit definition (`pct_change()` pads on pandas 2 and drops the move on pandas 3; see the limitations note), in `data/loader.py` and `framework/data.py`, and run the whole suite on both.
4. Finish the Release 1 gaps listed against P0-01, P0-02, P0-04, P0-06, P0-08, P0-09, P0-10, P0-11, P0-12 and P0-30.
5. A Model Verification view in the dashboard (the API and CLI exist; the page does not), and a dashboard view of the dataset audit and fold lineage (the report carries them; the page does not show them yet).
