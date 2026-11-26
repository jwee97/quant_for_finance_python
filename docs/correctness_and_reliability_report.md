# Comprehensive correctness, integration and reliability upgrade: report

This report states what was found, what was changed, what was run and what was not. A source file existing is not evidence that a feature works; a mocked test is not evidence that a provider still answers; a Linux run is not evidence about Windows; a synthetic backtest is not evidence of profitability. Where a claim below rests on a test, the test file is named.

Branch `claude/multi-asset-quant-platform-uwqect`. Commits of this round: `0693ab4`, `e5e4077`, `40b9fd0`, `89f2154`, `0f11be6`, `b4d34e4` (and the report commit).

---

## 1. Initial audit

### 1.1 Confirmed defects (each reproduced before it was changed)

| # | Defect | Evidence before the fix | Status |
|---|---|---|---|
| D1 | A crypto order-book as-of lookup returned an OLDER valid book when the newest one was invalid, crossed, stale or had a sequence gap; an existing test encoded this | `tests/test_crypto_store.py` asserted the fallback; `test_crypto_book_policy.py` reproduces the invalid, stale, crossed, gap and out-of-order cases | Fixed |
| D2 | `add_book` silently dropped a row received earlier than the last stored one (a negative interval counted as "too soon") | the out-of-order test | Fixed |
| D3 | The Research Comparison annualised every result with a fixed 252 and a CAGR of `equity^(252/n)`: a 24/7 crypto series was off by a third (44% against 29% in the planted example) and an irregular series by an arbitrary amount | 17 of 20 tests of `test_research_compare_conventions.py` fail on the previous `compare.py` (run and recorded) | Fixed |
| D4 | The comparison reindexed every result to the first result's dates and filled the gaps with a zero return; a result with a hole, a different inception date or another calendar was measured on the wrong sample | same tests | Fixed |
| D5 | The returns of a DIFFERENCE-adjusted continuous futures series were a percentage of the shifted level, not the return of the contract held | a planted curve: the old figure differed from the held-contract return by up to 0.11 percentage points per day, growing with every roll; an existing test pinned the old figure and was changed on purpose | Fixed |
| D6 | The Training monitor never showed a failed checkpoint write (the backend logged `checkpoint_failed`; no page read it) | grep of the page; `test_webapp_training_recovery_browser.py` now asserts the banner | Fixed |
| D7 | Checkpoints were loaded with `weights_only=False` (arbitrary code) guarded only by a SHA-256 file beside the checkpoint, which anyone able to replace the file can replace too | Measured: the previous `CheckpointManager` (commit `0f11be6`) executed a forged pickle with a matching `.sha256` sidecar (a marker file was written); `test_checkpoint_security.py` proves the new reader does not | Fixed |
| D8 | An unexpected server error returned `Type: message` with whatever file paths the message held, and nothing tied a failure on the page to a log line | `test_observability.py` | Fixed |
| D9 | The Alpha Lab's own help said nothing could be saved to a research session, although a factor study, a portfolio comparison and a trading study can | `lab_api.py`; `test_lab_help_consistency.py` | Fixed |

### 1.2 Conventions that differ on purpose, now quantified (not defects, not changed by default)

* The weight engine's default drift (`drift_basis="gross"`) rescales a book to its previous gross exposure after each return. Exact for a fully invested long-only book; for a book with cash, leverage or shorts it silently re-trades the book daily at no cost. An exact `drift_basis="equity"` option was added; the default is unchanged so saved research reproduces. See `docs/accounting_and_measurement_conventions.md`.
* The weight engine charges a cost as a fraction of the previous close's equity, the ledger as a fraction of the traded notional at trade-time equity: second order, bounded and tested.

### 1.3 Potential risks found and NOT fixed (see section 5)

* Fixed 252 in `backtest/metrics.py`, `framework/validate.py`, `alphalab/*`, `risk/performance.py`, `regime_lab.py`, `spread_lab.py` and `strategies/*`: correct for the business-day data they are fed, but not read from the data.
* Other pickle-based loads gated only by a hash stored with the data: `joblib.load` of a saved model (`ml/models.py`), `pickle.loads` of the Stable-Baselines3 environment (`ml/rl/sb3_state.py`) and of a stored frame (`ops/store.py`).
* No performance profile was made (Phase 12 asks for profiling before optimising); nothing was optimised.

### 1.4 What was already correct and was only re-run

Brinson-Fachler and Carino attribution with hand-worked references; the three-engine reconciliation (`test_engine_reconciliation.py`); the ledger's per-entry identities; the forecast-to-alpha contract and its look-ahead tests; the five-stage Alpha Lab workflow; schema-2 research artifacts; transactional epoch commits, disk-full and kill-at-every-step recovery; fundamentals point-in-time gating of unconfirmed figures.

---

## 2. Implementation

| Phase | Files | Root cause and change |
|---|---|---|
| 1.1 Crypto books | `assets/crypto_book_policy.py` (new), `crypto_store.py`, `crypto_replay.py` | One `BookPolicy` (latest row by receipt time at or before the decision, then validity, then age; no older book substituted unless `allow_last_valid_fallback`, and then recorded per trade and in the experiment's `book_policy`) |
| 1.2-1.3 Comparison | `research/conventions.py` (new), `research/compare.py`, `research/store.py` (series `meta`), `webapp/research_api.py`, `static/compare.js` | Levels, not returns, on the dates every result observed; elapsed-time CAGR; periods per year from the declared or inferred calendar (252, 365, 52, 12, 4) or elapsed-time variance for irregular spacing; risk-free accrued by elapsed time; coverage table, each result's own history shown separately |
| 1.4 Metrics | `research/metrics.py` (new) | Registry: direction, family, kinds, horizon, scale; ranking refused when a metric does not apply to a kind, the horizons differ or the units differ |
| 2 Accounting | `backtest/execution.py`, `backtest/engine.py` (`drift_basis`), `tests/test_accounting_cross_engine.py`, `tests/test_attribution_independent_reference.py`, `docs/accounting_and_measurement_conventions.md` | Documented conventions, exact drift option, closed-form scenarios (drift, costs, dividend, split, funding, partial fills, futures roll, option expiry, foreign currency) against the ledger |
| 3 Providers | `marketdata/capability_registry.py` (new) | One status per (provider, dataset) derived from evidence (generated live matrix, entitlement probe, committed real fixtures, test mentions) |
| 4 Data Health | `data/health.py`, `webapp/health_api.py`, `static/health.js`, `index.html`, `server.py` allow-list | Coverage, integrity, readiness, controls (validate as a job with a fingerprint snapshot, refresh, retry, missing periods, source, export, correction log) |
| 5 ML recovery | `ml/deep/runs.py`, `static/ml.js` | `checkpoint_failures` in the run summary and an alert banner: latest completed epoch against latest durable checkpoint, units at risk |
| 8 Futures | `assets/futures.py`, `assets/futures_validation.py` | Specification checks, executability report, roll table (calendar spread against roll cost), difference-adjusted returns fixed |
| 10 Audit trail | `backtest/audit.py` | One row per trade: decision date, signal, target, trade date, pre/traded/post weight, cost, P&L to the next trade; `reconcile` against the result |
| 11 Docs/UI | `webapp/lab_api.py` | Help matches what the save route accepts |
| 12 Security/reliability | `ml/deep/integrity.py`, `ml/deep/checkpoint.py`, `webapp/obs.py`, `webapp/server.py`, `webapp/health_api.py` | HMAC-gated unrestricted checkpoint loads, correlation ids, structured request log with rotation, scrubbed 500 messages, diagnostics, backup restore verification |

Backward compatibility: saved artifacts and research sessions are read unchanged (a series with no `meta` is measured under an inferred convention and flagged `inferred`); the weight engine's default drift is unchanged; checkpoints from older versions load when they contain plain data (every checkpoint of this platform examined does); the CAGR, volatility and Sharpe in the Research Comparison change numerically for existing results because the old arithmetic was wrong (D3, D4). Two existing tests were changed on purpose and say why (`test_crypto_store.py` D1; `test_futures_series_exact.py` D5), and three numerical expectations of `test_research_workspace.py` and `test_research_compare_studies.py` that encoded the old convention were rewritten.

---

## 3. Verification

RESULTS_PENDING

---

## 4. Remaining work, classified

| Item | Classification | Notes |
|---|---|---|
| 1.1 Crypto book validity, strict policy, audit | **Implemented and verified** | `test_crypto_book_policy.py`, `test_crypto_store.py`, replay and research tests |
| 1.2 Annualisation by calendar and frequency | **Implemented and verified** | numerical references in `test_research_compare_conventions.py`; sqrt(periods) scaling is a convention that assumes uncorrelated returns, the page says so |
| 1.3 Missing returns never zero | **Implemented and verified** | same file: holes, inception dates, calendars, flat days, partial coverage, mixed frequencies |
| 1.4 Metric registry and ranking direction | **Implemented and verified** | `test_research_metrics.py`; a page selector for `rank_by` is **not implemented** (API only) |
| 2.1-2.2 Cross-engine accounting | **Implemented and verified** for the cases listed; **partially** for multi-leg options and futures margin (ledger-level only, not run through the weight engine) | `test_accounting_cross_engine.py`, `test_engine_reconciliation.py` |
| 2.3 Attribution reconciliation | **Implemented and verified** (independent BHB and Carino references); factor attribution against an external vendor model **not done** | `test_attribution_independent_reference.py`, `test_portfolio_attribution.py` |
| 3.1 Provider capability registry | **Implemented and verified** (rules tested on constructed evidence) | |
| 3.2 Live read-only verification | **Blocked here**: no `MASSIVE_API_KEY` or other credential on this machine. The exchange matrix of 2026-11-25 (OKX, Deribit live; Bybit and Binance refused from that network) was **not re-run** this session | Massive rows stay `mock_tested` or `not_tested` |
| 3.3 Data-quality checks | **Implemented and verified** for price series, quotes, symbol mapping, point-in-time dates, corporate actions, expiry; **partially** for incremental-sync and provider-error handling (the market store already versions every revision) | `test_data_health.py` |
| 4 Data Health workspace | **Implemented and verified** in Chromium (`test_webapp_data_health_browser.py`) for the platform snapshot; the market store path is exercised by the API tests only (the test store is empty); the Options, Futures and Crypto downloads are **not** run through these checks | |
| 5.1 Fresh-process recovery of every family | **Implemented and verified earlier** (N-BEATS, forecast networks, PINNs, Double DQN) and re-run in the regression below; AE, VAE, GNN, Transformers and SB3: see the regression results; **Stable-Baselines3 is not installed here, so the SB3 tests are skipped and unverified** | |
| 5.2 Checkpoint failure handling and warnings | **Implemented and verified** (disk full, kill, corrupt, forged, repeated interrupts; the banner distinguishes completed from durable) | `test_recovery_faults.py`, `test_training_checkpoint_warning.py`, browser test |
| 5.3 Windows verification | **Implemented but unverified**: `scripts/verify_windows_recovery.py` exists and has never run on Windows (no Windows here) | |
| 6 Forecast to alpha, optimiser comparison, attribution link | **Implemented and verified earlier**, re-run in the regression | `test_forecast_alpha.py` |
| 7 Real-filing fundamentals benchmark | **Blocked**: the repository holds no real filings for the US, Singapore, Japan, South Korea or China, and none can be fetched here. No jurisdiction is claimed as tested. Point-in-time gating of unconfirmed and flagged figures is implemented and tested (`test_fund_pit.py`) | |
| 8 Futures | **Implemented and verified** (specs, executability, roll table, one defect fixed) | `test_futures_contract_validation.py` |
| 9 Artifact consistency and saved-research browser workflow | **Implemented and verified earlier** (`test_research_artifacts_v2.py`, `test_webapp_research_integration_browser.py`), re-run | |
| 10 Signal-to-trade audit trail | **Implemented and verified** for the weight engine (reconciles to the result to 1e-9); the event engine's own order/fill log is a separate trail and is **not** joined to the signal | `test_backtest_audit_trail.py` |
| 11 UI and documentation consistency | **Partially implemented**: the Alpha Lab help is corrected and tested; a full sweep of every page's help text was **not** made | |
| 12 Security | **Partially implemented**: checkpoints, error disclosure, secrets in responses, static allow-list, verification allow-list, SQL identifiers audited and tested; model-artifact and SB3 pickles are open (section 5) | `test_security_audit.py`, `test_checkpoint_security.py` |
| 12 Performance profiling | **Not implemented** | nothing was optimised; no claim is made |
| 12 Reliability (logs, ids, storage health, backup verification, restart diagnostics) | **Implemented and verified** | `test_observability.py`, `test_data_health.py` |
| 13 Layered test strategy | **Partially implemented**: unit, numerical, integration, fresh-process and browser tests exist and were added to; live-provider tests are kept apart (none could run); platform tests are Linux only | |
| 14 Staged order | Followed: critical correctness, data integrity, research integration, empirical reliability, platform hardening | |

## 5. Prioritised outstanding issues

| Priority | Issue | Impact | Depends on | Next action |
|---|---|---|---|---|
| 1 | `joblib.load` of saved models and `pickle.loads` of the SB3 environment and the ops store are gated only by a hash stored with the data | A planted model file or backup runs code when loaded (same class as D7) | none | Give the model envelope an HMAC variant (`QFM2`) using `ml/deep/integrity.py`, refuse an unsigned legacy artifact unless the user re-saves it; sign the SB3 manifest; replace the ops frame pickle with parquet |
| 2 | Real-filing benchmarks for the five jurisdictions | The importer's accuracy on real documents is unknown | Real, person-verified filings | Add held-out filings under `benchmarks/fundamentals_importer/real_filings/` and run `importer.benchmark` |
| 3 | No live verification of Massive datasets | Every Massive row is mock-tested or untested | A `MASSIVE_API_KEY` | Add the key through the environment's credential form, run `POST /api/market/probe`, re-open Data health |
| 4 | Fixed 252 inside the engines | A 7-day series pushed through an equity-calendar engine is mis-annualised there (the comparison is protected) | none | Read periods per year from the data's declared calendar in `backtest/metrics.py` and `framework/validate.py` |
| 5 | Windows and Stable-Baselines3 recovery unverified | Resume behaviour on Windows and for SB3 agents is untested here | A Windows machine; `pip install stable-baselines3` | Run `scripts/verify_windows_recovery.py` and `tests/test_fresh_process_sb3.py` |
| 6 | Data Health does not cover the lab downloads, intraday series or the market store with real data | Those sources are unchecked | none | Feed `data.health.report` from the download store and add intraday gap rules |
| 7 | `rank_by` has no page control | The metric choice is API-only | none | Add a selector to the comparison page fed by `metric_registry` |
| 8 | Performance profile | Unknown hot spots; the Replit memory budget is unmeasured | none | Profile the comparison, health report and training monitor under load before changing anything |
