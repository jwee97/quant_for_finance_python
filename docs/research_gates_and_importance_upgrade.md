# Research gates, permutation importance, Alpha Lab factors: audit, root causes, what was done

## 1. Gate A "P0-08: never run"

**Audit.** `src/ml/verification.py` (`REQUIREMENTS["P0-08"]` = `tests/test_ml_architecture_contracts.py` + `tests/test_ml_architecture_catalogue.py`), `src/ml/gates.py::_verification_items`, `src/webapp/ml_verify.py`, `src/webapp/ml_api.py::verification_run`, `src/webapp/static/ml.js` (the old gate tiles) and `evidence.js`.

**Root cause.** Items 1, 2 and 10 of the list: verification had never been requested, the web page had no control that calls `POST /api/ml/verification/run` (only the `ml-verify` command line did), and a status exists only when a run writes one. Everything else was fine: the test runner and both test files are present, the job starts, and a run of P0-08 in this checkout passes (21 tests, about 58 s) and is persisted (`tests/test_ml_gate_guidance.py::test_p0_08_runs_through_the_job_api_...`). Nothing was hard-coded or written into the registry by hand.

**What changed.**
* `ml/verification.py`: `preflight()` (runner, test files, busy), one run at a time (a lock), a per-file time limit, failure messages and tracebacks kept with each check, live progress (`LIVE`), counts, duration, source fingerprint and age per requirement, `history()`.
* `ml/gate_guidance.py` (new): purpose and "what passing does NOT mean" for each gate; the KIND of obstacle for every item (missing evidence, could not run, failed validation, stale, policy restriction, contaminated holdout, insufficient evidence); next action as data; `plan_readiness` and `prospective_readiness`.
* API: `GET /api/ml/verification/status`, `GET /api/ml/verification/history`, `GET /api/ml/plan-readiness`, `GET /api/ml/prospective-readiness`; `POST /api/ml/verification/run` now also takes `{"experiment_id", "scope": "pending"}` and `{"scope": "all_pending"}`. A request names registered requirement ids only (any other key is refused), at most 40 per request, authenticated like every POST, one verification at a time, 900 s per test file.
* `static/gates.js` (new): three cards (title, status, one-sentence purpose, evidence summary, exact blocker with its kind, last verification, one next action, technical details with per-requirement Run and History buttons and failure traces), "Run all pending requirements", "Retry failed verification", a progress panel (requirement, file, files done, counts, elapsed), an explainer for the three gates, and the plan and prospective workflows. The repeated "Read first" banners are gone.

**Gate B and C are not made easy.** A holdout read outside a frozen plan stays contaminated: the card says so and offers only "Plan a new frozen evaluation", which reads the ledger (consumed ranges, first date a new look could protect, the policy's minimum evidence) and lists the steps; it creates nothing. Gate C's card explains that prospective evidence is made only by waiting, and lists which of its steps (policy, strategy version, limits, recovery plan, forward test, monitoring) are done. Not built: a form that creates and freezes a plan, and a form that registers a strategy version (both remain API/CLI actions).

## 2. "What breaks when a feature is shuffled": no bars

**Root cause (reproduced).** On a universe of **two assets** a cross-sectional rank IC does not exist, so the drop in rank IC is undefined for every feature: the server sent 20 rows with `mean_drop: null` and the chart drew an empty value axis. ECharts' default axis for no data is 0 to 1 in steps of 0.2, which is exactly the "0.200 0.400 0.600 0.800" in the report; the labels overlapped because of the narrow phone width. With three or more assets the same chart draws correctly (checked at 1100 px and at 360 px with a device scale of 3), so this was not a rendering bug.

**What changed.** `ml/explainability.py`: `permutation_status` (computable, why not, minimum assets 3, interval definition, sample) and approximate 95% intervals; `static/importance.js`: a diverging bar chart (zero line, both signs, whiskers, top N plus the three most negative, exact tooltips, no overlapping axis labels at phone width), a Chart/Table toggle over the same rows, an explanation card instead of an empty axis, and "How to read this chart". `charts_ec.js::hbars`: undefined values draw nothing without breaking the other rows. No synthetic values anywhere. The importance is computed once when the experiment runs, on its out-of-sample blocks, and is not recomputed for exploration.

## 3. Alpha Lab

Audited: `src/alphalab/{ic,breadth,calibrate,studies,capacity,attribution}.py`, `src/webapp/lab_api.py`. Already present and left alone: rank IC, HAC t-statistic, decay and half-life, calibration, effective breadth, fundamental law, transfer coefficient, capacity, attribution; a study refuses fewer than 6 tickers.

Added: `src/models/regression.py::cross_sectional_ic` now refuses `min_assets < 3` (two assets give +1, -1 or undefined); `src/alphalab/factors.py`: `evaluate_factor` (rank and Pearson IC with a Newey-West interval, quantile spreads and the long-short return, turnover, coverage by date and asset, extreme-value share, thirds stability, decay, IC after neutralising supplied controls), `compare_factors` (Holm and Benjamini-Hochberg), `combine_factors` (equal weight, IC weighted, ridge, orthogonalised; weights from the past only; judged on the last 40% of dates; an "improvement" must beat the best single factor chosen on the training period and equal weight by more than 1.96 standard errors); `POST /api/lab/factors` (a job) and `alphalab.studies.factor_study`. **Not built:** the five-stage Alpha Lab interface (the new route has no page yet), sector/industry neutralisation inside the API (the library takes controls), optimizer comparison with an efficient frontier, weight stability and constraint-binding diagnostics, crowding proxies.

## 4. Forecasting (audit only)

Audited `src/forecast/{scoring,evaluate,baselines,verdict}.py`. Already present and tested in 17 test files: CRPS, pinball loss, interval coverage with an exact binomial test, PIT with a KS test, Diebold-Mariano with a Newey-West lag of `stride - 1` for overlapping origins, Clark-West and the Campbell-Thompson out-of-sample R-squared, directional skill against always-up, Holm correction, random-walk / RiskMetrics / historical / GBM baselines on common origins, and neural forecasters that run inside the job-level training context (checkpointed and resumable; all 11 architectures are killed and resumed in `tests/test_fresh_process_forecast.py`). **No change was made to Forecasting in this round**: the five-stage layout, fan-chart and decision-usefulness (after-cost) views of Part D were not built.

## 5. Fundamentals (audit only)

Audited `src/fundamentals/` (the importer's review-before-store model, in which scanner confidence never confirms a value, point-in-time availability in `pit.py`, coverage in `coverage.py`, lineage and trace in `importer/trace.py`). These exist from earlier rounds. **No change was made in this round**: the six-section workspace reorganisation, new valuation scenarios and the Fundamentals-to-Alpha-Lab connection were not built; real-document validation for the listed countries was not run (it needs the filings themselves).

## 6. Cross-workspace, data providers, UI standards

Not built in this round: shared research context across the three workspaces, experiment comparison across them, provider capability dashboards. The new pieces follow the shared chart kit (ECharts through `C.hbars`, no new library), have Chart/Table toggles where they chart, and never fill missing observations.

## 7. Tests and results

Environment: Linux 6.18, Python 3.11.18, PyTorch 2.15.0+cpu, 4 CPUs.

| Command | Result |
|---|---|
| `python -m pytest tests/test_ml_gate_guidance.py` | 12 passed (includes P0-08 end to end: 21 tests passed, persisted, history, status) |
| `python -m pytest tests/test_ml_permutation_importance.py` | 5 passed (two assets: all drops undefined and the reason; four assets: finite, signed, intervals; sign convention) |
| `python -m pytest tests/test_alphalab_factors.py tests/test_lab_api.py` | 8 + 10 passed (planted factor found, interval covers the truth, noise not, Holm and BH hand-computed, thin cross-section refused, neutralisation, no look-ahead in combinations) |
| `python -m pytest tests/test_webapp_ml_gates_browser.py` | 4 passed (cards, the P0-08 button with progress, plan/prospective panels, phone layout) |
| `python -m pytest tests/test_webapp_ml_importance_browser.py` | 14 passed (bars measured in the SVG for positive, negative, mixed, all-negative, all-zero; undefined values; chart and table identical; top N; long labels; phone axis labels; caption not covered; resize; hidden tab; real 4-asset and 2-asset experiments) |
| `python -m pytest tests -n 4 --ignore-glob='tests/*browser*'` (plus the recovery suites run in the previous round) | 7556 tests: 3 failed, 6 collection errors, 57 skipped; **all 9 also fail at commit ab798db before this work** (`src.marketdata` does not export `concat_events`/`DataContract`; the crypto view skips a heading level; `gymnasium` is not installed) |
| browser suites (ML, training, workspaces, main) | 131 tests; 2 failed only because source files were swapped under the running server by a baseline comparison; both pass alone |

## 8. Remaining limitations

Gate B/C cannot be passed by a button; the contaminated holdout stays contaminated. The importance chart explains an undefined result but cannot invent one: with fewer than three assets there is none. Alpha Lab's new factor evaluation has an API and tests but no page; Forecasting, Fundamentals, cross-workspace integration and provider dashboards were audited, not extended. No API key or extra historical data is needed for what was built.
