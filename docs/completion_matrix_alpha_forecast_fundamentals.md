# Completion matrix: regression reliability, Alpha research, Forecasting, Fundamentals, Integration

This round finishes the work the previous round only audited or built as a library. The rule used throughout: **a backend function is not a finished workflow until a person can reach it in the page, it says what it does not show, and a test has run it.** Each row below carries one of these statuses, and the statuses mean exactly this:

| Status | Meaning |
|---|---|
| **Implemented and tested** | Reachable in the page (or through the API where no page is meant), with numerical or browser tests that ran and passed in this round. |
| **Pre-existing, re-run** | Built before this round, not changed, its tests were run again and passed. |
| **Partially implemented** | Some of the request exists; the row says what is missing. |
| **Blocked** | Cannot be done here; the row says by what. |
| **Not implemented** | Not built. |

## Priority 1: regression reliability

| Item | Status | Evidence |
|---|---|---|
| `src.marketdata` export failures | **Implemented and tested** | Cause: commit `0369dd3` replaced the package's `__init__` with the Massive docstring and dropped every vendor-neutral export. Restored (`src/marketdata/__init__.py`). `tests/test_marketdata*`, the engine, swaps and stage-import tests pass. |
| Crypto heading-level accessibility failure | **Implemented and tested** | An unboxed block jumped from h2 to h4 (`[1,2,2,4,3]`). `subLevel()` in `static/workspace.js` puts a block's heading one level below the nearest heading above it. `tests/test_webapp_accessibility.py` passes. |
| Optional `gymnasium` / Stable-Baselines3 | **Implemented and tested** (requirements); **Blocked here** (the packages are not installed in this container) | `requirements.txt` lists `gymnasium`, `stable-baselines3`, `sb3-contrib`. `tests/test_stage_imports.py` treats exactly those three as optional and records them as missing, printing the list; any other import error still fails. The SB3 recovery tests need the packages: run them after `pip install -r requirements.txt` (not verified here). |
| Browser tests disturbed by concurrent source edits | **Implemented and tested** | The page's "restart needed" banner compared the code on disk with the code at launch, so touching any `src` file during a run raised it. `api.current_signature()` is frozen to the launch signature by `tests/conftest.py`; browser tests pass while a source file is being modified. |
| Baseline failures reported apart from new regressions | **Implemented** | See "Test results" below: what failed at the previous commit and what this round changed are listed separately. |

## Priority 2: Alpha Lab

The lab opens on a five-stage workflow (`static/alpharesearch.js`, mode "Alpha research"). Stages 1 to 3 are one server run (`POST /api/lab/factors`) because they read the same factor panels; stage 4 is `POST /api/lab/portfolio`, stage 5 `POST /api/lab/trading`.

| Item | Status | Where / evidence |
|---|---|---|
| Five stages: Discovery, Validation, Combination, Portfolio construction, Performance and attribution | **Implemented and tested** | `tests/test_webapp_alpha_research_browser.py` runs all five in Chromium, in order, and asserts what each draws. |
| Factor coverage | **Implemented and tested** | `factors.discovery`: median assets per date, coverage by date and by asset, worst months, whether the cross-section can be ranked and why not. `tests/test_alphalab_factors.py`. |
| Neutralisation | **Partially implemented** | Against market beta and 12-1 momentum, both computed from the same prices: the IC left, the share kept, and "nothing left" when a signal is only one of them. **Not** against sector or size: a price-only study has neither. |
| Stability | **Implemented and tested** | IC by thirds and whether the sign holds in all, drift of the signal between halves (Kolmogorov-Smirnov on raw values: a z-score would hide a change of level), rank persistence at 1 and 21 days. |
| Multiple-testing diagnostics | **Implemented and tested** | Holm and Benjamini-Hochberg over every factor tried, HAC intervals, clipping sensitivity at 0, 2 and 5 percent. The combination is judged on a test period nothing was fitted on and counts as an improvement only if it beats both the best single factor and equal weight by 1.96 standard errors. |
| Optimiser comparison on identical windows, costs and constraints | **Pre-existing, re-run** (`portfolio.lab.compare`), shown in stage 4 | Same lookback, rebalance dates, covariance, expected-return forecast, cap and cost for every method. `tests/test_portfolio_lab.py`. |
| Weight sensitivity and concentration | **Implemented and tested** | `portfolio.lab.diagnostics`: effective N, HHI, top-3 share, zero positions, cap binding, weight turnover, rank correlation of weights, movement under noisy expected returns and under another covariance estimate, cost drag, and a long-only efficient frontier under the same cap with each method placed on it. Closed-form checks (equal weight: effective N = N, HHI = 1/N, no turnover), the cap, the frontier, cost drag = turnover x cost. |
| Cross-sectional breadth guard in the UI | **Implemented and tested** | The page says before the run that a study needs 6 tickers and a rank correlation needs at least 3 assets with a value on the date (a date with fewer is skipped, never scored as zero), and disables the run button. A signal too thin to rank is listed in Discovery with the reason and not scored. |
| Attribution of the portfolio's return | **Implemented and tested** (follow-up round) | Factor and Brinson-Fachler attribution of every optimiser's return, reconciled; see `docs/research_integration_forecast_to_comparison.md`. |

## Priority 3: Forecasting

The Forecast tab keeps its methods (no model was added) and groups its cards into five stages: Generate, Walk-forward validation, Calibration, Model comparison, Research conclusions.

| Item | Status | Where / evidence |
|---|---|---|
| Consolidated workflow | **Implemented and tested** | `tests/test_webapp_forecast_browser.py` (9 tests). Existing cards were moved, not rewritten. |
| Fan chart | **Pre-existing, re-run** | |
| Realised against predicted | **Implemented and tested** | Per-origin median and 5/95% quantiles are now kept by the evaluation (`evaluate._series`); the page draws them over time with the central 90% range, and as a scatter of median forecast against outcome. Test: the series equals the independently recomputed log return, and the 90% hit equals "inside the quantiles". |
| PIT diagnostics and interval coverage | **Pre-existing, re-run**; **coverage over time implemented and tested** | Rolling coverage with the band chance alone allows for `window / stride` independent forecasts. |
| Rolling forecast errors | **Implemented and tested** | Rolling mean absolute error of the median, rolling skill against the random walk, rolling coverage; test recomputes each from the series. |
| Models compared on comparable dates | **Implemented and tested** | `evaluate._common`: the origins every method scored. A method that failed at some origins is scored there on fewer dates than the others; the comparison is recomputed on the common dates, with skill in each half, Holm-adjusted p-values on that sample, and a banner when the dates differ. Test with a method that refuses every third forecast: the common count, the lost count, and the skill all recomputed independently. |
| Appropriate baselines | **Pre-existing, re-run** | The random walk with the last year's spread (distribution), the five-year historical mean (mean forecast), and the simpler control each complex method must beat (`CONTROLS`). |
| Research conclusions | **Implemented and tested** | `verdict.conclusions`: each statement has a status (established, not established, caution). A skill that holds in one half only is a caution, not a finding; a comparison on different dates says so. |
| Checkpoint and experiment infrastructure | **Pre-existing, re-run** | Forecast jobs already run inside the job-level training context; `tests/test_fresh_process_forecast.py` passes. The forecast evaluation can now also be kept in a research session. |
| Realised-against-predicted for every horizon at once | **Not implemented** | Only the headline horizon is drawn; the scores of all horizons are in the existing skill matrix. |

## Priority 4: Fundamentals

| Item | Status | Where / evidence |
|---|---|---|
| Coverage and missing-data reporting | **Pre-existing, re-run** (coverage grid, data health); **point-in-time dates added** | The Data coverage view now also shows how many periods have a real filing date and how many an assumed one. |
| Filing-to-figure traceability and verification | **Pre-existing, re-run** | `lineage.explain`, the importer's provenance and review workflow; all `tests/test_fund_*.py` pass. Not changed this round. |
| Annual against quarterly comparison | **Implemented and tested** | `audit.annual_vs_quarterly`: the sum of the four quarters (or two halves) against the annual figure for revenue, net income and operating cash flow; reconciles, differs with the gap, or incomplete (never matching). Company page: "Compare each year with its quarters". |
| Accounting quality | **Pre-existing, re-run** | Beneish, Sloan accruals, Piotroski, Altman, quality flags. Not changed. |
| Valuation sensitivity | **Pre-existing, re-run** | `valuation.sensitivity` and the scenario grid. Not changed. |
| Point-in-time checks before fundamental factors enter Alpha Lab | **Implemented and tested** | `audit.pit_audit` (real against assumed dates, lags, dates before the period ends, restatements, verdict ready / ready with assumptions / blocked). `alphalab.fundamental.panels` refuses a blocked verdict and fewer than 6 companies. The audit, its assumptions and the coverage travel with the result. The unit tests check that a figure appears the day after its filing and not on the filing day. |
| Real-filing benchmark | **Blocked** | The repository holds no real filings: `benchmarks/fundamentals_importer/real_filings/README.md` says so and the harness (`importer.benchmark`) is ready for them. Nothing was run, so nothing is claimed. |
| Extraction support against demonstrated accuracy | **Implemented and tested** | `importer.support.matrix()`: every format and connector with its dependencies, its synthetic-benchmark evidence and its real-filing evidence. Statuses: shown on real filings (needs 5 person-verified held-out real filings), synthetic documents only, supported but never benchmarked, blocked, not implemented. Today **none** is shown on real filings; PDF, HTML and Excel are synthetic only; XBRL, CSV, AI-assisted extraction and every filing connector are unbenchmarked; OCR and Docling are blocked here (not installed). |

## Priority 5: Integration

No new framework: the existing research session, its context, artifacts and comparison are used.

| Item | Status | Where / evidence |
|---|---|---|
| Shared context | **Pre-existing, re-run**, extended | The Alpha research workflow reads the session's assets, first day and data source as before and shows them in a strip above the stages. |
| Experiment identifiers, data versions, assumptions | **Implemented and tested** | A factor study and a forecast evaluation are kept in a session as artifacts (`factor_study`, `forecast`) with the job id, a data fingerprint (assets, dates, source), the horizon, the factors or methods, the point-in-time status and its assumptions. |
| Historical availability | **Implemented and tested** | The point-in-time status, the number of assumed dates and the lag settings are in the artifact's configuration. |
| Transaction-cost settings | **Partially implemented** | The portfolio artifact keeps its cost assumption as before. A factor study and a forecast evaluation have no cost by construction. A trading study can now be kept in a session (follow-up round). |
| Prevent misleading comparisons across incompatible samples | **Implemented and tested** | In the workflow, a result whose sample or settings no longer match the form is marked stale and, for another sample, not drawn (browser test). In the comparison, studies of different kinds, horizons or dates are blocked from ranking; assumed point-in-time dates and mixed fundamental / price studies are warned about (API tests). A study has no return series, so it is listed and checked but never ranked by performance. |
| Fundamentals into Alpha Lab | **Implemented and tested** | Fundamental factors are chosen in Discovery, admitted after the point-in-time check, counted in the multiple-testing family, and shown with their check in Validation. |
| Forecast into Alpha Lab or Portfolio | **Implemented and tested** (follow-up round) | Typed forecast contract, explicit conversions, forecast, factor and combined expected returns into the optimisers. |
| Browser test of the Research Comparison with the new kinds | **Implemented and tested** (follow-up round) | `tests/test_webapp_research_integration_browser.py`. |

## Test results

Non-browser areas touched: 1,857 passed, 0 failed. Browser sweep: 279 tests, 4 failed; one (`test_the_fit_ranks_the_models_draws_the_smiles_and_says_what_it_fitted`) fails identically at commit 5dff41d, before this work, and three passed alone (load flakes). The full 8,116-test suite was not run end to end (about four hours on this machine). See `docs/research_integration_forecast_to_comparison.md` for the follow-up round.
