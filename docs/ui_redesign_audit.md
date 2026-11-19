# Repository audit for the research-workspace redesign (November 2026)

What this is: what the web application is made of today, read from the code before any of it was changed, and what that means for the redesign. It is the Phase 0 deliverable. The paths in the brief (`static/index.html` and so on) are all under `src/webapp/static/`.

Baseline: the branch was at commit `d0b9fd0` with CI run 129 green (full suite on Python 3.10 to 3.12, the 205-test browser lane on Chromium, lint, docs).

## 1. What exists

**Stack.** A Python standard-library HTTP server (`src/webapp/server.py`, 388 lines) serving static files and a JSON API; no front-end framework, no build step, no bundler. The page is 15 plain scripts loaded eagerly in `index.html` (about 10,700 lines in all, `style.css` 455 lines). One Content-Security-Policy forbids string evaluation, so there is no `eval` and no `new Function` anywhere.

**Navigation.** 16 horizontal tabs in a `nav.tabs` bar (`role="tab"`, ids `tab-<name>`, panels `view-<name>`), wrapped onto a second row below 1,800 px. Tab switching is `showTab(name)` in `app.js`; the last tab is kept in `localStorage` (`ql.tab`). There are no URLs: no hash, no history entries, no deep links, no breadcrumbs, no search.

| Tab | File | What it is | Kind |
|---|---|---|---|
| Backtest, Compare, Strategy builder | `app.js` | The strategy backtest, the comparison of runs held in memory, a builder | page |
| Tune, Signals | `tune.js`, `signals.js` | Parameter search with a hold-out; daily rule signals with a ledger and a paper portfolio | lab |
| Forecast, Scenarios, Rates | `forecast.js`, `scenarios.js`, `rates.js` | Distribution forecasts, generative scenarios, yield curves | lab |
| Options, Futures, Crypto | `options.js`, `futures.js`, `crypto.js` | Option analytics and strategy backtests; the same for futures and crypto | lab |
| Machine learning | `ml.js` (1,579 lines) | Datasets, 40 model kinds, experiments, reports, selection tests, champion/challenger | lab |
| Fundamentals | `fundamentals*.js` | Statements, ratios, factor research, manual entry | lab |
| Execution, Cash flows | `app.js` | Order-schedule simulation; liabilities and spending | lab |
| Guides | `app.js` | Documentation viewer | page |

**There is no regime tab, no overview and no saved-research page.** Regime detection exists only as a library (below) and as a setting inside the Backtest and ML forms.

**Components.** A "lab" is a plain object (`id, root, title, modes, sections, check, run, empty, build`) rendered by `renderLab` into a form column and a results column. Shared helpers are exported on `window.QL`: `h` (element builder), `chartCard`, `tile`, `simpleTable`, `actionsCard`, `labField`, `checkList`, `waitJob`, `store`, `toast`, number formatters. `charts.js` (646 lines) draws line, bar, histogram, heat-map and fan charts as SVG with touch tooltips and zoom (added in the previous review). The form controls are built per lab; there is no shared asset selector, date-range selector or model selector component.

**State.** Four mechanisms, none shared: (1) `S` in `app.js`, one in-memory object for the Backtest tab, **including the runs** (`S.runs`), which are lost on reload; (2) one object per lab (`lab.form`, `lab.results`, `lab.cache`); (3) `localStorage` under `ql.*` (theme, tab, tickers, source, a few per-tab choices), never the results; (4) the server's files. A change of ticker in one tab does not reach another.

**Persistence on the server.** Three independent stores, all additive and versioned: the ML registry (`artifacts/ml/registry.sqlite` plus per-experiment files, with numbered migrations, trial counts, lineage and holdout ledgers); `data/user/signals/` (JSON and JSON-lines, atomic writes); `data/user/jobs/` (one JSON record per job, a result file per finished job). Strategy backtests are stored nowhere: the Compare tab compares what is in memory. The Tune tab and the Options, Futures and Crypto backtests keep nothing either.

**Jobs.** `App.submit_task(label, work, kind, lane)` runs `work(progress)` on one of two single-thread pools (`compute`, `data`); the page polls `/api/job/<id>`; `JobLog` writes a record per job and marks jobs lost in a restart `interrupted`. States today: queued, running, done, error, interrupted. **There is no generic cancel**: Tune, Scenarios and Forecast each carry their own stop flag; every other job runs to the end. `/api/jobs` lists the last 60, with no link to what started them.

**Experiments.** `src/framework/experiments.py` (`ExperimentManager`, `spec_hash`, `git_commit`) for the pipeline; `src/ml/registry.py` (SQLite, 59 `/api/ml/*` routes) for ML: configuration, metrics, predictions, positions, returns, report, model, provenance (data and code hashes), lineage, holdout access, selection evidence. This is the strongest part of the application and the redesign reuses it unchanged.

**Regimes.** Three layers, all causal by construction: `framework/regimes.py` (7 detectors returning probabilities per date: vol-state, HMM walk-forward, GMM, BOCPD, macro, composite, static); `ml/regimes.py` and `ml/regime_exposure.py` (HMM and Markov-switching engines with a causal `filter`, a volatility-quantile engine, exposure rules); `ml/regime_eval.py` (regimes declared before the evaluation and hashed, labelled causal or retrospective, with HAC and bootstrap inference per regime and a `min_days` rule). `models/regimes.py::walk_forward_hmm` is the filtered, expanding-window estimate. Nothing exposes any of this to the page except as a signal setting.

**Charts.** Own SVG library, no dependency. Zoom, pan, pinch, tap tooltips, keyboard (previous review). No export of a chart image or data; no synchronised cursors or ranges between charts; every redraw rebuilds the SVG.

**Accessibility and responsive.** Axe-core audits of 22 views were done once and are held by `tests/test_webapp_accessibility.py` (heading order, tab roles, labels, contrast of both themes). One breakpoint (1,000 px) for the layout; the top bar wraps. Tests check that the Backtest page and four labs fit a 390 px phone.

## 2. Problems the redesign has to solve

| Problem | Evidence | Effect |
|---|---|---|
| 16 flat tabs, no hierarchy, no URLs | `index.html`; no `location.hash` anywhere | The user cannot find, link or return to a page; the browser's back button leaves the app |
| No shared context | `S`, `lab.form` and `ql.*` are separate | The ticker chosen in Backtest has to be chosen again in Options and ML, and the dates in Regime-like settings |
| Backtest results are not kept | `S.runs` in memory | A reload loses every run; nothing to compare or resume |
| No regime page | no route, no tab | A strategy's behaviour by regime cannot be seen; the regime library is invisible |
| No generic job cancel, no job-to-research link | `submit_task` | Long jobs cannot be stopped; a job cannot be traced to the research it served |
| Duplicated helpers | per-lab date, ticker and model controls | Inconsistent behaviour and a larger surface to test |
| 15 scripts loaded eagerly | `index.html` | About 10,700 lines parsed before the first paint; 11 of them serve tabs the user may never open |
| No request cancellation | `api()` has no `AbortController` | A slow answer for a page the user has left is still drawn into the shared state |
| Single breakpoint, top bar wraps | `style.css` | Tablet widths get the desktop layout squeezed |
| Backtest, ML and Options results are labelled inconsistently | per-tab banners | In-sample, out-of-sample and simulated results are not marked the same way everywhere |

## 3. Retain, refactor, extend, replace

| Area | Decision | Reason |
|---|---|---|
| The backtest engine, pricing models, ML and regime libraries, registry, jobs, data pipelines | **Retain**, untouched | They carry the research safeguards; the brief forbids competing engines; the previous review verified them |
| `window.QL` helpers, lab framework, `charts.js` | **Retain and extend** | Every tab uses them; new pages are labs or small pages built from the same helpers |
| `#tab-<name>` / `#view-<name>` ids, `role="tab"`, `showTab`, `ql.tab` | **Retain** | 21 test files and the accessibility audit address them; the sidebar is the same tab buttons laid out vertically |
| Top tab bar | **Replace** with a grouped, collapsible sidebar (same buttons) and a slim top bar | The crowding problem |
| `localStorage` keys `ql.*` | **Retain**; add `ql.shell.*` for navigation preferences and `ql.dash.*` for dashboard layout | No new storage mechanism for view preferences |
| Server-side persistence for research | **Extend with one small store**, `data/user/research/`, JSON with atomic writes and a schema version, the pattern of `signalbook/store.py` | Sessions and artifacts span strategies, options and regimes, which the ML registry does not hold; no new database, no new dependency. Large series stay on disk, never in the browser |
| `S.runs` | **Extend**: a run can be saved as a research artifact; the in-memory list stays | Existing Compare behaviour is preserved |
| Job infrastructure | **Extend**: cooperative cancel (flag checked at each progress message; queued jobs cancel at once), `cancelled` state, a session link on the record | No new queue |
| ML registry | **Reuse**; the Recent Experiments page and the comparison read it through the existing `/api/ml/experiments` and a new read-only returns route | Leakage controls, lineage and holdout ledgers stay where they are |
| Regime library | **Reuse**; a new `src/research/regime_lab.py` orchestrates `walk_forward_hmm`, the engines and `regime_eval` semantics and adds only what is missing (smoothed estimate for contrast, transition and duration statistics, performance by label) | No new detector without need |
| Per-lab asset/date controls | **Refactor gradually**: one shared "use the research context" action per lab rather than rewriting each form | The smallest change that makes context shared |
| Script loading | **Extend**: the new modules load on demand when their page is first opened (a small loader), existing scripts unchanged | Lower first-paint cost without touching the labs |

## 4. Plan, with dependencies and acceptance criteria

| Step | Depends on | Acceptance |
|---|---|---|
| A. Audit (this page) | | Findings above; every claim traceable to a file |
| B. Design tokens and sidebar shell (`shell.js`, CSS): groups, collapse, breadcrumbs, command palette, recents, shortcuts, drawer on small screens, preferences, hash routing | A | All 205 existing browser tests pass (with the mobile ones opening the drawer); `#/options` opens Options; old `ql.tab` still restores; keyboard-only operation; no horizontal scroll at 390, 768, 1,400 px |
| C. Page framework and new pages: Overview, Saved Research, Recent Experiments, Background Jobs, Data Management, Settings | B | Every figure on every page comes from an endpoint; a source label (real / historical / simulated / unavailable) on each widget; widgets can be hidden and reordered and the layout survives a reload |
| D. Research context and sessions: typed schema, validation, versioned store, API, context bar, compatibility check | B | A context that fails validation is refused with the reason; a saved session loads after a schema bump (migration test); changing the context never alters a saved result and flags incompatible ones |
| E. Market Regime Lab: filtered and smoothed estimates, probabilities, transitions, durations, conditional returns, by-regime performance of a saved backtest or an ML experiment | D | Filtered labels at *t* are identical when later data is removed (look-ahead test); the smoothed estimate is labelled retrospective and cannot be used to trade in the lab; regimes with too few days report "insufficient evidence" |
| F. Cross-module workflow: contextual actions in Strategy Lab, Options Lab linked to a session, ML experiment comparison, guided Setup to Report path | D, E | The SPY example of the brief works end to end in a browser test |
| G. Research Comparison: comparability engine, synchronised charts, CSV and report export | D | Experiments with different targets, periods or datasets are flagged and are not ranked as equivalent; the common window is stated |
| H. Jobs: cancel, cancelled state, session link, retry; request cancellation and caching on the page | B | A queued job cancels immediately and a running one at its next progress message; a cancelled job leaves no half result |
| I. Integrity labels everywhere (evidence kind, data kind, methodology notes) and the report | D to G | Every artifact carries `evidence` and `data_kind`; no page shows a result without them |
| J. Tests, documentation, migration notes, evidence run | all | Existing suites green before and after; new tests for each item of Phase 8; a browser pass at three viewport sizes |

## 5. Risks

- The sidebar replaces the nav the existing tests drive. Mitigation: the tab buttons keep their ids and roles; only the layout and, on a phone, the visibility change, and the tests that click a tab on a phone open the drawer first.
- `app.js` is one 1,600-line closure. It is not split in this programme; new behaviour goes into new files and a few small, tested hooks.
- Dark mode becomes the default for a person who has not chosen a theme. A saved choice, and the system preference when the person picks "system", still win.
- A "regime lab" can mislead if smoothed labels are used as if they were known at the time. The lab reports both, defaults to filtered, and marks smoothed output as retrospective in the interface and in the saved artifact.
