# The research workspace (November 2026)

What this is: the redesign of the web application from a row of sixteen separate tabs into one workspace, where the strategy, options, machine-learning and regime tools share a session, keep what they find, and can be compared. This page says what changed, how to use it, how it is built, what it will not let you do, and what is still missing. The audit that came before it, with the reasons for each decision, is [the redesign audit](ui_redesign_audit.md).

Nothing in the engines changed. A backtest, a model, a hedge and a regime estimate are computed by the code they always were; this layer chooses what to run on, keeps the outcome, labels it and puts results side by side.

## Where everything is now

The tab bar became a sidebar in four groups. Every old tab is still there under the same id, so a link, a test or a habit that addressed it still works.

| Group | Page | Was | Address |
|---|---|---|---|
| Workspace | Overview | new | `#/overview` |
| | Saved research | new | `#/saved` |
| | Recent experiments | new | `#/experiments` |
| Research | Strategy Lab | Backtest | `#/backtest` |
| | Compare runs | Compare | `#/compare` |
| | Strategy builder | Strategy builder | `#/builder` |
| | Tune | Tune | `#/tune` |
| | Signals | Signals | `#/signals` |
| | Options Lab | Options | `#/options` |
| | Machine Learning Lab | Machine learning | `#/ml` |
| | Market Regime Lab | new | `#/regime` |
| | Research comparison | new | `#/rcompare` |
| Markets & Analytics | Fundamentals, Futures, Crypto, Rates, Forecast | the same tabs | `#/fundamentals` and so on |
| | Risk scenarios | Scenarios | `#/scenarios` |
| | Execution | Execution | `#/exec` |
| | Portfolio cash flows | Cash flows | `#/cash` |
| System | Data management, Background jobs, Settings | new | `#/data`, `#/jobs`, `#/settings` |
| | Guides | Guides | `#/guides` |

**Navigation.** The address is `#/<page>` (and `#/<page>?key=value` for a little context, such as the result a page should open on). The back and forward buttons work; an unknown address falls back to a real page. Groups fold; the sidebar narrows to a rail; on a phone it is a drawer behind the Menu button. The breadcrumb names the group and the page, and the sidebar lists the pages you visited last. **Ctrl+K** (or `/`) opens a search over pages, their search words and your sessions; `g` then `o`, `b`, `p`, `m`, `r`, `c`, `j`, `d` or `s` jump to the Overview, Strategy Lab, Options Lab, Machine Learning Lab, Regime Lab, comparison, jobs, data or settings; `[` narrows the sidebar; `?` lists the shortcuts. Folded groups, the rail, the recent pages and the dashboard layout are kept in this browser (`ql.shell`, `ql.dash`): view preferences only.

**Appearance.** Dark is the default; Light and "follow the device" are in Settings. The design tokens (type scale, spacing, radii, the colour roles) are in `style.css`; the colour tokens are stated for both themes and tested for WCAG contrast. Settings also chooses where a visit opens (the last page, the Overview or the Strategy Lab). A first visit still opens on the Strategy Lab, as before.

## Sessions and the shared context

A **research session** keeps what the tools share and what you found: a context, the results you saved, notes and saved comparisons. The bar at the top of most pages shows the open session; **Use in this tab** fills that tab's form from it (the tickers and dates of the Strategy Lab, the underlying and the rate of the Options Lab, the universe of the Machine Learning Lab). Nothing is run by doing so.

The context is a small typed object, and every field is optional:

| Field | What it is | Checked |
|---|---|---|
| `assets` | tickers | each looks like a ticker; at most 40 |
| `asset_class` | equity, rates, crypto, futures, options, mixed … | one of the known classes |
| `start`, `end` | dates | real dates, start before end, none after tomorrow |
| `frequency` | daily | only daily is accepted: the engines work on daily bars |
| `data_source`, `dataset_version` | where the prices came from; a fingerprint of the universe actually used | the source must be one the app knows |
| `risk_free`, `benchmark` | annual rate, ticker | plausible range, valid ticker |
| `costs` | trading cost in basis points, borrow fee, fee per order | non-negative, bounded |
| `strategy`, `ml`, `regime` | what the tab chose | JSON-safe, size-limited |
| `options` | rate, dividend yield, where volatility comes from | ranges, one of model / real chain / your file |

An unknown field is refused, not kept. The server validates (`src/research/context.py`); the page shows the reason.

**A saved result is never edited to fit a new context.** It keeps a snapshot of the context it was made under (assets, dates, frequency, data source, dataset version, risk-free rate, benchmark, costs). When the context changes, the Strategy Lab compares the two and says, field by field, what no longer matches, calling a difference a *mismatch* when the numbers would change if the result were made again (other assets, other dates, another data source, other costs) and a *note* when they would not (a benchmark chosen only for the comparison). It then offers **Apply the context and run again**, which fills the form and starts a run when you press it. The shared context and each tab's own settings are separate: a tab keeps what you typed into it until you use the context there.

## From one module to the next

A finished backtest in the Strategy Lab has a **Research** card:

- **Save to session** keeps the result. The server stores it from its own record of the finished job, never from numbers sent by the page. It is labelled with its evidence (a historical simulation of the whole sample) and its data (platform, downloaded, simulated …), and carries the job, the code version and the dataset fingerprint.
- **Analyze by Market Regime** opens the Regime Lab on the same assets, and scores the saved result by regime once an estimate is saved.
- **Evaluate with ML** opens the Machine Learning Lab on the same tickers and dates.
- **Explore Options Hedge** opens the Options Lab's hedging simulation on the first asset. The hedging cost there is a simulation on model paths; it is not a record of trades that could have been made.
- **Add to Research Comparison** opens the comparison with the result selected.

A machine-learning experiment is added to a session from **Recent experiments** (its out-of-sample net returns come from the registry's own files).

The Overview's **research path** (Setup, Backtest, Machine learning, Regimes, Options hedging, Comparison, Report) shows which kinds of result the open session already has. It is a guide: every step opens on its own.

## The Market Regime Lab

Regime estimates come in two kinds, kept apart on the page and in the saved result:

- **Filtered (causal).** Each day's state probabilities use only data up to that day, and the model that gives them was fitted on earlier days only (an expanding window, refitted every *refit* days; `models.regimes.walk_forward_hmm`). A rule could have traded on it. The lab offers a hidden Markov model with 2 or 3 states, the volatility-state model and a Gaussian mixture of the framework's detectors.
- **Retrospective (smoothed).** One fit on the whole sample and the backward pass, so tomorrow's data helps label today. It describes the past better and **cannot be traded on**. The lab draws it only beside the filtered estimate (for the hidden Markov model; the other detectors are causal by construction and have none), says so on the chart, and reports how often the two labels agree.

What the page reports: the current state and its probabilities (as of the last day), the probability of each state through time, the transition frequencies between states (and the model's own), how long each state lasts (spells, mean, median, longest), the market's behaviour in each state, and diagnostics on how far to trust the labels (mean confidence, days with no clear state, switches per year, switches undone within five days, the range of the volatility a state stands for across refits).

**Performance by regime** pairs the label known at the close of day *t* with the return of day *t + 1* (what a position taken on the label could earn); the same-day figure is shown as descriptive, because for a volatility regime it contains the day that defined the label. A regime with fewer than the chosen number of days (60 by default) says *insufficient evidence* and shows no number. A historical backtest scored by regime carries a caution: its return in a regime is descriptive, not evidence that the strategy works there.

The tests include the look-ahead check: the filtered probabilities of every day up to *T* are identical whether the data ends at *T* or later, and the retrospective ones are not.

## The Research comparison

Choose results from a session, a reference, a window and a risk-free rate. The server (`src/research/compare.py`):

1. measures every result again **over the days they all share**, with one piece of arithmetic (the Backtest tab's tiles and `risk.performance`), so a ratio is never a ratio over different years;
2. lists the **differences in assumptions**, and says which of them stop a ranking: different kinds of evidence (an out-of-sample run against an in-sample simulation), different kinds of data, different assets, different dataset versions, different targets or horizons of machine-learning experiments, different frequency, or too short an overlap (fewer than 30 days). Different costs, simulated data, a window that cuts a result and results picked from many trials are warnings or notes;
3. **ranks by Sharpe only when nothing blocks it**, and then says the order describes this window and is not a forecast; otherwise it returns the reason and still shows the numbers;
4. returns the series for charts that share one window: growth of 1, drawdown, rolling Sharpe and volatility, the distribution of daily returns, the correlation of the returns and, against a chosen reference, beta, alpha, tracking error, information ratio and capture ratios.

A comparison can be saved to the session as a group, downloaded as CSV (with the window, the evidence and the data kind in its first rows) and as a Markdown report: every result with its references (job or experiment, configuration hash, dataset, code version), the assumptions, the differences and the caveats. Each chart has its Table view and a CSV of its data; the charts zoom and pan on their own, and the window of all of them is set by the From and To fields.

## Jobs

Background jobs lists every run with its kind, state, progress, session and time. A run can be **cancelled**: one that has not started leaves the queue at once; one that is running stops at its next progress message; a run that reports no progress finishes its work and its result is then **discarded**, and the state says so. A cancelled run keeps no result. A failed, cancelled or interrupted backtest or regime run can be **run again** with the request that started it (the new job says which one it repeats); nothing is resumed. Other kinds start again from their own tab, because their requests carry files or choices that may have changed. A run started from the Strategy Lab while a session is open is linked to it. `GET /api/jobs` gives each row `cancellable`, `cancelling`, `retryable`, `session` and `retry_of`.

## What it will not do

- It will not show a number it does not have. Every Overview block says where its numbers come from, or says why it has none. The market block is the platform's own snapshot with its date and the words "not a live feed"; the options block says no option analysis has been saved rather than drawing one.
- It will not rank results that are not comparable.
- It will not call a historical simulation out-of-sample, a smoothed regime tradeable, or a simulated hedge a record of trades.
- It will not edit a result because the context changed, or run anything because the context changed.

## Where things are kept

| What | Where | Notes |
|---|---|---|
| Sessions, notes, comparison groups | `data/user/research/sessions/<id>.json` | schema 1, atomic writes, ≤ 200 sessions, ≤ 1 MB each |
| Saved results with their series | `data/user/research/artifacts/<id>.json` | schema 1, ≤ 20,000 points and 8 MB each, `NaN` stored as `null` |
| Job records | `data/user/jobs/` | now also the session, the request (≤ 20 KB) and `retry_of` |
| Machine-learning experiments | `artifacts/ml/` | unchanged; read, never written, by this layer |
| Open session id, theme, folded groups, dashboard layout | browser `localStorage` (`ql.*`) | view preferences; no result or data |

Large series are never put in the browser. `data/user/` is ignored by git.

**Migration and compatibility.** Every stored record carries `schema`. A record from an older version is migrated when read and the original is kept next to it as `.v<n>.bak` (the first shape, "version 0", is converted; a test pins it). A record from a **newer** version is listed with the reason, never opened for writing and never rewritten. The existing tabs, their element ids and their API routes are unchanged; `API_VERSION` is 18, and a page that meets an older app says so as before.

## Programme interface

New endpoints (`src/webapp/research_api.py`): `GET /api/research/sessions|session|artifact|compat|overview`, `POST /api/research/session/create|update|delete|duplicate`, `/note/add|delete`, `/group/save|delete`, `/artifact/save|delete`, `/compare`, `/export`; `GET /api/regime/plan`, `POST /api/regime/run`, `/api/regime/by-result`; `POST /api/jobs/cancel|retry`. Libraries: `src/research/` (`context`, `store`, `extract`, `compare`, `regime_lab`).

## Tests

`tests/test_research_workspace.py` (context validation and compatibility; the store, limits, migration of an older record, refusal of a newer one, a crash between write and rename; extraction labels; the comparison's window, its refusals and its report; the regime lab's look-ahead test, planted-regime recovery and thin-regime rule), `tests/test_research_api.py` (the handlers through real backtest jobs and regime runs; the Overview shows only what exists), `tests/test_jobs_control.py` (cancel a queued job, a running job and one that cannot stop; retry; the session link), `tests/test_webapp_research_browser.py` (sidebar, addresses, back button, folding and the rail across a reload, palette and shortcuts, the phone drawer and tablet widths, Overview widgets, a session's validation and persistence, the whole backtest → regime → comparison → report workflow, a changed context flagging a saved result, settings), and the existing accessibility tests, which now also cover the new pages.

## What is still missing

- **The Overview opens on the Strategy Lab on a first visit**, not on itself, because the existing browser tests and habits assume it; Settings switches it. Widgets are hidden and reordered with buttons (not dragged), and one layout is kept per browser; named layouts are not built.
- **Only some results can be saved to a session**: Strategy Lab backtests, the rolling-strategy backtests of the Options, Futures and Crypto labs (an option backtest is labelled *simulated*), machine-learning experiments and regime estimates. Option chains, hedging-cost studies, sweeps, comparisons and forecasts cannot, so the Overview's options block shows a saved option *strategy* backtest or says there is none.
- **Strategy performance by regime for machine-learning experiments** works through the saved experiment (its out-of-sample returns), not through a per-regime training of the model. Regime-conditional *models* already exist in the Machine Learning Lab; this layer does not drive them.
- **Charts are not synchronised by cursor or zoom** across charts, only by one shared window; each comparison and regime chart can be saved as a standalone SVG (colours copied in) as well as a table and a CSV.
- **Table virtualisation, request cancellation in the page and result caching** were not added: the pages here are small tables and one request each. The existing labs keep their own caches.
- **Cancellation is cooperative.** A backtest that is already running its pipeline cannot be stopped inside it; its result is discarded when it ends.
- **The regime lab estimates on an equal-weight market proxy** of the chosen assets, daily; it does not use macro, volume or intraday data, and a hidden Markov model with two states is a deliberately simple description.
- **Frequency is daily only**, because every engine is.
