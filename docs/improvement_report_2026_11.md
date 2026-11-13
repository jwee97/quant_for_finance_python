# Correctness and research-standard review (November 2026)

What this is: the sixteen improvement items of the platform review, each checked against the code, with what was found, what was changed, the evidence, and what is still open. The order of work was the review's: correctness first, then trading realism, then statistical validity, then reproducibility.

How it was done: the starting point was the requirement record of the machine-learning programme (`docs/ml_correctness_v3/TRACEABILITY.md`: 86 requirements, 76 marked verified and 10 in progress when this began). Its labels were not taken on trust. For each item of the review the code path was read and, where a check could be run, it was run: the same decisions through all three engines, the headline statistics against definitions written out by hand, the published formulas transcribed from the papers. **Five defects were found that way and fixed.** None of them was known to the record.

No backtest result is claimed to have improved. Three of the fixes change reported numbers, and each makes a strategy look *worse or the same*, never better.

## 1. Defects found and fixed

| # | Defect | Where | How it was found | Effect |
|---|---|---|---|---|
| 1 | **Maximum drawdown understated when losses start on the first day.** The drawdown was measured on a growth curve whose first point is already after the first return, so the starting capital was never a peak | 24 call sites; the shared helper is `features.returns.drawdown(curve, initial)` | Writing the drawdown as a loop from the starting capital and comparing | An account that loses 5% on each of its first two days reported −5% and is −9.75%. The out-of-sample segment of a walk-forward and any slice of a run are exactly such series. A drawdown can only get deeper with the fix; Calmar and recovery ratios fall with it |
| 2 | **Sortino ratio divided by the wrong downside deviation.** `backtest.metrics.sortino_ratio` averaged the squared shortfalls over the losing days only; the definition (Sortino and van der Meer 1991; Sortino and Price 1994) averages over every day | `src/backtest/metrics.py` (Backtest and Tune tabs); `features.returns.describe_returns` used a third definition | Two other modules of the platform used the definition, so one return stream had two Sortino ratios | The old figure was too low by `sqrt(days / losing days)`, about 1.4. This is the one fix that raises a number; it raises it to what the other tabs already reported |
| 3 | **Ledoit-Wolf shrinkage intensity omitted a term of the paper.** For the constant-correlation target the paper's `rho` has an off-diagonal part; the code kept the diagonal only and attributed the shortcut to the paper | `src/portfolio/covariance.py::_shrinkage_intensity` (every allocator and the ML risk model that use `lw_constant_correlation`) | Transcribing Appendix B of Ledoit and Wolf (2004) independently | The intensity was too high: 0.156 where the paper gives 0.093 on twelve of the platform's ETFs over one year (0.083 against 0.049 over three). The covariance entries move by about 1% |
| 4 | **The realistic execution mode refused its own fills for large fractional orders.** Share counts were compared with an absolute tolerance of 1e-12 | `src/ml/realism.py::OrderBook.fill` | Running the three engines on the same weights with lots of a millionth | A fill of six million shares in fractional lots raised `OrderError`. Whole-share runs were not affected |
| 5 | **A whole number of fractional lots was rounded down by one lot** (`0.3 / 0.1` is `2.999...` in floating point) | `src/ml/realism.py::fill_with_limits` | The same run | One lot left unfilled, waiting until its order expired |

Also corrected, without a change of behaviour: the docstring of `backtest.metrics.deflated_sharpe_ratio` now says that, without the trials' Sharpe ratios, it uses an approximation of the paper's formula (their variance is taken to be `1 / n`), and the function takes the variance when it is known.

## 2. The sixteen items, verified

Status words: **in place** (implemented and the check below ran and passed), **closed here** (was open, done in this pass), **partly** (the named gap remains), **needs data** (cannot be done with what the repository holds).

### P0: correctness

| Item | Status | What was verified, and where |
|---|---|---|
| 1. One accounting across the three engines | **closed here** | `src/backtest/engine.py`, `src/engine/engine.py` and `src/ml/execution.py` were given the same weights. With the assumptions matched they are the same book to 1e-12: the event engine and the adapter when both size at the decision; the adapter and the vectorized engine when both size at the fill; long-only and long-short; with and without a commission (fees equal to nine digits). The three differences between their *defaults* are now a tested, documented list: sizing at the decision or at the fill; the event engine's 0.5% borrowing spread on negative cash; a cost taken from the return or paid from cash. `tests/test_engine_reconciliation.py` (14 tests). Before this only the adapter and the vectorized engine were reconciled |
| 2. Look-ahead and survivorship | **partly, needs data** | In place and tested: dated membership and delistings, corporate actions, point-in-time liquidity screens, session-aware alignment, a feature-dependency audit (requirements P1-01 to P1-03, P0-14, P0-16). Not possible here: macro vintages (the platform holds the latest vintage only and labels those series `estimated`), dated sector classifications (the graph models say `current classification, not point in time`), and a survivorship-free stock universe. Each needs a data source the repository does not have |
| 3. Labels and validation | **in place** | Purging by the actual interval of each label, embargo, combinatorial purged cross-validation, preprocessing fitted on training rows only, out-of-fold lineage (P0-13, P0-17 to P0-21), each against a brute-force oracle. Horizons are configurable. Open: one executable-price convention (the close after the lag); intraday conventions need intraday bars |
| 4. Statistical testing | **closed here** | Sharpe, volatility, growth, Sortino, tracking error, information ratio and drawdown against definitions written out in the test; the probabilistic Sharpe ratio against the worked example of Bailey and Lopez de Prado (0.913, and 0.982 for normal returns); the deflated Sharpe ratio's two implementations against each other; the standard error of a Sharpe ratio by simulated coverage; a wider interval under autocorrelation (Lo 2002). `tests/test_performance_statistics.py` (12 tests). Defects 1 and 2 came from this. White's Reality Check and Hansen's SPA were read against the papers and are correct; their small-sample behaviour is measured in section 4 |
| 5. Mathematical implementations | **partly** | Checked against the publication in this pass: Ledoit-Wolf (defect 3), Almgren-Chriss, the Sharpe-ratio inference, the snooping tests, the out-of-sample R squared of Gu, Kelly and Xiu. Checked earlier by the programme: network architectures against textbook equations, the HMM forward recursion in log space, reward functions, output contracts (P0-03, P0-08 to P0-10). Not audited line by line in this pass: the option-pricing models and the 27 forecasters, which have their own reference tests |

### P1: trading realism

| Item | Status | What was verified, and where |
|---|---|---|
| 6. Optimisation aware of execution | **closed here** | `cost_aware_mean_variance` now takes the account's capital and the traded volume: `signal.max_participation` caps each trade at that share of the name's average daily dollar volume, `signal.impact` adds the square-root impact cost to the objective. Volume and volatility are those up to the decision date. Asking for it without volume is refused, never answered as if liquidity were unlimited. `tests/test_ml_liquidity_allocation.py` (8 tests, one through a whole experiment). Before, the optimiser chose a book as if any size could trade, and liquidity was applied afterwards by the execution mode (the gap recorded against P2-16) |
| 7. Execution simulation | **partly** | In place: spreads, slippage, partial fills under a participation limit, rejected and expired orders, halts and closed sessions, borrow limits and recalls (P1-04 to P1-10). Open: a position whose price has been missing or halted for longer than the policy allows is still carried at its last good mark; the write-down the policy describes is not applied by the adapter (P1-10) |
| 8. Multi-asset accounting | **partly, needs data** | The ledger books currencies, futures margin and rolls and perpetual funding, tested as fixtures (P1-12, P1-13); the dashboard's futures and crypto backtests charge rolls and funding. Open: the ML pipeline's realistic mode is single-currency cash equity; no multi-currency or futures strategy runs through it. An FX history and more than about 30 days of futures history are data the repository does not hold |
| 9. Risk management | **in place** | Expected shortfall, stress windows, factor exposures, concentration, drawdown and liquidity measures exist and are tested (`src/risk`, `src/ml/risk_models.py`, P2-04). Drawdown figures were wrong in the case of defect 1 and are fixed |

### P2: machine learning and research

| Item | Status | What was verified, and where |
|---|---|---|
| 10. Audit of the ML algorithms | **in place** | Baselines, calibration, ablation and multi-seed studies exist per model family (P1-16 to P1-24). Not repeated in this pass |
| 11. Reinforcement learning | **partly** | In place: an independent quantity ledger, hard constraints with forced actions recorded as events, realistic rewards, checkpoints, evaluation on untouched data. Open: stepping forward one day at a time with a persisted recurrent state; every call replays its chain from the start, which is deterministic but costs time in proportion to the history (P0-02) |
| 12. Replication benchmarks | **closed here** | `src/validation/replication.py` and `python -m src replicate`: eleven benchmarks, each labelled. **Exact** (to rounding, against an independent transcription or the paper's printed number): the Ledoit-Wolf intensity and matrix; the probabilistic Sharpe ratio example; the Almgren-Chriss trajectory (the solver's path against the closed-form hyperbolic sine, to 3e-15) and frontier. **Simulation** (a statistical claim checked where the truth is known): shrinkage lowers the squared error; the deflated Sharpe ratio does not pass the best of 50 noise strategies; the probability of backtest overfitting is one half under noise; the snooping tests hold their level at 1,000 days; SPA keeps its power when poor alternatives are added and the Reality Check does not (0.64 against 0.00). **Approximate**: the expected maximum Sharpe ratio is the paper's own approximation, 2% high at ten trials. **Method only**: Gu, Kelly and Xiu. Their data (CRSP, 1957 to 2016) is not available, so nothing of their empirical tables is reproduced; their simulation design is, and on it least squares has a negative out-of-sample R squared (−3.8%), ridge a small positive one (0.1%) and boosted trees recover 4.3% of a true 5%. `tests/test_replication.py` (10 tests) |

### P3: research infrastructure

| Item | Status | What was verified, and where |
|---|---|---|
| 13. Reproducibility | **in place** | Every experiment records the dataset fingerprint, the code commit and source hash, parameters, seeds, library versions and the model artifact's hash; two runs can be compared component by component (P0-23, P0-24) |
| 14. Automated testing | **closed here in part** | Added: cross-engine reconciliation, independent-reference tests of the statistics, the replication benchmarks. The blocking CI lanes run them. Open: the scheduled heavy workflow (the 200-replication null study, multi-seed RL) has not run in Actions (P0-29) |
| 15. Forward testing | **in place** | Incremental predictions from a frozen bundle, drift detection, champion and challenger with rollback, a prospective ledger (P1-25, P3-02). The Signals tab keeps an append-only record with a paper portfolio |
| 16. Reporting | **in place** | Reports separate training, validation, test and forward segments, gross and net, turnover, costs, capacity, uncertainty and benchmarks; holdouts are protected and every look is logged. Rejected hypotheses are kept: an experiment that fails a gate stays in the registry with the reason |

## 3. Plan for what is open, in dependency order

| Order | Work | Depends on | Needs |
|---|---|---|---|
| 1 | Write-down of stale and over-long-halted positions in the adapter (item 7, P1-10) | the mark policy (done) | Code and fixtures. The independent gross P&L of the adapter must take the write-down too, or the run will not reconcile |
| 2 | Stepwise RL stepping with persisted recurrent state (item 11, P0-02) | the state schema (done) | Code; chunked, single-pass and stepwise replay equal to 1e-9 |
| 3 | Macro release instants in the loader, then a fundamentals feature group joined on filing dates (item 2) | 1 | Code and fixtures |
| 4 | Selectable daily execution conventions on open-high-low-close (item 3) | nothing | Code and fixtures. Intraday needs data |
| 5 | A multi-currency realistic mode, then futures and funding modes through the pipeline (item 8) | 1, 4 | Code and fixtures; an FX history and a longer futures history are data |
| 6 | The liquidity settings in the dashboard's form (item 6) | done in the API, the configuration and the library | A form field; today they are set through the API or a configuration file |
| 7 | The scheduled heavy workflow (item 14) | all it schedules | A run on the repository's Actions runner |

## 4. Limitations and risks that remain

- **The snooping tests reject too often in small samples.** Measured with 500 studies of ten skill-less candidates at the 10% level: over 250 days the Reality Check rejected 12% (blocks of one day), 13% (five) and 16% (21), and the SPA test 12%, 15% and 23%; over 1,000 days with blocks of five, 9% and 10%. The implementations follow the papers; this is the bootstrap with long blocks on a short sample. The selection layer of the ML tab already uses short blocks (`max(5, T^(1/3))`); the low-level functions default to 21 and their docstrings now say when that is too long. A result that passes on one year of data with long blocks deserves less trust than its p-value says.
- **The deflated Sharpe ratio of the Backtest tab is the approximate form** unless the trials' Sharpe ratios are supplied. It treats trials as independent, which sets the bar higher than the exact form.
- **Numbers reported before this change are not comparable for drawdown, Calmar, Sortino and anything using the constant-correlation shrinkage.** Stored experiments keep the figures they were saved with; a re-run gives the corrected ones.
- **The three engines agree where their assumptions are made to agree.** Their defaults still differ (section 2, item 1). A strategy moved from one to another will show a different result for those documented reasons.
- **Nothing here validates a strategy on survivorship-free data.** The universe is ETFs, futures and crypto that exist today, plus files the user brings.
- **The liquidity-aware allocation uses daily volume.** It knows nothing of intraday liquidity, auctions or the spread at the time of trading.
- **Gu, Kelly and Xiu's empirical results are not reproduced** and could not be with this data.

## 5. Evidence

Every figure below is from a run on an isolated copy of the final tree, not from memory.

| Check | Environment | Result |
|---|---|---|
| Lint (`ruff check src tests experiments`) | | clean |
| Documentation (`mkdocs build --strict`) | | builds |
| The new and the changed test files | Python 3.11, pandas 3.1 | 219 passed, 1 skipped (dask not installed) |
| Numerical lane (78 files) | Python 3.10, pandas 2.3 | 1,853 passed, 1 skipped. On the first run in a fresh copy one test failed (1,852 passed): `tests/test_option_models.py`, a command-line test that requires an empty standard output, ran before any other test had built the cache of processed prices, and building it writes log lines. It passes alone and the lane passes on a second run. It does not depend on this change, and CI holds no raw prices, so it cannot occur there |
| Dashboard and machine-learning lane (147 files) | Python 3.10, pandas 2.3 | 2,830 passed, 3 skipped (optional catboost) |
| Machine-learning correctness lane (93 files) | Python 3.11, pandas 3.1 | 1,704 passed |
| Browser lane (22 files, Chromium) | Python 3.11, pandas 3.1 | 205 passed |
| Replication benchmarks (`python -m src replicate`) | Python 3.11 | 11 of 11 held: 4 exact, 5 simulation, 1 approximate, 1 method only |

Tests whose expected values changed because the old value was the defect: two in `tests/test_performance_metrics.py` (Sortino), one in `tests/test_returns_dates_exact.py` (downside deviation and drawdown), one in `tests/test_covariance_exact.py` (it kept "the diagonal terms only, as the implementation says"), and one float-equality in `tests/test_portfolio_allocators.py` made a tolerance. Of 1,237 existing tests that mention a drawdown or a downside measure, 1,235 passed unchanged: none of them used a series that loses from its first day, which is why the drawdown defect had not been seen.

## 6. References

- Almgren and Chriss (2000), "Optimal execution of portfolio transactions", *Journal of Risk* 3(2).
- Bailey and Lopez de Prado (2012), "The Sharpe ratio efficient frontier", *Journal of Risk* 15(2).
- Bailey and Lopez de Prado (2014), "The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality", *Journal of Portfolio Management* 40(5).
- Bailey, Borwein, Lopez de Prado and Zhu (2017), "The probability of backtest overfitting", *Journal of Computational Finance* 20(4).
- Gu, Kelly and Xiu (2020), "Empirical asset pricing via machine learning", *Review of Financial Studies* 33(5).
- Hansen (2005), "A test for superior predictive ability", *Journal of Business and Economic Statistics* 23(4).
- Ledoit and Wolf (2004), "Honey, I shrunk the sample covariance matrix", *Journal of Portfolio Management* 30(4).
- Lo (2002), "The statistics of Sharpe ratios", *Financial Analysts Journal* 58(4).
- Sortino and Price (1994), "Performance measurement in a downside risk framework", *Journal of Investing* 3(3).
- Sortino and van der Meer (1991), "Downside risk", *Journal of Portfolio Management* 17(4).
- White (2000), "A reality check for data snooping", *Econometrica* 68(5).
