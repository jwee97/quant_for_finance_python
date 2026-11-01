# Assumptions and limitations (Releases 1 and 2)

## What the RL ledger models, and what it does not

- One currency, cash-equity accounting: cash, quantities, marks, a proportional cost plus an optional quadratic impact term, financing on debit cash and a borrow fee on short notional (both default to zero).
- Not modelled: futures margin and variation settlement, contract multipliers and rolls, FX and settlement calendars, corporate actions, borrow availability and recalls, partial fills, order rejection by a broker. Release 4 routes evaluations that
  need them through the existing event-driven engine and ledger.
- Prices are synthetic marks built from returns (the ledger starts at 1 and compounds returns); a return below -100% is a data error and stops the price at zero.
- The agent's private ledger starts flat at the first day of each chain. The platform backtest then charges the real transition between consecutive folds' books. The two are not the same account.
- Rewards are training signals, in percent per session. They do not sum to final equity.

## Data

- The ML futures universe is simulated and stays labelled so. Nothing in Release 1 uses it as evidence of a strategy.
- Sector and asset-class relationships used by the graph models are the dataset's current static map. They are not point in time, and the recorded relationship version says so (`static_group_map:<hash>`; `current classification, not point in time`).
- No historical constituents, borrow data, costs or fills were invented. None are used by Release 1.

## Statistics

- Release 1 makes no statistical claim about any strategy. Calibration quality, selection-bias control and uncertainty are Release 3 to 5 items.
- The HMM filter is checked against an independent log-space recursion; the Markov-switching filter is checked only against statsmodels' own. Both engines' fits start from random parameters, and the Markov-switching fit now passes its seed to statsmodels
  (which ignores numpy's global seed) and retries three seeds.

## Optional dependencies

- NGBoost depends on lifelines, which requires pandas below 3. On Python 3.11 or later it cannot be installed beside pandas 3, so the blocking `ml-correctness` job leaves it out and its serialization-matrix family is recorded as
  `NOT_APPLICABLE`; the Python 3.10 job runs it. A family skipped for any other reason makes its requirement `blocked`, not `pass`.
- The RL checks need Gymnasium and Stable-Baselines3; without them the environment tests skip, the ledger and reward tests still run, and the requirement is `blocked`.
- The GPU path has never been run.

## What this release does not do

Study lineage, holdout reservation, null controls, selection-test eligibility and the gate evaluator (Release 3); execution realism (Release 4); statistical and prospective validation (Release 5); alpha research,
strategy-improvement experiments and the advanced modules (Releases 6 to 8). Gate A, B and C are all `pending` or `not started` for every model.


# Release 2: causal data and training integrity

## Labels and calendars

- The label convention is daily close to close (`CONVENTION`). Intraday bars and executable-price conventions other than the close are not implemented; the convention is recorded, not selectable.
- Holidays are the data's own: a date with no row is not a trading day. Where a closure is carried as a price (the futures bundle with `hold_days`), the cell is a mark and is not tradable. The other loaders carry nothing over.
- The fold plan (`segment_plan`) cuts the history by fractions of the number of labelled dates, so appending data moves the fold boundaries. Prediction causality is a statement about a fixed plan on data of fixed length; the tests change values, never the length.
- Purged K-fold and CPCV exist as functions on label intervals. They train on later blocks and are research diagnostics (`chronological=False`); no experiment uses them yet.

## Point in time and sessions

- The platform has no macro vintage history: FRED serves the latest vintage, so the macro series are reported as `estimated` (a fixed release lag) except market prints and series computed from prices. Validation claims that rest on `estimated` or `unavailable` inputs are marked restricted. Nothing was backfilled or invented.
- The ML feature library has no fundamentals group. The fundamentals engine is reused for the availability join and its quality vocabulary is mapped, but no fundamentals feature enters a model yet.
- Sessions are a fixed table with daily bars. A ticker with no suffix, index symbol or asset class hint is assumed to trade US equity hours, and the report lists those assets. The macro loader applies release lags in days; release instants exist (`sessions.release_instant`) but it does not use them.
- Tokyo's close is modelled at 15:30 JST (the time since 5 November 2024). Before that the exchange closed at 15:00, so the model treats the bar as final half an hour later than it was: conservative, never early.

- **Returns at a missing price depend on the pandas version.** `bundle_from_prices` and `MarketData.returns()` use `DataFrame.pct_change()`. For prices `[1, NaN, 3, 4]` pandas 2.3 returns `[NaN, 0, 2, 1/3]` (it pads, so the day after a gap carries the whole move) and pandas 3.1 returns `[NaN, NaN, NaN, 1/3]` (that move is dropped). On prices without gaps the two agree, which is the case for every fixture of this release and for
  the platform's own cleaned data (small gaps are filled when it is built); a user-supplied series with a missing day inside it would differ between the two versions. This is a platform-wide definition, outside this release, and is listed as a next action; the features of this release name `fill_method=None` explicitly.

## Training and inference

- Perturbation checks run through six kinds (ridge, hmm_regime, lstm, gcn, model_ensemble, ppo). The other kinds of each family share the class and the binding but are not each run through the whole-walk-forward comparison.
- The pipeline has no data-driven feature-selection, dimensionality-reduction or probability-calibration stage. If one is added, it must fit on the eligible rows and be added to the isolation test.
- One-row-at-a-time inference equals the batch for the models without state through a call. The ensemble's dynamic weights and the PPO account carry state through a call: for them, a prefix of the block and a live replay (a view that ends on the decision date, given the block so far) give the batch values; a single isolated row would start the account flat.
- Sample weights are used by the linear, boosted-tree and regime models only; the others report `requested, not applied`. Sequential bootstrap is not implemented.
- The registry's `register_model` still stores plain model bytes; the inference bundle is written beside the experiment's model file. Experiments saved before this release have no bundle, no fold lineage and no audit, and say `not recorded`.
- A forward signal for a date the saved model had already seen is flagged as a replay and counted; it is not refused, because replaying history is a legitimate way to look at a model.

## Fixed in passing (behaviour that changed)

- Strategy combination by online learning back-filled the first volatility from later days and filled gaps with the whole-sample volatility; it now makes no update until 20 days of trailing volatility exist.
- The regime model's diagnostics described the prediction block as training data; they now stop at the fit window.
- `idio_vol_63` declared a 63-day lookback but needs 126.

# Release 3: research controls, reproducibility and protected holdouts

## What the controls protect, and what they do not

- They govern the application's research workflow: the API, the command line and the Python functions that go through `run_experiment` and the evaluation service. An owner of a local checkout can still edit the SQLite file
  or call lower-level model code; stronger isolation needs process and storage permissions. The append-only tables refuse updates and deletes through SQL triggers, which stop accidents and casual rewrites, not a determined owner.
- A holdout is *reserved* only when a plan reserves it. History that no plan covers is not protected: a direct run on it is allowed, recorded as an exposure and labelled exploratory, and that exposure makes a later reservation
  of the same history not untouched. The default of `POST /api/ml/train` stays `full` (every segment) so existing clients keep working; `mode: development` is the validation-only mode. This is a deliberate compatibility
  choice, recorded here rather than hidden: the specification asks for development mode by default.
- Overlap is found by shared assets and intersecting dates, and by identical outcome content. Two histories that are economically the same but share no asset name and have different returns (an index and its ETF) are not.
- `finalize: true` (deprecated) evaluates once as a *recorded re-use* when the history was already read, so the legacy flow keeps working; such a result can never count toward Gate B.
- A claim that crashed leaves the holdout consumed: outcomes may have been read, and nothing proves they were not. It can be resumed with the same key.

## Counts

- Raw look counts count recorded tries (done, failed and cancelled). They are not estimates of independent trials. The legacy `trial_count` (experiments + search trials + tombstones) is still what the deflated Sharpe ratio
  of a report uses; the study's raw count is reported beside it (`trials.study`). They can differ: the study counts feature searches, derived runs and combinations that the legacy count does not.
- A study is by default the universe, not the cache key. Two unrelated hypotheses on the same universe share a count unless `study` separates them; that errs towards more deflation.
- Experiments saved before Release 3 have no lineage, provenance, holdout status or gates evidence and say `not recorded`; none is reconstructed.

## Provenance

- The identity proves that the recorded inputs are the ones used; it does not prove they were right. Hashing covers the bundle the dataset was built from (when it was built in this version) and the dataset's own frames;
  a dataset built before Release 3 has no bundle manifest and its identity rests on the dataset hashes alone.
- Repeat-run tolerance is zero for the deterministic CPU families and a stated small number for neural and reinforcement-learning ones (`provenance.TOLERANCES`); a GPU run is looser. These are policies for comparing, not guarantees.
- The environment hash includes package versions; a different BLAS or CPU can change floating-point results without changing any version.

## Selection procedures

- Each procedure is applied to the validation net returns of the study's recorded trials that kept a stream (`trials/<study>/`). Trials run before Release 3, failed trials and trials of other studies have no stream; the
  procedure says how many tries it saw against how many were recorded. A study with no streams reports every procedure `unsupported`.
- DSR treats returns as serially independent in the standard error (no autocorrelation correction); PBO needs at least 3 candidates and 4 blocks of 8 observations; the Reality Check and SPA use a stationary bootstrap with a
  block length of about T^(1/3); FDR uses Benjamini-Yekutieli (valid under any dependence) for the pass criterion and reports Benjamini-Hochberg beside it. All test the validation sample on which the choice was made, not the holdout.
- The criteria (alpha 0.05, DSR probability 0.95, PBO at most 0.5) are parameters stated with each result; they are a policy and can be changed in a plan.
- Feature screens adjust the p-values of the rank-IC t-statistics for the number of features *tested*, rejected ones included (Holm or Benjamini-Yekutieli; Benjamini-Hochberg is refused because correlated features violate its assumption).
  Features tried in other screens of the same study are separate looks in the lineage.

## Null controls

- The placebo shifts the returns of all assets together against the unchanged features (circularly, with one zero-return seam). It breaks the feature-return link and nothing else; it is not a null for volatility
  forecasts, replications share the same features and are not independent samples of the market, and the interval treats them as independent draws of the procedure given the data.
- A study with fewer replications than its predeclared minimum, or an interval wider than its predeclared precision, validates the plumbing only. The `ml-correctness` lane runs a handful; the scheduled `ml-heavy` workflow runs 200 on
  a simulated market with no signal. That is evidence about the machinery, not about any real strategy.
- The default pipeline under test is the platform's staged search over a few model kinds with the DSR and SPA gate and a one-sided test on the placebo test segment. A different pipeline needs its own study.

## Gates

- Gate A is "implemented as specified", not "profitable". The family requirements are the `ml-verify` ones listed in `gates.family_requirements`; a requirement not registered in this checkout is `missing`, never skipped.
- Gate B cannot pass in this release: the realistic-execution evidence (Release 4) and the statistical-diagnostics evidence (Release 5) have no providers and report `blocked`. Gate C additionally needs prospective evidence and the
  operating controls, which no release before 8 provides. The tests exercise the logic with injected providers; no strategy has earned any gate.
- A stale decision (the code, data or policy changed) is shown as pending. The source-tree hash invalidates the provenance item of every experiment run before the change, which is intended.

## Fixed in passing

- The deflated Sharpe ratio of a finalised search counted every search trial twice (the registry already held them and `trials_extra` added them again). The API and `search` now pass the trials the registry does not hold.
- The tuning verdict's comparison of the best and starting settings needs the evaluation: without `finalize` it is not computed, and the response says why.
