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
- Gate B needs, besides the frozen plan, the selection controls and the null study, the realistic-execution evidence of Release 4 and the statistical diagnostics of Release 5 read against an **adopted validation policy**; with no policy adopted it is `blocked`, and the execution item is itself blocked for any result on simulated or unknown data, for a long-short book with no borrow records and for a result
  with no volume. Gate C additionally needs genuine prospective evidence (elapsed time that cannot be manufactured) and the operating controls of Release 5. The tests exercise the logic with injected providers and fixture rows; no strategy has earned any gate.
- A stale decision (the code, data or policy changed) is shown as pending. The source-tree hash invalidates the provenance item of every experiment run before the change, which is intended.

## Fixed in passing

- The deflated Sharpe ratio of a finalised search counted every search trial twice (the registry already held them and `trials_extra` added them again). The API and `search` now pass the trials the registry does not hold.
- The tuning verdict's comparison of the best and starting settings needs the evaluation: without `finalize` it is not computed, and the response says why.


# Release 4: executable trading and reconciled accounting

## What the realistic mode is, and is not

- It is a **cash-equity, single-currency, daily-bar** evaluation: orders decided at one close are filled at a later close at a modelled price. Opening-auction and intraday liquidity, queue position, latency, hidden liquidity, short squeezes and market-wide halts are not represented; a same-bar fill is refused because it needs an auction model that does not exist.
- The marks and the fill reference are the dataset's **research closes** (vendor-adjusted where the source adjusts them). No executable (unadjusted) price series and no corporate-action events come with the platform's data, so splits, dividends and mergers are not accounted for as events unless a caller supplies them (`market={"actions": ...}`); the report says so. Dividends in an adjusted series are reinvested implicitly.
- **Impact** is the square-root law of the event-driven stack with a coefficient that is an **assumption**, never a measured cost, because no calibration data exist (`impact_label` says `assumed (uncalibrated)`). Spread and slippage are proportional to the price and default to the backtest section's own costs; there is no observed spread.
- **Capacity** is measured at predeclared capital levels on the last held target, not over the history, from the prior average dollar volume (volume times the research price, which a split in the history can mis-scale). With no volume capacity is **not established** and the claim is blocked; with `max_participation: null` every order is assumed to fill in full and that is stated.
- **Short selling** needs borrow records. The platform has none: a long-short result is blocked at Gate B unless a `BorrowBook` is supplied or `shorts: refuse` is chosen (then the book is long-only where the strategy asked for shorts, and the report says so). The scenario fee of a loose book is visible and never described as a record.
- **Membership, halts and delistings**: eligibility acts at decision dates (weekly with the default rebalance), so a security that is delisted between two decisions needs a `delisting` action to be settled on its date. The stale-expiry and over-long-halt write-down of the mark policy is not applied by the adapter, which carries the last good mark (P1-10 is `in_progress`). The built-in universes carry no dated membership and are today's survivors.
- **Other asset classes**: multi-currency books, futures rolls with variation margin and perpetual funding are reconciled as **fixtures** against independent arithmetic (P1-12, P1-13), not run through the pipeline. The ledger books cash on the trade date; settlement dates are computed and reported (`FXBook.unsettled`), not simulated.

## How it differs from the vectorized engine

On the shared subset (the same weights, mask and volatility target, the same lag applied once, the same rebalance dates, close-to-close prices, costs proportional to traded notional) the two return the same stream (tested to 2e-9 at lags 1 and 2). They differ, by design, in: constant-gross renormalisation of the drifted book between rebalances (the engine) versus drifting shares (the ledger); whole shares, lots and the cash they leave;
participation limits and expired orders; impact when volume is available; interest on cash and the financing of borrowed cash; and, with `sizing: decision_close`, an order sized in shares at the decision close. A comparison block in every realistic report shows both numbers on the same days; the difference is not an error in either.

## Other limits

- The in-sample (train-segment) series and the robustness tests of a realistic run still use the vectorized engine, and the report says so.
- A realistic run is a look at the same outcomes as its parent: it is a counted `portfolio_variation` of the study and inherits the parent's holdout state; running it never makes an exposed history untouched.
- `GET /api/ml/trading-diagnostics` reads an experiment saved before execution modes existed as `vectorized_legacy`, flagged `recorded: false`.

## Fixed in passing

- Ineligible shorts: the first version of the adapter rejected the purchase that covers a short in an ineligible asset (it treated every buy as an addition). Exiting is always allowed; what is rejected is a request for more exposure than the book has.
- Orders with a lag of more than one session were cancelled by the next day's decision before they could fill; orders are now target positions queued per asset.


# Release 5: statistical and prospective validation

## What the numbers are, and are not

- **No threshold is implied.** Whether a deflated-Sharpe probability of 0.9, a PBO of 0.4 or sixty settled days is enough is a decision, made by a named person before the evaluation and recorded as a versioned policy. The illustrative example policy is a documented sketch of the *kind* of statement a policy makes; its numbers are not calibrated to any market or strategy. Nothing in the platform approves a strategy on
  its own: a correct implementation that loses fails Gate B and passes Gate A.
- **The person is a name, not an identity.** The platform has no authentication, so `actor` on an adoption, a halt, a rollback or an approval is the text the caller gave. The audit keeps it and the time; it does not prove who typed it. A deployment that needs that proof must put the API behind an authenticated front end.
- **Sharpe-ratio intervals.** The HAC interval is a delta-method, asymptotic statement that needs a long sample to estimate fourth moments; the Newey-West lag is a rule of thumb (a horizon of h forces at least h-1 lags); the bootstrap interval is a percentile interval, not bias-corrected, with Monte Carlo error that is recorded as the number of replications (at least 200). The naive interval is shown only to show what it ignores.
  Intervals describe sampling uncertainty of this sample; they say nothing about whether the sample resembles the future.
- **Matched benchmarks.** The platform has the equal-weight market and no other traded factor; any other factor is whatever the caller supplies. The sector-matched comparison uses the dataset's current group map (not a point-in-time classification). Exposures are those of the *target* weights through the vectorized engine's lag and schedule, not the executed holdings of a realistic run (the record says so). A beta estimated on 252 days is itself uncertain; the
  matched comparison removes the part of the return explained by that beta, no more.
- **Regimes.** The default definitions (trend over 63 days at plus or minus 5%, volatility terciles of the history seen so far, liquidity stress against a 252-day median, sideways) are conventional, not calibrated. A regime with fewer days than the policy's minimum is insufficient evidence and has no number. Crisis windows are retrospective slices chosen with hindsight and are labelled descriptive. The liquidity regime needs traded volume
  and is insufficient without it.
- **Selection tests** keep the limits of Release 3 (a raw count of tries is not a complete correction; the DSR standard error does not correct for serial dependence; CSCV treats blocks as exchangeable and is not a chronological validation; Reality Check and SPA test the family of recorded candidates only; Benjamini-Yekutieli is the criterion). A candidate whose return stream is constant has no Sharpe ratio: it is excluded from the statistics, the exclusion is
  recorded, and it still counts among the tries.
- **Neighbourhood, delay and capacity** re-run the *saved predictions* with other signal, backtest or execution settings. They do not refit the model, so the neighbourhood of a model hyper-parameter is not covered (the search and tuning flows record every trial). Each point is a child experiment and a counted look. The capacity analysis is the realistic cash-equity mode with an assumed impact coefficient.
- **Calibration and conformal intervals.** Brier, log loss, reliability (Wilson intervals), Platt and isotonic calibrators fitted on earlier data, pinball and interval scores, a rolling split conformal procedure and Adaptive Conformal Inference exist as library functions with independent tests. The rolling split procedure is labelled as having no time-series guarantee; ACI's bound is a long-run average, not a per-period coverage. The diagnostics run reports the class probabilities' scores by segment and, when the model
  stored a predictive distribution (the quantile models, NGBoost, the Temporal Fusion Transformer, the ensemble), the interval's coverage with its Wilson interval, width, interval score and pinball loss for the diagnosed segment, beside the two conformal procedures run asset by asset and calibrated only on forecasts whose outcomes had ended (horizon plus lag sessions earlier). The conformal intervals are centred on the stored centre forecast (a quantile model's median, a Normal model's mean) and are symmetric; a model whose interval is asymmetric is scored on its own interval and compared with a symmetric one. Forecasts before the first full calibration window have no conformal interval and are not scored.
- **Nested selection** is a tested procedure with an audit that recomputes its facts from the label intervals. The search and tuning flows are not yet run through it.
- **Multi-seed evaluation** reports the median over predeclared seeds and the spread across them beside each seed's time-series uncertainty; it never chooses a seed, and each seed is a counted look. It was exercised on non-reinforcement-learning families: the agents' deterministic policy evaluation depends on the optional packages and is not verified here.

## The prospective ledger

- A commitment is **prospective** only if it was written before the first session whose price its label needs, with data and a fitted model that already existed. The server's clock decides, so the daily step must run before the next session opens anywhere in the universe; a late step is a replay. The first label session is taken on a Monday-to-Friday calendar, which can classify a commitment made on a holiday as a replay although no session
  had traded (conservative; never the other way). History built from simulated or past data is always a replay, whatever it is called.
- Elapsed time cannot be manufactured: a policy's minimum number of settled prospective days is met by waiting, and replays are listed beside the count and never added to it.
- The equity of a prospective strategy is built from the committed holdings on the platform's daily research closes by the Release 4 adapter (whole shares, costs, cash). It is not a broker statement. A forecast horizon plays no part in it.
- A settlement is the realised return of each asset over the label window as the platform's data shows it on that date. A later revision of the data is a new settlement version with its reason; both evaluations are kept.

## Governance

- Limits are enforced where the platform commits a book (`commit_forecast`) and where it promotes a model (`Registry.promote`). The platform places no orders, so nothing stops a person who trades outside it.
- Monitoring is the population stability index of the features and the scores (alert at 0.25), the live rank IC against the backtest's (alert below two standard errors) and the limits on the committed holdings' returns. A breach of a hard limit halts the version; an alert does not.
- A strategy version's identity is the model artifact with the dataset, model, signal, execution and risk settings. A different limit is a different version and its evidence (commitments, monitoring, approval) starts again. The **current** version of an experiment is the latest one *created*; registering an identical identity again returns the existing version and does not make it current.
- The staleness limit is measured against the server's clock: a version with `max_staleness_days` that is monitored long after its last decision (a replay of old history, a forward test left idle) is halted by the check, which is what the limit is for.

## Fixed in passing

- A rejected conflicting commitment was audited inside the transaction that then raised, so the audit row was rolled back with the error; found by the test, now written before the error is raised.
- A percentile bootstrap interval with exactly 100 replications came back as `applied` with missing bounds; fewer than 200 is now `insufficient`.
- DSR, FDR, PBO and SPA returned `applied` with undefined statistics (or an unexplained failure) when a candidate's returns were constant; such candidates are now excluded and the exclusion is recorded.
- `ForwardTracker.summary` compounded each signal's horizon return divided by the horizon into a "cumulative" and a Sharpe ratio. That is not a daily equity curve; it is removed and replaced by the equity of the committed holdings.
- The neighbourhood analysis accepted a setting that no section has and reported a flat, flattering distribution; it now refuses it.
- Gate B's selection item no longer judges a result against the platform's built-in selection thresholds once a policy is adopted: the policy's thresholds do, and the evidence is recorded under them.

# Release 6: alpha research foundation

## What the diagnostics and comparisons are, and are not

- **Development evidence.** Alpha and portfolio diagnostics and every arm of a comparison read the validation segment. The test and forward segments are refused unless a frozen plan evaluated them once. A figure computed here is a description of development data; it can motivate a hypothesis and cannot support a claim about the future. A candidate called `supported_on_development` has only cleared its declared bar on that segment, once, with the multiple-testing correction over its own card: other cards on the same data are counted by the study, not by the card.
- **The rank IC is against one label.** Whatever target a model was trained on, the comparison measures its scores against the raw forward return of the baseline's horizon. A sector-relative or residual model is therefore judged on what a portfolio would have earned, not on how well it fitted its own target; each arm's IC against its own target is shown and never compared. For different horizons the ICs are not commensurable; the comparison is on the daily net returns of the book each horizon implies.
- **Relative targets.** The sector-relative target uses the bundle's group map, which is one static assignment applied to every date: it is not point in time, and the target's documentation says so. The residual target removes the benchmark's exposure with a loading estimated from the trailing 252 days up to the decision date (at least 120); it removes one factor and nothing else, because the platform has no point-in-time size, value or momentum factor returns.
- **Stable selection.** It is run on each fold's training rows only, with contiguous blocks of dates resampled. The published bound on the expected number of falsely selected features is reported with the result and is only below one at strict settings; at loose settings the report says the choice carries no error control and is a description of the training window. A fold with no stable feature is skipped, not fitted on everything.
- **The conditional IC and the blend** measure what a new signal adds to the signals it is conditioned on, on one sample. They are not causal and not a feature importance, and they do not condition on factors: no factor returns exist here beyond the equal-weight market.
- **The optimiser.** It maximises an expected return minus risk and cost in the units of one holding window; it does not calibrate the forecast it is given (a ridge prediction is a return forecast only in the platform's declared sense), it assumes daily returns are uncorrelated through time when scaling the covariance to the window, and it does not model integer lots, borrow or margin (the realistic execution mode applies them to its target). A cost large enough to remove every edge produces a book of zero, which is a result. The perturbation grid shows how the optimal book moves; it does not say which setting is right.
- **Covariance.** Estimators are compared on the risk a book actually realised over the next 21 days (or the horizon given), which is a statement about those dates. A repair of a non-positive-definite matrix is reported with its size.
- **Tranches.** The platform's holding period averages the last h daily books, which is h sleeves of 1/h of capital; during the first h - 1 days the average is over the sleeves that exist. A forecast produced every day is not h independent portfolios, and the combined book never carries more gross than a daily book.
- **Not done in this release:** the optimiser does not see capital or volume inside the pipeline (P2-16 is `in_progress`); the simplicity tie-break applies to comparison cards, not to the search or tuning winners (P2-17 is `in_progress`).

## Fixed in passing

- Candidates of a comparison are changes laid over the baseline, not standalone configurations: the first version compared a candidate with the defaults and reported everything the baseline had set as a difference.
- A constant paired difference (two arms that differ by a fixed amount) has no long-run variance and is `insufficient`, not a significant result.
- `conditional_ic` gave a signal the existing signals reproduce exactly the state `tied_signal`; it is now `explained_signal`, a different condition, and its IC is unavailable, not zero.


# Release 7: controlled strategy-improvement experiments

## What the overlays, ensembles and comparisons are, and are not

- **Development evidence.** Every arm of a comparison is a development run: the test and forward outcomes are not predicted, the verdicts are `supported_on_development`, `worse` or `inconclusive`, and the gate decision of a card is always `not_evaluated`. A gain here motivates a frozen evaluation; it is not that evaluation, and no strategy is promoted by it.
- **One change, then combinations.** A combination is built only from components that were each compared on their own, and its multiple-testing family includes theirs. Combining the components that looked best is still a selection among the components, and the card says so.
- **Arms that act after the forecasts re-run the baseline's saved predictions.** The two alternatives share forecasts, folds and risk model; that is why a no-trade region or a regime exposure is compared fairly, and it means the card measures the rule and not a retrained model.
- **No-trade region and trade filter.** Both read the model's score as an expected return over the dataset's horizon, so they refuse a rank or a probability. The no-trade region applies at the engine's rebalance dates, weighs utility under the platform's own covariance and costs, and has no market-impact model. The filter's error scale is the RMS forecast error of a trailing window of settled decisions; with no settled decision there is no estimate, nothing is filtered, and the count of such decisions is reported. A lower coverage is not a better result, and the log says how much diversification the rejections cost.
- **Calibrated sizing.** The score is mapped to return units by a cross-sectional slope estimated walk-forward on settled outcomes with a Newey-West error; with no demonstrated edge the book is flat, and the gross is whatever the edge and the risk imply (never scaled up). It does not make a probability calibrated.
- **Regime exposure.** States are filtered, never smoothed, and are numbered by volatility at each refit; the budgets are read from what the unscaled book earned the day after each state, over the rows before the refit. A state with too little evidence gets the neutral exposure, a model that cannot be fitted leaves the exposure neutral, and the overlay only lowers a book. It needs `regime_min_train` rows of the book's own history before it acts; because the book exists only where the experiment predicted, a short development-mode sample may never reach that, and `warmup_days` says so. A drawdown gain is read against a volatility-matched drawdown: a smaller position is not better timing.
- **Stacked ensemble weights** are fitted on the held-out stage, whose base predictions are out of sample, by non-negative least squares shrunk toward equal weights; a member that scores less than the coverage floor of the held-out rows carries no weight. They are static between refits, and `stacked_regime` fits them per regime with the pooled weights where a regime has fewer than 20 held-out dates.
- **The development-selected single model** is the best of N on the validation segment: an optimistic estimate of that model, so an ensemble that beats it has beaten a baseline selected in its favour. The regime-conditioned mixtures were tested on return streams built to have that structure and on noise; no claim is made about real markets.
- **Meta-labels** treat a bet as right if it earned more than the primary's declared one-way cost paid twice; that is not the itemised cost of each bet. The comparison of the filtered strategy with its base describes a pair; it is not a controlled one-change test.
- **Delay report.** A delay is a number of sessions on the dataset's own index, reported in calendar days; the book that executes is the one decided at the observation close, and no rule re-reads the signal later. Net results are the vectorized engine's on the saved book. Capacity is the share of the book one session's participation limit lets through, at the delayed session's liquidity, and needs a volume record and a limit; it is `not_established` without them.
- **Not done in this release:** P2-16 (capital and liquidity inside the optimiser's choice) and P2-17 (the tie-break on search and tuning winners) remain `in_progress`, and no real-data evidence that any improvement holds was produced.

## Fixed in passing

- A card's changed settings listed the settings the configuration fills in (`cost_bps` and the others); they are now listed apart as `derived`.
- Two components whose cards generated the same candidate name had the same ablation name; the card id now tells them apart.
- A member that scored too few held-out rows made an ensemble's mean IC an empty-slice warning and could blank every prediction; it is now left out below the coverage floor and its rows are renormalised.

# Release 8: optional advanced research and operations

## What the advanced modules are, and are not

- **Optional and off.** They are not part of an experiment's configuration, no existing result depends on them, and nothing runs on a schedule. An alert from drift or from the monitor is something for a person to read: no rule here retrains, rolls back, halts or promotes anything.
- **Drift.** The PSI levels are conventional heuristics that can be configured; they are not significance levels. The KS p-value uses an effective sample size from the lag-1 autocorrelation of each asset's series and adjusts across features; the dependence between assets on a date is not modelled, so it stays optimistic. The reference window is fixed at the model's fit cutoff, so a model refitted later gets a new reference. Performance drift reads settled development outcomes of the validation segment; it is a development statistic, not live monitoring.
- **Retraining.** A trigger is a proposal. Duplicates are linked to the open proposal, a cooldown and a data floor suppress, and a drift or performance trigger must be backed by an alert in a recorded drift report. A retraining runs only from code with a trainer the caller supplies (the platform supplies none), needs a named person's approval unless the policy allows otherwise, must stay in development mode, is validated against the incumbent, and ends as a candidate, never a promotion; repeated failures suspend it until a person documents a recovery.
- **Bayesian averaging.** Posterior probabilities follow from a stated likelihood and a stated prior (Zellner's g-prior, g = n), by the exact Bayes factor of each member against the intercept-only model. The predictive variance split uses a plug-in within-model variance. A weight derived from performance is a weighting scheme and is labelled as one.
- **Robust allocation.** The uncertainty set is a box on expected returns and a scale on the covariance, with radii calibrated from the history. A radius calibrated on the past describes the estimation error the past contained; it makes no statement about a regime change, and a larger radius buys protection against a wrong forecast at the price of a smaller book.
- **Horizon combination.** Forecasts are put in per-session units before they are combined; weights read only settled decisions; the sleeves share capital so the gross is never the sum of their grosses. The comparison with the single-horizon books is a development comparison.
- **Transfer.** One fit at the cutoff, no refits. The source and target share no security, the target is evaluated from the cutoff on, and the scaler and any components are fitted on the source's training rows only.
- **Discovery.** The best expression of a finite search is selected: the controls (Benjamini-Hochberg, Holm, a best-of-N correction) limit that, and do not remove it. A confirmation on later data is one look. Nothing mined here is independent evidence.
- **Off-policy evaluation.** An estimate exists only with logged behaviour probabilities and support; the platform's reinforcement-learning agents do not log them, so the estimator works on a log the caller supplies. The identification assumptions (positivity, ignorability, stationarity, no interference) are listed with every result and cannot be tested from the data.
- **Tax lots.** The module carries no jurisdiction's rules. A policy is supplied with its provenance and version; an absent, incomplete or unimplementable one gives no figure. Wash sales, short sales, dividends, withholding, currency conversion, brackets, carry-back and other tax years are not modelled. Tax is paid out of the portfolio and not fed back into position sizes (conservative on both counts), tax not yet due is deducted as a liability, and the tax on selling every open lot is reported separately as a deferred liability. The ledger and the return series are not reconciled. This is arithmetic under a supplied rule set, not tax advice.
- **Champion and challenger.** The schedule of looks, the spending rule and the shared assumptions are frozen before any observation; the levels are a union bound over the looks (conservative). When both ids are strategy versions registered with governance, their execution, cost/schedule and risk-limit settings must be identical and a difference is refused; otherwise the shared assumptions are only declared (the record says `checked: false`). The forecasts are supplied by the caller. A replay never counts; a look promotes nothing.
- **Uncertainty monitor.** Four checks, each with its own state; a missing estimate is `unavailable` or `not_applicable` with its reason and is never read as zero uncertainty or as health, and a monitor that could compute nothing is `unavailable`. A stored experiment gives the model's predictive distribution; an ensemble also keeps its members' calibrated forecasts (one more artifact column per member), so disagreement runs on it. A single model has no members (`not_applicable`), and an ensemble saved before they were kept is `unavailable`.
- **Reports.** A report is rendered from a snapshot stored with it and does not change when models or data are updated; whether its inputs have changed since is a separate question, answered by `verify_inputs`. A simulated data source is labelled in the report.
- **Not done in this release:** starting a retraining from the product (P3-02) remains `in_progress`: what a retraining means here is a product decision, because refitting the same configuration on more data reproduces the same walk-forward predictions. No advanced technique has been shown to improve a strategy on real data.

## Fixed in passing

- A drift or monitor payload carried its own `as_of`, which the response envelope reserves for provenance; it now moves into the envelope's field.
- An ensemble accepted an agent (whose output is a set of portfolio weights) as a member, and a forward step built positions from a saved rule without checking it against the model's output kind; both now refuse, with the contract's own message.
- An ensemble kept only the weighted result of its members; it now also returns each member's calibrated forecast, so that their disagreement can be monitored on a stored experiment.
- A panel-stacked series hid its serial dependence from the effective sample size; the effective size is computed per asset and averaged.
- The discovery module's look-ahead check compared two evaluations bit for bit. On pandas 2 a frame and its copy are laid out differently in memory, so a row-wise mean adds its terms in another order and differs in the last bit; a valid expression was reported as looking ahead (the Python 3.10 job found it). The check now requires agreement to within 1e-9, which rounding stays far below and any dependence on the scrambled future (noise of scale 10) far above, and a test shows that a rounding difference passes while a value that reads the next row, and a cell that goes missing because of the future, do not.
- What the reinforcement-learning environment overrode in an agent's book (a clipped action, a forced liquidation, a trimmed limit, a shortened trade, a cut-back of exposure) was returned from the call that made it and then dropped. The events of the days an agent is replayed over out of sample are now saved with the experiment (`extra_events.parquet`, one row for each day that had any, with a stable kind for each event) and summarised in an append-only `rl_events` record (counts by kind and segment, the dates, a hash of the table). Not recorded: the training episodes' events, and a forward step with an agent (prospective stepping with persisted state does not exist yet, P0-02).
- A CatBoost fit wrote its training log into a `catboost_info` folder in whatever directory the application was started in; it is now told to write nothing.
- A classifier's probabilities (`predict_proba`) were taken as column 1 of the scikit-learn output without checking the classes, and were not checked for bounds; they are now refused unless the classes are [0, 1] and every value is finite and inside [0, 1].
