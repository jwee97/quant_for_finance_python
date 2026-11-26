# Forecast → Alpha → Portfolio → Backtest → Attribution → Research Session → Comparison

This closes the four capabilities the previous round left open (attribution of an optimised portfolio's return, saving trading studies, forecasts as expected-return inputs, browser verification of the comparison with the new artifact kinds) and connects them. Nothing was rebuilt: attribution reuses `src/backtest/attribution.py` (Brinson-Fachler, Carino) and `src/riskengine/factor_model.py`; the optimisers are `src/portfolio/lab.py`'s; forecasts are the existing methods of `src/forecast/`; sessions, artifacts and the comparison are `src/research/`'s.

## 1. Attribution (`src/portfolio/attribution.py`)

**Factor attribution, ex post.** On each day the gross return is `sum_k (b_k' w_t) f_kt + sum_i w_it e_it`: the exposures of the weights held that day (drifted from the last rebalance) times the realised factor returns, plus each asset's residual. Net of costs, `net = gross - cost`; financing is reported as zero because the Lab models none (long only, fully invested). The residual of the reconciliation is `authoritative net - sum of the lines` and is its own line.

* *Point in time.* The exposures of a rebalance come from the window that ends the day **before** it (the same window the optimiser saw). Test: change every return from the second rebalance on, and the first two rebalances' exposures and risk shares do not move.
* *Ex ante against ex post.* Exposures `b'w` and the share of forecast variance from the factors are reported per rebalance (what the portfolio was set up to do); the contributions are what happened.
* *Models.* `market` (equal-weighted universe, betas from the past window) and `statistical` (principal components of the past window; the day's factor return is the projection of its returns on the fixed loadings).
* *Linking.* Each day's contribution is scaled by the portfolio value at the start of the day (wealth-weighted), so the cumulative contributions add to `V_T - 1` at **every** date.

**Brinson-Fachler.** The existing implementation, run daily on the earning weights of a method and of the benchmark (equal weight by default), by group: allocation `(Wp-Wb)(Rb_g-Rb)`, selection `Wb(Rp_g-Rb_g)`, interaction `(Wp-Wb)(Rp_g-Rb_g)`; they sum to the day's gross active return. Costs enter as their own line (portfolio minus benchmark), and everything is **Carino-linked on the net returns**, so allocation + selection + interaction + costs + reconciliation residual = the cumulative net active return exactly. Explicit rules: cash is a group of its own; an asset with no group is `Unclassified` with a warning; an asset the benchmark does not hold has zero benchmark weight (warned); a benchmark whose weights do not sum to one is refused; a classification may change over time (a list of effective-dated mappings, applied point in time) and a day with none in force is refused. Brinson is offered only when a classification exists and the benchmark method was run; otherwise the page says which is missing.

**Currency** (`currency_decomposition`): `(1+local)(1+fx)-1 = local + fx + local·fx`, weighted; the three add up to the base-currency return exactly. The Lab's data are single-currency, so the page does not offer it.

**Reconciliation results** (tests `tests/test_portfolio_attribution.py`): the replay of the target weights equals the Lab's own net return to 1e-14; factor contributions + specific − costs equal the net return with a largest daily residual of order 1e-16 and a cumulative gap of order 1e-15 (tolerance 1e-9, documented); Brinson daily identity error 2e-17, cumulative gap 1e-15. A deliberate 1e-4 error in one day of the authoritative series appears as exactly 1e-4 in the residual line and the status flips to `does_not_reconcile`; the other lines are unchanged (the gap is never spread over the factors).

## 2. Forecast-to-alpha (`src/forecast/alpha.py`, `src/alphalab/expected.py`)

`ForecastRecord`: instrument, origin, `available_at`, horizon, kind (price, return, volatility, distribution), units, value or draws, uncertainty, price basis, dividends, split factor, currency, model id and version, training-data fingerprint, evaluation status, provenance.

| Forecast | Becomes |
|---|---|
| volatility | **refused**: a spread is not a drift |
| price, adjusted basis | `pred / last - 1` (a dividend or a split factor is refused: already inside the prices) |
| price, raw basis | `(pred × split + dividends) / last - 1` |
| log return with spread | `exp(mu + σ²/2) - 1` (the mean); without spread `exp(mu) - 1`, labelled the median |
| distribution (draws) | `mean(exp(draw) - 1)`, with the standard deviation of the simple return |
| other currency | `(1 + local)(1 + fx forecast) - 1`; refused without the exchange-rate forecast |

Horizons: a forecast is used only at its own horizon (the rebalance step); another horizon is refused unless scaling is allowed (expected log return scaled linearly, spread by the square root, ratio capped at 3, assumption stated). `annualise` compounds. The only multiplication by 252/h is `to_optimiser_units`, which puts a horizon-`h` return in the units of the annualised covariance the optimisers use; it is applied to every source and is not reported as a forecast.

Sources of expected returns for the optimisers: `historical` (the Lab's default), `factor` (information coefficient measured on dates whose forward return was already complete × horizon volatility × z-score), `forecast` (any of the cheap mean-bearing methods, each from prices up to the origin, shrunk by the slope of realised on forecast over forecasts already resolved, zero weight until 20 have), `combined` (equal weights, or inverse past squared error over completed rebalances). All run through `portfolio.lab.compare(mu_fn=...)` with the **same** risk estimate, caps, costs and schedule; the historical mean fills any asset a source has no usable estimate for, and the page counts how often. Methods that ignore expected returns (minimum variance, risk parity, hierarchical, equal weight) give identical results under every source and are the control. Sensitivity of the weights to errors in the expected returns (noise of ¼, ½, 1, 2 cross-sectional standard deviations) is reported.

**Point-in-time proofs** (`tests/test_forecast_alpha.py`): a forecast from an origin is unchanged when every later price is multiplied by 7; reliability is unchanged by prices after its as-of day; the forecast provider ignores forecasts made on or after the last known day and stale ones; the factor provider's estimate and information coefficient do not move when prices from the decision day on are changed; combination weights depend on completed outcomes only; the first-day cost differs by exactly the cost of buying the book whatever the source.

## 3. Research sessions (`src/research/store.py`, `extract.py`, `src/webapp/research_api.py`)

Artifact schema 2 adds: `hypothesis`, `benchmark`, `costs`, `validation`, `attribution`, `source` (the job and the request that produced it), `version`, `parent`, and `extra` for any field this version does not know. A schema-1 artifact is read as schema 2 **in memory**: its file is never rewritten (a test compares the bytes), unknown fields move to `extra`, `migrated_from` says what it was; a newer schema is refused. Kinds saved: backtest, portfolio study (with attribution, validation, benchmark, costs, expected-return source), factor study, trading study (new), forecast evaluation, machine-learning experiment, regime estimate, strategy labs.

* *Immutable.* Nothing edits an artifact. Saving the same run twice keeps one study (`duplicate`); a new version is a separate artifact whose `parent` is the study it replaces, in the same session, with `version + 1`.
* *Re-run.* `GET /api/research/artifact/rerun?id=` returns the stored request; the page submits it and saves the result as a new version.
* *Save dialog* (`Q.research.saveStudy`): choose or create a session, name, hypothesis, notes; the button is off while saving; a failure is shown in the dialog, which stays open.

## 4. Research Comparison (`src/research/compare.py`, `static/compare.js`)

Results with a return series are recomputed over the days they all share (as before). New rules:

| Rule | Severity |
|---|---|
| different trading costs (basis points) among results with a return stream | stops a ranking |
| different construction (lookback, step, covariance estimator, cap, risk aversion, shrinkage) | stops a ranking |
| different currency | stops a ranking |
| different source of expected returns, everything else equal | note (it is the thing compared) |
| different optimiser **and** source | warning |
| different benchmark, risk-free rate | warning |
| studies without a return series: different kind, horizon, dates, instruments, data version | stops a ranking of the studies |
| attribution reports from different factor models | warning (lines not compared) |
| assumed point-in-time dates in fundamental factors | warning |

Predictive quality (factor-study information coefficient, forecast skill) and trading performance are in separate blocks and never ranked together; a study with no series says why nothing was recomputed. The page shows the assumptions side by side with the differing rows marked, the predictive block, the attribution differences, version and hypothesis, and an Open / Run again control per study (the opened study shows its saved attribution, notes, validation and versions).

## 5. Verification

See "Test results" at the end.

## 6. Status

| Capability | Status |
|---|---|
| Factor attribution of optimised portfolio returns (market and statistical models, daily and cumulative, ex-ante exposures, costs, residual) | Implemented and tested |
| Brinson-Fachler attribution against a benchmark (cash, missing groups, changing classification, linking, reconciliation) | Implemented and tested (library, API, browser) |
| Currency decomposition | Implemented and tested at library level; **not offered in the page** (no FX data in the Lab) |
| Financing costs | **Not modelled** by the Lab; reported as zero and labelled |
| Attribution against a benchmark other than a Lab method (an external index) | Not implemented |
| Forecast contract, conversions, availability rules | Implemented and tested |
| Forecast, factor and combined expected returns into the optimisers; same risk/costs; sensitivity | Implemented and tested |
| Price and raw-basis forecasts as inputs through the page | Partially implemented: the conversions are implemented and tested; the page generates distribution forecasts with the platform's own methods and cannot import an external price forecast |
| Forecast methods usable as sources | The cheap mean-bearing ones (historical mean, GBM, Hull-White, Kalman, Theta, ARMA, Markov returns, GP drift); neural and heavy methods are not offered |
| Save trading, portfolio, factor, forecast studies; hypotheses, notes; reopen; versions; re-run; duplicates | Implemented and tested |
| Legacy artifact loading | Implemented and tested |
| Saving from Backtesting / Strategy Explorer | Pre-existing, re-run through the same endpoint; not re-verified in a browser this round |
| Comparison rules and recomputation on common windows | Implemented and tested |
| Browser: end-to-end scenario, refresh, phone width, missing data, failing save, double submit, legacy study, re-run, tooltips, resizing | Implemented and tested (4 tests) |
| Stale forecast / invalid horizon through the page | API-level tests; the page derives the horizon from the rebalance step, so an invalid horizon cannot be typed |
