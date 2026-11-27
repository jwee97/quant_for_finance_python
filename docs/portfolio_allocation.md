# ML portfolio allocation: forecast-driven, risk-aware weights

## What was wrong

A model trained on a small universe (say NVDA and MU) produced a book that looked like 50/50 whatever the forecasts, volatilities or correlation. Tracing the weighting code (`src/ml/signals.py`) found three causes, none of them a bug in the accounting:

| Cause | Where | Effect on two stocks |
|---|---|---|
| Rank and proportional rules use only the *order* or the *sign* of scores | `top_n_long_short`, `long_only_top_quantile`, `proportional`, `rank_normalised`, `benchmark_tilt` | the cross-sectional z-score of two numbers is always ±0.707, so every rule that reads it gives the same two weights |
| The only magnitude-aware rule (`cost_aware_mean_variance`) needs at least four assets | `signals.py` | for two assets it silently returned a zero book |
| Equal weight is the *intended* benchmark | `backtest.equal_weight_benchmark`, `dataset.benchmark` | this is correct and stays |

Nothing in those rules reads volatility, correlation or how reliable the forecast has been.

## What was added

A new signal method, **`portfolio_allocation`** (`src/ml/portfolio_alloc.py`). It is used only when selected: **the default signal method, every existing method and every existing config hash are unchanged** (`tests/test_portfolio_allocation.py` pins the hashes and the numerical results of two existing rules recorded before the change). The method feeds the same authoritative backtest as every other rule, so execution lag, drift, costs and the ledger are unchanged.

### Allocators (one interface, `allocate`)

| Key | Objective | Needs | Fails when |
|---|---|---|---|
| `equal_weight` | 1/N within the limits | nothing | ignores risk and forecasts |
| `inverse_vol` | w ∝ 1/σ | a volatility | ignores correlation |
| `erc` | equal risk contributions (or a given risk budget) | positive-definite covariance | long-only only; unstable near singular |
| `mean_variance` (Markowitz 1952) | max μ'w − (λ/2) w'Σw − cost | calibrated μ, covariance | estimation error in μ is amplified |
| `black_litterman` (1992) | MV on the posterior of an equilibrium prior and the forecasts as uncertain views; view variance is the model's measured out-of-sample error variance | out-of-sample errors | no measured uncertainty → the prior (equal weight) |
| `hrp` (López de Prado 2016) | recursive bisection of the correlation tree | **at least 3 assets** | refused for 1–2 assets |
| `cvar` (Rockafellar–Uryasev 2000) | max μ'w − κ·CVaR over the trailing window's de-meaned scenarios | ≥ 60 scenarios | short windows give noisy tails |
| `robust_mv` | MV with each forecast reduced by κ × its parameter uncertainty | a calibration standard error | conservative |
| `kelly` | MV with risk aversion 1/f (fractional Kelly) | calibration t ≥ `kelly_min_t` | **not justified** (and reported so) when the forecast is not calibrated |

Documentation for each (objective, assumptions, needs, failure modes) is in the catalogue (`GET /api/ml/allocation/catalog`) and in each run's report.

### Forecast → expected return

* A forecast of a **return** is used as is; a **price** becomes `F/P − 1` at the decision date; a **distribution** contributes its mean and spread.
* A directional **probability is never read as an expected return**: it goes through a payoff model estimated from past outcomes.
* A forecast of another **horizon or currency** than the portfolio's is refused (`check_forecast_contract`), never rescaled.
* **Calibration uses past outcomes only**: through the origin on forecasts whose outcome ended at or before the decision (`s ≤ i − lag − h`); the t-statistic is deflated for the overlap of horizon-day windows. A forecast without a demonstrable slope is treated as *no edge*, and the book is set by risk and the limits alone (recorded per decision).
* **Forecast uncertainty** is the out-of-sample residual standard deviation from the same history.
* A rank or relative-return model output is refused by the output contract (`contracts._allocation_forecast`) with what to change.

### Risk models

`sample`, `ewma`, `lw_constant_correlation` (Ledoit–Wolf, default), `lw_identity`, `pca_denoised`, `factor` — the existing `risk_models` estimators, fitted on observations before each decision only, with missing data, ill-conditioning (repaired and counted) and short windows (no estimate yet → a flat book) handled and recorded.

### Constraints

Long-only or long-short, fully invested or cash allowed, min/max weight, sector cap, gross/net bounds, one-way turnover limit, volatility target, leverage cap, risk budget, and a transaction-cost + turnover penalty in the optimiser. **Feasibility is checked before optimisation** (`feasibility`): two stocks capped at 30% each cannot sum to 100%, and that is reported (`infeasible`, per decision, with the reason) and handled by the configured **failure policy** (`risk_only`, `hold`, `equal_weight`, `cash`) — a limit is never relaxed silently. After every solve the book is re-checked against all limits.

### Walk-forward

At every rebalance: forecasts available then → risk estimate from the past → calibration from resolved outcomes → solve → limits → trades from the previous book → the existing engine simulates execution, costs and realised P&L. Per-decision records (forecasts, calibrated forecasts, uncertainty, volatilities, weights, risk contributions, explanation, status, failure policy) are kept in `report.portfolio_allocation`. No future return enters a weight: `test_weights_at_a_date_do_not_depend_on_anything_after_it` rewrites the future and truncates the panel for six allocators.

### Why each stock got its weight

Exactly: `weight = equal-weight base + risk effect + forecast effect`, where the risk effect is the same allocator with a zero forecast minus the base. The page shows the three, the driver (forecast / risk and correlation / constraints / equal), the binding limits and the calibration state of the forecast.

### Comparing allocators (Phase 8)

`src/ml/allocator_compare.py`, `POST /api/ml/allocation/compare`, and the "Compare all allocators" run in the UI:

* identical forecasts, dates, limits, costs and execution for every allocator; the experiment's own position rule appears as one more row, and equal weight as the benchmark;
* net return, Sharpe, drawdown, turnover, cost, concentration (effective N, max weight) and weight stability;
* **selection on the validation segment only**; Holm-adjusted paired tests against equal weight (autocorrelation-robust) and the deflated Sharpe of the selected allocator for the number tried;
* test and forward are **not computed** unless requested, and then reported with `used_for_selection: false`; selecting on a protected segment is refused;
* **forecast quality is reported separately** from portfolio quality; a cross-sectional rank IC is estimated only with at least five assets on a date (it is `null` with a stated reason for two stocks).

## Using it

* UI: **Machine learning → Portfolio allocation**. Choose a saved experiment, an allocator, risk model, limits and costs, and *Allocate* (a child experiment; the parent is untouched) or *Compare all allocators*.
* API: `POST /api/ml/rebacktest` with `signal: {method: "portfolio_allocation", holding_period: 1, allocation: {...}}`; `POST /api/ml/allocation/compare`; `GET /api/ml/allocation/catalog`.
* The model must forecast an asset's own return (`fwd_ret_h`) or, with `forecast_kind: "probability"`, the direction of its move.

## Round two: holding period, calibration, stability, adapters, currency

### Why the holding period was restricted to one day, and what changed

Exact cause. `signal.holding_period` is, for every other position rule, applied *after* the book is built: `signals.holding_period` averages the last `H` daily books (staggered tranches). The allocation module is not intrinsically
one-day (it reads `window_days` and `opt_every`), but averaging its books after the fact would have meant that the weights the backtest held were an average of optima of different dates, which is a book that no optimisation
chose, with a cost term that no longer matched the trades. The guard `config.validate` raised `the portfolio allocation weighs forecast, risk and cost itself ... set signal.holding_period to 1` was the only thing enforcing it, and
the UI sent `holding_period: 1`. Removing the check alone would have silently averaged.

What was done instead: the holding period became a first-class input of the allocation (`alloc_horizon.py`), with four quantities kept apart:

| Quantity | Setting | Meaning |
|---|---|---|
| Prediction horizon `h` | `dataset.horizon` (`window_days`) | what the model's number forecasts; outcomes resolve `lag + h` sessions after the decision |
| Holding window `L` | `signal.holding_period` / `allocation.holding_days` | the window the optimisation is posed over and, for tranches, how long each is held; `h` when not set |
| Decision frequency | `allocation.opt_every` (and `backtest.rebalance` for when the engine may trade) | how often a new book is chosen |
| Execution lag | `dataset.lag + backtest.extra_lag` | how long after a decision the first return is earned |

Expected return over `L` is `mu_h * min(L, h) / h`: the edge accrues evenly within the forecast horizon and is **not extrapolated** beyond it. Covariance over `L` is one of `linear` (daily x `L`, the original rule),
`variance_ratio` (each asset's variance scaled by its Lo-MacKinlay variance ratio from past daily returns) or `multiperiod` (overlapping `L`-day returns, blended with the linear estimate by the number of *independent* windows
the past holds; with too few it is the linear estimate and says so).

Overlapping forecasts. `lifecycle: single_book` re-optimises one book at each decision; a new forecast replaces the previous, so overlapping forecasts are never stacked into a larger position (tested). `lifecycle:
staggered_tranches` opens a tranche every `opt_every` sessions and holds each for `L`; `ceil(L / opt_every)` overlap, each is a full book chosen by its own optimisation with the expiring tranche as its starting book (so the
cost term matches the trades), and the held book is their mean (feasible because the limits are convex). `holding_period = 1` with no `holding_days` is exactly the original behaviour; recorded runs reproduce to the digit.

Tested end to end through train, forecast, allocate, lagged execution and costs for 1-, 5-, 10- and 20-day horizons with their own holding windows, tranche counts and decision schedules.

### Calibration (`alloc_calibration.py`)

The pooled slope stays as the baseline. Optional: `asset` (own slope where an asset has `calib_asset_min_eff` independent windows), `partial_pooling` (empirical-Bayes shrinkage toward the pooled slope; DerSimonian-Laird
between-asset variance, **needs three assets**), `sector_pooling` (needs a sector map and three sectors) and `regularized` (the pooled slope shrunk toward zero by a normal prior). Standard errors: the original
`sqrt(horizon)` inflation, or `hac` (Newey-West on the per-date score, which carries the cross-asset correlation, with at least `horizon - 1` lags) or `block_bootstrap`. An effective sample size is reported. With two stocks the
asset-level methods fall back to the pooled slope and say why. `compare_calibration` scores each method out of sample on the validation segment only and calls a method better than the baseline **only** when the interval of the
paired HAC difference in squared error excludes zero; on the synthetic data used in the tests the methods are indistinguishable, and the output says so.

### Stabilisers (`alloc_stability.py`)

`w_final = (1 - alpha) w_baseline + alpha w_ML` with an equal-weight, inverse-volatility or ERC baseline; alpha is fixed, from forecast confidence, or from the best of a fixed grid by mean-variance utility over the last `blend_window`
**resolved** past decisions (baseline until `blend_min_obs`). Also: uncertainty shrinkage (slope minus `kappa` standard errors), minimum-evidence controls (`min_n_eff`, `min_confidence`), an explicit per-decision weight-change limit
(exact projection for a fully invested book; reported as infeasible if it cannot be honoured), concentration diagnostics, and forecast, risk-aversion and cost sensitivities. A blend that breaks a limit is projected onto the feasible
set and the deviation is reported. The default risk aversion stays 5. `preset: conservative` is a documented set of explicit values (risk aversion 20, HAC calibration with minimum t 2, one-standard-error shrinkage, a
confidence-weighted blend with inverse volatility capped at 0.5, 10-point weight-change limit, 60% position cap, small turnover penalty); explicit settings override it and it is not claimed to be optimal.

### Optimiser adapters (`alloc_adapters.py`)

| Key | Source | Reads the forecast? | Min assets | Native limits | Others |
|---|---|---|---|---|---|
| `herc` | `portfolio.hierarchical.herc_weights` | **no, risk-only** | 3 | none | projected, move reported |
| `nco` | `portfolio.nco` (minimum variance) | **no, risk-only** | 4 | none | projected, move reported |
| `nco_max_sharpe` | `portfolio.nco` (maximum Sharpe) | yes (calibrated mu, long-only per stage) | 4 | none | projected, move reported |
| `near_optimal` | `portfolio.near_optimal` | yes | 2 | box, sum | projected |
| `wasserstein_cvar` | `portfolio.robust_wasserstein` | **no, risk-only** | 2 (60 scenarios) | box, sum | projected |

Every adapter declares required inputs, the minimum asset count, whether it uses expected returns, that it is long-only (a long-short request is refused), what it honours natively, its window, its failure behaviour and its numerical
limitations (`GET /api/ml/allocation/catalog`). A limit an optimiser cannot honour (a position cap, a sector cap) is enforced by the Euclidean projection and the size of the move is recorded as `projection_l1`; the books are
re-checked against every limit after allocation (tests). HERC is implemented as the risk-only allocator it is and the page labels it so.

### Currency (`alloc_fx.py`)

Quotation is fixed: **base currency units per one unit of the instrument's currency** (a series quoted the other way is inverted on entry). For an unhedged holding `1 + R_base = (1 + R_local)(1 + R_fx)`; with a hedge ratio `k`,
`(1 + R_local)(1 + (1 - k) R_fx)` (carry ignored). Every rate has the time it refers to and the time it became available; a rate is used only if it was available before the date ended and is not older than `max_stale_days`.
A missing rate is never 1 and never "no change": the conversion raises, or with `missing="exclude"` removes the asset-day and reports how many. Observed returns use the rates of their own day. A forecast made at the decision date
needs an assumption about the exchange rate over the horizon: `random_walk` (expected FX return 0, labelled an assumption) or a caller-supplied expected FX return read only on that date; no exchange rate after the decision date is
read (tested by rewriting the future). Costs of converting are `fx_cost_bps` of the foreign exposure traded; `currency_attribution` splits the return exactly into local and currency parts (including the cross term) by currency;
`fx_exposure` reports exposure by currency. The original currency refusals stand when no FX context is supplied.

## Limitations and what was not implemented

* The experiment pipeline has no exchange-rate data source: the currency layer is complete at the computational level (`run(fx=...)`, `allocator_compare.compare(fx=...)`, tests) but is not reachable from the web form, which still refuses a
  forecast in a currency other than the portfolio's. A data ingestion for rates is the missing piece, not the arithmetic.
* FX costs are computed beside the backtest (one cost rate for every asset is all the engine has) and are not yet subtracted inside it.
* Partial hedging ignores the carry (interest differential) of the hedge.
* Asset-specific and hierarchical calibration cannot be evaluated on two stocks: they fall back to the pooled slope by design. Whether they beat it on larger universes is an empirical question the comparison answers per dataset;
  nothing here claims they do.
* HRP, HERC and NCO need three or four assets; the Wasserstein ball radius is an assumption (10% of the mean scenario standard deviation); NCO and HERC choose their cluster count by silhouette, which can change the book discontinuously.
* Sensitivities are computed for the latest decision only. Alpha from past resolved decisions needs a history of them (`blend_min_obs`); before that the baseline is used.
* Fractional Kelly with a full covariance and parameter uncertainty beyond the robust and fractional forms is not implemented. Equal weight beating the optimised books on a short sample is a common and expected result (DeMiguel, Garlappi and
  Uppal 2009): read the intervals, not the ranking.

## References

Markowitz (1952); Black and Litterman (1992); Rockafellar and Uryasev (2000); Ledoit and Wolf (2004); DeMiguel, Garlappi and Uppal (2009); Lo and MacKinlay (1988) for the variance ratio; Newey and West (1987); DerSimonian and Laird (1986);
Raffinot (2018) for HERC; Lopez de Prado (2016, 2019); Gu, Kelly and Xiu (2020).
