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

## Limitations and what was not implemented

* Holding period must be 1 (the allocator sets the book at each decision; averaging books over a holding window is refused).
* Forecast calibration is a slope through the origin per universe (pooled over assets); asset-specific slopes need far more history than a two-stock universe offers and are not estimated.
* HRP needs three assets; CVaR needs 60 scenarios; Kelly needs a significant calibration. When a method is not justified it reports so and the failure policy applies.
* Sector caps need a sector map in the dataset; without one the cap is refused rather than ignored.
* Currency conversion is not performed: a forecast in another currency is refused.
* Not implemented here: HERC and other Alpha Lab optimisers are not wired into this method (they remain in the Alpha Lab and `src/portfolio`). Equal weight beating the optimised books on a short sample is a common and expected result (DeMiguel, Garlappi and Uppal 2009): read the intervals, not the ranking.

## References

Markowitz (1952); Black and Litterman (1992); Rockafellar and Uryasev (2000); Ledoit and Wolf (2004); DeMiguel, Garlappi and Uppal (2009); López de Prado (2016); Gu, Kelly and Xiu (2020).
