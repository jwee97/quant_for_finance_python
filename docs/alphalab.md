# The Alpha, Risk and Portfolio Lab

This page covers the **Alpha & Portfolio Lab** tab, the `quant lab` command and the packages behind them: `src/alphalab`, `src/riskengine` and the new parts of `src/portfolio`. They follow four books: Grinold and Kahn, *Active Portfolio Management* and *Advances in Active Portfolio Management*; Cajas, *Advanced Portfolio Optimization*; and Paleologo, *Advanced Portfolio Management*. López de Prado's *Advances in Financial Machine Learning* is the [Financial ML Lab](finml.md).

Before anything was written each idea in the roadmap was checked against the code. A great deal already existed (Black-Litterman, HRP and HERC, CVaR optimisation, the multi-period optimiser and its no-trade region, Ledoit-Wolf and random-matrix covariance, copula scenarios, Brinson attribution). [The audit](roadmap_audit.md) lists every item with its status and where it lives, and a test imports every location it names. What follows is what was added to join them up.

## The pipeline

```
signal -> IC, decay, calibration -> expected residual return (alpha)
                                         |
          factor model -> risk of a portfolio -> optimiser -> trading policy -> attribution
```

Each stage speaks the same language: a panel of dates by assets, a covariance in per-period units, holdings as fractions of capital.

## Alpha (`src/alphalab`)

| Question | Answer | Where |
|---|---|---|
| How good is the signal, and how steadily? | the rank IC on each date; its mean with a Newey-West t-statistic (overlapping forward windows make neighbouring ICs dependent); the share of blocks of history in which its sign held | `ic.ic_summary`, `ic.stability`, `ic.rolling_ic`, `ic.ic_by_regime` |
| How fast does it go stale? | the IC of the signal against the return of the single day *k* days later, with an exponential fit: the half-life is `tau ln 2`. A signal whose first-lag IC is noise gets no half-life: a decay rate of noise is noise | `ic.lagged_ic`, `ic.fit_decay`, `ic.delay_cost` |
| What is it worth as a forecast? | `alpha = IC x omega x z`: the standardised signal times the residual volatility over the horizon times an IC estimated only from ICs whose forward windows had ended, shrunk by the signal-to-noise of its own average. `reliability` regresses the realised residual return on the alpha: a slope of one is calibrated | `calibrate.calibrate`, `calibrate.reliability` |
| How many independent bets does it make? | the participation ratio of the residual-return correlation matrix, times non-overlapping horizons a year, times the share of the signal that is new at each. An estimate, and labelled as one | `breadth.effective_breadth` |
| What does a constrained book keep of it? | the transfer coefficient: the cosine of `L^-1 alpha` and `L' h` for `Sigma = L L'`. It is exactly the ratio of the information ratio the holdings earn to the best any holdings could | `breadth.transfer_coefficient`, `breadth.constrained_transfer` |
| Is the signal point-in-time? | a versioned registry in which each signal carries the lag between its date and the moment it was known, and a check that counts values known after their date | `registry.SignalRegistry`, `registry.pit_violations` |

Combining several signals is the existing IC-weighted combination (`src.equity.alpha_model.optimal_alpha`): weights proportional to the inverse covariance of the ICs times their means, walk-forward.

## Risk (`src/riskengine`)

Three fits of `Sigma = B F B' + D`:

* **time series**: regress each asset on given factors;
* **cross-sectional**: regress the cross-section on each date on the *previous* date's exposures (sector, value, size, ...). The sector returns are constrained to net to zero so they and the market are not collinear. The slopes are the factor returns, which are the returns of factor-mimicking portfolios;
* **statistical**: principal components, the number set by the Marchenko-Pastur bound.

Then: the risk of a portfolio by factor, group and asset, with contributions that add exactly to the volatility; the risk of the active holdings against a benchmark; weights that follow a budget over factor groups; holdings neutral to chosen factors, changed as little as possible in risk terms; a minimum-variance hedge with named instruments; and **the bias statistic**, the standard deviation of the portfolio's return divided by the volatility forecast made the day before. It is 1 when the forecast is right, and its two-standard-error band is `1 +- 2/sqrt(2 w)`.

Two covariance estimators were added and reach `estimate_covariance`: the **graphical lasso** (a sparse inverse: a zero is a conditional independence) and a **PCA factor covariance**. **Detoned** correlation (the market component removed) is for clustering, not for inverting.

## Portfolios (`src/portfolio`)

New: **nested clustered optimisation**, **CDaR**, **maximum drawdown** and **EVaR** optimisation, **near-optimal centering**, **Wasserstein-robust CVaR** (the worst case over a ball of distributions around the scenarios: the CVaR plus `eps ||w|| / beta`, so a larger ball asks for a less concentrated portfolio), **neighbourhood caps on a graph of the assets** (the weight inside each asset's neighbourhood of the minimum spanning tree or TMFG is at most a cap; a hub that makes that impossible is refused), **Black-Litterman with views on factors** (the augmented-universe construction), and the comparison harness. Three are registered as allocators, so the Backtest tab can run them with its own engine, costs and tearsheet: `nco`, `near_optimal` and `cdar`.

The harness (`portfolio.lab.compare`) runs 19 methods like for like:

* the same trailing window on every rebalance, the same covariance estimator, the same shrunk expected returns, the same holding period with weights drifting between rebalances, the same trading cost, and the same cap on a position;
* a method that fails on a date falls back to equal weight there, and **the number of fallbacks is in the table**;
* the cap is applied after the method chooses, by clipping and redistributing; **the weight it moved is in the table**, so a method that needs the cap to be a sane portfolio can be seen to;
* `instability` is the mean one-way turnover between the weights on the real window and on a bootstrap resample of it;
* `sharpe_se` is there so that two Sharpe ratios are not compared without it. Over a few years, differences between methods are usually inside it.

## Trading (`alphalab.dynamic`)

The multi-period optimiser and the no-trade region are `src/equity/multiperiod.py` (Garleanu-Pedersen). `dynamic.simulate` puts four policies on the same forecasts, covariance and costs: **full** (hold the aim), **partial adjustment** (close the stationary share of the gap), **band** (hold while the gradient is inside the proportional cost) and **threshold**. `implementation_waterfall` re-runs the same forecasts with position limits, then costs, then a policy that respects them, then a day's delay, and reports the realised Sharpe at each step: where the paper alpha goes.

Also here: **selection, sizing and timing attribution** (an exact three-way split of profit against an equal-size, average-gross version of the same book), **stop-loss** and **volatility-target** simulations (both causal), **capacity** from the square-root impact law, **crowding** (position overlap, factor-exposure overlap and return correlation of several books), and **gearing**: Kelly leverage with its estimation error, and the leverage at which the long-run share of time spent a given depth below the peak equals a chosen probability.

## The tab

Four studies on a basket of tickers' daily prices: **Alpha**, **Risk**, **Portfolio**, **Trading**. Signals in the page are built from prices alone (momentum, reversal, low volatility, trend): they are examples of the interface, not recommendations. A signal from a model or from fundamentals is registered through `src.alphalab.registry` and studied with the same functions. "Keep this signal" stores the panel as a new version in the registry (`data/user/alpha_signals`); an unchanged panel keeps its version.

```
quant lab alpha SPY QQQ IWM EFA EEM TLT GLD --signal momentum --horizon 21
quant lab risk SPY QQQ IWM EFA EEM TLT GLD --kind statistical
quant lab portfolio SPY QQQ IWM EFA EEM TLT GLD --methods equal_weight hrp nco cdar --cov-method graphical_lasso --perturb 5
quant lab trading SPY QQQ IWM EFA EEM TLT GLD --signal reversal --cost-bps 10
```

## What this does not do

* **No claim that a method is better.** Each was tested against a world whose truth was planted (an AR(1) signal of known decay, a factor structure of known loadings, a covariance whose optimum is closed-form, Brownian motion with a known drawdown law) and against brute-force searches and hand calculations. That shows the code does what its definition says, not that the method pays.
* **Price-based signals only in the page.** Consensus-relative valuation and smart-beta decomposition need data the platform does not have; higher-moment optimisation is left out on purpose, because skewness and kurtosis are poorly estimated for large portfolios. See [the audit](roadmap_audit.md) for the full list.
* **The fundamental (cross-sectional) factor model is in the library, not the page**, because it needs exposures you supply.
* **A portfolio study can be kept in a research session, one method at a time** (*Keep a method in a research session*): its net equity curve, the equal-weight run as benchmark and its row of the table, so it can be compared with other results. Its Sharpe ratio is not corrected for having been chosen from the methods tried, and the saved note says so. The alpha, risk and trading studies are not saved yet.
* **Capacity is a model.** The square-root law has a coefficient of order one, and the page does not have volume data; `capacity_curve` takes yours. The backtest module's `capacity_curve` has the fuller version with volume bands.
