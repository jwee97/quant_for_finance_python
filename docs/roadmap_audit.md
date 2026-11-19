# The five-book roadmap: what is built, what already existed, what is not

This page is the audit of the roadmap for Quant Lab: López de Prado's *Advances in Financial Machine Learning*, Grinold and Kahn's *Active Portfolio Management* and *Advances in Active Portfolio Management*, Cajas's *Advanced Portfolio Optimization*, and Paleologo's *Advanced Portfolio Management*. It is written from `src/alphalab/catalog.py`, and a test imports every location named here, so it cannot describe something that is not there.

**Status.** *built*: written for this item. *existing*: the repository already had it, and the entry points to it. *partial*: some of it exists, and the note says what is missing. *not built*: nothing exists, and the note says why. **Priority** is the roadmap's proposed development order, not a statement of what was absent. Chapter numbers are as the roadmap gave them.

**Where to run things.** The *Alpha & Portfolio Lab* tab runs four studies (alpha, risk, portfolio, trading) on a basket of tickers; the Python modules are `src.alphalab`, `src.riskengine` and `src.portfolio`; see [the lab guide](alphalab.md).

## Alpha engine

| Priority | Item | Source | Status | Where | Notes |
|---|---|---|---|---|---|
| P0 | Signal registry with versions and information timestamps | Active PM | built | `src.alphalab.registry.SignalRegistry`, `src.alphalab.registry.pit_violations` | a content hash decides the version; each carries its information lag; a point-in-time check counts values known after their date |
| P0 | IC analysis: Pearson/rank, rolling, stability, by regime | Active PM ch. 10-13 | built | `src.alphalab.ic.ic_summary`, `src.alphalab.ic.rolling_ic`, `src.alphalab.ic.ic_by_regime`, `src.alphalab.ic.stability` | Newey-West t-statistic for overlapping forward windows; sign stability over blocks |
| P0 | IC decay and information horizon | Active PM; Advances ch. 5-11 | built | `src.alphalab.ic.lagged_ic`, `src.alphalab.ic.fit_decay`, `src.alphalab.ic.delay_cost` | single-period IC at each lag, exponential fit weighted by sampling error; no half-life is reported for a signal whose first-lag IC is noise |
| P0 | Alpha calibration: alpha = IC x omega x z | Active PM ch. 10 | built | `src.alphalab.calibrate.calibrate`, `src.alphalab.calibrate.reliability` | matured ICs only, shrunk by their own signal-to-noise; a reliability slope says whether the alphas are the right size |
| P0 | Signal combination weighted by IC covariance | Active PM ch. 11 | existing | `src.equity.alpha_model.optimal_alpha`, `src.equity.alpha_model.max_ir_weights`, `src.framework.ic_combination.optimal_ic_weights` | walk-forward max-IR weights on orthogonalised, standardised factors |
| P0 | Effective breadth | Active PM ch. 6 | built | `src.alphalab.breadth.effective_breadth`, `src.alphalab.breadth.effective_cross_section` | participation ratio of the residual-return correlation matrix x non-overlapping horizons x the share of the signal that is new; an estimate, labelled as one |
| P0 | Fundamental law and implied skill | Active PM ch. 6 | built | `src.alphalab.breadth.fundamental_law`, `src.alphalab.breadth.implied_ic` | IR = TC x IC x sqrt(BR) |
| P0 | Transfer coefficient | Active PM ch. 14; Clarke et al. | built | `src.alphalab.breadth.transfer_coefficient`, `src.alphalab.breadth.constrained_transfer` | the cosine of risk-adjusted alpha and risk-adjusted holdings: exactly the ratio of information ratios |

## Risk engine

| Priority | Item | Source | Status | Where | Notes |
|---|---|---|---|---|---|
| P0 | Time-series factor model | Paleologo ch. 3-5 | built | `src.riskengine.factor_model.fit_time_series` | betas to given factors; factor covariance and specific variance, optionally exponentially weighted |
| P0 | Cross-sectional (fundamental) factor model | Paleologo ch. 3-5 | built | `src.riskengine.factor_model.fit_cross_sectional` | per-date regression on lagged exposures; sector returns constrained to net to zero; needs exposures you supply (Fundamentals data in the library, not in the page) |
| P0 | Statistical factor model | Paleologo ch. 3-5 | built | `src.riskengine.factor_model.fit_statistical` | principal components; the number of factors from the Marchenko-Pastur bound |
| P0 | Risk decomposition by factor, group, asset; marginal contributions | Paleologo ch. 5 | built | `src.riskengine.decompose.risk_decomposition`, `src.riskengine.decompose.active_risk` | contributions add exactly to the volatility |
| P1 | Factor risk budgeting | Paleologo ch. 7; Cajas ch. 10 | built | `src.riskengine.decompose.factor_risk_budget_weights` | weights whose risk shares by factor group follow a budget |
| P1 | Factor-neutral and benchmark-relative holdings; hedging | Paleologo ch. 7 | built | `src.riskengine.hedge.neutralise`, `src.riskengine.hedge.min_variance_hedge` | smallest change in risk (or in weights) that zeroes chosen exposures; minimum-variance hedge with named instruments |
| P0 | Forecast versus realised risk; factor drift | Paleologo ch. 4 | built | `src.riskengine.monitor.bias_statistic`, `src.riskengine.monitor.portfolio_forecast_check`, `src.riskengine.monitor.exposure_drift` | the bias statistic with its confidence band |
| P0 | Ledoit-Wolf shrinkage and random-matrix denoising | Cajas ch. 3-4 | existing | `src.portfolio.covariance.ledoit_wolf_shrinkage`, `src.portfolio.covariance.pca_denoised_covariance`, `src.portfolio.covariance.ewma_covariance` | also a PCA factor covariance (src.ml.risk_models) |
| P0 | Detoning and graphical-lasso covariance | Cajas ch. 3-4 | built | `src.riskengine.covariance.detone_correlation`, `src.riskengine.covariance.graphical_lasso_covariance` | both reach estimate_covariance (graphical_lasso, factor); detoned correlation is for clustering, not inversion |

## Portfolio optimisation

| Priority | Item | Source | Status | Where | Notes |
|---|---|---|---|---|---|
| P0 | Cost-aware constrained optimisation | Active PM ch. 14-16; Cajas ch. 9 | existing | `tca_mvo` (allocator), `src.equity.portfolio.optimise_book`, `src.portfolio.constraints.Constraints` | costs inside the optimiser; long/short books with sector, beta and turnover limits |
| P1 | Black-Litterman | Cajas ch. 5 | existing | `src.portfolio.black_litterman.black_litterman_posterior`, `black_litterman` (allocator) | equilibrium prior and relative views; Idzorek confidence in the allocator |
| P1 | Augmented and Bayesian Black-Litterman | Cajas ch. 5 | built | `src.portfolio.bayesian.bayesian_weights`, `src.portfolio.augmented_bl.augmented_posterior` | Bayesian mean-variance (normal-inverse-Wishart) existed; views on factors as well as assets are the augmented-universe construction, which reduces to ordinary BL with asset views only. The book's own formulas for its augmented model were not available to check against |
| P1 | Drawdown and tail-risk optimisation: CVaR | Cajas ch. 7-8 | existing | `src.portfolio.cvar_optimize.mean_cvar_weights` | linear programme on scenarios |
| P1 | Drawdown and tail-risk optimisation: CDaR, maximum drawdown, EVaR | Cajas ch. 7-8 | built | `src.portfolio.drawdown.cdar_weights`, `src.portfolio.drawdown.max_drawdown_weights`, `src.portfolio.drawdown.evar_weights` | linear programmes on the scenario path (uncompounded) and a smooth convex programme for EVaR |
| P1 | HERC | Cajas ch. 12 | existing | `src.portfolio.hierarchical.herc_weights`, `src.portfolio.hierarchical.hrp_weights` | Ward clusters chosen by silhouette |
| P1 | Nested clustered optimisation | Cajas ch. 12 | built | `src.portfolio.nco.nco_weights` | minimum variance, maximum Sharpe or risk parity inside and between clusters; optional detoning for the clustering |
| P1 | Risk budgeting / risk parity | Cajas ch. 10 | existing | `src.portfolio.risk_parity.risk_parity_weights`, `risk_budget` (allocator) | Newton and SLSQP, budgets by asset class |
| P2 | Graph-based portfolios | Cajas ch. 13 | built | `src.portfolio.graph_portfolio.hsp_weights`, `src.portfolio.graph_portfolio.centrality_weights`, `src.portfolio.graph_constrained.graph_capped_weights` | MST/TMFG, centrality and hierarchical sensitivity parity existed; a cap on the weight inside every asset's neighbourhood of the graph is new, and refuses a hub that makes the cap impossible |
| P2 | Robust optimisation | Cajas ch. 11 | built | `src.portfolio.mean_variance.robust_weights`, `src.portfolio.mean_variance.resampled_weights`, `src.portfolio.robust_wasserstein.wasserstein_cvar_weights` | ellipsoidal uncertainty and Michaud resampling existed; the worst-case CVaR over a Wasserstein ball (CVaR + eps ||w|| / beta, attained by moving the worst scenarios) is new, solved exactly by cutting planes |
| P2 | Near-optimal centering | Cajas ch. 11 | built | `src.portfolio.near_optimal.near_optimal_center`, `src.portfolio.near_optimal.weight_dispersion` | the analytic centre of the portfolios within a tolerance of the best utility; resampling stability measured |
| P1 | Optimiser comparison on identical windows, costs and caps | Cajas ch. 1 | built | `src.portfolio.lab.compare` | 19 methods, six covariance estimators; fallbacks counted; cap and weight stability reported; one method can be kept in a research session |
| P2 | Higher-moment optimisation | Cajas ch. 3, 7 | not built | none | deliberately: skewness and kurtosis are poorly estimated for large portfolios; add only against a simpler benchmark that it beats out of sample |
| P2 | Copula scenarios and tail dependence | Cajas ch. 6, 14 | existing | `src.probability.copulas.simulate_joint`, `src.scenarios.generators.copula`, `src.probability.copulas.empirical_tail_dependence` | Gaussian, Student, Clayton, Gumbel, Frank; empirical or EVT marginals |

## Dynamic trading and capacity

| Priority | Item | Source | Status | Where | Notes |
|---|---|---|---|---|---|
| P0 | Multi-period optimisation, trade rate, no-trade region | Advances ch. 6-11 | existing | `src.equity.multiperiod.lq_solve`, `src.equity.multiperiod.mpc_plan`, `src.equity.multiperiod.no_trade_region`, `multi_period` (allocator) | Garleanu-Pedersen closed form; model-predictive control with constraints |
| P0 | Alpha-decay paths and trading-rule comparison | Advances ch. 5-11 | built | `src.alphalab.dynamic.decay_path`, `src.alphalab.dynamic.simulate`, `src.alphalab.dynamic.compare_policies` | full, partial adjustment, no-trade band and threshold on the same forecasts, covariance and costs |
| P0 | Implementation loss: paper to traded | Advances ch. 9-11 | built | `src.alphalab.dynamic.implementation_waterfall` | realised Sharpe as position limits, costs, a cost-aware policy and delay are added in turn |
| P1 | Capacity from market impact | Advances ch. 31 | built | `src.alphalab.capacity.capacity_curve`, `src.alphalab.capacity.capacity`, `src.backtest.impact.capacity_curve` | square-root law; capacity is the capital at which the Sharpe halves; the backtest module has a fuller version with ADV-banded execution |
| P1 | Crowding and common exposures | Advances ch. 30 | built | `src.alphalab.capacity.crowding_report` | position cosine, factor-exposure cosine and return correlation of books; effective independent books |
| P1 | Leverage and optimal gearing | Advances ch. 29 | built | `src.alphalab.capacity.kelly_leverage`, `src.alphalab.capacity.leverage_for_drawdown`, `src.alphalab.capacity.growth_rate` | Kelly with its estimation error; the stationary drawdown distribution of Brownian motion |

## Fundamental-quant and attribution

| Priority | Item | Source | Status | Where | Notes |
|---|---|---|---|---|---|
| P1 | Selection, sizing and timing attribution | Paleologo ch. 8 | built | `src.alphalab.attribution.selection_sizing_timing` | an exact three-way split of profit against an equal-size, average-gross version of the same book |
| P1 | Brinson attribution | Active PM | existing | `src.backtest.attribution.brinson_fachler` | allocation, selection and interaction against a benchmark |
| P1 | Stop-loss and leverage policy simulation | Paleologo ch. 8 | built | `src.alphalab.attribution.stop_loss`, `src.alphalab.attribution.volatility_target` | both causal: yesterday's equity or volatility only |
| P1 | Financial-statement factors | Paleologo ch. 5; Active PM | existing | `src.equity.factors.compute_all`, `src.equity.factors.compute` | value, quality and revision factors from point-in-time statements (needs your fundamentals file) |
| P1 | Event studies: earnings and announcements | Paleologo ch. 8 | existing | `src.stats.event_study.event_study`, `src.stats.event_study.buy_and_hold_abnormal_return` | market, factor and constant-mean models |
| P1 | Consensus-relative valuation signals | Paleologo ch. 5 | not built | none | needs analyst-estimate data the platform does not have |
| P2 | Factor-mimicking portfolios | Paleologo appendix 11.5 | built | `src.riskengine.factor_model.fit_cross_sectional` | the per-date regression slopes are the returns of factor-mimicking portfolios; custom exposures can be passed in |
| P2 | Smart-beta decomposition: factor return versus skill | Advances ch. 21-25 | not built | none | needs a manager's holdings history; the building blocks (factor model, attribution) are here |

## Validation (López de Prado)

| Priority | Item | Source | Status | Where | Notes |
|---|---|---|---|---|---|
| P0 | Triple-barrier labels, meta-labelling, sample uniqueness, purged CV, CPCV, deflated Sharpe | AFML | existing | `src.finml.pipeline.run_pipeline`, `src.finml.cv.purged_folds`, `src.finml.events.triple_barrier_events` | the Financial ML Lab |

## Summary

46 items: 30 built, 13 existing, 0 partial, 3 not built.
