# Capability matrix: the platform after Generation 8

An audit against the target of an *institutional multi-asset quantitative research platform*. Companion to [the Generation 2-4 audit](feature_audit.md); see the [research survey](research_survey.md) for why each capability matters and [the architecture](architecture.md) for how it fits together.

**Statuses.**

- **COMPLETE**: production code, no placeholder logic, tests that check it against a reference or a known truth, reachable from the normal workflow (the pipeline, the CLI, or a documented function), documented, with a runnable example.
- **PARTIAL**: built and tested, but something named in *Missing* is not, usually because it needs data, credentials or compute that this environment does not have.
- **NOT IMPLEMENTED**: not built, with the reason.

**What "complete" does not mean.** For asset classes with no free data (options, futures contracts, FX forwards, order books) the code is complete and tested against *synthetic markets with a stated truth*; those tests prove the code, and results on those markets say nothing about real markets. A capability that needs a vendor feed to say something about reality is marked PARTIAL for exactly that reason.

**Priority** is of the remaining gap: High (blocks use on real data for the capability), Medium, Low, or "n/a" when nothing is missing.

## Before and after

The state of the repository when this generation started (commit `0b67953`: 15 ETFs, 80 registered models, 12 allocators, the Generation 1-5 framework) against now.

| Area | Before | After |
|---|---|---|
| Statistics and inference | OLS with HAC in two modules, ADF in the pairs code, DSR/PBO/Reality Check/SPA in `validation` | a full `src/stats`: 8 covariance estimators, panels and Fama-MacBeth, unit roots, cointegration, Granger, event studies, 5 multiple-testing procedures, Sharpe inference, BCa and block bootstrap |
| Probability | block-bootstrap inside validation, an HMM for regimes | `src/probability`: resampling, Monte Carlo and importance sampling, EVT, copulas, drawdown/ruin/Kelly, MCMC, Markov chains |
| Econometrics | GARCH inside DCC, a Kalman filter in two strategies | `src/econometrics`: ARIMA, VAR/BVAR, VECM, GARCH/GJR/EGARCH, state space, dynamic factors, forecast loss functions |
| Asset classes | ETFs; crypto perpetuals from one venue | + futures (continuous contracts, carry, roll yield), commodity curves, FX, rates and bonds, a multi-asset bundle, options |
| Derivatives | none | pricing (BSM, Black-76, Bachelier, binomial, LSMC, Heston, Merton, SABR), Greeks, implied vol, SVI/SSVI, local vol, VIX-style index, a synthetic option market, an options backtester, 8 volatility strategies, dispersion |
| Microstructure | square-root impact in the daily engine | order-flow estimators, a limit-order-book simulator, Avellaneda-Stoikov market making, Almgren-Chriss execution |
| Machine learning | ridge, online ridge, deep window (PatchTST, TSMixer, N-BEATS, N-HiTS, TimeMixer), Chronos, TimesFM | + gradient-boosting and forest learners (`ml_trees`), TFT, autoencoder and contrastive representation learning (`deep_representation`) |
| Platform | experiment manager, SQLite research DB, distributed sweep, dashboard | + content-addressed artifact store, model registry, hyperparameter optimiser with selection-bias report, purged/CPCV cross-validation, profiling, `quant tune/benchmark/registry`, CI matrix and weekly benchmark |
| Strategies | 80 registered models, 12 allocators | 103 models (+ 4 econometric, 4 multi-asset carry/value, 2 ML in Generation 6; + 13 alpha, rebalancing and economic-outlook styles in Generation 8) and 17 allocators (`kelly`, `black_litterman`; `beta_neutral`, `liquidity_cap`, `tca_mvo`) |

## The matrix

### Asset classes

| Capability | Status | Evidence | Missing | Priority |
|---|---|---|---|---|
| ETFs and equity indices | COMPLETE | `src/data`, `data/processed/prices_adjusted.csv` (15 ETFs, 2006-2026), `quant data check`, bring-your-own-prices CSV | n/a | n/a |
| Single-stock equities | PARTIAL | any wide CSV of adjusted prices runs through the whole pipeline (`quant backtest --prices`); cross-sectional models and factor tools (`src/stats/panel.py`) work on large cross-sections | point-in-time constituents, delisting returns, corporate actions, fundamentals; no free source | High |
| Crypto spot and perpetuals, funding and basis | PARTIAL | `src/data/crypto.py`, `src/strategies/crypto.py` (`funding_carry`, `basis_reversion`, `stablecoin_flow`), `tests/test_crypto.py` | one venue only (Deribit); other exchanges refuse connections here; no cross-exchange spreads or order-book data | Medium |
| Futures: calendars, continuous contracts, roll, carry, basis momentum | COMPLETE | `src/assets/futures.py`, `tests/test_assets.py`, [guide](techniques/futures-and-commodity-curves.md) | real contract-level data (a vendor); the loader contract is documented | n/a (data) |
| Commodity curves and seasonality | COMPLETE | `src/assets/commodities.py` (Schwartz-Smith by Kalman filter, curve factors, seasonal factors with HAC), recovery test on a simulated curve | n/a for the code; real curves need a vendor | n/a (data) |
| Options: pricing, Greeks, implied vol | COMPLETE | `src/derivatives/pricing.py`, `greeks.py`, `iv.py`; checked against parity, tree convergence, Monte Carlo and finite differences, `tests/test_derivatives.py` | n/a | n/a |
| Options: volatility surface (SVI, SSVI, local vol, VIX) | COMPLETE | `src/derivatives/surface.py`; arbitrage conditions tested | n/a | n/a |
| Options: strategy backtesting with realistic fills | COMPLETE | `src/derivatives/backtest.py`, `strategies.py` (8 strategies); accounting identity, settlement, hedge cost tested | validated on a synthetic market only; no real chain history; margin and early exercise not modelled | High (data) |
| Volatility strategies (VRP, delta-hedged straddle, dispersion, skew, term structure) | COMPLETE | `strategies.py`, `volatility.py`; dispersion world with model-fair implied vols | single-stock option data for real dispersion | Medium (data) |
| Fixed income: curves, duration, convexity, key-rate durations, carry and roll-down | COMPLETE | `src/assets/rates.py` (Nelson-Siegel, Svensson, dynamic NS, bootstrap, bond analytics) | credit curves, callable and inflation-linked bonds; CDS data | Medium |
| FX: forwards, carry, momentum, the forward-premium test | COMPLETE | `src/assets/fx.py`, `bundles.fx_bundle`, synthetic market with a crash-risk carry premium | real spot and forward data; cross-currency basis | n/a (data) |
| Multi-asset universe through one pipeline | COMPLETE | `src/assets/bundles.py::multi_asset_demo_bundle`, `tests/test_assets.py` (every instrument carries a signal, models causal, pipeline runs) | demo is synthetic; a real mixed book needs the data above | n/a (data) |

### Strategy families

| Capability | Status | Evidence | Missing | Priority |
|---|---|---|---|---|
| Trend and time-series momentum | COMPLETE | `tsmom`, `dual_momentum`, moving-average systems, breakouts, `kalman_trend` and others in `src/strategies`; survey in [strategy_survey](strategy_survey.md) | n/a | n/a |
| Cross-sectional momentum, reversal, value, quality, low-vol | COMPLETE | `xs_momentum`, `residual_momentum`, `long_term_reversal`, `value_proxy`, `quality_proxy`, `low_volatility`, `bab`, ... | true value and quality need fundamentals (proxies are price-based) | High (data) |
| Mean reversion (time-series and cross-section) | COMPLETE | `ibs_reversion`, `rsi2`, `stochastic_reversion`, `ou_reversion`, `short_term_reversal`, ... | n/a | n/a |
| Carry (cross-asset) and basis momentum | COMPLETE | `carry_xs`, `carry_ts`, `basis_momentum`; recover the premium built into the simulation | not run on real futures/FX data (no data) | n/a (data) |
| Volatility targeting and managed exposure | COMPLETE | `vol_managed_long`, `garch_vol_managed`, `evt_risk_managed`, `src/portfolio` | n/a | n/a |
| Risk parity, HRP/HERC, min-variance, max diversification | COMPLETE | `src/portfolio`, allocators `static`, `min_variance`, `max_diversification` | n/a | n/a |
| Black-Litterman | COMPLETE | allocator `black_litterman` (views from the forecast panel), `tests/test_ml_additions.py` | n/a | n/a |
| Mean-CVaR | COMPLETE | `src/portfolio` (CVaR optimisation), [guide](techniques/mean-cvar.md) | n/a | n/a |
| Kelly sizing | COMPLETE | allocator `kelly`; `src/probability/ruin.py` (closed forms, drawdown constraint) | n/a | n/a |
| Pairs, cointegration, statistical arbitrage | COMPLETE | `cointegration_pairs`, `kalman_pairs`, `pca_residual`, `sparse_basket`; Engle-Granger, Johansen, Kalman hedge ratio | on 15 ETFs the relationships do not hold out of sample (a finding) | n/a |
| Machine-learning strategies | PARTIAL | `ml_ridge`, `ml_trees` (HistGB, random forest, extra trees, optional LightGBM/XGBoost/CatBoost), `online_ridge`, `deep_window`, `deep_representation`, all in one walk-forward protocol | CatBoost is exercised only through a mocked import in CI (the library is optional); no large per-learner hyperparameter search; none earned a positive Sharpe on the ETFs (a finding) | Low |
| Transformers and modern deep forecasters | PARTIAL | PatchTST, TSMixer, N-BEATS, N-HiTS, a compact TFT, a simplified TimeMixer in `deep_window`; Chronos and TimesFM zero-shot plug-ins | TimeMixer is "in the spirit of" the paper; TFT has tests but no study on real data; foundation-model weights and compute for heavier runs | Low |
| Market-making simulation | COMPLETE | `src/microstructure/market_making.py`; P&L decomposition (spread capture, adverse selection, inventory) tested | simulated book only; tick data, latency and queue position | Medium (data) |
| Crypto basis and funding | PARTIAL | see crypto row | multi-venue data | Medium |
| Macro timing and alternative data | PARTIAL | `src/data/macro.py`, `src/data/altdata.py` (CFTC positioning with release lags), `risk_on_off`, `yield_curve_regime`, ... | ISM PMI is proprietary (a regional Fed survey stands in); ETF flows, revisions, CDS spreads, order-flow data have no free point-in-time source | Medium (data) |
| Event-study strategies | PARTIAL | the event-study estimator (`src/stats/event_study.py`) with causal estimation windows | no registered strategy trades an event signal; needs an event calendar (earnings, announcements) | Low |
| Factor timing | PARTIAL | `src/portfolio`, `factor attribution` (Stage 40); regime-adaptive allocation (Stage 31) | no factor-return history beyond the factor data committed under `data/raw`; the adaptive study found no gain over risk parity | Low |

### Probability and statistics

| Capability | Status | Evidence | Missing | Priority |
|---|---|---|---|---|
| Bootstrap: i.i.d., moving-block, circular, stationary, wild, subsampling, BCa; automatic block length | COMPLETE | `src/probability/resampling.py`, `src/stats/inference.py`, `tests/test_probability.py`, `tests/test_stats.py` | n/a | n/a |
| Monte Carlo and variance reduction; importance sampling, cross-entropy | COMPLETE | `src/probability/montecarlo.py`; tests check rare-event probabilities against closed forms | n/a | n/a |
| Bayesian inference: MCMC with R-hat/ESS, Bayesian regression, Bayesian Sharpe, Bayes factors | COMPLETE | `src/probability/bayes.py`; recovery and diagnostics tests | no probabilistic-programming back end (by decision) | Low |
| Markov chains and Markov-switching, HMM regimes | COMPLETE | `src/probability/markov.py`, `src/framework/regimes.py`; filtered probabilities tested causal | n/a | n/a |
| Extreme value theory: GPD, Hill, GEV, extremal index, conditional EVT | COMPLETE | `src/probability/evt.py`; matches `scipy`; dynamic VaR backtested | n/a | n/a |
| Copulas: Gaussian, Student-t, Clayton, Gumbel, Frank; scenario simulation | COMPLETE | `src/probability/copulas.py`; family recovery and tail dependence tested | vine copulas | Low |
| Drawdown and ruin analytics; tail-risk estimation | COMPLETE | `src/probability/ruin.py`, `src/risk` | n/a | n/a |
| OLS/WLS/GLS, HAC, clustered, two-way clustered errors; rolling regression; diagnostics | COMPLETE | `src/stats/regression.py`; matched to `statsmodels` | n/a | n/a |
| Panel regression, fixed/random effects, Hausman, Fama-MacBeth, Shanken, GRS | COMPLETE | `src/stats/panel.py` | n/a | n/a |
| Stationarity (ADF, PP, KPSS, variance ratio), cointegration (EG, Johansen), Granger | COMPLETE | `src/stats/timeseries.py`, `src/econometrics/var.py` | n/a | n/a |
| Event studies | COMPLETE | `src/stats/event_study.py` (BMP, Patell, sign, clustering by date) | n/a | n/a |
| Forecast comparison tests | COMPLETE | `src/validation/forecast_tests.py` (Diebold-Mariano, paired Sharpe bootstrap), QLIKE | n/a | n/a |
| Reality Check, SPA, PBO, deflated Sharpe, haircut Sharpe, FDR (BH, BY, Storey), Romano-Wolf, min track record | COMPLETE | `src/validation`, `src/stats/inference.py`, `src/ops/hpo.py::Study.selection_report` | n/a | n/a |

### Econometrics

| Capability | Status | Evidence | Missing | Priority |
|---|---|---|---|---|
| AR, ARMA, ARIMA with exact likelihood and order selection | COMPLETE | `src/econometrics/arima.py`; matches `statsmodels` | seasonal ARIMA | Low |
| VAR, Bayesian (Minnesota) VAR, VECM, ECM | COMPLETE | `src/econometrics/var.py`, `factor.py::fit_bvar` | n/a | n/a |
| GARCH, GJR, TGARCH (Zakoian), EGARCH, Student-t, walk-forward forecasts | COMPLETE | `src/econometrics/garch.py`; matches `arch` | FIGARCH; realised GARCH | Low |
| DCC and multivariate GARCH | COMPLETE | `src/portfolio/dynamic_covariance.py`, allocator `dynamic_cov` | n/a | n/a |
| State space, Kalman filter and smoother, time-varying regression, dynamic factor models, nowcasting | COMPLETE | `src/econometrics/statespace.py`, `factor.py` | n/a | n/a |

### Machine learning

| Capability | Status | Evidence | Missing | Priority |
|---|---|---|---|---|
| Random forest, gradient boosting | COMPLETE | `ml_trees` (`hgb`, `random_forest`, `extra_trees`) | n/a | n/a |
| XGBoost, LightGBM, CatBoost | PARTIAL | optional learners in `make_tree_learner`; XGBoost and LightGBM tested for real when installed (they are in `requirements.txt`) | CatBoost is not installed in CI | Low |
| PatchTST, TSMixer, TimeMixer, N-BEATS, N-HiTS | PARTIAL | `deep_window` (see above) | TimeMixer simplified | Low |
| TFT | PARTIAL | `deep_window(kind=tft)`, `tests/test_ml_additions.py` | no study on real data; compact (three derived channels) | Low |
| Chronos, TimesFM | PARTIAL | zero-shot plug-ins, causality-checked on real weights where they were downloaded | weights are an external download; no pre-registered study | Low |
| Graph neural networks | PARTIAL | `src/models/graph_forecast.py`, Stage 33 | not a plug-in; added nothing over the historical mean | Low |
| Autoencoders and contrastive learning | COMPLETE | `deep_representation` (`kind` = autoencoder or contrastive) | n/a | n/a |
| Bayesian deep learning | PARTIAL | Monte-Carlo dropout (Stage 33) | stage-only; no better calibration found | Low |
| ML in the experiment framework | COMPLETE | every learner is a registry model with forecast distributions, costs, validation and trial counting | n/a | n/a |

### Platform

| Capability | Status | Evidence | Missing | Priority |
|---|---|---|---|---|
| Research database and experiment registry | COMPLETE | `src/research_db`, `src/framework/experiments.py`, `quant leaderboard/compare/sql` | SQLite only (by decision) | n/a |
| Model registry (versions, stages, lineage) | COMPLETE | `src/ops/store.py::ModelRegistry`, `quant registry`, `tests/test_ops.py`, `tests/test_cli_ops.py` | n/a | n/a |
| Artifact storage | COMPLETE | `src/ops/store.py::ArtifactStore` (content-addressed, tamper-detecting) | remote object stores (S3, GCS) | Low |
| Pipeline execution | COMPLETE | `Pipeline`, `quant run`, `experiments/run_all.py` | n/a | n/a |
| Hyperparameter optimisation | COMPLETE | `src/ops/hpo.py` (random, grid, TPE, successive halving), `quant tune`; reports deflated Sharpe, PBO, Romano-Wolf of the winner | multi-fidelity beyond halving; early-stopping integration with the pipeline | Low |
| Leakage-safe cross-validation | COMPLETE | `src/ops/cv.py` (purged k-fold, embargo, CPCV paths) | n/a | n/a |
| Distributed execution | PARTIAL | serial, joblib and Dask agree (`tests/test_experiment_manager.py`); a Ray adapter exists | Ray has never been run here | Low |
| Cloud compatibility | PARTIAL | `Dockerfile`, `docker-compose.yml`, CI image build; no cloud-specific dependencies | no deployment manifests (Kubernetes, batch); CI cannot be run locally | Low |
| Configuration management | COMPLETE | `config/*.yaml` with fingerprints, `PipelineSpec` YAML, run ids hash spec, data and config | n/a | n/a |
| CI/CD | COMPLETE | `.github/workflows/ci.yml` (tests with coverage, Python 3.10/3.12 numerical matrix, ruff, docker), `benchmark.yml` (weekly) | no automated deployment target | n/a |
| Benchmarking and profiling | COMPLETE | `src/ops/profiling.py`, `quant benchmark` | n/a | n/a |
| Dashboards | COMPLETE | `quant dashboard` (static), `quant serve` (interactive), `tests/test_webapp.py` | the dashboard does not yet show options or futures analytics | Low |
| Documentation per module | COMPLETE | 19 new technique guides with theory, repository use, findings, pitfalls and a runnable example; this page, the survey, the architecture | n/a | n/a |

### Multi-asset engine (Generation 7)

The contract-level engine: one engine, one ledger, one strategy interface for FX, futures, crypto, options and swaps. See the [architecture](architecture.md#the-multi-asset-engine-one-engine-one-ledger-one-strategy-interface) and the guides on the [instrument model](techniques/unified-instrument-model.md), [point-in-time data](techniques/point-in-time-market-data.md), [the engine and ledger](techniques/event-driven-engine-and-ledger.md), [contract lifecycle](techniques/contract-lifecycle.md), [swaps](techniques/interest-rate-swaps.md), [the strategy API](techniques/multi-asset-strategy-api.md), [costs and constraints](techniques/costs-constraints-and-capacity.md) and [paper trading](techniques/paper-trading-and-replay.md).

| Capability | Status | Evidence | Missing | Priority |
|---|---|---|---|---|
| Unified instrument layer (core fields, cash styles, calendars, registry, JSON round trip) | COMPLETE | `src/instruments`, `tests/test_instruments.py` | holiday rules are generated, not a vendor calendar; ad-hoc closures need `extra_holidays` | Low |
| FX spot, forwards, FX swaps | COMPLETE | `src/instruments/fx.py`; a forward is marked from the two discount curves and the spot, and settles by exchanging the currencies (`tests/test_swaps.py`) | non-deliverable forwards are specified and settle against a fixing but are not exercised in an engine run | Low |
| Cross-currency basis swaps | PARTIAL | `CrossCurrencyBasisSwap`: priced with notional exchanges, par spread solved and checked (`tests/test_swaps.py`); marked in the engine through the same model | no engine run with a cross-currency position; no cross-currency curve construction (the basis is a spread on the second leg) | Medium |
| Futures chains, roll methods, back-adjusted histories, roll cost attribution | COMPLETE | `src/instruments/futures.py`, `lifecycle.py`, `strategies/common.py`; independent recomputation through rolls (`tests/test_engine.py`) | n/a | n/a |
| Crypto spot, perpetuals (funding, mark, liquidation, margin modes), dated futures, venue conventions | COMPLETE | `src/instruments/crypto.py`, funding and liquidation tests | five venues' conventions are parameters, not live-verified; one venue's historical data (Deribit) | Medium |
| Crypto options | PARTIAL | the `Option` model carries an `inverse` flag for coin-settled options | no engine run or data for coin-settled options | Low |
| Listed options on spot, index and futures: exercise, assignment, expiry, Greeks | COMPLETE | `lifecycle.py`; analytic P&L tests for every settlement type (`tests/test_lifecycle.py`) | partial exercise of a position; early assignment is a configurable rule, not a model | Medium |
| Rate swaps: schedules, curves, cashflows, PV01, key rates, carry and roll-down, basis swaps | COMPLETE | `src/swaps`, `tests/test_swaps.py` | amortizing profiles are covered by unit tests but not by an engine run; no inflation or OIS-compounded legs | Medium |
| Bonds | PARTIAL | `Bond` specification; the cashflow dates function is shared with swaps | no engine run with a bond position; no yield-based marking in the engine | Medium |
| Vendor-neutral market data: events, availability, contracts, five file formats, SQL, REST, WebSocket | COMPLETE | `src/marketdata`, `tests/test_marketdata.py` (REST over a local HTTP server, WebSocket over a local server) | no real vendor adapters (each needs credentials and a licence) | n/a (data) |
| Event-driven engine: ordering, custom clock, latency, partial fills, bid/ask fills, depth-aware fills, limit/market/stop, accrual | COMPLETE | `src/engine`, `tests/test_engine.py` | no queue-position model for resting orders (a limit order fills when the market reaches it) | Medium |
| One ledger: multi-currency, margin, journal identity, reconciliation | COMPLETE | `src/ledger`, `tests/test_ledger.py` | portfolio margin is a scenario-scan approximation, not an exchange model | Medium |
| Invariants: cash and position conservation, no self-generated P&L, attributable P&L, no look-ahead, expired contracts, deterministic replay | COMPLETE | `tests/test_engine.py`, `test_ledger.py`, `test_marketdata.py` | n/a | n/a |
| Common strategy API with signals, targets, orders and the full hook set | COMPLETE | `src/engine/strategy.py` | n/a | n/a |
| Strategy families: trend, carry (futures, FX, funding), cash-and-carry and basis reversion, calendar, butterfly and FX-triangle relative value | COMPLETE | `src/engine/strategies/`, `tests/test_engine_strategies.py` (constructed data with a known answer) | cross-venue crypto spreads and curve-steepener futures are not built (swap curve trades are) | Medium |
| Volatility: variance premium, gamma, skew, term structure with vega sizing and delta hedging | COMPLETE | `strategies/volatility.py` | dispersion needs single-name option data; hedging is daily, not intraday | Medium (data) |
| Regime-aware ensemble (volatility and HMM regime filters, risk overlay, volatility target, drawdown brake) | COMPLETE | `strategies/ensemble.py` | Bayesian regime filters beyond the HMM; risk-parity overlay uses shadow returns, which ignore costs | Low |
| Machine learning on the common API: walk-forward ridge/lasso/trees, regime-conditioned models, online forecast combination, meta-labelling | COMPLETE | `strategies/ml.py`; causality and no-future-dependence tests | tree learners are small and shallow; no deep models on this API | Low |
| First-class costs and financing (commission, exchange fees, spread, slippage, impact, funding, borrow, margin interest, roll and FX conversion costs, participation limits) | COMPLETE | `src/engine/costs.py`, `tests/test_engine_risk.py` | calibration needs real fills (a regression is provided and tested on simulated trades) | Medium (data) |
| Constraints and mixed-portfolio risk (gross, net, leverage, margin, turnover, class, currency, factor, liquidity, concentration, drawdown and volatility targets; VaR, CVaR, Greeks, PV01, scenarios, funding and rollover risk) | COMPLETE | `constraints.py`, `risk.py`, `tests/test_engine_risk.py` | constraints apply per strategy, not across strategies | Medium |
| Cost-aware optimisation, impact calibration, capacity | COMPLETE | `src/engine/optimise.py` | capacity needs real volume data; the synthetic data has none | Medium (data) |
| Paper trading, deterministic replay, broker interface | PARTIAL | `src/engine/broker.py`; a paper session reproduces the backtest digest (`tests/test_engine.py`); `PaperBroker` | no real broker adapter: an implementation needs credentials and a sandbox account | High (credentials) |
| Mixed-asset example: FX, futures, crypto, options, swaps in one portfolio with a modern strategy | COMPLETE | `src/engine/demo.py`, `tests/test_engine_risk.py::test_the_mixed_asset_demo_meets_the_success_criteria` | runs on a synthetic market only | n/a (data) |
| Depth-aware fills (walking displayed book levels) | COMPLETE | `execution._walk_book`, `events_from_depth`, tests in `tests/test_engine.py` (VWAP, partial fills, FOK, limit and IOC, selling) | none for snapshots | n/a |
| Tick data replay with queue position and latency in the engine | NOT IMPLEMENTED | `src/microstructure` has a standalone book simulator; depth snapshots can be replayed | no queue-position model, no message-level book reconstruction; needs tick data to be meaningful | Medium (data) |
| Dashboard support for futures, derivatives and swaps | NOT IMPLEMENTED | the result's tables (`pnl_matrix`, `cost_summary`, `last_risk`) are the inputs | the dashboard still shows the ETF pipeline only | Low |

### Strategy styles, execution algorithms and cash flows (Generation 8)

The item-by-item map to the taxonomy this section answers is [algorithmic trading and investment strategies](algorithmic_trading.md), generated from the code. Execution and high-frequency capabilities are **simulators on a stylised market**: their tests check them against closed forms and known truths (the Almgren-Chriss solution, the risk of the unexecuted list, an edge that falls as the delay of the order grows), and a result says what a setting does to a cost, not what your trades would cost.

| Capability | Status | Evidence | Missing | Priority |
|---|---|---|---|---|
| Alpha-generating styles: long-term and short-term price rules, a walk-forward event study | COMPLETE | `src/strategies/alpha_styles.py`, `tests/test_alpha_styles.py`, [guide](techniques/alpha-generating-styles.md), [survey part two](strategy_survey_2.md) | none; on 15 ETFs none earned a net Sharpe above equal weight | n/a |
| Company outlook, company news and corporate-action strategies | PARTIAL | `news_sentiment`, `panel_signal` and `event_study_drift` read headlines, scores and events from `data/user/`; tested on planted data with lags and expiry | no free source of headlines, analyst revisions, corporate actions or index changes, so they were not run on real data | High (data) |
| Portfolio rebalancing styles: strategic weights with a calendar and a band, month-end flows, flight to quality, market outlook, market neutral | COMPLETE | `src/strategies/rebalance_styles.py`, `src/framework/allocators_overlay.py`, `tests/test_rebalance_styles.py`, `tests/test_allocators_overlay.py`, [guide](techniques/portfolio-rebalancing-styles.md) | index reconstitution needs an index-changes file | Low (data) |
| Risk reduction, hedging and liquidation costs | COMPLETE | `beta_neutral`, `liquidity_cap`, `tca_mvo`, `src/algo/liquidation.py`, [guide](techniques/portfolio-overlays-and-liquidation.md) | `tca_mvo` drops holdings in assets that become non-investable without charging for the sale | Low |
| Cash-flow strategies: deposits, redemptions, dividends, liabilities (LDI), payments | COMPLETE | `src/cashflow`, `tests/test_cashflow.py` (accounting, policy properties, closed forms), [guide](techniques/cash-flow-strategies.md), `quant cashflow`, the dashboard's Cash flows tab | liabilities use one flat discount yield and equal-weight buckets, not key-rate durations; spending is bootstrapped from the portfolio's own history | Low |
| Economic outlook: yield-curve and credit strategies | COMPLETE | `src/strategies/economic_outlook.py`, `tests/test_economic_outlook.py`, [guide](techniques/economic-outlook-strategies.md) | states are learned from about twenty years of history; the credit rule falls back to a credit-against-rates proxy when the spread series is missing | Low |
| Execution styles (aggressive, working order, passive) and single-stock algorithms (VWAP, TWAP, POV, arrival price, implementation shortfall) | COMPLETE | `src/algo/simulate.py`, `src/algo/algos.py`, `tests/test_algo.py`, [guide](techniques/execution-algorithms.md), `quant algo run`, the dashboard's Execution tab | impact parameters are illustrative values from the literature; the passive and dark fill rates are round numbers; calibrating to your own fills needs execution data | Medium (data) |
| Best-execution goals, adaptation tactics (target cost, aggressive and passive in the money), schedule optimisation (QP, exponential trade, exponential residual, trade rate) | COMPLETE | `src/algo/optimize.py`, `src/algo/tactics.py`, `tests/test_algo.py`, `tests/test_algo_catalog.py` | the tactics follow documented interpretations of one-line definitions | Low |
| Basket algorithms, minimum trading risk quantity, maximum trading opportunity, program-block decomposition, liquidity seeking | COMPLETE | `src/algo/basket.py`, `src/algo/liquidity.py`, `tests/test_basket.py`, [guide](techniques/basket-and-liquidity-algorithms.md) | the four tactics follow documented interpretations of one-line definitions; correlations in the demonstration basket come from one factor | Low |
| Black-box and high-frequency strategies: pair trading, ETF arbitrage, rebate and liquidity trading, auto market making | PARTIAL | `src/algo/blackbox.py`, `src/algo/hft.py`, `tests/test_blackbox_hft.py`, [guide](techniques/black-box-and-high-frequency-strategies.md) | stylised Ornstein-Uhlenbeck and Poisson markets: no order book, queue position, latency model or real tick data | Medium (data) |
| Dashboard tabs for execution and cash flows | COMPLETE | `src/webapp` (`/api/exec`, `/api/cash`), `tests/test_webapp_labs.py`, `tests/test_webapp_browser.py` | none | n/a |

## What cannot be completed in this environment

| Item | Why | What exists instead |
|---|---|---|
| Real option chains (history) | proprietary or paid (OPRA, CBOE DataShop, OptionMetrics) | schema, validation, vendor-CSV loader, synthetic market, full backtester |
| Real futures contracts, FX forwards, yield panels beyond FRED | vendor data | the documented loader contracts, bundle builders, synthetic worlds with known truth |
| Tick data, level-2 books, trades and quotes | proprietary | estimators and a Poisson limit-order-book simulator |
| Fundamentals, point-in-time constituents, earnings and revision data | proprietary | price-based proxies, the event-study estimator |
| Single-stock options (real dispersion), CDS | proprietary | a model-consistent dispersion world |
| Real broker or exchange adapters | an adapter needs credentials, a sandbox account and the venue's own error handling | the `Broker` interface, a paper broker on the same simulator and a position reconciliation (`LiveBroker` refuses to run) |
| Real vendor feeds for options, futures contracts, FX forwards, swap curves, tick data | licences | loaders, schemas, contracts, a point-in-time store and a synthetic multi-asset market with a known generating process |
| Live LLM calls | no API credentials in this environment | the template backend and the read-only SQL guard are tested; the `--llm` path is unexercised |
| Foundation-model fine-tuning, large transformer runs | compute | zero-shot plug-ins; compact networks on CPU |
| Ray, Docker and GitHub Actions executed locally | not available in the sandbox | the code and config are in place; CI is the evidence (see the commit status) |

## Future roadmap

1. **Data first.** Connect a vendor for option chains and futures contracts through the documented loaders; rerun the surface, strategy and carry studies on real data and replace the simulated results with real ones (or with honest negative results).
2. **Fundamentals and point-in-time equity data** to turn the value, quality and low-risk proxies into the real factors and to enable single-stock event studies.
3. **Options extensions:** early exercise and margin in the backtester, a local-stochastic-volatility model, a surface-dynamics (sticky-strike versus sticky-delta) study.
4. **Microstructure:** replay engine for tick data with queue position and latency; calibrate the impact model to the platform's own fills.
5. **Methodology:** vine copulas, seasonal and fractional ARIMA models, particle-filter stochastic volatility, multi-fidelity tuning that stops losing pipelines early.
6. **Platform:** a remote artifact store, deployment manifests, and options and futures analytics in the dashboard.
