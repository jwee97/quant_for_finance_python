# Roadmap coverage

Every item of the research roadmap this project was built against, with its status. **Built** means implemented, tested and run with a recorded result. **Partly** says what is missing. **Not built** says why.
Nothing here claims that a built item *works* as a way to make money; the [findings digest](generated/findings.md) and the reports say what each one showed.

## Generation 2: adaptive research platform

| Item | Status | Where |
|---|---|---|
| Market regime detection: HMM, Gaussian mixture, BOCPD, volatility-state, named rule-based regimes; `Regime(name, probability)` | Built | [regime-detection](techniques/regime-detection.md), Stages 16 and 31 |
| Regimes named Bull, Bear, HighVol, LowVol, Crisis, Inflation, Deflation | Partly | The framework's composite detector names LowVol, HighVol, Crisis, Inflation and Deflation; Bear and liquidity-crisis are Stage 16 rules (the liquidity-crisis rule fired on 34 days, below the declared minimum, and was reported untested). There is no Bull label |
| Dynamic allocation by regime (Crisis to mean-CVaR, LowVol to MVO, Inflation to commodity tilt) | Built | [regime-adaptive-allocation](techniques/regime-adaptive-allocation.md), Stage 31 |
| Forecast confidence: `Forecast(mean, std, confidence)` and confidence-aware sizing | Built | [probabilistic-forecasting](techniques/probabilistic-forecasting.md), Stages 19 and 30 |
| Alpha combination engine | Built | [forecast-combination](techniques/forecast-combination.md), Stage 24 |
| Hierarchical risk parity and HERC | Built | [hierarchical-risk-parity](techniques/hierarchical-risk-parity.md), Stage 18 |
| Dynamic covariance: DCC-GARCH, orthogonal GARCH | Built | [dynamic-covariance](techniques/dynamic-covariance.md), Stage 17 |
| Dynamic covariance filters; factor stochastic volatility | Not built | Kalman filters are used for pairs hedge ratios and online forecasting, not for covariance; factor stochastic volatility needs MCMC machinery that was not justified for a 15-asset universe |
| Portfolio attribution (allocation, selection, interaction, sleeves, factors) | Built | [attribution](techniques/attribution.md), Stages 20 and 40 |
| Attribution into named strategy components such as momentum, carry and residual | Built | Every multi-model tear sheet has an attribution by model that reconciles exactly to the portfolio return; Brinson-Fachler against equal weight is in every tear sheet. Factor attribution (Stage 40) is still a separate stage |
| Execution model: spread, impact, commission | Built | [execution-and-impact](techniques/execution-and-impact.md), Stage 25 |
| Slippage as a separate term | Not built | Not separable from spread and impact in daily ETF data |

## Generation 3: institutional research platform

| Item | Status | Where |
|---|---|---|
| Research database | Built (SQLite) | [experiment-database-and-reproducibility](techniques/experiment-database-and-reproducibility.md), Stage 29. DuckDB and Postgres were not used |
| Experiment manager: one command from experiment to comparison | Built | `quant backtest`, `quant run`, `quant leaderboard`, `quant compare` |
| Online learning: recursive least squares, Kalman filter, online gradient descent, Bayesian updating | Built | [online-learning](techniques/online-learning.md), Stage 22 (NLMS is the online gradient step for squared loss; Bayesian updating appears in the Bayesian portfolio stage, not as an online forecaster) |
| Macro layer: VIX, MOVE, yield curve, PMI proxy, inflation, Fed funds, unemployment, credit spreads | Built | [macro-and-alternative-data](techniques/macro-and-alternative-data.md), Stages 15 and 23. The ISM PMI is proprietary; the Philadelphia Fed survey is the proxy |
| Alternative data: CFTC positioning, options-implied volatility, credit spreads | Built | Stage 23 |
| ETF flows, analyst revisions, earnings revisions, CDS spreads | Not built | No free data source reachable from the build environment |
| Bayesian portfolio construction | Built | [bayesian-portfolio-construction](techniques/bayesian-portfolio-construction.md), Stage 21 |
| Explainable AI: SHAP, integrated gradients, feature importance | Built | [explainability](techniques/explainability.md), Stage 35 (Shapley values exact by enumeration, not the `shap` library) |
| Forecast calibration: Platt, isotonic, reliability diagrams | Built | [calibration](techniques/calibration.md), Stages 19 and 35 |
| Distributed research: Joblib, Dask, Ray | Built (Ray untested) | [distributed-experiments](techniques/distributed-experiments.md), Stage 26. Serial, joblib and dask were run and agree exactly; the Ray adapter exists in the executor but Ray is not installed here, so it was never run |
| Research dashboard | Built | `quant dashboard` (a self-contained HTML page: ledger, leaderboard, parameter sensitivity, IC, walk-forward, risk, figures, power). Not a live server |

## Generation 4: cutting-edge research

| Item | Status | Where |
|---|---|---|
| PatchTST, TSMixer, TimeMixer, N-BEATS, N-HiTS | Built | [deep-time-series-models](techniques/deep-time-series-models.md), Stages 27 and 33. The TimeMixer model is a simplified multiscale mixer in its spirit, not the paper's architecture |
| Chronos | Built (zero-shot) | [foundation-models](techniques/foundation-models.md), Stage 34 |
| TimesFM | Built (zero-shot plug-in) | `timesfm` model (TimesFM 2.5, 200M, optional dependency); causality-checked, one exploratory run in [feature_audit](feature_audit.md); no pre-registered stage |
| TimeGPT | Not built | Paid API; no credentials |
| Fine-tuning and transfer learning of foundation models | Not built | Zero-shot only: fine-tuning on a few hundred monthly points is overfitting by construction |
| Graph neural networks | Built | [graph-neural-networks](techniques/graph-neural-networks.md), Stage 33 (15 nodes) |
| Diffusion models | Built | [diffusion-scenarios](techniques/diffusion-scenarios.md), Stage 36 |
| Bayesian deep learning | Built | [bayesian-deep-learning](techniques/bayesian-deep-learning.md), Stage 33 |
| Causal inference: double ML, instrumental variables, difference in differences | Built | [causal-inference](techniques/causal-inference.md), Stage 37 |
| Causal forests | Partly | The R-learner with a ridge effect model is implemented; a forest-based effect model is not |
| Reinforcement learning | Built | [reinforcement-learning](techniques/reinforcement-learning.md), Stage 38 (evolution strategies, linear policy) |
| LLM research assistant | Partly | The document pipeline, extraction schema, cache, backends and read-only SQL guard exist; **no language model was ever called** (no API credentials); results use the offline lexicon. Earnings-call and macro-news summarisation were not built (no data) |

## Strategies

| Family | Built | Not built |
|---|---|---|
| Time series | Dual momentum, volatility breakout, Donchian, moving-average crossover, volatility carry (variance carry), trend with ATR filter | |
| Cross-sectional | Cross-sectional momentum, relative strength, low volatility, carry, defensive beta, value (price proxy), quality (price proxy) | Accounting-based value and quality |
| Statistical arbitrage | Cointegration, Kalman pairs, PCA residual, sparse mean reversion (LASSO basket), ETF NAV arbitrage (simulation) | Basket arbitrage on live constituents |
| Volatility | Volatility risk premium, variance carry, implied versus realised | Dispersion trading (conceptual only: no option-level data) |
| Fixed income | Curve steepener, butterfly, carry and roll-down, duration timing | |
| Macro | Inflation rotation, yield-curve regimes, dollar strength, commodity supercycle, risk-on/risk-off | |
| Crypto | Funding-rate carry (also the perpetual carry), basis trading, stablecoin flow | Cross-exchange spreads (one venue reachable) |

## The pipeline

Market data, feature factory (`src/features`), regime detection, forecast models, forecast confidence, forecast combination, portfolio construction, execution model, risk engine, walk-forward validation, attribution and the experiment database are all
built and run end to end by `Pipeline.run`; adding a strategy is writing one forecast model ([how to add a strategy](how_to_add_a_strategy.md)).

## Generation 7: one engine, one ledger, one strategy interface

The request: unify instruments, market data, engine, ledger, strategies, risk and costs, and analysis so FX, futures, crypto, options and swaps run in one portfolio. See the [architecture](architecture.md#the-multi-asset-engine-one-engine-one-ledger-one-strategy-interface) and the [capability matrix](capability_matrix.md).

| Item | Status | Where |
|---|---|---|
| Unified instrument layer with the core fields and FX, futures, crypto, option and swap families | Built | [unified-instrument-model](techniques/unified-instrument-model.md), `src/instruments` |
| Vendor-neutral point-in-time market data; CSV, Parquet, Arrow, JSONL, SQL, REST polling, WebSocket loaders; data contracts | Built | [point-in-time-market-data](techniques/point-in-time-market-data.md), `src/marketdata` |
| Deterministic event-driven engine (order, fill, funding, margin, roll, expiry, exercise, assignment, coupon, corporate-action events) | Built | [event-driven-engine-and-ledger](techniques/event-driven-engine-and-ledger.md), `src/engine` |
| One ledger with reconciliation and invariant tests | Built | `src/ledger`, `tests/test_ledger.py`, `tests/test_engine.py` |
| Futures rolls (calendar, volume, open interest), back-adjusted histories, roll costs | Built | [contract-lifecycle](techniques/contract-lifecycle.md); only the calendar rule is exercised by tests |
| Perpetuals: funding, mark price, liquidation, isolated and cross margin, inverse contracts | Built | [contract-lifecycle](techniques/contract-lifecycle.md) |
| Options: chains, expiry, exercise, assignment, hedging, premium and margin modes | Built | [contract-lifecycle](techniques/contract-lifecycle.md), `strategies/volatility.py` |
| Swaps module: contracts, schedules, cashflows, curves, pricing, risk, strategies; PV01, carry, roll-down, basis swaps | Built | [interest-rate-swaps](techniques/interest-rate-swaps.md), `src/swaps` |
| Cross-currency basis swaps in an engine run | Partly | priced and par-spread solved; never traded in a run |
| Common strategy API (Signal, Target, Order; all hooks) | Built | [multi-asset-strategy-api](techniques/multi-asset-strategy-api.md) |
| Strategy families: trend, carry, basis, relative value, volatility, regime ensembles, ML | Built | `src/engine/strategies/` |
| Bayesian regime filters | Partly | a two-state HMM and a volatility-quantile filter are built; a Bayesian change-point filter exists in `src/models/regimes.py` but is not wired to the ensemble |
| First-class costs, constraints, mixed-portfolio risk | Built | [costs-constraints-and-capacity](techniques/costs-constraints-and-capacity.md) |
| Cost-aware optimisation, capacity, stress scenarios | Built | `src/engine/optimise.py`, `src/engine/risk.py` |
| Paper-trading mode | Built | [paper-trading-and-replay](techniques/paper-trading-and-replay.md) |
| Broker adapters | Not built | an adapter needs credentials and a sandbox; the interface, a paper broker and a position reconciliation exist |
| Tick data / limit-order-book fills in the engine | Not built | needs tick data; the engine's event schema can carry depth, a book fill model is not written |
| Dashboard support for futures, derivatives and swaps | Not built | result tables are available; the dashboard shows the ETF pipeline only |
| Mixed-asset example with a modern strategy | Built | `src/engine/demo.py` (`run_mixed_asset_demo`) |

## Run but not demonstrated

Docker and the continuous-integration workflow exist as files; no Docker daemon was available to build the image, and the workflow has not run on a hosted runner.
