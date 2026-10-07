# Architecture

How the platform is organised, what each layer guarantees, and how to extend it. For the data flow of one backtest see [how the pieces connect](platform_integration.md); for the reasons behind the choices see the [research survey](research_survey.md); for what exists and what does not see the [capability matrix](capability_matrix.md).

## Layers

```
                         ┌────────────────────────────────────────────────────────────────────┐
  Interfaces             │  quant CLI · dashboard (quant serve) · notebooks · experiments/*.py │
                         └──────────────────────────────┬─────────────────────────────────────┘
                                                        │ PipelineSpec (YAML / dict)
                         ┌──────────────────────────────▼─────────────────────────────────────┐
  Orchestration          │  Pipeline → ExperimentManager (run id = hash(spec, data, config))   │
                         │  src/framework · src/research_db · src/ops (store, registry, HPO)  │
                         └───────┬─────────────────────┬──────────────────────┬───────────────┘
                                 │                     │                      │
                  ┌──────────────▼────────┐ ┌──────────▼──────────┐ ┌─────────▼──────────────┐
  Plug-ins        │ MODELS (90)           │ │ ALLOCATORS (14)     │ │ DETECTORS (7)          │
                  │ score / forecast      │ │ weights from        │ │ HMM, GMM, BOCPD, rules │
                  │ src/strategies        │ │ forecasts           │ │ src/framework          │
                  └──────────────┬────────┘ └──────────┬──────────┘ └─────────┬──────────────┘
                                 │                     │                      │
                         ┌───────▼─────────────────────▼──────────────────────▼───────────────┐
  Methods                │  src/stats · src/probability · src/econometrics · src/models        │
                         │  src/portfolio · src/risk · src/signals · src/validation            │
                         └──────────────────────────────┬─────────────────────────────────────┘
                                                        │
                         ┌──────────────────────────────▼─────────────────────────────────────┐
  Markets                │  MarketBundle: prices (total-return indices) · returns · investable │
                         │  · macro panel (signals, carry) · asset_class                       │
                         │  src/data (ETFs, macro, CFTC, crypto) · src/assets (futures, FX,    │
                         │  commodities, rates) · src/derivatives (options) · src/microstructure│
                         └──────────────────────────────────────────────────────────────────────┘
```

**The bundle is the seam.** A `MarketBundle` holds one total-return index per instrument (so the engine's simple returns are the instruments' excess returns), an `investable` mask, a macro panel carrying every per-instrument signal (`CARRY_<name>`, `BASISMOM_<name>`, macro series with their release lags) and an `asset_class` label. Everything above the seam, the 90 models, the allocators, the costs, validation and reports, is asset-class-agnostic. Adding an asset class means writing a function that returns a bundle (see `src/assets/bundles.py`), not touching the framework. This is what lets `carry_xs` run on commodity futures, currencies and bonds with the same code.

Options are the exception, by necessity: an option position is a contract with an expiry and a strike, not a return series, so `src.derivatives` has its own engine (`OptionBacktester`) with the same discipline (no look-ahead, explicit costs) and its own strategy interface. Microstructure is a set of simulators and estimators, not a backtest layer.

## Packages

| Package | Role | Key invariant |
|---|---|---|
| `src/data` | download, clean, hash and version raw data | raw files are immutable and checksummed; every series carries its availability date |
| `src/framework` | plugin registries, `Pipeline`, experiment manager, tear sheet, validation hooks | run id = hash of spec, data version and config; every model passes the causality check |
| `src/strategies` | the 90 registered models (trend, reversion, carry, value, volatility, ML, deep, econometric, multi-asset, crypto, ...) | a score on day `t` uses data through `t` only |
| `src/stats` | OLS/WLS/GLS, HAC and cluster covariances, panels, Fama-MacBeth, unit roots, cointegration, event studies, multiple testing, Sharpe inference | every estimator matches `statsmodels`, `arch` or a closed form |
| `src/probability` | block/stationary bootstrap, Monte Carlo and importance sampling, EVT, copulas, drawdown/ruin/Kelly, Bayesian inference, Markov models | every function takes a seed or a generator; closed forms are checked against simulation |
| `src/econometrics` | ARIMA, VAR/VECM, GARCH family, state space and Kalman, dynamic factors, BVAR | exact likelihoods; forecasts use only data before the origin |
| `src/derivatives` | pricing, Greeks, implied vol, SVI/SSVI, local vol, VIX-style index, synthetic option market, options backtester, volatility strategies | prices are checked against independent methods; fills cross the spread with a one-day lag |
| `src/assets` | futures calendars and continuous contracts, carry, commodity curves (Schwartz-Smith), FX conventions and carry, yield curves and bond analytics, bundles | roll gaps never appear as returns |
| `src/microstructure` | order-flow estimators, a Poisson limit order book, Avellaneda-Stoikov market making, Almgren-Chriss execution | simulators have a known mechanism the estimators must recover |
| `src/models` | neural and statistical forecasters (PatchTST, TSMixer, N-BEATS, N-HiTS, TimeMixer, TFT, autoencoder, contrastive, GNN, diffusion, ...) | deterministic given the seed; trained on matured labels only |
| `src/portfolio`, `src/risk`, `src/backtest` | allocation, covariance, VaR/ES, the daily engine, costs, impact | costs are charged on every rebalance |
| `src/validation` | walk-forward, robustness, deflated Sharpe, Reality Check, SPA, PBO | trials are counted |
| `src/ops` | content-addressed artifact store, model registry, hyperparameter optimisation, purged/CPCV cross-validation, profiling | an artifact is addressed by its hash; a search reports the bias of its own winner |
| `src/webapp`, `src/assistant`, `src/research_db` | dashboard server, explain mode and read-only SQL assistant, research database | read-only access to results; no network calls except the optional LLM |

## Invariants that every addition must keep

1. **No look-ahead.** A decision on date `t` may use data available at the close of `t`. The framework checks this for models by replacing all data after a cutoff with noise and requiring earlier outputs to be unchanged (`check_causality`); each package has its own version (event-study estimation windows, the options backtester's one-day lag, the walk-forward volatility forecasts, the filtered Markov probabilities).
2. **Count the trials.** Anything that looks at results to choose something (a parameter sweep, a tuning run, the strategy survey) records how many looks it took, and the deflated Sharpe ratio uses that number.
3. **Validate against a truth.** Each estimator is tested against a reference implementation, a closed form, or a simulation in which the answer is built in. A simulation-based test that passes proves the code, not the market, and the documentation says so wherever such a market is used.
4. **Reproducibility.** Seeds are explicit; raw data are hashed; run ids are content hashes; artifacts are content-addressed; the registry records the git commit and the config fingerprint.
5. **Honest reporting.** Results are reported as they came out. A number from one run is an observation. Items that cannot be done here (proprietary data, credentials, compute) are listed, not simulated into appearing done.

## Extension points

| To add | Do this | Reference |
|---|---|---|
| a strategy on any bundle | subclass `ForecastModel`, decorate with `@register_model`, implement `score`; set `book` and `rebalance` | [how to add a strategy](how_to_add_a_strategy.md) |
| an asset class | write a function returning a `MarketBundle` with the total-return indices and `CARRY_*`/signal columns | `src/assets/bundles.py` |
| real futures, FX or yield data | produce the long contract table (date, contract, expiry, price), the spot-and-rates table, or the yield panel documented in the module, then call the bundle builder | [futures](techniques/futures-and-commodity-curves.md), [FX](techniques/fx-carry-and-momentum.md), [rates](techniques/yield-curves-and-fixed-income.md) |
| real option chains | `src.derivatives.synthetic.load_option_csv` maps a vendor CSV onto the option schema, `validate_chain` rejects bad quotes, then use `OptionBacktester` as for the synthetic market | [option pricing](techniques/option-pricing-and-greeks.md) |
| an option strategy | subclass `OptionStrategy`; return `Order` and `Hedge` objects from the chain, positions and history | [option strategies](techniques/option-strategies-and-vol-premium.md) |
| an allocator | register in `ALLOCATORS`; receive the forecasts, regimes and covariance in a `Context` | `src/framework/allocators_portfolio.py` |
| a hyperparameter search | `Study(space).optimize(objective, method=...)` or `quant tune`; read `selection_report()` | [research operations](techniques/research-operations.md) |

## Testing strategy

| Kind | What it checks | Where |
|---|---|---|
| Unit and reference | each estimator against `statsmodels`, `arch`, `scipy` or a closed form to numerical precision | `tests/test_stats.py`, `test_econometrics.py`, `test_probability.py`, `test_derivatives.py`, `test_assets.py`, `test_microstructure.py`, `test_ops.py` |
| Leakage and causality | change the future, the past must not change; embargo and purging leave no overlap; estimation windows end before events | `tests/test_strategies.py`, `test_framework.py`, the causality tests in each package |
| Synthetic ground truth | the machinery recovers what the simulation built in (Schwartz-Smith factors, the FX carry premium, the variance premium, order-flow impact, planted slopes and jumps) | the `synthetic` tests in `test_assets.py`, `test_derivatives.py`, `test_microstructure.py`, `test_stats.py` |
| Backtest accounting | the sum of daily P&L equals the change in equity; orders fill at the stated prices; expiries settle at intrinsic value | `tests/test_derivatives.py`, `test_backtest.py` |
| Integration | every registered model runs through the pipeline; documentation, examples and notebooks agree with the code | `tests/test_strategies.py`, `test_docs.py`, `test_webapp.py`, `test_cli_ops.py` |
| Performance | profiling and scaling exponents; a weekly benchmark workflow keeps timings as an artifact | `src/ops/profiling.py`, `.github/workflows/benchmark.yml` |

## Deployment and execution

The `Dockerfile` builds a CPU-only image that runs the test suite as an unprivileged user; `docker-compose.yml` wires the volumes for reports and data. CI (`.github/workflows/ci.yml`) runs the tests with coverage on Python 3.11, the numerical packages on Python 3.10 and 3.12, ruff, and the image build. Experiment sweeps run serially, with joblib or with Dask; a Ray adapter exists but has never been run in this environment. Nothing in the platform needs a cloud service: the research database is SQLite, artifacts are files, and the optional LLM assistant is the only component that would call an external API (and has never been called here, for lack of credentials).
