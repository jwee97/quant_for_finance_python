# How the pieces connect

One specification drives the whole chain. `quant run examples/adaptive_pipeline.yaml --tearsheet --group examples` exercises every link below on the 15-ETF bundle.

```
data bundle (prices, macro levels with release lags, CFTC positioning)
   │
   ├─ forecast models  ──► Forecast(mean, std, confidence) per asset and day
   │     momentum, mean_reversion, ml_ridge, online_ridge, deep_window, chronos, timesfm, cftc_positioning, …
   │
   ├─ regime detector  ──► RegimeSeries (probabilities by day; hmm, gmm, bocpd, composite, …)
   │
   ├─ forecast: regime_spread     std ← std × κ(regime)            κ from matured standardised errors in that regime
   ├─ forecast: confidence        P(up) recalibrated (Platt / isotonic), matured labels only
   ├─ combination: decay_weighted weights ∝ IC at the holding period implied by each model's fitted alpha decay
   │
   ├─ allocation                  confidence · forecast_stack · regime_switch · static (HRP, risk parity, …)
   │                              dynamic_cov · bayesian · es_policy
   ├─ risk                        regime volatility targets, regime gross caps, drawdown de-risking
   ├─ execution                   spread + commission + square-root impact at the declared AUM
   │
   └─ evaluation                  tear sheet: net returns, benchmarks, by-regime, Brinson vs equal weight,
                                  attribution by model, cost breakdown, capacity by AUM, forecast quality,
                                  explainability, red flags, validation (causality, deflated Sharpe)
        │
        └─ experiment manager     run id = hash(spec, data version, config fingerprint); SQLite + files
                                  (metrics, tables, forecasts.csv.gz, meta.json with the git commit)
                                  quant leaderboard · compare · sweep · sql --db runs · dashboard
```

## Which spec key feeds what

| Link | Spec key | Code |
|---|---|---|
| Regime → forecast confidence | `forecast.regime_spread` | `src/framework/adaptive.py::regime_spread_scale` |
| Regime → allocation | `allocation: {allocator: regime_switch}` | `src/framework/allocation.py::RegimeSwitch` |
| Regime → volatility target | `risk.mode: regime`, `risk.regime_targets` | `src/framework/risk.py` |
| Regime → risk limits | `risk.limits.gross_caps`, `risk.limits.drawdown` | `src/framework/adaptive.py::RegimeRiskLimits` |
| Regime → backtest report | automatic | `by_regime`, `regime_spread_scale`, `risk_limits` tables |
| Confidence calibration | `forecast.confidence` | `src/framework/adaptive.py::calibrate_confidence` |
| Alpha combination (momentum, mean reversion, ML, confidence, decay) | `combination.rule: decay_weighted` | `src/framework/adaptive.py::decay_trust_weights` |
| Impact → costs and net returns | `execution.aum` | `src/backtest/impact.py` through `Pipeline.run` |
| Impact → strategy comparison | `execution.capacity` / `quant capacity` | `src/framework/analytics.py::capacity_by_aum` |
| Impact → attribution | `cost_breakdown` table | `src/framework/analytics.py::cost_breakdown` |
| Research database | automatic on `quant run` | `src/framework/experiments.py::ExperimentManager.record` |

Nothing here reads data after the date it is used for: every plug-in passes the repository's causality check (all data after a cutoff replaced by noise must leave earlier outputs unchanged), and `tests/test_adaptive_integration.py` repeats it for the adaptive layer.
