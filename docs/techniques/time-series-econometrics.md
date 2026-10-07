---
title: "Time-series econometrics: ARIMA, VAR, VECM, GARCH, Kalman and factor models"
slug: time-series-econometrics
difficulty: 3
chapter: Ch. 13
prerequisites: [unit-roots-and-cointegration, dynamic-covariance]
stages: []
files: [src/econometrics/arima.py, src/econometrics/var.py, src/econometrics/garch.py, src/econometrics/statespace.py, src/econometrics/factor.py]
figures: []
tests: [tests/test_econometrics.py]
models: []
---

# Time-series econometrics: ARIMA, VAR, VECM, GARCH, Kalman and factor models

## In one sentence

`src.econometrics` is the classical forecasting toolbox written from scratch, with every estimator checked against `statsmodels` or `arch`: ARMA models, vector autoregressions, error correction, volatility models, state-space filters and dynamic factors.

## The idea

**ARMA(p, q)** (`fit_arima`) writes a series as autoregressive plus moving-average terms. The likelihood is evaluated exactly, by running a Kalman filter on the state-space form (`arma_state_space`) with a stationary initial state; `select_arima` ranks orders by AIC or BIC. **VAR(p)** (`fit_var`) regresses a vector on its own lags, with impulse responses, forecast-error variance decompositions and Granger tests. For cointegrated series `johansen` finds the rank and `fit_vecm` and `error_correction_model` fit the error-correction form `dy = a b' y_{t-1} + sum G dy_{t-i} + e`. `fit_bvar` shrinks a VAR toward a random walk with the **Minnesota prior** so that many assets can share a short sample.

**GARCH** (`fit_garch`) models the variance as `s_t^2 = w + a e_{t-1}^2 + b s_{t-1}^2`; the `gjr`, `tgarch` (Zakoian's threshold model on the standard deviation) and `egarch` variants let bad news raise volatility more than good news, and the distribution can be normal or Student-t. `walk_forward_volatility` refits through time and returns forecasts using only past data, `ewma_volatility` is the RiskMetrics benchmark and `qlike` the loss function used to compare variance forecasts. **State-space** models (`StateSpace`, `local_level`, `local_linear_trend`, `time_varying_regression`, `kalman_hedge_ratio`) give the Kalman filter, smoother and likelihood; a time-varying hedge ratio is the standard use in pairs trading. **Dynamic factor** models (`fit_dynamic_factor`, `nowcast`, `bai_ng_criteria`) extract a few common factors from many series.

## Why it matters

These are the models behind most of the forecasting in the literature the platform is measured against. Knowing whether a volatility forecast beats EWMA, whether a lead-lag survives shrinkage, or whether a trend filter adds value over a moving average needs implementations whose likelihoods and forecasts you can trust.

## How this repo uses it

Four strategies in the library use these modules: `kalman_trend`, `garch_vol_managed`, `evt_risk_managed` and `bvar_lead_lag` (see [the guide](econometric-strategies.md)). The other users are pairs trading (Kalman hedge ratio), regime detection and the DCC covariance.

## What we found

The ARMA likelihood equals the `statsmodels` value to numerical precision once the stationary filter is used (the steady-state shortcut that makes it fast matches the full filter to 1e-12), VAR, VECM and Johansen results match `statsmodels`, and the GARCH family matches `arch` (for TGARCH, `arch` with power one: its constraint excludes some admissible solutions, so the test compares likelihoods and volatilities at our parameters). In [the findings](../institutional_findings.md), order selection on SPY returns shows what one expects for daily returns: very little linear predictability, so the selected orders are small and the BIC differences between candidates are modest.

## Pitfalls

- Daily returns have almost no linear predictability; a significant ARMA coefficient at 1% is usually microstructure or an outlier.
- GARCH parameters near the stationarity boundary (alpha + beta close to one) make long-horizon forecasts unstable.
- A VAR with many assets and few observations overfits; use the Bayesian version.
- Johansen is sensitive to the deterministic specification and the lag order.

## Try it

```python
import numpy as np, pandas as pd
from src.econometrics import fit_arima, fit_garch

rng = np.random.default_rng(0)
e = rng.standard_normal(1500)
y = pd.Series(np.zeros(1500))
for t in range(1, 1500):
    y[t] = 0.3 * y[t - 1] + e[t]
fit = fit_arima(y, order=(1, 0, 0))
print(np.round(fit.ar, 2))
g = fit_garch(pd.Series(rng.standard_t(6, 2000) * 0.01) * 100, model="garch", dist="t")
print({k: round(float(v), 3) for k, v in g.params.items()})
```
