---
title: "Heavy-tailed shocks and stochastic volatility in a forecast"
slug: stochastic-volatility-and-levy-forecasting
difficulty: 3
chapter: Platform
prerequisites: [monte-carlo-forecasting, option-pricing-models-and-calibration, short-rate-models-and-kalman-trackers]
stages: []
files: [src/forecast/stochvol.py, src/forecast/registry.py, src/derivatives/levy.py, src/probability/sde.py, src/econometrics/trackers.py, src/econometrics/garch.py]
figures: []
tests: [tests/test_forecast_stochvol.py, tests/test_forecast_methods.py, tests/test_levy.py, tests/test_sde.py, tests/test_trackers.py]
models: []
---

# Heavy-tailed shocks and stochastic volatility in a forecast

## In one sentence

Four more ways to forecast the *spread* of the next returns, borrowed from option pricing (a GARCH model whose shocks come from a fitted variance-gamma or NIG law, a stochastic-volatility Kalman filter and the Heston model) were built, tested on the platform's ETFs, and matched GARCH without beating it.

## The idea

A volatility forecast has two ingredients: where volatility is now and where it is going, and what the daily shock looks like once volatility is known. The existing methods settle the second one by assumption (a Student-t shock in `garch`) or by memory (the resampled past shocks of `fhs`). Option pricing offers richer answers.

**`garch_vg` and `garch_nig`.** The GJR-GARCH volatility model of `garch` is fitted as before (by normal quasi-maximum likelihood). Its standardised residuals are then fitted, by maximum likelihood, with a **variance-gamma** or a **normal inverse Gaussian** law: both have a free skew and a free kurtosis, and their tails are smooth and go beyond the largest shock in the sample, which a resampling cannot. A pool of 200,000 draws from the fitted law, rescaled to unit variance, replaces the Student-t shocks in the simulation. The fit range of the law's drift parameter is wider than for an option market's volatilities, because a daily shock needs a far larger drift parameter than a market volatility does for the same skew.

**`sv_kalman`.** A stochastic-volatility model whose log variance is a latent AR(1), filtered by the Kalman filter on the log of squared returns (the quasi-maximum-likelihood method of Harvey, Ruiz and Shephard). The filter says where log variance is *and how uncertain it is*, and the forecast simulates the volatility forward from that uncertain state: volatility has noise of its own, which GARCH (volatility as a function of past returns) does not have. It has no leverage term.

**`heston`.** The Heston model under the real-world measure, simulated with Andersen's quadratic-exponential scheme. Its parameters come from the **method of moments** on squared returns: the autocorrelation of squared returns decays at the speed of mean reversion and its level gives the variance of the variance, and the covariance of a return with later squared returns gives the leverage correlation. Translating a fitted GARCH through its diffusion limit was tried first and rejected: the GARCH persistence of Heston data is too close to one, and the reversion speed it implies is several times too slow.

None of the four forecasts a direction: the average return is set to zero, so they are tested on the spread, like `garch`.

## Why it matters

If the shock law or the volatility's own noise mattered, these methods would improve the tails and the longer horizons, where GARCH's constant-shape assumption is weakest. That is a testable claim, and the Forecast tab's walk-forward evaluation tests it on the same footing as every other method.

## How this repo uses it

`src/forecast/stochvol.py` has `GarchLevy` (with `GarchVG` and `GarchNIG`), `SVKalman`, `HestonMC` and `heston_moments`; they use `derivatives/levy.py` for the laws and their fits, `econometrics/trackers.py` for the filter and `probability/sde.py` for the Heston simulator. They are registered beside the others (`garch_vg`, `garch_nig` and `heston` in the Monte Carlo family, `sv_kalman` under State space), selectable in the Forecast tab and `quant forecast --methods`, and **not in the recommended set**: the evidence below does not earn them a place.

## What we found

- **In worlds with a known truth they work.** With variance-gamma shocks the fitted law reproduces the skew and the kurtosis of the residuals (a NIG fitted to variance-gamma shocks matches the moments and not the family, as it should) and the forecast has the longer left tail; in a Gaussian world the shocks come out with no skew and no excess kurtosis. The stochastic-volatility filter finds the planted persistence and volatility of log variance, widens the spread after a stormy month, and its ranges hold the outcome about nine times in ten over many origins. The method of moments recovers the Heston speed, level, volatility of variance and (roughly) the leverage correlation on average over simulated samples, and says the volatility of variance is small for returns with no clustering.
- **On real prices they match GARCH and do not beat it.** Walk-forward on SPY, QQQ, EEM, TLT and GLD (2018 to 2026, 202 forecasts a run), average skill against the random walk's CRPS:

| Method | 5 days | 21 days |
|---|---:|---:|
| GJR-GARCH (Student-t shocks) | +3.0% | +1.9% |
| Filtered historical simulation | +2.9% | +2.1% |
| GARCH, variance-gamma shocks | +3.3% | +2.1% |
| GARCH, NIG shocks | +2.8% | +2.0% |
| Heston, by moments | +2.5% | +1.5% |
| Stochastic-volatility Kalman filter | +2.0% | +0.7% |
| HAR | +2.7% | +0.8% |
| RiskMetrics | +1.6% | −0.2% |

  The shock laws are within 0.3 points of GARCH, which is inside what one history of five ETFs can resolve; Heston is about half a point below at both horizons; the filter is 1.0 and 1.2 points below (a stochastic-volatility model has no leverage). On SPY the two shock laws and Heston beat the random walk after the Holm correction (which counts twenty-two candidates: the twenty-one methods other than the random walk, and the blend) at both horizons, and the filter did at five days (+4.6%) but not at 21 (+3.0%); on GLD at 21 days none of them beat it (they were 1.6% to 1.8% behind), like most of the others. The reason the shock law adds little is in the data: the standardised residuals of GARCH fitted to the last five years of these ETFs are only mildly non-normal (excess kurtosis from 0.3 on TLT to 1.6 on EEM, skew from −0.1 to −0.5), so a law with a free skew and kurtosis has little to add to the Student-t it replaces.

## Going deeper

```
sample the shock:      pool = fitted_law.sample(dt = 1/252, 200_000);  z = (pool - mean) / std          (unit variance)
VG / NIG:              X = theta G + sigma W(G),  G a gamma / inverse-Gaussian clock with mean 1 and variance nu / kappa per unit time
SV state:              h_t = mu + phi (h_{t-1} - mu) + eta_t;   y_t = ln(r_t^2) = h_t - 1.2704 + noise(pi^2/2)
forecast variance:     E[sigma^2_{T+k}] = exp( mean_k + var_k / 2 ),   mean_k = mu + phi^k (h_T - mu),  var_k = phi^{2k} P_T + sigma_eta^2 (1 - phi^{2k}) / (1 - phi^2)
Heston moments:        ACF(r^2, k) = A e^{-kappa k dt};   Var(v) = 2 theta^2 A / (1 - 3 A);   xi^2 = 2 kappa Var(v) / theta;
                       Cov(r_t, r^2_{t+k}) = rho xi theta dt^2 e^{-kappa (k-1) dt}
```

## Pitfalls

- **A richer model of the shock is not a better forecast.** With a few thousand daily returns the likelihood cannot tell a variance-gamma from a NIG law, and the extra freedom is fitted to noise as readily as to signal; the walk-forward test is the arbiter, and it said no gain.
- The method of moments for Heston is a rough estimator: the speed and the volatility of variance are weakly determined by daily squared returns (the ACF amplitude is small), and a guard sets the volatility of variance to nearly zero when the amplitude is within noise.
- The two shock-law methods are slower than GARCH (the law is a second maximum-likelihood fit) and are flagged heavy in the tab; the filter's three-parameter optimisation also takes a moment.
- A stochastic-volatility filter without leverage misses the rise in volatility after a fall; Heston has it through its correlation but estimates it from the covariance of returns with later squares, which is noisy.
- None of them forecasts an average return, and none was tested for it.

## Try it

```bash
quant forecast SPY --methods garch garch_vg garch_nig heston sv_kalman --horizon 21 --effort quick
python -m experiments.forecast_study --methods garch fhs garch_vg garch_nig heston sv_kalman --horizons 21
python -m pytest tests/test_forecast_stochvol.py -q
```
