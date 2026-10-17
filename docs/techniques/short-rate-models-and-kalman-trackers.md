---
title: "Short-rate models and Kalman trackers of reversion and volatility"
slug: short-rate-models-and-kalman-trackers
difficulty: 3
chapter: Ch. 22
prerequisites: [sde-simulation-and-estimation, mean-reversion]
stages: []
files: [src/assets/short_rate.py, src/econometrics/trackers.py, src/probability/ou.py, src/probability/sde.py]
figures: []
tests: [tests/test_short_rate.py, tests/test_trackers.py, tests/test_ou.py]
models: []
---

# Short-rate models and Kalman trackers of reversion and volatility

## In one sentence

A whole yield curve can be a function of one number, today's short rate, if that rate is pulled back to a long-run level at some speed; and the same pull-back idea, with a Kalman filter on top, tracks how fast a spread is reverting *now* and how volatile a return series is *now*.

## The idea

**Vasicek** (`dr = kappa (theta - r) dt + sigma dW`) is Gaussian: the rate can go negative, as rates have. **Cox-Ingersoll-Ross** (`dr = kappa (theta - r) dt + sigma sqrt(r) dW`) keeps the rate positive when `2 kappa theta >= sigma^2`. In both, the price of a zero-coupon bond is `P = A(tau) exp(-B(tau) r)` with `A` and `B` in closed form, the solution of the bond equation `P_tau = kappa (theta - r) P_r + (sigma^2/2) r^{2 gamma} P_rr - r P` (`gamma = 0` for Vasicek, `1/2` for CIR). `B(tau)` is the bond's *duration to the short rate*: the percentage the price falls for a unit rise in `r`. The curve starts at the short rate and ends at a long yield below the long-run level `theta`, because the convexity of the bond price makes long bonds worth more than the average path of the rate suggests.

The parameters that price bonds are the *risk-neutral* ones; the history of rates follows the same equation with the physical long-run level, and the gap between the two is the market price of rate risk. One yield curve identifies the level and the starting rate well and the speed and volatility poorly (a faster pull to a lower level looks much like a slower pull to a higher one), so fit the speed and volatility from history and the level and starting rate from the curve.

A **Kalman filter** tracks a hidden state through noisy data. Two scalar versions are useful here. A **time-varying AR(1) coefficient**: for a demeaned spread `z_t = phi_t z_{t-1} + e_t` with `phi_t` a random walk, the filtered `phi_t` is the reversion speed *as the data through `t` say it is* (`theta_t = -ln(phi_t) / dt`), moving when the pair's behaviour changes, with its own standard deviation. A **stochastic-volatility** model (Harvey, Ruiz and Shephard 1994): with log variance `h_t` an AR(1) and `r_t = sigma_t z_t`, the log square `y_t = ln(r_t^2)` is `h_t` plus noise with mean -1.2704 and variance `pi^2/2` (the log of a chi-square with one degree of freedom); treating that noise as Gaussian makes the Kalman filter the best linear estimate of `h_t`, and the three parameters are fitted by maximising the Gaussian likelihood of the innovations.

## Why it matters

A fixed-window estimate of a half-life or a volatility answers "what was it on average over the window", late. A tracker answers "what is it now, and how sure am I", which is what a position size or an entry rule needs. A bond model gives rate sensitivities and a curve consistent with a stochastic rate where a flat discount rate would not.

## How this repo uses it

`assets/short_rate.py` has `bond_price`, `yield_curve`, `duration_to_short_rate`, `long_yield`, `bond_price_mc` (the same price as an expected discount factor over exact simulated paths, the check on the formula), `fit_curve` and `fit_history` (the exact-likelihood fit to a history of rates through `probability.ou` and `probability.sde`). `econometrics/trackers.py` has `kalman_ar1` (the coefficient tracker) and `sv_tracker` (the volatility tracker, with `forecast_variance` and `simulate_log_variance`). Both trackers are causal: the estimate at `t` uses the data through `t` only. The Forecast tab's `sv_kalman` method forecasts volatility from the stochastic-volatility tracker; the pairs strategy `ou_threshold_pairs` fits the Ornstein-Uhlenbeck process of each spread in a rolling window.

## What we found

- **The bond formulas solve their equation and agree with simulation.** The closed form satisfies the bond equation; it equals the expected discount factor over simulated paths (the exact Ornstein-Uhlenbeck transition for Vasicek, the non-central chi-square one for CIR); the curve starts at the short rate and ends at the long yield; a Gaussian model can price a bond above par at a negative rate, the square-root model never does.
- **A curve fit identifies what it can.** With the speed and volatility known it recovers the level and the starting rate; with all four free it still reproduces the curve, and the standard errors say that the four are not separately identified. A history fit recovers the physical parameters of each model.
- **The AR(1) tracker follows a change in the speed of reversion** (a spread whose speed of reversion changes part way through the sample), its uncertainty shrinks with data, with a fixed coefficient it settles near it, and a random walk shows no reversion. It is causal and copes with gaps.
- **The volatility tracker recovers the process:** the volatility path and the parameters of a simulated stochastic-volatility series, with zero returns and short samples handled; its variance forecast reverts to the unconditional level. As a *forecaster* it did not beat GARCH on the platform's ETFs (see the forecasting guide): a stochastic-volatility filter has no leverage term.

## Going deeper

```
Vasicek:   B = (1 - e^{-kappa tau}) / kappa;   ln A = (theta - sigma^2 / (2 kappa^2)) (B - tau) - sigma^2 B^2 / (4 kappa)
CIR:       h = sqrt(kappa^2 + 2 sigma^2);   B = 2 (e^{h tau} - 1) / ((h + kappa)(e^{h tau} - 1) + 2h)
long yield:   Vasicek  theta - sigma^2 / (2 kappa^2);     CIR  2 kappa theta / (kappa + h)
AR(1) tracker:   state phi_t = phi_{t-1} + w_t   (variance q);   observation  z_t = phi_t z_{t-1} + e_t   (variance r)
SV tracker:      h_t = mu + phi (h_{t-1} - mu) + eta_t;   y_t = ln(r_t^2 + offset) = h_t + (-1.2704) + noise(variance pi^2 / 2)
```

## Pitfalls

- **A short-rate model with one factor moves every maturity together:** it cannot match an arbitrary curve exactly (the one factor is the level), and a fit to the curve says little about the rate's real dynamics.
- Vasicek allows negative rates; CIR does not, and fails the positivity guarantee when the Feller condition is violated.
- The trackers have a noise variance and a state variance to estimate: with little data the tracker's output is mostly its starting values. Read the standard deviation it reports.
- The quasi-maximum-likelihood volatility fit treats a non-Gaussian log chi-square noise as Gaussian, which is consistent but not efficient; zero returns need an offset before the logarithm.
- A tracker that moves with the data also moves with the noise: a rule that trades on `theta_t` needs a threshold, not a point.

## Try it

```bash
python -m pytest tests/test_short_rate.py tests/test_trackers.py -q
python - <<'EOF'
import numpy as np
from src.assets import short_rate as sr
tau = np.array([0.25, 1, 2, 5, 10, 30])
print(sr.yield_curve("vasicek", 0.03, tau, kappa=0.5, theta=0.05, sigma=0.01).round(4))
print(sr.yield_curve("cir", 0.03, tau, kappa=0.5, theta=0.05, sigma=0.06).round(4))
EOF
```
