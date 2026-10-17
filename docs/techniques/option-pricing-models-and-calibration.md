---
title: "Option pricing models, Fourier pricing and calibration"
slug: option-pricing-models-and-calibration
difficulty: 3
chapter: Ch. 18
prerequisites: [option-pricing-and-greeks, option-chains-and-greeks]
stages: []
files: [src/derivatives/levy.py, src/derivatives/fourier.py, src/derivatives/calibration.py, src/assets/option_models.py, src/webapp/models_api.py, src/webapp/static/options.js]
figures: []
tests: [tests/test_levy.py, tests/test_fourier.py, tests/test_calibration.py, tests/test_option_models.py, tests/test_models_api.py, tests/test_webapp_models_browser.py]
models: []
---

# Option pricing models, Fourier pricing and calibration

## In one sentence

Black-Scholes gives every strike the same volatility, a real chain does not, and a model that has a *characteristic function* (the Fourier transform of the return's law) can be fitted to the whole smile at once and then say what the quotes imply about products nobody quotes.

## The idea

A **smile** is the market telling you that returns are not lognormal: crashes are likelier and rallies milder than Black-Scholes allows. The cure is a richer law for the return, and a family of them has one property that makes them cheap to use: the **characteristic function** `phi(u) = E[exp(i u X)]` of the log return `X = ln(S_T / S_0)` is known in closed form even when the density is not.

*Exponential Lévy models* keep the independent, stationary increments of Brownian motion and replace the normal law: **Merton** adds rare normal-sized jumps, **Kou** gives the jumps a double-exponential size, the **variance gamma** and the **normal inverse Gaussian (NIG)** run Brownian motion on a random clock (a gamma or an inverse-Gaussian one) so that there are infinitely many small jumps and a free skew and kurtosis. The **Heston** model makes the variance itself random and correlated with the price, which is not a Lévy process but has a characteristic function too. In every case the price is made a martingale by a convexity correction.

Given `phi`, the price of a European option is an integral. **Lewis (2001)** writes the call as one integral over `Re[e^{iuk} phi(u - i/2)] / (u^2 + 1/4)`; the integrand is even and analytic in a strip, so the plain trapezoid rule converges geometrically (error of order `exp(-pi/h)` for step `h`: a step of 0.1 gives 1e-13) and Simpson's rule, which weighs the points unevenly, loses exactly that. **Carr and Madan (1999)** damp the call so that it has a transform and invert it for a whole grid of strikes by FFT. Inverting `phi` itself gives the **risk-neutral density** of the price at expiry.

**Calibration** is the inverse problem: choose the parameters, shared by all expiries, that reprice the quotes. The residual of a quote is its price error divided by its Black-Scholes vega, which equals the implied-volatility error to first order, so every strike counts evenly in volatility terms (a plain price fit is dominated by the at-the-money options and ignores the wings). The honest summary of a fit is not the parameters but the error in volatility points, the standard errors of the parameters, the **conditioning** of the problem (large: some combination of parameters is not determined by the quotes), which parameters ended at the **edge of their allowed range**, and the benchmark every model must beat: **one flat volatility per expiry**.

## Why it matters

A vanilla quote tells you the price of that vanilla. A *model fitted to the vanillas* tells you something no quote does: what a digital, a barrier, an average-price or an American option on the same underlying should cost, and what the market is charging for the tail. Two models can reproduce the smile equally well and still disagree on the exotic; that disagreement beyond what their fit errors explain, the **model risk**, is the reason to fit more than one.

## How this repo uses it

`src/derivatives/levy.py` has the six models with exact samplers, densities and maximum-likelihood fits to returns; `fourier.py` prices by the Lewis integral and by FFT and inverts the density; `calibration.py` selects the liquid out-of-the-money quotes of a chain and fits a model by bounded least squares (two starting points, standard errors from the Jacobian, a flat-smile benchmark, `compare_models` to rank several).

The **Options tab** has three views built on them: **Model fit** (the five models fitted to the chain: a ranking in volatility points, the smile each draws beside the market's, the density each implies and its odds of a fall), **Model prices** and **Hedging cost** (see the guides on exotic and American options and on hedging costs). `src/assets/option_models.py` holds the reports; the page (`src/webapp/models_api.py`) and the terminal command `quant models fit SPY` call the same functions. Fits are kept for the session, keyed by the chain's content and the settings, so pricing a product after a fit does not repeat it.

## What we found

- **The machinery is exact where it can be checked.** The Lewis price matches the Black-Scholes closed form, the Merton series, the Heston price by Gil-Pelaez quadrature, the payoff integrated against the closed-form densities of the Lévy models, and exact Monte Carlo; the Carr-Madan FFT matches Lewis (to about 1e-8, set by the spline); the trapezoid error does fall geometrically with the step. A fit recovers the parameters that generated noise-free quotes, fits noisy quotes to the noise level with standard errors that describe the miss, ranks the wrong model below the right one and flags a strained fit.
- **It found a bug in an older function.** The Fourier price of a jump diffusion disagreed with the closed-form Merton series in `derivatives/pricing.py`, which counted the jump compensation twice and so over-priced by about 4% at two jumps a year. The series was fixed; the tests now pin the two to each other.
- **On the platform's simulated SPY chain (September 2026 prices) the stochastic-volatility model fits far better than the jump models.** Heston was 0.35 volatility points from the quotes (99% of the smile's variance explained against the flat benchmark of 3.79 points); Kou and Merton were about 1.5 points, NIG 2.2 and variance gamma 2.6. Kou's jump-probability and up-jump-rate parameters ended at the edge of their ranges with a conditioning number near 1e8: the fit was straining, and the page says so. The risk-neutral probability of a 10% fall in 32 days was 1.9% under Heston against 0.1% under Black-Scholes. These numbers describe a *simulated* surface (an SSVI model), so they show which family resembles that surface and nothing about what a market believes; `quant models fit SPY` reproduces them, and a downloaded real chain gives different ones.

## Going deeper

```
S_t = S_0 exp((r - q - w) t + L_t),   w = ln E[exp(L_1)]                  (the convexity correction)
E[exp(i u X_T)] = exp(i u (r - q - w) T + T psi(u))                       psi: the characteristic exponent
Lewis:   C = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2} / pi * Int_0^inf Re[e^{iuk} phi_c(u - i/2)] / (u^2 + 1/4) du,   k = ln(F/K)
residual of quote i:   (price_model_i - price_market_i) / vega_i   ~   iv_model_i - iv_market_i
parameter standard errors:   sqrt(diag(s^2 (J'J)^{-1})),   s^2 = residual sum of squares / (n - p)
share of the smile explained:   1 - (rmse_model / rmse_flat)^2
```
The VG and NIG laws are Brownian motion `theta t + sigma W` evaluated on a gamma or inverse-Gaussian clock with mean 1 and variance `nu` or `kappa` per unit time; Heston's characteristic function is written in the Albrecher et al. (2007) "little trap" form, which has no branch cuts, so that it does not jump branches at long maturities.

## Pitfalls

- **A fit is a description of the quotes, not a forecast.** It lives under the risk-neutral measure and includes the market's price of crash risk.
- **A fit to the platform's simulated chain is a fit of one model to another.** Download a real chain (the Free data card) for anything about a market.
- **A parameter at the edge of its range is not a measurement,** and a standard error from a singular fit is meaningless; read the flags before the parameters.
- **A good in-sample fit says little about the exotics.** Models fitted to the same quotes differ on a vanilla by how well each fits and, on some exotics, by more: the next two views set the exotic's spread beside the vanilla's so that the two can be told apart.
- Very short expiries are dominated by single jumps and rounding to the tick; the fit leaves out expiries under a week by default.
- Infinite-activity models (variance gamma, NIG) have a cusp in the density over a very short horizon that a truncated Fourier inversion cannot resolve; the report drops a density whose mass is not within 2% of one.

## Try it

```bash
quant models list                         # the models, their parameters, the products
quant models fit SPY                      # about half a minute for all five; --models merton heston for two
quant models fit --chain my_chain.csv --json fit.json --notes
python -m pytest tests/test_levy.py tests/test_fourier.py tests/test_calibration.py tests/test_option_models.py -q
```
In the dashboard: **Options → Model fit → Fit the models**; tick a real chain in the *Real option chains (free)* card first to calibrate to quotes.
