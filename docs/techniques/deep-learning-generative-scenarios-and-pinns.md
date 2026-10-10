---
title: "Deep networks, generative models and physics-informed pricing"
slug: deep-learning-generative-scenarios-and-pinns
difficulty: 3
chapter: Platform
prerequisites: [deep-time-series-models, diffusion-scenarios, option-pricing-and-greeks]
stages: []
files: [src/models/deep_sequence.py, src/models/deep_family.py, src/models/generative.py, src/models/pinn.py, experiments/deep_world.py]
figures: []
tests: [tests/test_generative_pinn.py, tests/test_deep_models.py]
models: [deep_window]
---

# Deep networks, generative models and physics-informed pricing

## In one sentence

Neural networks can forecast from a window of returns (feed-forward, convolutional, recurrent and Transformer families), generate new return vectors that look like the data (a Wasserstein GAN, a factor-structured variational autoencoder), and solve a pricing equation without a grid by minimising its residual (a physics-informed network), and on every test here each did its job roughly and none did it better than a simple baseline where a baseline exists.

## The idea

**Forecasting networks.** `deep_window` feeds the 252 daily returns before a month-end, each divided by the volatility at the origin, to a network that predicts the next month's return in volatility units. The kinds are `fnn` (two hidden layers on the window average-pooled in threes), `cnn` (a temporal convolutional network: dilated residual convolutions whose receptive field grows with depth), `lstm` and `gru` (recurrent networks over block sums of three days), `transformer` (an encoder with sinusoidal positions over blocks of six days), and the earlier `patchtst`, `tsmixer`, `nbeats`, `nhits`, `timemixer` and `tft`. All are refit once a year on origins whose labels had finished, with early stopping on a later block; they differ in what structure they can represent cheaply, not in what they are given.

**A Wasserstein GAN with gradient penalty.** A generator maps noise to a vector of returns; a *critic* (no sigmoid) scores vectors and is trained to widen the gap between its mean score on real and on generated data, which estimates the Wasserstein-1 distance; a penalty on the gradient's norm between real and generated points keeps it 1-Lipschitz. The critic keeps a useful gradient when the two distributions barely overlap, which the original GAN's loss does not, and that is why training is steadier.

**A factor VAE (NeuralFactors-style).** Returns of `N` assets are driven by `K` latent factors: `r = B z + g(z) + noise`, with a linear loading matrix, a small nonlinear correction and heavy-tailed (Student) noise with a scale per asset. An encoder gives the approximate posterior of `z`; the model maximises the evidence lower bound; sampling draws `z` and the noise. This follows the idea of the NeuralFactors model, not the paper's model.

**Physics-informed networks.** The solution of a pricing PDE `u(tau, x)` is a network trained to make the equation's residual small at random points (derivatives by automatic differentiation) and to respect the boundaries. The payoff is imposed at maturity exactly by writing `u = smoothed payoff + tau * network`. The smoothing matters: with the exact kink the second derivative is zero everywhere the computer looks, and the network can leave the payoff alone, which is not a solution.

## Why it matters

These are the tools of the newest research and the easiest to be fooled by: a network that fits is not a network that forecasts, a generator that matches means and correlations may be missing exactly the tails a risk model needs, and a PINN that "solves" a PDE to a fraction of a percent has not made the closed form obsolete. The honest use is to check each against a case with a known answer first.

## How this repo uses it

`src/models/deep_sequence.py` adds the classical families to the network registry of `src/models/deep_family.py` (so `MODELS.create("deep_window", kind="gru")` works, with the same walk-forward refits, embargo and causality tests as the earlier kinds). `src/models/generative.py` has `WGANGP` and `FactorVAE` with `fit` and `sample`; `src/scenarios/generators.py` wraps them (and the diffusion model) behind one interface, see the [scenario guide](scenario-generation-and-backtesting.md). `src/models/pinn.py` has `BSMPINN`, `VasicekPINN` and `HestonPINN` (a Bates model when `lam > 0`: the jump term is an expectation of the network at shifted log-prices, by Gauss–Hermite quadrature), the references `vasicek_bond` and `bates_price` (Fourier inversion of the Heston characteristic function times the jump factor, with the drift compensated), and Adam followed by L-BFGS on one fixed draw of collocation points. Everything is torch, on one CPU thread, deterministic for a seed.

## What we found

`python -m experiments.deep_world` (about nine minutes); everything is one seed and one run.

*Forecasting networks.* In a simulated world of 12 assets and 3000 days with a planted nonlinear effect (next month tilts by tanh of the last month's cumulative return), the mean monthly rank IC is 0.20 (t 6.5) for the GRU, 0.13 (t 4.1) for the CNN, 0.05 for the patch transformer and the feed-forward network, 0.04 for the plain Transformer, 0.01 for N-BEATS and -0.02 for the LSTM. The spread between the GRU and the LSTM, two recurrent networks, is mostly the luck of one training run: do not read it as a ranking. On the platform's 15 ETFs (188 monthly origins, from 2010) the ICs are 0.109 (t 3.4) for the Transformer, 0.063 (t 2.1) for the feed-forward network, 0.058 (t 1.8) for the CNN and 0.031 (t 0.9) for the LSTM, against 0.019 (t 0.6) for a ridge regression on price characteristics over its own, longer sample. One architecture of four reaching t 3.4 is the kind of result that a search for it produces by itself, and the repository's earlier stages found deep networks no better than ridge; treat it as a lead, not a finding.

*Generative models* (details and the whole comparison in the [scenario guide](scenario-generation-and-backtesting.md)). On six simulated assets driven by two factors with fat-tailed noise, the factor VAE reproduces the correlations to 0.05 and the volatilities to 4%, and the WGAN-GP, given 2500 critic-and-generator updates and 128 units, to 0.11 and 8%; with 800 updates the GAN is far off (correlation error 0.5). Both under-reproduce the tails: the kurtosis of the samples is about 0.5 to 0.7 of the data's. On five ETFs the learned generators match marginals worse than a bootstrap (Kolmogorov–Smirnov distances 0.03 to 0.04 against 0.01) and, drawing days independently, have none of the data's volatility clustering.

*Physics-informed pricing* (strike 100; maximum error over five spots; seconds to train): Black–Scholes–Merton call 0.42, 0.42 and 0.31 at maturities 0.25, 0.5 and 1 (19 s); Vasicek bond, per unit of face value, 0.0015, 0.0012 and 0.0009 at maturities 1, 3 and 5 (12 s); Heston call 1.04, 0.74 and 0.68 at maturities 0.25, 0.5 and 1 (36 s); Bates call 0.88, 0.68 and 0.48 (53 s), against Fourier prices. That is 0.3% to 1% of the strike, and for the Heston and Bates calls about 10% of an at-the-money price, worst at the shortest maturity where the payoff's kink is least smoothed. A Black–Scholes price from the formula costs microseconds and is exact.

## Pitfalls

- A network's out-of-sample IC from one seed is a draw. Average over seeds (`seeds=(0, 1, 2)` on `deep_window`) and over architectures counted as trials before believing any one.
- A generator that matches the marginals and the correlation matrix can still have the wrong tails, the wrong clustering and the wrong behaviour over a horizon; look at all the numbers of `quality_report`.
- Learned generators are fitted on the data they are given; pass only what was known at the time if the scenarios feed a backtest.
- PINN accuracy is limited by the payoff smoothing (sharpness 30 means about 2% of the strike at the kink at maturity), by the boundary treatment, and by the training budget; error is largest near maturity and at the edges. Fix the PDE and its boundary conditions before trusting the network, and compare with a finite-difference or Fourier price whenever one exists.
- The Heston PINN's variance boundary at zero is handled by sampling more points near it, not by a proven condition; far from the middle of its domain it should not be used.

## Try it

```bash
python -m experiments.deep_world
quant backtest --model deep_window --param kind=gru --param max_epochs=8
python - <<'PY'
from src.models.pinn import BSMPINN
net = BSMPINN(0.03, 0.01, 0.25, 1.0).fit()
print(net.price([90, 100, 110], 100, 1.0))
PY
```
