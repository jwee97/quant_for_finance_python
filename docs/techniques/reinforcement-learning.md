---
title: "Reinforcement learning for allocation (and why it comes last)"
slug: reinforcement-learning
difficulty: 3
chapter: Ch. 19
prerequisites: [backtest-engine-and-costs, multiple-testing]
stages: [38]
files: [src/models/rl_allocation.py, experiments/stage38_rl.py]
figures: [75, 76]
tests: [tests/test_rl_allocation.py]
models: []
---

# Reinforcement learning for allocation (and why it comes last)

## In one sentence

Learn an allocation rule by trial and error against a reward; in finance the reward is a noisy backtest, so the main result is usually overfitting.

## The idea

The policy here is a linear softmax over features (zero parameters equals equal weight). The reward is mean-variance utility net of trading costs. Evolution strategies perturb the parameters, rank the
perturbations by reward and move toward the better ones; it needs no gradients through the simulator. It is run with three seeds, annually refitted on matured months only.

## Why it matters

Most reinforcement-learning finance results do not survive transaction costs and honest out-of-sample testing. Showing this once, with a fair setup, is more useful than a promising demo.

## How this repo uses it

`src/models/rl_allocation.py` has the policy and the optimiser; Stage 38 evaluates the average policy through the Generation 1 engine against equal weight and reports the overfitting gap and a random-tilt null.

## What we found

The learned policy's training-period utility gain over equal weight (about +0.0104 per month) turned negative out of sample (about -0.0031); its net Sharpe was 0.21 against 0.75 for equal weight
(paired difference -0.54, significantly worse), with annual turnover of about 13.6 times (declared hypothesis h_rl, rejected).

## Going deeper

```
policy:    w = softmax( X theta + b )                        theta, b = 0  <=>  equal weight
reward:    sum_t [ r_p,t - 0.5 * gamma * r_p,t^2 ] - cost * turnover,   gamma = 5
ES update: theta <- theta + lr / (n sigma) * sum_i  F_i * eps_i      F_i = centred rank of the reward of perturbation eps_i (antithetic pairs)
```
`F_i` uses ranks, not reward values, so one lucky outlier month cannot dominate the update.

## Pitfalls

- A reward that sums over 100 months has very few independent observations.
- Turnover is part of the reward; a policy that ignores it learns to churn.
- Compare with the null of random tilts, not only with the benchmark.

## Try it

```bash
python -m experiments.stage38_rl
```
