---
title: "Dynamic programming and optimal control: MDPs, Merton, HJB, Pontryagin and Schrödinger bridges"
slug: dynamic-programming-and-optimal-control
difficulty: 3
chapter: Platform
prerequisites: [multi-period-trading-and-order-routing, drawdown-ruin-and-kelly, option-pricing-and-greeks]
stages: []
files: [src/control/mdp.py, src/control/merton.py, src/control/hjb.py, src/control/pontryagin.py, src/control/schrodinger.py, experiments/control_world.py]
figures: []
tests: [tests/test_control_mdp.py, tests/test_control_continuous.py]
models: []
---

# Dynamic programming and optimal control: MDPs, Merton, HJB, Pontryagin and Schrödinger bridges

## In one sentence

A sequence of decisions under uncertainty can be solved by working backwards from the end (Bellman's principle: whatever happened so far, the rest must be optimal), in discrete time as a Markov decision process, in continuous time as a partial differential equation (Hamilton–Jacobi–Bellman) or along one path as a boundary value problem (Pontryagin), and the same machinery covers hidden states (POMDPs), risk-averse agents (entropic utility), trading costs (a no-trade band) and the cheapest way to steer a distribution to a target (Schrödinger bridges).

## The idea

**Markov decision processes.** States, actions, transition probabilities `P(s' | s, a)`, rewards `R(s, a)` and a discount `gamma`. The optimal value solves `V(s) = max_a [R + gamma sum P V(s')]`. Value iteration applies the right side repeatedly (a contraction, so it converges from anywhere), policy iteration alternates exact evaluation with greedy improvement, and backward induction does a finite horizon. *Risk-sensitive* control replaces the expectation by an entropic one, `(1/theta) log E exp(theta (r + gamma V'))`: negative `theta` penalises variance and bad outcomes (to second order, mean plus `theta/2` times the variance), positive `theta` likes risk, and `theta -> 0` is the ordinary problem. A *partially observed* MDP gives the agent observations instead of the state; it holds a belief updated by Bayes' rule, the problem is an MDP on beliefs, and the value is a convex, piecewise linear maximum of "alpha" vectors, which point-based value iteration improves at a grid of beliefs.

**Merton's problem.** Split wealth between a riskless asset and a risky one with power utility of relative risk aversion `gamma`: the optimal risky fraction is the constant `(mu - r) / (gamma sigma^2)` at every wealth and every horizon. With a proportional trading cost the answer changes character: the investor does nothing while the fraction is inside a *no-trade region* around the Merton fraction and trades to its edge when it drifts out. For small costs `kappa` its half-width is about `(3 pi*^2 (1 - pi*)^2 kappa / (2 gamma))^(1/3)`, growing with the cube root of the cost.

**Hamilton–Jacobi–Bellman and viscosity solutions.** In continuous time the value function solves `V_t + max_u [drift . V_x + (1/2) sigma^2 V_xx + reward] = 0`. It is often not smooth (an American option's value has a kink at the exercise boundary, a bound on the control makes the equation degenerate), so the classical equation has no solution there. The *viscosity solution* is the one a numerical scheme converges to if it is **monotone** (every neighbour enters with a non-negative weight), stable and consistent. An upwind difference for the drift and a central one for the diffusion, an implicit time step and Howard's policy iteration at every step give such a scheme.

**Pontryagin.** Along one optimal path the Hamiltonian `H = L + lambda . f` is minimised by the control, the costate follows `lambda' = -dH/dx`, and `lambda(T) = dphi/dx(x(T))`: a two-point boundary value problem. It is cheaper than HJB (one path, not the whole value function) and gives only a necessary condition. For a linear-quadratic problem it agrees with the Riccati equation; for liquidation with a risk penalty it gives the Almgren–Chriss `sinh` schedule.

**Schrödinger bridges and linearly solvable control.** The process closest in relative entropy to a reference chain that starts at `p0` and ends at `pT` is the reference reweighted by two functions of the endpoints, found by Sinkhorn's alternating rescaling. With state cost `q` and a control cost equal to the KL divergence from passive dynamics, the Bellman equation becomes *linear* in the desirability `z = exp(-v)`.

## Why it matters

Almost every decision in this repository's books is a sequence: how fast to trade a basket (a control problem), how much to hold when trading is costly (a band), when to exercise (a stopping problem), how to hold a book that must be rebalanced later. These modules are the reference solutions to those problems in the cases that have exact or near-exact answers, so that the approximate methods built elsewhere (the multi-period allocator, the schedule optimisers, the reinforcement learners) can be checked against something that is known to be right.

## How this repo uses it

`src/control/mdp.py`: `MDP`, `value_iteration`, `policy_iteration`, `backward_induction`, `risk_sensitive_value_iteration`, and `POMDP` with `belief_update`, `expectimax` (exact tree search, for small cases) and `pbvi` (point-based value iteration). `src/control/merton.py`: the closed forms and `CostDP`, a discrete-time dynamic programme over the pre-trade risky fraction that exploits the homotheticity of power utility, so the fraction is the only state. `src/control/hjb.py`: `merton_hjb` (log-wealth grid, upwind drift, policy iteration, the homothetic boundary condition `V_x = (1 - gamma) V`) and `american_put_hjb`, the obstacle problem `max(V_tau - L V, payoff - V) = 0` solved with policy iteration on the stop-or-continue decision. `src/control/pontryagin.py`: `solve_pmp` (a general boundary value solver on `scipy.integrate.solve_bvp`), `lqr_pmp` against `lqr_riccati`, `liquidation_pmp` against `almgren_chriss_closed_form`, and `ramsey_pmp`. `src/control/schrodinger.py`: `sinkhorn`, `markov_bridge`, `path_kl`, `solve_lmdp_first_exit` and `kl_control_cost`. Everything is exact mathematics on small problems; none of it forecasts anything.

## What we found

`python -m experiments.control_world` (the first half takes seconds). Tiger problem (two doors, a tiger behind one, listening is cheap and 85% accurate): the exact tree search gives -1.00, -1.95, 2.31, 1.80, 2.76 at one to five steps; point-based value iteration reaches 19.37 at 200 sweeps (the published value, to two decimals) and chooses to listen at 50/50, and to open the right door when 97% sure the tiger is on the left. At 10 sweeps it is still at -5.3, because it starts from a pessimistic bound and discounting at 0.95 forgets that slowly. A gamble worth 1.1 against a sure 1.0 is taken by the risk-neutral and risk-seeking agents and refused by the risk-averse (`theta = -0.5` and `-2` both take the sure thing).

Merton's HJB on a log-wealth grid: the value function is within 0.5%, 0.27% and 0.14% of the closed form on grids of 101x25, 201x50 and 401x100 for `gamma = 3` (error halving with the grid), and the risky fraction is 0.475, 0.50, 0.50 against 0.50; with `gamma = 0.5` the unconstrained answer, 3, is capped at the bound and the scheme reports 2.95, 2.975, 3.0. The cost dynamic programme finds no band at zero cost and, with `gamma = 3`, `pi* = 0.5`, a half-width of 0.015, 0.030, 0.055 and 0.095 at costs of 5, 20, 80 and 320 basis points (the small-cost formula says 0.025, 0.040, 0.063 and 0.100): the right order and the right shape (the band widens with the cost, and the formula overstates it at the smallest costs and matches it at the largest). A 64-fold rise in cost widens the band 6.3 times where the cube-root law says 4: the smallest bands are only a few grid steps (0.005) wide, so the ratio is coarse. The American put (spot = strike = 100, 5% rate, 30% volatility, one year) comes out at 9.946, 9.915, 9.894 and 9.883 on grids doubling from 100, against 9.870 for a 3000-step binomial tree, the error roughly halving with each doubling: first-order convergence, with an exercise boundary near 68 to 69. Pontryagin agrees with the Riccati solution to 2e-11, with the `sinh` liquidation schedule to 1e-9 shares, and the Ramsey model's capital ends at its steady state with consumption within 0.012 of it.

A Schrödinger bridge between random start and end laws over a sticky chain costs 0.19 nats of path entropy against 0.39 for the independent coupling (five random chains, six steps), needs 16 Sinkhorn sweeps on average, and reproduces the targets exactly. Linearly solvable control matches the Bellman equation `v = q - log E exp(-v')` to rounding.

## Pitfalls

- A closed form is a *test*, not a result about markets. Merton's fraction assumes constant drift and volatility known exactly; fed estimated inputs it inherits the estimation error of the drift, which is large (the repository's other guides on shrinkage and Bayesian portfolios are about exactly this).
- The no-trade band depends on the cost, risk aversion and volatility assumed. The small-cost formula is an asymptotic result and overstates the width at the smallest costs here; use the dynamic programme when the numbers matter, and a finer grid when the band is only a few steps wide.
- A scheme that is not monotone can converge to something that is not the viscosity solution or oscillate near a kink. The upwind choice is the point of the schemes here, and a bound on the control is part of the problem: with `gamma = 0.5` the "answer" is the bound.
- PBVI is a lower bound that improves with the number of sweeps and beliefs; judge it by comparing sweeps (10 is useless, 60 is close, 200 converged here), not by a single run. Tree search is exact but exponential in the horizon.
- Pontryagin is necessary, not sufficient, and `solve_bvp` needs a decent starting guess for nonlinear problems; the Ramsey example pins the end state at the steady state, which is the turnpike assumption.
- Sinkhorn on a kernel with tiny entries underflows; the bridge is only as informative as the reference process.

## Try it

```bash
python -m experiments.control_world
python - <<'PY'
from src.control.merton import CostDP
sol = CostDP(0.08, 0.02, 0.2, 3.0, kappa=0.002, periods=240, grid=301).solve()
print(sol.lower, sol.merton, sol.upper)
PY
```
