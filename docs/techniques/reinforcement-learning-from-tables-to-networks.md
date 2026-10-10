---
title: "Reinforcement learning from tables to networks: TD control, policy gradients, G-learning and deep agents"
slug: reinforcement-learning-from-tables-to-networks
difficulty: 3
chapter: Platform
prerequisites: [reinforcement-learning, dynamic-programming-and-optimal-control, walk-forward-and-leakage]
stages: []
files: [src/rl/envs.py, src/rl/tabular.py, src/rl/policy_gradient.py, src/rl/glearning.py, src/rl/deep.py, src/rl/exposure.py, src/framework/allocators_rl.py, experiments/control_world.py]
figures: []
tests: [tests/test_rl_numpy.py, tests/test_rl_deep.py, tests/test_allocators_rl.py]
models: []
---

# Reinforcement learning from tables to networks: TD control, policy gradients, G-learning and deep agents

## In one sentence

Reinforcement learning solves the dynamic programming problem when the model is unknown, by learning the value of actions (SARSA, Q-learning), or the policy directly (REINFORCE, actor-critic, deterministic policy gradient), or both with neural networks (DQN, DDPG, TD3, SAC, PPO), and G-learning and its inverse (GIRL) add a cost for departing from a prior policy and recover a reward from observed behaviour.

## The idea

**Value-based control.** Learn `Q(s, a)` by moving it toward a one-step target. SARSA uses the action actually taken next (on-policy, so it learns the value of the exploring policy), expected SARSA averages over the exploring policy (less noise), Q-learning uses the greedy next action (off-policy: it learns the optimal values whatever it does), and double Q-learning keeps two tables so that one picks the action and the other values it, removing the upward bias of the maximum of noisy estimates. DQN does the same with a network and a replay buffer; double DQN transplants the double trick.

**Policy gradients.** REINFORCE follows `grad log pi(a | s) (G - baseline)`, the actor-critic methods replace the return by a one-step or `n`-step advantage from a learned critic, and the *deterministic* policy gradient moves a deterministic policy up the slope of the critic in the action, which is the only practical way with a continuous action such as a portfolio fraction. DDPG is its deep version; TD3 adds two critics (the smaller as the target), delayed policy updates and noise on the target action; SAC maximises reward plus entropy with an automatically tuned temperature; PPO takes several gradient steps on a clipped surrogate over a batch of on-policy data with generalised advantage estimates.

**G-learning and GIRL.** Penalise the relative entropy of the policy from a prior `pi_0`, in units of `1/beta`. The optimum satisfies the soft Bellman equation `G = R + gamma E F(s')`, `F(s) = (1/beta) log sum_a pi_0 exp(beta G)`, `pi = pi_0 exp(beta (G - F))`: `beta -> infinity` is ordinary Q-learning, `beta -> 0` returns the prior. The inverse problem takes observed behaviour as soft-optimal and finds the reward weights (a reward linear in features) that make it likely.

## Why it matters

A portfolio problem with costs, constraints and a state (a regime, a drawdown, the current holdings) is a dynamic programme, and for most realistic versions there is no closed form. Reinforcement learning is the family of methods built for that, and the honest question about each is how much data it needs and whether it finds the answer when the answer is known, before it is trusted where it is not.

## How this repo uses it

`src/rl/envs.py`: tabular environments that carry their exact MDP (`MDPEnv`, `CliffWalking`, `random_mdp_env`) and `RegimeFractionEnv`, a portfolio problem whose optimal fraction in each regime is known by quadrature. `src/rl/tabular.py`: `train` for the four temporal-difference methods. `src/rl/policy_gradient.py`: `reinforce`, `a2c` and `deterministic_policy_gradient` with linear features. `src/rl/glearning.py`: `soft_value_iteration` (the model-based fixed point), `g_learning` (sample-based), `girl` (maximum-likelihood inverse). `src/rl/deep.py` (torch): `DQN` (`double=True` for double DQN), `DDPG`, `TD3`, `SAC`, `PPO` behind one interface. `src/rl/exposure.py` and the allocator `q_learning_exposure`: fitted Q-iteration on the empirical transitions of a book's own history chooses the exposure (0, half or full) from a trend-and-volatility state, learned from months already observed, with the reward of every action computed exactly (holding less does not change the market), so no exploration is needed. The older `es_policy` allocator (evolution strategies) and its guide, [Reinforcement learning for allocation](reinforcement-learning.md), remain.

## What we found

Everything in the first sections is a world with a known answer (`python -m experiments.control_world`, about six minutes with the deep agents).

*Temporal-difference control* on a random four-state, two-action MDP (gamma 0.7): with 25 episodes of up to 100 steps all four methods have the optimal greedy policy but the table is off by 0.4 to 0.5; by 400 episodes the errors against the target are 0.021 (SARSA), 0.010 (expected SARSA), 0.010 (Q-learning) and 0.018 (double Q), where the target of the on-policy methods is the value of the 5%-exploring policy they follow, not the optimal value. On the cliff (five seeds, 1000 episodes, 10% exploration) SARSA averages -15.0 per episode while exploring and learns a greedy policy worth -11; Q-learning averages -29.8 while exploring (it walks the edge and falls) and its greedy policy is the shortest path, -9. That is the textbook difference between learning the value of what you do and of what you would do.

*Policy gradients*: REINFORCE and the actor-critic find an optimal policy on the same MDP, and the compatible-critic deterministic policy gradient (three seeds, 40,000 steps) learns regime fractions within 0.35 of the optimal 1.67 and 0.62.

*G-learning*: as `beta` rises from 0.1 to 50 the soft value rises toward the optimum (the mean gap shrinks from -5.1 to -0.1), the largest action probability rises from 0.37 to 0.95 and the policy converges on the greedy one. The sample version reaches a maximum error of 0.016 in `F` at 300 episodes and 0.009 at 1000; it needs a little uniform exploration, because with none the actions the soft policy rarely picks keep stale values that `F` still counts (the error then stalls near 0.42).

*GIRL*: from 10 demonstrations of 30 steps (three random problems) the correlation between recovered and planted reward weights is 0.999 and the implied policy is within 0.002 nats of the true one, against 0.54 for "no information"; with 300 demonstrations 0.2e-3. That is an easy case (a known discount, temperature and features, a demonstrator that is exactly soft-optimal).

*Deep agents* on the regime portfolio problem (optimum fractions 1.67 and 0.62; three seeds; regret in units of 1e-4 log growth per period): PPO 9.9 (fractions 1.51, 0.72), SAC 22 (1.72, 0.56), TD3 36 (1.67, 0.85), DDPG 58 (1.45, 1.02), double DQN 97 (1.58, 0.42, on a grid of eleven), DQN 405 (2.00, 1.08). The plain DQN overestimates the value of the extreme action (its worst seed chose the maximum fraction in both regimes, a regret of 963), and the double version removes most of that, which is the reason for the second network. The off-policy methods are noisy at these budgets (worst-seed regret two to four times the mean); PPO's advantage here reflects a problem where the action does not change the next state, so on-policy data lose nothing.

*The exposure allocator on the platform's 15 ETFs* (from 2010, equal-weight book as the base, cash at zero): equal weight returns 8.0% with 9.7% volatility, Sharpe 0.82, drawdown -20.4%; `q_learning_exposure` returns 5.8% with 8.2% volatility, Sharpe 0.71, drawdown -18.4%, at a mean exposure of 63% (0.70 and 59% with no trading cost). It cut risk and return together, about in proportion, and did not improve the risk-adjusted return. This is the honest outcome for a method with two bits of state and sixteen years of monthly data.

## Pitfalls

- A learner can only be as good as its world. A reinforcement learning agent trained on a backtest learns the backtest; the walk-forward discipline of the other guides applies in full, and `q_learning_exposure` fits only on months whose following month had ended (it is checked by a test that changes the future).
- The `RegimeFractionEnv` results measure whether the algorithm finds a known optimum; they say nothing about whether markets have exploitable regimes.
- Off-policy value methods overestimate (the maximum of noisy values is biased upward); on-policy ones learn the value of their own exploration. Neither is wrong, they answer different questions.
- Deep agents are sensitive to the seed, the learning rate and the budget; quote the spread over seeds, not the best run. The tests use generous tolerances for that reason.
- GIRL recovers a reward only up to what leaves the policy unchanged, and assumes the demonstrator is (soft-)optimal with a known discount and temperature. Real investors are neither.
- G-learning needs exploration even though its update is off-policy; the value of the prior and of `beta` are modelling choices, not estimates.

## Try it

```bash
python -m experiments.control_world
quant backtest --model characteristic_regression --allocator q_learning_exposure --alloc-param cost_bps=10
python - <<'PY'
from src.rl import envs, deep
env = envs.RegimeFractionEnv()
print([env.optimal_fraction(i) for i in range(2)])
print(deep.PPO().train(env, 16000, seed=0).policy_table(env))
PY
```
