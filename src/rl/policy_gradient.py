"""Policy-gradient methods with linear function approximation: REINFORCE, advantage actor-critic and the deterministic policy gradient.

**REINFORCE** (Williams 1992). A stochastic policy ``pi_theta(a | s) = softmax(phi(s)' theta)_a`` and the likelihood-ratio gradient ``grad J = E[sum_t grad log pi(a_t | s_t) (G_t - b(s_t))]``: raise the probability
of actions followed by more than the baseline ``b`` (a running mean of the return from each state: it removes variance without bias).

**Advantage actor-critic** (A2C). A critic ``V_w(s) = phi(s)' w`` learned by temporal differences supplies the advantage ``A_t = r_t + gamma V(s_{t+1}) - V(s_t)`` (here an ``n``-step version), so the actor learns
online, without waiting for the episode to end.

**Deterministic policy gradient** (Silver et al. 2014). For continuous actions, a deterministic policy ``mu_theta(s) = phi(s)' theta`` follows ``grad J = E[grad_theta mu(s) grad_a Q(s, a)|_{a = mu(s)}]``. With a
*compatible* critic ``Q_w(s, a) = V_v(s) + (a - mu(s)) phi(s)' w`` the gradient estimate is unbiased: the critic's slope in the action is exactly ``phi(s)' w``. Behaviour is the policy plus Gaussian noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .envs import Env, TabularEnv


def one_hot(n: int) -> Callable:
    def phi(s):
        v = np.zeros(n)
        v[int(s)] = 1.0
        return v
    return phi


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class PGResult:
    theta: np.ndarray
    critic: np.ndarray | None = None
    returns: list = field(default_factory=list)

    def probabilities(self, phi_s: np.ndarray) -> np.ndarray:
        return softmax(phi_s @ self.theta)


def reinforce(env: TabularEnv, episodes: int = 2000, gamma: float = 0.95, lr: float = 0.1, max_steps: int = 50, features: Callable | None = None, baseline: bool = True, seed: int = 0) -> PGResult:
    """REINFORCE with linear-softmax policy ``theta`` of shape ``(n_features, n_actions)`` (tabular by default)."""
    rng = np.random.default_rng(seed)
    phi = features or one_hot(env.n_states)
    d = len(phi(0))
    theta = np.zeros((d, env.n_actions))
    base = np.zeros(env.n_states)
    count = np.zeros(env.n_states)
    returns = []
    for ep in range(episodes):
        s = env.reset(seed=seed if ep == 0 else None)
        traj = []
        for _ in range(max_steps):
            x = phi(s)
            p = softmax(x @ theta)
            a = int(rng.choice(len(p), p=p))
            s2, r, done = env.step(a)
            traj.append((s, x, a, r))
            s = s2
            if done:
                break
        G = 0.0
        grads = np.zeros_like(theta)
        for s_t, x, a, r in reversed(traj):
            G = r + gamma * G
            adv = G - (base[s_t] if baseline else 0.0)
            p = softmax(x @ theta)
            g = -p
            g[a] += 1.0                                                                          # grad of log softmax w.r.t. the logits
            grads += np.outer(x, g) * adv
            count[s_t] += 1
            base[s_t] += (G - base[s_t]) / count[s_t]
        theta += lr * grads / max(len(traj), 1)
        returns.append(sum(r for *_, r in traj))
    return PGResult(theta, None, returns)


def a2c(env: TabularEnv, steps: int = 60000, gamma: float = 0.95, actor_lr: float = 0.05, critic_lr: float = 0.1, n_step: int = 5, features: Callable | None = None, entropy: float = 0.0,
        max_episode: int = 100, seed: int = 0) -> PGResult:
    """``n``-step advantage actor-critic with linear softmax actor and linear critic, updated every ``n_step`` transitions."""
    rng = np.random.default_rng(seed)
    phi = features or one_hot(env.n_states)
    d = len(phi(0))
    theta, w = np.zeros((d, env.n_actions)), np.zeros(d)
    s = env.reset(seed=seed)
    t_ep = 0
    buf = []
    returns, total = [], 0.0
    for step in range(steps):
        x = phi(s)
        p = softmax(x @ theta)
        a = int(rng.choice(len(p), p=p))
        s2, r, done = env.step(a)
        t_ep += 1
        total += r
        truncated = t_ep >= max_episode
        buf.append((x, a, r))
        if done or truncated or len(buf) == n_step:
            boot = 0.0 if done else float(phi(s2) @ w)
            G = boot
            for x_t, a_t, r_t in reversed(buf):
                G = r_t + gamma * G
                adv = G - float(x_t @ w)
                w += critic_lr * adv * x_t
                p_t = softmax(x_t @ theta)
                g = -p_t
                g[a_t] += 1.0
                ent_grad = -p_t * (np.log(p_t + 1e-12) + 1.0 - (p_t * (np.log(p_t + 1e-12) + 1.0)).sum())
                theta += actor_lr * np.outer(x_t, adv * g + entropy * ent_grad)
            buf = []
        if done or truncated:
            returns.append(total)
            total, t_ep = 0.0, 0
            s = env.reset()
        else:
            s = s2
    return PGResult(theta, w, returns)


@dataclass
class DPGResult:
    theta: np.ndarray       # policy weights: mu(s) = phi(s) . theta
    slope: np.ndarray       # compatible critic's action slope weights
    value: np.ndarray
    rewards: list = field(default_factory=list)


def deterministic_policy_gradient(env: Env, steps: int = 40000, gamma: float = 0.9, noise: float = 0.4, actor_lr: float = 0.01, slope_lr: float = 0.05, value_lr: float = 0.05,
                                  init: float = 1.0, seed: int = 0) -> DPGResult:
    """Compatible-critic DPG (COPDAC-Q) for a scalar action in ``[0, env.max_fraction]`` and observations that are feature vectors (the one-hot regime of :class:`RegimeFractionEnv`).
    ``mu(s) = phi(s) . theta`` starts at ``init``; exploration noise is Gaussian with standard deviation ``noise``."""
    rng = np.random.default_rng(seed)
    d = env.obs_dim
    theta, w, v = np.full(d, init), np.zeros(d), np.zeros(d)
    hi = env.max_fraction
    x = env.reset(seed=seed)
    rewards = []
    for _ in range(steps):
        mu = float(x @ theta)
        a = float(np.clip(mu + noise * rng.normal(), 0.0, hi))
        x2, r, _done = env.step(a)
        psi = (a - mu) * x                                                                      # compatible features: grad_theta mu * (a - mu)
        q = float(x @ v + psi @ w)
        q_next = float(x2 @ v)                                                                  # Q(s', mu(s')) = V(s'): the action term vanishes at the policy's own action
        delta = r + gamma * q_next - q
        v += value_lr * delta * x
        w += slope_lr * delta * psi
        theta += actor_lr * x * float(x @ w)                                                    # grad_theta mu * grad_a Q = phi * (phi . w)
        theta = np.clip(theta, 0.0, hi)
        x = x2
        rewards.append(r)
    return DPGResult(theta, w, v, rewards)
