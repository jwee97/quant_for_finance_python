"""Deep reinforcement learning in torch: DQN and double DQN, DDPG, TD3, SAC and PPO.

All five act in an environment with a vector observation (``env.obs_dim``) and a scalar action in ``[0, env.max_fraction]`` (see :class:`~src.rl.envs.RegimeFractionEnv`), and share one interface:
``agent.train(env, steps, seed)`` then ``agent.act(obs)``. They differ in how they learn the value of an action and how they improve the policy:

    DQN, double DQN   value-based, discrete actions (a grid of fractions): regress ``Q(s, a)`` on ``r + gamma max_a' Q_target(s', a')`` from a replay buffer; double DQN lets the online network pick the
                      action and the target network value it, which removes the upward bias of the maximum (van Hasselt et al. 2016)
    DDPG              deterministic actor ``mu(s)`` trained to climb the critic ``Q(s, a)``, off-policy with a replay buffer, slowly moving target networks and exploration noise (Lillicrap et al. 2016)
    TD3               DDPG with three repairs: two critics and the smaller one as the target (against overestimation), a delayed actor, and noise added to the target action (Fujimoto et al. 2018)
    SAC               a stochastic squashed-Gaussian actor trained to maximise ``Q - alpha log pi``: reward plus entropy, the temperature ``alpha`` tuned to a target entropy (Haarnoja et al. 2018)
    PPO               on-policy: collect a batch, estimate advantages by GAE(lambda), and take several gradient steps on a clipped surrogate that keeps the new policy near the old (Schulman et al. 2017)

The environment's time limit is not a terminal state, so every bootstrap continues through the end of an episode.
"""

from __future__ import annotations

import copy

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:                                                                       # pragma: no cover
    raise ImportError("src.rl.deep needs torch") from exc


def mlp(inp: int, out: int, hidden: int = 64, layers: int = 2) -> nn.Module:
    dims = [inp] + [hidden] * layers
    mods = []
    for i in range(layers):
        mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    return nn.Sequential(*mods, nn.Linear(dims[-1], out))


class Replay:
    def __init__(self, obs_dim: int, capacity: int = 50000):
        self.o, self.a = np.zeros((capacity, obs_dim), np.float32), np.zeros((capacity, 1), np.float32)
        self.r, self.o2 = np.zeros((capacity, 1), np.float32), np.zeros((capacity, obs_dim), np.float32)
        self.n, self.cap, self.i = 0, capacity, 0

    def add(self, o, a, r, o2):
        self.o[self.i], self.a[self.i], self.r[self.i], self.o2[self.i] = o, a, r, o2
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int, rng):
        idx = rng.integers(0, self.n, size=batch)
        return tuple(torch.from_numpy(x[idx]) for x in (self.o, self.a, self.r, self.o2))


def _seed(seed: int):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    return np.random.default_rng(seed)


def _soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1 - tau).add_(tau * s)


class Agent:
    hi: float = 1.0

    def act(self, obs, deterministic: bool = True) -> float:
        raise NotImplementedError

    def policy_table(self, env, n: int | None = None) -> np.ndarray:
        """The action in each of the environment's observations (one per regime)."""
        out = []
        for i in range(env.n_regimes):
            o = np.zeros(env.obs_dim, dtype=np.float32)
            o[i] = 1.0
            out.append(self.act(o, True))
        return np.asarray(out)


# ------------------------------------------------------------------------------------------------------------------ DQN
class DQN(Agent):
    def __init__(self, n_actions: int = 11, double: bool = False, gamma: float = 0.5, lr: float = 2e-3, batch: int = 64, hidden: int = 64, target_every: int = 100, eps: tuple = (1.0, 0.05),
                 warmup: int = 200):
        self.n_actions, self.double, self.gamma, self.lr, self.batch, self.hidden, self.target_every, self.eps, self.warmup = n_actions, double, gamma, lr, batch, hidden, target_every, eps, warmup
        self.grid = None

    def train(self, env, steps: int = 6000, seed: int = 0):
        rng = _seed(seed)
        self.hi = env.max_fraction
        self.grid = np.linspace(0.0, self.hi, self.n_actions).astype(np.float32)
        self.q = mlp(env.obs_dim, self.n_actions, self.hidden)
        self.target = copy.deepcopy(self.q)
        opt = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        buf = Replay(env.obs_dim)
        o = env.reset(seed=seed).astype(np.float32)
        for t in range(steps):
            eps = self.eps[0] + (self.eps[1] - self.eps[0]) * min(1.0, t / (0.7 * steps))
            k = int(rng.integers(self.n_actions)) if rng.random() < eps else int(self.q(torch.from_numpy(o)).argmax())
            o2, r, done = env.step(self.grid[k])
            o2 = o2.astype(np.float32)
            buf.add(o, k, r, o2)
            o = env.reset().astype(np.float32) if done else o2
            if buf.n >= self.warmup:
                bo, ba, br, bo2 = buf.sample(self.batch, rng)
                with torch.no_grad():
                    if self.double:
                        nxt = self.target(bo2).gather(1, self.q(bo2).argmax(1, keepdim=True))
                    else:
                        nxt = self.target(bo2).max(1, keepdim=True).values
                    y = br + self.gamma * nxt
                q = self.q(bo).gather(1, ba.long())
                loss = nn.functional.mse_loss(q, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if t % self.target_every == 0:
                    self.target.load_state_dict(self.q.state_dict())
        return self

    def act(self, obs, deterministic: bool = True) -> float:
        with torch.no_grad():
            return float(self.grid[int(self.q(torch.as_tensor(obs, dtype=torch.float32)).argmax())])

    def q_values(self, obs) -> np.ndarray:
        with torch.no_grad():
            return self.q(torch.as_tensor(obs, dtype=torch.float32)).numpy()


# ------------------------------------------------------------------------------------------------------------------ DDPG and TD3
class DDPG(Agent):
    twin = False

    def __init__(self, gamma: float = 0.5, actor_lr: float = 1e-3, critic_lr: float = 2e-3, tau: float = 0.01, noise: float = 0.4, batch: int = 64, hidden: int = 64, warmup: int = 200,
                 policy_delay: int = 1, target_noise: float = 0.0, noise_clip: float = 0.5):
        self.gamma, self.alr, self.clr, self.tau, self.noise, self.batch, self.hidden, self.warmup = gamma, actor_lr, critic_lr, tau, noise, batch, hidden, warmup
        self.policy_delay, self.target_noise, self.noise_clip = policy_delay, target_noise, noise_clip

    def _actor(self, obs):
        return self.hi * torch.sigmoid(self.actor(obs))

    def _q(self, nets, o, a):
        x = torch.cat([o, a], dim=1)
        return [net(x) for net in nets]

    def train(self, env, steps: int = 5000, seed: int = 0):
        rng = _seed(seed)
        self.hi = env.max_fraction
        d = env.obs_dim
        self.actor = mlp(d, 1, self.hidden)
        self.critics = [mlp(d + 1, 1, self.hidden) for _ in range(2 if self.twin else 1)]
        self.actor_t, self.critics_t = copy.deepcopy(self.actor), [copy.deepcopy(c) for c in self.critics]
        a_opt = torch.optim.Adam(self.actor.parameters(), lr=self.alr)
        c_opt = torch.optim.Adam([p for c in self.critics for p in c.parameters()], lr=self.clr)
        buf = Replay(d)
        o = env.reset(seed=seed).astype(np.float32)
        for t in range(steps):
            if t < self.warmup:
                a = float(rng.uniform(0.0, self.hi))
            else:
                a = float(np.clip(self.act(o, False) + self.noise * rng.normal(), 0.0, self.hi))
            o2, r, done = env.step(a)
            o2 = o2.astype(np.float32)
            buf.add(o, a, r, o2)
            o = env.reset().astype(np.float32) if done else o2
            if buf.n < self.warmup:
                continue
            bo, ba, br, bo2 = buf.sample(self.batch, rng)
            with torch.no_grad():
                a2 = self.hi * torch.sigmoid(self.actor_t(bo2))
                if self.target_noise > 0:
                    a2 = (a2 + (self.target_noise * torch.randn_like(a2)).clamp(-self.noise_clip, self.noise_clip)).clamp(0.0, self.hi)
                q_next = torch.stack(self._q(self.critics_t, bo2, a2)).min(dim=0).values
                y = br + self.gamma * q_next
            loss = sum(nn.functional.mse_loss(q, y) for q in self._q(self.critics, bo, ba))
            c_opt.zero_grad()
            loss.backward()
            c_opt.step()
            if t % self.policy_delay == 0:
                a_loss = -self._q(self.critics[:1], bo, self._actor(bo))[0].mean()
                a_opt.zero_grad()
                a_loss.backward()
                a_opt.step()
                _soft_update(self.actor_t, self.actor, self.tau)
                for ct, c in zip(self.critics_t, self.critics):
                    _soft_update(ct, c, self.tau)
        return self

    def act(self, obs, deterministic: bool = True) -> float:
        with torch.no_grad():
            return float(self._actor(torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)))

    def q_value(self, obs, action: float) -> float:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            return float(self._q(self.critics[:1], o, torch.full((1, 1), float(action)))[0])


class TD3(DDPG):
    """Twin critics, delayed policy updates and target-policy smoothing."""
    twin = True

    def __init__(self, policy_delay: int = 2, target_noise: float = 0.2, **kw):
        super().__init__(policy_delay=policy_delay, target_noise=target_noise, **kw)


# ------------------------------------------------------------------------------------------------------------------ SAC
class SAC(Agent):
    def __init__(self, gamma: float = 0.5, lr: float = 1e-3, tau: float = 0.01, batch: int = 64, hidden: int = 64, warmup: int = 200, target_entropy: float = -1.0, init_alpha: float = 0.2):
        self.gamma, self.lr, self.tau, self.batch, self.hidden, self.warmup, self.target_entropy, self.init_alpha = gamma, lr, tau, batch, hidden, warmup, target_entropy, init_alpha

    def _sample(self, obs):
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        std = log_std.clamp(-5.0, 1.0).exp()
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mean) / std) ** 2 - std.log() - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True) - torch.log(1 - a ** 2 + 1e-6).sum(-1, keepdim=True)
        return self.hi * 0.5 * (a + 1.0), logp - np.log(self.hi * 0.5)

    def train(self, env, steps: int = 5000, seed: int = 0):
        rng = _seed(seed)
        self.hi = env.max_fraction
        d = env.obs_dim
        self.actor = mlp(d, 2, self.hidden)
        self.critics = [mlp(d + 1, 1, self.hidden) for _ in range(2)]
        self.critics_t = [copy.deepcopy(c) for c in self.critics]
        self.log_alpha = torch.tensor(float(np.log(self.init_alpha)), requires_grad=True)
        a_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        c_opt = torch.optim.Adam([p for c in self.critics for p in c.parameters()], lr=2 * self.lr)
        al_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)
        buf = Replay(d)
        o = env.reset(seed=seed).astype(np.float32)
        for t in range(steps):
            if t < self.warmup:
                a = float(rng.uniform(0.0, self.hi))
            else:
                with torch.no_grad():
                    a = float(self._sample(torch.from_numpy(o).reshape(1, -1))[0])
            o2, r, done = env.step(a)
            o2 = o2.astype(np.float32)
            buf.add(o, a, r, o2)
            o = env.reset().astype(np.float32) if done else o2
            if buf.n < self.warmup:
                continue
            bo, ba, br, bo2 = buf.sample(self.batch, rng)
            alpha = self.log_alpha.exp().detach()
            with torch.no_grad():
                a2, logp2 = self._sample(bo2)
                x2 = torch.cat([bo2, a2], dim=1)
                q_next = torch.min(self.critics_t[0](x2), self.critics_t[1](x2)) - alpha * logp2
                y = br + self.gamma * q_next
            x = torch.cat([bo, ba], dim=1)
            c_loss = sum(nn.functional.mse_loss(c(x), y) for c in self.critics)
            c_opt.zero_grad()
            c_loss.backward()
            c_opt.step()
            a_new, logp = self._sample(bo)
            xn = torch.cat([bo, a_new], dim=1)
            a_loss = (alpha * logp - torch.min(self.critics[0](xn), self.critics[1](xn))).mean()
            a_opt.zero_grad()
            a_loss.backward()
            a_opt.step()
            al_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            al_opt.zero_grad()
            al_loss.backward()
            al_opt.step()
            for ct, c in zip(self.critics_t, self.critics):
                _soft_update(ct, c, self.tau)
        return self

    def act(self, obs, deterministic: bool = True) -> float:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            if deterministic:
                mean = self.actor(o).chunk(2, dim=-1)[0]
                return float(self.hi * 0.5 * (torch.tanh(mean) + 1.0))
            return float(self._sample(o)[0])

    @property
    def temperature(self) -> float:
        return float(self.log_alpha.exp().detach())


# ------------------------------------------------------------------------------------------------------------------ PPO
class PPO(Agent):
    def __init__(self, gamma: float = 0.5, lam: float = 0.95, lr: float = 3e-3, clip: float = 0.2, rollout: int = 400, epochs: int = 6, minibatch: int = 100, hidden: int = 64,
                 init_std: float = 0.5, entropy: float = 0.0):
        self.gamma, self.lam, self.lr, self.clip, self.rollout, self.epochs, self.minibatch, self.hidden, self.init_std, self.entropy = gamma, lam, lr, clip, rollout, epochs, minibatch, hidden, init_std, entropy

    def _dist(self, obs):
        mean = self.hi * torch.sigmoid(self.actor(obs))
        return torch.distributions.Normal(mean, self.log_std.exp())

    def train(self, env, steps: int = 20000, seed: int = 0):
        rng = _seed(seed)
        self.hi = env.max_fraction
        d = env.obs_dim
        self.actor, self.critic = mlp(d, 1, self.hidden), mlp(d, 1, self.hidden)
        self.log_std = nn.Parameter(torch.full((1,), float(np.log(self.init_std))))
        opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], lr=self.lr)
        o = env.reset(seed=seed).astype(np.float32)
        done_steps = 0
        while done_steps < steps:
            O, A, R, V, LP, O2 = [], [], [], [], [], []
            for _ in range(self.rollout):
                ot = torch.from_numpy(o).reshape(1, -1)
                with torch.no_grad():
                    dist = self._dist(ot)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                    v = self.critic(ot)
                o2, r, done = env.step(float(a.clamp(0.0, self.hi)))
                O.append(o); A.append(float(a)); R.append(r); V.append(float(v)); LP.append(float(lp)); O2.append(o2.astype(np.float32))
                o = env.reset().astype(np.float32) if done else o2.astype(np.float32)
            done_steps += self.rollout
            O, A, R, LP, O2 = (torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in (O, A, R, LP, O2))
            with torch.no_grad():
                V = self.critic(O).squeeze(-1)
                Vn = self.critic(O2).squeeze(-1)                                                   # bootstrap from the true next observation (episode ends are time limits)
                delta = R + self.gamma * Vn - V
                adv = torch.zeros_like(delta)
                run = 0.0
                for i in reversed(range(len(delta))):
                    run = float(delta[i]) + self.gamma * self.lam * run
                    adv[i] = run
                ret = adv + V
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(self.epochs):
                perm = torch.from_numpy(rng.permutation(len(O)))
                for k in range(0, len(O), self.minibatch):
                    idx = perm[k:k + self.minibatch]
                    dist = self._dist(O[idx])
                    ratio = (dist.log_prob(A[idx].unsqueeze(-1)).squeeze(-1) - LP[idx]).exp()
                    surr = torch.min(ratio * adv[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                    v_loss = nn.functional.mse_loss(self.critic(O[idx]).squeeze(-1), ret[idx])
                    loss = -surr + 0.5 * v_loss - self.entropy * dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        return self

    def act(self, obs, deterministic: bool = True) -> float:
        with torch.no_grad():
            dist = self._dist(torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1))
            return float(dist.mean if deterministic else dist.sample().clamp(0.0, self.hi))


AGENTS = {"dqn": lambda **kw: DQN(**kw), "double_dqn": lambda **kw: DQN(double=True, **kw), "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
