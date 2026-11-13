"""Deep hedging (Buehler, Gonon, Teichmann and Wood, Quantitative Finance 2019): a neural network is trained, on simulated paths, to choose the hedge of a sold option when every trade costs money.

The classical rules each rest on an approximation: the Black-Scholes delta ignores the cost, Leland turns it into extra volatility, Whalley and Wilmott expand for a small cost. Deep hedging makes none.
It states the problem as it is (sell the option, trade at the given dates, pay the cost on every trade, be judged on the profit and loss at expiry by a risk measure) and searches the trading rules by
gradient descent. The rule is a small network that sees, at each date, the moneyness, the time left and the Black-Scholes delta, and answers with a band of holdings; the hedge moves to the band's nearest edge or, inside
it, does not trade. With costs the best trade depends on where you stand, and a band is the form that takes.

The risk measure is the entropic one of exponential utility, ``-(1/a) log E[exp(-a PnL)]`` (the certainty equivalent), with the same risk aversion as the Whalley-Wilmott band, so the two can be compared
on the criterion both aim at. The network starts as a narrow band around the delta, so it begins close to the delta hedge and widens or shifts the band only where that pays.

PyTorch is needed; the caller treats its absence as "this rule is not available".
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

FEATURES = 3
START_BIAS = -3.0                # the band starts about 0.05 shares either side of the delta: close to the delta hedge, and wide enough for the gradient to move it


class HedgePolicy(nn.Module):
    """A no-transaction band network (Imaki, Imajo, Ito, Minami and Nakagawa 2021). From the moneyness, the time left and the Black-Scholes delta the network gives a band ``[delta - below, delta + above]``;
    the holding is the previous one moved to the nearest edge of the band, or left alone inside it. A network that outputs the holding itself has to learn to stand still; this one stands still by
    construction and only has to learn where the edges are, which is the shape the theory says the best rule has."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(FEATURES, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, START_BIAS)

    def band(self, moneyness, time_left, delta_abs):
        """``(lower, upper)`` edges of the holding's size, each between 0 and 1."""
        width = nn.functional.softplus(self.head(self.body(torch.stack([moneyness, time_left, delta_abs], dim=1))))
        return (delta_abs - width[:, 0]).clamp(0.0, 1.0), (delta_abs + width[:, 1]).clamp(0.0, 1.0)

    def forward(self, moneyness, time_left, delta_abs, held_abs):
        low, high = self.band(moneyness, time_left, delta_abs)
        return torch.minimum(torch.maximum(held_abs, low), high)


def _delta_abs(S, K: float, tau: float, r: float, q: float, sigma: float, call: bool):
    """The size of the Black-Scholes delta (the put's is negative; the sign is put back by the caller)."""
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
    return math.exp(-q * tau) * torch.special.ndtr(d1 if call else -d1)


def run_policy(policy: HedgePolicy, paths, K: float, T: float, r: float, q: float, sigma: float, cost: float, call: bool, premium: float):
    """Sell the option for ``premium`` and follow ``policy`` along ``paths`` (``[n, steps + 1]``, a tensor), with the accounting of ``costs.simulate_hedging``: the cost is paid on every trade, the first
    purchase and the final liquidation; cash earns ``r`` and shares pay ``q``. Returns the discounted profit and loss, the discounted cost paid and the number of trades of each path."""
    n, steps = paths.shape[0], paths.shape[1] - 1
    dt, sign = T / steps, (1.0 if call else -1.0)
    held = torch.zeros(n, dtype=paths.dtype)
    cash = torch.full((n,), float(premium), dtype=paths.dtype)
    paid, trades = torch.zeros(n, dtype=paths.dtype), torch.zeros(n, dtype=paths.dtype)
    for i in range(steps + 1):
        Si = paths[:, i]
        if i == steps:
            move = -held
        else:
            tau = T - i * dt
            target = sign * policy(torch.log(Si / K) / (sigma * math.sqrt(T)), torch.full_like(Si, tau / T), _delta_abs(Si, K, tau, r, q, sigma, call), held.abs())
            move = target - held
        fee = cost * Si * move.abs()
        cash = cash - Si * move - fee
        paid, trades, held = paid + fee, trades + (move.abs() > 1e-6).to(paths.dtype), held + move
        if i < steps:
            cash = cash * math.exp(r * dt) + held * Si * (math.exp(q * dt) - 1.0)
    payoff = torch.clamp((paths[:, -1] - K) if call else (K - paths[:, -1]), min=0.0)
    return (cash - payoff) * math.exp(-r * T), paid * math.exp(-r * T), trades


def certainty_equivalent(pnl, risk_aversion: float) -> float:
    """``-(1/a) log mean(exp(-a pnl))``: the sure amount an investor with exponential utility and risk aversion ``a`` (per dollar) values the uncertain profit and loss at. Higher is better."""
    x = -float(risk_aversion) * np.asarray(pnl, dtype=float)
    top = x.max()
    return float(-(top + math.log(np.mean(np.exp(x - top)))) / float(risk_aversion))


def gbm_paths(S: float, growth: float, sigma: float, T: float, steps: int, n: int, generator: torch.Generator, dtype=torch.float32):
    z = torch.randn(n, steps, generator=generator, dtype=dtype)
    dt = T / steps
    logs = torch.cumsum((growth - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z, dim=1)
    return torch.cat([torch.full((n, 1), float(S), dtype=dtype), float(S) * torch.exp(logs)], dim=1)


def train(S: float, K: float, T: float, r: float, q: float, sigma: float, cost: float, steps: int, call: bool, risk_aversion: float, premium: float, iterations: int = 150, batch: int = 2048, hidden: int = 32,
          learning_rate: float = 0.01, seed: int = 0, progress=None) -> tuple[HedgePolicy, list[float]]:
    """Train a policy to maximise the certainty equivalent of selling the option at ``premium`` and hedging it. Every iteration draws fresh paths, so the network cannot memorise a sample. Returns the policy
    and the certainty equivalent of each iteration's batch (in dollars)."""
    if steps < 1 or iterations < 1 or batch < 64 or risk_aversion <= 0 or cost < 0:
        raise ValueError("deep hedging needs at least one date, one iteration, 64 paths a batch, a positive risk aversion and a cost that is not negative")
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed) + 7919)                                       # the training paths are not the paths the rules are compared on
    policy = HedgePolicy(hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations, eta_min=learning_rate * 0.05)
    a, history = float(risk_aversion), []
    for it in range(iterations):
        paths = gbm_paths(S, r - q, sigma, T, steps, batch, gen)
        pnl, _, _ = run_policy(policy, paths, K, T, r, q, sigma, cost, call, premium)
        loss = (torch.logsumexp(-a * pnl, dim=0) - math.log(batch)) / a                         # minus the certainty equivalent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        history.append(-float(loss.detach()))
        if progress is not None and (it + 1) % 25 == 0:
            progress(f"training the hedging network: {it + 1} of {iterations} iterations")
    return policy.eval(), history


@torch.no_grad()
def evaluate(policy: HedgePolicy, paths: np.ndarray, K: float, T: float, r: float, q: float, sigma: float, cost: float, call: bool, premium: float) -> dict:
    """The policy on given paths (``[n, steps + 1]``), in the shape ``costs.simulate_hedging`` returns."""
    pnl, paid, trades = run_policy(policy.double(), torch.as_tensor(np.asarray(paths), dtype=torch.float64), K, T, r, q, sigma, cost, call, premium)
    pnl, n = pnl.numpy(), paths.shape[0]
    return {"strategy": "deep", "premium": float(premium), "mean": float(pnl.mean()), "std": float(pnl.std(ddof=1)), "se": float(pnl.std(ddof=1) / math.sqrt(n)), "cost": float(paid.mean()),
            "trades": float(trades.mean()), "pnl": pnl, "steps": int(paths.shape[1] - 1), "n_paths": int(n)}


@torch.no_grad()
def band_curve(policy: HedgePolicy, spots: np.ndarray, K: float, tau: float, T: float, r: float, q: float, sigma: float, call: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)``: the edges of the network's no-trade band at each price in ``spots`` with ``tau`` left, as signed holdings (a put's are negative, lower edge first)."""
    policy.double()
    s = torch.as_tensor(np.asarray(spots), dtype=torch.float64)
    low, high = policy.band(torch.log(s / K) / (sigma * math.sqrt(T)), torch.full_like(s, tau / T), _delta_abs(s, K, tau, r, q, sigma, call))
    return (low.numpy(), high.numpy()) if call else (-high.numpy(), -low.numpy())
