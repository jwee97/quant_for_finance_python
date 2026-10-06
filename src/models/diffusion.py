"""A small denoising diffusion probabilistic model (Ho et al. 2020) for low-dimensional return vectors (Generation 5, Stage 36).

Sampling uses the posterior-mean form of the reverse step with the predicted x_0 clipped to +-``clip`` standardised units (a standard
stabiliser; the plain epsilon form amplifies the network's error ~30x at the last noise step).

Forward process: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, with a cosine noise schedule (Nichol and Dhariwal 2021). The network predicts
eps from (x_t, t); sampling runs the reverse chain with variance beta_t. Pure torch, deterministic on CPU for a given seed.

Also here: the Kupiec (1995) proportion-of-failures test and the pinball loss, the two scores a value-at-risk forecast is judged by.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2


def cosine_schedule(steps: int, s: float = 0.008) -> np.ndarray:
    t = np.arange(steps + 1) / steps
    abar = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = abar / abar[0]
    return np.clip(1.0 - abar[1:] / abar[:-1], 1e-5, 0.999)


def build_denoiser(dim: int, hidden: int = 128, layers: int = 3, embed: int = 32):
    import torch
    from torch import nn

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = embed
            self.t_proj = nn.Sequential(nn.Linear(embed, hidden), nn.SiLU())
            self.inp = nn.Linear(dim, hidden)
            self.body = nn.ModuleList(nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(layers))
            self.out = nn.Linear(hidden, dim)

        def forward(self, x, t):
            half = self.embed // 2
            freq = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / half)
            ang = t.float()[:, None] * freq[None]
            emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
            h = self.inp(x) + self.t_proj(emb)
            for block in self.body:
                h = h + block(h)
            return self.out(h)

    return Denoiser()


class Diffusion:
    """Fit on an (n, dim) array; ``sample(m, seed)`` returns (m, dim) in the original units."""

    def __init__(self, steps: int = 100, hidden: int = 128, layers: int = 3, epochs: int = 600, lr: float = 1e-3, batch: int = 128, seed: int = 11, clip: float = 6.0):
        self.clip = clip
        self.steps, self.hidden, self.layers, self.epochs, self.lr, self.batch, self.seed = steps, hidden, layers, epochs, lr, batch, seed
        self.betas = cosine_schedule(steps)
        self.alphas = 1.0 - self.betas
        self.abar = np.cumprod(self.alphas)

    def fit(self, X: np.ndarray) -> "Diffusion":
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        self.mean, self.std = X.mean(axis=0), X.std(axis=0, ddof=1)
        self.std[self.std == 0] = 1.0
        x0 = torch.as_tensor((X - self.mean) / self.std, dtype=torch.float32)
        self.net = build_denoiser(X.shape[1], self.hidden, self.layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        abar = torch.as_tensor(self.abar, dtype=torch.float32)
        self.losses = []
        for _ in range(self.epochs):
            order = torch.randperm(len(x0), generator=gen)
            running = 0.0
            for i in range(0, len(order), self.batch):
                x = x0[order[i:i + self.batch]]
                t = torch.randint(0, self.steps, (len(x),), generator=gen)
                eps = torch.randn(x.shape, generator=gen)
                a = abar[t][:, None]
                opt.zero_grad()
                loss = torch.mean((self.net(a.sqrt() * x + (1 - a).sqrt() * eps, t) - eps) ** 2)
                loss.backward()
                opt.step()
                running += float(loss.detach()) * len(x)
            self.losses.append(running / len(x0))
        self.net.eval()
        return self

    def sample(self, m: int, seed: int) -> np.ndarray:
        import torch

        gen = torch.Generator().manual_seed(seed)
        dim = len(self.mean)
        x = torch.randn((m, dim), generator=gen)
        abar_prev = np.concatenate([[1.0], self.abar[:-1]])
        with torch.no_grad():
            for t in reversed(range(self.steps)):
                eps_hat = self.net(x, torch.full((m,), t, dtype=torch.long))
                x0_hat = ((x - math.sqrt(1.0 - self.abar[t]) * eps_hat) / math.sqrt(self.abar[t])).clamp(-self.clip, self.clip)
                mean = (math.sqrt(abar_prev[t]) * self.betas[t] / (1.0 - self.abar[t])) * x0_hat \
                    + (math.sqrt(self.alphas[t]) * (1.0 - abar_prev[t]) / (1.0 - self.abar[t])) * x
                if t > 0:
                    var = self.betas[t] * (1.0 - abar_prev[t]) / (1.0 - self.abar[t])
                    x = mean + math.sqrt(var) * torch.randn((m, dim), generator=gen)
                else:
                    x = mean
        return x.numpy().astype(float) * self.std + self.mean


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.where(d >= 0, tau * d, (tau - 1.0) * d)


def kupiec_pof(exceedances: int, n: int, p: float) -> dict:
    """Kupiec's unconditional coverage test: are exceedances of the p-quantile forecast a Binomial(n, p) count? LR ~ chi2(1)."""
    x = int(exceedances)
    if n <= 0:
        return {"lr": float("nan"), "p_value": float("nan")}
    phat = x / n

    def ll(prob: float) -> float:
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return (n - x) * math.log(1 - prob) + x * math.log(prob)

    lr = max(0.0, -2.0 * (ll(p) - ll(phat)))
    return {"lr": lr, "p_value": float(chi2.sf(lr, 1))}
