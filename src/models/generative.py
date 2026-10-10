"""Generative models of return vectors: a Wasserstein GAN with gradient penalty and a variational autoencoder with a factor structure (NeuralFactors-style).

Both learn the joint distribution of a vector of ``dim`` numbers (a day's returns on ``N`` assets, or a short path flattened) from samples and then draw new ones, which is what a scenario generator needs: scenarios
that look like the data in their marginals, their dependence and their tails, without being copies of it. Inputs are standardised internally and ``sample`` returns the original units.

**WGAN-GP** (Arjovsky et al. 2017; Gulrajani et al. 2017). A generator maps noise to a vector; a *critic* (not a classifier: it has no sigmoid) scores vectors, and is trained to maximise the gap between its mean score on
real and on generated data, which estimates the Wasserstein-1 distance; a penalty on the norm of its gradient at points between real and generated samples keeps it 1-Lipschitz. The generator moves to close the gap.
Unlike the original GAN's Jensen-Shannon objective the critic's loss keeps a useful gradient when the two distributions barely overlap, which is why training is steadier.

**Factor VAE** (in the spirit of Gopal 2024, "NeuralFactors"). The returns of ``N`` assets are driven by ``K`` latent factors ``z ~ N(0, I)``: ``r = B z + g(z) + noise``, a linear loading matrix ``B``, a small
nonlinear correction ``g`` (so that factor effects can be asymmetric), and heavy-tailed idiosyncratic noise (Student ``t`` with learned degrees of freedom and a scale per asset). An encoder gives the approximate
posterior of ``z`` from a return vector and the model is fitted by maximising the evidence lower bound. Sampling draws ``z`` and the noise, so the cross-asset dependence passes through ``K`` factors and the tails
come from both the noise and the nonlinearity. It is not the paper's model (which conditions on stock characteristics and uses a normalising-flow-style likelihood); the name says "style".
"""

from __future__ import annotations

import math

import numpy as np


def _standardise(X: np.ndarray):
    mean, std = X.mean(axis=0), X.std(axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    return (X - mean) / std, mean, std


def _mlp(inp: int, out: int, hidden: int, layers: int, act):
    from torch import nn

    parts, size = [], inp
    for _ in range(layers):
        parts += [nn.Linear(size, hidden), act()]
        size = hidden
    parts.append(nn.Linear(size, out))
    return nn.Sequential(*parts)


class WGANGP:
    """Wasserstein GAN with gradient penalty on an ``(n, dim)`` array. ``fit`` runs ``steps`` generator updates with ``n_critic`` critic updates each."""

    def __init__(self, latent: int = 16, hidden: int = 128, layers: int = 3, steps: int = 2500, n_critic: int = 3, batch: int = 128, lr: float = 2e-4, penalty: float = 10.0, seed: int = 0, clip: float = 8.0):
        self.latent, self.hidden, self.layers, self.steps, self.n_critic, self.batch, self.lr, self.penalty, self.seed, self.clip = latent, hidden, layers, steps, n_critic, batch, lr, penalty, seed, clip

    def fit(self, X: np.ndarray) -> "WGANGP":
        import torch
        from torch import nn

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Z, self.mean, self.std = _standardise(np.asarray(X, dtype=float))
        Z = np.clip(Z, -self.clip, self.clip)
        data = torch.as_tensor(Z, dtype=torch.float32)
        dim = Z.shape[1]
        self.G = _mlp(self.latent, dim, self.hidden, self.layers, nn.LeakyReLU)
        self.D = _mlp(dim, 1, self.hidden, self.layers, lambda: nn.LeakyReLU(0.2))
        g_opt = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.critic_gap, self.latent_dim = [], self.latent
        for step in range(self.steps):
            for _ in range(self.n_critic):
                real = data[torch.as_tensor(rng.integers(0, len(data), self.batch))]
                fake = self.G(torch.randn(self.batch, self.latent)).detach()
                eps = torch.rand(self.batch, 1)
                mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
                grad = torch.autograd.grad(self.D(mix).sum(), mix, create_graph=True)[0]
                gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
                gap = self.D(real).mean() - self.D(fake).mean()
                loss = -gap + self.penalty * gp
                d_opt.zero_grad()
                loss.backward()
                d_opt.step()
            g_loss = -self.D(self.G(torch.randn(self.batch, self.latent))).mean()
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()
            if step % 25 == 0:
                self.critic_gap.append(float(gap.detach()))
        self.G.eval()
        return self

    def sample(self, m: int, seed: int = 0) -> np.ndarray:
        import torch

        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            z = self.G(torch.randn(m, self.latent, generator=gen)).numpy().astype(float)
        return z * self.std + self.mean


class FactorVAE:
    """A variational autoencoder whose decoder is a ``K``-factor model with a nonlinear correction and Student-t noise. ``fit`` maximises the ELBO; ``sample`` draws new vectors."""

    def __init__(self, factors: int = 3, hidden: int = 64, epochs: int = 300, batch: int = 128, lr: float = 2e-3, beta: float = 1.0, nu_init: float = 6.0, seed: int = 0, clip: float = 10.0):
        self.K, self.hidden, self.epochs, self.batch, self.lr, self.beta, self.nu_init, self.seed, self.clip = factors, hidden, epochs, batch, lr, beta, nu_init, seed, clip

    def _build(self, dim: int):
        import torch
        from torch import nn

        K = self.K
        self.enc = _mlp(dim, 2 * K, self.hidden, 2, nn.Tanh)
        self.B = nn.Parameter(0.3 * torch.randn(dim, K))
        self.g = _mlp(K, dim, self.hidden // 2, 1, nn.Tanh)
        with torch.no_grad():
            self.g[-1].weight.mul_(0.1)
            self.g[-1].bias.zero_()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.log_nu = nn.Parameter(torch.tensor(math.log(self.nu_init - 2.0)))
        return list(self.enc.parameters()) + [self.B] + list(self.g.parameters()) + [self.log_scale, self.log_nu]

    def _mean(self, z):
        return z @ self.B.T + self.g(z)

    def _log_t(self, x, mean):
        import torch

        nu = 2.0 + self.log_nu.exp()
        scale = self.log_scale.exp()
        r = (x - mean) / scale
        return (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi) - self.log_scale - (nu + 1) / 2 * torch.log1p(r ** 2 / nu)).sum(dim=1)

    def fit(self, X: np.ndarray) -> "FactorVAE":
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Z, self.mean, self.std = _standardise(np.asarray(X, dtype=float))
        Z = np.clip(Z, -self.clip, self.clip)
        data = torch.as_tensor(Z, dtype=torch.float32)
        params = self._build(Z.shape[1])
        opt = torch.optim.Adam(params, lr=self.lr)
        self.losses = []
        for _ in range(self.epochs):
            order = rng.permutation(len(data))
            total = 0.0
            for i in range(0, len(order), self.batch):
                x = data[torch.as_tensor(order[i:i + self.batch])]
                mu, logvar = self.enc(x).chunk(2, dim=1)
                logvar = logvar.clamp(-8.0, 4.0)
                z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
                recon = self._log_t(x, self._mean(z))
                kl = 0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar).sum(dim=1)
                loss = -(recon - self.beta * kl).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(x)
            self.losses.append(total / len(data))
        return self

    @property
    def degrees_of_freedom(self) -> float:
        return float(2.0 + self.log_nu.exp().detach())

    def loadings(self) -> np.ndarray:
        """The linear loading matrix ``B`` in standardised units, shape ``(dim, K)``."""
        return self.B.detach().numpy().copy()

    def sample(self, m: int, seed: int = 0) -> np.ndarray:
        import torch

        gen = torch.Generator().manual_seed(seed)
        nu = float(2.0 + self.log_nu.exp().detach())
        with torch.no_grad():
            z = torch.randn(m, self.K, generator=gen)
            mean = self._mean(z)
            g = np.random.default_rng(seed)
            noise = g.standard_t(nu, size=mean.shape) * self.log_scale.exp().numpy()
            x = mean.numpy() + noise
        return x.astype(float) * self.std + self.mean
