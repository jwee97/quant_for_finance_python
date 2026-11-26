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
from ..ml import live


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
        live.start(epochs=self.steps, patience=None, device="cpu", n_params=int(sum(q.numel() for q in list(self.G.parameters()) + list(self.D.parameters()))), n_train=int(len(data)), batch_size=int(self.batch), name="wgan-gp (steps)")
        from ..ml.deep.session import TrainingSession, data_fingerprint

        block = 25                                                                                      # a reporting unit is 25 generator updates: the metrics row, the gradient summary and the checkpoint come at its end
        blocks = -(-self.steps // block)
        sess = TrainingSession("wgan_gp", config={"latent": self.latent, "hidden": self.hidden, "layers": self.layers, "n_critic": self.n_critic, "batch": self.batch, "lr": self.lr, "penalty": self.penalty, "seed": self.seed, "clip": self.clip},
                               data=data_fingerprint(Z), modules={"G": self.G, "D": self.D}, optimizers={"g": g_opt, "d": d_opt}, rngs={"data": rng}, units=blocks, family="gan",
                               capture=lambda: {"critic_gap": list(self.critic_gap)}, apply=lambda s: self.critic_gap.__init__(s["critic_gap"]), steps_per_unit=block * self.n_critic)
        mon_g, mon_d = sess.monitor("generator", self.G), sess.monitor("critic", self.D)
        with sess:
            resumed = sess.restore()
            for blk in range(resumed.start if resumed else 0, blocks):
                sess.begin_unit(blk, block)
                d_k = g_k = 0
                gaps, g_losses, gps = [], [], []
                for step in range(blk * block, min((blk + 1) * block, self.steps)):
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
                        mon_d.observe(d_k)
                        mon_d.before_step(d_k)
                        d_opt.step()
                        mon_d.after_step(d_k)
                        d_k += 1
                    g_loss = -self.D(self.G(torch.randn(self.batch, self.latent))).mean()
                    g_opt.zero_grad()
                    g_loss.backward()
                    mon_g.observe(g_k)
                    mon_g.before_step(g_k)
                    g_opt.step()
                    mon_g.after_step(g_k)
                    g_k += 1
                    gaps.append(float(gap.detach()))
                    g_losses.append(float(g_loss.detach()))
                    gps.append(float(gp.detach()))
                    if step % 25 == 0:
                        self.critic_gap.append(float(gap.detach()))
                        live.epoch(step, float(g_loss.detach()), float(gap.detach()), lr=self.lr, epochs=self.steps)
                if sess.end_unit(blk, {"train_loss": float(np.mean(g_losses)), "val_loss": None, "critic_gap": float(np.mean(gaps)), "gradient_penalty": float(np.mean(gps))}) == "stop":
                    break
            sess.finish()
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
        live.start(epochs=self.epochs, patience=None, device="cpu", n_params=int(sum(q.numel() for q in params)), n_train=int(len(data)), batch_size=int(self.batch))
        from torch import nn
        from ..ml.deep.session import TrainingSession, data_fingerprint

        holder = nn.ParameterDict({"B": self.B, "log_scale": self.log_scale, "log_nu": self.log_nu})   # the factor loadings, the noise scale and the t degrees of freedom are parameters outside any module: kept together to be saved
        kl_hist, rec_hist = [], []
        sess = TrainingSession("factor_vae", config={"factors": self.K, "hidden": self.hidden, "batch": self.batch, "lr": self.lr, "beta": self.beta, "nu_init": self.nu_init, "seed": self.seed, "clip": self.clip}, data=data_fingerprint(Z),
                               modules={"enc": self.enc, "g": self.g, "factor": holder}, optimizers={"adam": opt}, rngs={"order": rng}, units=self.epochs,
                               capture=lambda: {"losses": list(self.losses), "kl": list(kl_hist), "recon": list(rec_hist)},
                               apply=lambda s: (self.losses.__init__(s["losses"]), kl_hist.__init__(s["kl"]), rec_hist.__init__(s["recon"])))
        mon = sess.monitor("adam", {**{f"enc.{k}": p for k, p in self.enc.named_parameters()}, **{f"g.{k}": p for k, p in self.g.named_parameters()}, "factor.B": self.B, "factor.log_scale": self.log_scale, "factor.log_nu": self.log_nu})
        with sess:
            resumed = sess.restore()
            for epoch in range(resumed.start if resumed else 0, self.epochs):
                sess.begin_unit(epoch, -(-len(data) // self.batch))
                order = rng.permutation(len(data))
                total = rec_total = kl_total = 0.0
                for step, i in enumerate(range(0, len(order), self.batch)):
                    x = data[torch.as_tensor(order[i:i + self.batch])]
                    mu, logvar = self.enc(x).chunk(2, dim=1)
                    logvar = logvar.clamp(-8.0, 4.0)
                    z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
                    recon = self._log_t(x, self._mean(z))
                    kl = 0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar).sum(dim=1)
                    loss = -(recon - self.beta * kl).mean()
                    opt.zero_grad()
                    loss.backward()
                    mon.observe(step)
                    mon.before_step(step)
                    opt.step()
                    mon.after_step(step)
                    total += float(loss.detach()) * len(x)
                    rec_total += float(recon.detach().mean()) * len(x)
                    kl_total += float(kl.detach().mean()) * len(x)
                self.losses.append(total / len(data))
                rec_hist.append(rec_total / len(data))
                kl_hist.append(kl_total / len(data))
                live.epoch(epoch, self.losses[-1], None, lr=self.lr, epochs=self.epochs)
                if sess.end_unit(epoch, {"train_loss": self.losses[-1], "val_loss": None, "reconstruction_loglik": rec_hist[-1], "kl": kl_hist[-1], "beta": self.beta, "degrees_of_freedom": float(2.0 + self.log_nu.exp().detach())}) == "stop":
                    break
            sess.finish()
        live.finish(final_train=self.losses[-1] if self.losses else None)
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
