"""Unsupervised representation learning for return windows: an autoencoder and a contrastive (SimCLR-style) encoder.

Labels (next-month returns) are scarce and noisy, but windows of past returns are plentiful. An encoder is first trained WITHOUT labels on every window available at the refit date, then a small
supervised model (ridge) maps the learned embedding to the forecast using only the few matured labels. If the embedding captures regularities (volatility regime, trend, mean-reversion shape) that
a ridge on twelve hand-made features cannot, the second stage should forecast better; if it does not, the representation was not informative about returns.

* ``autoencoder``: ``x -> z (k dims) -> x_hat`` trained to reconstruct the window (a nonlinear PCA when ``k`` is small).
* ``contrastive``: two random augmentations of the same window (random scaling, Gaussian jitter, a masked stretch of days) must map to nearby embeddings and different windows to distant ones,
  with the NT-Xent loss (Chen et al. 2020); the embedding learns what is invariant to those nuisances.

Everything here is deterministic given the seed.
"""

from __future__ import annotations

import numpy as np

KINDS = ("autoencoder", "contrastive")


def _nets(window: int, embed_dim: int, hidden: int):
    from torch import nn

    encoder = nn.Sequential(nn.Linear(window, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
    decoder = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, window))
    projector = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
    return encoder, decoder, projector


def augment(x, generator, scale: float = 0.2, jitter: float = 0.3, mask_fraction: float = 0.1):
    """Random amplitude scaling, additive Gaussian noise, and one masked contiguous stretch (set to zero) per window."""
    import torch

    n, w = x.shape
    out = x * (1.0 + scale * torch.randn(n, 1, generator=generator)) + jitter * torch.randn(n, w, generator=generator)
    length = max(1, int(mask_fraction * w))
    starts = torch.randint(0, w - length + 1, (n,), generator=generator)
    idx = torch.arange(w).unsqueeze(0)
    mask = (idx >= starts.unsqueeze(1)) & (idx < (starts + length).unsqueeze(1))
    return out.masked_fill(mask, 0.0)


def nt_xent(z1, z2, temperature: float = 0.2):
    """Normalised temperature-scaled cross entropy: each embedding's positive is its other view; the other ``2n - 2`` embeddings are negatives."""
    import torch
    import torch.nn.functional as F

    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.T / temperature
    n = z1.shape[0]
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool), float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, target)


def train_encoder(X: np.ndarray, kind: str = "autoencoder", embed_dim: int = 8, hidden: int = 64, epochs: int = 15, batch_size: int = 256, lr: float = 1e-3, seed: int = 0) -> dict:
    """Fit an encoder on the windows ``X`` (n, window) without labels. Returns ``{"encode": fn, "loss": per-epoch loss list}`` where ``encode`` maps windows to embeddings."""
    import torch

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n, window = X.shape
    encoder, decoder, projector = _nets(window, embed_dim, hidden)
    params = list(encoder.parameters()) + (list(decoder.parameters()) if kind == "autoencoder" else list(projector.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    data = torch.as_tensor(X, dtype=torch.float32)
    history = []
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch_size):
            batch = data[order[i:i + batch_size]]
            if kind == "autoencoder":
                loss = torch.mean((decoder(encoder(batch)) - batch) ** 2)
            else:
                if len(batch) < 4:
                    continue
                loss = nt_xent(projector(encoder(augment(batch, gen))), projector(encoder(augment(batch, gen))))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        history.append(total / n)
    encoder.eval()

    def encode(W: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return encoder(torch.as_tensor(W, dtype=torch.float32)).numpy().astype(float)

    return {"encode": encode, "loss": history}
