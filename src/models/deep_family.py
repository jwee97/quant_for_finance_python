"""The Stage 33 forecasters that read the same 252-day normalised window as the Stage 27 networks.

* ``nbeats``    - N-BEATS, generic configuration (Oreshkin et al. 2020): a stack of blocks, each a small fully connected network that
  emits a *backcast* (what it explains of the window, subtracted before the next block sees it) and a *forecast* (summed over blocks).
* ``nhits``     - N-HiTS (Challu et al. 2023): N-BEATS blocks that each read the window average-pooled at a different rate, so a block
  that reads the slow rate sees a short summary and cannot memorise the daily noise. Backcasts are produced at the pooled resolution and
  expanded by repetition (the paper interpolates; repetition is the simpler nearest-neighbour variant).
* ``timemixer`` - a simplified multiscale mixer in the spirit of TimeMixer (Wang et al. 2024): the window at several scales, each through
  its own MLP, mixed by a head. It does NOT implement the decomposable past/future mixing of the paper; the name says "style".

Every network takes ``(x, asset)`` with ``x`` of shape (batch, window) and returns a z-score forecast of shape (batch,), so it plugs into
``deep_forecast.train_and_predict`` unchanged. The asset enters as a learned embedding concatenated to each block's input.
"""

from __future__ import annotations

FAMILY = ("nbeats", "nhits", "timemixer")


def build_family_network(kind: str, n_assets: int, window: int = 252, **cfg):
    import torch
    from torch import nn

    emb_dim = int(cfg.get("asset_embedding", 8))
    dropout = float(cfg.get("dropout", 0.2))

    def mlp(n_in: int, width: int, layers: int, n_out: int) -> nn.Sequential:
        parts, size = [], n_in
        for _ in range(layers):
            parts += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
            size = width
        parts.append(nn.Linear(size, n_out))
        return nn.Sequential(*parts)

    class Residual(nn.Module):
        """N-BEATS / N-HiTS: blocks with backcast residuals; ``pools`` = (1, 1, 1) is plain N-BEATS."""

        def __init__(self, pools: list[int], layers: int, width: int):
            super().__init__()
            for p in pools:
                if window % p:
                    raise ValueError(f"pool {p} does not divide the window {window}")
            self.pools = list(pools)
            self.asset = nn.Embedding(n_assets, emb_dim)
            nn.init.normal_(self.asset.weight, std=0.02)
            self.blocks = nn.ModuleList(mlp(window // p + emb_dim, width, layers, window // p + 1) for p in self.pools)

        def forward(self, x, asset):
            emb = self.asset(asset)
            residual, total = x, 0.0
            for pool, block in zip(self.pools, self.blocks):
                seen = residual if pool == 1 else residual.reshape(residual.shape[0], -1, pool).mean(dim=2)
                theta = block(torch.cat([seen, emb], dim=1))
                backcast, forecast = theta[:, :-1], theta[:, -1]
                residual = residual - (backcast if pool == 1 else backcast.repeat_interleave(pool, dim=1))
                total = total + forecast
            return total

    class Multiscale(nn.Module):
        def __init__(self, scales: list[int], width: int):
            super().__init__()
            for s in scales:
                if window % s:
                    raise ValueError(f"scale {s} does not divide the window {window}")
            self.scales = list(scales)
            self.asset = nn.Embedding(n_assets, emb_dim)
            nn.init.normal_(self.asset.weight, std=0.02)
            self.branches = nn.ModuleList(nn.Sequential(nn.Linear(window // s, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width), nn.GELU())
                                          for s in self.scales)
            self.head = nn.Sequential(nn.Linear(width * len(self.scales) + emb_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))

        def forward(self, x, asset):
            parts = [b(x if s == 1 else x.reshape(x.shape[0], -1, s).mean(dim=2)) for s, b in zip(self.scales, self.branches)]
            return self.head(torch.cat(parts + [self.asset(asset)], dim=1)).squeeze(-1)

    if kind == "nbeats":
        return Residual([1] * int(cfg.get("blocks", 3)), int(cfg.get("layers", 4)), int(cfg.get("width", 64)))
    if kind == "nhits":
        return Residual(list(cfg.get("pools", [12, 3, 1])), int(cfg.get("layers", 3)), int(cfg.get("width", 64)))
    if kind == "timemixer":
        return Multiscale(list(cfg.get("scales", [1, 3, 7])), int(cfg.get("width", 32)))
    raise ValueError(f"unknown family model '{kind}'")
