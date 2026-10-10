"""The Stage 33 forecasters that read the same 252-day normalised window as the Stage 27 networks.

* ``nbeats``    - N-BEATS, generic configuration (Oreshkin et al. 2020): a stack of blocks, each a small fully connected network that
  emits a *backcast* (what it explains of the window, subtracted before the next block sees it) and a *forecast* (summed over blocks).
* ``nhits``     - N-HiTS (Challu et al. 2023): N-BEATS blocks that each read the window average-pooled at a different rate, so a block
  that reads the slow rate sees a short summary and cannot memorise the daily noise. Backcasts are produced at the pooled resolution and
  expanded by repetition (the paper interpolates; repetition is the simpler nearest-neighbour variant).
* ``timemixer`` - a simplified multiscale mixer in the spirit of TimeMixer (Wang et al. 2024): the window at several scales, each through
  its own MLP, mixed by a head. It does NOT implement the decomposable past/future mixing of the paper; the name says "style".

* ``tft``       - a compact Temporal Fusion Transformer (Lim et al. 2021) for ONE observed series plus a static asset identity: gated residual networks (GRN), a variable-selection
  network over three per-step inputs (the normalised return, its absolute value and the running sum), an LSTM encoder, static enrichment, interpretable multi-head attention (one
  value projection shared across heads) with a gate and layer norm, and a position-wise GRN. The window is average-pooled by ``stride`` first so the LSTM runs on 84 steps, not 252.
  Known-future inputs and multi-horizon quantile outputs of the paper are not used: the target is a single z-score.

Every network takes ``(x, asset)`` with ``x`` of shape (batch, window) and returns a z-score forecast of shape (batch,), so it plugs into
``deep_forecast.train_and_predict`` unchanged. The asset enters as a learned embedding concatenated to each block's input.
"""

from __future__ import annotations

FAMILY = ("nbeats", "nhits", "timemixer", "tft", "fnn", "cnn", "lstm", "gru", "transformer")


def build_family_network(kind: str, n_assets: int, window: int = 252, **cfg):
    from .deep_sequence import SEQUENCE, build_sequence_network

    if kind in SEQUENCE:                                                                              # Generation 6: the classical families
        return build_sequence_network(kind, n_assets, window, **cfg)
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

    class GLU(nn.Module):
        def __init__(self, d_in: int, d_out: int):
            super().__init__()
            self.lin = nn.Linear(d_in, 2 * d_out)

        def forward(self, x):
            a, b = self.lin(x).chunk(2, dim=-1)
            return a * torch.sigmoid(b)

    class GRN(nn.Module):
        """Gated residual network: ``LayerNorm(skip(x) + GLU(W2 ELU(W1 x + W3 c)))``; the optional context ``c`` is a static covariate."""

        def __init__(self, d_in: int, d_hidden: int, d_out: int, d_ctx: int = 0):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.ctx = nn.Linear(d_ctx, d_hidden, bias=False) if d_ctx else None
            self.fc2 = nn.Linear(d_hidden, d_hidden)
            self.glu = GLU(d_hidden, d_out)
            self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
            self.norm = nn.LayerNorm(d_out)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, c=None):
            h = self.fc1(x) + (self.ctx(c) if self.ctx is not None and c is not None else 0.0)
            h = self.fc2(nn.functional.elu(h))
            return self.norm(self.skip(x) + self.glu(self.drop(h)))

    class TFT(nn.Module):
        def __init__(self, d: int, heads: int, stride: int):
            super().__init__()
            if window % stride:
                raise ValueError(f"stride {stride} does not divide the window {window}")
            if d % heads:
                raise ValueError("d_model must be divisible by heads")
            self.stride, self.heads, self.d = stride, heads, d
            self.asset = nn.Embedding(n_assets, emb_dim)
            nn.init.normal_(self.asset.weight, std=0.02)
            self.static = GRN(emb_dim, d, d)
            self.var_in = nn.ModuleList(nn.Linear(1, d) for _ in range(3))                          # one embedding per observed input variable
            self.var_weights = GRN(3 * d, d, 3, d_ctx=d)                                               # variable selection weights, conditioned on the static context
            self.var_grn = nn.ModuleList(GRN(d, d, d) for _ in range(3))
            self.lstm = nn.LSTM(d, d, batch_first=True)
            self.post_lstm = GLU(d, d)
            self.norm_lstm = nn.LayerNorm(d)
            self.enrich = GRN(d, d, d, d_ctx=d)
            self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
            self.v = nn.Linear(d, d // heads)                                                           # one value projection shared by the heads (interpretable attention)
            self.out = nn.Linear(d // heads, d)
            self.post_attn = GLU(d, d)
            self.norm_attn = nn.LayerNorm(d)
            self.ff = GRN(d, d, d)
            self.head = nn.Linear(d, 1)
            self.last_selection = None

        def forward(self, x, asset):
            b = x.shape[0]
            xp = x.reshape(b, -1, self.stride).sum(dim=2)                                              # sum of the stride returns: the coarser return series
            feats = torch.stack([xp, xp.abs(), xp.cumsum(dim=1) / (xp.shape[1] ** 0.5)], dim=-1)       # (b, T, 3)
            ctx = self.static(self.asset(asset))
            emb = [lin(feats[..., i:i + 1]) for i, lin in enumerate(self.var_in)]                      # three (b, T, d)
            w = torch.softmax(self.var_weights(torch.cat(emb, dim=-1), ctx.unsqueeze(1)), dim=-1)     # (b, T, 3)
            self.last_selection = w.detach().mean(dim=(0, 1))
            sel = sum(w[..., i:i + 1] * grn(e) for i, (e, grn) in enumerate(zip(emb, self.var_grn)))
            h, _ = self.lstm(sel)
            h = self.norm_lstm(sel + self.post_lstm(h))
            h = self.enrich(h, ctx.unsqueeze(1))
            T = h.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)           # causal attention within the window
            q = self.q(h).reshape(b, T, self.heads, -1).transpose(1, 2)
            k = self.k(h).reshape(b, T, self.heads, -1).transpose(1, 2)
            attn = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            attn = torch.softmax(attn.masked_fill(mask, float("-inf")), dim=-1)
            v = self.v(h).unsqueeze(1)                                                                  # (b, 1, T, d/h) shared across heads
            mixed = self.out((attn @ v).mean(dim=1))
            h = self.norm_attn(h + self.post_attn(mixed))
            h = self.ff(h)
            return self.head(h[:, -1]).squeeze(-1)

    if kind == "tft":
        return TFT(int(cfg.get("d_model", 16)), int(cfg.get("heads", 2)), int(cfg.get("stride", 3)))
    if kind == "nbeats":
        return Residual([1] * int(cfg.get("blocks", 3)), int(cfg.get("layers", 4)), int(cfg.get("width", 64)))
    if kind == "nhits":
        return Residual(list(cfg.get("pools", [12, 3, 1])), int(cfg.get("layers", 3)), int(cfg.get("width", 64)))
    if kind == "timemixer":
        return Multiscale(list(cfg.get("scales", [1, 3, 7])), int(cfg.get("width", 32)))
    raise ValueError(f"unknown family model '{kind}'")
