"""The classical network families on the 252-day normalised window: feed-forward, convolutional, recurrent (LSTM, GRU) and a plain Transformer encoder.

They take ``(x, asset)`` with ``x`` of shape (batch, window) and return a z-score forecast of shape (batch,), as every network in :mod:`src.models.deep_family` does, so they plug into
``deep_forecast.train_and_predict`` and the ``deep_window`` model unchanged.

* ``fnn``          feed-forward: the window (average-pooled by ``pool``) and a learned asset embedding through two hidden layers
* ``cnn``          a temporal convolutional network: dilated one-dimensional convolutions with residual connections over the window, then pooling over time; the receptive field grows geometrically with depth
* ``lstm``, ``gru`` a recurrent network over the window summed in blocks of ``stride`` days, reading the last hidden state
* ``transformer``  an encoder with sinusoidal positions over blocks of ``stride`` days and mean pooling (the patch transformer of ``patchtst`` is the same idea with learned positions and patches of 21 days)
"""

from __future__ import annotations

import math

SEQUENCE = ("fnn", "cnn", "lstm", "gru", "transformer")


def build_sequence_network(kind: str, n_assets: int, window: int = 252, **cfg):
    import torch
    from torch import nn

    emb_dim = int(cfg.get("asset_embedding", 8))
    dropout = float(cfg.get("dropout", 0.2))

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.asset = nn.Embedding(n_assets, emb_dim)
            nn.init.normal_(self.asset.weight, std=0.02)

    def blocks(x, stride):
        if window % stride:
            raise ValueError(f"stride {stride} does not divide the window {window}")
        return x.reshape(x.shape[0], -1, stride).sum(dim=2) / math.sqrt(stride)                       # block sums, rescaled to unit variance

    class FNN(Base):
        def __init__(self, width: int, pool: int):
            super().__init__()
            if window % pool:
                raise ValueError(f"pool {pool} does not divide the window {window}")
            self.pool = pool
            self.net = nn.Sequential(nn.Linear(window // pool + emb_dim, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width // 2, 1))

        def forward(self, x, asset):
            seen = x if self.pool == 1 else x.reshape(x.shape[0], -1, self.pool).mean(dim=2)
            return self.net(torch.cat([seen, self.asset(asset)], dim=1)).squeeze(-1)

    class TCN(Base):
        def __init__(self, channels: int, depth: int, kernel: int):
            super().__init__()
            self.inp = nn.Conv1d(1, channels, 1)
            self.convs = nn.ModuleList(nn.Conv1d(channels, channels, kernel, dilation=2 ** i, padding=(kernel - 1) * 2 ** i // 2) for i in range(depth))
            self.norms = nn.ModuleList(nn.GroupNorm(1, channels) for _ in range(depth))
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(2 * channels + emb_dim, 1)

        def forward(self, x, asset):
            h = self.inp(x.unsqueeze(1))
            for conv, norm in zip(self.convs, self.norms):
                h = h + self.drop(torch.relu(norm(conv(h))))                                          # residual dilated convolution
            pooled = torch.cat([h.mean(dim=2), h[:, :, -window // 4:].mean(dim=2)], dim=1)            # whole window and the most recent quarter
            return self.head(torch.cat([pooled, self.asset(asset)], dim=1)).squeeze(-1)

    class Recurrent(Base):
        def __init__(self, cell: str, hidden: int, stride: int, layers: int):
            super().__init__()
            self.stride = stride
            rnn = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = rnn(1, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden + emb_dim, 1)

        def forward(self, x, asset):
            out, _ = self.rnn(blocks(x, self.stride).unsqueeze(-1))
            return self.head(torch.cat([out[:, -1], self.asset(asset)], dim=1)).squeeze(-1)

    class Transformer(Base):
        def __init__(self, d: int, heads: int, layers: int, stride: int):
            super().__init__()
            if d % heads:
                raise ValueError("d_model must be divisible by heads")
            self.stride = stride
            steps = window // stride
            self.embed = nn.Linear(1, d)
            pos = torch.zeros(steps, d)
            t = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
            pos[:, 0::2], pos[:, 1::2] = torch.sin(t * div), torch.cos(t * div)[:, : d // 2]
            self.register_buffer("pos", pos.unsqueeze(0))
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.asset_proj = nn.Linear(emb_dim, d)
            self.head = nn.Linear(d, 1)

        def forward(self, x, asset):
            tokens = self.embed(blocks(x, self.stride).unsqueeze(-1)) + self.pos + self.asset_proj(self.asset(asset))[:, None, :]
            return self.head(self.norm(self.body(tokens)).mean(dim=1)).squeeze(-1)

    if kind == "fnn":
        return FNN(int(cfg.get("width", 64)), int(cfg.get("pool", 3)))
    if kind == "cnn":
        return TCN(int(cfg.get("channels", 16)), int(cfg.get("depth", 5)), int(cfg.get("kernel", 3)))
    if kind in ("lstm", "gru"):
        return Recurrent(kind, int(cfg.get("hidden", 16)), int(cfg.get("stride", 3)), int(cfg.get("rnn_layers", 1)))
    if kind == "transformer":
        return Transformer(int(cfg.get("d_model", 16)), int(cfg.get("heads", 2)), int(cfg.get("layers", 2)), int(cfg.get("stride", 6)))
    raise ValueError(f"unknown sequence model '{kind}'")
