"""The networks. Every forecaster maps a window ``[batch, lookback, feature]`` to ``[batch, n_out]`` (one output per forecast horizon; the Temporal Fusion Transformer gives three quantiles per
horizon), so the trainer and the model wrapper do not care which one they hold. The autoencoders map a window to a reconstruction and a latent code.

These are compact PyTorch implementations written for panels of a few dozen features and windows of a few weeks, not ports of the reference code; where a design choice departs from the paper it is said in
the class's docstring.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RNNNet(nn.Module):
    """LSTM or GRU over the window; the last hidden state goes through a normalised linear head."""

    def __init__(self, kind: str, n_features: int, n_out: int, hidden: int = 32, layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        cell = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cell(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class _CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))                        # left padding only: the output at t never sees t+1
        h = self.drop(F.relu(self.conv2(F.pad(h, (self.pad, 0)))))
        return F.relu(h + self.skip(x))


class TCNNet(nn.Module):
    """Temporal convolutional network (Bai, Kolter and Koltun 2018): causal, dilated, residual. The receptive field is 1 + 2 (kernel - 1) (2^layers - 1) steps."""

    def __init__(self, n_features: int, n_out: int, hidden: int = 32, layers: int = 3, kernel_size: int = 3, dropout: float = 0.1, **_):
        super().__init__()
        self.blocks = nn.Sequential(*[_CausalBlock(n_features if i == 0 else hidden, hidden, int(kernel_size), 2 ** i, dropout) for i in range(layers)])
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x.transpose(1, 2))
        return self.head(h[:, :, -1])


class PatchTST(nn.Module):
    """Patch time-series transformer (Nie et al. 2023). Channel-independent: each channel's window is cut into patches, embedded, and read by a transformer encoder shared by all channels; a shared linear
    layer summarises each channel and a final linear layer mixes the channels into the outputs. A panel has dozens of correlated features, and a transformer pass per feature is slow on a CPU, so a linear
    layer first reduces the features to ``channels`` learned channels (0 keeps every feature as its own channel, as the paper does). No reversible instance normalisation: the features are already standardised."""

    def __init__(self, n_features: int, lookback: int, n_out: int, patch_len: int = 4, stride: int = 2, d_model: int = 32, n_heads: int = 4, ff_dim: int = 64, layers: int = 2, dropout: float = 0.1,
                 channels: int = 8, **_):
        super().__init__()
        channels = int(channels) if 0 < int(channels) < n_features else n_features
        self.reduce = nn.Linear(n_features, channels) if channels != n_features else nn.Identity()
        if lookback < patch_len:
            raise ValueError("lookback must be at least patch_len")
        if d_model % n_heads:
            raise ValueError("d_model must be a multiple of n_heads")
        self.patch_len, self.stride = int(patch_len), int(stride)
        n_patches = (lookback - self.patch_len) // self.stride + 1
        self.embed = nn.Linear(self.patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.summary = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        self.mix = nn.Linear(channels, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        b, length, c = x.shape
        patches = x.transpose(1, 2).reshape(b * c, length).unfold(1, self.patch_len, self.stride)            # [b*c, patches, patch_len]
        h = self.encoder(self.embed(patches) + self.pos)
        per_channel = self.summary(h.mean(dim=1)).view(b, c)
        return self.mix(self.drop(per_channel))


class _MLP(nn.Module):
    def __init__(self, d_in: int, width: int, layers: int, dropout: float):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
            d = width
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NBeats(nn.Module):
    """N-BEATS, generic architecture (Oreshkin et al. 2020): a stack of blocks, each a fully connected network that emits a backcast (subtracted from the block's input) and a forecast (added to the
    output). The window of all features is flattened, so the network is multivariate, which the paper's univariate series are not."""

    def __init__(self, n_features: int, lookback: int, n_out: int, blocks: int = 3, width: int = 64, layers: int = 3, dropout: float = 0.1, **_):
        super().__init__()
        d_in = n_features * lookback
        self.body = nn.ModuleList([_MLP(d_in, width, layers, dropout) for _ in range(blocks)])
        self.back = nn.ModuleList([nn.Linear(width, d_in) for _ in range(blocks)])
        self.fore = nn.ModuleList([nn.Linear(width, n_out) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual, out = x.flatten(1), 0.0
        for body, back, fore in zip(self.body, self.back, self.fore):
            h = body(residual)
            residual = residual - back(h)
            out = out + fore(h)
        return out


class NHiTS(nn.Module):
    """N-HiTS (Challu et al. 2023): N-BEATS blocks that each read the window max-pooled at their own rate (coarse blocks first) and emit the backcast at that coarse resolution, interpolated back to the
    full window. With only a handful of forecast horizons the forecast needs no interpolation, so each block's forecast has one coefficient per output."""

    def __init__(self, n_features: int, lookback: int, n_out: int, pool_sizes=(4, 2, 1), width: int = 64, layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        self.pools = [int(k) for k in pool_sizes]
        self.length, self.n_features = int(lookback), int(n_features)
        self.knots = [math.ceil(lookback / k) for k in self.pools]
        self.body = nn.ModuleList([_MLP(n_features * q, width, layers, dropout) for q in self.knots])
        self.back = nn.ModuleList([nn.Linear(width, n_features * q) for q in self.knots])
        self.fore = nn.ModuleList([nn.Linear(width, n_out) for _ in self.pools])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual, out = x.transpose(1, 2), 0.0                                              # [batch, feature, time]
        for k, q, body, back, fore in zip(self.pools, self.knots, self.body, self.back, self.fore):
            pooled = F.max_pool1d(residual, k, k, ceil_mode=True) if k > 1 else residual
            h = body(pooled.flatten(1))
            coarse = back(h).view(-1, self.n_features, q)
            residual = residual - (F.interpolate(coarse, size=self.length, mode="linear", align_corners=False) if q != self.length else coarse)
            out = out + fore(h)
        return out


class GRN(nn.Module):
    """Gated residual network of the Temporal Fusion Transformer."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_in, d_hidden), nn.Linear(d_hidden, d_out * 2)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.fc2(F.elu(self.fc1(x))))
        return self.norm(self.skip(x) + F.glu(h, dim=-1))


class TFT(nn.Module):
    """A compact Temporal Fusion Transformer (Lim et al. 2021) for a panel with no static inputs: variable selection (softmax weights over the features at each step, from a gated residual network, applied
    to per-feature embeddings), an LSTM encoder with a gated skip, multi-head attention from the last step over the whole window, a gated position-wise network and a linear map to three quantiles per
    horizon (quantile ``i`` of horizon ``h`` is output ``3h + i``). Standard multi-head attention stands in for the paper's interpretable variant; ``last_attention`` and ``last_variables`` keep the batch
    means of the attention weights and the variable-selection weights for the diagnostics."""

    def __init__(self, n_features: int, lookback: int, n_out: int, hidden: int = 32, n_heads: int = 2, lstm_layers: int = 1, dropout: float = 0.1, **_):
        super().__init__()
        if hidden % n_heads:
            raise ValueError("hidden must be a multiple of n_heads")
        self.select = GRN(n_features, hidden, n_features, dropout)
        self.embed = nn.Parameter(torch.randn(n_features, hidden) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.lstm = nn.LSTM(hidden, hidden, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.gate1 = nn.Sequential(nn.Linear(hidden, hidden * 2))
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.gate2 = nn.Linear(hidden, hidden * 2)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = GRN(hidden, hidden, hidden, dropout)
        self.out = nn.Linear(hidden, n_out * 3)
        self.last_attention = self.last_variables = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.select(x), dim=-1)                                            # [batch, time, feature]
        h = torch.einsum("blf,fd->bld", w * x, self.embed) + self.bias
        enc, _ = self.lstm(h)
        enc = self.norm1(h + F.glu(self.gate1(enc), dim=-1))
        q = enc[:, -1:]
        att, weights = self.attn(q, enc, enc, need_weights=True, average_attn_weights=True)
        z = self.norm2(q + F.glu(self.gate2(att), dim=-1))
        z = self.ff(z)[:, 0]
        self.last_attention, self.last_variables = weights.detach().mean(dim=(0, 1)), w.detach().mean(dim=(0, 1))
        return self.out(z)


class AutoEncoder(nn.Module):
    """A fully connected autoencoder over the flattened window; with ``variational`` the encoder outputs a mean and a log-variance and the code is sampled (the reparameterisation trick)."""

    def __init__(self, n_features: int, lookback: int, latent_dim: int = 8, hidden: int = 32, layers: int = 1, dropout: float = 0.1, variational: bool = False, **_):
        super().__init__()
        d_in = n_features * lookback
        self.variational, self.latent_dim = bool(variational), int(latent_dim)
        self.encoder = _MLP(d_in, hidden, layers, dropout)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim) if variational else None
        self.decoder = nn.Sequential(_MLP(latent_dim, hidden, layers, 0.0), nn.Linear(hidden, d_in))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.encoder(x.flatten(1))
        return self.mu(h), (self.logvar(h).clamp(-8.0, 8.0) if self.variational else None)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if (self.variational and self.training) else mu
        return self.decoder(z).view_as(x), mu, logvar


FORECASTERS = ("lstm", "gru", "tcn", "patchtst", "nbeats", "nhits", "tft")


def build_net(arch: str, n_features: int, lookback: int, n_out: int, cfg: dict) -> nn.Module:
    """The network for ``arch`` with the settings in ``cfg`` (the model's parameters)."""
    p = dict(cfg)
    hidden, layers, dropout = int(p.get("hidden", 32)), int(p.get("layers", 2)), float(p.get("dropout", 0.1))
    if arch in ("lstm", "gru"):
        return RNNNet(arch, n_features, n_out, hidden, layers, dropout)
    if arch == "tcn":
        return TCNNet(n_features, n_out, hidden, layers, int(p.get("kernel_size", 3)), dropout)
    if arch == "patchtst":
        return PatchTST(n_features, lookback, n_out, int(p.get("patch_len", 4)), int(p.get("stride", 2)), int(p.get("d_model", 32)), int(p.get("n_heads", 4)), int(p.get("ff_dim", 64)), layers, dropout, int(p.get("channels", 8)))
    if arch == "nbeats":
        return NBeats(n_features, lookback, n_out, int(p.get("blocks", 3)), int(p.get("width", 64)), layers, dropout)
    if arch == "nhits":
        pools = [int(k) for k in str(p.get("pool_sizes", "4,2,1")).replace(" ", "").split(",") if k]
        return NHiTS(n_features, lookback, n_out, pools or (1,), int(p.get("width", 64)), layers, dropout)
    if arch == "tft":
        return TFT(n_features, lookback, n_out, hidden, int(p.get("n_heads", 2)), int(p.get("lstm_layers", 1)), dropout)
    if arch in ("autoencoder", "vae"):
        return AutoEncoder(n_features, lookback, int(p.get("latent_dim", 8)), hidden, layers, dropout, variational=arch == "vae")
    raise ValueError(f"unknown network '{arch}'")
