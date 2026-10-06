"""Patch transformer and MLP-mixer forecasters of 21-day returns (Generation 4, Priority 17).

Both models see the same thing: the 252 daily returns before an origin, each divided by the EWMA daily
volatility AT the origin and clipped, cut into 21-day patches. They predict the 21-day return divided by the
EWMA volatility forecast (a z-score), so the forecast is ``mu_z * sigma`` and the volatility forecast of
Stage 19 can be reused unchanged: only the conditional mean differs.

Everything here is deterministic on CPU for a given seed (single thread, deterministic algorithms), so a refit
can run as an independent task in the distributed executor and give the same answer as a serial run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .probabilistic import ewma_sigma

PATCH = 21


def normalised_windows(returns: pd.DataFrame, sigma_daily: pd.DataFrame, positions: np.ndarray, asset_index: np.ndarray,
                       window: int = 252, clip: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Windows of normalised returns ending AT each (position, asset); rows with any missing value are flagged invalid.

    ``positions`` are integer day positions; the window is ``returns[position - window + 1 : position + 1]`` for the
    asset, divided by that asset's daily volatility at ``position`` (through ``position`` itself, as the Stage 19
    volatility forecast is) and clipped at ``+-clip``. Returns ``(X, valid)`` with ``X`` of shape ``(n, window)``.
    """
    values = returns.to_numpy(dtype=float)
    sig = sigma_daily.to_numpy(dtype=float)
    n = len(positions)
    X = np.full((n, window), np.nan)
    for i, (p, a) in enumerate(zip(positions, asset_index)):
        if p - window + 1 < 0:
            continue
        X[i] = values[p - window + 1:p + 1, a] / sig[p, a]
    valid = np.isfinite(X).all(axis=1)
    X = np.clip(np.where(valid[:, None], X, 0.0), -clip, clip)
    return X, valid


def daily_sigma(returns: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """EWMA daily volatility (horizon 1) through each day."""
    return ewma_sigma(returns, halflife, 1)


@dataclass
class TrainSettings:
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5


def build_network(kind: str, n_assets: int, window: int = 252, d_model: int = 32, layers: int = 2, heads: int = 4,
                  ff: int = 64, dropout: float = 0.2, family: dict | None = None):
    from .deep_family import FAMILY, build_family_network

    if kind in FAMILY:                                   # Generation 5 (Stage 33); the Stage 27 networks below are untouched
        return build_family_network(kind, n_assets, window, **(family or {}))
    import torch
    from torch import nn

    patches = window // PATCH

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(PATCH, d_model)
            self.position = nn.Parameter(torch.zeros(1, patches, d_model))
            self.asset = nn.Embedding(n_assets, d_model)
            nn.init.normal_(self.position, std=0.02)
            nn.init.normal_(self.asset.weight, std=0.02)
            if kind == "patchtst":
                layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
                self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            elif kind == "tsmixer":
                self.time_norm = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers))
                self.time_mlp = nn.ModuleList(nn.Sequential(nn.Linear(patches, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, patches))
                                              for _ in range(layers))
                self.feat_norm = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers))
                self.feat_mlp = nn.ModuleList(nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d_model))
                                              for _ in range(layers))
            else:
                raise ValueError(f"unknown deep model '{kind}'")
            self.out_norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x, asset):
            tokens = self.embed(x.reshape(x.shape[0], patches, PATCH)) + self.position + self.asset(asset)[:, None, :]
            if kind == "patchtst":
                tokens = self.body(tokens)
            else:
                for tn, tm, fn_, fm in zip(self.time_norm, self.time_mlp, self.feat_norm, self.feat_mlp):
                    tokens = tokens + tm(tn(tokens).transpose(1, 2)).transpose(1, 2)
                    tokens = tokens + fm(fn_(tokens))
            return self.head(self.out_norm(tokens).mean(dim=1)).squeeze(-1)

    return Backbone()


def parameter_count(kind: str, n_assets: int, **kwargs) -> int:
    return sum(p.numel() for p in build_network(kind, n_assets, **kwargs).parameters())


def train_and_predict(kind: str, X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, seed: int, n_assets: int,
                      net_kwargs: dict | None = None, settings: TrainSettings = TrainSettings(), mc_samples: int = 0) -> dict:
    """Train one network with early stopping on a validation block and predict the test rows.

    Returns ``{"pred": mu_z for the test rows, "val_loss": per-epoch validation MSE, "best_epoch": int, "train_loss": [...]}``.
    With ``mc_samples > 0`` it also returns ``"mc_pred"`` of shape (mc_samples, n_test): predictions with dropout left ON
    (Monte Carlo dropout, Gal and Ghahramani 2016). With the default 0 nothing extra is drawn, so Stage 27 is unchanged.
    """
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    net = build_network(kind, n_assets, **(net_kwargs or {}))
    optimiser = torch.optim.Adam(net.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    tx, ta, ty = (torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(a_train, dtype=torch.long), torch.as_tensor(y_train, dtype=torch.float32))
    vx, va, vy = (torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(a_val, dtype=torch.long), torch.as_tensor(y_val, dtype=torch.float32))
    best, best_state, best_epoch, history, train_history = np.inf, None, 0, [], []
    for epoch in range(settings.max_epochs):
        net.train()
        order = torch.randperm(len(tx), generator=generator)
        running = 0.0
        for i in range(0, len(order), settings.batch_size):
            idx = order[i:i + settings.batch_size]
            optimiser.zero_grad()
            loss = torch.mean((net(tx[idx], ta[idx]) - ty[idx]) ** 2)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((net(vx, va) - vy) ** 2))
        history.append(val_loss)
        train_history.append(running / len(order))
        if val_loss < best - 1e-9:
            best, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        elif epoch - best_epoch >= settings.patience:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(torch.as_tensor(X_test, dtype=torch.float32), torch.as_tensor(a_test, dtype=torch.long)).numpy().astype(float)
    out = {"pred": pred, "val_loss": history, "train_loss": train_history, "best_epoch": int(best_epoch)}
    if mc_samples > 0:
        out["mc_pred"] = mc_dropout_predict(net, X_test, a_test, mc_samples, seed)
    return out


def mc_dropout_predict(net, X_test, a_test, samples: int, seed: int) -> np.ndarray:
    """``samples`` stochastic forward passes with dropout active (the nets use LayerNorm, never batch norm, so train mode only switches dropout on)."""
    import torch

    torch.manual_seed(seed + 100_003)
    net.train()
    tx, ta = torch.as_tensor(X_test, dtype=torch.float32), torch.as_tensor(a_test, dtype=torch.long)
    with torch.no_grad():
        draws = np.stack([net(tx, ta).numpy().astype(float) for _ in range(samples)])
    net.eval()
    return draws


def ridge_window(X_train, y_train, X_test, alpha: float = 1000.0) -> np.ndarray:
    """The linear control: ridge regression on the same normalised window."""
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha).fit(X_train, y_train).predict(X_test)
