"""A graph attention forecaster over the ETF universe (Generation 5, Stage 33).

Nodes are the ETFs on one day. Each node carries standardised price features plus a learned asset embedding; edges join each ETF to its
``k`` most correlated others over the trailing window, recomputed for every day (so the graph at an origin uses returns through the
origin only). One graph-attention layer (Velickovic et al. 2018; additive attention, LeakyReLU, softmax over the node's neighbours and
itself) is followed by a readout that also sees the node's own projection. Pure torch, no graph library.

The target is the same z-score as Stage 27: the 21-day return over the EWMA volatility forecast. The model is trained on whole days
(every node of a day is one training example set), with a masked loss for nodes whose features or target are missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_neighbours(returns: pd.DataFrame, positions: np.ndarray, window: int = 252, k: int = 4) -> np.ndarray:
    """For each position, the indices of each asset's ``k`` most correlated other assets over ``returns[p - window + 1 : p + 1]``.

    Returns an int array (len(positions), n_assets, k). Pairs without enough joint history rank last; if an asset has fewer than ``k``
    usable neighbours the remainder is filled with the asset itself (a self-loop, which the attention then treats as a duplicate of itself).
    """
    values = returns.to_numpy(dtype=float)
    n = values.shape[1]
    out = np.zeros((len(positions), n, k), dtype=np.int64)
    for i, p in enumerate(positions):
        block = pd.DataFrame(values[max(0, p - window + 1):p + 1])
        corr = block.corr(min_periods=window // 2).to_numpy().copy()
        np.fill_diagonal(corr, np.nan)
        for a in range(n):
            row = np.where(np.isfinite(corr[a]), corr[a], -np.inf)
            order = np.argsort(-row, kind="stable")[:k]
            chosen = [int(j) for j in order if np.isfinite(row[j])]
            out[i, a] = chosen + [a] * (k - len(chosen))
    return out


def build_graph_network(n_assets: int, n_features: int, hidden: int = 32, heads: int = 2, embedding: int = 8, dropout: float = 0.2):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class GAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.asset = nn.Embedding(n_assets, embedding)
            nn.init.normal_(self.asset.weight, std=0.02)
            self.proj = nn.Linear(n_features + embedding, hidden * heads, bias=False)
            self.att_src = nn.Parameter(torch.empty(heads, hidden))
            self.att_dst = nn.Parameter(torch.empty(heads, hidden))
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)
            self.heads, self.hidden = heads, hidden
            self.drop = nn.Dropout(dropout)
            self.readout = nn.Sequential(nn.Linear(2 * hidden * heads, hidden), nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x, nbr):
            """``x``: (batch, n_assets, n_features); ``nbr``: (batch, n_assets, k) neighbour indices -> (batch, n_assets) z-scores."""
            b, n, _ = x.shape
            emb = self.asset.weight[None].expand(b, -1, -1)
            h = self.proj(torch.cat([x, emb], dim=2)).reshape(b, n, self.heads, self.hidden)          # (b, n, H, D)
            members = torch.cat([torch.arange(n, device=x.device)[None, :, None].expand(b, -1, 1), nbr], dim=2)   # self first, then neighbours
            batch_idx = torch.arange(b, device=x.device)[:, None, None].expand_as(members)
            neighbour_h = h[batch_idx, members]                                                         # (b, n, k+1, H, D)
            score = (neighbour_h * self.att_src).sum(-1) + (h * self.att_dst).sum(-1)[:, :, None, :]    # (b, n, k+1, H)
            alpha = self.drop(torch.softmax(F.leaky_relu(score, 0.2), dim=2))
            attended = (alpha[..., None] * neighbour_h).sum(dim=2)                                      # (b, n, H, D)
            joined = torch.cat([F.elu(attended).reshape(b, n, -1), h.reshape(b, n, -1)], dim=2)
            return self.readout(joined).squeeze(-1)

    return GAT()


def train_and_predict_graph(X_train, N_train, y_train, X_val, N_val, y_val, X_test, N_test, seed: int, n_assets: int, settings, net_kwargs: dict | None = None,
                            mc_samples: int = 0) -> dict:
    """Train on days (arrays of shape (days, assets, features) / (days, assets, k) / (days, assets)); NaN targets are masked out of the loss."""
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    net = build_graph_network(n_assets, X_train.shape[2], **(net_kwargs or {}))
    optimiser = torch.optim.Adam(net.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)

    def tensors(X, N, y=None):
        out = [torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(N, dtype=torch.long)]
        if y is not None:
            out.append(torch.as_tensor(np.nan_to_num(y, nan=0.0), dtype=torch.float32))
            out.append(torch.as_tensor(np.isfinite(y), dtype=torch.float32))
        return out

    tx, tn, ty, tm = tensors(X_train, N_train, y_train)
    vx, vn, vy, vm = tensors(X_val, N_val, y_val)

    def masked_mse(pred, y, m):
        return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)

    best, best_state, best_epoch, history, train_history = np.inf, None, 0, [], []
    # a "batch" is a set of days; the day count per batch keeps the node count per batch near the Stage 27 batch size
    days_per_batch = max(1, settings.batch_size // n_assets)
    for epoch in range(settings.max_epochs):
        net.train()
        order = torch.randperm(len(tx), generator=generator)
        running = 0.0
        for i in range(0, len(order), days_per_batch):
            idx = order[i:i + days_per_batch]
            optimiser.zero_grad()
            loss = masked_mse(net(tx[idx], tn[idx]), ty[idx], tm[idx])
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            val_loss = float(masked_mse(net(vx, vn), vy, vm))
        history.append(val_loss)
        train_history.append(running / len(order))
        if val_loss < best - 1e-9:
            best, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        elif epoch - best_epoch >= settings.patience:
            break
    net.load_state_dict(best_state)
    net.eval()
    sx, sn = tensors(X_test, N_test)
    with torch.no_grad():
        pred = net(sx, sn).numpy().astype(float)
    out = {"pred": pred, "val_loss": history, "train_loss": train_history, "best_epoch": int(best_epoch)}
    if mc_samples > 0:
        torch.manual_seed(seed + 100_003)
        net.train()
        with torch.no_grad():
            out["mc_pred"] = np.stack([net(sx, sn).numpy().astype(float) for _ in range(mc_samples)])
        net.eval()
    return out
