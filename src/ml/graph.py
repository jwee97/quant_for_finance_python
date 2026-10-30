"""Graph neural networks over the assets: ``gcn`` (Kipf and Welling) and ``gat`` (Velickovic et al.).

Each date is one graph. The nodes are the assets, a node's features are the asset's features on that date, and the edges say which assets should inform each other: the ``neighbors`` most correlated
assets (correlation of daily returns over the TRAINING window only) and the assets that share a sector or asset class. A layer lets every asset read its neighbours' features, so a score can use what the
rest of the market, and the asset's own sector, are doing on the same day (all of it known by that day's close).

Two outputs per date: a node score (the asset's return relative to the day's average) and a portfolio-level forecast (the day's average target, read from the mean of the node embeddings); the score is their
sum, in the target's units like every other model, and the portfolio forecast is also reported on its own. Graphs are dense (a few dozen assets); a missing asset on a date is a zero vector that the loss ignores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .deep.data import FeatureScaler, SeqAccessor
from .deep.trainer import TrainConfig, fit_network, resolve_device
from .models import Model

MIN_DATES = 120
MIN_NODES = 3


def build_adjacency(returns: pd.DataFrame, groups: dict, mode: str = "both", neighbors: int = 4, corr_weight: float = 1.0, sector_weight: float = 1.0) -> np.ndarray:
    """Symmetric non-negative edge weights between the columns of ``returns`` (no self loops): the correlation of each asset's ``neighbors`` nearest assets, and ``sector_weight`` between assets of one group."""
    names = list(returns.columns)
    n = len(names)
    A = np.zeros((n, n))
    if mode in ("both", "correlation"):
        corr = returns.corr(min_periods=60).fillna(0.0).to_numpy()
        np.fill_diagonal(corr, -np.inf)
        k = max(1, min(int(neighbors), n - 1))
        for i in range(n):
            for j in np.argsort(corr[i])[::-1][:k]:
                A[i, j] = max(A[i, j], float(corr_weight) * max(corr[i, j], 0.0))
    if mode in ("both", "sector"):
        label = [groups.get(a) for a in names]
        if all(v is None for v in label):
            if mode == "sector":
                raise ValueError("the dataset has no sector or asset-class labels to make edges from; choose graph=correlation")
        else:
            for i in range(n):
                for j in range(n):
                    if i != j and label[i] is not None and label[i] == label[j]:
                        A[i, j] = max(A[i, j], float(sector_weight))
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


class GraphNet(nn.Module):
    def __init__(self, kind: str, n_features: int, adjacency: np.ndarray, hidden: int, layers: int, dropout: float, heads: int = 2):
        super().__init__()
        self.kind = kind
        A = torch.tensor(adjacency, dtype=torch.float32) + torch.eye(len(adjacency))
        deg = A.sum(1)
        self.register_buffer("norm", A / torch.sqrt(deg[:, None] * deg[None, :]))              # D^-1/2 (A + I) D^-1/2
        self.register_buffer("mask", A > 0)
        self.inp = nn.Linear(n_features, hidden)
        self.heads = int(heads) if kind == "gat" else 1
        if hidden % self.heads:
            raise ValueError("hidden must be a multiple of heads")
        self.lin = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.att_src = nn.ParameterList([nn.Parameter(torch.randn(self.heads, hidden // self.heads) * 0.1) for _ in range(layers)]) if kind == "gat" else None
        self.att_dst = nn.ParameterList([nn.Parameter(torch.randn(self.heads, hidden // self.heads) * 0.1) for _ in range(layers)]) if kind == "gat" else None
        self.drop = nn.Dropout(dropout)
        self.node = nn.Linear(hidden, 1)
        self.portfolio = nn.Linear(hidden, 1)
        self.last_attention = None

    def _gcn(self, h: torch.Tensor, i: int) -> torch.Tensor:
        return torch.einsum("ij,bjh->bih", self.norm, self.lin[i](h))

    def _gat(self, h: torch.Tensor, i: int) -> torch.Tensor:
        b, n, _ = h.shape
        wh = self.lin[i](h).view(b, n, self.heads, -1)
        e = F.leaky_relu((wh * self.att_src[i]).sum(-1)[:, :, None, :] + (wh * self.att_dst[i]).sum(-1)[:, None, :, :], 0.2)          # [batch, i, j, head]
        e = e.masked_fill(~self.mask[None, :, :, None], float("-inf"))
        alpha = torch.softmax(e, dim=2)
        self.last_attention = alpha.detach().mean(dim=(0, 3))
        return torch.einsum("bijh,bjhd->bihd", alpha, wh).reshape(b, n, -1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.inp(x))
        for i in range(len(self.lin)):
            h = h + self.drop(F.elu(self._gcn(h, i) if self.kind == "gcn" else self._gat(h, i)))
        pooled = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        return self.node(h).squeeze(-1), self.portfolio(pooled).squeeze(-1)


class GraphModel(Model):
    needs_dataset = True

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        if task != "regression":
            raise ValueError(f"{kind} is a regression")
        p = self.params
        if p["graph"] not in ("both", "correlation", "sector"):
            raise ValueError("graph must be both, correlation or sector")
        if int(p["hidden"]) < 1 or int(p["layers"]) < 1 or not 0.0 <= float(p["dropout"]) < 1.0 or int(p["neighbors"]) < 1 or int(p["epochs"]) < 1:
            raise ValueError(f"invalid setting(s) for {kind}")
        resolve_device(str(p["device"]))
        self._ds = self._base = None
        self.net_ = None

    def bind(self, ds) -> "GraphModel":
        self._ds, self._base = ds, None
        return self

    # ------------------------------------------------------------------------------------------------------------ the panel as graphs
    def _axis(self, X: pd.DataFrame) -> tuple[pd.DatetimeIndex, list]:
        assets = list(getattr(self, "assets_", None) or (self._ds.assets if self._ds is not None else sorted(X.index.get_level_values(1).unique())))
        dates = self._ds.dates if self._ds is not None else pd.DatetimeIndex(sorted(X.index.get_level_values(0).unique()))
        return dates, assets

    def _graphs(self, X: pd.DataFrame, rows_only: bool):
        """``(cross_sections [dates, assets, features], present [dates, assets], date_of_row, asset_of_row, dates)`` for the dates of ``X``. With ``rows_only`` False the assets a date's rows lack are filled
        from the dataset's own features of that date (a few rows asked about, the rest of the day's graph is known)."""
        cols = self.features_
        axis, assets = self._axis(X)
        t_row, a_row = axis.get_indexer(X.index.get_level_values(0)), pd.Index(assets).get_indexer(X.index.get_level_values(1))
        if (t_row < 0).any() or (a_row < 0).any():
            raise ValueError("some rows are on a date or for an asset that is not in the model's graph")
        used = np.unique(t_row)
        local = np.searchsorted(used, t_row)
        sub = np.full((len(used), len(assets), len(cols)), np.nan, dtype=np.float32)
        sub[local, a_row] = X[cols].to_numpy(dtype=np.float32)
        present = np.zeros(sub.shape[:2], dtype=bool)
        present[local, a_row] = True
        if self._ds is not None and not rows_only:
            if self._base is None:
                self._base = SeqAccessor.from_frame(self._ds.X[cols], self._ds.dates, self._ds.assets)
            fill = self._base.arr[used]
            hole = ~present
            sub[hole] = fill[hole]
            present |= np.isfinite(fill).any(axis=2)
        return sub, present, local, a_row, axis[used]

    # ------------------------------------------------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "GraphModel":
        if self._ds is None:
            raise ValueError(f"{self.kind} needs the dataset's returns and labels to build its graph: it must be bound to a dataset before it is fitted")
        Xs = self._select(X)
        self.features_ = list(Xs.columns)
        self.n_train_ = len(Xs)
        p = self.params
        d_all = Xs.index.get_level_values(0)
        end = d_all.max()
        self.assets_ = list(self._ds.assets)
        returns = self._ds.returns[self.assets_].loc[:end].tail(504)
        groups = {a: (self._ds.group.get(a) or self._ds.asset_class.get(a)) for a in self.assets_}
        self.adjacency_ = build_adjacency(returns, groups, str(p["graph"]), int(p["neighbors"]), float(p["corr_weight"]), float(p["sector_weight"]))
        target = self._prepare_y(y).astype(np.float32)
        sub, present, local, a_row, ds_dates = self._graphs(Xs, rows_only=True)
        n_t, n_a = present.shape
        Yg = np.full((n_t, n_a), np.nan, dtype=np.float32)
        Yg[local, a_row] = target
        ok = np.isfinite(Yg)
        count = ok.sum(1)
        mean_d = np.where(count > 0, np.nansum(np.where(ok, Yg, 0.0), 1) / np.maximum(count, 1), 0.0).astype(np.float32)
        node_y, port_y = np.where(ok, Yg - mean_d[:, None], 0.0).astype(np.float32), mean_d
        usable = np.flatnonzero(count >= MIN_NODES)
        pos = np.asarray(dates) if dates is not None else None
        if pos is None:
            date_pos = np.arange(n_t)
        else:
            first = pd.Series(pos).groupby(d_all.to_numpy()).first()
            date_pos = first.reindex(ds_dates).to_numpy()
        usable = usable[np.isfinite(date_pos[usable])]
        if len(usable) < MIN_DATES:
            raise ValueError(f"{len(usable)} dates with at least {MIN_NODES} labelled assets are too few for a graph network (at least {MIN_DATES})")
        order = usable[np.argsort(date_pos[usable])]
        n_val = max(8, int(round(len(order) * float(p["val_fraction"]))))
        purge = self._ds.lag + self._ds.horizon
        val_idx = order[-n_val:]
        train_idx = order[: max(0, len(order) - n_val - purge)]
        if len(train_idx) < MIN_DATES // 2:
            raise ValueError("too few dates left to train on after the validation tail and the purge")
        self.scaler_ = FeatureScaler().fit(sub[train_idx][present[train_idx]])
        device, note = resolve_device(str(p["device"]))
        torch.manual_seed(self.seed)
        net = GraphNet(self.kind, len(self.features_), self.adjacency_, int(p["hidden"]), int(p["layers"]), float(p["dropout"]), int(p.get("heads", 1))).to(device)
        sets = {"train": train_idx, "val": val_idx}

        def batch_fn(split, idx):
            t = sets[split][idx]
            x = torch.from_numpy(self.scaler_.transform(sub[t])).to(device)
            valid = torch.from_numpy(ok[t].astype(np.float32)).to(device)
            return x, valid, torch.from_numpy(node_y[t]).to(device), torch.from_numpy(port_y[t]).to(device)

        def loss_fn(model, batch):
            x, valid, ny, py = batch
            node, port = model(x, valid)
            return (((node - ny) ** 2) * valid).sum() / valid.sum().clamp(min=1.0) + 0.5 * ((port - py) ** 2).mean()

        cfg = TrainConfig(int(p["epochs"]), int(p["batch_size"]), float(p["learning_rate"]), float(p["weight_decay"]), int(p["patience"]), float(p["grad_clip"]), str(p["device"]), self.seed)
        self.history_ = fit_network(net, batch_fn, loss_fn, len(train_idx), len(val_idx), cfg)
        self.history_.device_note = note
        self.net_ = net
        self.net_spec_ = (self.kind, len(self.features_), int(p["hidden"]), int(p["layers"]), float(p["dropout"]), int(p.get("heads", 1)))
        self.importance_ = self._saliency(batch_fn, len(val_idx), device)
        self.n_graphs_ = (int(len(train_idx)), int(len(val_idx)))
        return self

    def _saliency(self, batch_fn, n_val: int, device) -> np.ndarray:
        take = np.arange(0, n_val, max(1, n_val // 32))[:32]
        x, valid, _, _ = batch_fn("val", take)
        x.requires_grad_(True)
        self.net_.eval()
        node, port = self.net_(x, valid)
        (node * valid).sum().backward()
        g = x.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        return g / (g.sum() + 1e-12)

    # ------------------------------------------------------------------------------------------------------------ inference
    @torch.no_grad()
    def _forward(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        device, _ = resolve_device(str(self.params["device"]))
        self.net_.to(device).eval()
        sub, present, local, a_row, dates = self._graphs(X, rows_only=False)
        nodes, ports = [], []
        for i in range(0, len(sub), 256):
            x = torch.from_numpy(self.scaler_.transform(sub[i:i + 256])).to(device)
            node, port = self.net_(x, torch.from_numpy(present[i:i + 256].astype(np.float32)).to(device))
            nodes.append(node.cpu().numpy())
            ports.append(port.cpu().numpy())
        node, port = np.concatenate(nodes), np.concatenate(ports)
        return node[local, a_row], port, local

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        node, port, local = self._forward(X)
        return (node + port[local]) * self.y_std_ + self.y_mean_

    def predict_extras(self, X: pd.DataFrame) -> dict:
        _, port, _ = self._forward(X)
        dates = pd.DatetimeIndex(sorted(set(X.index.get_level_values(0))))
        return {"portfolio": pd.DataFrame({"predicted": port * self.y_std_ + self.y_mean_}, index=dates)}

    def importance(self) -> pd.Series:
        return pd.Series(self.importance_, index=self.features_)

    def diagnostics(self) -> dict:
        h = self.history_.to_dict()
        A = self.adjacency_
        iu = np.dstack(np.unravel_index(np.argsort(-np.triu(A), axis=None)[:30], A.shape))[0]
        edges = [[self.assets_[i], self.assets_[j], float(A[i, j])] for i, j in iu if A[i, j] > 0]
        d = {"arch": self.kind, "graph": {"assets": self.assets_, "edges": edges, "mode": self.params["graph"], "n_edges": int((np.triu(A) > 0).sum()), "mean_degree": float((A > 0).sum(1).mean())},
             "graphs": list(self.n_graphs_), **{k: h[k] for k in ("train_loss", "val_loss", "lr", "grad_norm", "best_epoch", "stopped_early", "device", "device_note", "n_params", "seconds", "peak_memory_mb", "epochs_run")}}
        att = getattr(self.net_, "last_attention", None)
        if att is not None:
            att = att.cpu().numpy()
            top = np.dstack(np.unravel_index(np.argsort(-att, axis=None)[:30], att.shape))[0]
            d["attention"] = [[self.assets_[i], self.assets_[j], float(att[i, j])] for i, j in top if i != j][:20]
        return d

    # ------------------------------------------------------------------------------------------------------------ serialization
    def __getstate__(self):
        import io

        state = dict(self.__dict__)
        state.update(_ds=None, _base=None)
        if self.net_ is not None:
            buf = io.BytesIO()
            torch.save({k: v.cpu() for k, v in self.net_.state_dict().items()}, buf)
            state.update(net_=None, _state_bytes=buf.getvalue())
        return state

    def __setstate__(self, state):
        import io

        blob = state.pop("_state_bytes", None)
        self.__dict__.update(state)
        if blob is not None:
            kind, n_features, hidden, layers, dropout, heads = self.net_spec_
            net = GraphNet(kind, n_features, self.adjacency_, hidden, layers, dropout, heads)
            net.load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True))
            self.net_ = net.eval()


def build(kind: str, task: str, params: dict, seed: int) -> Model:
    return GraphModel(kind, task, params, seed)
