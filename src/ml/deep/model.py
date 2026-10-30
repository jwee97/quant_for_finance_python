"""The networks as models of the zoo: ``lstm``, ``gru``, ``tcn``, ``patchtst``, ``nbeats``, ``nhits``, ``tft``, ``autoencoder`` and ``vae``.

**Look-ahead.** A row's window holds that asset's features on the row's day and the days before it, nothing later. The model is bound to the dataset by the trainer so that the first days of a test block
still find their lookback in the dataset's feature history. The scaler is fitted on the training rows. The last ``val_fraction`` of the training dates is held out for early stopping, separated from the
rest by ``lag + horizon`` days (the purge of the walk-forward) so that no label of a training row overlaps a validation day. Extra forecast horizons (``horizons``) must be shorter than the dataset's, so
their labels are complete whenever the primary label is.

**Serialization.** ``to_bytes`` / ``Model.from_bytes`` (what the registry uses) keep the network as a state dict with the settings that rebuild it; ``save_model`` / ``load_model`` write the same to a file
with a checksum and the versions of the libraries it was made with.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge

from ..models import Model
from ..targets import forward_return
from .data import FeatureScaler, SeqAccessor
from .nets import FORECASTERS, build_net
from .trainer import TrainConfig, fit_network, resolve_device

AUTOENCODERS = ("autoencoder", "vae")
MIN_ROWS = 200


class _Source:
    """Windows for the rows of one frame: the dataset's history (when bound) with the frame's own rows laid over it."""

    def __init__(self, model: "DeepModel", X: pd.DataFrame, context: pd.DataFrame | None = None):
        cols = model.features_
        frame = X[cols] if context is None else pd.concat([context[cols], X[cols]])
        frame = frame[~frame.index.duplicated(keep="last")]
        self.length = int(model.params["lookback"])
        if model._ds is not None:
            self.acc = model._base_accessor()
            self.overlay = self.acc.overlay(frame)
        else:
            self.acc, self.overlay = SeqAccessor.from_frame(frame), None
        self.t, self.a = self.acc.locate(X.index)
        self.values = X[cols].to_numpy(dtype=np.float32)

    def get(self, idx: np.ndarray) -> np.ndarray:
        win = self.acc.windows(self.t[idx], self.a[idx], self.length, self.overlay)
        win[:, -1, :] = self.values[idx]                                          # the last step is the row itself, even when the index repeats a row (a perturbed copy of it)
        return win


class DeepModel(Model):
    needs_dataset = True

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        if task != "regression":
            raise ValueError(f"{kind} is a regression")
        p = self.params
        checks = {"lookback": int(p["lookback"]) >= 2, "hidden": int(p["hidden"]) >= 1, "layers": int(p["layers"]) >= 1, "epochs": int(p["epochs"]) >= 1, "batch_size": int(p["batch_size"]) >= 8,
                  "learning_rate": float(p["learning_rate"]) > 0, "patience": int(p["patience"]) >= 1, "dropout": 0.0 <= float(p["dropout"]) < 1.0, "val_fraction": 0.05 <= float(p["val_fraction"]) <= 0.5}
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"invalid setting(s) for {kind}: {bad}")
        if kind == "tft" and not 0.0 < float(p["alpha_low"]) < 0.5 < float(p["alpha_high"]) < 1.0:
            raise ValueError("alpha_low must be below 0.5 and alpha_high above it")
        resolve_device(str(p["device"]))
        self.arch = kind
        self._ds = self._base = None
        self.net_ = self.scaler_ = self.history_ = None
        self.heads_: list = []

    # ------------------------------------------------------------------------------------------------------------ data access
    def bind(self, ds) -> "DeepModel":
        self._ds, self._base = ds, None
        return self

    def _base_accessor(self) -> SeqAccessor:
        if self._base is None:
            self._base = SeqAccessor.from_frame(self._ds.X[self.features_], axis=self._ds.dates, assets=self._ds.assets)
        return self._base

    def _horizons(self) -> list[int]:
        raw = str(self.params.get("horizons") or "").replace(" ", "")
        hs = sorted({int(h) for h in raw.split(",") if h})
        if not hs:
            return []
        if self._ds is None:
            raise ValueError("extra forecast horizons need the dataset (the model must be bound to it)")
        bad = [h for h in hs if not 1 <= h < self._ds.horizon]
        if bad:
            raise ValueError(f"extra horizons {bad} must be at least 1 and shorter than the dataset's horizon of {self._ds.horizon} days")
        return hs

    def _aux_targets(self, index: pd.MultiIndex, horizons: list[int]) -> np.ndarray:
        """Cross-sectionally demeaned forward returns over each extra horizon, for these rows (complete whenever the primary label is, since they end sooner)."""
        cols = []
        t, a = self._ds.dates.get_indexer(index.get_level_values(0)), pd.Index(self._ds.assets).get_indexer(index.get_level_values(1))
        for h in horizons:
            fwd = forward_return(self._ds.prices, h, self._ds.lag).reindex(index=self._ds.dates, columns=self._ds.assets)
            wide = fwd.sub(fwd.mean(axis=1), axis=0).to_numpy(dtype=np.float32)
            cols.append(wide[t, a])
        return np.column_stack(cols) if cols else np.empty((len(index), 0), dtype=np.float32)

    # ------------------------------------------------------------------------------------------------------------ fitting
    @property
    def _per_head(self) -> int:
        return 3 if self.arch == "tft" else 1

    def _config(self, fit_key: str = "") -> TrainConfig:
        p = self.params
        folder = p.get("checkpoint_dir") or None
        if folder and fit_key:
            folder = str(Path(folder) / f"{self.kind}-{fit_key}")                       # one folder per fit: a walk-forward refits on other windows, and resuming from another window's weights would skip the training
        return TrainConfig(int(p["epochs"]), int(p["batch_size"]), float(p["learning_rate"]), float(p["weight_decay"]), int(p["patience"]), float(p["grad_clip"]), str(p["device"]), self.seed,
                           int(p.get("num_threads", 0)), folder, bool(p.get("resume", False)))

    def _split(self, pos: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Training rows, then (after a purge gap) the validation tail, by time."""
        upos = np.unique(pos)
        n_val = max(1, int(round(len(upos) * float(self.params["val_fraction"]))))
        start = upos[-n_val]
        purge = (self._ds.lag + self._ds.horizon) if self._ds is not None else 0
        train, val = np.flatnonzero(pos <= start - 1 - purge), np.flatnonzero(pos >= start)
        if len(train) < MIN_ROWS or len(val) < MIN_ROWS // 4:
            raise ValueError(f"too few rows for a network ({len(train)} to train on, {len(val)} to validate); use more history")
        cap = int(self.params["max_rows"])
        if 0 < cap < len(train):
            train = np.sort(rng.choice(train, cap, replace=False))
        if 0 < cap // 3 < len(val):
            val = np.sort(rng.choice(val, cap // 3, replace=False))
        return train, val

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "DeepModel":
        Xs = self._select(X)
        self.features_ = list(Xs.columns)
        self.n_train_ = len(Xs)
        self._base = None
        rng = np.random.default_rng(self.seed)
        target = self._prepare_y(y).astype(np.float32)
        horizons = self._horizons()
        aux = self._aux_targets(Xs.index, horizons)
        self.head_stats_ = [(float(np.nanmean(aux[:, i])), float(np.nanstd(aux[:, i])) or 1.0) for i in range(aux.shape[1])] + [(self.y_mean_, self.y_std_)]
        Y = np.column_stack([(aux[:, i] - self.head_stats_[i][0]) / self.head_stats_[i][1] for i in range(aux.shape[1])] + [target]).astype(np.float32)
        self.heads_ = horizons + [self._ds.horizon if self._ds is not None else 0]
        pos = np.asarray(dates) if dates is not None else pd.Series(Xs.index.get_level_values(0)).rank(method="dense").to_numpy()
        train, val = self._split(pos, rng)
        self.scaler_ = FeatureScaler().fit(Xs.to_numpy(dtype=np.float32)[train])
        src = _Source(self, Xs)
        device, note = resolve_device(str(self.params["device"]))
        n_out = len(self.heads_)
        self.net_spec_ = (self.arch, len(self.features_), int(self.params["lookback"]), n_out)
        torch.manual_seed(self.seed)
        net = build_net(self.arch, len(self.features_), int(self.params["lookback"]), n_out, self.params).to(device)
        rows = {"train": train, "val": val}
        finite = np.isfinite(Y)
        Yz = np.nan_to_num(Y)
        is_ae = self.arch in AUTOENCODERS

        def batch_fn(split, idx):
            r = rows[split][idx]
            raw = src.get(r)
            x = torch.from_numpy(self.scaler_.transform(raw)).to(device)
            if is_ae:
                return x, torch.from_numpy(np.isfinite(raw).astype(np.float32)).to(device)
            return x, torch.from_numpy(Yz[r]).to(device), torch.from_numpy(finite[r].astype(np.float32)).to(device)

        loss_fn = self._ae_loss if is_ae else (self._pinball_loss if self.arch == "tft" else self._mse_loss)
        key = hashlib.sha1(f"{pos.min()}|{pos.max()}|{len(train)}|{len(val)}|{self.seed}|{len(self.features_)}".encode()).hexdigest()[:10]       # the fit's training window, so that a resumed run is the same fit
        self.history_ = fit_network(net, batch_fn, loss_fn, len(train), len(val), self._config(key))
        self.history_.device_note = note
        self.net_ = net
        if is_ae:
            self._fit_readout(src, device, Y[:, -1])
        self.importance_ = self._saliency(src, rows["val"], device)
        self.n_windows_ = (int(len(train)), int(len(val)))
        return self

    # -- the losses
    def _mse_loss(self, net, batch):
        x, y, m = batch
        return (((net(x) - y) ** 2) * m).sum() / m.sum().clamp(min=1.0)

    def _pinball_loss(self, net, batch):
        x, y, m = batch
        out = net(x).view(x.shape[0], -1, 3)
        err = y.unsqueeze(-1) - out
        q = torch.tensor([float(self.params["alpha_low"]), 0.5, float(self.params["alpha_high"])], device=x.device)
        return ((torch.maximum(q * err, (q - 1.0) * err)) * m.unsqueeze(-1)).sum() / (m.sum().clamp(min=1.0) * 3.0)

    def _ae_loss(self, net, batch):
        x, valid = batch
        recon, mu, logvar = net(x)
        loss = (((recon - x) ** 2) * valid).sum() / valid.sum().clamp(min=1.0)
        if logvar is not None:
            kl = -0.5 * (1.0 + logvar - mu ** 2 - logvar.exp()).sum(dim=1).mean()
            loss = loss + float(self.params["beta"]) * kl / x[0].numel()
        return loss

    def _fit_readout(self, src: _Source, device, target: np.ndarray) -> None:
        """The score of an autoencoder: a ridge regression on the latent code, fitted on the same rows (the code itself never saw the target)."""
        idx = np.flatnonzero(np.isfinite(target))
        Z = self._encode(src, idx, device)
        self.readout_ = Ridge(alpha=float(self.params["ridge_alpha"])).fit(Z, target[idx])
        self.recon_error_ = float(self.history_.val_loss[self.history_.best_epoch])

    # ------------------------------------------------------------------------------------------------------------ inference
    @torch.no_grad()
    def _encode(self, src: _Source, idx: np.ndarray, device) -> np.ndarray:
        self.net_.eval()
        out = []
        for i in range(0, len(idx), 2048):
            x = torch.from_numpy(self.scaler_.transform(src.get(idx[i:i + 2048]))).to(device)
            out.append(self.net_.encode(x)[0].cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, int(self.params["latent_dim"])))

    @torch.no_grad()
    def _forward(self, X: pd.DataFrame, context: pd.DataFrame | None = None) -> dict:
        """All the model's outputs for the rows of ``X``, in the target's own units: ``score`` always; ``heads``, ``quantiles`` or ``latent`` when the architecture has them."""
        device, _ = resolve_device(str(self.params["device"]))
        self.net_.to(device).eval()
        src = _Source(self, X, context)
        n = len(X)
        if self.arch in AUTOENCODERS:
            Z = self._encode(src, np.arange(n), device)
            out = {"score": self.readout_.predict(Z) * self.y_std_ + self.y_mean_, "latent": Z}
        else:
            raw = np.concatenate([self.net_(torch.from_numpy(self.scaler_.transform(src.get(np.arange(i, min(i + 2048, n))))).to(device)).cpu().numpy() for i in range(0, n, 2048)]) if n else \
                np.empty((0, len(self.heads_) * self._per_head))
            k = len(self.heads_)
            raw = raw.reshape(n, k, self._per_head)
            scale = np.array([s for _, s in self.head_stats_])[None, :, None]
            shift = np.array([m for m, _ in self.head_stats_])[None, :, None]
            vals = raw * scale + shift
            if self.arch == "tft":
                vals = np.sort(vals, axis=2)
                out = {"score": vals[:, -1, 1], "heads": vals[:, :, 1], "quantiles": vals[:, -1, :]}
            else:
                out = {"score": vals[:, -1, 0], "heads": vals[:, :, 0]}
        return out

    def predict(self, X: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        return self.raw_predict(X, context)

    def raw_predict(self, X: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        return self._forward(X, context)["score"]

    def predict_extras(self, X: pd.DataFrame) -> dict:
        out, frames = self._forward(X), {}
        if "heads" in out and out["heads"].shape[1] > 1:
            frames["heads"] = pd.DataFrame(out["heads"], index=X.index, columns=[f"h{h}" for h in self.heads_])
        if "quantiles" in out:
            lo, mid, hi = out["quantiles"].T
            z = stats.norm.ppf(float(self.params["alpha_high"])) - stats.norm.ppf(float(self.params["alpha_low"]))
            frames["distribution"] = pd.DataFrame({"mean": mid, "std": (hi - lo) / z, "lower": lo, "upper": hi}, index=X.index)
        if "latent" in out:
            frames["latent"] = pd.DataFrame(out["latent"], index=X.index, columns=[f"z{i}" for i in range(out["latent"].shape[1])])
        return frames

    def latent_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """The latent code of each row (autoencoders only): the representation the score is read from, usable as features of another model."""
        if self.arch not in AUTOENCODERS:
            raise ValueError(f"{self.arch} has no latent code; use autoencoder or vae")
        return self.predict_extras(X)["latent"]

    # ------------------------------------------------------------------------------------------------------------ explanation
    def _saliency(self, src: _Source, val_rows: np.ndarray, device) -> np.ndarray:
        """Mean absolute gradient of the score with respect to each feature over a sample of validation windows: what the network responds to."""
        take = val_rows[:: max(1, len(val_rows) // 256)][:256]
        x = torch.from_numpy(self.scaler_.transform(src.get(take))).to(device).requires_grad_(True)
        self.net_.eval()
        with torch.backends.cudnn.flags(enabled=False):
            if self.arch in AUTOENCODERS:
                coef = torch.tensor(self.readout_.coef_, dtype=torch.float32, device=device)
                score = (self.net_.encode(x)[0] * coef).sum()
            else:
                out = self.net_(x).view(len(take), len(self.heads_), self._per_head)
                score = out[:, -1, 1 if self.arch == "tft" else 0].sum()
            score.backward()
        grad = x.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        if self.arch == "tft" and getattr(self.net_, "last_variables", None) is not None:
            grad = 0.5 * grad / (grad.sum() + 1e-12) + 0.5 * self.net_.last_variables.cpu().numpy()
        return grad / (grad.sum() + 1e-12)

    def importance(self) -> pd.Series:
        return pd.Series(self.importance_, index=self.features_)

    def diagnostics(self) -> dict:
        h = self.history_.to_dict()
        d = {"arch": self.arch, "lookback": int(self.params["lookback"]), "heads": list(self.heads_), "windows": list(self.n_windows_), **{k: h[k] for k in (
            "train_loss", "val_loss", "lr", "grad_norm", "best_epoch", "stopped_early", "device", "device_note", "n_params", "seconds", "peak_memory_mb", "epochs_run")}}
        if self.arch == "tft":
            d["interval_level"] = float(self.params["alpha_high"]) - float(self.params["alpha_low"])
            if getattr(self.net_, "last_attention", None) is not None:
                d["attention"] = [float(v) for v in self.net_.last_attention.cpu()]
        if self.arch in AUTOENCODERS:
            d["latent_dim"] = int(self.params["latent_dim"])
            d["reconstruction_loss"] = self.recon_error_
        return d

    # ------------------------------------------------------------------------------------------------------------ serialization
    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_ds=None, _base=None)
        if self.net_ is not None:
            buf = io.BytesIO()
            torch.save({k: v.cpu() for k, v in self.net_.state_dict().items()}, buf)
            state.update(net_=None, _state_bytes=buf.getvalue())
        return state

    def __setstate__(self, state):
        blob = state.pop("_state_bytes", None)
        self.__dict__.update(state)
        if blob is not None:
            arch, n_features, lookback, n_out = self.net_spec_
            net = build_net(arch, n_features, lookback, n_out, self.params)
            net.load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True))
            self.net_ = net.eval()


def build(kind: str, task: str, params: dict, seed: int) -> Model:
    if kind not in FORECASTERS and kind not in AUTOENCODERS:
        raise ValueError(f"unknown model '{kind}'")
    return DeepModel(kind, task, params, seed)


def _versions() -> dict:
    import sklearn

    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def save_model(model: Model, path: str | Path) -> dict:
    """Write a model to ``path`` (its serialized bytes) with ``path.json`` beside it: the kind, settings, SHA-256 of the bytes and the versions of the libraries. Returns the metadata."""
    path = Path(path)
    blob = model.to_bytes()
    meta = {"kind": model.kind, "params": model.params, "seed": model.seed, "features": list(model.features_), "n_train": int(model.n_train_), "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob), "versions": _versions()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def load_model(path: str | Path, verify: bool = True) -> Model:
    """Read a model written by ``save_model``; with ``verify`` the checksum in the metadata must match the bytes."""
    path = Path(path)
    blob = path.read_bytes()
    meta_path = path.with_suffix(path.suffix + ".json")
    if verify and meta_path.exists() and json.loads(meta_path.read_text())["sha256"] != hashlib.sha256(blob).hexdigest():
        raise ValueError(f"{path} does not match its checksum: the file was changed or damaged")
    return Model.from_bytes(blob)
