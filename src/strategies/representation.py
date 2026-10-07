"""A forecast from a learned representation: an unsupervised encoder on return windows, then a ridge on the embedding (see ``models.representation``)."""

from __future__ import annotations

import numpy as np

from ..features.sleeves import month_end_dates
from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..framework.types import ForecastPanel
from ..models.representation import KINDS, train_encoder
from .deep import WINDOW, _origin_rows, _panel


@register_model("deep_representation", "machine learning", "Autoencoder or contrastive encoder trained WITHOUT labels on volatility-normalised return windows, then a ridge from the embedding to the next 21-day return")
class DeepRepresentation(ForecastModel):
    """Labels are scarce but windows are plentiful: learn what a year of returns typically looks like, then ask whether where a window sits in that learned space predicts the next month."""

    name, family = "deep_representation", "machine learning"

    def __init__(self, kind: str = "autoencoder", embed_dim: int = 8, hidden: int = 64, epochs: int = 15, ridge_alpha: float = 100.0, min_train: int = 1260, refit_every: int = 252,
                 embargo: int = 21, seed: int = 0):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if embed_dim < 2 or hidden < embed_dim or epochs < 1 or ridge_alpha < 0 or min_train < WINDOW + 252 or refit_every < 21 or embargo < 0:
            raise ValueError("embed_dim >= 2, hidden >= embed_dim, epochs >= 1, ridge_alpha >= 0, min_train >= 504, refit_every >= 21, embargo >= 0")
        self.kind, self.embed_dim, self.hidden, self.epochs, self.ridge_alpha = kind, embed_dim, hidden, epochs, ridge_alpha
        self.min_train, self.refit_every, self.embargo, self.seed = min_train, refit_every, embargo, seed

    def forecast(self, data, min_observations: int = 504, allow_negative: bool = False, halflife: float = 40.0) -> ForecastPanel:
        from sklearn.linear_model import Ridge

        h, index = self.horizon, data.index
        origins = month_end_dates(index)
        pos, asset_idx, X, ok, z, sigma = _origin_rows(data, origins, h)
        mean = np.full(len(pos), np.nan)
        for start in range(self.min_train, len(index), self.refit_every):
            unlabelled = (pos <= start) & ok                                         # every window that ends on or before the refit date: no labels, so no leakage
            labelled = (pos + h + self.embargo <= start) & ok & np.isfinite(z)
            test = (pos >= start) & (pos < start + self.refit_every) & ok
            if unlabelled.sum() < 200 or labelled.sum() < 100 or not test.any():
                continue
            enc = train_encoder(X[unlabelled], self.kind, self.embed_dim, self.hidden, self.epochs, seed=self.seed)["encode"]
            E_l, E_t = enc(X[labelled]), enc(X[test])
            mu, sd = E_l.mean(axis=0), np.where(E_l.std(axis=0) > 0, E_l.std(axis=0), 1.0)
            assets = np.eye(len(data.assets))[asset_idx]
            Z_l = np.hstack([(E_l - mu) / sd, assets[labelled]])
            Z_t = np.hstack([(E_t - mu) / sd, assets[test]])
            model = Ridge(alpha=self.ridge_alpha).fit(Z_l, z[labelled])
            mean[test] = model.predict(Z_t) * sigma[test]
        return _panel(self, data, origins, mean, halflife)

    def score(self, data):
        return self.forecast(data).mean
