"""Online learning as a forecast model: train, trade, update, trade, update.

The batch ``ml_ridge`` refits once a year. This model instead updates a linear forecast every month, from the labels that have just
matured, with an exponential forgetting factor so the coefficients can follow a drifting relationship. It is the plug-in form of the
Generation 3 updaters in ``src.models.online`` (recursive ridge, normalised LMS, random-walk Kalman).

Causality: at month-end origin ``t`` the coefficients have absorbed only origins ``s`` with ``s + horizon <= t`` (their 21-day return is
complete), and features are standardised with statistics from rows up to ``t`` only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.macro import expanding_zscore
from ..features.sleeves import month_end_dates
from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..framework.types import ForecastPanel
from ..models.online import NLMS, KalmanRW, OnlineRidge
from ..models.probabilistic import ewma_sigma, price_features

UPDATERS = ("ridge", "nlms", "kalman")


@register_model("online_ridge", "machine learning", "Monthly-updated linear forecast of the next 21-day return (recursive ridge, NLMS or Kalman coefficients) that follows a drifting signal")
class OnlineLinear(ForecastModel):
    """If the link between price features and the next return drifts, a model that forgets old months adapts faster than one refit yearly."""

    name, family = "online_ridge", "machine learning"

    def __init__(self, updater: str = "ridge", forgetting: float = 0.97, ridge: float = 50.0, step: float = 0.05, process_to_observation: float = 1e-3,
                 min_updates: int = 36, features: str = "price"):
        if updater not in UPDATERS:
            raise ValueError(f"updater must be one of {UPDATERS}")
        if features not in ("price", "price_macro"):
            raise ValueError("features must be 'price' or 'price_macro'")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        if ridge < 0 or step <= 0 or process_to_observation <= 0 or min_updates < 1:
            raise ValueError("ridge must be >= 0; step, process_to_observation > 0; min_updates >= 1")
        self.updater, self.forgetting, self.ridge, self.step = updater, forgetting, ridge, step
        self.process_to_observation, self.min_updates, self.features = process_to_observation, min_updates, features

    def _learner(self, dim: int):
        if self.updater == "ridge":
            return OnlineRidge(dim, self.forgetting, self.ridge)
        if self.updater == "nlms":
            return NLMS(dim, self.step)
        return KalmanRW(dim, self.process_to_observation)

    def _rows(self, data, origins: pd.DatetimeIndex) -> pd.DataFrame:
        frames = {n: f.loc[origins] for n, f in price_features(data.prices).items()}
        if self.features == "price_macro":
            z = expanding_zscore(data.macro.dropna(axis=1, how="all"), 504, 4.0).reindex(origins)
            for c in z.columns:
                frames[f"macro_{c}"] = pd.DataFrame({a: z[c] for a in data.prices.columns})
        return pd.concat({n: f.stack(future_stack=True) for n, f in frames.items()}, axis=1)

    def forecast(self, data, min_observations: int = 504, allow_negative: bool = False, halflife: float = 40.0) -> ForecastPanel:
        self.require(data)
        h = self.horizon
        index = data.index
        origins = month_end_dates(index)
        X = self._rows(data, origins)
        sigma = ewma_sigma(data.returns, 40.0, h)
        forward = (data.prices.shift(-h) / data.prices - 1.0)
        z = (forward / sigma.where(sigma > 0)).clip(-5, 5)
        position = pd.Series(np.arange(len(index)), index=index)
        origin_pos = position.reindex(origins).to_numpy()
        dim = X.shape[1] + 1
        learner = self._learner(dim)
        n_seen, total, total_sq = 0, np.zeros(dim - 1), np.zeros(dim - 1)
        pending: list[tuple[int, np.ndarray, np.ndarray]] = []            # (origin position, standardised rows, standardised labels) awaiting maturity
        mean = pd.DataFrame(np.nan, index=origins, columns=data.assets)
        updates = 0
        for k, origin in enumerate(origins):
            rows = X.xs(origin, level=0).reindex(data.assets)
            ok = rows.notna().all(axis=1).to_numpy()
            values = rows.to_numpy(dtype=float)
            if ok.any():                                                      # running feature statistics: rows up to and including today
                total += values[ok].sum(axis=0)
                total_sq += (values[ok] ** 2).sum(axis=0)
                n_seen += int(ok.sum())
            mu = total / max(n_seen, 1)
            sd = np.sqrt(np.maximum(total_sq / max(n_seen, 1) - mu ** 2, 0.0))
            sd = np.where(sd > 1e-12, sd, 1.0)
            std_rows = np.column_stack([(values - mu) / sd, np.ones(len(values))])
            matured = [item for item in pending if item[0] + h <= origin_pos[k]]
            if matured:                                                       # absorb the labels that are now fully realised (one update per month)
                pending = pending[len(matured):]
                xs, ys = [], []
                for _, rows_j, targets in matured:
                    xs.append(rows_j)
                    ys.append(targets)
                x, y = np.vstack(xs), np.concatenate(ys)
                if len(y):
                    learner.update(x, y)
                    updates += 1
            if updates >= self.min_updates and ok.any():
                prediction = np.asarray(learner.predict(std_rows)).ravel()
                mean.loc[origin, rows.index[ok]] = prediction[ok] * sigma.loc[origin, rows.index[ok]].to_numpy()
            target = z.loc[origin].reindex(data.assets).to_numpy(dtype=float)
            usable = ok & np.isfinite(target)
            # the label is stored now but only fed to the learner once it has matured
            pending.append((int(origin_pos[k]), std_rows[usable], target[usable]))
        out = mean.reindex(index).ffill().where(data.investable)
        std = ewma_sigma(data.returns, halflife, h).reindex_like(out)
        return ForecastPanel.from_mean_std(out, std.where(out.notna()), h, self.name)

    def score(self, data):
        return self.forecast(data).mean
