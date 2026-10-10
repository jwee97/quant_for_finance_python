"""Deep forecasters as plug-in models: window networks refit walk-forward, and a zero-shot foundation model.

``deep_window`` runs one of the Generation 4/5 networks (patch transformer, MLP-mixer, N-BEATS, N-HiTS, TimeMixer-style, Temporal Fusion Transformer) on the 252 daily returns before each
month-end origin, each divided by the EWMA daily volatility at the origin. The target is the next 21-day return in the same units (a z-score), so the forecast
is ``mu_z * sigma`` and only the conditional mean is learned. The network is refit once a year on origins whose label (plus a 21-day embargo) was complete by the
refit date; the last fifth of those origins (after a 42-day gap) is the early-stopping block.

``chronos`` feeds the same window to Chronos-Bolt with no fitting at all (optional dependency ``chronos-forecasting``; the weights come from the Hugging Face hub).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.sleeves import month_end_dates
from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..framework.types import ForecastPanel
from ..models.deep_forecast import TrainSettings, daily_sigma, normalised_windows, train_and_predict
from ..models.probabilistic import ewma_sigma

KINDS = ("patchtst", "tsmixer", "nbeats", "nhits", "timemixer", "tft", "fnn", "cnn", "lstm", "gru", "transformer")
WINDOW = 252


def _origin_rows(data, origins: pd.DatetimeIndex, horizon: int, halflife: float = 40.0):
    """Every (origin, asset) row: normalised window, validity, z-score label (NaN while unmatured at the end of the sample), and the origin's sigma."""
    index, assets = data.index, list(data.assets)
    position = pd.Series(np.arange(len(index)), index=index)
    pos = np.repeat(position.reindex(origins).to_numpy(), len(assets))
    asset_idx = np.tile(np.arange(len(assets)), len(origins))
    sigma_d = daily_sigma(data.returns, halflife)
    X, valid = normalised_windows(data.returns, sigma_d, pos, asset_idx, WINDOW)
    sigma_h = ewma_sigma(data.returns, halflife, horizon).to_numpy()
    forward = (data.prices.shift(-horizon) / data.prices - 1.0).to_numpy()
    sigma = sigma_h[pos, asset_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.clip(forward[pos, asset_idx] / np.where(sigma > 0, sigma, np.nan), -5.0, 5.0)
    live = data.investable.to_numpy()[pos, asset_idx]
    return pos, asset_idx, X, valid & live & np.isfinite(sigma), z, sigma


def _panel(model, data, origins, mean: np.ndarray, halflife: float) -> ForecastPanel:
    """Month-end forecasts held until the next origin, with the EWMA volatility as the spread."""
    monthly = pd.DataFrame(mean.reshape(len(origins), len(data.assets)), index=origins, columns=list(data.assets))
    out = monthly.reindex(data.index).ffill().where(data.investable)
    std = ewma_sigma(data.returns, halflife, model.horizon).reindex_like(out)
    return ForecastPanel.from_mean_std(out, std.where(out.notna()), model.horizon, model.name)


@register_model("deep_window", "machine learning", "A small neural network (patch transformer, MLP-mixer, N-BEATS, N-HiTS, TimeMixer-style, Temporal Fusion Transformer, or a plain feed-forward, convolutional, LSTM, GRU or Transformer network) on the volatility-normalised 252-day return window, refit yearly")
class DeepWindow(ForecastModel):
    """If a flexible sequence model finds structure in the last year of returns that a ridge on twelve features cannot, it should beat the ridge out of sample.
    Stage 27 and 33 found it did not; the plug-in makes that test repeatable on any bundle."""

    name, family = "deep_window", "machine learning"

    def __init__(self, kind: str = "nbeats", min_train: int = 1260, refit_every: int = 252, embargo: int = 21, max_epochs: int = 15, patience: int = 3,
                 seeds: tuple = (0,), d_model: int = 16):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if min_train < WINDOW + 252 or refit_every < 21 or embargo < 0 or max_epochs < 1 or patience < 1 or not seeds or d_model < 4:
            raise ValueError("min_train >= 504, refit_every >= 21, embargo >= 0, max_epochs >= 1, patience >= 1, at least one seed, d_model >= 4")
        self.kind, self.min_train, self.refit_every, self.embargo = kind, min_train, refit_every, embargo
        self.max_epochs, self.patience, self.seeds, self.d_model = max_epochs, patience, tuple(seeds), d_model

    def forecast(self, data, min_observations: int = 504, allow_negative: bool = False, halflife: float = 40.0) -> ForecastPanel:
        h, index = self.horizon, data.index
        origins = month_end_dates(index)
        pos, asset_idx, X, ok, z, sigma = _origin_rows(data, origins, h)
        settings = TrainSettings(max_epochs=self.max_epochs, patience=self.patience)
        net_kwargs = {"window": WINDOW, "d_model": self.d_model, "ff": 2 * self.d_model, "heads": 2 if self.d_model % 2 == 0 else 1}
        mean = np.full(len(pos), np.nan)
        for start in range(self.min_train, len(index), self.refit_every):
            usable = (pos + h + self.embargo <= start) & ok & np.isfinite(z)
            test = (pos >= start) & (pos < start + self.refit_every) & ok
            days = np.unique(pos[usable])
            if len(days) < 24 or not test.any():
                continue
            cut = days[int(np.floor(0.8 * len(days)))]
            train, val = usable & (pos <= cut - 42), usable & (pos > cut)
            if train.sum() < 100 or val.sum() < 20:
                continue
            preds = [train_and_predict(self.kind, X[train], asset_idx[train], z[train], X[val], asset_idx[val], z[val], X[test], asset_idx[test], int(seed),
                                       len(data.assets), net_kwargs, settings)["pred"] for seed in self.seeds]
            mean[test] = np.mean(preds, axis=0) * sigma[test]
        return _panel(self, data, origins, mean, halflife)

    def score(self, data):
        return self.forecast(data).mean


def _zero_shot(model, data, predict, batch: int, min_history: int, halflife: float = 40.0) -> ForecastPanel:
    """Feed the normalised window to ``predict(windows) -> summed 21-day path in normalised units`` for every month-end origin with enough history."""
    origins = month_end_dates(data.index)
    pos, _, X, ok, _, sigma = _origin_rows(data, origins, model.horizon)
    ok &= pos >= min_history
    mean = np.full(len(pos), np.nan)
    rows = np.flatnonzero(ok)
    if len(rows):
        paths = [np.asarray(predict(X[rows[i:i + batch]]), dtype=float) for i in range(0, len(rows), batch)]
        mean[rows] = np.concatenate(paths) * sigma[rows]
    return _panel(model, data, origins, mean, halflife)


@register_model("chronos", "machine learning", "Chronos-Bolt foundation model used zero-shot on the volatility-normalised 252-day return window (optional dependency, weights from the Hugging Face hub)")
class ChronosZeroShot(ForecastModel):
    """A pre-trained time-series model needs no fitting, so there is nothing to leak. Its pre-training corpus overlaps the sample and may contain these prices,
    so a good result here is weaker evidence than a good result from a model trained walk-forward."""

    name, family = "chronos", "machine learning"

    def __init__(self, model: str = "amazon/chronos-bolt-small", batch: int = 256, min_history: int = WINDOW + 63):
        if batch < 1 or min_history < WINDOW:
            raise ValueError("batch >= 1 and min_history >= 252")
        self.model, self.batch, self.min_history = model, batch, min_history

    def forecast(self, data, min_observations: int = 504, allow_negative: bool = False, halflife: float = 40.0) -> ForecastPanel:
        try:
            import torch
            from chronos import ChronosBoltPipeline
        except ImportError as error:
            raise ImportError("the 'chronos' model needs its optional dependencies: pip install -e '.[chronos]'") from error
        pipeline = ChronosBoltPipeline.from_pretrained(self.model, device_map="cpu", dtype=torch.float32)

        def predict(windows):
            q, _ = pipeline.predict_quantiles(torch.as_tensor(windows, dtype=torch.float32), prediction_length=self.horizon, quantile_levels=[0.5])
            return q[:, :, 0].sum(dim=1).numpy()

        return _zero_shot(self, data, predict, self.batch, self.min_history, halflife)

    def score(self, data):
        return self.forecast(data).mean


@register_model("timesfm", "machine learning", "TimesFM 2.5 (200M) foundation model used zero-shot on the volatility-normalised 252-day return window (optional dependency, weights from the Hugging Face hub)")
class TimesFMZeroShot(ForecastModel):
    """Same protocol and same caveat as ``chronos``: nothing is fitted, and the pre-training corpus may overlap the sample."""

    name, family = "timesfm", "machine learning"

    def __init__(self, model: str = "google/timesfm-2.5-200m-pytorch", batch: int = 64, min_history: int = WINDOW + 63):
        if batch < 1 or min_history < WINDOW:
            raise ValueError("batch >= 1 and min_history >= 252")
        self.model, self.batch, self.min_history = model, batch, min_history

    def forecast(self, data, min_observations: int = 504, allow_negative: bool = False, halflife: float = 40.0) -> ForecastPanel:
        try:
            import timesfm
        except ImportError as error:
            raise ImportError("the 'timesfm' model needs the optional dependency: pip install 'timesfm[torch]'") from error
        net = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model)
        net.compile(timesfm.ForecastConfig(max_context=256, max_horizon=self.horizon, normalize_inputs=True, use_continuous_quantile_head=False,
                                           force_flip_invariance=False, infer_is_positive=False, fix_quantile_crossing=False))

        def predict(windows):
            point, _ = net.forecast(horizon=self.horizon, inputs=[w.astype("float32") for w in windows])
            return point[:, :self.horizon].sum(axis=1)

        return _zero_shot(self, data, predict, self.batch, self.min_history, halflife)

    def score(self, data):
        return self.forecast(data).mean
