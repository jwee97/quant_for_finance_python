"""Feature windows from a (date, asset) panel, gathered without look-ahead.

A sequence model reads, for each row, the last ``lookback`` days of that asset's features, the row's own day included. ``SeqAccessor`` holds the panel as a dense ``[date, asset, feature]`` array
and gathers those windows batch by batch (a window is index arithmetic on the array, so the whole set of windows is never built in memory). A window only ever reaches backwards, and positions
before the first date are padded with missing values, which the scaler turns into zeros.

When a model is bound to a dataset the array holds the dataset's whole feature history, so the first days of a test block still have their lookback; rows handed to ``predict`` are overlaid on it
(so a stress test that perturbs the features is felt by the windows). Without a dataset the array is built from the rows given.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class FeatureScaler:
    """Standardises features with the mean and spread of the TRAINING rows, clips at +-``clip`` and turns missing values into 0 (the training mean)."""

    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.mu_ = self.sd_ = None

    def fit(self, values: np.ndarray) -> "FeatureScaler":
        v = np.asarray(values, dtype=np.float64)
        mu, sd = np.nanmean(v, axis=0), np.nanstd(v, axis=0)
        self.mu_ = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
        self.sd_ = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0).astype(np.float32)
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        out = np.clip((arr - self.mu_) / self.sd_, -self.clip, self.clip)
        return np.nan_to_num(out, nan=0.0).astype(np.float32, copy=False)


@dataclass
class Overlay:
    xpos: np.ndarray                    # [date, asset] -> row number in ``values`` (-1 where the overlay has no row)
    values: np.ndarray                  # [rows, feature]


class SeqAccessor:
    def __init__(self, axis: pd.DatetimeIndex, assets: list, arr: np.ndarray | None, n_features: int):
        self.axis, self.assets, self.arr, self.n_features = axis, list(assets), arr, int(n_features)
        self._asset_index = pd.Index(self.assets)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, axis: pd.DatetimeIndex | None = None, assets: list | None = None) -> "SeqAccessor":
        dates, names = frame.index.get_level_values(0), frame.index.get_level_values(1)
        axis = axis if axis is not None else pd.DatetimeIndex(sorted(dates.unique()))
        assets = list(assets) if assets is not None else sorted(names.unique())
        t, a = axis.get_indexer(dates), pd.Index(assets).get_indexer(names)
        ok = (t >= 0) & (a >= 0)
        arr = np.full((len(axis), len(assets), frame.shape[1]), np.nan, dtype=np.float32)
        arr[t[ok], a[ok]] = frame.to_numpy(dtype=np.float32)[ok]
        return cls(axis, assets, arr, frame.shape[1])

    def locate(self, index: pd.MultiIndex) -> tuple[np.ndarray, np.ndarray]:
        t, a = self.axis.get_indexer(index.get_level_values(0)), self._asset_index.get_indexer(index.get_level_values(1))
        if (t < 0).any() or (a < 0).any():
            raise ValueError("some rows are on a date or for an asset the model's dataset does not have")
        return t, a

    def overlay(self, frame: pd.DataFrame) -> Overlay:
        t, a = self.axis.get_indexer(frame.index.get_level_values(0)), self._asset_index.get_indexer(frame.index.get_level_values(1))
        ok = (t >= 0) & (a >= 0)
        xpos = np.full((len(self.axis), len(self.assets)), -1, dtype=np.int64)
        xpos[t[ok], a[ok]] = np.arange(len(frame))[ok]
        return Overlay(xpos, frame.to_numpy(dtype=np.float32))

    def windows(self, t: np.ndarray, a: np.ndarray, lookback: int, overlay: Overlay | None = None) -> np.ndarray:
        """Raw ``[len(t), lookback, feature]`` windows ending at (t, a); missing and pre-history steps are NaN."""
        steps = t[:, None] + np.arange(-lookback + 1, 1)[None, :]
        before = steps < 0
        idx = np.clip(steps, 0, None)
        cols = a[:, None]
        out = self.arr[idx, cols] if self.arr is not None else np.full((len(t), lookback, self.n_features), np.nan, dtype=np.float32)
        if overlay is not None:
            rows = overlay.xpos[idx, cols]
            use = rows >= 0
            out = out.copy() if self.arr is not None else out
            out[use] = overlay.values[rows[use]]
        out[before] = np.nan
        return out


@dataclass
class SequenceData:
    """Windows, targets and the row index of a small panel, as arrays: ``X [rows, lookback, feature]`` and ``y [rows]``."""

    X: np.ndarray
    y: np.ndarray
    index: pd.MultiIndex
    features: list


def sequence_dataset(X: pd.DataFrame, y: pd.Series | None, lookback: int, scaler: FeatureScaler | None = None) -> SequenceData:
    """The windows of every row of ``X`` (earlier rows of the same asset in ``X`` fill the lookback), scaled with ``scaler`` (fitted here on ``X`` when not given). For a panel that fits in memory;
    the models gather their windows batch by batch instead."""
    acc = SeqAccessor.from_frame(X)
    t, a = acc.locate(X.index)
    raw = acc.windows(t, a, int(lookback))
    scaler = scaler or FeatureScaler().fit(X.to_numpy(dtype=np.float64))
    return SequenceData(scaler.transform(raw), np.full(len(X), np.nan) if y is None else y.to_numpy(dtype=np.float32), X.index, list(X.columns))
