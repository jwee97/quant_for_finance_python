"""What every forecaster shares: a price history in, a predictive distribution of the next days' return out.

A forecaster does not say "the price will be 101". It says how the price might move: ``Forecast.samples`` holds ``n`` simulated cumulative log returns from the forecast origin (``samples[j, k - 1]`` is
``log(P[T + k] / P[T])`` in draw ``j``), for every step ``k`` up to the longest horizon asked for. Quantiles of a column are the fan of the price, and the same draws are scored against what happened
(``forecast.scoring``). A method that simulates paths (GARCH, a Markov chain) gives rows that are coherent paths; a method that models one horizon at a time (the Gaussian process on features, the
analogues) gives the horizons asked for exactly and fills the days between by interpolating the quantile function (``fill_between``), so its fan is smooth but its rows are not paths.

Causality is structural. ``fit`` and ``predict`` are handed a ``History`` that ends at the forecast origin, so there is no future in it to leak; the evaluation (``forecast.evaluate``) cuts the data that way
and the tests replace everything after the origin with noise to check that no forecast changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd


class ForecastError(ValueError):
    """A method could not be fitted or could not forecast from this history (too short, constant, a model that did not converge)."""


@dataclass(frozen=True)
class History:
    """Daily prices up to a forecast origin: the adjusted close, and the day's high and low where there are some (a range-based volatility estimate needs them)."""

    close: np.ndarray
    index: pd.DatetimeIndex | None = None
    high: np.ndarray | None = None
    low: np.ndarray | None = None

    def __post_init__(self):
        close = np.asarray(self.close, dtype=float)
        if close.ndim != 1 or len(close) < 3 or not np.all(np.isfinite(close)) or np.any(close <= 0):
            raise ForecastError("a history is a one-dimensional run of at least three finite, positive prices")
        object.__setattr__(self, "close", close)
        for name in ("high", "low"):
            side = getattr(self, name)
            if side is not None:
                side = np.asarray(side, dtype=float)
                if side.shape != close.shape:
                    raise ForecastError(f"{name} must have as many days as the close")
                object.__setattr__(self, name, side)
        if self.index is not None and len(self.index) != len(close):
            raise ForecastError("the dates must match the prices")

    @classmethod
    def from_frame(cls, close: pd.Series, high: pd.Series | None = None, low: pd.Series | None = None) -> "History":
        """From pandas series on the same dates (the high and low are used only where they cover every day and bracket the close)."""
        close = close.dropna()
        hi = lo = None
        if high is not None and low is not None:
            hi, lo = high.reindex(close.index), low.reindex(close.index)
            if hi.isna().any() or lo.isna().any() or (hi < lo).any() or (hi <= 0).any() or (lo <= 0).any():
                hi = lo = None
        return cls(close.to_numpy(float), close.index, None if hi is None else hi.to_numpy(float), None if lo is None else lo.to_numpy(float))

    @property
    def n(self) -> int:
        return len(self.close)

    @cached_property
    def r(self) -> np.ndarray:
        """Daily log returns (one fewer than the prices)."""
        return np.diff(np.log(self.close))

    def upto(self, i: int) -> "History":
        """The history ending at position ``i`` (inclusive): what was known on that day."""
        return self.slice(None, i + 1)

    def slice(self, start: int | None, stop: int | None) -> "History":
        s = slice(start, stop)
        return History(self.close[s], None if self.index is None else self.index[s], None if self.high is None else self.high[s], None if self.low is None else self.low[s])

    def tail(self, n: int) -> "History":
        return self.slice(max(0, self.n - n), None)


@dataclass
class Forecast:
    """``samples``: ``(n_paths, steps)`` cumulative log returns at steps 1 to ``steps``. ``info``: what the method learned (its fitted numbers), for the report."""

    samples: np.ndarray
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.ndim != 2 or not np.all(np.isfinite(self.samples)):
            raise ForecastError("a forecast is a finite (paths, steps) array")

    @property
    def steps(self) -> int:
        return self.samples.shape[1]

    def at(self, horizon: int) -> np.ndarray:
        """The draws of the ``horizon``-day return."""
        return self.samples[:, horizon - 1]


class Forecaster:
    """A way of forecasting. Subclasses say how to ``fit`` on a history and how to ``predict`` from a fitted state and a (possibly later) history. ``fit`` is the expensive part and the evaluation repeats it only every
    few weeks; ``predict`` brings the state up to the origin with the fitted numbers held fixed, so a forecast uses every day up to the origin and no fit uses a day after it."""

    name = "base"
    short = "A forecaster"         # for a chart label
    label = "A forecaster"
    family = "other"
    text = ""                      # one sentence for the form
    assumes = ""                   # what it takes the market to be like
    fails = ""                     # where it breaks
    mean_signal = False            # does it forecast a non-zero average return (a direction), or only the spread?
    needs_range = False            # uses the day's high and low when the data have them
    heavy = False                  # a fit that takes seconds: the evaluation refits it less often
    min_history = 500              # days of prices needed
    window = 1260                  # days a fit looks at (about five years)
    lookback = 504                 # days a forecast brings its state up to date over

    def fit(self, h: History, horizons: tuple[int, ...]):
        """Estimate what the method needs from ``h`` (the last ``window`` days up to the origin); ``horizons`` are the horizons that will be asked for (a method that learns one horizon at a time needs them)."""
        raise NotImplementedError

    def predict(self, state, h: History, horizons: tuple[int, ...], n: int, rng: np.random.Generator) -> Forecast:
        raise NotImplementedError

    def view(self) -> dict:
        return {"id": self.name, "label": self.label, "short": self.short, "family": self.family, "text": self.text, "assumes": self.assumes, "fails": self.fails, "mean_signal": self.mean_signal, "heavy": self.heavy,
                "min_history": self.min_history}


# ---------------------------------------------------------------------------------------------------------------------------------- helpers shared by the methods
def standard_t(rng: np.random.Generator, shape, nu: float) -> np.ndarray:
    """Student-t draws scaled to unit variance (``nu`` > 2); the normal as ``nu`` grows."""
    if not np.isfinite(nu) or nu > 300:
        return rng.standard_normal(shape)
    return rng.standard_t(nu, shape) * np.sqrt((nu - 2.0) / nu)


def cumulate(returns: np.ndarray) -> np.ndarray:
    """Daily returns ``(n, steps)`` to cumulative returns from the origin."""
    return np.cumsum(returns, axis=1)


def fill_between(anchors: dict[int, np.ndarray], steps: int) -> np.ndarray:
    """Draws of the cumulative return at every step from draws at a few horizons, for methods that model one horizon at a time. ``anchors`` maps a horizon to ``n`` draws of its return. The draws of each
    horizon are sorted and the days between two horizons interpolate the two quantile functions linearly in time (and start from zero at the origin), so row ``j`` is the ``j``-th quantile at every step: a
    smooth fan, not a simulated path."""
    if not anchors:
        raise ForecastError("no horizon to forecast")
    hs = sorted(anchors)
    if hs[-1] < steps:
        raise ForecastError("the longest horizon of a method must reach the last step")
    n = len(anchors[hs[0]])
    cols = {h: np.sort(np.asarray(anchors[h], dtype=float)) for h in hs}
    out = np.empty((n, steps))
    left_h, left = 0, np.zeros(n)
    for h in hs:
        right = cols[h]
        for k in range(left_h + 1, min(h, steps) + 1):
            w = (k - left_h) / (h - left_h)
            out[:, k - 1] = left + w * (right - left)
        left_h, left = h, right
        if h >= steps:
            break
    return out


def trailing_rms(r: np.ndarray, days: int) -> float:
    """Root mean square of the last ``days`` returns (zero-mean volatility): what a forecast with no view on direction should use for the spread."""
    x = r[-days:]
    if len(x) < 20:
        raise ForecastError("too few days to measure volatility")
    return float(np.sqrt(np.mean(x * x)))
