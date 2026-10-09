"""A stylised single-stock market for testing execution algorithms, and the order an algorithm has to work.

There is no tick data in this repository, so every algorithm in :mod:`src.algo` is tried on a *model* of a trading day: ``n`` equal intervals with a U-shaped volume profile (busy at the open and close,
quiet at midday), a volatility profile, a quoted spread, and the market's response to our own trading (the impact model in :mod:`src.algo.impact`). Nothing here is fitted to a real stock: the defaults
are round numbers in the range the academic literature reports for liquid US equities, and every one of them is a parameter. Treat results as a comparison of *methods* under stated assumptions, not as a
forecast of what an order will cost.

Units. Shares are shares; prices are dollars; ``sigma`` is the DAILY volatility as a fraction of price; ``spread`` is the quoted spread as a fraction of price; a *bps* is a hundredth of a percent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def u_shape(n: int, strength: float = 2.0) -> np.ndarray:
    """``n`` fractions summing to one, high at both ends and low in the middle: ``1 + strength (2t - 1)^2`` at the interval midpoints ``t``. ``strength = 0`` is flat."""
    t = (np.arange(n) + 0.5) / n
    raw = 1.0 + strength * (2.0 * t - 1.0) ** 2
    return raw / raw.sum()


@dataclass(frozen=True)
class Market:
    """One stock's market for ``days`` trading days of ``n`` intervals each.

    ``eta, beta`` set the temporary (while-trading) impact ``eta * sigma * (q / V)^beta`` as a fraction of price for a slice of ``q`` shares when ``V`` shares trade in the interval (Almgren, Thum, Hauptmann
    and Li 2005 report ``eta = 0.142`` and ``beta = 0.6`` for US stocks). ``gamma`` sets the permanent impact ``gamma * sigma * q / adv`` that stays in the price. ``volume_noise`` is the standard deviation
    of the log of an interval's volume around its profile and ``day_noise`` that of the whole day's volume around ``adv`` (both shock the volume a VWAP schedule did not expect).
    """

    price: float = 50.0
    adv: float = 2_000_000.0
    sigma: float = 0.02
    spread: float = 0.0004
    n: int = 26
    days: int = 1
    eta: float = 0.142
    beta: float = 0.6
    gamma: float = 0.30
    volume_noise: float = 0.25
    day_noise: float = 0.15
    u_strength: float = 2.0
    volume_fractions: tuple | None = None            # your own intraday volume profile (fractions per interval, summing to one); default the U shape
    variance_fractions: tuple | None = None          # your own intraday variance profile; default a milder U

    def __post_init__(self):
        if self.price <= 0 or self.adv <= 0 or self.sigma <= 0 or self.spread < 0 or self.n < 2 or self.days < 1:
            raise ValueError("price, adv, sigma > 0; spread >= 0; n >= 2; days >= 1")
        if self.eta < 0 or self.beta <= 0 or self.gamma < 0 or self.volume_noise < 0 or self.day_noise < 0:
            raise ValueError("eta >= 0, beta > 0, gamma >= 0, noise >= 0")
        for name in ("volume_fractions", "variance_fractions"):
            given = getattr(self, name)
            if given is not None and (len(given) != self.n or min(given) < 0 or abs(sum(given) - 1.0) > 1e-6):
                raise ValueError(f"{name} needs {self.n} non-negative numbers that sum to one")

    # ------------------------------------------------------------------------------------------------------------------ profiles
    @property
    def intervals(self) -> int:
        return self.n * self.days

    def volume_profile(self) -> np.ndarray:
        """Expected fraction of a day's volume in each interval of ONE day."""
        return np.asarray(self.volume_fractions, float) if self.volume_fractions is not None else u_shape(self.n, self.u_strength)

    def variance_profile(self) -> np.ndarray:
        """Fraction of a day's price variance in each interval of one day (the volatility U is milder than the volume U)."""
        return np.asarray(self.variance_fractions, float) if self.variance_fractions is not None else u_shape(self.n, 0.5 * self.u_strength)

    def expected_volume(self) -> np.ndarray:
        """Shares expected to trade in each of the ``intervals`` intervals, others' trading only."""
        return np.tile(self.volume_profile(), self.days) * self.adv

    def variance_weights(self) -> np.ndarray:
        """Each interval's share of ONE day's variance, tiled over the days (it sums to ``days``)."""
        return np.tile(self.variance_profile(), self.days)

    def tau(self) -> float:
        """The length of an interval in days."""
        return 1.0 / self.n


@dataclass(frozen=True)
class Order:
    """A parent order: ``side`` +1 buys and -1 sells ``shares`` over the first ``horizon`` intervals (default: all of them) starting at interval ``start``, measured from ``arrival`` (default: the market price).

    ``max_participation`` caps the share of an interval's volume the algorithm may be of the total (ours included): a wish above it waits for the next interval.
    """

    side: int = 1
    shares: float = 200_000.0
    arrival: float | None = None
    start: int = 0
    horizon: int | None = None
    max_participation: float = 0.35

    def __post_init__(self):
        if self.side not in (1, -1) or self.shares <= 0 or self.start < 0 or (self.horizon is not None and self.horizon < 1) or not 0 < self.max_participation <= 1:
            raise ValueError("side must be +1 or -1; shares > 0; start >= 0; horizon >= 1; 0 < max_participation <= 1")

    def window(self, market: Market) -> slice:
        stop = market.intervals if self.horizon is None else min(self.start + self.horizon, market.intervals)
        if self.start >= stop:
            raise ValueError("the order's window is empty")
        return slice(self.start, stop)

    def length(self, market: Market) -> int:
        w = self.window(market)
        return w.stop - w.start

    def reference(self, market: Market) -> float:
        return market.price if self.arrival is None else float(self.arrival)


@dataclass(frozen=True)
class Scenario:
    """How the day behaves. ``drift_bps`` is the expected price move over the order's window (positive = up) and ``persistence`` the first-order autocorrelation of interval returns (positive = momentum, which
    makes a favourable move likely to continue; negative = mean reversion). A *stress window* (``stress = (from, to)`` as fractions of the horizon) multiplies the volatility, the spread, the impact and the traded
    volume for the intervals inside it: a crisis is wide spreads, jumpy prices and thin books, with a lot of panic volume."""

    name: str = "normal"
    drift_bps: float = 0.0
    persistence: float = 0.0
    stress: tuple | None = None
    vol_mult: float = 1.0
    spread_mult: float = 1.0
    impact_mult: float = 1.0
    volume_mult: float = 1.0
    description: str = ""

    def multipliers(self, intervals: int) -> dict[str, np.ndarray]:
        """Per-interval multipliers for ``intervals`` intervals."""
        out = {k: np.ones(intervals) for k in ("vol", "spread", "impact", "volume")}
        if self.stress is not None:
            lo, hi = int(round(self.stress[0] * intervals)), int(round(self.stress[1] * intervals))
            inside = np.zeros(intervals, bool)
            inside[lo:hi] = True
            for key, m in (("vol", self.vol_mult), ("spread", self.spread_mult), ("impact", self.impact_mult), ("volume", self.volume_mult)):
                out[key][inside] = m
        return out


SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario("normal", description="no drift, no memory: prices are a random walk and volume follows the usual U"),
    Scenario("trend_up", drift_bps=40.0, persistence=0.25, description="the stock drifts up 40 bps over the order, and moves tend to continue (bad for a buyer who waits)"),
    Scenario("trend_down", drift_bps=-40.0, persistence=0.25, description="the stock drifts down 40 bps over the order, and moves tend to continue (bad for a seller who waits)"),
    Scenario("mean_reverting", persistence=-0.35, description="prices overshoot and come back: a dip is a chance to buy, and a rally a chance to sell"),
    Scenario("crisis", drift_bps=-60.0, persistence=0.15, stress=(0.35, 0.70), vol_mult=3.0, spread_mult=4.0, impact_mult=2.5, volume_mult=1.6,
             description="a shock in the middle of the order and a falling price (60 bps): three times the volatility, spreads four times wider, a book that is two and a half times thinner, and panic volume. It hurts a seller and helps a buyer who waits"),
)}
