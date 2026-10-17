"""The forecasters by name, in the order the form lists them, and the metadata the form shows."""

from __future__ import annotations

from .base import Forecaster
from .baselines import GBM, Historical, HullWhite, RandomWalk, RiskMetrics
from .gp import GPDrift, GPPrice, GPReturns
from .markov import HiddenMarkov, MarkovReturns, MarkovVolatility
from .montecarlo import FilteredHistorical, GarchMC, JumpDiffusion
from .others import HAR, Analogues, ARMAReturns, KalmanTrend
from .stochvol import GarchNIG, GarchVG, HestonMC, SVKalman

BASELINE = "random_walk"
BLEND = "blend"
CLASSES = (RandomWalk, RiskMetrics, Historical, HullWhite, GBM, GarchMC, FilteredHistorical, GarchVG, GarchNIG, HestonMC, JumpDiffusion, SVKalman, HAR, MarkovReturns, MarkovVolatility, HiddenMarkov, GPReturns, GPDrift, GPPrice, KalmanTrend, ARMAReturns, Analogues)
METHODS: dict[str, Forecaster] = {c.name: c() for c in CLASSES}
FAMILIES = ("baseline", "Monte Carlo", "Volatility", "Markov", "Gaussian process", "State space", "Time series", "Pattern matching", "Combination")
DEFAULT = ("random_walk", "riskmetrics", "hull_white", "garch", "fhs", "har", "markov_volatility", "hmm", "gp_returns", "gp_drift", "kalman", "analog")
BLEND_VIEW = {"id": BLEND, "label": "Equal-weight blend of the other methods", "short": "Blend (equal weight)", "family": "Combination", "mean_signal": True, "heavy": False, "min_history": 0,
              "text": "Pools the forecasts of every other method you chose, drawing equally from each. Combining forecasts is the most dependable way to improve on the best of them, and an equal-weight pool is hard to beat.",
              "assumes": "No method is reliably best, so each is trusted equally.", "fails": "It inherits the errors the methods share, and a bad member drags it down by its share."}


def create(name: str) -> Forecaster:
    if name not in METHODS:
        raise KeyError(f"unknown forecasting method '{name}'")
    return METHODS[name]


def catalog() -> list[dict]:
    """Every method for the form (the baseline first), with its blend last."""
    return [m.view() for m in METHODS.values()] + [BLEND_VIEW]
